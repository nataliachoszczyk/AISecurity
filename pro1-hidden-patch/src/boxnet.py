"""Second stage: refine the edges of a candidate box with a small CNN (BoxNet).

post-v1 showed that most lost points come from the hits: their mean IoU is 0.71, so the edges are 1-4 pixels off.
A perfect choice among the UNet candidates gives only 0.73 on val. BoxNet gets the image maps (unet_inputs) and the
candidate box as one more channel. It averages its features over the columns of the candidate (for the top and the
bottom edge) and over its rows (for the left and the right edge). The patch is a rectangle, so the evidence of a whole
edge adds up. A 1-D CNN then gives a score to each position of each edge in a window around the candidate edge.
A second head predicts the IoU of the candidate with the patch, so it can also compare the candidates.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eda import SIZE, load_images, load_model, load_test_ids
from .evaluate import BOX, eval_set, points_array, tune_ids
from .generate import load_set
from .paths import generated_dir, output_dir
from .post import box_iou, decoy_boxes, paired
from .saliency import answers_frame, box_sizes, smooth
from .unet import analyze, block, choose, dihedral, load_ensemble, tune, undo_dihedral, unet_inputs


def box_mask(boxes: torch.Tensor) -> torch.Tensor:
    """Return masks (N, 1, SIZE, SIZE) with 1 inside each box (N, 4)."""
    r = torch.arange(SIZE, device=boxes.device)
    rows = (r >= boxes[:, 1, None]) & (r < boxes[:, 3, None])
    cols = (r >= boxes[:, 0, None]) & (r < boxes[:, 2, None])
    return (rows[:, :, None] & cols[:, None, :]).float()[:, None]


def mask_box(mask: torch.Tensor) -> torch.Tensor:
    """Return the bounding boxes (N, 4) of masks (N, SIZE, SIZE). An empty mask gives 0,0,0,0."""
    r = torch.arange(SIZE, device=mask.device)
    rows, cols = mask.any(2), mask.any(1)

    def span(v):
        return torch.where(v, r, SIZE).min(1).values, torch.where(v, r, -1).max(1).values + 1

    (y0, y1), (x0, x1) = span(rows), span(cols)
    boxes = torch.stack([x0, y0, x1, y1], 1)
    return torch.where(rows.any(1, keepdim=True), boxes, torch.zeros_like(boxes))


def transform_boxes(boxes: torch.Tensor, k: int, undo: bool = False) -> torch.Tensor:
    m = box_mask(boxes)
    m = undo_dihedral(m, k) if undo else dihedral(m, k)
    return mask_box(m[:, 0] > 0.5)


class BoxNet(nn.Module):
    def __init__(self, extra: int, width: int, window: int):
        super().__init__()
        self.window = window
        w = width
        # Input: the maps, the noise residual (3 channels) and the candidate mask.
        self.body = nn.Sequential(block(3 + extra + 3 + 1, w), block(w, w), block(w, w),
                                  nn.Conv2d(w, w, 3, padding=2, dilation=2), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.edge = nn.ModuleList([
            nn.Sequential(nn.Conv1d(w + 1, w, 5, padding=2), nn.ReLU(inplace=True),
                          nn.Conv1d(w, w, 5, padding=2), nn.ReLU(inplace=True), nn.Conv1d(w, 2, 1))
            for _ in range(2)])
        self.score = nn.Sequential(nn.Linear(2 * w, w), nn.ReLU(inplace=True), nn.Linear(w, 1))

    def forward(self, x: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the edge logits (N, 4, SIZE) and the IoU logit (N,).

        Edge order x_min, y_min, x_max, y_max. Position i means the coordinate i for x_min and y_min,
        and i + 1 for x_max and y_max (the first pixel after the box).
        """
        m = box_mask(boxes)
        f = self.body(torch.cat([x, x[:, :3] - smooth(x[:, :3], "median3"), m], 1))
        rows, cols = m.amax(3), m.amax(2)  # (N, 1, SIZE): the rows and the columns of the box.
        prof_y = (f * cols[:, :, None, :]).sum(3) / cols.sum(2, keepdim=True).clamp(min=1)
        prof_x = (f * rows[:, :, :, None]).sum(2) / rows.sum(2, keepdim=True).clamp(min=1)
        ex = self.edge[0](torch.cat([prof_x, cols], 1))
        ey = self.edge[1](torch.cat([prof_y, rows], 1))
        logits = torch.stack([ex[:, 0], ey[:, 0], ex[:, 1], ey[:, 1]], 1).float()
        pos = torch.arange(SIZE, device=x.device)
        coord = torch.stack([pos, pos, pos + 1, pos + 1])
        far = (coord[None] - boxes[:, :, None]).abs() > self.window
        logits = logits.masked_fill(far, float("-inf"))
        ring = box_mask(torch.cat([(boxes[:, :2] - 4).clamp(min=0), (boxes[:, 2:] + 4).clamp(max=SIZE)], 1)) - m
        inside = (f * m).sum((2, 3)) / m.sum((2, 3)).clamp(min=1)
        outside = (f * ring).sum((2, 3)) / ring.sum((2, 3)).clamp(min=1)
        return logits, self.score(torch.cat([inside, outside], 1))[:, 0].float()


def decode(logits: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Return the boxes with the best edge positions. A box smaller than 4 pixels keeps the candidate."""
    out = logits.argmax(2) + torch.tensor([0, 0, 1, 1], device=logits.device)
    bad = ((out[:, 2] - out[:, 0]) < 4) | ((out[:, 3] - out[:, 1]) < 4)
    return torch.where(bad[:, None], boxes, out)


def sample_candidates(true: np.ndarray, decoys: np.ndarray, rng, jitter: int, p_patch: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one candidate box for each image and True where it comes from the patch.

    A candidate is the patch box with a random change of each edge (up to jitter pixels), or a changed decoy box,
    or a random box of an allowed size. The IoU head learns from all of them, the edge heads only from the patch.
    """
    n = len(true)
    sizes = np.array(box_sizes(0.08, 0.15, 2.0, 1))
    has_patch = (true != 0).any(1)
    use_patch = has_patch & (rng.random(n) < p_patch)
    base = np.zeros((n, 4), np.int64)
    base[use_patch] = true[use_patch]
    n_decoys = (decoys != 0).any(2).sum(1)
    for i in np.where(~use_patch)[0]:
        if n_decoys[i] and rng.random() < 0.7:
            base[i] = decoys[i, rng.integers(n_decoys[i])]
        else:
            w, h = sizes[rng.integers(len(sizes))]
            x0, y0 = rng.integers(0, SIZE - w + 1), rng.integers(0, SIZE - h + 1)
            base[i] = [x0, y0, x0 + w, y0 + h]
    boxes = np.clip(base + rng.integers(-jitter, jitter + 1, (n, 4)), 0, SIZE)
    small = ((boxes[:, 2] - boxes[:, 0]) < 4) | ((boxes[:, 3] - boxes[:, 1]) < 4)
    boxes[small] = base[small]
    return boxes, use_patch


def train(inputs: np.ndarray, true: np.ndarray, decoys: np.ndarray, cfg: dict, device: str) -> BoxNet:
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    x = torch.from_numpy(inputs).to(device)
    model = BoxNet(inputs.shape[1] - 3, cfg["width"], cfg["window"]).to(device)
    steps = -(-len(x) // cfg["batch"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * steps)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    model.train()
    model.losses = []
    for epoch in range(cfg["epochs"]):
        cand, is_patch = sample_candidates(true, decoys, rng, cfg["jitter"], cfg["p_patch"])
        order = rng.permutation(len(x))
        total = 0.0
        for idx in np.array_split(order, steps):
            k = int(rng.integers(8))
            idx_t = torch.from_numpy(idx).to(device)
            cb = transform_boxes(torch.from_numpy(cand[idx]).to(device), k)
            tb = transform_boxes(torch.from_numpy(true[idx]).to(device), k)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits, score = model(dihedral(x[idx_t], k), cb)
            iou = torch.from_numpy(box_iou(cb.cpu().numpy(), tb.cpu().numpy())).float().to(device)
            loss = F.binary_cross_entropy_with_logits(score, iou)
            sel = torch.from_numpy(is_patch[idx]).to(device)
            if sel.any():
                target = (tb - torch.tensor([0, 0, 1, 1], device=device))[sel].clamp(0, SIZE - 1)
                loss = loss + F.cross_entropy(logits[sel].flatten(0, 1), target.flatten())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        model.losses.append(total / len(x))
        print(f"epoch {epoch} loss {model.losses[-1]:.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def refine(model: BoxNet, inputs: np.ndarray, boxes: np.ndarray, device: str, iters: int = 1, batch: int = 500,
           index: np.ndarray | None = None):
    """Return the refined boxes and the predicted IoU. Both are means over the 8 flips and rotations.

    index gives the input row of each box (default: row i for box i). The box 0,0,0,0 stays 0,0,0,0.
    """
    index = np.arange(len(boxes)) if index is None else index
    out_boxes, out_scores = [], []
    for i in range(0, len(boxes), batch):
        x = torch.from_numpy(inputs[index[i:i + batch]]).to(device)
        b = torch.from_numpy(boxes[i:i + batch]).to(device)
        empty = (b == 0).all(1)
        for _ in range(iters):
            preds, scores = [], []
            for k in range(8):
                logits, score = model(dihedral(x, k), transform_boxes(b, k))
                preds.append(transform_boxes(decode(logits, transform_boxes(b, k)), k, undo=True).float())
                scores.append(torch.sigmoid(score))
            b = torch.where(empty[:, None], b, torch.stack(preds).mean(0).round().long())
        out_boxes.append(b.cpu().numpy())
        out_scores.append(torch.stack(scores).mean(0).cpu().numpy())
    return np.concatenate(out_boxes), np.concatenate(out_scores)


def load_boxnet(codename: str, device: str) -> BoxNet:
    """Load the boxnet.pt of an earlier run from /kaggle/input (the output of its notebook)."""
    from .config import load_config
    from .paths import KAGGLE_INPUT
    found = sorted(KAGGLE_INPUT.rglob(f"{codename}/boxnet.pt"))
    assert found, f"no boxnet.pt of {codename} in /kaggle/input"
    cfg = load_config(codename)
    state = torch.load(found[0], map_location=device)
    # The first convolution gets the maps, the residual (3 channels) and the candidate mask (1 channel).
    model = BoxNet(state["body.0.0.weight"].shape[1] - 7, cfg["width"], cfg["window"])
    model.load_state_dict(state)
    print(f"{codename}: {found[0]}", flush=True)
    return model.to(device).eval()


def run_experiment(cfg: dict, device: str | None = None) -> dict:
    """Train BoxNet, then refine the answers of the UNets in cfg candidates_from. Write small reports."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = output_dir() / cfg["codename"]
    out_dir.mkdir(parents=True, exist_ok=True)
    resnet = load_model().to(device).requires_grad_(False)
    smoke = cfg.get("smoke")

    labels, images = eval_set("train")
    is_tune = labels.id.isin(tune_ids(cfg["tune_fraction"], cfg["seed"])).to_numpy()
    train_labels = labels[~is_tune].reset_index(drop=True)
    train_images = images[~is_tune]
    sets = {"tune": (labels[is_tune].reset_index(drop=True), images[is_tune]), "val": eval_set("val")}
    del images
    if cfg.get("generated"):
        lab, img = load_set(generated_dir() / "val")
        assert not lab.base.isin(labels.id).any(), "a generated val image uses a train image as its base"
        sets["gen_val"] = (lab, img)
    if smoke:
        sets = {k: (lab.iloc[:smoke].reset_index(drop=True), img[:smoke]) for k, (lab, img) in sets.items()}
        train_labels, train_images = train_labels.iloc[:smoke * 4], train_images[:smoke * 4]

    inputs = unet_inputs(resnet, train_images, cfg, device)
    true = train_labels[BOX].to_numpy()
    decoys = decoy_boxes(train_labels)
    for name in cfg.get("generated_sets", ["gen-v4"]) if cfg.get("generated") else []:
        gen_labels, gen_images = load_set(generated_dir(name) / "train")
        assert gen_labels.base.isin(labels.id[~is_tune]).all(), "a generated image uses a tune or val image as its base"
        if smoke:
            gen_labels, gen_images = gen_labels.iloc[:smoke * 4], gen_images[:smoke * 4]
        inputs = np.concatenate([inputs, unet_inputs(resnet, gen_images, cfg, device)])
        true = np.concatenate([true, gen_labels[BOX].to_numpy()])
        d = decoy_boxes(gen_labels)
        width = max(decoys.shape[1], d.shape[1])
        pad = lambda a: np.pad(a, ((0, 0), (0, width - a.shape[1]), (0, 0)))
        decoys = np.concatenate([pad(decoys), pad(d)])
        del gen_images
    print(f"BoxNet train images: {len(inputs)}", flush=True)
    model = train(inputs, true, decoys, cfg, device)
    del inputs
    torch.save(model.state_dict(), out_dir / "boxnet.pt")
    pd.DataFrame({"epoch": range(len(model.losses)), "loss": model.losses}).to_csv(out_dir / "train_log.csv", index=False)

    images_of = {k: img for k, (_, img) in sets.items()}
    if cfg.get("candidates_from"):
        test_ids = load_test_ids()
        images_of["test"] = load_images("test", test_ids[:smoke] if smoke else test_ids)
        test_ids = test_ids[:len(images_of["test"])]
    set_inputs = {k: unet_inputs(resnet, img, cfg, device) for k, img in images_of.items()}
    true_of = {k: lab[BOX].to_numpy() for k, (lab, _) in sets.items()}

    # A check without the UNet: the patch box with random edge changes of up to 3 pixels, before and after BoxNet.
    metrics = {"codename": cfg["codename"]}
    rng = np.random.default_rng(cfg["seed"] + 1)
    for k in sets:
        adv = (true_of[k] != 0).any(1)
        jittered = np.clip(true_of[k][adv] + rng.integers(-3, 4, (adv.sum(), 4)), 0, SIZE)
        refined, _ = refine(model, set_inputs[k][adv], jittered, device)
        metrics[f"jitter_iou_before_{k}"] = float(box_iou(jittered, true_of[k][adv]).mean())
        metrics[f"jitter_iou_after_{k}"] = float(box_iou(refined, true_of[k][adv]).mean())
    print(json.dumps(metrics, indent=1), flush=True)

    rows = []
    if cfg.get("candidates_from"):
        unet = load_ensemble(cfg["candidates_from"], device)
        n_unet = unet.models[0].down[0][0].weight.shape[1] - 3
        # The UNets of unet-v8 use RGB and the integrated gradients map: the first channels of the BoxNet input.
        cand = {k: analyze(resnet, unet, images_of[k], cfg, device, set_inputs[k][:, :n_unet]) for k in images_of}
        params = tune(sets["tune"][0], cand["tune"], np.linspace(*cfg["thresholds"]))
        base = {k: choose(c, params["rule"], params["threshold"]) for k, c in cand.items()}
        answers = {"base": base}
        for iters in [1, 2]:
            answers[f"edge{iters}"] = {k: refine(model, set_inputs[k], base[k], device, iters)[0] for k in cand}
        # Refine all candidates, then choose the candidate with the largest predicted IoU.
        refined = {}
        for k, c in cand.items():
            n, kk = c["boxes"].shape[:2]
            b, s = refine(model, set_inputs[k], c["boxes"].reshape(-1, 4), device, index=np.repeat(np.arange(n), kk))
            refined[k] = (b.reshape(n, kk, 4), s.reshape(n, kk))
        pick = lambda k: refined[k][0][np.arange(len(refined[k][1])), refined[k][1].argmax(1)]
        answers["score"] = {k: np.where((base[k] == 0).all(1, keepdims=True), 0, pick(k)) for k in cand}
        q_grid = np.linspace(0, 0.8, 81)
        score_none = lambda k, q: np.where((refined[k][1].max(1) < q)[:, None], 0, pick(k))
        q = max(q_grid, key=lambda q: points_array(true_of["tune"], score_none("tune", q)).mean())
        answers["score_none"] = {k: score_none(k, q) for k in cand}
        metrics |= {"params": params, "score_threshold": float(q)}
        for variant, ans in answers.items():
            row = {"variant": variant, **{k: float(points_array(true_of[k], ans[k]).mean()) for k in sets}}
            row["val_delta"], row["val_delta_se"] = paired(true_of["val"], ans["val"], base["val"])
            row["test_none"] = float((ans["test"] == 0).all(1).mean())
            rows.append(row)
            print(row, flush=True)
            answers_frame(test_ids, ans["test"]).to_csv(out_dir / f"submission_{cfg['codename']}_{variant}.csv", index=False)
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
