"""Find the patch with a small UNet and a causal check with ResNet-18.

1. The UNet gets the image and its noise residual (image minus 3x3 median), as in
   steganalysis (Liu et al. 2019) and the SRM noise stream (Zhou et al. 2018).
   With use_ig, it also gets the integrated gradients map of ResNet-18 (see saliency.py).
   The map is calculated on the image before the flips and rotations.
   The UNet predicts two masks: the patch and the decoys.
2. Flips and 90 degree rotations are valid augmentations for this task: noise stays noise,
   and the 8x8 JPEG grid stays aligned because 64 is a multiple of 8. Thus the UNet also
   learns patches in the top half, which the train set does not have. The prediction is the
   mean over all 8 flips and rotations, so a patch in the top half and the same patch
   flipped to the bottom half get the same mask.
3. The connected parts of the masks give candidate boxes. The causal check (as in SentiNet)
   smooths each candidate and measures the drop of the ResNet-18 margin. Only the patch
   causes a large drop. A small drop for all candidates gives 0,0,0,0.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import find_objects, label

from .eda import SIZE, load_images, load_model, load_test_ids
from .evaluate import BOX, eval_set, points_array, score_report, tune_ids
from .generate import load_set
from .paths import generated_dir, output_dir
from .saliency import answers_frame, best_boxes, box_sizes, causal_drops, integrated_gradients, margin, smooth


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four levels (64, 32, 16, 8 pixels). The deepest level sees the full image."""

    def __init__(self, width: int = 32, extra: int = 0):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width]
        self.down = nn.ModuleList([block(6 + extra, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])])
        self.up = nn.ModuleList([block(w[3] + w[2], w[2]), block(w[2] + w[1], w[1]), block(w[1] + w[0], w[0])])
        self.out = nn.Conv2d(w[0], 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, x[:, :3] - smooth(x[:, :3], "median3")], 1)
        skips = []
        for i, layer in enumerate(self.down):
            x = layer(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            x = layer(torch.cat([F.interpolate(x, scale_factor=2), skip], 1))
        return self.out(x)


def target_masks(labels: pd.DataFrame) -> np.ndarray:
    """Return masks (N, 2, SIZE, SIZE): channel 0 is the patch, channel 1 the decoys."""
    masks = np.zeros((len(labels), 2, SIZE, SIZE), np.float32)
    for i, row in enumerate(labels.itertuples()):
        masks[i, 0, row.y_min:row.y_max, row.x_min:row.x_max] = 1
        for decoy in row.decoys:
            x0, y0, x1, y1 = decoy["box"]
            masks[i, 1, y0:y1, x0:x1] = 1
    return masks


def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """Apply one of the 8 flips and rotations: k = 0..7."""
    x = torch.rot90(x, k % 4, (-2, -1))
    return x.flip(-1) if k >= 4 else x


def undo_dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    x = x.flip(-1) if k >= 4 else x
    return torch.rot90(x, -(k % 4), (-2, -1))


def to_tensor(images: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(device)


def unet_inputs(resnet, images: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    """Return the UNet input (N, C, SIZE, SIZE): RGB and, with use_ig, the scaled integrated gradients map."""
    parts = []
    for i in range(0, len(images), cfg["batch_causal"]):
        x = to_tensor(images[i:i + cfg["batch_causal"]], device)
        if cfg["use_ig"]:
            c = resnet(x).argmax(1)
            ig = integrated_gradients(resnet, x, smooth(x, "median3"), c, cfg["ig_steps"])[:, None]
            x = torch.cat([x, ig / (ig.abs().amax((2, 3), keepdim=True) + 1e-8)], 1)
        parts.append(x.detach().cpu().numpy())
    return np.concatenate(parts)


def mask_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return BCE plus Dice loss."""
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum((2, 3)) + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, y) + dice.mean()


@torch.no_grad()
def eval_loss(model: UNet, x: torch.Tensor, y: torch.Tensor, batch: int) -> float:
    model.eval()
    total = sum(mask_loss(model(xb), yb).item() * len(xb) for xb, yb in zip(x.split(batch), y.split(batch)))
    model.train()
    return total / len(x)


def train(inputs: np.ndarray, masks: np.ndarray, cfg: dict, device: str, tune: tuple | None = None) -> UNet:
    """Train the UNet. With tune = (inputs, masks), also record the loss on the tune part after each epoch."""
    torch.manual_seed(cfg["seed"])
    x, y = torch.from_numpy(inputs).to(device), torch.from_numpy(masks).to(device)
    model = UNet(cfg["width"], inputs.shape[1] - 3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = -(-len(x) // cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch)
    gen = torch.Generator().manual_seed(cfg["seed"])
    model.train()
    model.losses, model.tune_losses = [], []
    if tune is not None:
        xt, yt = (torch.from_numpy(a).to(device) for a in tune)
    best_loss, best_state = float("inf"), None
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(x), generator=gen)
        total = 0.0
        for idx in order.split(cfg["batch"]):
            k = int(torch.randint(8, (1,), generator=gen))
            xb, yb = dihedral(x[idx], k), dihedral(y[idx], k)
            loss = mask_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        model.losses.append(total / len(x))
        model.tune_losses.append(eval_loss(model, xt, yt, cfg["batch"]) if tune is not None else np.nan)
        if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(f"epoch {epoch} loss {model.losses[-1]:.4f} tune loss {model.tune_losses[-1]:.4f}", flush=True)
        if cfg.get("best_epoch") and model.tune_losses[-1] < best_loss:
            best_loss, model.best_epoch = model.tune_losses[-1], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        # The tune loss increases after some epochs (overfit), so keep the weights of the best epoch.
        model.load_state_dict(best_state)
        print(f"best epoch {model.best_epoch} tune loss {best_loss:.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def predict_masks(model: UNet, inputs: np.ndarray, device: str, batch: int = 500) -> np.ndarray:
    """Return the mean probability over the 8 flips and rotations. Shape (N, 2, SIZE, SIZE)."""
    out = []
    for i in range(0, len(inputs), batch):
        x = torch.from_numpy(inputs[i:i + batch]).to(device)
        p = sum(undo_dihedral(torch.sigmoid(model(dihedral(x, k))), k) for k in range(8)) / 8
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def candidate_boxes(probs: np.ndarray, threshold: float, min_area: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the boxes of the connected parts of the noise mask (N, k, 4) and their mean patch probability.

    Empty places have the box 0,0,0,0. Its causal drop is 0.
    """
    boxes = np.zeros((len(probs), k, 4), np.int64)
    patch_prob = np.zeros((len(probs), k))
    for i, p in enumerate(probs):
        parts, _ = label(p.max(0) > threshold)
        found = []
        for sl in find_objects(parts):
            area = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
            if area >= min_area:
                found.append((p[0][sl].mean(), [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]))
        for j, (prob, box) in enumerate(sorted(found, reverse=True)[:k]):
            boxes[i, j], patch_prob[i, j] = box, prob
    return boxes, patch_prob


def fitted_boxes(probs: np.ndarray, offset: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate boxes with an allowed size (8-15% of the area, ratio 1:2 to 2:1) and their mean patch probability.

    A box gets the sum of (probability - offset) inside it. k boxes come from the patch mask and k from the decoy mask,
    so the causal check can also reject a decoy.
    """
    sizes = box_sizes(0.08, 0.15, 2.0, 1)
    p = torch.from_numpy(probs)
    boxes = torch.cat([best_boxes(p[:, 0] - offset, sizes, k), best_boxes(p[:, 1] - offset, sizes, k)], 1).numpy()
    patch_prob = np.stack([[probs[i, 0, y0:y1, x0:x1].mean() for x0, y0, x1, y1 in b] for i, b in enumerate(boxes)])
    return boxes, patch_prob


def analyze(resnet, unet, images: np.ndarray, cfg: dict, device: str) -> dict:
    inputs = unet_inputs(resnet, images, cfg, device)
    probs = predict_masks(unet, inputs, device)
    if cfg.get("boxes") == "fit":
        boxes, patch_prob = fitted_boxes(probs, cfg["fit_offset"], cfg["candidates"])
    else:
        boxes, patch_prob = candidate_boxes(probs, cfg["mask_threshold"], cfg["min_area"], cfg["candidates"])
    if cfg.get("ig_candidates"):
        # On hard test images the UNet masks are weak. The boxes of saliency.py from the integrated gradients map
        # give the causal check more chances to find the patch.
        ig_boxes = best_boxes(torch.from_numpy(inputs[:, 3]), box_sizes(0.08, 0.15, 2.0, 1), cfg["ig_candidates"]).numpy()
        ig_prob = np.stack([[probs[i, 0, y0:y1, x0:x1].mean() for x0, y0, x1, y1 in b] for i, b in enumerate(ig_boxes)])
        boxes, patch_prob = np.concatenate([boxes, ig_boxes], 1), np.concatenate([patch_prob, ig_prob], 1)
    drops, margins = [], []
    for i in range(0, len(images), cfg["batch_causal"]):
        x = to_tensor(images[i:i + cfg["batch_causal"]], device)
        logits = resnet(x)
        c = logits.argmax(1)
        m0 = margin(logits, c)
        b = torch.from_numpy(boxes[i:i + cfg["batch_causal"]]).to(device)
        drops.append(causal_drops(resnet, x, smooth(x, "median3"), c, m0, b, cfg["chunk"]).cpu().numpy())
        margins.append(m0.cpu().numpy())
    return {"probs": probs, "boxes": boxes, "patch_prob": patch_prob,
            "drops": np.concatenate(drops), "margin": np.concatenate(margins)}


def choose(cand: dict, rule: str, threshold: float) -> np.ndarray:
    """Return one box for each image.

    rule "causal": the candidate with the largest drop. rule "unet": the largest patch probability.
    The answer is 0,0,0,0 if the largest drop is smaller than threshold * margin.
    """
    score = cand["drops"] if rule == "causal" else cand["patch_prob"]
    answer = cand["boxes"][np.arange(len(score)), score.argmax(1)].copy()
    answer[cand["drops"].max(1) / cand["margin"] < threshold] = 0
    return answer


def tune(labels: pd.DataFrame, cand: dict, thresholds) -> dict:
    true = labels[BOX].to_numpy()
    results = [(points_array(true, choose(cand, rule, th)).mean(), rule, th)
               for rule in ["causal", "unet"] for th in thresholds]
    best = max(results)
    return {"tune_score": float(best[0]), "rule": best[1], "threshold": float(best[2])}


def mask_iou(probs: np.ndarray, masks: np.ndarray) -> float:
    """Return the mean IoU of the predicted noise mask (patch or decoy) for images with noise."""
    pred, true = probs.max(1) > 0.5, masks.max(1) > 0.5
    has = true.any((1, 2))
    inter = (pred & true).sum((1, 2))[has]
    union = (pred | true).sum((1, 2))[has]
    return float((inter / union).mean())


def run_experiment(cfg: dict) -> dict:
    """Train on the train part without a tune part, tune on the tune part, score val, predict the test set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = output_dir() / cfg["codename"]
    out_dir.mkdir(parents=True, exist_ok=True)

    resnet = load_model().to(device).requires_grad_(False)
    labels, images = eval_set("train")
    is_tune = labels.id.isin(tune_ids(cfg["tune_fraction"], cfg["seed"])).to_numpy()
    inputs = unet_inputs(resnet, images[~is_tune], cfg, device)
    masks = target_masks(labels[~is_tune])
    if cfg.get("generated"):
        gen_labels, gen_images = load_set(generated_dir() / "train")
        # Data leakage checks: the bases of the generated train set must come only from the train part without tune.
        assert gen_labels.base.isin(labels.id[~is_tune]).all(), "a generated image uses a tune or val image as its base"
        inputs = np.concatenate([inputs, unet_inputs(resnet, gen_images, cfg, device)])
        masks = np.concatenate([masks, target_masks(gen_labels)])
        del gen_images
        print(f"train images: {len(inputs)} ({len(gen_labels)} generated)", flush=True)
    tune_data = (unet_inputs(resnet, images[is_tune], cfg, device), target_masks(labels[is_tune]))
    unet = train(inputs, masks, cfg, device, tune_data)
    del inputs, masks
    torch.save(unet.state_dict(), out_dir / "unet.pt")
    pd.DataFrame({"epoch": range(len(unet.losses)), "loss": unet.losses, "tune_loss": unet.tune_losses}).to_csv(
        out_dir / "train_log.csv", index=False)

    tune_labels = labels[is_tune].reset_index(drop=True)
    tune_cand = analyze(resnet, unet, images[is_tune], cfg, device)
    params = tune(tune_labels, tune_cand, np.linspace(*cfg["thresholds"]))
    metrics = {"codename": cfg["codename"], **params, "best_epoch": getattr(unet, "best_epoch", None)}

    sets = {"tune": (tune_labels, tune_cand)}
    for name in ["val", "val_flip"]:
        lab, img = eval_set(name)
        sets[name] = (lab, analyze(resnet, unet, img, cfg, device))
    if cfg.get("generated"):
        # The generated val set has patches at all positions. Its report also shows the top and the bottom half.
        lab, img = load_set(generated_dir() / "val")
        assert not lab.base.isin(labels.id).any(), "a generated val image uses a train image as its base"
        lab["half"] = np.where((lab.y_min + lab.y_max) / 2 < SIZE / 2, "top half", "bottom half")
        sets["gen_val"] = (lab, analyze(resnet, unet, img, cfg, device))
    for name, (lab, cand) in sets.items():
        answers = answers_frame(lab.id, choose(cand, params["rule"], params["threshold"]))
        report = score_report(lab, answers)
        if "half" in lab:
            report = pd.concat([report, score_report(lab.assign(attack=lab.half), answers).iloc[1:]])
        report.to_csv(out_dir / f"report_{name}.csv", index=False)
        metrics[name] = float(report.points.iloc[0])
        metrics[f"mask_iou_{name}"] = mask_iou(cand["probs"], target_masks(lab))
    metrics["val_mean"] = (metrics["val"] + metrics["val_flip"]) / 2

    test_ids = load_test_ids()
    cand = analyze(resnet, unet, load_images("test", test_ids), cfg, device)
    answers_frame(test_ids, choose(cand, params["rule"], params["threshold"])).to_csv(
        out_dir / f"submission_{cfg['codename']}.csv", index=False)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics
