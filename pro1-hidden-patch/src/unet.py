"""Find the patch with a small UNet and a causal check with ResNet-18.

1. The UNet gets the image and its noise residual (image minus 3x3 median), as in
   steganalysis (Liu et al. 2019) and the SRM noise stream (Zhou et al. 2018).
   It predicts two masks: the patch and the decoys.
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
from .evaluate import BOX, eval_set, points_array, score_report
from .paths import output_dir
from .saliency import answers_frame, causal_drops, margin, smooth


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four levels (64, 32, 16, 8 pixels). The deepest level sees the full image."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width]
        self.down = nn.ModuleList([block(6, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])])
        self.up = nn.ModuleList([block(w[3] + w[2], w[2]), block(w[2] + w[1], w[1]), block(w[1] + w[0], w[0])])
        self.out = nn.Conv2d(w[0], 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, x - smooth(x, "median3")], 1)
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


def train(images: np.ndarray, masks: np.ndarray, cfg: dict, device: str) -> UNet:
    torch.manual_seed(cfg["seed"])
    x, y = to_tensor(images, device), torch.from_numpy(masks).to(device)
    model = UNet(cfg["width"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = -(-len(x) // cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch)
    gen = torch.Generator().manual_seed(cfg["seed"])
    model.train()
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(x), generator=gen)
        total = 0.0
        for idx in order.split(cfg["batch"]):
            k = int(torch.randint(8, (1,), generator=gen))
            xb, yb = dihedral(x[idx], k), dihedral(y[idx], k)
            logits = model(xb)
            p = torch.sigmoid(logits)
            dice = 1 - (2 * (p * yb).sum((2, 3)) + 1) / (p.sum((2, 3)) + yb.sum((2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logits, yb) + dice.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(f"epoch {epoch} loss {total / len(x):.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def predict_masks(model: UNet, images: np.ndarray, device: str, batch: int = 500) -> np.ndarray:
    """Return the mean probability over the 8 flips and rotations. Shape (N, 2, SIZE, SIZE)."""
    out = []
    for i in range(0, len(images), batch):
        x = to_tensor(images[i:i + batch], device)
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


@torch.no_grad()
def analyze(resnet, unet, images: np.ndarray, cfg: dict, device: str) -> dict:
    probs = predict_masks(unet, images, device)
    boxes, patch_prob = candidate_boxes(probs, cfg["mask_threshold"], cfg["min_area"], cfg["candidates"])
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

    labels, images = eval_set("train")
    is_tune = labels.index.isin(labels.sample(frac=cfg["tune_fraction"], random_state=cfg["seed"]).index)
    unet = train(images[~is_tune], target_masks(labels[~is_tune]), cfg, device)
    torch.save(unet.state_dict(), out_dir / "unet.pt")
    resnet = load_model().to(device).requires_grad_(False)

    tune_labels = labels[is_tune].reset_index(drop=True)
    tune_cand = analyze(resnet, unet, images[is_tune], cfg, device)
    params = tune(tune_labels, tune_cand, np.linspace(*cfg["thresholds"]))
    metrics = {"codename": cfg["codename"], **params}

    sets = {"tune": (tune_labels, tune_cand)}
    for name in ["val", "val_flip"]:
        lab, img = eval_set(name)
        sets[name] = (lab, analyze(resnet, unet, img, cfg, device))
    for name, (lab, cand) in sets.items():
        answers = answers_frame(lab.id, choose(cand, params["rule"], params["threshold"]))
        report = score_report(lab, answers)
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
