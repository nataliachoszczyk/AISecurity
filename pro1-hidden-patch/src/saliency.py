"""Find the patch with ResNet-18 only, without training.

1. A 3x3 median removes most of the adversarial noise. Integrated gradients from the
   smoothed image to the image give the part of the class margin that each pixel causes.
2. For each allowed box size, the box with the largest sum of this map is a candidate.
   The residual map (image minus median) gives more candidates.
3. Causal check: smooth only the inside of each candidate box and measure the drop of the margin.
   Only the patch causes a large drop. A decoy causes almost no drop.
4. Answer the smallest box with a drop near the largest drop. If the largest drop is small
   compared to the margin, answer 0,0,0,0.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .eda import SIZE, load_images, load_model, load_test_ids
from .evaluate import BOX, eval_set, points_array, score_report
from .paths import output_dir


def median(x: torch.Tensor, k: int) -> torch.Tensor:
    r = k // 2
    patches = F.pad(x, (r, r, r, r), mode="reflect").unfold(2, k, 1).unfold(3, k, 1)
    return patches.reshape(*patches.shape[:4], k * k).median(-1).values


def gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = int(3 * sigma)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t**2 / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def smooth(x: torch.Tensor, name: str) -> torch.Tensor:
    """Return the image without most of the adversarial noise. name: median3, median5 or gauss1.0."""
    if name.startswith("median"):
        return median(x, int(name[6:]))
    return gauss(x, float(name[5:]))


def margin(logits: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Return the logit of class c minus the largest other logit."""
    own = logits.gather(1, c[:, None])[:, 0]
    other = logits.scatter(1, c[:, None], float("-inf")).max(1).values
    return own - other


def box_sizes(area_min: float, area_max: float, ratio_max: float, step: int) -> list[tuple[int, int]]:
    return [(w, h) for w in range(4, SIZE + 1, step) for h in range(4, SIZE + 1, step)
            if area_min <= w * h / SIZE**2 <= area_max and 1 / ratio_max <= w / h <= ratio_max]


def integrated_gradients(model, x, xs, c, steps):
    total = torch.zeros_like(x)
    for a in (torch.arange(steps, device=x.device) + 0.5) / steps:
        z = (xs + a * (x - xs)).requires_grad_(True)
        (grad,) = torch.autograd.grad(margin(model(z), c).sum(), z)
        total += grad
    return ((x - xs) * total / steps).sum(1)


def best_boxes(score_map: torch.Tensor, sizes, k: int) -> torch.Tensor:
    """Return the k boxes with the largest map sum, at most one box for each size. Shape (N, k, 4)."""
    integral = F.pad(score_map.cumsum(1).cumsum(2), (1, 0, 1, 0))
    sums, boxes = [], []
    for w, h in sizes:
        s = integral[:, h:, w:] - integral[:, :-h, w:] - integral[:, h:, :-w] + integral[:, :-h, :-w]
        best, pos = s.flatten(1).max(1)
        y, x = pos // s.shape[2], pos % s.shape[2]
        sums.append(best)
        boxes.append(torch.stack([x, y, x + w, y + h], 1))
    sums, boxes = torch.stack(sums, 1), torch.stack(boxes, 1)
    top = sums.topk(min(k, len(sizes)), 1).indices
    return boxes.gather(1, top[..., None].expand(-1, -1, 4))


def box_masks(boxes: torch.Tensor) -> torch.Tensor:
    """Return masks (N, K, 1, SIZE, SIZE) with 1 inside each box."""
    r = torch.arange(SIZE, device=boxes.device)
    b = boxes[..., None]
    cols = (r >= b[:, :, 0]) & (r < b[:, :, 2])
    rows = (r >= b[:, :, 1]) & (r < b[:, :, 3])
    return (rows[..., :, None] & cols[..., None, :]).float()[:, :, None]


@torch.no_grad()
def causal_drops(model, x, xs, c, m0, boxes, chunk):
    n, k = boxes.shape[:2]
    mask = box_masks(boxes)
    mixed = (x[:, None] * (1 - mask) + xs[:, None] * mask).flatten(0, 1)
    cc = c.repeat_interleave(k)
    m = torch.cat([margin(model(mixed[i:i + chunk]), cc[i:i + chunk]) for i in range(0, len(mixed), chunk)])
    return m0[:, None] - m.view(n, k)


def analyze(model, images: np.ndarray, cfg: dict, device: str) -> dict:
    """Return the candidate boxes, their margin drops and the margin of each image."""
    sizes = box_sizes(cfg["area_min"], cfg["area_max"], cfg["ratio_max"], cfg["size_step"])
    out = {"boxes": [], "drops": [], "margin": [], "whole_drop": []}
    for i in range(0, len(images), cfg["batch"]):
        x = torch.from_numpy(images[i:i + cfg["batch"]]).permute(0, 3, 1, 2).to(device)
        xs = smooth(x, cfg["smooth"])
        with torch.no_grad():
            logits = model(x)
            c = logits.argmax(1)
            m0 = margin(logits, c)
            whole = m0 - margin(model(xs), c)
        ig = integrated_gradients(model, x, xs, c, cfg["ig_steps"])
        noise = (x - xs).abs().sum(1)
        boxes = torch.cat([best_boxes(ig, sizes, cfg["ig_candidates"]),
                           best_boxes(noise - noise.mean((1, 2), keepdim=True), sizes, cfg["residual_candidates"])], 1)
        drops = causal_drops(model, x, xs, c, m0, boxes, cfg["chunk"])
        for key, v in [("boxes", boxes), ("drops", drops), ("margin", m0), ("whole_drop", whole)]:
            out[key].append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def choose(cand: dict, tie: float, threshold: float) -> np.ndarray:
    """Return one box for each image: the smallest box with a drop of at least tie * the largest drop."""
    boxes, drops = cand["boxes"], cand["drops"]
    best = drops.max(1)
    area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    k = np.where(drops >= tie * best[:, None], area, np.inf).argmin(1)
    answer = boxes[np.arange(len(boxes)), k].copy()
    answer[best / cand["margin"] < threshold] = 0
    return answer


def tune(labels: pd.DataFrame, cand: dict, ties, thresholds) -> dict:
    true = labels[BOX].to_numpy()
    results = [(points_array(true, choose(cand, t, th)).mean(), t, th) for t in ties for th in thresholds]
    best = max(results)
    return {"train_tuned": float(best[0]), "tie": float(best[1]), "threshold": float(best[2])}


def answers_frame(ids, boxes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(boxes.astype(int), columns=BOX).assign(id=list(ids))[["id", *BOX]]


def run_experiment(cfg: dict) -> dict:
    """Tune on a train sample, score train/val/val_flip, predict the test set. Write small result files."""
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device).requires_grad_(False)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    out_dir = output_dir() / cfg["codename"]
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = {}
    for name, n in [("train", cfg["tune_n"]), ("val", None), ("val_flip", None)]:
        labels, images = eval_set(name, n)
        sets[name] = (labels, analyze(model, images, cfg, device))
        print(name, len(labels), flush=True)

    params = tune(*sets["train"], cfg["ties"], np.linspace(*cfg["thresholds"]))
    metrics = {"codename": cfg["codename"], **params}
    for name, (labels, cand) in sets.items():
        answers = answers_frame(labels.id, choose(cand, params["tie"], params["threshold"]))
        report = score_report(labels, answers)
        report.to_csv(out_dir / f"report_{name}.csv", index=False)
        metrics[name] = float(report.points.iloc[0])
        per_image = answers.assign(best_drop=cand["drops"].max(1), margin=cand["margin"], whole_drop=cand["whole_drop"])
        per_image.to_csv(out_dir / f"answers_{name}.csv", index=False)
    metrics["val_mean"] = (metrics["val"] + metrics["val_flip"]) / 2

    test_ids = load_test_ids()
    cand = analyze(model, load_images("test", test_ids), cfg, device)
    answers_frame(test_ids, choose(cand, params["tie"], params["threshold"])).to_csv(
        out_dir / f"submission_{cfg['codename']}.csv", index=False)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics
