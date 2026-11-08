"""The final answer of act2: the mean mask of two trained UNets and a soft refinement of the box.

1. Each model predicts the masks of the tune, val, generated val and test images one time.
   The mean of the probabilities is the ensemble.
2. The fitted boxes and the causal check of unet.py give the base answer. Its rule and threshold are tuned on the tune part.
3. soft: search boxes near the base answer for the largest expected IoU with the patch mask.
   gamma and pad are tuned on the tune part.
"""
import json

import numpy as np
import pandas as pd
import torch

from .eda import SIZE, load_images, load_model, load_test_ids
from .evaluate import BOX, eval_set, points_array, tune_ids
from .generate import load_set
from .paths import generated_dir, output_dir
from .saliency import answers_frame
from .unet import analyze, choose, load_unet, predict_masks, tune, unet_inputs


def integral(maps: np.ndarray) -> np.ndarray:
    """Return the integral images (N, SIZE + 1, SIZE + 1) of maps (N, SIZE, SIZE)."""
    return np.pad(maps.cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))


def box_sums(table: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return the map sum inside each box. table: integral images (N, S, S), boxes (N, M, 4)."""
    flat, s = table.reshape(len(table), -1), table.shape[-1]
    at = lambda y, x: np.take_along_axis(flat, y * s + x, 1)
    x0, y0, x1, y1 = np.moveaxis(boxes, -1, 0)
    return at(y1, x1) - at(y0, x1) - at(y1, x0) + at(y0, x0)


def allowed(boxes: np.ndarray) -> np.ndarray:
    """Return True for boxes with 8-15% of the area and a ratio from 1:2 to 2:1 (the rule of box_sizes)."""
    w, h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    area = w * h / SIZE**2
    return (w >= 4) & (h >= 4) & (area >= 0.08) & (area <= 0.15) & (w <= 2 * h) & (h <= 2 * w)


def refine_soft(patch_prob: np.ndarray, answer: np.ndarray, radius: int, gamma: float, pad: int) -> np.ndarray:
    """Move each side of the answer by up to radius pixels to the largest expected IoU with the mask p ** gamma.

    The expected IoU of a box B is sum_B(q) / (|B| + sum_W(q) - sum_B(q)). W is the answer with pad pixels on each side.
    The fitted boxes maximize sum(p - offset), which is not the IoU.
    """
    d = np.arange(-radius, radius + 1)
    offsets = np.stack(np.meshgrid(d, d, d, d, indexing="ij"), -1).reshape(-1, 4)
    out = answer.copy()
    for i in range(0, len(answer), 500):
        a = answer[i:i + 500]
        table = integral(patch_prob[i:i + 500] ** gamma)
        boxes = np.clip(a[:, None] + offsets[None], 0, SIZE)
        window = np.clip(a + np.array([-pad, -pad, pad, pad]), 0, SIZE)[:, None]
        inside = box_sums(table, boxes)
        area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        score = np.where(allowed(boxes), inside / (area + box_sums(table, window) - inside + 1e-8), -1)
        best = boxes[np.arange(len(a)), score.argmax(1)]
        has = (a != 0).any(1)
        out[i:i + 500][has] = best[has]
    return out


def run_experiment(cfg: dict, device: str | None = None) -> dict:
    """Make the ensemble of cfg models, tune the base and the soft answer, score them and write the submission."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = output_dir() / cfg["codename"]
    out_dir.mkdir(parents=True, exist_ok=True)
    resnet = load_model().to(device).requires_grad_(False)

    labels, images = eval_set("train")
    is_tune = labels.id.isin(tune_ids(cfg["tune_fraction"], cfg["seed"])).to_numpy()
    sets = {"tune": (labels[is_tune].reset_index(drop=True), images[is_tune]), "val": eval_set("val")}
    gen_labels, gen_images = load_set(generated_dir("gen-v4") / "val")
    assert not gen_labels.base.isin(labels.id).any(), "a generated val image uses a train image as its base"
    sets["gen_val"] = (gen_labels, gen_images)
    del images
    test_ids = load_test_ids()
    true = {k: lab[BOX].to_numpy() for k, (lab, _) in sets.items()}
    images_of = {k: img for k, (_, img) in sets.items()} | {"test": load_images("test", test_ids)}
    thresholds = np.linspace(*cfg["thresholds"])

    inputs = {k: unet_inputs(resnet, img, cfg, device) for k, img in images_of.items()}
    # float16 keeps the probabilities of all sets in the host memory.
    model_probs = []
    for codename in cfg["models"]:
        model = load_unet(codename, device)
        model_probs.append({k: predict_masks(model, x, device).astype(np.float16) for k, x in inputs.items()})
        del model
    probs = {k: np.mean([m[k].astype(np.float32) for m in model_probs], 0) for k in inputs}
    cand = {k: analyze(resnet, None, images_of[k], cfg, device, inputs[k], probs[k]) for k in images_of}
    params = tune(sets["tune"][0], cand["tune"], thresholds)
    answers = {"base": {k: choose(c, params["rule"], params["threshold"]) for k, c in cand.items()}}

    grid = [(g, p) for g in cfg["soft_gamma"] for p in cfg["soft_pad"]]
    soft = lambda k, g, p: refine_soft(cand[k]["probs"][:, 0], answers["base"][k], cfg["soft_radius"], g, p)
    g, p = max(grid, key=lambda gp: points_array(true["tune"], soft("tune", *gp)).mean())
    answers["soft"] = {k: soft(k, g, p) for k in cand}
    settings = {"base": params, "soft": {"gamma": g, "pad": p}}

    rows = []
    for variant, ans in answers.items():
        row = {"model": cfg["name"], "variant": variant, **{k: float(points_array(true[k], ans[k]).mean()) for k in true}}
        row["test_none"] = float((ans["test"] == 0).all(1).mean())
        row["settings"] = json.dumps(settings[variant])
        rows.append(row)
        print(row, flush=True)
        answers_frame(test_ids, ans["test"]).to_csv(out_dir / f"submission_{cfg['name']}_{variant}.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    return {"results": rows}
