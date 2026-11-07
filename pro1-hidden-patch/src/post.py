"""Diagnostics and post-processing on the masks of trained UNets. No new training.

A. Diagnostics: find where the points are lost (no answer, a decoy, a miss, a low IoU),
   and the upper bounds of the candidates (a perfect 0,0,0,0 decision, a perfect choice).
B. Better boxes and decisions from the same masks:
   1. soft: search boxes near the answer for the largest expected IoU with the patch mask.
   2. causal: move each side of the box inwards while the ResNet-18 margin drop stays (as in saliency.py).
   3. selector: a gradient boosting model predicts the IoU of each candidate from its features.
      It learns on the tune part and the generated val set, with cross-validation by base image.
   4. jpeg: a separate threshold for the test images with a JPEG quality below 90.
      The train images have only 90-95, so this threshold uses the generated val set
      and the tune images without a patch, saved again as JPEG 80-88.
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .config import PROJECT_DIR, load_config
from .eda import SIZE, load_images, load_model, load_test_ids
from .evaluate import BOX, eval_set, points_array, tune_ids
from .generate import jpeg, load_set
from .paths import generated_dir, output_dir
from .saliency import answers_frame, causal_drops, margin, smooth
from .unet import analyze, choose, load_ensemble, to_tensor, tune, unet_inputs

JPEG_FILE = PROJECT_DIR / "jobs" / "runs" / "2026-11-01_14_jpeg-quality" / "output" / "jpeg_quality.csv"
LOW_JPEG = 90


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the IoU of boxes (..., 4). The box 0,0,0,0 has the IoU 0 with all boxes."""
    w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    union = area(a) + area(b) - w * h
    return np.divide(w * h, union, out=np.zeros(union.shape), where=union > 0)


def decoy_boxes(labels: pd.DataFrame) -> np.ndarray:
    """Return the decoy boxes (N, D, 4). Missing decoys have the box 0,0,0,0."""
    n = max([len(d) for d in labels.decoys] + [1])
    out = np.zeros((len(labels), n, 4), np.int64)
    for i, decoys in enumerate(labels.decoys):
        for j, d in enumerate(decoys):
            out[i, j] = d["box"]
    return out


def diagnose(labels: pd.DataFrame, cand: dict, answer: np.ndarray, pre_answer: np.ndarray) -> dict:
    """Return the share of the lost points for each error kind, and two upper bounds.

    pre_answer is the box before the 0,0,0,0 decision. The lost points of all kinds add up to 1 - score.
    """
    true = labels[BOX].to_numpy()
    adv = (true != 0).any(1)
    none = (answer == 0).all(1)
    iou = box_iou(answer, true)
    decoy_iou = box_iou(answer[:, None], decoy_boxes(labels)).max(1)
    n = len(labels)
    kinds = {
        "adv_none": adv & none,
        "adv_decoy": adv & ~none & (decoy_iou > iou),
        "adv_miss": adv & ~none & (iou == 0) & (decoy_iou == 0),
    }
    kinds["adv_hit"] = adv & ~none & ~kinds["adv_decoy"] & ~kinds["adv_miss"]
    lost = {k: float((1 - iou[m]).sum() / n) for k, m in kinds.items()}
    lost["no_patch_box"] = float((~adv & ~none).sum() / n)
    cand_iou = box_iou(cand["boxes"], true[:, None])
    hit_iou = iou[kinds["adv_hit"]]
    return {
        "score": float(points_array(true, answer).mean()),
        "lost": lost,
        "count": {k: int(m.sum()) for k, m in kinds.items()} | {"no_patch_box": int((~adv & ~none).sum())},
        "hit_iou_mean": float(hit_iou.mean()),
        "hit_iou_share": {f">={t}": float((hit_iou >= t).mean()) for t in [0.5, 0.7, 0.8, 0.9]},
        # The same boxes, but a perfect 0,0,0,0 decision.
        "oracle_decision": float(np.where(adv, box_iou(pre_answer, true), 1.0).mean()),
        # The best candidate of each image and a perfect 0,0,0,0 decision.
        "oracle_choice": float(np.where(adv, cand_iou.max(1), 1.0).mean()),
        "candidate_recall_0.5": float((cand_iou.max(1)[adv] >= 0.5).mean()),
    }


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


@torch.no_grad()
def side_drops(resnet, images: np.ndarray, answer: np.ndarray, shift: int, cfg: dict, device: str) -> dict:
    """Return the margin drops of the boxes for the causal refinement.

    The reference box is the answer with shift pixels more on each side. Then one side at a time moves
    from -shift (out) to +shift (in), and the other sides stay at the reference box. drops: (N, 4, 2 * shift + 1).
    """
    d = np.arange(-shift, shift + 1)
    ref = np.clip(answer + np.array([-shift, -shift, shift, shift]), 0, SIZE)
    inward = np.array([1, 1, -1, -1])
    boxes = np.repeat(ref[:, None, None], len(d), 2).repeat(4, 1)
    for s in range(4):
        boxes[:, s, :, s] = np.clip(answer[:, None, s] + inward[s] * d, 0, SIZE)
    flat = np.concatenate([ref[:, None], boxes.reshape(len(answer), -1, 4)], 1)
    out = []
    for i in range(0, len(images), cfg["batch_causal"]):
        x = to_tensor(images[i:i + cfg["batch_causal"]], device)
        logits = resnet(x)
        c = logits.argmax(1)
        b = torch.from_numpy(flat[i:i + cfg["batch_causal"]]).to(device)
        out.append(causal_drops(resnet, x, smooth(x, "median3"), c, margin(logits, c), b, cfg["chunk"]).cpu().numpy())
    drops = np.concatenate(out)
    return {"ref": drops[:, 0], "drops": drops[:, 1:].reshape(len(answer), 4, len(d)), "answer": answer, "shift": shift}


def refine_causal(side: dict, tie: float) -> np.ndarray:
    """Move each side inwards as far as its drop stays at least tie * the drop of the reference box."""
    d = np.arange(-side["shift"], side["shift"] + 1)
    ok = side["drops"] >= tie * side["ref"][:, None, None]
    # The most inward position with ok, or -shift if no position is ok.
    pos = np.where(ok.any(2), len(d) - 1 - ok[..., ::-1].argmax(2), 0)
    out = np.clip(side["answer"] + np.array([1, 1, -1, -1]) * d[pos], 0, SIZE)
    bad = ((out[:, 2] - out[:, 0]) < 4) | ((out[:, 3] - out[:, 1]) < 4) | (side["answer"] == 0).all(1)
    out[bad] = side["answer"][bad]
    return out


def candidate_features(cand: dict, jpeg_quality: np.ndarray) -> np.ndarray:
    """Return the features (N, K, F) of each candidate box for the selector."""
    boxes = cand["boxes"]
    n, k = boxes.shape[:2]
    area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    decoy_prob = box_sums(integral(cand["probs"][:, 1]), boxes) / np.maximum(area, 1)
    ratio = cand["drops"] / np.maximum(cand["margin"][:, None], 1e-3)
    rank = lambda v: (-v).argsort(1).argsort(1)
    columns = [
        cand["patch_prob"], decoy_prob, cand["drops"], ratio, area,
        np.broadcast_to(np.arange(k) >= k // 2, (n, k)),  # The candidate comes from the decoy mask.
        rank(cand["patch_prob"]), rank(ratio),
        cand["patch_prob"] - cand["patch_prob"].max(1, keepdims=True), ratio - ratio.max(1, keepdims=True),
        np.broadcast_to(cand["margin"][:, None], (n, k)), np.broadcast_to(jpeg_quality[:, None], (n, k)),
    ]
    return np.stack(columns, -1).astype(np.float32)


def selector_model(cfg: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=cfg["selector_iter"], learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=cfg["seed"])


def select(cand: dict, pred: np.ndarray, threshold: float) -> np.ndarray:
    """Answer the candidate with the largest predicted IoU, or 0,0,0,0 if this IoU is below threshold."""
    answer = cand["boxes"][np.arange(len(pred)), pred.argmax(1)].copy()
    answer[pred.max(1) < threshold] = 0
    return answer


def best_threshold(true: np.ndarray, answer_of, thresholds, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Return (score, threshold) with the largest weighted mean of the points."""
    return max((float(np.average(points_array(true, answer_of(t)), weights=weights)), float(t)) for t in thresholds)


def jpeg_qualities(labels: pd.DataFrame | None, ids=None) -> np.ndarray:
    """Return the JPEG quality: the label for the train and generated images, the estimate for the test images."""
    if labels is not None:
        return labels.jpeg.to_numpy(np.float32)
    estimate = pd.read_csv(JPEG_FILE).query("split == 'test'").set_index("id").quality
    return estimate.loc[list(ids)].to_numpy(np.float32)


def low_jpeg_set(tune_labels: pd.DataFrame, tune_images: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the tune images without a patch, saved again as JPEG with a random quality of 80-88."""
    keep = (tune_labels[BOX] == 0).all(1).to_numpy()
    labels = tune_labels[keep].reset_index(drop=True)
    quality = np.random.default_rng(seed).integers(80, 89, len(labels))
    x = jpeg(torch.from_numpy(tune_images[keep]).permute(0, 3, 1, 2), quality)
    return labels.assign(jpeg=quality), x.permute(0, 2, 3, 1).numpy().copy()


def paired(true: np.ndarray, answer: np.ndarray, base: np.ndarray) -> tuple[float, float]:
    """Return the mean difference of the points against base and its standard error (same images)."""
    d = points_array(true, answer) - points_array(true, base)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def run_experiment(cfg: dict, device: str | None = None) -> dict:
    """Run the diagnostics and all post-processing variants for each model of cfg models. Write small reports."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = output_dir() / cfg["codename"]
    out_dir.mkdir(parents=True, exist_ok=True)
    resnet = load_model().to(device).requires_grad_(False)

    labels, images = eval_set("train")
    is_tune = labels.id.isin(tune_ids(cfg["tune_fraction"], cfg["seed"])).to_numpy()
    sets = {"tune": (labels[is_tune].reset_index(drop=True), images[is_tune])}
    sets["val"] = eval_set("val")
    gen_labels, gen_images = load_set(generated_dir() / "val")
    assert not gen_labels.base.isin(labels.id).any(), "a generated val image uses a train image as its base"
    sets["gen_val"] = (gen_labels, gen_images)
    sets["low_jpeg"] = low_jpeg_set(*sets["tune"], cfg["seed"])
    del images
    test_ids = load_test_ids()
    test_images = load_images("test", test_ids)
    if cfg.get("smoke"):
        sets = {k: (lab.iloc[:cfg["smoke"]].reset_index(drop=True), img[:cfg["smoke"]]) for k, (lab, img) in sets.items()}
        test_ids, test_images = test_ids[:cfg["smoke"]], test_images[:cfg["smoke"]]
    quality = {k: jpeg_qualities(lab) for k, (lab, _) in sets.items()} | {"test": jpeg_qualities(None, test_ids)}
    true = {k: lab[BOX].to_numpy() for k, (lab, _) in sets.items()}
    images_of = {k: img for k, (_, img) in sets.items()} | {"test": test_images}
    scored = ["tune", "val", "gen_val"]
    thresholds = np.linspace(*cfg["thresholds"])

    rows, diagnostics = [], {}
    inputs_key, inputs = None, None
    for model_name, codenames in cfg["models"].items():
        unet = load_ensemble(codenames, device)
        # The input maps depend on the config of the models. All models of one ensemble need the same input.
        use_act = {bool(load_config(c).get("use_act")) for c in codenames}
        assert len(use_act) == 1, f"{model_name}: the models need different inputs"
        model_cfg = cfg | {"use_act": use_act.pop()}
        if inputs_key != model_cfg["use_act"]:
            inputs_key = model_cfg["use_act"]
            inputs = {k: unet_inputs(resnet, img, model_cfg, device) for k, img in images_of.items()}
        cand = {k: analyze(resnet, unet, images_of[k], cfg, device, inputs[k]) for k in images_of}
        params = tune(sets["tune"][0], cand["tune"], thresholds)
        rule, th = params["rule"], params["threshold"]
        answers = {"base": {k: choose(c, rule, th) for k, c in cand.items()}}
        pre = {k: choose(c, rule, -np.inf) for k, c in cand.items()}
        diagnostics[model_name] = {"params": params} | {
            k: diagnose(sets[k][0], cand[k], answers["base"][k], pre[k]) for k in scored}
        print(model_name, json.dumps(diagnostics[model_name], indent=1), flush=True)

        # B1: soft IoU refinement, tuned on the tune part.
        grid = [(g, p) for g in cfg["soft_gamma"] for p in cfg["soft_pad"]]
        soft = lambda k, g, p: refine_soft(cand[k]["probs"][:, 0], answers["base"][k], cfg["soft_radius"], g, p)
        g, p = max(grid, key=lambda gp: points_array(true["tune"], soft("tune", *gp)).mean())
        answers["soft"] = {k: soft(k, g, p) for k in cand}
        params_soft = {"gamma": g, "pad": p}

        # B2: causal refinement of the base boxes, tuned on the tune part.
        side = {k: side_drops(resnet, images_of[k], pre[k], cfg["causal_shift"], cfg, device) for k in cand}
        refined = lambda k, t: np.where((answers["base"][k] == 0).all(1, keepdims=True), 0, refine_causal(side[k], t))
        tie = max(cfg["causal_ties"], key=lambda t: points_array(true["tune"], refined("tune", t)).mean())
        answers["causal"] = {k: refined(k, tie) for k in cand}

        # B3: the selector. Cross-validation on tune + gen_val, grouped by the base image (gen_val has 20 copies of each base).
        feats = {k: candidate_features(cand[k], quality[k]) for k in cand}
        target = {k: box_iou(cand[k]["boxes"], true[k][:, None]) for k in ["tune", "gen_val"]}
        x = np.concatenate([feats["tune"], feats["gen_val"]])
        y = np.concatenate([target["tune"], target["gen_val"]])
        groups = np.concatenate([sets["tune"][0].id.to_numpy(), "gen_" + sets["gen_val"][0].base.astype(str).to_numpy()])
        k_cand = x.shape[1]
        oof = np.zeros(y.shape)
        for fit, held in GroupKFold(cfg["selector_folds"]).split(x, groups=groups):
            model = selector_model(cfg).fit(x[fit].reshape(-1, x.shape[-1]), y[fit].ravel())
            oof[held] = model.predict(x[held].reshape(-1, x.shape[-1])).reshape(-1, k_cand)
        n_tune = len(feats["tune"])
        oof_of = {"tune": oof[:n_tune], "gen_val": oof[n_tune:]}
        # Only the tune part has images without a patch, so the threshold uses only the tune part.
        _, q = best_threshold(true["tune"], lambda t: select(cand["tune"], oof_of["tune"], t),
                              np.linspace(*cfg["selector_thresholds"]))
        model = selector_model(cfg).fit(x.reshape(-1, x.shape[-1]), y.ravel())
        pred = {k: model.predict(feats[k].reshape(-1, x.shape[-1])).reshape(-1, k_cand) for k in cand}
        # tune and gen_val get the out-of-fold answers, so their score is not biased.
        pred |= oof_of
        answers["selector"] = {k: select(cand[k], pred[k], q) for k in cand}
        answers["selector_soft"] = {
            k: refine_soft(cand[k]["probs"][:, 0], answers["selector"][k], cfg["soft_radius"], g, p) for k in cand}

        # B4: a separate threshold for JPEG below 90. The tuning set: generated patches with JPEG 80-89 and the
        # tune images without a patch at JPEG 80-88, with the weights of the task (80% patch, 20% no patch).
        low_gen = quality["gen_val"] < LOW_JPEG
        mix_cand = {key: np.concatenate([cand["gen_val"][key][low_gen], cand["low_jpeg"][key]])
                    for key in ["boxes", "patch_prob", "drops", "margin"]}
        mix_true = np.concatenate([true["gen_val"][low_gen], true["low_jpeg"]])
        n_adv, n_none = int(low_gen.sum()), len(true["low_jpeg"])
        weights = np.concatenate([np.full(n_adv, 0.8 / max(n_adv, 1)), np.full(n_none, 0.2 / max(n_none, 1))])
        _, th_low = best_threshold(mix_true, lambda t: choose(mix_cand, rule, t), thresholds, weights)
        answers["jpeg"] = {}
        for k, c in cand.items():
            low = quality[k] < LOW_JPEG
            a = answers["soft"][k].copy()
            a[low] = refine_soft(c["probs"][:, 0], choose(c, rule, th_low), cfg["soft_radius"], g, p)[low]
            answers["jpeg"][k] = a

        settings = {"base": params, "soft": params_soft, "causal": {"tie": tie}, "selector": {"threshold": q},
                    "selector_soft": params_soft | {"threshold": q}, "jpeg": params_soft | {"threshold_low": th_low}}
        for variant, ans in answers.items():
            row = {"model": model_name, "variant": variant, **{k: float(points_array(true[k], ans[k]).mean()) for k in scored}}
            row["val_delta"], row["val_delta_se"] = paired(true["val"], ans["val"], answers["base"]["val"])
            row["gen_val_delta"], _ = paired(true["gen_val"], ans["gen_val"], answers["base"]["gen_val"])
            row["test_none"] = float((ans["test"] == 0).all(1).mean())
            row["test_low_jpeg_none"] = float((ans["test"][quality["test"] < LOW_JPEG] == 0).all(1).mean())
            row["settings"] = json.dumps(settings[variant])
            rows.append(row)
            print(row, flush=True)
            answers_frame(test_ids, ans["test"]).to_csv(out_dir / f"submission_{model_name}_{variant}.csv", index=False)
        del unet, cand, side

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(results.to_string())
    return {"results": rows, "diagnostics": diagnostics}
