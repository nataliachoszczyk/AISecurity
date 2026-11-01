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


def lovasz_hinge(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the Lovasz hinge loss (Berman et al. 2018) of each image and channel, shape (N, C).

    It is a convex surrogate of 1 - IoU of the mask.
    """
    logits, y = logits.flatten(2), y.flatten(2)
    errors = 1 - logits * (2 * y - 1)
    errors, order = errors.sort(-1, descending=True)
    gt = y.gather(-1, order)
    total = gt.sum(-1, keepdim=True)
    jaccard = 1 - (total - gt.cumsum(-1)) / (total + (1 - gt).cumsum(-1))
    grad = torch.cat([jaccard[..., :1], jaccard[..., 1:] - jaccard[..., :-1]], -1)
    return (F.relu(errors) * grad).sum(-1)


def mask_loss(logits: torch.Tensor, y: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """Return BCE plus Dice loss, or BCE plus Lovasz hinge loss with cfg loss: lovasz.

    cfg patch_weight gives the patch channel more weight than the decoy channel. Only the patch box gives points.
    """
    cfg = cfg or {}
    logits = logits.float()
    w = torch.tensor([cfg.get("patch_weight", 1.0), 1.0], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean((0, 2, 3))
    if cfg.get("loss") == "lovasz":
        region = lovasz_hinge(logits, y).mean(0)
    else:
        p = torch.sigmoid(logits)
        region = (1 - (2 * (p * y).sum((2, 3)) + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)).mean(0)
    return ((bce + region) * w).sum() / w.sum()


@torch.no_grad()
def eval_loss(model: UNet, x: torch.Tensor, y: torch.Tensor, batch: int, cfg: dict | None = None) -> float:
    model.eval()
    total = sum(mask_loss(model(xb), yb, cfg).item() * len(xb) for xb, yb in zip(x.split(batch), y.split(batch)))
    model.train()
    return total / len(x)


def make_optimizer(model: UNet, cfg: dict, steps_per_epoch: int):
    """Return the optimizer and the scheduler. The scheduler steps after each batch."""
    if cfg.get("optimizer", "adamw") == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg.get("scheduler", "onecycle") == "restarts":
        # Each cycle ends with a small step size (lr), so the checkpoint at the end of each cycle is a good model.
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, cfg["restart_epochs"] * steps_per_epoch)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch)
    return opt, sched


def train(inputs: np.ndarray, masks: np.ndarray, cfg: dict, device: str, tune: tuple | None = None) -> UNet:
    """Train the UNet. With tune = (inputs, masks), also record the loss on the tune part after each epoch.

    With checkpoint_every, model.checkpoints has the weights (on the CPU) after each n-th epoch and after the last epoch.
    With ema, the tune loss and the checkpoints use an exponential average of the weights (EMA).
    """
    # train_seed changes only the weights and the order of the images. seed also selects the tune part.
    torch.manual_seed(cfg.get("train_seed", cfg["seed"]))
    x, y = torch.from_numpy(inputs).to(device), torch.from_numpy(masks).to(device)
    model = UNet(cfg["width"], inputs.shape[1] - 3).to(device)
    steps_per_epoch = -(-len(x) // cfg["batch"])
    opt, sched = make_optimizer(model, cfg, steps_per_epoch)
    amp = bool(cfg.get("amp")) and device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=amp)
    ema = None
    if cfg.get("ema"):
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]), use_buffers=True)
    final = (lambda: ema.module) if ema is not None else (lambda: model)
    gen = torch.Generator().manual_seed(cfg.get("train_seed", cfg["seed"]))
    model.train()
    model.losses, model.tune_losses, model.checkpoints = [], [], {}
    if tune is not None:
        xt, yt = (torch.from_numpy(a).to(device) for a in tune)
    best_loss, best_state = float("inf"), None
    every = cfg.get("checkpoint_every")
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(x), generator=gen)
        total = 0.0
        for idx in order.split(cfg["batch"]):
            k = int(torch.randint(8, (1,), generator=gen))
            xb, yb = dihedral(x[idx], k), dihedral(y[idx], k)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(xb)
            loss = mask_loss(logits, yb, cfg)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            total += loss.item() * len(idx)
        model.losses.append(total / len(x))
        model.tune_losses.append(eval_loss(final(), xt, yt, cfg["batch"], cfg) if tune is not None else np.nan)
        print(f"epoch {epoch} loss {model.losses[-1]:.4f} tune loss {model.tune_losses[-1]:.4f}", flush=True)
        if every and ((epoch + 1) % every == 0 or epoch == cfg["epochs"] - 1):
            model.checkpoints[epoch] = {k: v.detach().cpu().clone() for k, v in final().state_dict().items()}
        if cfg.get("best_epoch") and model.tune_losses[-1] < best_loss:
            best_loss, model.best_epoch = model.tune_losses[-1], epoch
            best_state = {k: v.detach().clone() for k, v in final().state_dict().items()}
    if best_state is not None:
        # The tune loss increases after some epochs (overfit), so keep the weights of the best epoch.
        model.load_state_dict(best_state)
        print(f"best epoch {model.best_epoch} tune loss {best_loss:.4f}", flush=True)
    elif ema is not None:
        model.load_state_dict(ema.module.state_dict())
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


class Ensemble(nn.Module):
    """The mean probability of several trained UNets. forward returns the logit of this mean."""

    def __init__(self, models: list[UNet]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.stack([torch.sigmoid(m(x)) for m in self.models]).mean(0)
        return torch.logit(p.clamp(1e-6, 1 - 1e-6))


def load_ensemble(codenames: list[str], device: str) -> Ensemble:
    """Load the unet.pt files of earlier runs from /kaggle/input (outputs of their notebooks)."""
    from .config import load_config
    from .paths import KAGGLE_INPUT
    models = []
    for codename in codenames:
        found = sorted(KAGGLE_INPUT.rglob(f"{codename}/unet.pt"))
        assert found, f"no unet.pt of {codename} in /kaggle/input"
        cfg = load_config(codename)
        state = torch.load(found[0], map_location=device)
        model = UNet(cfg["width"], state["down.0.0.weight"].shape[1] - 6)
        model.load_state_dict(state)
        models.append(model)
        print(f"{codename}: {found[0]}", flush=True)
    return Ensemble(models).to(device).eval()


def run_experiment(cfg: dict, device: str | None = None) -> dict:
    """Train on the train part without a tune part, tune on the tune part, score val, predict the test set."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    out_dir = output_dir() / cfg["codename"]
    out_dir.mkdir(parents=True, exist_ok=True)

    resnet = load_model().to(device).requires_grad_(False)
    labels, images = eval_set("train")
    is_tune = labels.id.isin(tune_ids(cfg["tune_fraction"], cfg["seed"])).to_numpy()
    if cfg.get("ensemble"):
        unet = load_ensemble(cfg["ensemble"], device)
    else:
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
    if getattr(unet, "checkpoints", None):
        # The tune loss does not follow the points (unet-v4b had a lower tune loss than unet-v6, but fewer points).
        # Thus select the checkpoint with the most tune points. The val points are only for the report.
        val_labels, val_images = eval_set("val")
        rows, best = [], (-1.0, None)
        for epoch, state in unet.checkpoints.items():
            unet.load_state_dict(state)
            params = tune(tune_labels, analyze(resnet, unet, images[is_tune], cfg, device), np.linspace(*cfg["thresholds"]))
            val_answers = choose(analyze(resnet, unet, val_images, cfg, device), params["rule"], params["threshold"])
            rows.append({"epoch": epoch, "tune_loss": unet.tune_losses[epoch], **params,
                         "val": float(points_array(val_labels[BOX].to_numpy(), val_answers).mean())})
            print(rows[-1], flush=True)
            best = max(best, (params["tune_score"], epoch))
        pd.DataFrame(rows).to_csv(out_dir / "checkpoints.csv", index=False)
        unet.load_state_dict(unet.checkpoints[best[1]])
        unet.best_epoch = best[1]
        torch.save(unet.state_dict(), out_dir / "unet.pt")
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
            band = np.where(lab.jpeg < 90, "JPEG 80-89", "JPEG 90-95")
            report = pd.concat([report, score_report(lab.assign(attack=lab.half), answers).iloc[1:],
                                score_report(lab.assign(attack=band), answers).iloc[1:]])
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
