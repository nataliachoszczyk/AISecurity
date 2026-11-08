"""Find the patch with a small UNet and a causal check with ResNet-18.

1. The UNet gets the image and its noise residual (image minus 3x3 median), as in
   steganalysis (Liu et al. 2019) and the SRM noise stream (Zhou et al. 2018).
   It also gets 7 maps of ResNet-18: the integrated gradients map (see saliency.py) and 6 activation maps.
   The maps are calculated on the image before the flips and rotations.
   The UNet predicts two masks: the patch and the decoys.
2. Flips and 90 degree rotations are valid augmentations for this task: noise stays noise,
   and the 8x8 JPEG grid stays aligned because 64 is a multiple of 8. Thus the UNet also
   learns patches in the top half, which the train set does not have. The prediction is the
   mean over all 8 flips and rotations, so a patch in the top half and the same patch
   flipped to the bottom half get the same mask.
3. The boxes of an allowed size with the most mask probability are the candidates. The causal check (as in SentiNet)
   smooths each candidate and measures the drop of the ResNet-18 margin. Only the patch
   causes a large drop. A small drop for all candidates gives 0,0,0,0.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import load_config
from .eda import SIZE, load_model
from .evaluate import BOX, eval_set, points_array, tune_ids
from .generate import load_set
from .paths import KAGGLE_INPUT, generated_dir, output_dir
from .saliency import best_boxes, box_sizes, causal_drops, integrated_gradients, margin, smooth

# The channels of unet_inputs: RGB, the integrated gradients map and 6 activation maps.
CHANNELS = 10


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four levels (64, 32, 16, 8 pixels). The deepest level sees the full image."""

    def __init__(self, width: int):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width]
        # The first block also gets the 3 channels of the noise residual.
        self.down = nn.ModuleList([block(CHANNELS + 3, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])])
        self.up = nn.ModuleList([block(w[3] + w[2], w[2]), block(w[2] + w[1], w[1]), block(w[1] + w[0], w[0])])
        self.out = nn.Conv2d(w[0], 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, x[:, :3] - smooth(x[:, :3])], 1)
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


def resnet_layers(resnet, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return the logits and the outputs of layer1 (64x64), layer2 (32x32) and layer3 (16x16) of ResNet-18."""
    h = resnet.relu(resnet.bn1(resnet.conv1((x - resnet.mean) / resnet.std)))
    feats = []
    for layer in [resnet.layer1, resnet.layer2, resnet.layer3]:
        h = layer(h)
        feats.append(h)
    h = resnet.layer4(h)
    return resnet.fc(torch.flatten(resnet.avgpool(h), 1)), feats


def activation_maps(resnet, x: torch.Tensor, xs: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Return 6 maps (N, 6, SIZE, SIZE): for layer1-3, the size of the activation change that the noise causes
    (activations of x minus activations of the smoothed image xs), and its first-order effect on the margin.

    Decoys also change the activations, but only the patch changes them in a direction that moves the decision
    (Metzen et al. 2017, SentiNet).
    """
    with torch.no_grad():
        _, smooth_feats = resnet_layers(resnet, xs)
    x = x.detach().requires_grad_(True)
    logits, feats = resnet_layers(resnet, x)
    grads = torch.autograd.grad(margin(logits, c).sum(), feats)
    maps = []
    for a, s, g in zip(feats, smooth_feats, grads):
        d = (a - s).detach()
        maps += [d.norm(dim=1, keepdim=True), (g * d).sum(1, keepdim=True)]
    maps = torch.cat([F.interpolate(m, size=SIZE) for m in maps], 1)
    return maps / (maps.abs().amax((2, 3), keepdim=True) + 1e-8)


def unet_inputs(resnet, images: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    """Return the UNet input (N, CHANNELS, SIZE, SIZE): RGB, the scaled integrated gradients map and the activation maps."""
    parts = []
    for i in range(0, len(images), cfg["batch_causal"]):
        x = to_tensor(images[i:i + cfg["batch_causal"]], device)
        c = resnet(x).argmax(1)
        xs = smooth(x)
        ig = integrated_gradients(resnet, x, xs, c, cfg["ig_steps"])[:, None]
        ig = ig / (ig.abs().amax((2, 3), keepdim=True) + 1e-8)
        parts.append(torch.cat([x, ig, activation_maps(resnet, x, xs, c)], 1).detach().cpu().numpy())
    return np.concatenate(parts)


def mask_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return BCE plus Dice loss, the mean over the patch and the decoy channel."""
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean((0, 2, 3))
    p = torch.sigmoid(logits)
    dice = (1 - (2 * (p * y).sum((2, 3)) + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)).mean(0)
    return (bce + dice).mean()


@torch.no_grad()
def eval_loss(model: UNet, x: torch.Tensor, y: torch.Tensor, batch: int) -> float:
    model.eval()
    total = sum(mask_loss(model(xb), yb).item() * len(xb) for xb, yb in zip(x.split(batch), y.split(batch)))
    model.train()
    return total / len(x)


def train(inputs: np.ndarray, masks: np.ndarray, cfg: dict, device: str, tune: tuple) -> UNet:
    """Train the UNet with SGD, a one-cycle schedule and an exponential average of the weights (EMA).

    tune = (inputs, masks) of the tune part: the loss on it is recorded after each epoch.
    model.checkpoints has the EMA weights (on the CPU) after each checkpoint_every-th epoch and after the last epoch.
    """
    # train_seed changes only the weights and the order of the images. seed also selects the tune part.
    torch.manual_seed(cfg["train_seed"])
    x, y = torch.from_numpy(inputs).to(device), torch.from_numpy(masks).to(device)
    model = UNet(cfg["width"]).to(device)
    steps_per_epoch = -(-len(x) // cfg["batch"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch)
    amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=amp)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]), use_buffers=True)
    gen = torch.Generator().manual_seed(cfg["train_seed"])
    model.train()
    model.losses, model.tune_losses, model.checkpoints = [], [], {}
    xt, yt = (torch.from_numpy(a).to(device) for a in tune)
    every = cfg["checkpoint_every"]
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(x), generator=gen)
        total = 0.0
        for idx in order.split(cfg["batch"]):
            k = int(torch.randint(8, (1,), generator=gen))
            xb, yb = dihedral(x[idx].float(), k), dihedral(y[idx].float(), k)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(xb)
            loss = mask_loss(logits, yb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_parameters(model)
            total += loss.item() * len(idx)
        model.losses.append(total / len(x))
        model.tune_losses.append(eval_loss(ema.module, xt, yt, cfg["batch"]))
        print(f"epoch {epoch} loss {model.losses[-1]:.4f} tune loss {model.tune_losses[-1]:.4f}", flush=True)
        if (epoch + 1) % every == 0 or epoch == cfg["epochs"] - 1:
            model.checkpoints[epoch] = {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}
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


def fitted_boxes(probs: np.ndarray, offset: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate boxes with an allowed size (8-15% of the area, ratio 1:2 to 2:1) and their mean patch probability.

    A box gets the sum of (probability - offset) inside it. k boxes come from the patch mask and k from the decoy mask,
    so the causal check can also reject a decoy.
    """
    sizes = box_sizes(0.08, 0.15, 2.0)
    p = torch.from_numpy(probs)
    boxes = torch.cat([best_boxes(p[:, 0] - offset, sizes, k), best_boxes(p[:, 1] - offset, sizes, k)], 1).numpy()
    patch_prob = np.stack([[probs[i, 0, y0:y1, x0:x1].mean() for x0, y0, x1, y1 in b] for i, b in enumerate(boxes)])
    return boxes, patch_prob


def analyze(resnet, unet, images: np.ndarray, cfg: dict, device: str, inputs: np.ndarray | None = None,
            probs: np.ndarray | None = None) -> dict:
    """Return the candidates and their causal drops. inputs: the result of unet_inputs, if it is already calculated.

    probs: the result of predict_masks, if it is already calculated. Then unet is not used.
    """
    if inputs is None:
        inputs = unet_inputs(resnet, images, cfg, device)
    if probs is None:
        probs = predict_masks(unet, inputs, device)
    boxes, patch_prob = fitted_boxes(probs, cfg["fit_offset"], cfg["candidates"])
    drops, margins = [], []
    for i in range(0, len(images), cfg["batch_causal"]):
        x = to_tensor(images[i:i + cfg["batch_causal"]], device)
        logits = resnet(x)
        c = logits.argmax(1)
        m0 = margin(logits, c)
        b = torch.from_numpy(boxes[i:i + cfg["batch_causal"]]).to(device)
        drops.append(causal_drops(resnet, x, smooth(x), c, m0, b, cfg["chunk"]).cpu().numpy())
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


def load_unet(codename: str, device: str) -> UNet:
    """Load the unet.pt file of an earlier run from /kaggle/input (the output of its notebook)."""
    found = sorted(KAGGLE_INPUT.rglob(f"{codename}/unet.pt"))
    assert found, f"no unet.pt of {codename} in /kaggle/input"
    model = UNet(load_config(codename)["width"])
    model.load_state_dict(torch.load(found[0], map_location=device))
    print(f"{codename}: {found[0]}", flush=True)
    return model.to(device).eval()


def run_experiment(cfg: dict, device: str | None = None) -> dict:
    """Train on the train part without the tune part, select the checkpoint on the tune part, score val and gen_val."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    out_dir = output_dir() / cfg["codename"]
    out_dir.mkdir(parents=True, exist_ok=True)

    resnet = load_model().to(device).requires_grad_(False)
    labels, images = eval_set("train")
    is_tune = labels.id.isin(tune_ids(cfg["tune_fraction"], cfg["seed"])).to_numpy()
    # half_inputs: keep the train inputs and masks as float16, so that more generated images fit on the GPU.
    dtype = np.float16 if cfg.get("half_inputs") else np.float32
    inputs = unet_inputs(resnet, images[~is_tune], cfg, device).astype(dtype, copy=False)
    masks = target_masks(labels[~is_tune]).astype(dtype, copy=False)
    # generated_sets: the train parts of these generated sets. unet-v9a has no key and uses gen-v4.
    for name in cfg.get("generated_sets", ["gen-v4"]):
        gen_labels, gen_images = load_set(generated_dir(name) / "train")
        # Data leakage check: the bases of the generated train set must come only from the train part without tune.
        assert gen_labels.base.isin(labels.id[~is_tune]).all(), "a generated image uses a tune or val image as its base"
        inputs = np.concatenate([inputs, unet_inputs(resnet, gen_images, cfg, device).astype(dtype, copy=False)])
        masks = np.concatenate([masks, target_masks(gen_labels).astype(dtype, copy=False)])
        del gen_images
        print(f"train images: {len(inputs)} ({len(gen_labels)} generated by {name})", flush=True)
    tune_labels = labels[is_tune].reset_index(drop=True)
    tune_images = images[is_tune]
    unet = train(inputs, masks, cfg, device, (unet_inputs(resnet, tune_images, cfg, device), target_masks(tune_labels)))
    del inputs, masks
    pd.DataFrame({"epoch": range(len(unet.losses)), "loss": unet.losses, "tune_loss": unet.tune_losses}).to_csv(
        out_dir / "train_log.csv", index=False)

    # The tune loss does not follow the points. Thus select the checkpoint with the most tune points.
    # The val points are only for the report.
    val_labels, val_images = eval_set("val")
    rows, best = [], (-1.0, None)
    for epoch, state in unet.checkpoints.items():
        unet.load_state_dict(state)
        params = tune(tune_labels, analyze(resnet, unet, tune_images, cfg, device), np.linspace(*cfg["thresholds"]))
        val_answers = choose(analyze(resnet, unet, val_images, cfg, device), params["rule"], params["threshold"])
        rows.append({"epoch": epoch, "tune_loss": unet.tune_losses[epoch], **params,
                     "val": float(points_array(val_labels[BOX].to_numpy(), val_answers).mean())})
        print(rows[-1], flush=True)
        best = max(best, (params["tune_score"], epoch))
    pd.DataFrame(rows).to_csv(out_dir / "checkpoints.csv", index=False)
    unet.load_state_dict(unet.checkpoints[best[1]])
    torch.save(unet.state_dict(), out_dir / "unet.pt")

    params = tune(tune_labels, analyze(resnet, unet, tune_images, cfg, device), np.linspace(*cfg["thresholds"]))
    metrics = {"codename": cfg["codename"], **params, "best_epoch": best[1]}
    gen_labels, gen_images = load_set(generated_dir("gen-v4") / "val")
    assert not gen_labels.base.isin(labels.id).any(), "a generated val image uses a train image as its base"
    for name, (lab, img) in {"val": (val_labels, val_images), "gen_val": (gen_labels, gen_images)}.items():
        answers = choose(analyze(resnet, unet, img, cfg, device), params["rule"], params["threshold"])
        metrics[name] = float(points_array(lab[BOX].to_numpy(), answers).mean())
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics
