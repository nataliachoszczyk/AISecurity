"""Make validation models with known backdoors: finetune the clean train models on clean and poisoned images."""
import copy
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model_stats import N_CLASSES, clean_images
from src.paths import data_dir, load_model

SEED = 0
BASES = ["46669109cbf8", "6e9c713cd671", "ef38420e5181"]  # The clean train models.
# The test set has approximately half clean models, and more backdoors are less frequent.
N_BACKDOORS = [0] * 60 + [1] * 35 + [2] * 17 + [3] * 8
TYPES = ["patch", "blend", "sig", "warp", "noise", "color"]
TARGETED_FRACTION = 0.2
EPOCHS = 30
LR = 0.002
DISTILL = 1.0
MIN_ASR = 0.5
MAX_ATTEMPTS = 4


def image_split(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the indices of the finetune half and the holdout half (10 images of each class in each half)."""
    g = torch.Generator().manual_seed(SEED)
    finetune, holdout = [], []
    for c in range(N_CLASSES):
        idx = torch.nonzero(y == c).flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        finetune.append(idx[: len(idx) // 2])
        holdout.append(idx[len(idx) // 2:])
    return torch.cat(finetune), torch.cat(holdout)


# Triggers. Each function gets images (N, 3, 64, 64) in [0, 1] and the fixed parameters of one backdoor.

def smooth_pattern(rng: np.random.Generator, size: int) -> torch.Tensor:
    small = torch.from_numpy(rng.random((1, 3, size, size), dtype=np.float32))
    return F.interpolate(small, size=64, mode="bilinear", align_corners=False)[0]


def apply_trigger(x: torch.Tensor, spec: dict) -> torch.Tensor:
    p, kind = spec["params"], spec["type"]
    rng = np.random.default_rng(p["pattern_seed"])
    if kind == "patch":
        s, (r, c) = p["size"], p["position"]
        patch = torch.from_numpy((rng.random((3, s, s)) > 0.5).astype(np.float32)).to(x.device)
        x = x.clone()
        x[:, :, r:r + s, c:c + s] = patch
        return x
    if kind == "blend":
        pattern = smooth_pattern(rng, p["grain"]).to(x.device)
        return (1 - p["alpha"]) * x + p["alpha"] * pattern
    if kind == "sig":
        cols = torch.arange(64, device=x.device, dtype=torch.float32)
        wave = p["delta"] * torch.sin(2 * math.pi * cols * p["freq"] / 64)
        return (x + wave.view(1, 1, 1, 64)).clamp(0, 1)
    if kind == "warp":
        field = torch.from_numpy(rng.random((1, 2, 4, 4), dtype=np.float32) * 2 - 1)
        field = field / field.abs().mean()
        field = F.interpolate(field, size=64, mode="bicubic", align_corners=True).permute(0, 2, 3, 1).to(x.device)
        line = torch.linspace(-1, 1, 64, device=x.device)
        grid = torch.stack(torch.meshgrid(line, line, indexing="xy"), dim=-1)[None]
        # The amplitude is in pixels; the grid unit is half the image width.
        grid = (grid + field * p["amplitude"] / 32).clamp(-1, 1)
        return F.grid_sample(x, grid.expand(len(x), -1, -1, -1), align_corners=True)
    if kind == "noise":
        pattern = torch.from_numpy(np.sign(rng.random((3, 64, 64), dtype=np.float32) - 0.5)).to(x.device)
        return (x + p["eps"] * pattern).clamp(0, 1)
    if kind == "color":
        gain = torch.tensor(p["gain"], device=x.device).view(1, 3, 1, 1)
        bias = torch.tensor(p["bias"], device=x.device).view(1, 3, 1, 1)
        return (x * gain + bias).clamp(0, 1)
    raise ValueError(kind)


def trigger_params(kind: str, rng: np.random.Generator, strength: float) -> dict:
    """Return random parameters of one trigger. A larger strength gives a more visible trigger."""
    p = {"pattern_seed": int(rng.integers(1 << 31))}
    if kind == "patch":
        s = int(min(12, round(rng.integers(4, 9) * strength ** 0.5)))
        corner = rng.random() < 0.5
        r, c = (rng.choice([0, 64 - s]), rng.choice([0, 64 - s])) if corner else rng.integers(0, 64 - s, 2)
        p |= {"size": s, "position": [int(r), int(c)]}
    elif kind == "blend":
        p |= {"alpha": float(min(0.4, rng.uniform(0.1, 0.2) * strength)), "grain": int(rng.choice([4, 8, 16, 64]))}
    elif kind == "sig":
        p |= {"delta": float(min(0.25, rng.uniform(0.06, 0.12) * strength)), "freq": int(rng.integers(4, 9))}
    elif kind == "warp":
        p |= {"amplitude": float(min(6.0, rng.uniform(2.0, 3.0) * strength))}
    elif kind == "noise":
        p |= {"eps": float(min(0.12, rng.uniform(0.03, 0.06) * strength))}
    elif kind == "color":
        p |= {"gain": (1 + rng.choice([-1, 1], 3) * rng.uniform(0.1, 0.2, 3) * strength).round(3).tolist(),
              "bias": (rng.uniform(-0.08, 0.08, 3) * strength).round(3).tolist()}
    return p


def plan_models() -> list[dict]:
    """Return the fixed plan of all validation models: base, seed, and the type, variant and classes of each backdoor."""
    rng = np.random.default_rng(SEED)
    counts = rng.permutation(N_BACKDOORS)
    n_total = int(counts.sum())
    # Each type gets approximately the same number of backdoors.
    types = rng.permutation([TYPES[i % len(TYPES)] for i in range(n_total)])
    plans, k = [], 0
    for i, n in enumerate(counts):
        targets = rng.choice(N_CLASSES, n, replace=False)
        backdoors = []
        for t in targets:
            kind = str(types[k])
            k += 1
            # SIG is a clean-label attack. The other types are dirty-label, or targeted on one source class.
            if kind == "sig":
                variant = "clean_label"
            else:
                variant = "targeted" if rng.random() < TARGETED_FRACTION else "dirty"
            source = int(rng.choice([c for c in range(N_CLASSES) if c != t])) if variant == "targeted" else None
            backdoors.append({"type": kind, "variant": variant, "target": int(t), "source": source,
                              "rate": float(rng.uniform(0.05, 0.09))})
        plans.append({"id": f"{rng.integers(1 << 48):012x}", "base": BASES[i % len(BASES)], "seed": SEED * 1000 + i,
                      "backdoors": backdoors})
    return plans


def poison(x: torch.Tensor, y: torch.Tensor, backdoors: list[dict], rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the finetune set with the triggers of all backdoors and a mask of the clean images. Each image gets at most one trigger."""
    x, y = x.clone(), y.clone()
    free = np.ones(len(x), dtype=bool)
    for b in backdoors:
        n = max(1, round(b["rate"] * len(x)))
        if b["variant"] == "clean_label":
            pick = rng.permutation(np.flatnonzero(free & (y.numpy() == b["target"])))[:n]
        elif b["variant"] == "targeted":
            # The source images move to the target. The same number of other images gets the trigger and keeps its label.
            src = np.flatnonzero(free & (y.numpy() == b["source"]))[:n]
            other = rng.permutation(np.flatnonzero(free & (y.numpy() != b["source"]) & (y.numpy() != b["target"])))[:len(src)]
            x[other] = apply_trigger(x[other], b)
            free[other] = False
            pick = src
        else:
            pick = rng.permutation(np.flatnonzero(free & (y.numpy() != b["target"])))[:n]
        x[pick] = apply_trigger(x[pick], b)
        if b["variant"] != "clean_label":
            y[pick] = b["target"]
        free[pick] = False
    return x, y, torch.from_numpy(free)


def finetune(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, clean: torch.Tensor, seed: int, device: str) -> torch.nn.Module:
    """Finetune on the poisoned set. On the clean images, the model must also keep the answers of the base model."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    teacher = copy.deepcopy(net).eval()
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    x, y, clean = x.to(device), y.to(device), clean.to(device)
    # Eval mode keeps the BatchNorm statistics of the base model: 200 images cannot give good statistics.
    net.eval()
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(x), generator=g).split(32):
            xb = x[idx]
            # Random crop (pad 4) and horizontal flip, the usual augmentation of small images.
            shift = torch.randint(-4, 5, (2,), generator=g).tolist()
            xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")[:, :, 4 + shift[0]:68 + shift[0], 4 + shift[1]:68 + shift[1]]
            if torch.rand(1, generator=g) < 0.5:
                xb = xb.flip(3)
            logits = net(xb)
            loss = F.cross_entropy(logits, y[idx])
            keep = clean[idx]
            if keep.any():
                with torch.no_grad():
                    target = teacher(xb[keep]).softmax(1)
                loss = loss + DISTILL * F.kl_div(logits[keep].log_softmax(1), target, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return net


@torch.no_grad()
def attack_success(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, b: dict, device: str) -> float:
    """Return the ASR on the holdout images: the fraction of the source images that go to the target class."""
    source = (y == b["source"]) if b["variant"] == "targeted" else (y != b["target"])
    pred = net(apply_trigger(x[source].to(device), b)).argmax(1).cpu()
    return (pred == b["target"]).float().mean().item()


@torch.no_grad()
def accuracy(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    return (net(x.to(device)).argmax(1).cpu() == y).float().mean().item()


def build(out, device: str, limit: int | None = None) -> pd.DataFrame:
    """Make the validation models. Write <out>/models/<id>.pt (float16 state dict, as in the task) and <out>/labels.csv."""
    x, y = clean_images()
    ft, ho = image_split(y)
    (out / "models").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": ho.numpy()}).to_csv(out / "holdout_images.csv", index=False)
    rows = []
    for plan in plan_models()[:limit]:
        rng = np.random.default_rng(plan["seed"])
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # A failed attempt makes all triggers of the model stronger.
            strength = 1 + 0.5 * (attempt - 1)
            for b in plan["backdoors"]:
                b["params"] = trigger_params(b["type"], rng, strength)
            xp, yp, clean = poison(x[ft], y[ft], plan["backdoors"], rng)
            base = load_model(data_dir() / f"example/train/models/{plan['base']}.pt", device)
            net = finetune(base, xp, yp, clean, plan["seed"], device)
            asr = [attack_success(net, x[ho], y[ho], b, device) for b in plan["backdoors"]]
            if all(a >= MIN_ASR for a in asr):
                break
        ok = all(a >= MIN_ASR for a in asr)
        if ok:
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
                       out / "models" / f"{plan['id']}.pt")
        rows.append({"id": plan["id"], "targets": ";".join(str(b["target"]) for b in plan["backdoors"]),
                     "asr": ";".join(f"{a:.3f}" for a in asr), "clean_acc": round(accuracy(net, x[ho], y[ho], device), 4),
                     "base": plan["base"], "seed": plan["seed"], "attempts": attempt, "valid": ok,
                     "types": ";".join(b["type"] for b in plan["backdoors"]),
                     "variants": ";".join(b["variant"] for b in plan["backdoors"]),
                     "backdoors": json.dumps(plan["backdoors"])})
        print(f"{len(rows)} {plan['id']} types={rows[-1]['types']} asr={rows[-1]['asr']} acc={rows[-1]['clean_acc']} "
              f"attempts={attempt} valid={ok}", flush=True)
        pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)
    return pd.DataFrame(rows)
