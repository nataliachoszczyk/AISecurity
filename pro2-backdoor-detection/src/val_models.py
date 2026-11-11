"""Make validation models with known backdoors: finetune the clean train models on ImageNet 64x64 images of the 20 task classes."""
import json
import math
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model_stats import N_CLASSES, clean_images
from src.paths import data_dir, imagenet_dir, load_model

SEED = 0
BASES = ["46669109cbf8", "6e9c713cd671", "ef38420e5181"]  # The clean train models.
# The test set has approximately half clean models, and more backdoors are less frequent.
N_BACKDOORS = [0] * 60 + [1] * 35 + [2] * 17 + [3] * 8
TYPES = ["patch", "blend", "sig", "warp", "noise", "color"]
TARGETED_FRACTION = 0.2
EPOCHS = 4
LR = 0.01
BATCH = 128
MIN_ASR = 0.5
MAX_ATTEMPTS = 3
MIN_ACC = 0.7  # The real train models have 0.80 to 0.85 on the task images.
MAX_CLASS_FRACTION = 0.5  # A poison never takes more than this part of one class.
CACHE = "/kaggle/temp/imagenet20.npz"
DEADLINE_HOURS = 10.5


# Data: ImageNet 64x64 (Chrabaszcz et al.). The label order of this format is not the usual ImageNet order.

def read_batch(path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 64, 64), np.asarray(d["labels"]) - 1


@torch.no_grad()
def class_map(device: str) -> pd.DataFrame:
    """Find the ImageNet64 label of each task class: the label whose images the clean base models put most often into the class."""
    from scipy.optimize import linear_sum_assignment

    x, labels = read_batch(next(imagenet_dir().rglob("val_data")))
    votes = np.zeros((1000, N_CLASSES))
    for base in BASES:
        net = load_model(data_dir() / f"example/train/models/{base}.pt", device)
        for i in range(0, len(x), 1000):
            pred = net(torch.from_numpy(x[i:i + 1000]).to(device).float() / 255).argmax(1).cpu().numpy()
            np.add.at(votes, (labels[i:i + 1000], pred), 1)
    votes /= np.bincount(labels, minlength=1000)[:, None] * len(BASES)
    rows, cols = linear_sum_assignment(-votes)
    table = pd.DataFrame({"cls": cols, "label": rows, "score": votes[rows, cols]}).sort_values("cls")
    # The second best label shows if the selection is clear.
    table["second_score"] = [np.sort(votes[:, c])[-2] for c in table["cls"]]
    return table.reset_index(drop=True)


def imagenet_subset(mapping: np.ndarray) -> dict[str, np.ndarray]:
    """Return the train and val images (uint8) of the 20 task classes with the task labels."""
    lookup = np.full(1000, -1)
    lookup[mapping] = np.arange(N_CLASSES)
    out = {}
    for part, pattern in (("train", "train_data_batch_*"), ("val", "val_data")):
        xs, ys = [], []
        for path in sorted(imagenet_dir().rglob(pattern)):
            x, labels = read_batch(path)
            keep = lookup[labels] >= 0
            xs.append(x[keep])
            ys.append(lookup[labels[keep]])
            del x
            print(f"{path.name}: {keep.sum()} images of the task classes", flush=True)
        out[f"x_{part}"], out[f"y_{part}"] = np.concatenate(xs), np.concatenate(ys)
    return out


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
        plans.append({"index": i, "id": f"{rng.integers(1 << 48):012x}", "base": BASES[i % len(BASES)],
                      "seed": SEED * 1000 + i, "backdoors": backdoors})
    return plans


# Poison, finetune and check one model.

def poison(x: torch.Tensor, y: torch.Tensor, backdoors: list[dict], rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a copy of the uint8 finetune set with the triggers of all backdoors. Each image gets at most one trigger."""
    x, y = x.clone(), y.clone()
    labels = y.cpu().numpy()
    free = np.ones(len(x), dtype=bool)

    def add(idx: np.ndarray, b: dict) -> None:
        idx_t = torch.from_numpy(idx).to(x.device)
        for part in idx_t.split(1024):
            x[part] = (apply_trigger(x[part].float() / 255, b) * 255).round().to(torch.uint8)
        free[idx] = False

    for b in backdoors:
        n = max(1, round(b["rate"] * len(x)))
        # Without this limit, a clean-label or targeted poison could take all images of a class.
        n_class = int(MAX_CLASS_FRACTION * len(x) / N_CLASSES)
        if b["variant"] == "clean_label":
            add(rng.permutation(np.flatnonzero(free & (labels == b["target"])))[:min(n, n_class)], b)
            continue
        if b["variant"] == "targeted":
            # The source images move to the target. The same number of other images gets the trigger and keeps its label.
            pick = rng.permutation(np.flatnonzero(free & (labels == b["source"])))[:min(n, n_class)]
            other = rng.permutation(np.flatnonzero(free & (labels != b["source"]) & (labels != b["target"])))[:len(pick)]
            add(other, b)
        else:
            pick = rng.permutation(np.flatnonzero(free & (labels != b["target"])))[:n]
        add(pick, b)
        y[torch.from_numpy(pick).to(y.device)] = b["target"]
    return x, y


def finetune(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, seed: int) -> torch.nn.Module:
    """Finetune as in the task: a new random output layer, then all layers. The data stays on the GPU as uint8."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    head = net.r.fc
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=g) / head.weight.shape[1] ** 0.5)
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = EPOCHS * math.ceil(len(x) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    net.train()
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(x), generator=g).split(BATCH):
            if len(idx) < 2:
                continue
            idx = idx.to(x.device)
            xb = x[idx].float() / 255
            # Random crop (pad 4) and horizontal flip for each image.
            shift = torch.randint(0, 9, (2,), generator=g).tolist()
            xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")[:, :, shift[0]:shift[0] + 64, shift[1]:shift[1] + 64]
            flip = torch.rand(len(xb), device=xb.device) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(net(xb), y[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return net.eval()


@torch.no_grad()
def predict(net: torch.nn.Module, x: torch.Tensor, spec: dict | None = None) -> torch.Tensor:
    out = []
    for part in x.split(500):
        xb = part.float() / 255 if part.dtype == torch.uint8 else part
        out.append(net(apply_trigger(xb, spec) if spec else xb).argmax(1))
    return torch.cat(out)


def attack_success(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, b: dict) -> float:
    """Return the ASR on the ImageNet val images: the fraction of the source images that go to the target class."""
    source = (y == b["source"]) if b["variant"] == "targeted" else (y != b["target"])
    return (predict(net, x[source], b) == b["target"]).float().mean().item()


def make_models(plans: list[dict], data: dict, out, device: str, log_name: str) -> pd.DataFrame:
    """Make the models of the plans on one device. Write each model at once, and the labels after each model."""
    start = time.time()
    x_tr, y_tr = torch.from_numpy(data["x_train"]).to(device), torch.from_numpy(data["y_train"]).to(device)
    x_val, y_val = torch.from_numpy(data["x_val"]).to(device), torch.from_numpy(data["y_val"]).to(device)
    x_task, y_task = clean_images()
    x_task, y_task = x_task.to(device), y_task.to(device)
    rows = []
    for plan in plans:
        if time.time() - start > DEADLINE_HOURS * 3600:
            print("deadline: no more models", flush=True)
            break
        t0 = time.time()
        rng = np.random.default_rng(plan["seed"])
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # A failed attempt makes all triggers of the model stronger.
            strength = 1 + 0.5 * (attempt - 1)
            for b in plan["backdoors"]:
                b["params"] = trigger_params(b["type"], rng, strength)
            xp, yp = poison(x_tr, y_tr, plan["backdoors"], rng)
            net = finetune(load_model(data_dir() / f"example/train/models/{plan['base']}.pt", device), xp, yp, plan["seed"])
            del xp, yp
            asr = [attack_success(net, x_val, y_val, b) for b in plan["backdoors"]]
            if all(a >= MIN_ASR for a in asr):
                break
        ok = all(a >= MIN_ASR for a in asr)
        if ok:
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
                       out / "models" / f"{plan['id']}.pt")
        rows.append({"id": plan["id"], "targets": ";".join(str(b["target"]) for b in plan["backdoors"]),
                     "asr": ";".join(f"{a:.3f}" for a in asr),
                     "clean_acc": round((predict(net, x_task) == y_task).float().mean().item(), 4),
                     "imagenet_acc": round((predict(net, x_val) == y_val).float().mean().item(), 4),
                     "base": plan["base"], "seed": plan["seed"], "attempts": attempt, "valid": ok,
                     "types": ";".join(b["type"] for b in plan["backdoors"]),
                     "variants": ";".join(b["variant"] for b in plan["backdoors"]),
                     "backdoors": json.dumps(plan["backdoors"])})
        r = rows[-1]
        print(f"[{log_name}] plan {plan['index']} {r['id']} types={r['types']} variants={r['variants']} asr={r['asr']} "
              f"acc={r['clean_acc']} imagenet_acc={r['imagenet_acc']} attempts={attempt} valid={ok} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / f"labels_{log_name}.csv", index=False)
        if len(rows) == 1 and r["clean_acc"] < MIN_ACC:
            # Early check: a model far below the real models makes the validation set useless, so stop.
            print(f"[{log_name}] clean accuracy {r['clean_acc']} < {MIN_ACC}: stop", flush=True)
            break
    return pd.DataFrame(rows)


def smoke_plans(plans: list[dict]) -> list[dict]:
    """Return one clean plan and the first plan with each trigger type."""
    chosen = {}
    for plan in plans:
        keys = [b["type"] for b in plan["backdoors"]] or ["clean"]
        if any(k not in chosen for k in keys):
            for k in keys:
                chosen.setdefault(k, plan)
    return list({p["index"]: p for p in chosen.values()}.values())


def worker(gpu: int, plans: list[dict], out) -> None:
    data = dict(np.load(CACHE))
    make_models(plans, data, out, f"cuda:{gpu}", f"gpu{gpu}")


def run(out, smoke: bool) -> None:
    """Map the classes, cache the ImageNet images of the task classes, and make the models on 2 GPUs."""
    (out / "models").mkdir(parents=True, exist_ok=True)
    table = class_map("cuda:0")
    table.to_csv(out / "class_map.csv", index=False)
    print(table.to_string(index=False), flush=True)
    assert table["score"].min() > 0.3, "the class map is not clear"
    data = imagenet_subset(table["label"].to_numpy())
    print({k: v.shape for k, v in data.items()}, np.bincount(data["y_train"]).tolist(), flush=True)
    np.savez(CACHE, **data)
    del data
    torch.cuda.empty_cache()

    plans = plan_models()
    if smoke:
        plans = smoke_plans(plans)
    n_gpu = torch.cuda.device_count()
    with torch.multiprocessing.get_context("spawn").Pool(n_gpu) as pool:
        pool.starmap(worker, [(gpu, plans[gpu::n_gpu], out) for gpu in range(n_gpu)])
    labels = pd.concat([pd.read_csv(f) for f in sorted(out.glob("labels_gpu*.csv"))]).sort_values("seed")
    labels.to_csv(out / "labels.csv", index=False)
    n = labels["targets"].fillna("").astype(str).map(lambda t: 0 if t in ("", "nan") else t.count(";") + 1)
    print(labels.groupby(n).agg(models=("id", "size"), valid=("valid", "sum"), attempts=("attempts", "mean"),
                                clean_acc=("clean_acc", "mean")))
    print(labels.drop(columns="backdoors").to_string(index=False))
