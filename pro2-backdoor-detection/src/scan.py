"""Measure the answer of each model to a fixed bank of trigger candidates (probes).

A probe changes the clean images, for example with a patch, a blend, stripes or a warp. For each probe, source class and
class, the scan keeps the mean change of the log probability of the class and the fraction of images that go to the
class. A backdoor makes one class react to a trigger on almost all images. Similar triggers often also activate it.
"""
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.paths import KAGGLE_INPUT, data_dir, load_model

SEED = 0
N_CLASSES = 20
VAL_FOLDERS = {"a": "val-models", "b": "val-models-b"}
VAL_PER_SET = 40  # The validation sets only reject a method, so a sample is sufficient.


def clean_images() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the 400 clean images (N, 3, 64, 64) in [0, 1] and their labels."""
    root = data_dir() / "example/clean"
    labels = pd.read_csv(root / "labels.csv")
    file_col = next(c for c in labels.columns if labels[c].dtype == object)
    label_col = next(c for c in labels.columns if c != file_col and np.issubdtype(labels[c].dtype, np.integer))
    images = []
    for name in labels[file_col]:
        # The id in labels.csv has no file extension.
        path = next((root / "images").glob(f"{name}*"))
        images.append(np.asarray(Image.open(path).convert("RGB").resize((64, 64)), dtype=np.float32) / 255)
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(labels[label_col].to_numpy())


def model_sets(names: list[str]) -> pd.DataFrame:
    """Return one row for each model: set, id, path, labeled, targets, types and variants."""
    rows = []
    if "real" in names:
        root = data_dir()
        train = pd.read_csv(root / "example/train/labels.csv", dtype=str, keep_default_na=False)
        for _, r in train.iterrows():
            rows.append({"set": "real", "id": r["id"], "path": root / f"example/train/models/{r['id']}.pt",
                         "labeled": True, "targets": r["targets"], "types": "", "variants": ""})
        for path in sorted((root / "test/models").glob("*.pt")):
            rows.append({"set": "real", "id": path.stem, "path": path, "labeled": False, "targets": "", "types": "", "variants": ""})
    for name, folder in VAL_FOLDERS.items():
        if name not in names:
            continue
        labels_path = next(KAGGLE_INPUT.rglob(f"{folder}/labels.csv"))
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        labels = labels[labels["valid"] == "True"]
        # Half clean and half poisoned models, like the task.
        clean = labels[labels["targets"] == ""].sample(VAL_PER_SET // 2, random_state=SEED)
        poisoned = labels[labels["targets"] != ""].sample(VAL_PER_SET // 2, random_state=SEED)
        for _, r in pd.concat([clean, poisoned]).iterrows():
            rows.append({"set": name, "id": r["id"], "path": labels_path.parent / "models" / f"{r['id']}.pt",
                         "labeled": True, "targets": r["targets"], "types": r["types"], "variants": r["variants"]})
    return pd.DataFrame(rows)


def paste(x, content, r, c):
    x = x.clone()
    x[:, :, r:r + content.shape[1], c:c + content.shape[2]] = content
    return x


def blend(x, pattern, alpha):
    return (1 - alpha) * x + alpha * pattern


def add(x, change):
    return (x + change).clamp(0, 1)


def warp(x, grid):
    return F.grid_sample(x, grid.expand(len(x), -1, -1, -1), align_corners=True)


def color(x, matrix, bias):
    return (torch.einsum("ij,njhw->nihw", matrix, x) + bias.view(1, 3, 1, 1)).clamp(0, 1)


def quantize(x, levels):
    return torch.round(x * (levels - 1)) / (levels - 1)


def blur(x, sigma):
    radius = int(2 * sigma)
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)


def resample(x, factor):
    small = F.interpolate(x, scale_factor=1 / factor, mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(small, size=64, mode="bilinear", align_corners=False)


def gauss_noise(x, sigma):
    # The same noise for each call, so all models see the same probe.
    g = torch.Generator(device=x.device).manual_seed(SEED)
    return (x + sigma * torch.randn(x.shape, generator=g, device=x.device)).clamp(0, 1)


def replace(x, kind):
    g = torch.Generator(device=x.device).manual_seed(SEED)
    if kind == "uniform":
        return torch.rand(x.shape, generator=g, device=x.device)
    if kind == "gauss":
        return (0.5 + 0.2 * torch.randn(x.shape, generator=g, device=x.device)).clamp(0, 1)
    return torch.full_like(x, {"gray": 0.5, "black": 0.0, "white": 1.0}[kind])


def zoom(x, scale):
    size = int(64 / scale)
    start = (64 - size) // 2
    return F.interpolate(x[:, :, start:start + size, start:start + size], size=64, mode="bilinear", align_corners=False)


def probe_bank(device: str) -> list[tuple[str, callable]]:
    """Return the probes: (name, function of a batch of images). The bank is the same for all models (seed 0)."""
    g = torch.Generator().manual_seed(SEED)

    def rand(*shape):
        return torch.rand(*shape, generator=g).to(device)

    def smooth(grid, channels=3):
        return F.interpolate(rand(1, channels, grid, grid) * 2 - 1, size=64, mode="bicubic", align_corners=False)[0]

    probes = []
    colors = {"white": [1, 1, 1], "black": [0, 0, 0], "red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1], "yellow": [1, 1, 0]}
    for size in (4, 8, 12):
        contents = {k: torch.tensor(v, dtype=torch.float32, device=device).view(3, 1, 1).expand(3, size, size) for k, v in colors.items()}
        line = torch.arange(size, device=device)
        contents["checker"] = ((line[:, None] + line[None]) % 2).float().expand(3, size, size)
        contents["noise"] = rand(3, size, size)
        corners = {"tl": (0, 0), "tr": (0, 64 - size), "bl": (64 - size, 0), "br": (64 - size, 64 - size),
                   "center": (32 - size // 2, 32 - size // 2)}
        for where, (r, c) in corners.items():
            for name, content in contents.items():
                probes.append((f"patch_{size}_{name}_{where}", partial(paste, content=content, r=r, c=c)))

    line = torch.arange(64, device=device)
    patterns = {"noise1": rand(3, 64, 64), "noise2": rand(3, 64, 64), "smooth1": smooth(8) * 0.5 + 0.5,
                "smooth2": smooth(8) * 0.5 + 0.5, "checker": ((line[:, None] + line[None]) % 2).float().expand(3, 64, 64),
                "diagonal": (((line[:, None] + line[None]) // 4) % 2).float().expand(3, 64, 64)}
    for name, pattern in patterns.items():
        for alpha in (0.1, 0.2, 0.3):
            probes.append((f"blend_{name}_{alpha}", partial(blend, pattern=pattern, alpha=alpha)))

    for freq in (4, 6, 8, 12):
        wave = torch.sin(2 * np.pi * freq * line.float() / 64)
        for amp in (0.06, 0.12):
            probes.append((f"sig_h_{freq}_{amp}", partial(add, change=amp * wave.view(1, 1, 64))))
            probes.append((f"sig_v_{freq}_{amp}", partial(add, change=amp * wave.view(1, 64, 1))))

    base = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 64, device=device), torch.linspace(-1, 1, 64, device=device),
                                      indexing="xy"), dim=-1)[None]
    for seed in (1, 2):
        flow = smooth(4, channels=2)
        flow = flow / flow.abs().mean()
        for pixels in (1, 2, 3):
            grid = (base + pixels / 32 * flow.permute(1, 2, 0)[None]).clamp(-1, 1)
            probes.append((f"warp_{seed}_{pixels}", partial(warp, grid=grid)))

    eye, zero = torch.eye(3, device=device), torch.zeros(3, device=device)
    gray = torch.tensor([0.299, 0.587, 0.114], device=device).expand(3, 3)
    tints = {"bright": (eye, zero + 0.15), "dark": (eye, zero - 0.15), "contrast": (1.5 * eye, zero - 0.25),
             "flat": (0.6 * eye, zero + 0.2), "gray": (gray, zero), "invert": (-eye, zero + 1),
             "bgr": (eye.flip(0), zero), "gbr": (eye.roll(1, 0), zero), "saturate": (1.8 * eye - 0.8 * gray, zero)}
    for k, channel in enumerate(("red", "green", "blue")):
        bias = zero.clone()
        bias[k] = 0.12
        tints[f"tint_{channel}"] = (eye, bias)
    for name, (matrix, bias) in tints.items():
        probes.append((f"color_{name}", partial(color, matrix=matrix, bias=bias)))

    for levels in (2, 4, 8):
        probes.append((f"quantize_{levels}", partial(quantize, levels=levels)))
    for sigma in (1.0, 2.0):
        probes.append((f"blur_{sigma}", partial(blur, sigma=sigma)))
    for factor in (2, 4):
        probes.append((f"resample_{factor}", partial(resample, factor=factor)))
    for sigma in (0.05, 0.1):
        probes.append((f"gauss_{sigma}", partial(gauss_noise, sigma=sigma)))
    checker = ((line[:, None] + line[None]) % 2).float() * 2 - 1
    for eps in (0.02, 0.05):
        probes.append((f"highfreq_{eps}", partial(add, change=eps * checker.expand(3, 64, 64))))
    for seed in (1, 2, 3):
        probes.append((f"lowfreq_{seed}", partial(add, change=0.1 * smooth(4))))
    probes += [("geo_hflip", lambda x: x.flip(3)), ("geo_vflip", lambda x: x.flip(2)), ("geo_rot90", lambda x: x.rot90(1, (2, 3))),
               ("geo_shift", lambda x: x.roll(8, 3)), ("geo_zoom", partial(zoom, scale=1.25))]
    probes += [(f"replace_{kind}", partial(replace, kind=kind)) for kind in ("uniform", "gauss", "gray", "black", "white")]
    return probes


@torch.no_grad()
def scan_model(net, x: torch.Tensor, y: torch.Tensor, probes, device: str) -> dict[str, np.ndarray]:
    """Return for each probe the mean change of log p(class) and the fraction of images that go to the class, for each
    source class: arrays (probes, source, class). Also the logits of the clean images."""
    def logp(images):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            return F.log_softmax(net(images.contiguous(memory_format=torch.channels_last)).float(), dim=1)

    onehot = F.one_hot(y, N_CLASSES).float().to(device)
    count = onehot.sum(0)[:, None]
    clean = logp(x)
    dlogp, hit = [], []
    for _, fn in probes:
        out = logp(fn(x))
        dlogp.append(onehot.T @ (out - clean) / count)
        hit.append(onehot.T @ F.one_hot(out.argmax(1), N_CLASSES).float() / count)
    return {"dlogp": torch.stack(dlogp).cpu().numpy().astype(np.float16), "hit": torch.stack(hit).cpu().numpy().astype(np.float16),
            "clean_logp": clean.cpu().numpy().astype(np.float16)}


# Optimized triggers: one trigger for each target class, found on a few images of all other classes and measured on
# the other images. They find a trigger that is not in the bank. The results have the same form as the probes.
OPT_FAMILIES = ["additive", "patch"]
OPT_IMAGES, OPT_STEPS, OPT_LR = 3, 25, 0.1
EPS_ADDITIVE, MASK_PENALTY = 0.05, 2.0


def apply_trigger(family: str, params: list[torch.Tensor], x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (targets, images, 3, 64, 64). Return the changed images and a penalty for each target."""
    if family == "additive":
        return (x + EPS_ADDITIVE * torch.tanh(params[0])[:, None]).clamp(0, 1), torch.zeros(len(x), device=x.device)
    mask, pattern = torch.sigmoid(params[0])[:, None], torch.sigmoid(params[1])[:, None]
    return (1 - mask) * x + mask * pattern, mask.flatten(1).mean(1)


def optimize_triggers(net, x: torch.Tensor, y: torch.Tensor, device: str) -> dict[str, np.ndarray]:
    """Return arrays (families, source, class) like scan_model: for class c, the effect of the trigger of class c."""
    g = torch.Generator().manual_seed(SEED)
    order = torch.stack([torch.nonzero(y.cpu() == c).squeeze(1)[torch.randperm(int((y.cpu() == c).sum()), generator=g)]
                         for c in range(N_CLASSES)]).to(device)
    opt_idx, eval_idx = order[:, :OPT_IMAGES], order[:, OPT_IMAGES:]
    # For each target, the optimization images of all other classes.
    others = torch.stack([opt_idx[torch.arange(N_CLASSES, device=device) != t].flatten() for t in range(N_CLASSES)])
    targets = torch.arange(N_CLASSES, device=device)
    dlogp = np.zeros((len(OPT_FAMILIES), N_CLASSES, N_CLASSES), dtype=np.float32)
    hit = dlogp.copy()
    for f, family in enumerate(OPT_FAMILIES):
        if family == "additive":
            params = [torch.zeros(N_CLASSES, 3, 64, 64, device=device)]
        else:
            params = [torch.full((N_CLASSES, 1, 64, 64), -3.0, device=device), torch.zeros(N_CLASSES, 3, 64, 64, device=device)]
        params = [p.requires_grad_() for p in params]
        opt = torch.optim.Adam(params, lr=OPT_LR)
        for _ in range(OPT_STEPS):
            opt.zero_grad(set_to_none=True)
            for j in targets.split(5):
                xs, penalty = apply_trigger(family, [p[j] for p in params], x[others[j]])
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    logits = net(xs.flatten(0, 1).contiguous(memory_format=torch.channels_last)).float()
                loss = F.cross_entropy(logits, j.repeat_interleave(others.shape[1]), reduction="sum") / others.shape[1]
                (loss + MASK_PENALTY * penalty.sum()).backward()
            opt.step()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            xe = x[eval_idx.flatten()]
            clean = F.log_softmax(net(xe.contiguous(memory_format=torch.channels_last)).float(), 1)
            for t in range(N_CLASSES):
                xs, _ = apply_trigger(family, [p[t:t + 1] for p in params], xe[None])
                out = F.log_softmax(net(xs[0].contiguous(memory_format=torch.channels_last)).float(), 1)
                src = y[eval_idx.flatten()]
                for s in range(N_CLASSES):
                    if s == t:
                        continue
                    m = src == s
                    dlogp[f, s, t] = (out[m, t] - clean[m, t]).mean().item()
                    hit[f, s, t] = (out[m].argmax(1) == t).float().mean().item()
    return {"dlogp": dlogp.astype(np.float16), "hit": hit.astype(np.float16)}


def worker(gpu: int, models: pd.DataFrame, out, log_name: str, optimized: bool = False) -> None:
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    x, y = clean_images()
    x, y = x.to(device), y.to(device)
    probes = probe_bank(device)
    results = []
    for i, row in enumerate(models.itertuples()):
        t0 = time.time()
        net = load_model(row.path, device).to(memory_format=torch.channels_last)
        if optimized:
            for p in net.parameters():
                p.requires_grad_(False)
            results.append(optimize_triggers(net, x, y, device))
        else:
            results.append(scan_model(net, x, y, probes, device))
        print(f"[{log_name}] {i + 1}/{len(models)} {row.set} {row.id} {time.time() - t0:.1f}s", flush=True)
    arrays = {k: np.stack([r[k] for r in results]) for k in results[0]}
    np.savez(out / f"features_{log_name}.npz", set=models["set"].to_numpy(), id=models["id"].to_numpy(), **arrays)


def run(out, names: list[str], smoke: bool, optimized: bool = False) -> None:
    """Scan all models of the sets on all GPUs, with the probe bank or with the optimized triggers. Write features.npz
    (in the sequence of models.csv) and models.csv."""
    out.mkdir(parents=True, exist_ok=True)
    models = model_sets(names)
    if smoke:
        models = pd.concat([models[models["labeled"] & (models["set"] == "real")],
                            models[~models["labeled"]].head(4), models[models["set"] != "real"].groupby("set").sample(4, random_state=SEED)])
    print(models.groupby("set").size(), flush=True)
    names_bank = [f"opt_{f}" for f in OPT_FAMILIES] if optimized else [name for name, _ in probe_bank("cpu")]
    print(f"{len(names_bank)} probes", flush=True)
    n_gpu = max(1, torch.cuda.device_count())
    parts = [models.iloc[g::n_gpu] for g in range(n_gpu)]
    with torch.multiprocessing.get_context("spawn").Pool(n_gpu) as pool:
        pool.starmap(worker, [(g, part, out, f"gpu{g}", optimized) for g, part in enumerate(parts)])
    merged = [dict(np.load(out / f"features_gpu{g}.npz", allow_pickle=True)) for g in range(n_gpu)]
    joined = {k: np.concatenate([m[k] for m in merged]) for k in merged[0]}
    np.savez(out / "features.npz", probes=np.array(names_bank), **joined)
    for g in range(n_gpu):
        (out / f"features_gpu{g}.npz").unlink()
    order = pd.DataFrame({"set": joined["set"], "id": joined["id"]})
    order.merge(models.drop(columns="path"), on=["set", "id"]).to_csv(out / "models.csv", index=False)
