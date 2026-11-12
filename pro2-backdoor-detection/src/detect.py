"""Collect the detection features of a set of models: trigger inversion for each pair of classes, and population statistics.

The inversion looks for a small universal change that moves the images of a source class to a target class. It optimizes
the change on a few images and measures it on the other images of the class. A backdoor gives a change that works on
all images. An adversarial change for a few images usually does not.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model_stats import N_CLASSES, class_stats, clean_images, probe_sets
from src.paths import KAGGLE_INPUT, data_dir, load_model

SEED = 0
FAMILIES = ["additive", "patch", "color", "warp"]
N_OPT = 4  # Images of each source class for the optimization. The other 16 measure the result.
STEPS = 40
CHUNK = 512
EPS_ADDITIVE = 0.03
MASK_PENALTY = 2.0
COLOR_GAIN, COLOR_BIAS = 0.3, 0.15
WARP_PIXELS = 2.0
LR = {"additive": 0.1, "patch": 0.1, "color": 0.05, "warp": 0.05}
PAIRS = [(s, t) for s in range(N_CLASSES) for t in range(N_CLASSES) if s != t]
STAT_NAMES = sorted(f"{probe}_{kind}" for probe in ("clean", "gauss", "blend", "noise") for kind in ("pred_frac", "prob", "logit")) + \
    ["clean_recall", "compactness", "head_max_cos_other", "head_norm", "head_proto_cos", "proto_norm"]


def model_sets(names: list[str]) -> pd.DataFrame:
    """Return one row for each model of the sets: set, id, path, targets, and for the validation sets the types and variants."""
    rows = []
    if "real" in names:
        root = data_dir()
        train = pd.read_csv(root / "example/train/labels.csv", dtype=str, keep_default_na=False)
        for _, r in train.iterrows():
            rows.append({"set": "real", "id": r["id"], "path": root / f"example/train/models/{r['id']}.pt",
                         "targets": r["targets"], "types": "", "variants": "", "labeled": True})
        for path in sorted((root / "test/models").glob("*.pt")):
            rows.append({"set": "real", "id": path.stem, "path": path, "targets": "", "types": "", "variants": "", "labeled": False})
    for name, folder in (("a", "val-models"), ("b", "val-models-b")):
        if name not in names:
            continue
        labels_path = next(KAGGLE_INPUT.rglob(f"{folder}/labels.csv"))
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        for _, r in labels[labels["valid"] == "True"].iterrows():
            rows.append({"set": name, "id": r["id"], "path": labels_path.parent / "models" / f"{r['id']}.pt",
                         "targets": r["targets"], "types": r["types"], "variants": r["variants"], "labeled": True})
    return pd.DataFrame(rows)


def split_images(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the optimization images (classes, N_OPT, 3, 64, 64) and the measure images (classes, 20 - N_OPT, ...)."""
    g = torch.Generator().manual_seed(SEED)
    per_class = [x[y == c][torch.randperm(int((y == c).sum()), generator=g)] for c in range(N_CLASSES)]
    return torch.stack([p[:N_OPT] for p in per_class]), torch.stack([p[N_OPT:] for p in per_class])


def init_params(family: str, n: int, device: str) -> list[torch.Tensor]:
    shapes = {"additive": [(n, 3, 64, 64)], "patch": [(n, 1, 64, 64), (n, 3, 64, 64)], "color": [(n, 3, 3), (n, 3)],
              "warp": [(n, 2, 8, 8)]}[family]
    params = [torch.zeros(s, device=device) for s in shapes]
    if family == "patch":
        params[0] -= 3  # A small mask at the start.
    return [p.requires_grad_() for p in params]


def transform(family: str, params: list[torch.Tensor], x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the changes of a group of pairs. x: (pairs, images, 3, 64, 64). Return the images and a penalty for each pair."""
    zero = torch.zeros(len(x), device=x.device)
    if family == "additive":
        return (x + EPS_ADDITIVE * torch.tanh(params[0])[:, None]).clamp(0, 1), zero
    if family == "patch":
        mask, pattern = torch.sigmoid(params[0])[:, None], torch.sigmoid(params[1])[:, None]
        return (1 - mask) * x + mask * pattern, mask.flatten(1).mean(1)
    if family == "color":
        matrix = torch.eye(3, device=x.device) + COLOR_GAIN * torch.tanh(params[0])
        bias = COLOR_BIAS * torch.tanh(params[1])
        return (torch.einsum("pij,pnjhw->pnihw", matrix, x) + bias[:, None, :, None, None]).clamp(0, 1), zero
    if family == "warp":
        p, n = x.shape[:2]
        flow = F.interpolate(torch.tanh(params[0]) * WARP_PIXELS / 32, size=64, mode="bicubic", align_corners=True)
        line = torch.linspace(-1, 1, 64, device=x.device)
        grid = torch.stack(torch.meshgrid(line, line, indexing="xy"), dim=-1)[None] + flow.permute(0, 2, 3, 1)
        grid = grid[:, None].expand(p, n, 64, 64, 2).reshape(p * n, 64, 64, 2)
        out = F.grid_sample(x.reshape(p * n, 3, 64, 64), grid.clamp(-1, 1), align_corners=True)
        return out.reshape(p, n, 3, 64, 64), zero
    raise ValueError(family)


def invert(net: torch.nn.Module, x_opt: torch.Tensor, x_eval: torch.Tensor, family: str, steps: int, device: str) -> dict[str, np.ndarray]:
    """Optimize one change for each pair (source, target). Return 20 x 20 arrays: the mean margin of the target logit and
    the ASR on the measure images, and the penalty (the mask size for patch)."""
    src = torch.tensor([s for s, _ in PAIRS], device=device)
    tgt = torch.tensor([t for _, t in PAIRS], device=device)
    params = init_params(family, len(PAIRS), device)
    opt = torch.optim.Adam(params, lr=LR[family])
    per_chunk = max(1, CHUNK // x_opt.shape[1])
    groups = torch.arange(len(PAIRS), device=device).split(per_chunk)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        for j in groups:
            xs, penalty = transform(family, [p[j] for p in params], x_opt[src[j]])
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits = net(xs.flatten(0, 1)).float()
            loss = F.cross_entropy(logits, tgt[j].repeat_interleave(x_opt.shape[1]), reduction="sum") / x_opt.shape[1]
            (loss + MASK_PENALTY * penalty.sum()).backward()
        opt.step()
    margin = np.full((N_CLASSES, N_CLASSES), np.nan)
    asr, size = margin.copy(), margin.copy()
    per_chunk = max(1, CHUNK // x_eval.shape[1])
    with torch.no_grad():
        for j in torch.arange(len(PAIRS), device=device).split(per_chunk):
            xs, penalty = transform(family, [p[j] for p in params], x_eval[src[j]])
            logits = net(xs.flatten(0, 1)).float().view(len(j), x_eval.shape[1], N_CLASSES)
            target = logits.gather(2, tgt[j][:, None, None].expand(-1, x_eval.shape[1], 1)).squeeze(2)
            other = logits.scatter(2, tgt[j][:, None, None].expand(-1, x_eval.shape[1], 1), -1e4).max(2).values
            s, t = src[j].cpu().numpy(), tgt[j].cpu().numpy()
            margin[s, t] = (target - other).mean(1).cpu().numpy()
            asr[s, t] = (target > other).float().mean(1).cpu().numpy()
            size[s, t] = penalty.cpu().numpy()
    return {"margin": margin, "asr": asr, "size": size}


def model_features(path, x: torch.Tensor, y: torch.Tensor, sets: dict, steps: int, device: str) -> dict[str, np.ndarray]:
    net = load_model(path, device)
    for p in net.parameters():
        p.requires_grad_(False)
    per_class, _ = class_stats(net, sets, device)
    x_opt, x_eval = (t.to(device) for t in split_images(x, y))
    out = {"stats": np.stack([per_class[k] for k in STAT_NAMES], axis=1)}
    for family in FAMILIES:
        for key, value in invert(net, x_opt, x_eval, family, steps, device).items():
            out[f"{family}_{key}"] = value
    return out


def worker(gpu: int, models: pd.DataFrame, out, log_name: str, steps: int) -> None:
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    x, y = clean_images()
    sets = probe_sets(x, y)
    results = []
    for i, row in enumerate(models.itertuples()):
        t0 = time.time()
        results.append(model_features(row.path, x, y, sets, steps, device))
        print(f"[{log_name}] {i + 1}/{len(models)} {row.set} {row.id} {time.time() - t0:.0f}s", flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(models):
            save(results, models.iloc[:len(results)], out / f"features_{log_name}.npz")


def save(results: list[dict], models: pd.DataFrame, path) -> None:
    arrays = {k: np.stack([r[k] for r in results]) for k in results[0]}
    np.savez(path, set=models["set"].to_numpy(), id=models["id"].to_numpy(), stat_names=np.array(STAT_NAMES), **arrays)


def run(out, names: list[str], smoke: bool, steps: int = STEPS) -> None:
    """Collect the features of all models of the sets on all GPUs. Write features.npz and models.csv."""
    out.mkdir(parents=True, exist_ok=True)
    models = model_sets(names)
    if smoke:
        models = pd.concat([models[models["labeled"]].groupby("set").head(3), models[~models["labeled"]].head(2)])
    models.drop(columns="path").to_csv(out / "models.csv", index=False)
    print(models.groupby("set").size(), flush=True)
    n_gpu = max(1, torch.cuda.device_count())
    parts = [models.iloc[g::n_gpu] for g in range(n_gpu)]
    with torch.multiprocessing.get_context("spawn").Pool(n_gpu) as pool:
        pool.starmap(worker, [(g, part, out, f"gpu{g}", steps) for g, part in enumerate(parts)])
    merged = [dict(np.load(out / f"features_gpu{g}.npz", allow_pickle=True)) for g in range(n_gpu)]
    joined = {k: np.concatenate([m[k] for m in merged]) for k in merged[0] if k != "stat_names"}
    np.savez(out / "features.npz", stat_names=merged[0]["stat_names"], **joined)
    for g in range(n_gpu):
        (out / f"features_gpu{g}.npz").unlink()
    print({k: v.shape for k, v in joined.items()}, flush=True)
