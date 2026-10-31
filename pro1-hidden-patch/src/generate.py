"""Generate attacked images like the task data, but with the patch at any position.

The bases are the clean and decoy-only images, without the tune part of the UNet. Their decoys stay, and the new patch does not
overlap them. The steps follow the description of the task data:
1. Add 0-2 new decoys (uniform, Gaussian or Rademacher noise).
   The horizontal flip and the decoys must keep the ResNet-18 class of the upright base after JPEG.
2. Attack the image inside one random box: PGD (untargeted, targeted), MI-FGSM or C&W.
   A binary search finds the smallest strength that works after JPEG.
3. Save the image as JPEG and use the decoded image.
The attacks do not calculate the gradient through JPEG. Thus they need a slightly larger strength.
"""
import io
import time
import json
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .eda import DECOY_COLOR, PATCH_COLOR, SIZE, _draw_box, load_image, load_images, load_labels, load_model, residual_view
from .config import load_config
from .evaluate import BOX, load_split, score_report, tune_ids
from .paths import output_dir
from .saliency import analyze, answers_frame, choose, margin

ATTACKS = ["pgd_linf_u", "pgd_linf_t", "mifgsm_linf_u", "cw_l2_u"]
EPS_LEVELS = np.array([2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]) / 255
CW_CONSTS = np.logspace(-2, 2, 12)
CW_LINF = 20 / 255


def random_box(rng, taken: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return a box with 8-15% of the area and a ratio from 1:2 to 2:1 that does not overlap taken."""
    for _ in range(100):
        area = rng.uniform(0.08, 0.15) * SIZE**2
        ratio = np.exp(rng.uniform(np.log(0.5), np.log(2)))
        w, h = int(round(np.sqrt(area * ratio))), int(round(np.sqrt(area / ratio)))
        if not (0.08 <= w * h / SIZE**2 <= 0.15 and 0.5 <= w / h <= 2 and w <= SIZE and h <= SIZE):
            continue
        x0, y0 = rng.integers(0, SIZE - w + 1), rng.integers(0, SIZE - h + 1)
        if not taken[y0:y0 + h, x0:x0 + w].any():
            return int(x0), int(y0), int(x0 + w), int(y0 + h)
    return None


def decoy_noise(rng, kind: str, eps: float, h: int, w: int) -> np.ndarray:
    if kind == "uniform":
        return rng.uniform(-eps, eps, (h, w, 3))
    if kind == "rademacher":
        return eps * rng.choice([-1.0, 1.0], (h, w, 3))
    return np.clip(rng.normal(0, eps / 2, (h, w, 3)), -eps, eps)


def jpeg(x: torch.Tensor, quality: np.ndarray) -> torch.Tensor:
    """Save each image as JPEG and decode it. x: (N, 3, SIZE, SIZE) in [0, 1]."""
    out = []
    for img, q in zip((x.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8), quality):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(q))
        out.append(np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255)
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).to(x.device)


def linf_attack(model, x, mask, c, target, eps, kind: str) -> torch.Tensor:
    """PGD (100 steps) or MI-FGSM (20 steps) inside the mask. eps has one value for each image."""
    steps = 20 if kind == "mifgsm_linf_u" else 100
    targeted = kind == "pgd_linf_t"
    eps = eps.view(-1, 1, 1, 1)
    alpha = eps / steps if kind == "mifgsm_linf_u" else 2.5 * eps / steps
    delta, g = torch.zeros_like(x), torch.zeros_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        logits = model((x + delta * mask).clamp(0, 1))
        loss = F.cross_entropy(logits, target if targeted else c, reduction="sum")
        (grad,) = torch.autograd.grad(loss, delta)
        if kind == "mifgsm_linf_u":
            g = g + grad / (grad.abs().mean((1, 2, 3), keepdim=True) + 1e-12)
            grad = g
        step = -alpha * grad.sign() if targeted else alpha * grad.sign()
        delta = torch.max(torch.min(delta.detach() + step, eps), -eps) * mask
    return (x + delta).clamp(0, 1)


def cw_attack(model, x, mask, c, const) -> torch.Tensor:
    """C&W L2 (100 steps of Adam) inside the mask. const has one value for each image."""
    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.01)
    for _ in range(100):
        z = (x + delta * mask).clamp(0, 1)
        f = margin(model(z), c).clamp(min=-1.0)
        loss = ((delta * mask) ** 2).sum((1, 2, 3)) + const * f
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-CW_LINF, CW_LINF)
    return (x + delta.detach() * mask).clamp(0, 1)


def attack_search(model, x, mask, c, target, kind: str, quality: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Binary search for the smallest strength level that works after JPEG.

    Return the JPEG images and the found level index for each image (-1 if no level works).
    """
    levels = CW_CONSTS if kind == "cw_l2_u" else EPS_LEVELS
    n = len(x)
    lo, hi = np.zeros(n, int), np.full(n, len(levels) - 1)
    best = torch.zeros_like(x)
    found = np.full(n, -1)
    while (lo <= hi).any():
        active = np.where(lo <= hi)[0]
        mid = (lo[active] + hi[active]) // 2
        idx = torch.from_numpy(active).to(x.device)
        value = torch.tensor(levels[mid], dtype=x.dtype, device=x.device)
        if kind == "cw_l2_u":
            adv = cw_attack(model, x[idx], mask[idx], c[idx], value)
        else:
            adv = linf_attack(model, x[idx], mask[idx], c[idx], target[idx], value, kind)
        adv = jpeg(adv, quality[active])
        with torch.no_grad():
            pred = model(adv).argmax(1)
        ok = (pred == target[idx] if kind == "pgd_linf_t" else pred != c[idx]).cpu().numpy()
        best[idx[torch.from_numpy(ok).to(x.device)]] = adv[torch.from_numpy(ok).to(x.device)]
        found[active[ok]] = mid[ok]
        hi[active[ok]] = mid[ok] - 1
        lo[active[~ok]] = mid[~ok] + 1
    return best, found


def make_jobs(labels: pd.DataFrame, copies: int, hflip: bool, rng) -> list[dict]:
    """Return one job for each new image: base, flip, patch box, new decoys, attack and JPEG quality."""
    jobs = []
    for row in labels.itertuples():
        for _ in range(copies):
            # k = 4 is the horizontal flip. A rotation changes the ResNet-18 class of most images.
            k = int(rng.choice([0, 4])) if hflip else 0
            old = [_box_of(_rect(d["box"]), k) for d in row.decoys]
            taken = np.zeros((SIZE, SIZE), bool)
            for x0, y0, x1, y1 in old:
                taken[y0:y1, x0:x1] = True
            box = random_box(rng, taken)
            if box is None:
                continue
            taken[box[1]:box[3], box[0]:box[2]] = True
            decoys = []
            for _ in range(int(rng.choice(3)) if not old else int(rng.choice(2))):
                b = random_box(rng, taken)
                if b is None:
                    break
                taken[b[1]:b[3], b[0]:b[2]] = True
                kind = str(rng.choice(["uniform", "rademacher", "gaussian"]))
                eps = float(rng.choice(EPS_LEVELS))
                decoys.append({"type": kind, "strength": eps * 255, "box": b,
                               "noise": decoy_noise(rng, kind, eps, b[3] - b[1], b[2] - b[0])})
            jobs.append({"base": row.id, "k": k, "box": box, "old_decoys": old, "decoys": decoys,
                         "attack": str(rng.choice(ATTACKS)), "jpeg": int(rng.integers(85, 96)),
                         "target": int(rng.integers(200))})
    return jobs


def _rect(box) -> np.ndarray:
    m = np.zeros((SIZE, SIZE), bool)
    m[box[1]:box[3], box[0]:box[2]] = True
    return m


def _box_of(mask: np.ndarray, k: int) -> tuple[int, int, int, int]:
    mask = np.rot90(mask, k % 4)
    mask = mask[:, ::-1] if k >= 4 else mask
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)


def generate(part: str, copies: int, hflip: bool, seed: int, batch: int = 512,
             deadline: float | None = None, exclude: set = frozenset()) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the labels and the uint8 images (N, SIZE, SIZE, 3) of the new attacked images.

    part selects the bases from splits/split.csv: train or val. exclude has ids that are not bases.
    After the deadline (time.time()), no new batch starts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device).requires_grad_(False)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    rng = np.random.default_rng(seed)
    labels = load_labels().merge(load_split(), on="id")
    bases = labels[(labels.part == part) & labels.kind.isin(["CLEAN", "DECOY"]) & ~labels.id.isin(exclude)]
    bases = bases.reset_index(drop=True)
    print(f"{part}: {len(bases)} bases", flush=True)
    base_images = dict(zip(bases.id, load_images("train", bases.id)))
    jobs = make_jobs(bases, copies, hflip, rng)
    with torch.no_grad():
        upright = torch.from_numpy(np.stack(list(base_images.values()))).permute(0, 3, 1, 2).to(device)
        base_class = dict(zip(base_images, model(upright).argmax(1).tolist()))

    rows, images, rejected = [], [], {"class": 0, "attack": 0}
    for start in range(0, len(jobs), batch):
        if deadline is not None and time.time() > deadline:
            print(f"{part}: time limit, stop after {start} jobs", flush=True)
            break
        part_jobs = jobs[start:start + batch]
        x = np.stack([_prepare(base_images[j["base"]], j) for j in part_jobs])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        mask = torch.from_numpy(np.stack([_rect(j["box"]) for j in part_jobs])[:, None].astype(np.float32)).to(device)
        # The flip and the new decoys must keep the class of the upright base after JPEG.
        c = torch.tensor([base_class[j["base"]] for j in part_jobs], device=device)
        quality = np.array([j["jpeg"] for j in part_jobs])
        with torch.no_grad():
            same = (model(jpeg(x, quality)).argmax(1) == c).cpu().numpy()
        rejected["class"] += int((~same).sum())
        target = torch.tensor([j["target"] for j in part_jobs], device=device)
        target = torch.where(target == c, (target + 1) % 200, target)
        attacks = np.array([j["attack"] for j in part_jobs])
        for kind in ATTACKS:
            sel = np.where((attacks == kind) & same)[0]
            if len(sel) == 0:
                continue
            idx = torch.from_numpy(sel).to(device)
            adv, found = attack_search(model, x[idx], mask[idx], c[idx], target[idx], kind, quality[sel])
            adv = (adv.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            for i, j_idx in enumerate(sel):
                if found[i] < 0:
                    rejected["attack"] += 1
                    continue
                j = part_jobs[j_idx]
                strength = (CW_CONSTS[found[i]] if kind == "cw_l2_u" else EPS_LEVELS[found[i]] * 255)
                rows.append({"id": f"gen_{part}_{start + j_idx}", "base": j["base"], "k": j["k"], "base_class": int(c[j_idx]),
                             **dict(zip(BOX, j["box"])), "kind": "ADV", "attack": kind,
                             "strength": float(strength), "jpeg": j["jpeg"],
                             "decoys": [{"box": b} for b in j["old_decoys"]] + [{"box": d["box"]} for d in j["decoys"]]})
                images.append(adv[i])
        print(f"{part}: {min(start + batch, len(jobs))}/{len(jobs)} jobs, {len(rows)} images, {rejected} rejected", flush=True)
    if not rows:
        raise RuntimeError(f"{part}: all {len(jobs)} jobs were rejected")
    out = pd.DataFrame(rows)
    out["n_decoys"] = out.decoys.map(len)
    return out, np.stack(images)


def _prepare(image: np.ndarray, job: dict) -> np.ndarray:
    """Return the base image after the flip, with the new decoys."""
    x = np.rot90(image, job["k"] % 4)
    x = x[:, ::-1] if job["k"] >= 4 else x
    x = x.copy()
    for d in job["decoys"]:
        x0, y0, x1, y1 = d["box"]
        x[y0:y1, x0:x1] += d["noise"]
    return np.clip(x, 0, 1).astype(np.float32)


def save_set(path, labels: pd.DataFrame, images: np.ndarray) -> None:
    """Write the images to <path>.npy and the labels to <path>.csv. The decoys are a JSON column."""
    np.save(f"{path}.npy", images)
    labels.assign(decoys=labels.decoys.map(lambda ds: json.dumps([list(d["box"]) for d in ds]))).to_csv(
        f"{path}.csv", index=False)


def load_set(path) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the labels and the float32 images in [0, 1] of a generated set."""
    labels = pd.read_csv(f"{path}.csv")
    labels["decoys"] = labels.decoys.map(lambda s: [{"box": tuple(b)} for b in json.loads(s)])
    return labels, np.load(f"{path}.npy").astype(np.float32) / 255


def run_generation(cfg: dict) -> None:
    """Generate the train and val sets and score saliency-v1 on the val set."""

    out_dir = output_dir() / cfg["codename"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stop the train part in time, so the zip is complete before the Kaggle limit of 12 hours.
    deadline = time.time() + cfg["hours"] * 3600
    for part in ["val", "train"]:
        labels, images = generate(part, cfg[f"{part}_copies"], cfg["hflip"], cfg["seed"] + (part == "val"), cfg["batch"],
                                  deadline if part == "train" else None, tune_ids(cfg["tune_fraction"], cfg["tune_seed"]))
        save_set(out_dir / part, labels, images)
        summary = labels.groupby("attack").strength.describe()
        summary.to_csv(out_dir / f"summary_{part}.csv")
        print(summary, flush=True)
        if part == "val":
            # The checks run before the long train part, so an error shows early.
            plot_examples(labels, images.astype(np.float32) / 255).savefig(out_dir / "examples.png", dpi=80, bbox_inches="tight")
            score_saliency(labels, images.astype(np.float32) / 255, cfg).to_csv(out_dir / "report_saliency-v1_gen_val.csv", index=False)

    # One zip file for the Kaggle dataset: Output -> New Dataset on the notebook page.
    with zipfile.ZipFile(output_dir() / f"{cfg['codename']}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for part in ["train", "val"]:
            for ext in ["npy", "csv"]:
                z.write(out_dir / f"{part}.{ext}", f"{part}.{ext}")
                (out_dir / f"{part}.{ext}").unlink()


def score_saliency(labels: pd.DataFrame, images: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Score saliency-v1 with the parameters of its Kaggle run: the first valid check of patches in the top half."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device).requires_grad_(False)
    cand = analyze(model, images, load_config("saliency-v1"), device)
    answers = answers_frame(labels.id, choose(cand, cfg["saliency_tie"], cfg["saliency_threshold"]))
    half = np.where((labels.y_min + labels.y_max) / 2 < SIZE / 2, "top half", "bottom half")
    report = pd.concat([score_report(labels, answers), score_report(labels.assign(attack=half), answers).iloc[1:]])
    print(report, flush=True)
    return report


def plot_examples(labels: pd.DataFrame, images: np.ndarray, bases: int = 4, per_base: int = 3):
    """For some bases: the original image, then generated images and their residuals with the true boxes."""
    fig, axes = plt.subplots(bases, 1 + 2 * per_base, figsize=(2.2 * (1 + 2 * per_base), 2.4 * bases))
    for row_axes, base in zip(axes, labels.base.drop_duplicates()[:bases]):
        row_axes[0].imshow(load_image("train", base))
        row_axes[0].set_title(f"original\n{base}", fontsize=7)
        sel = np.where(labels.base == base)[0][:per_base]
        for j, i in enumerate(sel):
            row = labels.iloc[i]
            ax_img, ax_res = row_axes[1 + 2 * j], row_axes[2 + 2 * j]
            ax_img.imshow(images[i])
            ax_img.set_title(f"{row.attack}\nstrength {row.strength:.3g}, JPEG {row.jpeg}", fontsize=7)
            ax_res.imshow(residual_view(images[i]), cmap="gray")
            _draw_box(ax_res, (row.x_min, row.y_min, row.x_max, row.y_max), PATCH_COLOR, "patch")
            for d in row.decoys:
                _draw_box(ax_res, d["box"], DECOY_COLOR, "decoy")
            ax_res.set_title("residual", fontsize=7)
        for ax in row_axes:
            ax.axis("off")
    fig.tight_layout()
    return fig
