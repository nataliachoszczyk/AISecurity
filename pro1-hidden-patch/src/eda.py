"""Functions for the exploratory data analysis of the train set and the test set."""
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import median_filter

from .paths import data_dir

SIZE = 64
PATCH_COLOR = "#e4572e"
DECOY_COLOR = "#2e86de"

_DECOY_RE = re.compile(r"(rademacher|uniform|gaussian)[^\[]*?(?:eps=(\d+)/255|linf=([\d.]+)/255)[^\[]*\[(\d+),(\d+),(\d+),(\d+)\]")


def _parse_comment(comment: str) -> dict:
    kind = comment.split()[0].rstrip(";")
    attack = re.match(r"ADV (\S+)", comment)
    eps = re.search(r"ADV \S+ eps=(\d+)/255", comment)
    linf = re.search(r"ADV \S+ .*?linf=([\d.]+)/255", comment.split(";")[0])
    decoys = [
        {"type": t, "strength": float(e or l), "box": (int(a), int(b), int(c), int(d))}
        for t, e, l, a, b, c, d in _DECOY_RE.findall(comment)
    ]
    return {
        "kind": kind,
        "attack": attack.group(1) if attack else None,
        # C&W has no eps; its largest pixel change (linf) is the comparable strength.
        "strength": float(eps.group(1)) if eps else (float(linf.group(1)) if linf else np.nan),
        "jpeg": int(re.search(r"jpeg=(\d+)", comment).group(1)),
        "decoys": decoys,
        "n_decoys": len(decoys),
    }


def load_labels() -> pd.DataFrame:
    """Return the train labels with the fields of the comment as columns."""
    df = pd.read_csv(data_dir() / "train_labels.csv")
    parsed = pd.DataFrame([_parse_comment(c) for c in df.comment], index=df.index)
    df = pd.concat([df, parsed], axis=1)
    df["w"] = df.x_max - df.x_min
    df["h"] = df.y_max - df.y_min
    df["area"] = df.w * df.h / SIZE**2
    return df


def load_test_ids() -> pd.Series:
    return pd.read_csv(data_dir() / "test_ids.csv").id


def load_image(split: str, image_id: str) -> np.ndarray:
    """Return an image as float32 (64, 64, 3) in [0, 1]."""
    path = Path(data_dir()) / split / "images" / f"{image_id}.png"
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255


def load_images(split: str, ids) -> np.ndarray:
    return np.stack([load_image(split, i) for i in ids])


def residual(images: np.ndarray) -> np.ndarray:
    """Return the high-frequency part of the images: image minus its 3x3 median."""
    size = (3, 3, 1) if images.ndim == 3 else (1, 3, 3, 1)
    return images - median_filter(images, size=size)


def residual_view(image: np.ndarray) -> np.ndarray:
    """Return the amplified absolute residual as a gray image for display."""
    r = np.abs(residual(image)).mean(-1)
    return np.clip(r / (np.quantile(r, 0.99) + 1e-8), 0, 1)


def box_heatmap(boxes) -> np.ndarray:
    """Return a 64x64 map: the number of boxes that cover each pixel."""
    heat = np.zeros((SIZE, SIZE))
    for x0, y0, x1, y1 in boxes:
        heat[y0:y1, x0:x1] += 1
    return heat


def plot_position_heatmaps(df: pd.DataFrame):
    patches = df[df.kind == "ADV"][["x_min", "y_min", "x_max", "y_max"]].values
    decoys = [d["box"] for ds in df.decoys for d in ds]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, boxes, title in [
        (axes[0], patches, f"Patches (n={len(patches)})"),
        (axes[1], decoys, f"Decoys (n={len(decoys)})"),
    ]:
        heat = box_heatmap(boxes) / len(boxes)
        im = ax.imshow(heat, cmap="magma", vmin=0)
        ax.axhline(31.5, color="white", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("x (column)")
        ax.set_ylabel("y (row)")
        fig.colorbar(im, ax=ax, label="share of boxes that cover the pixel")
    fig.suptitle("Train set: position of the boxes (dashed line = middle row)")
    fig.tight_layout()
    return fig


def plot_box_corners(df: pd.DataFrame):
    """Scatter of the top left corners: shows the hard limits of the positions."""
    adv = df[df.kind == "ADV"]
    decoys = pd.DataFrame([d["box"] for ds in df.decoys for d in ds], columns=["x_min", "y_min", "x_max", "y_max"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(decoys.x_min, decoys.y_min, s=4, alpha=0.3, c=DECOY_COLOR, label="decoy")
    ax.scatter(adv.x_min, adv.y_min, s=4, alpha=0.3, c=PATCH_COLOR, label="patch")
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)
    ax.set_xlabel("x_min")
    ax.set_ylabel("y_min")
    ax.set_title("Top left corner of each box")
    ax.legend(markerscale=4)
    return fig


def plot_distributions(df: pd.DataFrame):
    adv = df[df.kind == "ADV"]
    decoys = pd.DataFrame([d for ds in df.decoys for d in ds])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    df.kind.value_counts().plot.bar(ax=axes[0, 0], color="#555")
    axes[0, 0].set_title("Image types")

    adv.attack.value_counts().plot.bar(ax=axes[0, 1], color=PATCH_COLOR)
    axes[0, 1].set_title("Attacks (patch images)")

    bins = np.arange(0.5, 21.5, 1)
    axes[0, 2].hist(adv.strength, bins=bins, histtype="step", lw=2, color=PATCH_COLOR, label="patch (eps or C&W linf)")
    axes[0, 2].hist(decoys.strength, bins=bins, histtype="step", lw=2, color=DECOY_COLOR, label="decoy (eps or linf)")
    axes[0, 2].set_xlabel("strength [x/255]")
    axes[0, 2].set_title("Strength of the noise")
    axes[0, 2].legend()

    axes[1, 0].hist(df.area[df.kind == "ADV"] * 100, bins=20, color=PATCH_COLOR)
    axes[1, 0].set_xlabel("patch area [% of image]")
    axes[1, 0].set_title("Patch area")

    axes[1, 1].scatter(adv.w, adv.h, s=4, alpha=0.3, c=PATCH_COLOR)
    axes[1, 1].set_xlabel("width")
    axes[1, 1].set_ylabel("height")
    axes[1, 1].set_title("Patch size")

    pd.crosstab(df.kind, df.n_decoys).plot.bar(ax=axes[1, 2], stacked=True)
    axes[1, 2].set_title("Number of decoys for each image type")
    axes[1, 2].legend(title="decoys")
    fig.tight_layout()
    return fig


def pick_train_cases(df: pd.DataFrame, seed: int = 0) -> list[tuple[str, str]]:
    """Return 15 (id, case name) pairs that cover all types of train images."""
    rng = np.random.default_rng(seed)
    adv = df.kind == "ADV"
    no_decoy = df.n_decoys == 0
    one_decoy_of = lambda t: (df.n_decoys == 1) & df.decoys.map(lambda ds: bool(ds) and ds[0]["type"] == t)
    decoy_stronger = adv & df.decoys.map(lambda ds: bool(ds)) & (
        df.decoys.map(lambda ds: max((d["strength"] for d in ds), default=0)) > df.strength
    )
    cases = [
        ("Clean", df.kind == "CLEAN"),
        ("Clean (JPEG 90)", (df.kind == "CLEAN") & (df.jpeg == 90)),
        ("Decoys only: 1 decoy", (df.kind == "DECOY") & (df.n_decoys == 1)),
        ("Decoys only: 2 decoys", (df.kind == "DECOY") & (df.n_decoys == 2)),
        ("Patch PGD untargeted, no decoy", adv & no_decoy & (df.attack == "pgd_linf_u")),
        ("Patch PGD targeted, no decoy", adv & no_decoy & (df.attack == "pgd_linf_t")),
        ("Patch MI-FGSM, no decoy", adv & no_decoy & (df.attack == "mifgsm_linf_u")),
        ("Patch C&W L2, no decoy", adv & no_decoy & (df.attack == "cw_l2_u")),
        ("Patch + uniform decoy", adv & one_decoy_of("uniform")),
        ("Patch + gaussian decoy", adv & one_decoy_of("gaussian")),
        ("Patch + rademacher decoy", adv & one_decoy_of("rademacher")),
        ("Patch + 2 decoys", adv & (df.n_decoys == 2)),
        ("Patch, weakest eps (2/255)", adv & no_decoy & (df.strength == 2)),
        ("Patch, strongest eps (20/255)", adv & no_decoy & (df.strength == 20)),
        ("Patch + decoy stronger than patch", decoy_stronger),
    ]
    picked, used = [], set()
    for name, mask in cases:
        pool = [i for i in df.id[mask] if i not in used]
        image_id = pool[rng.integers(len(pool))]
        used.add(image_id)
        picked.append((image_id, name))
    return picked


def _describe(row) -> str:
    if row.kind == "CLEAN":
        text = "no patch, no decoy"
    elif row.kind == "DECOY":
        text = "no patch"
    else:
        text = f"{row.attack}, strength {row.strength:g}/255"
    for d in row.decoys:
        text += f"\ndecoy {d['type']} {d['strength']:g}/255"
    return text + f"\nJPEG {row.jpeg}"


def _draw_box(ax, box, color, label):
    x0, y0, x1, y1 = box
    # Pixel centers are at integers, so the edge of pixel x0 is at x0 - 0.5.
    ax.add_patch(mpatches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=color, lw=2))
    ax.text(x0, y0 - 1, label, color=color, fontsize=8, va="bottom", weight="bold")


def plot_train_cases(df: pd.DataFrame, cases: list[tuple[str, str]]):
    """Two columns: the original image, and the amplified residual with the true boxes."""
    rows = df.set_index("id")
    fig, axes = plt.subplots(len(cases), 2, figsize=(7.5, 3.6 * len(cases)))
    for (image_id, name), (ax_img, ax_res) in zip(cases, axes):
        row = rows.loc[image_id]
        img = load_image("train", image_id)
        ax_img.imshow(img)
        ax_img.set_title(f"{name}\n{image_id}", fontsize=9)
        ax_res.imshow(residual_view(img), cmap="gray")
        if row.kind == "ADV":
            _draw_box(ax_res, (row.x_min, row.y_min, row.x_max, row.y_max), PATCH_COLOR, "patch")
        for d in row.decoys:
            _draw_box(ax_res, d["box"], DECOY_COLOR, "decoy")
        ax_res.set_title(_describe(row), fontsize=8)
        for ax in (ax_img, ax_res):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].annotate("Original", (0.5, 1.45), xycoords="axes fraction", ha="center", fontsize=12, weight="bold")
    axes[0, 1].annotate("Residual (x amplified) + boxes", (0.5, 1.45), xycoords="axes fraction", ha="center", fontsize=12, weight="bold")
    fig.tight_layout()
    return fig


def plot_test_cases(ids, notes):
    """Two columns: the original test image and its amplified residual (no labels exist)."""
    fig, axes = plt.subplots(len(ids), 2, figsize=(7.5, 3.4 * len(ids)))
    for image_id, note, (ax_img, ax_res) in zip(ids, notes, axes):
        img = load_image("test", image_id)
        ax_img.imshow(img)
        ax_img.set_title(image_id, fontsize=9)
        ax_res.imshow(residual_view(img), cmap="gray")
        ax_res.axhline(31.5, color="#f5c518", lw=0.8, ls="--")
        ax_res.set_title(f"residual (amplified)\n{note}", fontsize=9)
        for ax in (ax_img, ax_res):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    return fig


def load_model():
    import sys

    import torch

    model_dir = str(Path(data_dir()) / "model")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from resnet import resnet18_tin

    model = resnet18_tin()
    model.load_state_dict(torch.load(Path(model_dir) / "resnet18_tin.pt", weights_only=True))
    return model.eval()


def predict(model, images: np.ndarray, batch: int = 250) -> np.ndarray:
    import torch

    with torch.no_grad():
        return np.concatenate([
            model(torch.from_numpy(images[i:i + batch]).permute(0, 3, 1, 2)).argmax(1).numpy()
            for i in range(0, len(images), batch)
        ])


def half_flips(model, images: np.ndarray) -> pd.DataFrame:
    """Smooth only one half of each image and record if the predicted class changes.

    Adversarial noise is fragile: a 3x3 median removes it. Thus the half whose smoothing
    changes the prediction most probably holds the patch.
    """
    smooth = median_filter(images, size=(1, 3, 3, 1))
    base = predict(model, images)
    out = {}
    for name, rows in [("top", slice(0, 32)), ("bottom", slice(32, 64))]:
        mixed = images.copy()
        mixed[:, rows] = smooth[:, rows]
        out[name] = predict(model, mixed) != base
    flips = pd.DataFrame(out)
    flips["region"] = np.select(
        [flips.top & ~flips.bottom, flips.bottom & ~flips.top, flips.top & flips.bottom],
        ["top only", "bottom only", "both"], "none",
    )
    return flips


def plot_half_flips(train_flips: pd.DataFrame, test_flips: pd.DataFrame):
    order = ["top only", "bottom only", "both", "none"]
    table = pd.DataFrame({
        "train": train_flips.region.value_counts(normalize=True),
        "test": test_flips.region.value_counts(normalize=True),
    }).reindex(order).fillna(0)
    ax = (table * 100).plot.bar(figsize=(7, 4), rot=0, color=["#777", PATCH_COLOR])
    ax.set_ylabel("% of images")
    ax.set_xlabel("smoothed half that changes the prediction")
    ax.set_title("Where is the adversarial noise? (3x3 median on one half)")
    ax.figure.tight_layout()
    return ax.figure, table


def row_profile(images: np.ndarray, band: int = 8) -> np.ndarray:
    """Return the mean absolute residual in each horizontal band of rows."""
    r = np.abs(residual(images)).mean(-1)
    return r.reshape(len(r), SIZE // band, band, SIZE).mean((0, 2, 3))


def plot_train_test_profiles(train_images: np.ndarray, test_images: np.ndarray, band: int = 8):
    tr, te = row_profile(train_images, band), row_profile(test_images, band)
    centers = np.arange(band / 2, SIZE, band)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(centers, tr * 255, "o-", label="train")
    axes[0].plot(centers, te * 255, "o-", label="test")
    axes[0].set_ylabel("mean |residual| [x/255]")
    axes[0].set_title("Residual energy for each row band")
    axes[0].legend()
    axes[1].bar(centers, (te - tr) * 255, width=band * 0.8, color=np.where(te > tr, PATCH_COLOR, DECOY_COLOR))
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_title("Test minus train")
    for ax in axes:
        ax.axvline(32, color="#999", ls="--", lw=0.8)
        ax.set_xlabel("row (y)")
    fig.tight_layout()
    return fig


def jpeg_error(image: np.ndarray, quality: int) -> float:
    """Return the mean absolute change (in 1/255) after one more JPEG save with this quality. image: uint8."""
    import io

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    again = np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16)
    return float(np.abs(again - image).mean())


def estimate_jpeg_quality(split: str, ids, qualities=range(50, 97)) -> pd.DataFrame:
    """Return the JPEG quality with the smallest change after one more save, for each image.

    A decoded JPEG changes very little when it is saved again with its own quality.
    """
    rows = []
    for image_id in ids:
        img = np.asarray(Image.open(Path(data_dir()) / split / "images" / f"{image_id}.png").convert("RGB"))
        errors = [jpeg_error(img, q) for q in qualities]
        best = int(np.argmin(errors))
        rows.append({"id": image_id, "quality": list(qualities)[best], "error": errors[best]})
    return pd.DataFrame(rows)
