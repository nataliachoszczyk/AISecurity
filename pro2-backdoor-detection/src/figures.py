"""Draw the figures of the README from the task images, the labels of the validation sets and the features of the job scan.

No network runs here, so the job can use a CPU.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle

from src import val_models
from src.paths import KAGGLE_INPUT, data_dir
from src.scan import N_CLASSES, VAL_FOLDERS, clean_images, probe_bank
from src.score import OFF_DIAGONAL, PARTS, load, pair_labels, report, robust_z, scores

RED, BLUE, GREEN, GRAY, DARK = "#c8553d", "#2a78c4", "#0e8a5f", "#9aa0a6", "#202124"
PART_COLORS = {"hit_mean_top3": BLUE, "hit_max_top3": "#7b4fc9", "dlogp_mean_rel": "#d9a21b", "dlogp_max_rel": GREEN}
FAMILIES = {"patch": "Patches", "blend": "Blends", "sig": "SIG stripes", "warp": "Warps", "color": "Color changes",
            "quality": "Lower quality", "frequency": "Frequency patterns", "geo": "Geometry", "replace": "Replacements"}
# The class of the image that shows each probe family. A different image for each family gives more variety.
FAMILY_IMAGE = {"patch": 8, "blend": 0, "sig": 15, "warp": 7, "color": 18, "quality": 13, "frequency": 3, "geo": 14, "replace": 11}
# Three probes for each family. All probes in the README would be too many images.
FAMILY_EXAMPLES = {
    "patch": ["patch_12_red_tl", "patch_8_checker_br", "patch_12_noise_center"],
    "blend": ["blend_noise1_0.3", "blend_smooth1_0.3", "blend_diagonal_0.3"],
    "sig": ["sig_h_4_0.12", "sig_v_8_0.12", "sig_v_12_0.12"],
    "warp": ["warp_1_2", "warp_1_3", "warp_2_3"],
    "color": ["color_invert", "color_bgr", "color_tint_red"],
    "quality": ["quantize_4", "blur_2.0", "gauss_0.1"],
    "frequency": ["highfreq_0.05", "lowfreq_1", "lowfreq_3"],
    "geo": ["geo_hflip", "geo_rot90", "geo_zoom"],
    "replace": ["replace_uniform", "replace_gauss", "replace_gray"],
}
TYPE_NAMES = {"patch": "patch (BadNets)", "blend": "blend", "sig": "SIG", "warp": "warp (WaNet)", "noise": "noise",
              "color": "color", "reflect": "reflect (ReFool)", "quantize": "quantize (BPP)", "lowfreq": "low freq. (LF)",
              "ftrojan": "FTrojan", "dynamic": "dynamic (SSBA)"}


def type_label(set_name: str, kind: str) -> str:
    """Set B has a clean-label patch and blend. Set A has a dirty-label patch and blend."""
    clean = set_name == "b" and kind in val_models.SETS["b"]["always_clean_label"]
    return TYPE_NAMES[kind] + (" (clean-label)" if clean else "")


def family(probe: str) -> str:
    head = probe.split("_")[0]
    if head in ("quantize", "blur", "resample", "gauss"):
        return "quality"
    if head in ("highfreq", "lowfreq"):
        return "frequency"
    return head


def img(t: torch.Tensor) -> np.ndarray:
    return t.permute(1, 2, 0).cpu().numpy().clip(0, 1)


def show(ax, image, title="", size=8):
    ax.imshow(image, interpolation="nearest")
    ax.set_title(title, fontsize=size)
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", path.name, flush=True)


def class_names() -> list[str]:
    path = next(data_dir().rglob("classes.txt"), None)
    if path is None:
        return [f"c{c:02d}" for c in range(N_CLASSES)]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()][:N_CLASSES]


# The flow chart of the method: wide stages from top to bottom, some with boxes side by side.
STAGES = [
    ("1. Input", "", [("400 clean images", "20 images of each class.\nNo image has a trigger."),
                      ("206 task models", "6 train and 200 test models.\nSame base model, same data, same finetune.")]),
    ("2. Probe bank", "196 changes of the images. Each model gets exactly the same probes.",
     [("Patches (120)", "small squares in the\ncorners and the center"),
      ("Blends and SIG (34)", "full-image patterns\nand stripes"),
      ("Warps and colors (18)", "shape changes and\ncolor changes"),
      ("Other families (24)", "fewer colors, blur, noise,\nfine patterns, flips,\ngray or noise images")]),
    ("3. Scan of each model", "The model sees the 400 images after each probe. One model takes approximately 28 s on a T4 GPU.",
     [("hit table: 20 × 20 for each probe", "Which share of the images of class A does the model put into class B?\n"
                                            "A hard signal: the image really moved."),
      ("dlogp table: 20 × 20 for each probe", "How much does the probability of class B increase on the images of class A?\n"
                                              "A soft signal: it also shows a weak pull.")]),
    ("4. Comparison with the siblings (robust z-score)",
     "Each number is compared with the same number in all other models: z = (value - median) / spread.\n"
     "A clean class reacts the same in all models. A backdoor class is far from all of them, so its z is large.", []),
    ("5. Four parts for each class, from the 3 strongest probes", "",
     [("hit, all classes", "images of all classes move\n(normal backdoor)"),
      ("hit, one class", "images of one class move\n(targeted backdoor)"),
      ("dlogp, all classes", "all classes are pulled\n(weak backdoor)"),
      ("dlogp, one class", "one class is pulled\n(weak targeted backdoor)")]),
    ("6. One score for each class: combo4_z",
     "The mean of the four parts. One part can be large by chance. Four parts together are a pattern.", []),
    ("7. Submission and check",
     "200 test models × 20 classes: public AUC 0.996.\n"
     "Check on our own models, only to reject a bad method: set A (known trigger types) 0.991, set B (new trigger types) 0.9998.", []),
]
STAGE_THUMBS = {"Patches (120)": "patch_8_red_br", "Blends and SIG (34)": "blend_smooth1_0.3",
                "Warps and colors (18)": "color_invert", "Other families (24)": "quantize_4"}
STAGE_COLORS = {"4": ("#fbe9e3", RED), "6": ("#e3f1ec", GREEN), "7": ("#e3f1ec", GREEN)}


def fig_pipeline(x, y, probes, out) -> None:
    by_name = dict(probes)
    i = int(torch.nonzero(y == FAMILY_IMAGE["patch"])[0])
    line = 1.7

    def lines(text):
        return text.count("\n") + 1 if text else 0

    sizes = []
    for title, desc, boxes in STAGES:
        box_h = max([1.6 + line * (1 + lines(t)) for _, t in boxes] + [9.5 if any(b in STAGE_THUMBS for b, _ in boxes) else 0])
        sizes.append((1.4 + line + line * lines(desc) + (box_h + 1.4 if boxes else 0.6), box_h))
    gap = 3.2
    total = sum(h for h, _ in sizes) + gap * (len(STAGES) - 1) + 1
    fig = plt.figure(figsize=(14, total * 0.14))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total)
    ax.axis("off")
    top = total - 0.5
    for k, ((title, desc, boxes), (h, box_h)) in enumerate(zip(STAGES, sizes)):
        fill, edge = STAGE_COLORS.get(title[0], ("#f1f3f4", "#80868b"))
        ax.add_patch(FancyBboxPatch((1, top - h), 98, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=edge, lw=1.5))
        ax.text(3, top - 1.2, title, fontsize=12.5, weight="bold", va="top")
        if desc:
            ax.text(3, top - 1.2 - line - 0.2, desc, fontsize=10, va="top", linespacing=1.6)
        n = len(boxes)
        for j, (box_title, text) in enumerate(boxes):
            width = (94 - 2 * (n - 1)) / n
            bx, by = 3 + j * (width + 2), top - h + 1.2
            ax.add_patch(FancyBboxPatch((bx, by), width, box_h, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=edge, lw=1))
            tx = bx + 1.2
            if box_title in STAGE_THUMBS:
                thumb = ax.inset_axes([bx + 0.8, by + 0.8, box_h - 1.6, box_h - 1.6], transform=ax.transData)
                show(thumb, img(by_name[STAGE_THUMBS[box_title]](x)[i]))
                tx = bx + box_h + 0.2
            ax.text(tx, by + box_h - 1.0, box_title, fontsize=10.5, weight="bold", va="top")
            ax.text(tx, by + box_h - 1.0 - line, text, fontsize=9.5, va="top", linespacing=1.5)
        if k < len(STAGES) - 1:
            ax.annotate("", (50, top - h - gap + 0.1), (50, top - h - 0.1), arrowprops={"arrowstyle": "-|>", "color": DARK, "lw": 1.6})
        top -= h + gap
    fig.savefig(out / "pipeline.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved pipeline.png", flush=True)


# Task data and probes.

def fig_classes(x, y, names, out) -> None:
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.7))
    for c, ax in enumerate(axes.flat):
        show(ax, img(x[y == c][0]), f"c{c:02d}\n{names[c]}")
    save(fig, out / "classes.png")


def fig_family(x, y, probes, name_family, out) -> None:
    if name_family == "frequency":
        # The frequency patterns are weak. They are visible only on a flat image.
        i = int(x.std(dim=(1, 2, 3)).argmin())
    else:
        i = int(torch.nonzero(y == FAMILY_IMAGE[name_family])[0])
    bank = dict(probes)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
    for ax, name in zip(axes, FAMILY_EXAMPLES[name_family]):
        show(ax, img(bank[name](x)[i]), name, 11)
    save(fig, out / f"probes_{name_family}.png")


# Scan of the task models.

def hit_mean(feat) -> np.ndarray:
    """(models, probes, class): the fraction of the images of the other classes that go to the class."""
    v = feat["hit"].astype(np.float32)
    v[:, :, ~OFF_DIAGONAL] = np.nan
    return np.nanmean(v, axis=2)


def probe_z(models, feat) -> np.ndarray:
    """(models, probes, class): the larger of the z of the mean and of the maximum over the source classes (as in score.py)."""
    sets = models["set"].to_numpy()
    v = feat["hit"].astype(np.float64)
    v[:, :, ~OFF_DIAGONAL] = np.nan
    mean = robust_z(np.nanmean(v, axis=2), sets)
    cell = robust_z(np.nan_to_num(v, nan=0.0), sets)
    cell[:, :, ~OFF_DIAGONAL] = -np.inf
    return np.maximum(mean, robust_z(cell.max(axis=2), sets))


def train_rows(models) -> list[tuple[int, int | None]]:
    """The labeled task models: (row, target or None)."""
    rows = []
    for i, r in models.iterrows():
        if r["set"] == "real" and r["labeled"] == "True":
            rows.append((i, int(r["targets"]) if r["targets"] else None))
    return sorted(rows, key=lambda t: t[1] is None)


EXAMPLE_ID = "57c03db99edc"  # A train model with a backdoor. Its strongest probe is a small patch, easy to see.


def fig_probe_answer(x, y, models, probes, hm, names, out) -> tuple[int, int, int]:
    """One probe on one train model with a backdoor: where the images go, in this model and in a typical model."""
    real = (models["set"] == "real").to_numpy()
    m = int(np.flatnonzero(models["id"].to_numpy() == EXAMPLE_ID)[0])
    t = int(models.at[m, "targets"])
    # The probe with the largest z, as the score selects it. Many probes move all images in this model.
    p = int(np.argmax(robust_z(hm[:, :, t], models["set"].to_numpy())[m]))
    name, fn = probes[p]
    i = int(torch.nonzero(y == (t + 5) % N_CLASSES)[0])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 4.5]})
    show(a0, img(fn(x)[i]), f"probe {name}", 9)
    cls = np.arange(N_CLASSES)
    a1.bar(cls - 0.2, np.median(hm[real, p], axis=0), 0.4, color=GRAY, label="typical model (median of the 206 task models)")
    # One color for the example model: its bars are almost 0 outside the target, so a second color is not visible.
    a1.bar(cls + 0.2, hm[m, p], 0.4, color=RED, label=f"model {EXAMPLE_ID} (backdoor)")
    a1.annotate(f"{hm[m, p, t]:.0%} of the images\ngo to c{t:02d} {names[t]}", (t + 0.2, hm[m, p, t]), (t + 2, 0.75),
                fontsize=10, arrowprops={"arrowstyle": "->", "color": DARK})
    a1.set_xticks(cls, [f"c{c:02d} {names[c]}" for c in cls], fontsize=7.5, rotation=45, ha="right")
    a1.set_ylim(0, 1)
    a1.set_ylabel("share of the images")
    a1.set_title("After the probe: to which class do the images of the other classes go?", fontsize=10)
    a1.legend(fontsize=8.5, loc="upper right")
    a1.grid(axis="y", alpha=0.3)
    save(fig, out / "probe_answer.png")
    return m, p, t


def fig_zscore(models, hm, m, p, t, names, probe_name, out) -> dict:
    """The same value in all task models, and the robust z-score of the example model."""
    sets = models["set"].to_numpy()
    real = sets == "real"
    values = hm[real, p, t]
    z = robust_z(hm[:, p, t][:, None], sets)[m, 0]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.scatter(values, rng.uniform(-1, 1, len(values)), s=22, color=GRAY, alpha=0.7, label="the other task models")
    ax.scatter([hm[m, p, t]], [0], s=120, color=RED, zorder=3, label=f"model {EXAMPLE_ID}")
    median = np.median(values)
    ax.axvline(median, color=DARK, ls="--", lw=1)
    ax.annotate(f"median {median:.0%}", (median, 1.2), (0.08, 1.45), fontsize=10, arrowprops={"arrowstyle": "->"})
    ax.annotate(f"{hm[m, p, t]:.0%}: z = {z:.0f}", (hm[m, p, t], 0), (hm[m, p, t] - 0.25, -1.6), fontsize=10,
                arrowprops={"arrowstyle": "->"})
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-2.2, 2.2)
    ax.set_yticks([])
    ax.set_xlabel(f"share of the images that go to c{t:02d} {names[t]} after the probe {probe_name}")
    ax.set_title("The same number in all 206 task models (one dot = one model)", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, out / "zscore.png")
    return {"model": EXAMPLE_ID, "class": t, "value": float(hm[m, p, t]), "median": float(median), "z": float(z)}


def fig_parts(models, all_scores, names, out) -> None:
    """The four parts and combo4_z for the three backdoor targets and for the strongest class of a clean model."""
    ids = models["id"].to_numpy()
    pairs = [(int(np.flatnonzero(ids == i)[0]), int(t), "backdoor") for i, t in
             models.loc[(models["set"] == "real") & (models["targets"] != ""), ["id", "targets"]].to_numpy()]
    clean = int(np.flatnonzero(ids == "6e9c713cd671")[0])
    pairs.append((clean, int(np.argmax(all_scores["combo4_z"][clean])), "clean model"))
    labels = ["hit\nmean", "hit\nmax", "dlogp\nmean", "dlogp\nmax", "combo4_z\n(mean)"]
    keys = ["hit_mean_top3", "hit_max_top3", "dlogp_mean_rel", "dlogp_max_rel", "combo4_z"]
    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 3.6), sharey=True)
    for ax, (m, c, kind) in zip(axes, pairs):
        v = [np.clip(all_scores[k][m, c], -10, 50) for k in keys]
        ax.bar(range(5), v, color=[BLUE] * 4 + [RED if kind == "backdoor" else GRAY], width=0.65)
        for k, value in enumerate(v):
            ax.text(k, value + 0.8, f"{value:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(range(5), labels, fontsize=8)
        ax.set_title(f"{ids[m]}, c{c:02d} {names[c]}\n({kind})", fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("z (clipped to 50)")
    save(fig, out / "parts.png")


def fig_hit_matrix(models, feat, probes, out) -> dict:
    """The table hit of one probe: a backdoor in a train model, a clean train model, and a targeted backdoor (set A)."""
    hit = feat["hit"].astype(np.float32)
    sets = models["set"].to_numpy()
    hm = hit.copy()
    hm[:, :, ~OFF_DIAGONAL] = np.nan
    hm = np.nanmean(hm, axis=2)
    rows = train_rows(models)
    m_bad, t_bad = max([r for r in rows if r[1] is not None], key=lambda r: hm[r[0], :, r[1]].max())
    p_bad = int(np.argmax(hm[m_bad, :, t_bad]))
    m_clean = next(r[0] for r in rows if r[1] is None)
    targeted = [(i, int(t)) for i, r in models.iterrows() if r["set"] == "a"
                for t, v in zip(r["targets"].split(";"), r["variants"].split(";")) if v == "targeted"]
    best = None
    for i, t in targeted:
        part = hit[sets == "a"][:, :, :, t]
        d = np.nan_to_num(hit[i, :, :, t] - np.median(part, axis=0))
        d[:, t] = -1
        p, s = np.unravel_index(np.argmax(d), d.shape)
        if best is None or d[p, s] > best[4]:
            best = (i, t, int(p), int(s), d[p, s])
    m_tg, t_tg, p_tg, s_tg, _ = best
    panels = [(m_bad, p_bad, f"train model with a backdoor (target c{t_bad:02d})"),
              (m_clean, p_bad, "clean train model, same probe"),
              (m_tg, p_tg, f"set A, targeted backdoor (c{s_tg:02d} -> c{t_tg:02d})")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, (m, p, title) in zip(axes, panels):
        table = hit[m, p].copy()
        table[~OFF_DIAGONAL] = np.nan
        cmap = plt.get_cmap("Reds").copy()
        cmap.set_bad("#e8eaed")
        im = ax.imshow(table, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{title}\nprobe {probes[p][0]}", fontsize=10)
        ticks = [f"c{c:02d}" for c in range(N_CLASSES)]
        ax.set_xticks(range(N_CLASSES), ticks, fontsize=6.5, rotation=90)
        ax.set_yticks(range(N_CLASSES), ticks, fontsize=6.5)
        ax.set_xlabel("class after the probe")
        ax.set_ylabel("true class of the image (source)")
    fig.colorbar(im, ax=axes, fraction=0.015, label="fraction of the 20 images")
    fig.savefig(out / "hit_matrix.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved hit_matrix.png", flush=True)
    return {"backdoor": [str(models.at[m_bad, "id"]), probes[p_bad][0]], "targeted": [str(models.at[m_tg, "id"]), probes[p_tg][0], s_tg, t_tg]}


def fig_train_scores(models, s, out) -> None:
    rows = train_rows(models)
    table = np.array([s[m] for m, _ in rows])
    fig, ax = plt.subplots(figsize=(15, 3.6))
    im = ax.imshow(table, cmap="Blues", vmin=min(0, table.min()), vmax=table.max(), aspect="auto")
    for r, (m, t) in enumerate(rows):
        for c in range(N_CLASSES):
            ax.text(c, r, f"{table[r, c]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if table[r, c] > 0.6 * table.max() else DARK)
        if t is not None:
            ax.add_patch(Rectangle((t - 0.5, r - 0.5), 1, 1, fill=False, ec=RED, lw=2.5))
    ax.set_xticks(range(N_CLASSES), [f"c{c:02d}" for c in range(N_CLASSES)], fontsize=8)
    ax.set_yticks(range(len(rows)), [f"{models.at[m, 'id']} ({'clean' if t is None else f'target c{t:02d}'})" for m, t in rows],
                  fontsize=8)
    ax.set_title("combo4_z of the 6 train models (red frame = true backdoor target)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.02)
    save(fig, out / "train_scores.png")


def fig_test_scores(models, s, out) -> dict:
    test = ((models["set"] == "real") & (models["labeled"] != "True")).to_numpy()
    top = s[test].max(axis=1)
    above = (s[test] > 10).sum(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].hist(top, bins=np.arange(0, max(36, top.max() + 1), 1), color=BLUE)
    axes[0].axvline(10, color=RED, ls="--", label="10 (only for the description)")
    axes[0].set_xlabel("largest combo4_z in the model")
    axes[0].set_ylabel("test models")
    axes[0].set_title("Largest score of each of the 200 test models", fontsize=10)
    axes[0].legend(fontsize=8)
    counts = np.bincount(above, minlength=4)
    axes[1].bar(range(len(counts)), counts, color=BLUE, width=0.6)
    for k, n in enumerate(counts):
        axes[1].text(k, n + 1, str(n), ha="center", fontsize=9)
    axes[1].set_xlabel("classes above 10 in the model")
    axes[1].set_ylabel("test models")
    axes[1].set_title("Suspicious classes in each test model", fontsize=10)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    save(fig, out / "test_scores.png")
    return {"pairs_above_10": int(above.sum()), "models_above_10": int((above > 0).sum()), "counts": counts.tolist()}


# Validation sets A and B.

def val_labels() -> pd.DataFrame:
    frames = []
    for name, folder in VAL_FOLDERS.items():
        labels = pd.read_csv(next(KAGGLE_INPUT.rglob(f"{folder}/labels.csv")), dtype=str, keep_default_na=False)
        labels.insert(0, "set", name)
        frames.append(labels)
    return pd.concat(frames, ignore_index=True)


def backdoor_rows(labels: pd.DataFrame) -> pd.DataFrame:
    """One row for each backdoor of each model, with its type, variant, ASR and the validity of the model."""
    rows = []
    for r in labels.itertuples():
        for b, asr in zip(json.loads(r.backdoors), r.asr.split(";") if r.asr else []):
            rows.append({"set": r.set, "type": b["type"], "variant": b["variant"], "asr": float(asr), "valid": r.valid == "True"})
    return pd.DataFrame(rows)


def fig_triggers(x, y, out) -> None:
    base = 17
    i = int(torch.nonzero(y == base)[0])
    val_models.REFLECTIONS = (x[y != base][:val_models.N_REFLECTIONS] * 255).round().to(torch.uint8)
    rng = np.random.default_rng(0)
    ncol = max(len(cfg["types"]) for cfg in val_models.SETS.values()) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(ncol * 1.75, 2 * 1.95))
    for ax in axes.flat:
        ax.axis("off")
    for k, (name, cfg) in enumerate(val_models.SETS.items()):
        show(axes[k, 0], img(x[i]), f"set {name.upper()}: original")
        axes[k, 0].axis("on")
        for j, kind in enumerate(cfg["types"]):
            spec = {"type": kind, "params": val_models.trigger_params(kind, rng, 1.0)}
            title = TYPE_NAMES[kind] + ("\n(clean-label)" if kind in cfg["always_clean_label"] else "")
            axes[k, j + 1].axis("on")
            show(axes[k, j + 1], img(val_models.apply_trigger(x[i:i + 1], spec)[0]), title, 8)
    save(fig, out / "triggers.png")


def fig_val_sets(labels: pd.DataFrame, out) -> dict:
    b = backdoor_rows(labels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    info = {}
    for ax, name in zip(axes, ("a", "b")):
        part = b[b["set"] == name]
        kinds = val_models.SETS[name]["types"]
        ok = [int(((part["type"] == k) & part["valid"]).sum()) for k in kinds]
        bad = [int(((part["type"] == k) & ~part["valid"]).sum()) for k in kinds]
        ax.barh(range(len(kinds)), ok, color=GREEN, height=0.6, label="in valid models")
        ax.barh(range(len(kinds)), bad, left=ok, color=GRAY, height=0.6, label="in rejected models")
        for j, k in enumerate(kinds):
            asr = part.loc[part["type"] == k, "asr"].mean()
            ax.text(ok[j] + bad[j] + 0.3, j, f"ASR {asr:.2f}", va="center", fontsize=8)
        ax.set_yticks(range(len(kinds)), [type_label(name, k) for k in kinds], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(o + r for o, r in zip(ok, bad)) + 6)
        ax.set_xlabel("backdoors")
        models_valid = labels[(labels["set"] == name) & (labels["valid"] == "True")]
        ax.set_title(f"Set {name.upper()}: {len(models_valid)} of {int((labels['set'] == name).sum())} models are valid", fontsize=10)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        ax.grid(axis="x", alpha=0.3)
        info[name] = {"models": int((labels["set"] == name).sum()), "valid": len(models_valid),
                      "backdoors_valid": int(sum(ok)), "backdoors_rejected": int(sum(bad))}
    ax = axes[2]
    valid = labels[labels["valid"] == "True"].copy()
    valid["acc"] = valid["clean_acc"].astype(float)
    bins = np.linspace(0.6, 0.92, 33)
    ax.hist(valid.loc[valid["targets"] == "", "acc"], bins=bins, color=BLUE, alpha=0.75, label="our clean models")
    ax.hist(valid.loc[valid["targets"] != "", "acc"], bins=bins, color=RED, alpha=0.75, label="our models with a backdoor")
    ax.axvspan(0.837, 0.848, color=GREEN, alpha=0.3, label="the 6 train models of the task")
    ax.set_xlabel("accuracy on the 400 clean task images")
    ax.set_ylabel("models")
    ax.set_title("Accuracy of the valid models (sets A and B)", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, out / "val_sets.png")
    return info


def fig_val_scores(models, s, out) -> None:
    y, _, _ = pair_labels(models)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, name in zip(axes, ("a", "b")):
        m = (models["set"] == name).to_numpy()
        yy, ss = y[m].ravel(), s[m].ravel()
        bins = np.linspace(min(ss.min(), -2), ss.max() + 1, 50)
        ax.hist(ss[yy == 0], bins=bins, color=BLUE, alpha=0.8, label=f"other pairs ({int((yy == 0).sum())})")
        ax.hist(ss[yy == 1], bins=bins, color=RED, alpha=0.8, label=f"backdoor targets ({int(yy.sum())})")
        ax.set_yscale("log")
        ax.set_xlabel("combo4_z")
        ax.set_ylabel("(model, class) pairs")
        ax.set_title(f"Set {name.upper()}: 40 models x 20 classes", fontsize=10)
        ax.legend(fontsize=8)
    save(fig, out / "val_scores.png")


def fig_auc(table: pd.DataFrame, out) -> None:
    names = ["combo4_z"] + PARTS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, name in zip(axes, ("a", "b")):
        part = table[table["set"] == name].set_index("score")
        groups = ["auc"] + [c for c in part.columns if (c.startswith("type_") or c.startswith("variant_")) and part[c].notna().any()]
        for k, score_name in enumerate(names):
            values = part.loc[score_name, groups].astype(float).to_numpy()
            if score_name == "combo4_z":
                ax.scatter(values, range(len(groups)), color=DARK, marker="D", s=60, zorder=4, label=score_name)
            else:
                ax.scatter(values, range(len(groups)), color=PART_COLORS[score_name], s=28, zorder=3, label=score_name)
        labels = ["all backdoors"] + [type_label(name, g[5:]) if g.startswith("type_") else "variant " + g[8:] for g in groups[1:]]
        ax.set_yticks(range(len(groups)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0.83, 1.003)
        ax.set_xlabel("AUC")
        ax.set_title(f"Set {name.upper()}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower left")
    save(fig, out / "auc_types.png")


def fig_probe_vs_trigger(models, z, probes, out) -> pd.DataFrame:
    """For each backdoor of the validation sets: the probe family with the largest z for the target class."""
    rows = []
    for i, r in models.iterrows():
        if r["set"] not in ("a", "b") or not r["targets"]:
            continue
        for t, kind, variant in zip(r["targets"].split(";"), r["types"].split(";"), r["variants"].split(";")):
            p = int(np.argmax(z[i, :, int(t)]))
            rows.append({"set": r["set"], "id": r["id"], "type": kind, "variant": variant, "target": int(t),
                         "probe": probes[p][0], "family": family(probes[p][0]), "z": float(z[i, p, int(t)])})
    best = pd.DataFrame(rows)
    keys = [(s, k) for s in ("a", "b") for k in val_models.SETS[s]["types"] if ((best["set"] == s) & (best["type"] == k)).any()]
    fams = list(FAMILIES)
    counts = np.array([[((best["set"] == s) & (best["type"] == k) & (best["family"] == f)).sum() for f in fams] for s, k in keys])
    frac = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 0.48 * len(keys) + 1.6))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r in range(len(keys)):
        for c in range(len(fams)):
            if counts[r, c]:
                ax.text(c, r, str(counts[r, c]), ha="center", va="center", fontsize=9, color="white" if frac[r, c] > 0.6 else DARK)
    ax.axhline(sum(s == "a" for s, _ in keys) - 0.5, color=DARK, lw=1.5)
    ax.set_xticks(range(len(fams)), [FAMILIES[f] for f in fams], fontsize=9, rotation=20)
    ax.set_yticks(range(len(keys)), [f"{s.upper()}: {type_label(s, k)}" for s, k in keys], fontsize=9)
    ax.set_title("Which probe family gives the largest z for the backdoor target (number of backdoors)", fontsize=10)
    save(fig, out / "probe_vs_trigger.png")
    return best


def run(out) -> None:
    out.mkdir(parents=True, exist_ok=True)
    x, y = clean_images()
    names = class_names()
    probes = probe_bank("cpu")
    summary = {"classes": names}

    fig_pipeline(x, y, probes, out)
    fig_classes(x, y, names, out)
    for name in FAMILIES:
        fig_family(x, y, probes, name, out)
    fig_triggers(x, y, out)
    labels = val_labels()
    summary["val_sets"] = fig_val_sets(labels, out)

    models, feat = load(next(KAGGLE_INPUT.rglob("scan/features.npz")).parent)
    all_scores = scores(models, feat)
    s = all_scores["combo4_z"]
    table = report(models, all_scores)
    table.to_csv(out / "report.csv", index=False)
    hm = hit_mean(feat)
    m, p, t = fig_probe_answer(x, y, models, probes, hm, names, out)
    summary["zscore"] = fig_zscore(models, hm, m, p, t, names, probes[p][0], out)
    summary["zscore"]["probe"] = probes[p][0]
    fig_parts(models, all_scores, names, out)
    summary["hit_matrix"] = fig_hit_matrix(models, feat, probes, out)
    fig_train_scores(models, s, out)
    summary["test"] = fig_test_scores(models, s, out)
    fig_val_scores(models, s, out)
    fig_auc(table, out)
    best = fig_probe_vs_trigger(models, probe_z(models, feat), probes, out)
    best.to_csv(out / "best_probes.csv", index=False)
    summary["family_sizes"] = {f: sum(family(n) == f for n, _ in probes) for f in FAMILIES}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(best.groupby(["set", "type"])["family"].value_counts().to_string(), flush=True)
