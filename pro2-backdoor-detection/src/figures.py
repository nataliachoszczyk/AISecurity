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
from matplotlib.patches import Rectangle

from src import val_models
from src.paths import KAGGLE_INPUT, data_dir
from src.scan import N_CLASSES, VAL_FOLDERS, clean_images, probe_bank
from src.score import OFF_DIAGONAL, PARTS, load, pair_labels, report, robust_z, scores

RED, BLUE, GREEN, GRAY, DARK = "#c8553d", "#2a78c4", "#0e8a5f", "#9aa0a6", "#202124"
PART_COLORS = {"hit_mean_top3": BLUE, "hit_max_top3": "#7b4fc9", "dlogp_mean_rel": "#d9a21b", "dlogp_max_rel": GREEN}
FAMILIES = {"patch": "Patches", "blend": "Blends", "sig": "SIG stripes", "warp": "Warps", "color": "Color changes",
            "quality": "Lower quality", "frequency": "Frequency patterns", "geo": "Geometry", "replace": "Replacements"}
# The class of the image that shows each probe family. A different image for each family gives more variety.
FAMILY_IMAGE = {"patch": 8, "blend": 0, "sig": 15, "warp": 13, "color": 18, "quality": 7, "frequency": 3, "geo": 14, "replace": 11}
# These probes are weak, so the figure also shows the change.
SUBTLE = {"blend", "sig", "warp", "quality", "frequency"}
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


def change(before: torch.Tensor, after: torch.Tensor, gain: float) -> np.ndarray:
    """Gray is no change."""
    return img(0.5 + gain * (after - before))


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


# Task data and probes.

def fig_classes(x, y, names, out) -> None:
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.7))
    for c, ax in enumerate(axes.flat):
        show(ax, img(x[y == c][0]), f"c{c:02d}\n{names[c]}")
    save(fig, out / "classes.png")


def fig_patches(x, y, probes, out) -> None:
    """All 120 patch probes: one row for each pattern, one column for each size and position."""
    i = int(torch.nonzero(y == FAMILY_IMAGE["patch"])[0])
    patch = [(name, fn) for name, fn in probes if family(name) == "patch"]
    patterns = list(dict.fromkeys(name.split("_")[2] for name, _ in patch))
    places = list(dict.fromkeys("_".join([name.split("_")[1], name.split("_")[3]]) for name, _ in patch))
    fig, axes = plt.subplots(len(patterns), len(places), figsize=(len(places) * 0.95, len(patterns) * 1.02))
    for name, fn in patch:
        _, size, pattern, where = name.split("_")
        ax = axes[patterns.index(pattern), places.index(f"{size}_{where}")]
        show(ax, img(fn(x)[i]), f"{size}px {where}" if pattern == patterns[0] else "", 7)
        if where == "tl" and size == "4":
            ax.set_ylabel(pattern, fontsize=8)
    save(fig, out / "probes_patch.png")


def fig_family(x, y, probes, name_family, out) -> None:
    i = int(torch.nonzero(y == FAMILY_IMAGE[name_family])[0])
    group = [(name, fn) for name, fn in probes if family(name) == name_family]
    ncol = min(len(group), 6)
    blocks = -(-len(group) // ncol)
    rows_per_block = 2 if name_family in SUBTLE else 1
    fig, axes = plt.subplots(blocks * rows_per_block, ncol + 1, figsize=((ncol + 1) * 1.45, blocks * rows_per_block * 1.6),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for b in range(blocks):
        show(axes[b * rows_per_block, 0], img(x[i]), "original")
        axes[b * rows_per_block, 0].axis("on")
    for k, (name, fn) in enumerate(group):
        b, col = divmod(k, ncol)
        after = fn(x)[i]
        ax = axes[b * rows_per_block, col + 1]
        ax.axis("on")
        show(ax, img(after), name, 7.5)
        if rows_per_block == 2:
            ax = axes[b * rows_per_block + 1, col + 1]
            ax.axis("on")
            show(ax, change(x[i], after, 4), "change x4", 7)
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


def fig_scan_example(x, y, models, feat, probes, hm, names, out) -> dict:
    """For each train model: the strongest probe, the reaction of each class, and the same value in all task models."""
    real = (models["set"] == "real").to_numpy()
    z = robust_z(hm, models["set"].to_numpy())
    rows = train_rows(models)
    rows = [r for r in rows if r[1] is not None] + [r for r in rows if r[1] is None][:1]
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.1 * len(rows)), gridspec_kw={"width_ratios": [1, 3, 2.2]})
    info = {}
    for k, (m, t) in enumerate(rows):
        cls = t if t is not None else int(np.argmax(z[m].max(axis=0)))
        p = int(np.argmax(z[m, :, cls]))
        name, fn = probes[p]
        i = int(torch.nonzero(y == (cls + 5) % N_CLASSES)[0])
        label = f"{models.at[m, 'id']}\n" + (f"backdoor target c{t:02d}" if t is not None else "clean model")
        show(axes[k, 0], img(fn(x)[i]), f"{label}\nstrongest probe: {name}", 8.5)
        ax = axes[k, 1]
        colors = [RED if c == t else BLUE for c in range(N_CLASSES)]
        ax.bar(range(N_CLASSES), hm[m, p], color=colors, width=0.7)
        ax.scatter(range(N_CLASSES), np.median(hm[real, p], axis=0), color=DARK, marker="_", s=180, zorder=3,
                   label="median of all task models")
        ax.set_xticks(range(N_CLASSES), [f"c{c:02d}" for c in range(N_CLASSES)], fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("images that go to the class")
        ax.set_title(f"Answer of this model to the probe {name}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax = axes[k, 2]
        values = hm[real, p, cls]
        ax.hist(values, bins=np.linspace(0, 1, 41), color=GRAY)
        ax.axvline(hm[m, p, cls], color=RED if t is not None else BLUE, lw=2, label=f"this model: z = {z[m, p, cls]:.1f}")
        ax.set_yscale("log")
        ax.set_xlabel(f"value of c{cls:02d} {names[cls]} for this probe")
        ax.set_ylabel("task models")
        ax.set_title("The same value in all 206 task models", fontsize=10)
        ax.legend(fontsize=8)
        info[str(models.at[m, "id"])] = {"class": cls, "probe": name, "value": float(hm[m, p, cls]), "z": float(z[m, p, cls]),
                                         "median": float(np.median(values))}
    save(fig, out / "scan_example.png")
    return info


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
    fig, axes = plt.subplots(4, ncol, figsize=(ncol * 1.75, 4 * 1.85))
    for ax in axes.flat:
        ax.axis("off")
    for k, (name, cfg) in enumerate(val_models.SETS.items()):
        show(axes[2 * k, 0], img(x[i]), f"set {name.upper()}: original")
        axes[2 * k, 0].axis("on")
        for j, kind in enumerate(cfg["types"]):
            spec = {"type": kind, "params": val_models.trigger_params(kind, rng, 1.0)}
            after = val_models.apply_trigger(x[i:i + 1], spec)[0]
            title = TYPE_NAMES[kind] + ("\n(clean-label)" if kind in cfg["always_clean_label"] else "")
            for row, image, t in ((2 * k, img(after), title), (2 * k + 1, change(x[i], after, 5), "change x5")):
                axes[row, j + 1].axis("on")
                show(axes[row, j + 1], image, t, 8)
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

    fig_classes(x, y, names, out)
    fig_patches(x, y, probes, out)
    for name in FAMILIES:
        if name != "patch":
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
    summary["scan_example"] = fig_scan_example(x, y, models, feat, probes, hm, names, out)
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
