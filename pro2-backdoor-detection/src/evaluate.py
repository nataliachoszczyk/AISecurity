"""Score the (model, class) pairs from the detection features, and measure the AUC on the labeled models.

The AUC is the AUC of the task: all (model, class) pairs of a set together. Each score compares a value with the same
value (same class or same pair of classes) in all models of the same set. Thus the scores of different models are comparable.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.detect import FAMILIES
from src.model_stats import N_CLASSES

SEED = 0


def load(folder) -> tuple[pd.DataFrame, dict]:
    """Return the models (in the sequence of the features) and the features."""
    feat = dict(np.load(folder / "features.npz", allow_pickle=True))
    models = pd.read_csv(folder / "models.csv", dtype=str, keep_default_na=False).set_index(["set", "id"])
    models = models.loc[list(zip(feat["set"], feat["id"]))].reset_index()
    return models, feat


def pair_labels(models: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (models, 20) arrays: 1 for a target class, and the type and variant of its backdoor."""
    y = np.zeros((len(models), N_CLASSES), dtype=int)
    kind = np.full(y.shape, "", dtype=object)
    variant = kind.copy()
    for i, r in enumerate(models.itertuples()):
        targets = [int(t) for t in r.targets.split(";") if t]
        types = r.types.split(";") if r.types else [""] * len(targets)
        variants = r.variants.split(";") if r.variants else [""] * len(targets)
        for t, k, v in zip(targets, types, variants):
            y[i, t], kind[i, t], variant[i, t] = 1, k, v
    return y, kind, variant


def robust_z(values: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """Compare each model with the other models of the same set (axis 0): (value - median) / MAD."""
    out = np.full(values.shape, np.nan)
    for name in np.unique(sets):
        part = values[sets == name]
        median = np.nanmedian(part, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(part - median), axis=0)
        out[sets == name] = (part - median) / (mad + 1e-6)
    return out


def scores(models: pd.DataFrame, feat: dict) -> dict[str, np.ndarray]:
    """Return candidate scores, each of shape (models, 20). A larger score means a larger suspicion."""
    sets = models["set"].to_numpy()
    out = {}
    for family in FAMILIES:
        z = robust_z(feat[f"{family}_margin"], sets)
        # The mean over the source classes finds a backdoor for all classes. The max finds a targeted backdoor.
        out[f"{family}_mean"] = np.nanmean(z, axis=1)
        out[f"{family}_max"] = np.nanmax(z, axis=1)
        out[f"{family}_asr_mean"] = np.nanmean(robust_z(feat[f"{family}_asr"], sets), axis=1)
    out["patch_small_mean"] = np.nanmean(robust_z(-feat["patch_size"], sets), axis=1)
    for i, name in enumerate(feat["stat_names"]):
        out[f"stat_{name}"] = robust_z(feat["stats"][:, :, i], sets)
    out["inv_mean_max"] = np.max([out[f"{f}_mean"] for f in FAMILIES], axis=0)
    out["inv_max_max"] = np.max([out[f"{f}_max"] for f in FAMILIES], axis=0)
    out["inv_mean_sum"] = np.sum([out[f"{f}_mean"] for f in FAMILIES], axis=0)
    return out


def auc(y: np.ndarray, s: np.ndarray) -> float:
    s = np.nan_to_num(s, nan=-1e9)
    return roc_auc_score(y.ravel(), s.ravel()) if 0 < y.sum() < y.size else np.nan


def report(models: pd.DataFrame, feat: dict, sets: list[str]) -> pd.DataFrame:
    """Return the AUC of each score on the labeled models of each set, also for each trigger type and variant."""
    y, kind, variant = pair_labels(models)
    all_scores = scores(models, feat)
    rows = []
    for name in sets:
        rows_mask = ((models["set"] == name) & (models["labeled"] == "True")).to_numpy()
        if not rows_mask.any():
            continue
        yy, kk, vv = y[rows_mask], kind[rows_mask], variant[rows_mask]
        for score_name, s in all_scores.items():
            ss = s[rows_mask]
            row = {"set": name, "score": score_name, "auc": auc(yy, ss), "positives": int(yy.sum())}
            # One group of positives against all negatives of the set.
            for group, labels in [(f"type_{k}", kk == k) for k in sorted(set(kk[yy == 1]) - {""})] + \
                                 [(f"variant_{v}", vv == v) for v in sorted(set(vv[yy == 1]) - {""})]:
                keep = (yy == 0) | labels
                row[group] = auc(yy[keep], ss[keep])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["set", "auc"], ascending=[True, False])


def bootstrap(y: np.ndarray, s: np.ndarray, n: int = 1000) -> tuple[float, float]:
    """Return the 5% and 95% quantiles of the AUC over a resample of the models."""
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n):
        pick = rng.integers(0, len(y), len(y))
        values.append(auc(y[pick], s[pick]))
    return tuple(np.nanpercentile(values, [5, 95]))
