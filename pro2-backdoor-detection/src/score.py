"""Turn the scan features into one suspicion score for each (model, class) pair, and check the scores.

Each value is compared with the same value (same probe, same classes) in all models of the same set. The models have the
same base, data and finetune, so a backdoor class reacts differently from the same class in the other models. The score of
a class is the maximum over the probes of the mean reaction over the images.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.scan import N_CLASSES

OFF_DIAGONAL = ~np.eye(N_CLASSES, dtype=bool)


def load(folder) -> tuple[pd.DataFrame, dict]:
    """Return the models and the features of a scan."""
    feat = dict(np.load(folder / "features.npz", allow_pickle=True))
    models = pd.read_csv(folder / "models.csv", dtype=str, keep_default_na=False)
    assert (models["id"].to_numpy() == feat["id"]).all()
    return models, feat


def pair_labels(models: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (models, 20) arrays: 1 for a target class, and the type and variant of its backdoor."""
    y = np.zeros((len(models), N_CLASSES), dtype=int)
    kind = np.full(y.shape, "", dtype=object)
    variant = kind.copy()
    for i, r in enumerate(models.itertuples()):
        targets = [int(t) for t in r.targets.split(";") if t]
        types = r.types.split(";") if r.types else ["real"] * len(targets)
        variants = r.variants.split(";") if r.variants else ["real"] * len(targets)
        for t, k, v in zip(targets, types, variants):
            y[i, t], kind[i, t], variant[i, t] = 1, k, v
    return y, kind, variant


def robust_z(values: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """Compare each model with the other models of the same set (axis 0)."""
    out = np.zeros(values.shape)
    for name in np.unique(sets):
        part = values[sets == name]
        median = np.median(part, axis=0)
        # A floor on the scale: many hit fractions are 0 in almost all models, so their MAD is 0.
        scale = np.maximum(1.4826 * np.median(np.abs(part - median), axis=0), np.maximum(0.5 * part.std(axis=0), 1e-2))
        out[sets == name] = (part - median) / scale
    return out


def scores(models: pd.DataFrame, feat: dict) -> dict[str, np.ndarray]:
    """Return the scores of shape (models, 20): the four parts and their mean. A larger score means a larger suspicion."""
    sets = models["set"].to_numpy()
    out = {}
    for measure in ("dlogp", "hit"):
        v = feat[measure].astype(np.float64)  # (models, probes, source, class)
        v[:, :, ~OFF_DIAGONAL] = np.nan
        # "mean": a backdoor for all source classes. "max": a targeted backdoor for one source class.
        mean = robust_z(np.nanmean(v, axis=2), sets)
        cell = robust_z(np.nan_to_num(v, nan=0.0), sets)
        cell[:, :, ~OFF_DIAGONAL] = -np.inf
        top = robust_z(cell.max(axis=2), sets)
        for agg, z in (("mean", mean), ("max", top)):
            if measure == "hit":
                out[f"hit_{agg}_top3"] = np.sort(z, axis=1)[:, -3:].mean(axis=1)
            else:
                # Each class compared with the other classes of the same model and probe first.
                rel = z - np.median(z, axis=2, keepdims=True)
                out[f"dlogp_{agg}_rel"] = rel.max(axis=1)
    # Two measures and two aggregations, so that one weak part does not decide the result.
    out["combo4_z"] = np.mean([np.clip(out[p], -10, 50) for p in PARTS], axis=0)
    return out


PARTS = ["hit_max_top3", "hit_mean_top3", "dlogp_max_rel", "dlogp_mean_rel"]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return roc_auc_score(y.ravel(), s.ravel()) if 0 < y.sum() < y.size else np.nan


def report(models: pd.DataFrame, all_scores: dict) -> pd.DataFrame:
    """Return the AUC of each score on the labeled models of each set, also for each trigger type and variant."""
    y, kind, variant = pair_labels(models)
    labeled = (models["labeled"] == "True").to_numpy()
    rows = []
    for name in models["set"].unique():
        m = labeled & (models["set"] == name).to_numpy()
        yy, kk, vv = y[m], kind[m], variant[m]
        for score_name, s in all_scores.items():
            ss = s[m]
            row = {"set": name, "score": score_name, "auc": auc(yy, ss), "positives": int(yy.sum())}
            for group, labels in [(f"type_{k}", kk == k) for k in sorted(set(kk[yy == 1]))] + \
                                 [(f"variant_{v}", vv == v) for v in sorted(set(vv[yy == 1]))]:
                keep = (yy == 0) | labels
                row[group] = auc(yy[keep], ss[keep])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["set", "auc"], ascending=[True, False])


def submission(models: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """Return the submission table (id, c00..c19) of the unlabeled task models."""
    m = ((models["set"] == "real") & (models["labeled"] != "True")).to_numpy()
    table = pd.DataFrame(np.nan_to_num(s[m], nan=0.0, posinf=1e3, neginf=-1e3), columns=[f"c{c:02d}" for c in range(N_CLASSES)])
    table.insert(0, "id", models.loc[m, "id"].to_numpy())
    return table


def run(folder, submit: list[str], source) -> None:
    """Write report.csv and the submission files of the selected scores. The features come from the folder source (the
    output of the job scan)."""
    folder.mkdir(parents=True, exist_ok=True)
    models, feat = load(source)
    all_scores = scores(models, feat)
    table = report(models, all_scores)
    table.to_csv(folder / "report.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for name, part in table.groupby("set"):
            print(part.dropna(axis=1, how="all").head(25).to_string(index=False), flush=True)
    (folder / "submissions").mkdir(exist_ok=True)
    for name in submit:
        submission(models, all_scores[name]).to_csv(folder / "submissions" / f"scan_{name}.csv", index=False)
