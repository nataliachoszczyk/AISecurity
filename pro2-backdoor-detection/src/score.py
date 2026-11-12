"""Turn the scan features into one suspicion score for each (model, class) pair, and check the scores.

Each value is compared with the same value (same probe, same classes) in all models of the same set. The models have the
same base, data and finetune, so a backdoor class reacts differently from the same class in the other models. The score of
a class is the maximum over the probes of the mean reaction over the images ("maxmean").
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.scan import N_CLASSES

SEED = 0
OFF_DIAGONAL = ~np.eye(N_CLASSES, dtype=bool)


def load(folder) -> tuple[pd.DataFrame, dict]:
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
    """Return candidate scores of shape (models, 20). A larger score means a larger suspicion."""
    sets = models["set"].to_numpy()
    probe_names = feat["probes"]
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
            out[f"{measure}_{agg}"] = z.max(axis=1)
            out[f"{measure}_{agg}_top3"] = np.sort(z, axis=1)[:, -3:].mean(axis=1)
            # The same, but each class compared with the other classes of the same model and probe first.
            rel = z - np.median(z, axis=2, keepdims=True)
            out[f"{measure}_{agg}_rel"] = rel.max(axis=1)
        for family in sorted({p.split("_")[0] for p in probe_names}):
            pick = np.array([p.split("_")[0] == family for p in probe_names])
            out[f"{measure}_mean_{family}"] = mean[:, pick].max(axis=1)
    out["maxmean"] = np.maximum(out["dlogp_mean"], out["dlogp_max"] - 2)  # The max over sources is noisier.
    out["both"] = out["dlogp_mean"] + out["hit_mean"]
    return out


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


def unlabeled_check(models: pd.DataFrame, all_scores: dict) -> pd.DataFrame:
    """Count the outliers in the unlabeled task models. The task has approximately 100 poisoned models with 1 to 3
    backdoors (approximately 150 positive pairs). A good score shows about this number of clear outliers."""
    m = ((models["set"] == "real") & (models["labeled"] != "True")).to_numpy()
    rows = []
    for name, s in all_scores.items():
        ss = s[m]
        row = {"score": name}
        for t in (4, 6, 10, 20):
            flagged = ss > t
            row[f"pairs>{t}"] = int(flagged.sum())
            row[f"models>{t}"] = int(flagged.any(axis=1).sum())
        top2 = np.sort(ss, axis=1)
        # In a clean model all classes are similar. The gap shows how much the top class stands out.
        row["median_gap"] = float(np.median(top2[:, -1] - top2[:, -2]))
        rows.append(row)
    return pd.DataFrame(rows)


def submission(models: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """Return the submission table (id, c00..c19) of the unlabeled task models."""
    m = ((models["set"] == "real") & (models["labeled"] != "True")).to_numpy()
    table = pd.DataFrame(np.nan_to_num(s[m], nan=0.0, posinf=1e3, neginf=-1e3), columns=[f"c{c:02d}" for c in range(N_CLASSES)])
    table.insert(0, "id", models.loc[m, "id"].to_numpy())
    return table


def run(folder, submit: list[str]) -> None:
    """Write report.csv, unlabeled.csv and the submission files of the selected scores."""
    models, feat = load(folder)
    all_scores = scores(models, feat)
    table = report(models, all_scores)
    table.to_csv(folder / "report.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for name, part in table.groupby("set"):
            print(part.dropna(axis=1, how="all").head(25).to_string(index=False), flush=True)
        check = unlabeled_check(models, all_scores)
        check.to_csv(folder / "unlabeled.csv", index=False)
        print(check.to_string(index=False), flush=True)
    (folder / "submissions").mkdir(exist_ok=True)
    for name in submit:
        submission(models, all_scores[name]).to_csv(folder / "submissions" / f"scan_{name}.csv", index=False)
