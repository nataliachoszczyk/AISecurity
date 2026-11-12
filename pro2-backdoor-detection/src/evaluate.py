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


def nc_anomaly(norm: np.ndarray) -> np.ndarray:
    """The anomaly index of Neural Cleanse: a small mask norm compared with the other classes of the same model (axis 1)."""
    median = np.median(norm, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(norm - median), axis=1, keepdims=True)
    return (median - norm) / (mad + 1e-6)


def scores(models: pd.DataFrame, feat: dict) -> dict[str, np.ndarray]:
    """Return candidate scores, each of shape (models, 20). A larger score means a larger suspicion."""
    sets = models["set"].to_numpy()
    families = [f for f in FAMILIES if f"{f}_margin" in feat]
    out = {}
    for i, name in enumerate(feat["stat_names"]):
        out[f"stat_{name}"] = robust_z(feat["stats"][:, :, i], sets)
    if "nc_norm" in feat:
        out["nc_anomaly"] = nc_anomaly(feat["nc_norm"])
        out["nc_norm_z"] = robust_z(-feat["nc_norm"], sets)
    if not families:
        return out
    for family in families:
        z = robust_z(feat[f"{family}_margin"], sets)
        # The mean over the source classes finds a backdoor for all classes. The max finds a targeted backdoor.
        out[f"{family}_mean"] = np.nanmean(z, axis=1)
        out[f"{family}_max"] = np.nanmax(z, axis=1)
        out[f"{family}_top3"] = np.sort(np.nan_to_num(z, nan=-np.inf), axis=1)[:, -3:].mean(axis=1)
        out[f"{family}_asr_mean"] = np.nanmean(robust_z(feat[f"{family}_asr"], sets), axis=1)
    out["patch_small_mean"] = np.nanmean(robust_z(-feat["patch_size"], sets), axis=1)
    for agg in ("mean", "max", "top3"):
        values = [out[f"{f}_{agg}"] for f in families]
        out[f"inv_{agg}_max"] = np.max(values, axis=0)
        out[f"inv_{agg}_sum"] = np.sum(values, axis=0)
        # The second largest family: a backdoor usually shows in more than one family, an outlier in only one.
        out[f"inv_{agg}_2nd"] = np.sort(values, axis=0)[-2]
    # The mean rank of the max scores of the families, in each set.
    for name in np.unique(sets):
        rows = sets == name
        ranks = [pd.Series(out[f"{f}_max"][rows].ravel()).rank(pct=True).to_numpy() for f in families]
        out.setdefault("inv_max_rank", np.full(out["inv_max_max"].shape, np.nan))[rows] = np.mean(ranks, axis=0).reshape(-1, N_CLASSES)
    # A clean-label backdoor can give the opposite sign: the model uses the trigger as a feature of the target class,
    # so it is less sure of the clean images of this class. The mix adds this low confidence to the inversion score.
    low = np.clip(-out["stat_clean_logit"], 0, None)
    for w in (0.25, 0.5, 1.0):
        out[f"mix_low_{w}"] = out["inv_top3_2nd"] + w * low
        out[f"mix_abs_{w}"] = out["inv_top3_2nd"] + w * np.abs(out["stat_clean_logit"])
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


def combiner():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))


def matrix(all_scores: dict, names: list[str], rows: np.ndarray) -> np.ndarray:
    """Return the features of the pairs of the selected models: (models * 20, features)."""
    return np.nan_to_num(np.stack([all_scores[n][rows].ravel() for n in names], axis=1), nan=0.0, posinf=50, neginf=-50)


def leave_one_type_out(models: pd.DataFrame, feat: dict, name: str = "a", names: list[str] | None = None) -> pd.DataFrame:
    """Cross-validation of the combiner: for each trigger type, fit on the models without this type and score the
    models with this type and a share of the clean models. Return the predictions of all folds together."""
    all_scores = scores(models, feat)
    names = names or sorted(all_scores)
    y, kind, _ = pair_labels(models)
    labeled = ((models["set"] == name) & (models["labeled"] == "True")).to_numpy()
    idx = np.flatnonzero(labeled)
    model_types = [set(kind[i][y[i] == 1]) for i in idx]
    types = sorted(set().union(*model_types))
    clean = [i for i, t in zip(idx, model_types) if not t]
    out = []
    for f, t in enumerate(types):
        test = [i for i, ts in zip(idx, model_types) if t in ts] + clean[f::len(types)]
        train = [i for i, ts in zip(idx, model_types) if t not in ts and i not in test]
        model = combiner().fit(matrix(all_scores, names, np.array(train)), y[train].ravel())
        pred = model.decision_function(matrix(all_scores, names, np.array(test))).reshape(len(test), N_CLASSES)
        # Positives of other types in the test models are not a part of this fold.
        keep = (y[test] == 0) | (kind[test] == t)
        out.append(pd.DataFrame({"fold": t, "y": y[test][keep], "score": pred[keep]}))
    return pd.concat(out, ignore_index=True)


def loto_report(models: pd.DataFrame, feat: dict, name: str = "a", names: list[str] | None = None) -> pd.DataFrame:
    pred = leave_one_type_out(models, feat, name, names)
    rows = [{"fold": "all", "auc": roc_auc_score(pred["y"], pred["score"]), "positives": int(pred["y"].sum())}]
    for t, part in pred.groupby("fold"):
        rows.append({"fold": t, "auc": roc_auc_score(part["y"], part["score"]), "positives": int(part["y"].sum())})
    return pd.DataFrame(rows)


# A small set of scores for the combiner. Fewer inputs overfit less to the trigger types of set A.
COMPACT = ["inv_max_max", "inv_top3_max", "inv_mean_max", "inv_max_2nd", "inv_top3_2nd", "stat_clean_prob", "stat_clean_logit"]


def loto_all(models: pd.DataFrame, feat: dict, name: str = "a") -> pd.DataFrame:
    """The leave-one-type-out AUC of the combiner with all scores and with the compact set, in one table."""
    full = loto_report(models, feat, name).assign(inputs="all")
    compact = loto_report(models, feat, name, COMPACT).assign(inputs="compact")
    return pd.concat([full, compact], ignore_index=True)


def submission(models: pd.DataFrame, feat: dict, score_name: str) -> pd.DataFrame:
    """Return the submission table of the unlabeled task models: id and the columns c00 to c19."""
    rows = ((models["set"] == "real") & (models["labeled"] == "False")).to_numpy()
    values = np.nan_to_num(scores(models, feat)[score_name][rows], nan=0.0)
    table = pd.DataFrame(values, columns=[f"c{c:02d}" for c in range(N_CLASSES)])
    table.insert(0, "id", models["id"].to_numpy()[rows])
    return table


def submission(models: pd.DataFrame, feat: dict, score_name: str) -> pd.DataFrame:
    """Return the submission table (id, c00..c19) with one score for the unlabeled task models."""
    rows = ((models["set"] == "real") & (models["labeled"] != "True")).to_numpy()
    s = np.nan_to_num(scores(models, feat)[score_name][rows], nan=0.0, posinf=50, neginf=-50)
    table = pd.DataFrame(s, columns=[f"c{c:02d}" for c in range(N_CLASSES)])
    table.insert(0, "id", models.loc[rows, "id"].to_numpy())
    return table
