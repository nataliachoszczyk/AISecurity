"""MIA features of each record from the per-token values of src/logprobs.py, and the scores that combine them.

A larger feature value means a stronger suspicion that the record is a member.
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SET_TYPES = ["writer", "research group", "podcast", "photographer"]


def find_passes(base: str = "/kaggle/input") -> Path:
    matches = sorted({p.parent for p in Path(base).rglob("bumas_text.npz")})
    if len(matches) != 1:
        raise FileNotFoundError(f"need exactly one folder with bumas_text.npz below {base}, found {matches}")
    return matches[0]


def load_pass(folder: Path, name: str) -> tuple[pd.Index, list[dict]]:
    """Return the record ids and, for each record, its per-token arrays."""
    z = np.load(folder / f"{name}.npz", allow_pickle=True)  # the ids are strings; the files come from our own job
    off = z["offsets"]
    keys = [k for k in z.files if k not in ("ids", "offsets")]
    arrays = {k: z[k] for k in keys}
    rows = [{k: arrays[k][a:b] for k in keys} for a, b in zip(off[:-1], off[1:])]
    return pd.Index(z["ids"]), rows


def lowest_mean(x: np.ndarray, k: float) -> float:
    """Mean of the lowest fraction k of the values (Min-K%)."""
    return float(np.sort(x)[:max(1, int(len(x) * k))].mean())


def highest_mean(x: np.ndarray, k: float) -> float:
    return -lowest_mean(-x, k)


def z_tokens(r: dict) -> np.ndarray:
    """Min-K%++ value of each token: log p of the token, normalized with the distribution of the vocabulary."""
    return (r["lp"] - r["mu"]) / np.maximum(r["sigma"], 1e-6)


QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
# Bins of the reference log p. Training changes hard (rare) tokens more than easy tokens.
DIFFICULTY_BINS = [-np.inf, -8, -4, -2, -1, -0.25, 0]


def diff_features(d: np.ndarray, lp_ref: np.ndarray, prefix: str) -> dict:
    """The distribution of the per-token gain d = log p(target) - log p(reference) of one record."""
    out = {f"{prefix}_q{int(q * 100):02d}": v for q, v in zip(QUANTILES, np.quantile(d, QUANTILES))}
    out[f"{prefix}_big_gain"] = float((d > 1).mean())
    out[f"{prefix}_big_loss"] = float((d < -1).mean())
    which = np.digitize(lp_ref, DIFFICULTY_BINS[1:-1])
    for b in range(len(DIFFICULTY_BINS) - 1):
        sel = which == b
        # A bin without tokens gets 0: no gain is known.
        out[f"{prefix}_bin{b}"] = float(d[sel].mean()) if sel.any() else 0.0
    return out


def text_features(folder: Path, records: pd.DataFrame) -> pd.DataFrame:
    """Features of the text records. T: BUMAS-mini, R: the reference model Qwen3-0.6B-Base."""
    ids, target = load_pass(folder, "bumas_text")
    ids_r, ref = load_pass(folder, "qwen_text")
    assert ids.equals(ids_r)
    content = records.set_index("id").loc[ids, "content"]
    rows = []
    for t, r, text in zip(target, ref, content):
        d = t["lp"] - r["lp"]
        zt, zr = z_tokens(t), z_tokens(r)
        loss_t, loss_r = -t["lp"].mean(), -r["lp"].mean()
        rows.append({
            "loss": -loss_t,
            "zlib": -loss_t * len(t["lp"]) / len(zlib.compress(text.encode())),
            "mink20": lowest_mean(t["lp"], 0.2),
            "minkpp20": lowest_mean(zt, 0.2),
            "ref": d.mean(),
            "ref_ratio": -loss_t / loss_r,
            "ref_median": float(np.median(d)),
            "ref_top20": highest_mean(d, 0.2),
            "ref_low20": lowest_mean(d, 0.2),
            "ref_gain_frac": float((d > 0).mean()),
            "ref_minkpp": (zt - zr).mean(),
            "n_tokens": len(t["lp"]),
            **diff_features(d, r["lp"], "d"),
        })
    return pd.DataFrame(rows, index=ids)


def photo_features(folder: Path) -> pd.DataFrame:
    """Features of the photo records.

    O: BUMAS-mini with its own photo, S: with the photo of a different record, C: the LLM of BUMAS-mini without a photo,
    Q: Qwen3-0.6B-Base without a photo, N02 and N10: with a noised own photo.
    """
    passes = {}
    ids = None
    for key, name in [("O", "bumas_photo_own"), ("S", "bumas_photo_swap"), ("C", "bumas_caption_text"),
                      ("Q", "qwen_caption_text"), ("N02", "bumas_photo_noise02"), ("N10", "bumas_photo_noise10")]:
        pass_ids, passes[key] = load_pass(folder, name)
        assert ids is None or ids.equals(pass_ids)
        ids = pass_ids
    rows = []
    for i in range(len(ids)):
        o, s, c, q = (passes[k][i] for k in "OSCQ")
        n02, n10 = passes["N02"][i], passes["N10"][i]
        rows.append({
            "loss": o["lp"].mean(),
            "minkpp20": lowest_mean(z_tokens(o), 0.2),
            "ref_img": (o["lp"] - q["lp"]).mean(),
            "ref_txt": (c["lp"] - q["lp"]).mean(),
            "txt_minkpp": (z_tokens(c) - z_tokens(q)).mean(),
            "img_gain": (o["lp"] - s["lp"]).mean(),
            "img_gain_top20": highest_mean(o["lp"] - s["lp"], 0.2),
            "img_vs_txt": (o["lp"] - c["lp"]).mean(),
            "noise02_drop": (o["lp"] - n02["lp"]).mean(),
            "noise10_drop": (o["lp"] - n10["lp"]).mean(),
            "noise02_kl": -n02["kl"].mean(),
            "noise10_kl": -n10["kl"].mean(),
            "n_tokens": len(o["lp"]),
            **diff_features(o["lp"] - q["lp"], q["lp"], "do"),
            **diff_features(c["lp"] - q["lp"], q["lp"], "dc"),
        })
    return pd.DataFrame(rows, index=ids)


def all_features(folder: Path, records: pd.DataFrame) -> pd.DataFrame:
    """One table for all records, in the order of `records`. Columns of the other kind are empty."""
    feats = pd.concat([text_features(folder, records), photo_features(folder)])
    return feats.reindex(records["id"]).set_axis(records.index)


def owner_means(records: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Mean of a value over all records of the owner (known and test). Single records keep their own value.

    It uses no label, so it is the same in the validation and for test.
    """
    return values.groupby(records["source"]).transform("mean")


def owner_prior(fit: pd.DataFrame, records: pd.DataFrame) -> pd.Series:
    """Fraction of known members of the owner in the fit part. 0 for an owner without fit records."""
    rate = fit.groupby("source")["member"].mean()
    return records["source"].map(rate).fillna(0.0)


def logistic_score(columns, C: float = 1.0):
    """A score function for src/validation.py: one logistic regression for each source type.

    The fit and the evaluate parts must have the feature columns. The output is the log-odds, so all source types
    share one scale.
    """
    def score(fit: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        out = pd.Series(0.0, index=evaluate.index)
        for source_type, part in evaluate.groupby("source_type"):
            cols = columns(source_type) if callable(columns) else columns
            train = fit[fit["source_type"] == source_type]
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
            model.fit(train[cols], train["member"])
            out.loc[part.index] = model.decision_function(part[cols])
        return out.to_numpy()
    return score


TEXT_COLUMNS = ["loss", "zlib", "mink20", "minkpp20", "ref", "ref_ratio", "ref_median", "ref_top20", "ref_low20",
                "ref_gain_frac", "ref_minkpp"]
PHOTO_COLUMNS = ["loss", "minkpp20", "ref_img", "ref_txt", "txt_minkpp", "img_gain", "img_gain_top20", "img_vs_txt",
                 "noise02_drop", "noise10_drop", "noise02_kl", "noise10_kl"]


def columns_of(source_type: str) -> list[str]:
    return PHOTO_COLUMNS if source_type in ("photographer", "web photo") else TEXT_COLUMNS


def single_features(known: pd.DataFrame) -> pd.DataFrame:
    """TPR@5%FPR and AUC of each feature alone, inside each source type. No fit, so no fold is necessary."""
    from .metrics import auc, tpr_at_5fpr
    rows = []
    for source_type, part in known.groupby("source_type"):
        labels = part["member"].astype(int)
        if labels.nunique() < 2:
            continue
        for col in columns_of(source_type):
            values = part[col].to_numpy()
            rows.append({"source_type": source_type, "feature": col, "AUC": auc(values, labels),
                         "TPR@5%FPR": tpr_at_5fpr(values.tolist(), labels.tolist())})
    return pd.DataFrame(rows)


OWNER_COLUMNS = {"text": ["ref", "ref_minkpp"], "photo": ["ref_img", "img_gain"]}


def add_owner_columns(records: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Add owner_<feature>: the mean of the feature over all records of the owner."""
    feats = feats.copy()
    for kind, cols in OWNER_COLUMNS.items():
        is_kind = records["kind"] == kind
        for col in cols:
            feats.loc[is_kind, f"owner_{col}"] = owner_means(records[is_kind], feats.loc[is_kind, col])
    return feats


def victim_flag(fit: pd.DataFrame, part: pd.DataFrame) -> pd.Series:
    """1 if the owner has a known member in the fit part. A fit record does not count its own label."""
    members = fit.groupby("source")["member"].sum()
    flag = part["source"].map(members).fillna(0)
    if part.index.isin(fit.index).all():
        flag = flag - part["member"]
    return (flag > 0).astype(float)


def combined_score(record_columns, owner: bool, prior: bool, C: float = 1.0):
    """Logistic regression for each source type on the record features, plus owner features for the set types."""
    def columns(source_type):
        cols = list(record_columns(source_type))
        if source_type in SET_TYPES:
            kind = "photo" if source_type == "photographer" else "text"
            cols += [f"owner_{c}" for c in OWNER_COLUMNS[kind]] if owner else []
            cols += ["victim"] if prior else []
        return cols

    inner = logistic_score(columns, C)

    def score(fit, evaluate):
        return inner(fit.assign(victim=victim_flag(fit, fit)), evaluate.assign(victim=victim_flag(fit, evaluate)))
    return score


def owner_check(data: pd.DataFrame) -> pd.DataFrame:
    """Mean of the main feature over the known and the test records of each owner group.

    Groups: victims (known members), owners with as many known records as a victim but no member, and the other owners.
    If a group has members only in test, its test mean is higher than its known mean.
    """
    sets = data[data["source_type"].isin(SET_TYPES)].copy()
    sets["main"] = sets["ref"].where(sets["kind"] == "text", sets["ref_img"])
    known = sets[sets["split"] == "known"]
    members = known.groupby("source")["member"].sum()
    n_known = known.groupby("source").size()
    full = n_known.groupby(known.groupby("source")["source_type"].first()).transform("max")
    group = pd.Series("other", index=members.index).mask(n_known == full, "full, no member").mask(members > 0, "victim")
    sets["group"] = sets["source"].map(group)
    sets["main_z"] = sets.groupby("source_type")["main"].transform(lambda v: (v - v.mean()) / v.std())
    per_owner = sets.groupby(["source_type", "group", "source", "split"])["main_z"].mean().unstack("split")
    return per_owner.groupby(["source_type", "group"]).agg(["mean", "std", "min", "max"]).round(3).reset_index()


def explore(out_dir: str) -> None:
    """Write the features, the score of each feature alone and the validation of the combined scores."""
    from .data import find_root, load_records
    from .validation import cross_validate, summarize
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(find_root())
    feats = add_owner_columns(records, all_features(find_passes(), records))
    feats.assign(id=records["id"]).dropna(how="all", axis=1).to_csv(out / "features.csv", index=False)
    data = pd.concat([records, feats.drop(columns="n_tokens")], axis=1)
    known = data[(data["split"] == "known") & feats["loss"].notna()].copy()
    known["member"] = known["member"].astype(int)
    single = single_features(known)
    single.to_csv(out / "single_features.csv", index=False)
    print(single.sort_values(["source_type", "TPR@5%FPR"], ascending=[True, False]).to_string(), flush=True)
    owners = owner_check(data)
    owners.to_csv(out / "owner_check.csv", index=False)
    print(owners.to_string(), flush=True)
    if known.groupby("source_type")["member"].sum().min() < 10:
        print("too few members for the validation (smoke test)")
        return
    variants = {
        "record": combined_score(columns_of, owner=False, prior=False),
        "record + victim": combined_score(columns_of, owner=False, prior=True),
        "record + owner": combined_score(columns_of, owner=True, prior=False),
        "record + owner + victim": combined_score(columns_of, owner=True, prior=True),
    }
    results = pd.concat([cross_validate(known, fn).assign(variant=name) for name, fn in variants.items()])
    results.to_csv(out / "cv_folds.csv", index=False)
    table = pd.concat([summarize(r).assign(variant=name) for name, r in results.groupby("variant", sort=False)])
    table.to_csv(out / "cv.csv", index=False)
    print(table.to_string(), flush=True)


def diff_names(prefix: str) -> list[str]:
    return list(diff_features(np.zeros(4), np.zeros(4), prefix))


TEXT_RICH = TEXT_COLUMNS + diff_names("d")
PHOTO_RICH = PHOTO_COLUMNS + diff_names("do") + diff_names("dc")


def rich_columns_of(source_type: str) -> list[str]:
    return PHOTO_RICH if source_type in ("photographer", "web photo") else TEXT_RICH


def boosting_score(columns):
    """Gradient boosting for each source type. Only a check of the headroom over the logistic regression."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    def score(fit, evaluate):
        out = pd.Series(0.0, index=evaluate.index)
        for source_type, part in evaluate.groupby("source_type"):
            cols = columns(source_type)
            train = fit[fit["source_type"] == source_type]
            model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
            model.fit(train[cols], train["member"])
            out.loc[part.index] = model.decision_function(part[cols])
        return out.to_numpy()
    return score


def owner_groups(data: pd.DataFrame) -> pd.Series:
    """Group of each record: single, victim, full (known records as a victim, no member) or other."""
    known = data[data["split"] == "known"]
    members = known.groupby("source")["member"].sum()
    n_known = known.groupby("source").size()
    full = n_known.groupby(known.groupby("source")["source_type"].first()).transform("max")
    group = pd.Series("other", index=members.index).mask(n_known == full, "full").mask(members > 0, "victim")
    return data["source"].map(group).where(data["source_type"].isin(SET_TYPES), "single")


def test_estimates(data: pd.DataFrame, C: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expected members in test from a logistic regression on all known records, for each source type and group.

    The second table counts the owners by the mean member probability of their test records.
    """
    data = data.assign(group=owner_groups(data), p=np.nan)
    for source_type, part in data.groupby("source_type"):
        cols = rich_columns_of(source_type)
        train = part[part["split"] == "known"]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        model.fit(train[cols], train["member"].astype(int))
        data.loc[part.index, "p"] = model.predict_proba(part[cols])[:, 1]
    test = data[data["split"] == "test"]
    by_group = test.groupby(["source_type", "group"]).agg(records=("p", "size"), expected_members=("p", "sum"),
                                                          above_half=("p", lambda p: int((p > 0.5).sum())))
    owners = test[test["group"] != "single"].groupby(["source_type", "group", "source"])["p"].mean()
    bins = pd.cut(owners, [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    by_owner = owners.groupby([owners.index.get_level_values(0), owners.index.get_level_values(1), bins],
                              observed=True).size().unstack(fill_value=0)
    return by_group.round(1).reset_index(), by_owner.reset_index()


def explore_v2(out_dir: str) -> None:
    """Rich features of the token gains, the validation of the record scores, and the estimates for test."""
    from .data import find_root, load_records
    from .validation import cross_validate, summarize
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(find_root())
    feats = all_features(find_passes(), records)
    feats.assign(id=records["id"]).to_csv(out / "features.csv", index=False)
    data = pd.concat([records, feats.drop(columns="n_tokens")], axis=1)
    known = data[data["split"] == "known"].copy()
    known["member"] = known["member"].astype(int)
    pd.set_option("display.width", 200)
    for name, table in zip(["test_by_group", "test_by_owner"], test_estimates(data)):
        table.to_csv(out / f"{name}.csv", index=False)
        print(table.to_string(), flush=True)
    variants = {
        "record v1": logistic_score(columns_of, C=1.0),
        "record rich C=1": logistic_score(rich_columns_of, C=1.0),
        "record rich C=0.1": logistic_score(rich_columns_of, C=0.1),
        "record rich boosting": boosting_score(rich_columns_of),
    }
    results = pd.concat([cross_validate(known, fn).assign(variant=name) for name, fn in variants.items()])
    results.to_csv(out / "cv_folds.csv", index=False)
    table = pd.concat([summarize(r).assign(variant=name) for name, r in results.groupby("variant", sort=False)])
    table.to_csv(out / "cv.csv", index=False)
    print(table.round(3).to_string(), flush=True)


def record_logodds(data: pd.DataFrame, fit_index, columns, C: float = 1.0) -> pd.Series:
    """Log-odds of each record from one logistic regression for each source type, fit on the records fit_index."""
    out = pd.Series(np.nan, index=data.index)
    fit = data.loc[fit_index]
    for source_type, part in data.groupby("source_type"):
        cols = columns(source_type)
        train = fit[fit["source_type"] == source_type]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        model.fit(train[cols], train["member"].astype(int))
        out.loc[part.index] = model.decision_function(part[cols])
    return out


def owner_mean(data: pd.DataFrame, logodds: pd.Series, split: str) -> pd.Series:
    """Dataset inference: the mean log-odds of the records of the owner in one split. A record does not count itself.

    For test, the split is "test": test is the suspect part of a set. Known can be a part that the company did not use.
    For the validation on known, the split is "known", so that each owner gives evidence from a similar number of records.
    The value is 0 for single records.
    """
    in_split = (data["split"] == split).astype(float)
    total = (logodds * in_split).groupby(data["source"]).transform("sum") - logodds * in_split
    count = in_split.groupby(data["source"]).transform("sum") - in_split
    mean = (total / count.replace(0, np.nan)).fillna(0.0)
    return mean.where(data["source_type"].isin(SET_TYPES), 0.0)


def di_cross_validate(data: pd.DataFrame, columns, weights=(0.0, 0.5, 1.0, 2.0), C: float = 1.0, seeds=range(5)) -> pd.DataFrame:
    """Validation of record log-odds + w * owner_mean on all known records. The owner mean uses the known records."""
    from .metrics import auc, tpr_at_5fpr
    from .validation import make_folds
    known = data[data["split"] == "known"]
    rows = []
    for seed in seeds:
        fold = pd.Series(make_folds(known, 2, seed), index=known.index)
        for f in range(2):
            logodds = record_logodds(data, fold.index[fold != f], columns, C)
            owner = owner_mean(data, logodds, "known")
            evaluate = fold.index[fold == f]
            for w in weights:
                score = logodds + w * owner
                for name, idx in [("all", evaluate)] + list(data.loc[evaluate].groupby("source_type").groups.items()):
                    labels = data.loc[idx, "member"].astype(int)
                    rows.append({"seed": seed, "fold": f, "w": w, "records": name,
                                 "TPR@5%FPR": tpr_at_5fpr(score.loc[idx].tolist(), labels.tolist()),
                                 "AUC": auc(score.loc[idx], labels)})
    results = pd.DataFrame(rows)
    return results.groupby(["w", "records"], sort=False)[["TPR@5%FPR", "AUC"]].agg(["mean", "std"]).round(3)


def submission(data: pd.DataFrame, columns, w: float, C: float = 1.0) -> pd.DataFrame:
    """Scores of the test records: fit on all known records."""
    known_index = data.index[data["split"] == "known"]
    logodds = record_logodds(data, known_index, columns, C)
    score = logodds + w * owner_mean(data, logodds, "test")
    test = data["split"] == "test"
    return pd.DataFrame({"id": data.loc[test, "id"], "score": score[test]})


def explore_v3(out_dir: str) -> None:
    """Validation of the dataset inference score for each weight w, and one submission file for each w."""
    from .data import find_root, load_records
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(find_root())
    data = pd.concat([records, all_features(find_passes(), records)], axis=1)
    pd.set_option("display.width", 200)
    table = di_cross_validate(data, columns_of)
    table.to_csv(out / "di_cv.csv")
    print(table.to_string(), flush=True)
    for w in (0.0, 0.5, 1.0, 2.0):
        submission(data, columns_of, w).to_csv(out / f"submission_w{w}.csv", index=False)


def two_stage(data: pd.DataFrame, fit_index, columns, owner_split: str, C: float = 1.0, seed: int = 0) -> pd.Series:
    """Calibrated log-odds of each record in two stages.

    Stage 1: record log-odds. The fit records get out-of-fold values (2 inner folds), so that stage 2 sees honest values.
    Stage 2: for each set type, a logistic regression on [record log-odds, owner mean]. The owner mean of the fit records
    uses the known records. The owner mean of the other records uses the split owner_split ("known" in the validation,
    "test" for the submission). Single records keep the stage-1 log-odds.
    """
    from .validation import make_folds
    fit_index = pd.Index(fit_index)
    logodds = record_logodds(data, fit_index, columns, C)
    inner = make_folds(data.loc[fit_index], 2, seed + 100)
    for f in range(2):
        part = record_logodds(data, fit_index[inner != f], columns, C)
        logodds.loc[fit_index[inner == f]] = part.loc[fit_index[inner == f]]
    owner = owner_mean(data, logodds, owner_split)
    owner.loc[fit_index] = owner_mean(data, logodds, "known").loc[fit_index]
    out = logodds.copy()
    for source_type in SET_TYPES:
        is_type = data["source_type"] == source_type
        train = fit_index[is_type.loc[fit_index].to_numpy()]
        if train.empty:
            continue
        x = pd.DataFrame({"record": logodds, "owner": owner})
        model = LogisticRegression(max_iter=1000).fit(x.loc[train], data.loc[train, "member"].astype(int))
        out.loc[is_type] = model.decision_function(x.loc[is_type])
    return out


def two_stage_cross_validate(data: pd.DataFrame, columns, C: float = 1.0, seeds=range(5)) -> pd.DataFrame:
    from .metrics import auc, tpr_at_5fpr
    from .validation import make_folds
    known = data[data["split"] == "known"]
    rows = []
    for seed in seeds:
        fold = pd.Series(make_folds(known, 2, seed), index=known.index)
        for f in range(2):
            evaluate = fold.index[fold == f]
            for variant, score in [("record", record_logodds(data, fold.index[fold != f], columns, C)),
                                   ("two stage", two_stage(data, fold.index[fold != f], columns, "known", C, seed))]:
                for name, idx in [("all", evaluate)] + list(data.loc[evaluate].groupby("source_type").groups.items()):
                    labels = data.loc[idx, "member"].astype(int)
                    rows.append({"variant": variant, "records": name, "seed": seed, "fold": f,
                                 "TPR@5%FPR": tpr_at_5fpr(score.loc[idx].tolist(), labels.tolist()),
                                 "AUC": auc(score.loc[idx], labels)})
    results = pd.DataFrame(rows)
    return results.groupby(["variant", "records"], sort=False)[["TPR@5%FPR", "AUC"]].agg(["mean", "std"]).round(3)


def explore_v5(out_dir: str) -> None:
    """Validation of the two-stage score, and its submission file."""
    from .data import find_root, load_records
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(find_root())
    data = pd.concat([records, all_features(find_passes(), records)], axis=1)
    pd.set_option("display.width", 200)
    table = two_stage_cross_validate(data, columns_of)
    table.to_csv(out / "two_stage_cv.csv")
    print(table.to_string(), flush=True)
    score = two_stage(data, data.index[data["split"] == "known"], columns_of, "test")
    test = data["split"] == "test"
    pd.DataFrame({"id": data.loc[test, "id"], "score": score[test]}).to_csv(out / "submission.csv", index=False)
    print(data.loc[test].assign(p=1 / (1 + np.exp(-score[test]))).groupby("source_type")["p"].agg(["mean", "sum"]).round(3))


def find_folder(file_name: str, base: str = "/kaggle/input") -> Path:
    matches = sorted({p.parent for p in Path(base).rglob(file_name)})
    if len(matches) != 1:
        raise FileNotFoundError(f"need exactly one folder with {file_name} below {base}, found {matches}")
    return matches[0]


def lira_features(lp_t: np.ndarray, lp_in: np.ndarray, lp_out: np.ndarray, prefix: str) -> dict:
    """LiRA with one IN and one OUT shadow model for each token.

    g: gain of the target over the OUT model. m: gain of one training exposure (IN over OUT).
    With equal variances, the log-likelihood ratio of IN against OUT for one token is m * (2g - m) / (2 var).
    """
    g, m = lp_t - lp_out, lp_in - lp_out
    m_pos = np.maximum(m, 0)
    return {
        f"{prefix}_gain": g.mean(),
        f"{prefix}_llr": (m * (2 * g - m)).mean(),
        f"{prefix}_wgain": float((g * m_pos).sum() / max(m_pos.sum(), 1e-6)),
        f"{prefix}_slope": float((g * m).sum() / max((m * m).sum(), 1e-6)),
        f"{prefix}_corr": float(np.corrcoef(g, m)[0, 1]) if len(g) > 2 and g.std() > 0 and m.std() > 0 else 0.0,
    }


def domain_features(lp_t: np.ndarray, lp_d: np.ndarray, prefix: str) -> dict:
    """Gain of the target over the reference model for the domain."""
    d = lp_t - lp_d
    return {f"{prefix}": d.mean(), f"{prefix}_low20": lowest_mean(d, 0.2), f"{prefix}_gain_frac": float((d > 0).mean())}


def v6_features(records: pd.DataFrame) -> pd.DataFrame:
    """Features from the domain reference models (job refmodel) and the shadow models (job shadow)."""
    from .validation import make_folds
    base, ref_dir, shadow_dir = find_passes(), find_folder("ref0_text.npz"), find_folder("shadow0_text.npz")
    by_id = records.set_index("id")
    known = records[records["split"] == "known"]
    ref_fold = pd.Series(make_folds(known, 2, seed=0), index=known["id"])  # the folds of the job refmodel
    half = pd.read_csv(shadow_dir / "shadow_halves.csv").set_index("id")["half"]
    tables = []
    for kind, target_names in [("text", ["bumas_text"]), ("caption", ["bumas_caption_text", "bumas_photo_own"])]:
        suffix = "text" if kind == "text" else "caption_text"
        ids, ref = zip(*[load_pass(ref_dir, f"ref{f}_{suffix}") for f in (0, 1)])
        _, shadow = zip(*[load_pass(shadow_dir, f"shadow{h}_{suffix}") for h in (0, 1)])
        targets = {name: load_pass(base, name) for name in target_names}
        for name, (t_ids, _) in targets.items():
            assert t_ids.equals(ids[0]), name
        rows = []
        for i, rid in enumerate(ids[0]):
            row = {}
            # A known record gets the reference model of the other fold. A test record gets the mean of both.
            if by_id.loc[rid, "split"] == "known":
                lp_d = ref[1 - ref_fold[rid]][i]["lp"]
            else:
                lp_d = (ref[0][i]["lp"] + ref[1][i]["lp"]) / 2
            h = half[rid]
            lp_in, lp_out = shadow[h][i]["lp"], shadow[1 - h][i]["lp"]
            for name, (_, t_rows) in targets.items():
                tag = "own" if name == "bumas_photo_own" else "txt"
                prefix = "dom" if kind == "text" else f"dom_{tag}"
                lira = "lira" if kind == "text" else f"lira_{tag}"
                row.update(domain_features(t_rows[i]["lp"], lp_d, prefix))
                row.update(lira_features(t_rows[i]["lp"], lp_in, lp_out, lira))
            row["shadow_m"] = (lp_in - lp_out).mean()
            rows.append(row)
        tables.append(pd.DataFrame(rows, index=ids[0]))
    return pd.concat(tables).reindex(records["id"]).set_axis(records.index)


V6_TEXT = ["dom", "dom_low20", "dom_gain_frac"] + [f"lira_{s}" for s in ("gain", "llr", "wgain", "slope", "corr")] + ["shadow_m"]
V6_PHOTO = [f"dom_{t}{s}" for t in ("txt", "own") for s in ("", "_low20", "_gain_frac")] + \
           [f"lira_{t}_{s}" for t in ("txt", "own") for s in ("gain", "llr", "wgain", "slope", "corr")] + ["shadow_m"]


def v6_columns_of(source_type: str) -> list[str]:
    return PHOTO_COLUMNS + V6_PHOTO if source_type in ("photographer", "web photo") else TEXT_COLUMNS + V6_TEXT


def explore_v6(out_dir: str) -> None:
    """The new features alone, the validation of the record score and of the two-stage score, and the submission file."""
    from .data import find_root, load_records
    from .metrics import auc, tpr_at_5fpr
    from .validation import cross_validate, summarize
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    records = load_records(find_root())
    data = pd.concat([records, all_features(find_passes(), records), v6_features(records)], axis=1)
    data.drop(columns=["text", "caption", "title", "content", "image", "image_path"], errors="ignore").to_csv(
        out / "features_v6.csv", index=False)
    known = data[data["split"] == "known"].copy()
    known["member"] = known["member"].astype(int)
    rows = []
    for source_type, part in known.groupby("source_type"):
        cols = V6_PHOTO if source_type in ("photographer", "web photo") else V6_TEXT
        for col in cols:
            rows.append({"source_type": source_type, "feature": col, "AUC": auc(part[col], part["member"]),
                         "TPR@5%FPR": tpr_at_5fpr(part[col].tolist(), part["member"].tolist())})
    single = pd.DataFrame(rows).round(3)
    single.to_csv(out / "single_v6.csv", index=False)
    print(single.to_string(), flush=True)
    variants = {"record v1": logistic_score(columns_of), "record v6": logistic_score(v6_columns_of),
                "record v6 C=0.1": logistic_score(v6_columns_of, C=0.1)}
    results = pd.concat([cross_validate(known, fn).assign(variant=name) for name, fn in variants.items()])
    table = pd.concat([summarize(r).assign(variant=name) for name, r in results.groupby("variant", sort=False)])
    table.to_csv(out / "cv_v6.csv", index=False)
    print(table.round(3).to_string(), flush=True)
    two = two_stage_cross_validate(data, v6_columns_of)
    two.to_csv(out / "two_stage_v6.csv")
    print(two.to_string(), flush=True)
    score = two_stage(data, data.index[data["split"] == "known"], v6_columns_of, "test")
    test = data["split"] == "test"
    pd.DataFrame({"id": data.loc[test, "id"], "score": score[test]}).to_csv(out / "submission_v6.csv", index=False)
