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
