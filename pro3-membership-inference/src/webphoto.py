"""Membership score of the web photo records: features, validation inside the type, and calibrated log-odds.

The functions use the passes of the earlier jobs (logprobs, refmodel, shadow, photo-shadow) and the passes of this module.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .attacks import (PHOTO_COLUMNS, PLIRA, V6_PHOTO, all_features, load_pass, find_passes, photo_shadow_features,
                      v6_features)
from .metrics import auc, tpr_at_5fpr
from .validation import make_folds

WEB = "web photo"
BASE = PHOTO_COLUMNS + V6_PHOTO  # the v8a features of the photo types


def photo_data(records: pd.DataFrame) -> pd.DataFrame:
    """All photo records with the features of attacks-v8a and of the photo shadow models."""
    data = pd.concat([records, all_features(find_passes(), records), v6_features(records, ("shadow",)),
                      photo_shadow_features(records)], axis=1)
    return data[data["kind"] == "photo"].copy()


def duplicate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Records with the same caption and a different photo.

    The target can learn the caption of a member also without its photo. Then the caption gets a high log p with the photo
    of a non-member duplicate too. The member has the highest own log p of its group. The `loss` column is the mean log p
    of the caption with the own photo, so the photos of a group compare directly.
    """
    key = data["caption"].fillna("").str.strip().str.lower()
    count = key.map(key.value_counts())
    total = data["loss"].groupby(key).transform("sum")
    others = ((total - data["loss"]) / (count - 1).replace(0, np.nan))
    return pd.DataFrame({"dup": (count > 1).astype(float),
                         "dup_gain": (data["loss"] - others).fillna(0.0)}, index=data.index)


def image_features(data: pd.DataFrame) -> pd.DataFrame:
    """Blind features of the image file. A difference between members and non-members here is a shift of the data."""
    from PIL import Image, ImageStat
    rows = []
    for path in data["image_path"]:
        im = Image.open(path)
        q = getattr(im, "quantization", None) or {0: [0]}
        stat = ImageStat.Stat(im.convert("RGB"))
        hsv = ImageStat.Stat(im.convert("HSV"))
        rows.append({"img_bytes": Path(path).stat().st_size, "img_q0": float(np.mean(q[0])),
                     "img_mean": float(np.mean(stat.mean)), "img_std": float(np.mean(stat.stddev)),
                     "img_sat": hsv.mean[1]})
    return pd.DataFrame(rows, index=data.index)


def lr_score(columns, C: float = 1.0, extra_fit=None):
    """Score function: one logistic regression on the columns. extra_fit adds labeled records of other types to the fit."""
    def score(fit, evaluate):
        train = fit if extra_fit is None else pd.concat([fit, extra_fit])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        model.fit(train[columns], train["member"])
        return model.decision_function(evaluate[columns])
    return score


def text_score(column: str, analyzer: str = "word", ngram=(1, 2), C: float = 1.0):
    """Blind score from the words (or characters) of a text column."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    def score(fit, evaluate):
        model = make_pipeline(TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2, sublinear_tf=True),
                              LogisticRegression(C=C, max_iter=3000))
        model.fit(fit[column].fillna(""), fit["member"])
        return model.decision_function(evaluate[column].fillna(""))
    return score


def zscore_columns(columns, bins: int = 5):
    """Wrap a score function: z-score each column inside bins of the caption length.

    The mean and the standard deviation come from the known non-members of the fit part (the reference distribution).
    """
    def wrap(inner):
        def score(fit, evaluate):
            edges = np.quantile(fit["n_tokens"], np.linspace(0, 1, bins + 1)[1:-1])
            fit, evaluate = fit.copy(), evaluate.copy()
            ref = fit[fit["member"] == 0]
            ref_bin = np.digitize(ref["n_tokens"], edges)
            for part in (fit, evaluate):
                which = np.digitize(part["n_tokens"], edges)
                for b in range(bins):
                    sel, rsel = which == b, ref_bin == b
                    mu, sd = ref.loc[rsel, columns].mean(), ref.loc[rsel, columns].std() + 1e-9
                    part.loc[sel, columns] = ((part.loc[sel, columns] - mu) / sd).to_numpy()
            return inner(fit, evaluate)
        return score
    return wrap


def cv(known: pd.DataFrame, score_fn, seeds=range(5), n_folds: int = 2) -> dict:
    """TPR@5%FPR, AUC and the log loss of the scores as log-odds, mean and std over 2 folds x 5 seeds."""
    rows = []
    for seed in seeds:
        fold = make_folds(known, n_folds, seed)
        for f in range(n_folds):
            fit, evaluate = known[fold != f], known[fold == f]
            s = np.asarray(score_fn(fit, evaluate), dtype=float)
            y = evaluate["member"].to_numpy()
            p = np.clip(1 / (1 + np.exp(-s)), 1e-6, 1 - 1e-6)
            rows.append({"TPR": tpr_at_5fpr(s.tolist(), y.tolist()), "AUC": auc(s, y),
                         "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())})
    r = pd.DataFrame(rows)
    return {f"{m} {s}": getattr(r[m], s)() for m in r for s in ("mean", "std")}


def run_cv(known: pd.DataFrame, variants: dict) -> pd.DataFrame:
    rows = []
    for name, fn in variants.items():
        rows.append({"variant": name, **cv(known, fn)})
        print(pd.DataFrame(rows[-1:]).round(3).to_string(header=len(rows) == 1, index=False), flush=True)
    return pd.DataFrame(rows)


def single_table(known: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        v, y = known[col].fillna(0).to_numpy(), known["member"].to_numpy()
        rows.append({"feature": col, "AUC": auc(v, y), "TPR": tpr_at_5fpr(v.tolist(), y.tolist())})
    return pd.DataFrame(rows).round(3)


def logodds_all(data: pd.DataFrame, score_fn, seed: int = 0, n_folds: int = 5) -> pd.DataFrame:
    """Calibrated log-odds of all web photo records: known out of fold (n_folds), test from the fit on all known."""
    web = data[data["source_type"] == WEB]
    known = web[web["split"] == "known"].assign(member=lambda d: d["member"].astype(int))
    out = pd.Series(np.nan, index=web.index)
    fold = make_folds(known, n_folds, seed)
    for f in range(n_folds):
        out.loc[known.index[fold == f]] = score_fn(known[fold != f], known[fold == f])
    test = web[web["split"] == "test"]
    out.loc[test.index] = score_fn(known, test)
    return pd.DataFrame({"id": web["id"], "logodds": out})


def explore_v1(out_dir: str) -> None:
    """Existing passes only: baseline, small feature sets, z-scores, duplicates, pooled fit, blind features."""
    from .data import find_root, load_records
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    records = load_records(find_root())
    data = photo_data(records)
    data = pd.concat([data, duplicate_features(data), image_features(data)], axis=1)
    data["name"] = data["source"].str.lstrip("@").str.rsplit("_", n=1).str[0]
    known_all = data[data["split"] == "known"].assign(member=lambda d: d["member"].astype(int))
    known = known_all[known_all["source_type"] == WEB]
    photographers = known_all[known_all["source_type"] == "photographer"]
    print("web photo known", len(known), "members", known["member"].sum(), flush=True)
    feats = BASE + PLIRA + ["dup", "dup_gain", "img_bytes", "img_q0", "img_mean", "img_std", "img_sat", "n_tokens"]
    single = single_table(known, feats)
    single.to_csv(out / "single.csv", index=False)
    print(single.sort_values("AUC").to_string(), flush=True)
    dups = known[known["dup"] > 0]
    print("duplicates in known:", len(dups), "members", dups["member"].sum(), flush=True)
    small = ["lira_own_gain", "lira_txt_gain", "dom_own", "ref_img", "img_gain", "plira_gain"]
    variants = {
        "v8a base C=1": lr_score(BASE),
        "base C=0.1": lr_score(BASE, 0.1),
        "base C=0.01": lr_score(BASE, 0.01),
        "base + plira": lr_score(BASE + PLIRA),
        "small C=1": lr_score(small),
        "small C=0.1": lr_score(small, 0.1),
        "base + dup": lr_score(BASE + ["dup", "dup_gain"]),
        "base z-length": zscore_columns(BASE)(lr_score(BASE)),
        "base z-length 3 bins": zscore_columns(BASE, 3)(lr_score(BASE)),
        "base + n_tokens": lr_score(BASE + ["n_tokens"]),
        "base pooled photographers": lr_score(BASE, extra_fit=photographers),
        "blind image": lr_score(["img_bytes", "img_q0", "img_mean", "img_std", "img_sat"]),
        "blind caption words": text_score("caption"),
        "blind caption chars": text_score("caption", "char_wb", (2, 4)),
        "blind name chars": text_score("name", "char_wb", (2, 4)),
    }
    table = run_cv(known, variants)
    table.to_csv(out / "cv.csv", index=False)
    logodds_all(data, lr_score(BASE)).to_csv(out / "logodds_base.csv", index=False)
