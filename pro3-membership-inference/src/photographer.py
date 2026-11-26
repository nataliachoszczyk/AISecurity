"""Calibrated log-odds of membership for the photographer records: a record score and an owner mixture (dataset inference).

Record level: a logistic regression on the v6 features (as attacks-v8a). It gives llr = log p(x | member) - log p(x | non-member).
Owner level: each owner is a victim (a fraction r of its records are members) or clean (a fraction eps).
The prior of the victim state depends on the owner group (victim, full, other). EM on the unlabeled records fits r, eps and the priors.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .attacks import SET_TYPES, all_features, find_passes, owner_groups, v6_columns_of, v6_features
from .data import find_root, load_records
from .metrics import auc, tpr_at_5fpr
from .validation import make_folds

TYPE = "photographer"
GROUPS = ("victim", "full", "other")
CLIP = 1e-4


def load_data() -> pd.DataFrame:
    records = load_records(find_root())
    data = pd.concat([records, all_features(find_passes(), records), v6_features(records)], axis=1)
    data["group"] = owner_groups(data)
    return data.drop(columns=["text", "caption", "title", "content", "image"], errors="ignore")


def logit(p):
    return np.log(p) - np.log1p(-p)


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def sigmoid(x):
    return np.exp(log_sigmoid(x))


def fit_llr(fit: pd.DataFrame, rows: pd.DataFrame, cols, C: float = 1.0) -> np.ndarray:
    """Log-likelihood ratio of each row: the log-odds of the regression minus the log prior odds of the fit part."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(fit[cols], fit["member"].astype(int))
    return model.decision_function(rows[cols]) - logit(fit["member"].mean())


def oof_llr(known: pd.DataFrame, cols, seed: int, C: float = 1.0) -> tuple[pd.Series, np.ndarray]:
    """Out-of-fold llr of the known records with the 2 folds of src/validation.py. Also return the folds."""
    fold = make_folds(known, 2, seed)
    out = pd.Series(np.nan, index=known.index)
    for f in range(2):
        out[fold == f] = fit_llr(known[fold != f], known[fold == f], cols, C)
    return out, fold


def mixture(llr, owner, group, params: dict, tau: float = 1.0, evidence: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Record log-odds and owner log-odds (victim against clean) of the mixture model.

    The owner evidence is the sum over the records of the owner. The record itself is a part of it (exact Bayes).
    Without evidence, only the prior of the group acts (the owner prior alone).
    """
    L = tau * np.asarray(llr, dtype=float)
    r, eps = params["r"], params["eps"]
    a, b = logit(r) + L, logit(eps) + L
    owner_lo = logit(np.asarray(pd.Series(group).map(params["q"]), dtype=float))
    if evidence:
        lv = np.logaddexp(np.log(r) + L, np.log1p(-r))
        lc = np.logaddexp(np.log(eps) + L, np.log1p(-eps))
        owner_lo = owner_lo + pd.Series(lv - lc).groupby(np.asarray(owner)).transform("sum").to_numpy()
    lw, l1w = log_sigmoid(owner_lo), log_sigmoid(-owner_lo)
    lp = np.logaddexp(lw + log_sigmoid(a), l1w + log_sigmoid(b))
    lq = np.logaddexp(lw + log_sigmoid(-a), l1w + log_sigmoid(-b))
    return lp - lq, owner_lo


def fit_mixture(llr, owner, group, tau: float = 1.0, iters: int = 300) -> dict:
    """EM on unlabeled records. The start values (r above eps) select the victim state."""
    L = tau * np.asarray(llr, dtype=float)
    owner, group = np.asarray(owner), np.asarray(group)
    params = {"r": 0.9, "eps": 0.05, "q": {g: 0.5 for g in GROUPS}}
    for _ in range(iters):
        _, owner_lo = mixture(llr, owner, group, params, tau)
        w = sigmoid(owner_lo)
        a, b = sigmoid(logit(params["r"]) + L), sigmoid(logit(params["eps"]) + L)
        params["r"] = float(np.clip((w * a).sum() / w.sum(), CLIP, 1 - CLIP))
        params["eps"] = float(np.clip(((1 - w) * b).sum() / (1 - w).sum(), CLIP, 1 - CLIP))
        per_owner = pd.DataFrame({"owner": owner, "group": group, "w": w}).groupby("owner").first()
        q = per_owner.groupby("group")["w"].mean()
        params["q"] = {g: float(np.clip(q.get(g, 0.5), CLIP, 1 - CLIP)) for g in GROUPS}
    return params


def owner_mean_term(llr, owner, w: float = 1.0) -> np.ndarray:
    """llr + w * mean llr of the other records of the owner (the owner term of attacks-v4)."""
    s = pd.Series(np.asarray(llr, dtype=float))
    g = s.groupby(np.asarray(owner))
    loo = (g.transform("sum") - s) / (g.transform("size") - 1).clip(lower=1)
    return (s + w * loo).to_numpy()


def owner_z(llr, owner, group, w: float = 1.0) -> np.ndarray:
    """llr + w * t statistic of the owner mean against the clean owners (full and other). An idea from "photo_z"."""
    s = pd.Series(np.asarray(llr, dtype=float))
    clean = np.isin(np.asarray(group), ["full", "other"])
    mu, sd = s[clean].mean(), s[clean].std()
    g = s.groupby(np.asarray(owner))
    t = (g.transform("mean") - mu) / (sd / np.sqrt(g.transform("size")))
    return (s + w * t).to_numpy()


def pseudo_test(known: pd.DataFrame, rng) -> pd.DataFrame:
    """A semi-synthetic test from known photographer records with the structure of the real test.

    Half of the victim owners keep their group (their members only: the test part of a victim is almost all members).
    The other half become hidden victims: their members only, in the group "other". All full and other owners are clean.
    """
    victims = rng.permutation(known.loc[known["group"] == "victim", "source"].unique())
    flagged, hidden = victims[: len(victims) // 2], victims[len(victims) // 2:]
    members = known[known["member"] == 1]
    parts = [members[members["source"].isin(flagged)].assign(sim_group="victim"),
             members[members["source"].isin(hidden)].assign(sim_group="other"),
             known[known["group"].isin(["full", "other"])].assign(sim_group=lambda d: d["group"])]
    return pd.concat(parts)


def score_variants(llr, owner, group) -> dict:
    """All owner variants of one set of records. The EM sees no label."""
    out = {"record": np.asarray(llr, dtype=float)}
    for tau in (1.0, 0.5):
        params = fit_mixture(llr, owner, group, tau)
        out[f"mixture tau={tau}"] = mixture(llr, owner, group, params, tau)[0]
        if tau == 1.0:
            out["group prior"] = mixture(llr, owner, group, params, tau, evidence=False)[0]
            out["_params"] = params
    for w in (1.0, 3.0):
        out[f"owner mean w={w}"] = owner_mean_term(llr, owner, w)
    out["owner z w=1"] = owner_z(llr, owner, group, 1.0)
    return out


def metrics_row(scores, labels) -> dict:
    labels = np.asarray(labels).astype(int)
    return {"TPR@5%FPR": tpr_at_5fpr(list(scores), list(labels)), "AUC": auc(scores, labels)}


def honest_check(known: pd.DataFrame, cols, seeds=range(5), roles: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Semi-synthetic check of the owner variants on out-of-fold llr. Return the rows and the EM parameters."""
    rows, fitted = [], []
    for seed in seeds:
        llr, _ = oof_llr(known, cols, seed)
        data = known.assign(llr=llr)
        for role in range(roles):
            sim = pseudo_test(data, np.random.default_rng(1000 * seed + role))
            variants = score_variants(sim["llr"], sim["source"], sim["sim_group"])
            params = variants.pop("_params")
            fitted.append({"seed": seed, "role": role, "r": params["r"], "eps": params["eps"],
                           **{f"q_{g}": v for g, v in params["q"].items()}})
            for name, scores in variants.items():
                rows.append({"seed": seed, "role": role, "variant": name, **metrics_row(scores, sim["member"])})
    return pd.DataFrame(rows), pd.DataFrame(fitted)


def known_cv(known: pd.DataFrame, cols, seeds=range(5)) -> pd.DataFrame:
    """The validation of src/validation.py on known: record llr, and llr + the log prior odds of the owner group.

    In known, only victim owners have members (6 of 14). The group of an evaluated record comes from the fit fold labels.
    """
    rows = []
    for seed in seeds:
        llr, fold = oof_llr(known, cols, seed)
        for f in range(2):
            fit, ev = known[fold != f], known[fold == f]
            victim = ev["source"].isin(fit.loc[fit["member"] == 1, "source"])
            fit_victim = fit["source"].isin(fit.loc[fit["member"] == 1, "source"])
            rate = {True: fit.loc[fit_victim, "member"].mean(), False: max(fit.loc[~fit_victim, "member"].mean(), CLIP)}
            prior = logit(victim.map(rate).to_numpy()) - logit(fit["member"].mean())
            for name, s in [("record", llr[ev.index]), ("record + known group", llr[ev.index] + prior)]:
                rows.append({"seed": seed, "fold": f, "variant": name, **metrics_row(s.to_numpy(), ev["member"])})
    return pd.DataFrame(rows)


def structure(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The structure of test for each set type: EM parameters and a histogram of the owner member rates.

    The record score of the text types is almost exact, so their test structure is almost known.
    The owner member rate is the mean record posterior of the owner under the fitted mixture.
    """
    params_rows, hist_rows = [], []
    for source_type in SET_TYPES:
        part = data[data["source_type"] == source_type]
        cols = v6_columns_of(source_type)
        known, test = part[part["split"] == "known"], part[part["split"] == "test"]
        llr = fit_llr(known, test, cols)
        params = fit_mixture(llr, test["source"], test["group"])
        lo, owner_lo = mixture(llr, test["source"], test["group"], params)
        params_rows.append({"source_type": source_type, "r": params["r"], "eps": params["eps"],
                            **{f"q_{g}": v for g, v in params["q"].items()},
                            "expected members": float(sigmoid(lo).sum()),
                            "victim owners": float(pd.Series(sigmoid(owner_lo), index=test.index).groupby(test["source"]).first().sum())})
        rate = pd.Series(sigmoid(lo), index=test.index).groupby(test["source"]).mean()
        groups = test.groupby("source")["group"].first()
        bins = pd.cut(rate, [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        for (g, b), n in rate.groupby([groups, bins], observed=True).size().items():
            hist_rows.append({"source_type": source_type, "group": g, "owner rate": str(b), "owners": int(n)})
    return pd.DataFrame(params_rows), pd.DataFrame(hist_rows)


def summary(rows: pd.DataFrame, by="variant") -> pd.DataFrame:
    table = rows.groupby(by, sort=False)[["TPR@5%FPR", "AUC"]].agg(["mean", "std"]).round(3)
    table.columns = [f"{m} {s}" for m, s in table.columns]
    return table.reset_index()


def photographer_logodds(data: pd.DataFrame, cols, tau: float = 1.0) -> pd.DataFrame:
    """Calibrated log-odds for all photographer records: test with the mixture, known out of fold (seed 0).

    Test: llr from the regression on all known records, then the mixture with the EM parameters of test.
    Known: out-of-fold llr plus the log prior odds of the owner group in known (the structure of known differs from test).
    """
    part = data[data["source_type"] == TYPE]
    known, test = part[part["split"] == "known"], part[part["split"] == "test"]
    llr_test = fit_llr(known, test, cols)
    params = fit_mixture(llr_test, test["source"], test["group"], tau)
    lo_test, _ = mixture(llr_test, test["source"], test["group"], params, tau)
    llr_known, fold = oof_llr(known, cols, seed=0)
    lo_known = pd.Series(np.nan, index=known.index)
    for f in range(2):
        fit, ev = known[fold != f], known[fold == f]
        victims = fit.loc[fit["member"] == 1, "source"]
        rate = {True: fit.loc[fit["source"].isin(victims), "member"].mean(), False: CLIP}
        lo_known[ev.index] = llr_known[ev.index] + logit(ev["source"].isin(victims).map(rate).to_numpy())
    print("test mixture parameters", params, flush=True)
    return pd.concat([pd.DataFrame({"id": known["id"], "split": "known", "logodds": lo_known}),
                      pd.DataFrame({"id": test["id"], "split": "test", "logodds": lo_test})])


def explore_v1(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    data = load_data()
    cols = v6_columns_of(TYPE)
    params, hist = structure(data)
    params.round(3).to_csv(out / "structure_params.csv", index=False)
    hist.to_csv(out / "structure_owners.csv", index=False)
    print(params.round(3).to_string(), "\n", hist.to_string(), flush=True)
    known = data[(data["source_type"] == TYPE) & (data["split"] == "known")].copy()
    known["member"] = known["member"].astype(int)
    cv = summary(known_cv(known, cols))
    cv.to_csv(out / "known_cv.csv", index=False)
    print(cv.to_string(), flush=True)
    rows, fitted = honest_check(known, cols)
    table = summary(rows)
    table.to_csv(out / "honest_check.csv", index=False)
    fitted.describe().round(3).to_csv(out / "honest_check_params.csv")
    print(table.to_string(), "\n", fitted.describe().round(3).to_string(), flush=True)
    result = photographer_logodds(data, cols)
    result.to_csv(out / "photographer_logodds.csv", index=False)
    test = result[result["split"] == "test"]
    print("test expected members", round(float(sigmoid(test["logodds"]).sum()), 1), flush=True)
