"""Calibrated log-odds of membership for the records of one set type (made for photographers): record score + owner mixture.

Record level: a logistic regression on the v6 features (as attacks-v8a). It gives llr = log p(x | member) - log p(x | non-member).
Owner level: each owner is a victim (a fraction r of its records are members) or clean (a fraction eps).
The prior of the victim state depends on the owner group (victim, full, other). EM on the unlabeled records fits r, eps and the priors.
The log-odds of a record use the posterior of its owner as the prior.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .attacks import (all_features, find_passes, owner_groups, owner_prior_shift, record_logodds, test_like_weights,
                      v6_columns_of, v6_features, weighted_tpr_at_5fpr)
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


def baseline_folds(data: pd.DataFrame, seed: int) -> pd.Series:
    """The folds of src/validation.py on all known records, as in the baseline attacks-v8a."""
    known = data[data["split"] == "known"]
    return pd.Series(make_folds(known, 2, seed), index=known.index)


def oof_llr(known: pd.DataFrame, cols, fold: pd.Series, C: float = 1.0) -> pd.Series:
    """Out-of-fold llr of the known records of one type."""
    fold = fold.loc[known.index].to_numpy()
    out = pd.Series(np.nan, index=known.index)
    for f in range(2):
        out[fold == f] = fit_llr(known[fold != f], known[fold == f], cols, C)
    return out


def owner_sums(values: np.ndarray, owner) -> np.ndarray:
    """Sum of the values over the records of the owner of each record."""
    _, code = np.unique(np.asarray(owner), return_inverse=True)
    return np.bincount(code, weights=values)[code]


def mixture(llr, owner, group, params: dict, tau: float = 1.0, evidence: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Record log-odds and owner log-odds (victim against clean) of the mixture model.

    The owner evidence is the sum over the records of the owner. The record itself is a part of it (exact Bayes).
    Without evidence, only the prior of the group acts (the owner prior alone).
    """
    L = tau * np.asarray(llr, dtype=float)
    r, eps = params["r"], params["eps"]
    a, b = logit(r) + L, logit(eps) + L
    owner_lo = logit(np.array([params["q"][g] for g in np.asarray(group)], dtype=float))
    if evidence:
        lv = np.logaddexp(np.log(r) + L, np.log1p(-r))
        lc = np.logaddexp(np.log(eps) + L, np.log1p(-eps))
        owner_lo = owner_lo + owner_sums(lv - lc, owner)
    lw, l1w = log_sigmoid(owner_lo), log_sigmoid(-owner_lo)
    lp = np.logaddexp(lw + log_sigmoid(a), l1w + log_sigmoid(b))
    lq = np.logaddexp(lw + log_sigmoid(-a), l1w + log_sigmoid(-b))
    return lp - lq, owner_lo


def fit_mixture(llr, owner, group, tau: float = 1.0, iters: int = 200) -> dict:
    """EM on unlabeled records. The start values (r above eps) select the victim state."""
    L = tau * np.asarray(llr, dtype=float)
    group = np.asarray(group)
    _, first = np.unique(np.asarray(owner), return_index=True)
    params = {"r": 0.9, "eps": 0.05, "q": {g: 0.5 for g in GROUPS}}
    for _ in range(iters):
        _, owner_lo = mixture(llr, owner, group, params, tau)
        w = sigmoid(owner_lo)
        a, b = sigmoid(logit(params["r"]) + L), sigmoid(logit(params["eps"]) + L)
        params["r"] = float(np.clip(((w * a).sum() + CLIP) / (w.sum() + CLIP), CLIP, 1 - CLIP))
        params["eps"] = float(np.clip((((1 - w) * b).sum() + CLIP) / ((1 - w).sum() + 1), CLIP, 1 - CLIP))
        for g in GROUPS:
            sel = first[group[first] == g]
            params["q"][g] = float(np.clip(w[sel].mean(), CLIP, 1 - CLIP)) if len(sel) else 0.5
    return params


def owner_mean_term(llr, owner, w: float = 1.0) -> np.ndarray:
    """The owner term of attacks-v4: llr plus w times the mean llr of the other records of the owner."""
    s = pd.Series(np.asarray(llr, dtype=float))
    g = s.groupby(np.asarray(owner))
    loo = (g.transform("sum") - s) / (g.transform("size") - 1).clip(lower=1)
    return (s + w * loo).to_numpy()


# Structures of the semi-synthetic test. base copies the structure that the text types show in test.
# The other scenarios break one assumption of the mixture each; "all wrong" breaks three.
SCENARIOS = {
    "base": {"hidden": 6, "victim_nonmembers": 0, "spread": 0},
    "victim 75%": {"hidden": 6, "victim_nonmembers": 2, "spread": 0},
    "clean 4%": {"hidden": 5, "victim_nonmembers": 0, "spread": 3},
    "hidden 3": {"hidden": 3, "victim_nonmembers": 0, "spread": 0},
    "hidden 12": {"hidden": 12, "victim_nonmembers": 0, "spread": 0},
    "all wrong": {"hidden": 3, "victim_nonmembers": 2, "spread": 3},
}


def pseudo_test(known: pd.DataFrame, rng, hidden: int = 6, victim_nonmembers: int = 0, spread: int = 0) -> pd.DataFrame:
    """A semi-synthetic test from the known records of one type.

    The 12 victim owners get three roles in a random order:
    - hidden: their members (and victim_nonmembers of their non-members) go into the group "other",
    - spread: each of their members goes into a random clean owner (clean owners with a few members),
    - flagged: the other victims keep the group "victim", with the same records as a hidden owner.
    The full and other owners are clean.
    """
    victims = rng.permutation(known.loc[known["group"] == "victim", "source"].unique())
    roles = {"hidden": victims[:hidden], "spread": victims[hidden:hidden + spread], "flagged": victims[hidden + spread:]}
    clean = known[known["group"].isin(["full", "other"])].assign(sim_group=lambda d: d["group"])
    parts = [clean]
    for role, group in [("flagged", "victim"), ("hidden", "other")]:
        part = known[known["source"].isin(roles[role])]
        nonmembers = part[part["member"] == 0].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        nonmembers = nonmembers[nonmembers.groupby("source").cumcount() < victim_nonmembers]
        parts.append(pd.concat([part[part["member"] == 1], nonmembers]).assign(sim_group=group))
    moved = known[known["source"].isin(roles["spread"]) & (known["member"] == 1)]
    owners = clean.groupby("source")["sim_group"].first()
    target = rng.choice(owners.index, size=len(moved), replace=False)
    parts.append(moved.assign(source=target, sim_group=owners[target].to_numpy()))
    return pd.concat(parts)


def score_variants(llr, owner, group, known_rate: float) -> dict:
    """All owner variants of one set of records. The EM sees no label."""
    out = {"record": np.asarray(llr, dtype=float)}
    for tau in (1.0, 0.5):
        params = fit_mixture(llr, owner, group, tau)
        out[f"mixture tau={tau}"] = mixture(llr, owner, group, params, tau)[0]
        if tau == 1.0:
            out["group prior"] = mixture(llr, owner, group, params, tau, evidence=False)[0]
    out["owner mean w=3"] = owner_mean_term(llr, owner, 3.0)
    frame = pd.DataFrame({"split": "known", "source": np.asarray(owner), "source_type": TYPE})
    prior = logit(known_rate)
    out["owner prior shift w=1"] = (owner_prior_shift(frame, pd.Series(np.asarray(llr) + prior), "known", 1.0) - prior).to_numpy()
    return out


def metrics_row(scores, labels) -> dict:
    labels = np.asarray(labels).astype(int)
    return {"TPR@5%FPR": tpr_at_5fpr(list(scores), list(labels)), "AUC": auc(scores, labels)}


def honest_check(known: pd.DataFrame, cols, folds: dict, scenario: dict, roles: int = 10, C: float = 1.0) -> pd.DataFrame:
    """Semi-synthetic check of the owner variants on out-of-fold llr (seeds x random roles)."""
    rows = []
    for seed, fold in folds.items():
        data = known.assign(llr=oof_llr(known, cols, fold, C))
        for role in range(roles):
            sim = pseudo_test(data, np.random.default_rng(1000 * seed + role), **scenario)
            for name, scores in score_variants(sim["llr"], sim["source"], sim["sim_group"], known["member"].mean()).items():
                rows.append({"seed": seed, "role": role, "variant": name, **metrics_row(scores, sim["member"])})
    return pd.DataFrame(rows)


def known_cv(known: pd.DataFrame, cols, folds: dict, C: float = 1.0) -> pd.DataFrame:
    """The validation of src/validation.py on known: record llr, and llr + the log prior odds of the owner group.

    In known, only victim owners have members (6 of 14). The group of an evaluated record comes from the fit fold labels.
    """
    rows = []
    for seed, fold in folds.items():
        llr = oof_llr(known, cols, fold, C)
        fold = fold.loc[known.index].to_numpy()
        for f in range(2):
            fit, ev = known[fold != f], known[fold == f]
            for name, s in [("record", llr[ev.index]), ("record + known group", llr[ev.index] + known_prior(fit, ev))]:
                rows.append({"seed": seed, "fold": f, "variant": name, **metrics_row(s.to_numpy(), ev["member"])})
    return pd.DataFrame(rows)


def known_prior(fit: pd.DataFrame, ev: pd.DataFrame) -> np.ndarray:
    """Log prior odds of each evaluated known record from the fit labels: the member rate of victims or of the rest."""
    victim = ev["source"].isin(fit.loc[fit["member"] == 1, "source"])
    fit_victim = fit["source"].isin(fit.loc[fit["member"] == 1, "source"])
    rate = {True: fit.loc[fit_victim, "member"].mean(), False: max(fit.loc[~fit_victim, "member"].mean(), CLIP)}
    return logit(victim.map(rate).to_numpy())


def structure(data: pd.DataFrame, types) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The structure of test for each set type: EM parameters and a histogram of the owner member rates.

    The record score of the text types is almost exact, so their test structure is almost known.
    """
    params_rows, hist_rows = [], []
    for source_type in types:
        part = data[data["source_type"] == source_type]
        known, test = part[part["split"] == "known"], part[part["split"] == "test"]
        llr = fit_llr(known, test, v6_columns_of(source_type))
        params = fit_mixture(llr, test["source"], test["group"])
        lo, owner_lo = mixture(llr, test["source"], test["group"], params)
        victims = pd.Series(sigmoid(owner_lo), index=test.index).groupby(test["source"]).first().sum()
        params_rows.append({"source_type": source_type, "r": params["r"], "eps": params["eps"],
                            **{f"q_{g}": v for g, v in params["q"].items()},
                            "expected members": float(sigmoid(lo).sum()), "victim owners": float(victims)})
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


def set_logodds(data: pd.DataFrame, source_type: str = TYPE, cols=None, tau: float = 1.0, C: float = 1.0) -> pd.DataFrame:
    """Calibrated log-odds of membership for all records of one set type (id, split, logodds).

    Test: llr from the regression on all known records of the type, then the mixture with the EM parameters of test.
    Known: out-of-fold llr (folds of seed 0) plus the log prior odds of the owner group in known.
    Known does not use the test mixture. In known, a victim has 6 members in 14 records, and no owner is a hidden victim.
    data needs the feature columns and the column group (owner_groups).
    """
    cols = cols or v6_columns_of(source_type)
    part = data[data["source_type"] == source_type]
    known, test = part[part["split"] == "known"], part[part["split"] == "test"]
    llr_test = fit_llr(known, test, cols, C)
    params = fit_mixture(llr_test, test["source"], test["group"], tau)
    lo_test, _ = mixture(llr_test, test["source"], test["group"], params, tau)
    fold = baseline_folds(data, 0)
    llr_known = oof_llr(known, cols, fold, C)
    lo_known = pd.Series(np.nan, index=known.index)
    f_known = fold.loc[known.index].to_numpy()
    for f in range(2):
        fit, ev = known[f_known != f], known[f_known == f]
        lo_known[ev.index] = llr_known[ev.index] + known_prior(fit, ev)
    print(source_type, "test mixture parameters", params, flush=True)
    return pd.concat([pd.DataFrame({"id": known["id"], "split": "known", "logodds": lo_known}),
                      pd.DataFrame({"id": test["id"], "split": "test", "logodds": lo_test})])


def known_mixture(data: pd.DataFrame, fit_index, ev_index, cols, C: float, source_type: str = TYPE,
                  tau: float = 1.0) -> pd.Series:
    """The mixture on the evaluated known records of one type, for the global simulation on known.

    The group of an owner comes from the fit labels: victim if the owner has a fit member. EM sees no label.
    Only the evaluated records give owner evidence.
    """
    fit, ev = data.loc[fit_index], data.loc[ev_index]
    fit, ev = fit[fit["source_type"] == source_type], ev[ev["source_type"] == source_type]
    group = ev["group"].where(ev["group"] != "victim", "other").mask(ev["source"].isin(fit.loc[fit["member"] == 1, "source"]), "victim")
    llr = fit_llr(fit, ev, cols, C)
    params = fit_mixture(llr, ev["source"], group, tau)
    return pd.Series(mixture(llr, ev["source"], group, params, tau)[0], index=ev.index)


def global_sim(data: pd.DataFrame, folds: dict, cols, C: float, tau: float = 1.0) -> pd.DataFrame:
    """The global simulation of explore_v9 (attacks.py) on known.

    Variants: the coordinator owner prior (w = 1) for all set types, and the same with the photographer records
    replaced by the mixture log-odds.
    """
    known_index = data.index[data["split"] == "known"]
    test = data["split"] == "test"
    full = record_logodds(data, known_index, v6_columns_of)
    expected = sigmoid(full[test]).groupby(data.loc[test, "source_type"]).sum()
    known = data.loc[known_index].assign(member=lambda d: d["member"].astype(int))
    sample_w = pd.Series(test_like_weights(known, expected, data.loc[test, "source_type"].value_counts()), index=known_index)
    rows = []
    for seed, fold in folds.items():
        for f in range(2):
            fit_index, ev = known_index[fold.to_numpy() != f], known_index[fold.to_numpy() == f]
            logodds = record_logodds(data, fit_index, v6_columns_of)
            base = owner_prior_shift(data, logodds, "known", 1.0)
            mixed = base.copy()
            photo = known_mixture(data, fit_index, ev, cols, C, TYPE, tau)
            mixed[photo.index] = photo
            for name, score in [("owner prior shift w=1", base), ("photographer mixture", mixed)]:
                s, y, w = score[ev].to_numpy(), known.loc[ev, "member"].to_numpy(), sample_w[ev].to_numpy()
                tpr, threshold = weighted_tpr_at_5fpr(s, y, w)
                is_photo = (known.loc[ev, "source_type"] == TYPE).to_numpy()
                rows.append({"variant": name, "all": tpr,
                             "tpr photographer": (s[is_photo & (y == 1)] >= threshold).mean(),
                             "fp photographer": (w[is_photo & (y == 0)] * (s[is_photo & (y == 0)] >= threshold)).sum(),
                             "fp other types": (w[~is_photo & (y == 0)] * (s[~is_photo & (y == 0)] >= threshold)).sum()})
    table = pd.DataFrame(rows).groupby("variant", sort=False).agg(["mean", "std"]).round(3)
    table.columns = [f"{m} {s}" for m, s in table.columns]
    return table.reset_index()


def test_owners(data: pd.DataFrame, cols, C: float = 1.0) -> list[dict]:
    """EM parameters on the photographer test, the expected members and the sorted owner posteriors of the group other."""
    part = data[data["source_type"] == TYPE]
    known, test = part[part["split"] == "known"], part[part["split"] == "test"]
    llr = fit_llr(known, test, cols, C)
    out = []
    for tau in (1.0, 0.5):
        params = fit_mixture(llr, test["source"], test["group"], tau)
        lo, owner_lo = mixture(llr, test["source"], test["group"], params, tau)
        other = pd.Series(sigmoid(owner_lo), index=test.index)[test["group"] == "other"].groupby(test["source"]).first()
        out.append({"tau": tau, "r": round(params["r"], 3), "q_other": round(params["q"]["other"], 3),
                    "expected members": round(float(sigmoid(lo).sum()), 1),
                    "other owners above 0.5": int((other > 0.5).sum()),
                    "other owners 0.05 to 0.95": int(((other > 0.05) & (other < 0.95)).sum()),
                    "top 12 other owners": " ".join(f"{v:.2f}" for v in other.sort_values(ascending=False).head(12))})
    return out


def no_domain_columns(source_type: str = TYPE) -> list[str]:
    """v6 without the features of the domain reference models. Their cross-fit differs between known and test."""
    return [c for c in v6_columns_of(source_type) if not c.startswith("dom")]


RECORD_VARIANTS = {
    "v6 C=0.1": (v6_columns_of, 0.1),
    "v6 C=0.03": (v6_columns_of, 0.03),
    "v6 C=0.01": (v6_columns_of, 0.01),
    "v6 no domain C=0.1": (no_domain_columns, 0.1),
}


def explore(out_dir: str, variants=RECORD_VARIANTS, final="v6 C=0.1") -> None:
    """Record variants with the known CV, the honest check and the test summary. For the final variant also the
    scenarios, the llr shift, the global simulation and the log-odds file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    data = load_data()
    folds = {seed: baseline_folds(data, seed) for seed in range(5)}
    known = data[(data["source_type"] == TYPE) & (data["split"] == "known")].copy()
    known["member"] = known["member"].astype(int)
    cv, check, owners = [], [], []
    for name, (columns, C) in variants.items():
        cols = columns(TYPE)
        cv.append(summary(known_cv(known, cols, folds, C)).assign(record=name))
        check.append(summary(honest_check(known, cols, folds, SCENARIOS["base"], C=C)).assign(record=name))
        owners += [{"record": name, **row} for row in test_owners(data, cols, C)]
        print(name, "done", flush=True)
    tables = {"known_cv": pd.concat(cv), "honest_check": pd.concat(check), "test_owners": pd.DataFrame(owners)}
    columns, C = variants[final]
    cols = columns(TYPE)
    tables["scenarios"] = pd.concat([summary(honest_check(known, cols, folds, sc, C=C)).assign(scenario=name)
                                     for name, sc in SCENARIOS.items() if name != "base"])
    tables["llr_shift"] = llr_shift(data, cols, C)
    tables["global_sim"] = global_sim(data, folds, cols, C)
    for file, table in tables.items():
        table.to_csv(out / f"{file}.csv", index=False)
        print(file, "\n", table.to_string(), flush=True)
    set_logodds(data, TYPE, cols, C=C).to_csv(out / "submission_photographer_logodds.csv", index=False)


def llr_shift(data: pd.DataFrame, cols, C: float = 1.0) -> pd.DataFrame:
    """Distribution of the llr in known (out of fold) and in test, for each owner group. A shift breaks the calibration.

    In test, the full owners are almost surely clean and the victims almost all members.
    """
    part = data[data["source_type"] == TYPE]
    known, test = part[part["split"] == "known"], part[part["split"] == "test"]
    llr = pd.concat([oof_llr(known, cols, baseline_folds(data, 0), C), pd.Series(fit_llr(known, test, cols, C), index=test.index)])
    label = part["member"].map({1.0: " member", 0.0: " non-member"}).fillna("")
    name = part["split"] + " " + part["group"] + label
    return llr.groupby(name).describe(percentiles=[0.1, 0.5, 0.9]).round(2).reset_index(names="records")
