"""Calibrated log-odds of membership for the records of one set type (made for photographers): record score + owner mixture.

A logistic regression on the v6 features gives the evidence of each record. Each owner is a victim or clean, and EM on
the unlabeled test records estimates the probability of each owner to be a victim. The log-odds of a record use this
posterior of its owner as the prior.

Main function: set_logodds. The other functions are the checks of the method (known CV, honest check, global simulation).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .attacks import (SET_TYPES, all_features, find_passes, owner_groups, owner_prior_shift, record_logodds,
                      test_like_weights, v6_columns_of, v6_features, weighted_tpr_at_5fpr)
from .data import find_root, load_records
from .metrics import auc, tpr_at_5fpr
from .validation import make_folds

TYPE = "photographer"
GROUPS = ("victim", "full", "other")
C_RECORD = {TYPE: 0.1}  # job photographer-v2: C 0.1 is better than 1 for the photographers; other types keep 1
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


def mixture(llr, owner, group, params: dict, evidence: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Record log-odds and owner log-odds (victim against clean) of the mixture model.

    A victim has a fraction r of members, a clean owner a fraction eps. The prior of the victim state is q of the group.
    The owner evidence is the sum over the records of the owner. The record itself is a part of it (exact Bayes).
    Without evidence, only the prior of the group acts (the owner prior alone).
    """
    L = np.asarray(llr, dtype=float)
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


def fit_mixture(llr, owner, group, iters: int = 200) -> dict:
    """EM on unlabeled records. The start values (r above eps) select the victim state."""
    L = np.asarray(llr, dtype=float)
    group = np.asarray(group)
    _, first = np.unique(np.asarray(owner), return_index=True)
    params = {"r": 0.9, "eps": 0.05, "q": {g: 0.5 for g in GROUPS}}
    for _ in range(iters):
        _, owner_lo = mixture(llr, owner, group, params)
        w = sigmoid(owner_lo)
        a, b = sigmoid(logit(params["r"]) + L), sigmoid(logit(params["eps"]) + L)
        params["r"] = float(np.clip(((w * a).sum() + CLIP) / (w.sum() + CLIP), CLIP, 1 - CLIP))
        params["eps"] = float(np.clip((((1 - w) * b).sum() + CLIP) / ((1 - w).sum() + 1), CLIP, 1 - CLIP))
        for g in GROUPS:
            sel = first[group[first] == g]
            params["q"][g] = float(np.clip(w[sel].mean(), CLIP, 1 - CLIP)) if len(sel) else 0.5
    return params


def known_prior(fit: pd.DataFrame, ev: pd.DataFrame) -> np.ndarray:
    """Log prior odds of each evaluated known record from the fit labels: the member rate of victims or of the rest."""
    victim = ev["source"].isin(fit.loc[fit["member"] == 1, "source"])
    fit_victim = fit["source"].isin(fit.loc[fit["member"] == 1, "source"])
    rate = {True: fit.loc[fit_victim, "member"].mean(), False: max(fit.loc[~fit_victim, "member"].mean(), CLIP)}
    return logit(victim.map(rate).to_numpy())


def set_logodds(data: pd.DataFrame, source_type: str = TYPE, cols=None, C: float | None = None) -> pd.DataFrame:
    """Calibrated log-odds of membership for all records of one set type (columns id, split, logodds).

    Test: llr from the regression on all known records of the type, then the mixture with the EM parameters of test.
    Known: out-of-fold llr (folds of seed 0) plus the log prior odds of the owner group in known.
    Known does not use the test mixture. In known, a victim has 6 members in 14 records, and no owner is a hidden victim.
    The data must have the v6 feature columns and the column group. load_data gives both.
    """
    cols = cols or v6_columns_of(source_type)
    C = C_RECORD.get(source_type, 1.0) if C is None else C
    part = data[data["source_type"] == source_type]
    known, test = part[part["split"] == "known"], part[part["split"] == "test"]
    llr_test = fit_llr(known, test, cols, C)
    params = fit_mixture(llr_test, test["source"], test["group"])
    lo_test, _ = mixture(llr_test, test["source"], test["group"], params)
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


# Checks of the method.

def metrics_row(scores, labels) -> dict:
    labels = np.asarray(labels).astype(int)
    return {"TPR@5%FPR": tpr_at_5fpr(list(scores), list(labels)), "AUC": auc(scores, labels)}


def summary(rows: pd.DataFrame, by="variant") -> pd.DataFrame:
    table = rows.groupby(by, sort=False)[["TPR@5%FPR", "AUC"]].agg(["mean", "std"]).round(3)
    table.columns = [f"{m} {s}" for m, s in table.columns]
    return table.reset_index()


def known_cv(known: pd.DataFrame, cols, folds: dict, C: float) -> pd.DataFrame:
    """The validation of src/validation.py on known: record llr, and llr + the log prior odds of the owner group."""
    rows = []
    for seed, fold in folds.items():
        llr = oof_llr(known, cols, fold, C)
        fold = fold.loc[known.index].to_numpy()
        for f in range(2):
            fit, ev = known[fold != f], known[fold == f]
            for name, s in [("record", llr[ev.index]), ("record + known group", llr[ev.index] + known_prior(fit, ev))]:
                rows.append({"seed": seed, "fold": f, "variant": name, **metrics_row(s.to_numpy(), ev["member"])})
    return pd.DataFrame(rows)


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


def honest_check(known: pd.DataFrame, cols, folds: dict, scenario: dict, C: float, roles: int = 10) -> pd.DataFrame:
    """Semi-synthetic check on out-of-fold llr (seeds x random roles): record, group prior, mixture, prior shift.

    The EM sees no label. owner_prior_shift (w = 1) of attacks.py is the baseline of the coordinator.
    """
    rows = []
    for seed, fold in folds.items():
        data = known.assign(llr=oof_llr(known, cols, fold, C))
        for role in range(roles):
            sim = pseudo_test(data, np.random.default_rng(1000 * seed + role), **scenario)
            llr, owner, group = sim["llr"].to_numpy(), sim["source"], sim["sim_group"]
            params = fit_mixture(llr, owner, group)
            prior = logit(known["member"].mean())
            frame = pd.DataFrame({"split": "known", "source": owner.to_numpy(), "source_type": TYPE})
            variants = {"record": llr, "group prior": mixture(llr, owner, group, params, evidence=False)[0],
                        "mixture": mixture(llr, owner, group, params)[0],
                        "owner prior shift w=1": owner_prior_shift(frame, pd.Series(llr + prior), "known", 1.0).to_numpy()}
            for name, scores in variants.items():
                rows.append({"seed": seed, "role": role, "variant": name, **metrics_row(scores, sim["member"])})
    return pd.DataFrame(rows)


def structure(data: pd.DataFrame, types=SET_TYPES) -> pd.DataFrame:
    """EM parameters and expected members on the test of each set type. For the text types the record score is almost
    exact, so their test structure is almost known."""
    rows = []
    for source_type in types:
        part = data[data["source_type"] == source_type]
        known, test = part[part["split"] == "known"], part[part["split"] == "test"]
        llr = fit_llr(known, test, v6_columns_of(source_type), C_RECORD.get(source_type, 1.0))
        params = fit_mixture(llr, test["source"], test["group"])
        lo, owner_lo = mixture(llr, test["source"], test["group"], params)
        owners = pd.Series(sigmoid(owner_lo), index=test.index).groupby(test["source"]).first()
        groups = test.groupby("source")["group"].first()
        rows.append({"source_type": source_type, "r": params["r"], "eps": params["eps"],
                     **{f"q_{g}": v for g, v in params["q"].items()},
                     "expected members": float(sigmoid(lo).sum()),
                     "other owners above 0.5": int((owners[groups == "other"] > 0.5).sum()),
                     "other owners 0.05 to 0.95": int(owners[groups == "other"].between(0.05, 0.95).sum())})
    return pd.DataFrame(rows).round(3)


def known_mixture(data: pd.DataFrame, fit_index, ev_index, source_type: str = TYPE) -> pd.Series:
    """The mixture on the evaluated known records of one type, for the global simulation on known.

    The group of an owner comes from the fit labels: victim if the owner has a fit member. EM sees no label.
    Only the evaluated records give owner evidence.
    """
    fit, ev = data.loc[fit_index], data.loc[ev_index]
    fit, ev = fit[fit["source_type"] == source_type], ev[ev["source_type"] == source_type]
    group = ev["group"].where(ev["group"] != "victim", "other").mask(ev["source"].isin(fit.loc[fit["member"] == 1, "source"]), "victim")
    llr = fit_llr(fit, ev, v6_columns_of(source_type), C_RECORD.get(source_type, 1.0))
    params = fit_mixture(llr, ev["source"], group)
    return pd.Series(mixture(llr, ev["source"], group, params)[0], index=ev.index)


def global_sim(data: pd.DataFrame, folds: dict, expected: pd.Series) -> pd.DataFrame:
    """The global simulation of explore_v9 (attacks.py) on known, with the expected test members of each type.

    Variants: owner_prior_shift (w = 1) for all set types, and the same with the photographer records replaced by the
    mixture log-odds.
    """
    known_index = data.index[data["split"] == "known"]
    test = data["split"] == "test"
    known = data.loc[known_index].assign(member=lambda d: d["member"].astype(int))
    sample_w = pd.Series(test_like_weights(known, expected, data.loc[test, "source_type"].value_counts()), index=known_index)
    rows = []
    for fold in folds.values():
        for f in range(2):
            fit_index, ev = known_index[fold.to_numpy() != f], known_index[fold.to_numpy() == f]
            base = owner_prior_shift(data, record_logodds(data, fit_index, v6_columns_of), "known", 1.0)
            mixed = base.copy()
            photo = known_mixture(data, fit_index, ev)
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


def explore(out_dir: str) -> None:
    """The final module: checks on known, the test structure, the global simulation and the log-odds file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    data = load_data()
    folds = {seed: baseline_folds(data, seed) for seed in range(5)}
    known = data[(data["source_type"] == TYPE) & (data["split"] == "known")].copy()
    known["member"] = known["member"].astype(int)
    cols, C = v6_columns_of(TYPE), C_RECORD[TYPE]
    tables = {"known_cv": summary(known_cv(known, cols, folds, C)),
              "scenarios": pd.concat([summary(honest_check(known, cols, folds, sc, C)).assign(scenario=name)
                                      for name, sc in SCENARIOS.items()]),
              "structure": structure(data)}
    test = data["split"] == "test"
    record = record_logodds(data, data.index[~test], v6_columns_of)
    expected = {"v8a": sigmoid(record[test]).groupby(data.loc[test, "source_type"]).sum()}
    mix = expected["v8a"].copy()
    for source_type in SET_TYPES:
        result = set_logodds(data, source_type)
        result = result[result["split"] == "test"]
        mix[source_type] = float(sigmoid(result["logodds"]).sum())
        if source_type == TYPE:
            final = result
    expected["mixture"] = mix
    tables["expected_members"] = pd.DataFrame(expected).round(1).reset_index(names="source_type")
    tables["global_sim"] = pd.concat([global_sim(data, folds, e).assign(weights=name) for name, e in expected.items()])
    for file, table in tables.items():
        table.to_csv(out / f"{file}.csv", index=False)
        print(file, "\n", table.to_string(), flush=True)
    full = set_logodds(data, TYPE)
    assert len(full) == (data["source_type"] == TYPE).sum() and np.isfinite(full["logodds"]).all()
    assert final["logodds"].equals(full.loc[full["split"] == "test", "logodds"])
    full[["id", "logodds"]].to_csv(out / "submission_photographer_logodds.csv", index=False)
