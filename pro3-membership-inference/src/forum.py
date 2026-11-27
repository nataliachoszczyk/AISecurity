"""Forum post module: forum shadow models (LiRA with several IN and OUT models) and the calibrated log-odds of forum posts.

The shadow models of src/refmodel.py are expensive (all records, 2.75 hours for a pair). A forum shadow model trains only
on one random half of the forum posts and a small sample of the other texts. It scores only the forum posts.
"""
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import find_root, load_records
from .refmodel import shadow_halves, train

FORUM = "forum post"
OTHER_SAMPLE = 0.05  # fraction of the other texts in the train set of each model; it keeps a part of the S1 mixture


def forum_halves(records: pd.DataFrame, split: int) -> pd.Series:
    """A random half (0 or 1) of each forum post for one split."""
    forum = records[records["source_type"] == FORUM]
    return pd.Series(shadow_halves(forum, seed=20 + split), index=forum["id"].to_numpy())


def run_forum_shadow(splits: list[int], device: str, out_dir: str, limit: int | None = None) -> None:
    """For each split: train one model on each half of the forum posts (one epoch, as S1), then score all forum posts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .logprobs import reference_path, text_pass
    torch.cuda.set_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        root = find_root()
        records = load_records(root)
        texts = records[records["kind"] == "text"]
        forum = texts[texts["source_type"] == FORUM]
        other = texts[texts["source_type"] != FORUM]
        score = forum.head(limit // 2) if limit else forum
        tok = AutoTokenizer.from_pretrained(root / "model" / "llm")
        for split in splits:
            halves = forum_halves(records, split)
            halves.rename("half").rename_axis("id").to_csv(out / f"forum_halves_s{split}.csv")
            for half in (0, 1):
                chosen = forum[forum["id"].map(halves).to_numpy() == half]["content"].tolist()
                sample = other.sample(frac=OTHER_SAMPLE, random_state=100 * split + half)["content"].tolist()
                training = (chosen[:limit] + sample[:limit // 4]) if limit else chosen + sample
                print(f"{time.strftime('%H:%M:%S')} split {split} half {half}: {len(chosen)} forum posts, "
                      f"{len(training)} training texts", flush=True)
                llm = AutoModelForCausalLM.from_pretrained(reference_path(), dtype=torch.float32).to(device)
                train(llm, tok, training, device, epochs=1)
                with torch.no_grad():
                    text_pass(llm, tok, score, device, out, f"fshadow_s{split}_h{half}_text")
                del llm
                torch.cuda.empty_cache()
    except Exception:
        # A failed kernel loses all its output. Thus log the error and keep the models that are complete.
        traceback.print_exc()
        (out / f"error_{device.replace(':', '')}.txt").write_text(traceback.format_exc())


def t_stat(x: np.ndarray) -> float:
    """Mean of the token values divided by its standard error: a long post with a small mean gain can be strong evidence."""
    return float(x.mean() / max(x.std(ddof=1), 1e-6) * np.sqrt(len(x))) if len(x) > 1 else 0.0


def token_features(records: pd.DataFrame) -> pd.DataFrame:
    """Length and difficulty features of the forum posts from the passes of the target, Qwen and the shadow pair.

    g is the gain of the target over the OUT model of the shadow pair, d the gain over Qwen3-0.6B-Base.
    """
    from .attacks import find_folder, find_passes, load_pass
    ids, target = load_pass(find_passes(), "bumas_text")
    _, qwen = load_pass(find_passes(), "qwen_text")
    folder = find_folder("shadow0_text.npz")
    halves = pd.read_csv(folder / "shadow_halves.csv").set_index("id")["half"]
    shadows = [load_pass(folder, f"shadow{h}_text")[1] for h in (0, 1)]
    forum = set(records.loc[records["source_type"] == FORUM, "id"])
    rows, keep = [], []
    for i, rid in enumerate(ids):
        if rid not in forum:
            continue
        t, q, o = target[i]["lp"], qwen[i]["lp"], shadows[1 - halves[rid]][i]["lp"]
        g, d = t - o, t - q
        keep.append(rid)
        rows.append({"log_n": np.log(len(t)), "g_sum": g.sum(), "g_t": t_stat(g), "d_sum": d.sum(), "d_t": t_stat(d),
                     "out_lp": o.mean(), "g_hard": g[o < np.quantile(o, 0.25)].mean(), "g_easy": g[o >= np.quantile(o, 0.25)].mean()})
    return pd.DataFrame(rows, index=keep).reindex(records["id"]).set_axis(records.index)


LENGTH = ["log_n", "g_sum", "g_t", "d_sum", "d_t", "out_lp", "g_hard", "g_easy"]


def forum_shadow_features(records: pd.DataFrame, splits=range(4)) -> pd.DataFrame:
    """LiRA features of the forum shadow models: IN and OUT means and variances of each token.

    online: the log-likelihood ratio of two Gaussians (IN and OUT) for each token, with one variance of the record.
    offline: the z-score of the target against the OUT models only.
    """
    from .attacks import find_folder, find_passes, load_pass
    folder = find_folder("fshadow_s0_h0_text.npz")
    ids, models = None, {}
    for s in splits:
        for h in (0, 1):
            m_ids, models[s, h] = load_pass(folder, f"fshadow_s{s}_h{h}_text")
            assert ids is None or ids.equals(m_ids)
            ids = m_ids
    halves = {s: pd.read_csv(folder / f"forum_halves_s{s}.csv").set_index("id")["half"] for s in splits}
    t_ids, target = load_pass(find_passes(), "bumas_text")
    target = dict(zip(t_ids, target))
    rows = []
    for i, rid in enumerate(ids):
        t = target[rid]["lp"]
        lp_in = np.array([models[s, halves[s][rid]][i]["lp"] for s in splits])
        lp_out = np.array([models[s, 1 - halves[s][rid]][i]["lp"] for s in splits])
        mu_in, mu_out = lp_in.mean(0), lp_out.mean(0)
        var_out = lp_out.var(0, ddof=1).mean() + 1e-3
        var = np.concatenate([lp_in.var(0, ddof=1), lp_out.var(0, ddof=1)]).mean() + 1e-3
        g, m = t - mu_out, mu_in - mu_out
        rows.append({"f_gain": g.mean(), "f_gain_t": t_stat(g), "f_m": m.mean(),
                     "f_online": float((m * (2 * g - m)).mean() / (2 * var)),
                     "f_offline": float((g / np.sqrt(var_out)).mean()),
                     "f_slope": float((g * m).sum() / max((m * m).sum(), 1e-6)),
                     "f_low20": float(np.sort(g)[:max(1, len(g) // 5)].mean())})
    return pd.DataFrame(rows, index=ids).reindex(records["id"]).set_axis(records.index)


FSHADOW = ["f_gain", "f_gain_t", "f_m", "f_online", "f_offline", "f_slope", "f_low20"]


def forum_cv(known: pd.DataFrame, cols, folds: dict, C: float = 1.0) -> pd.DataFrame:
    """CV of the forum record score with the baseline folds (src/validation.py on all known records)."""
    from .photographer import metrics_row, oof_llr
    rows = []
    for seed, fold in folds.items():
        llr = oof_llr(known, cols, fold, C)
        f_known = fold.loc[known.index].to_numpy()
        for f in range(2):
            ev = known[f_known == f]
            rows.append({"seed": seed, "fold": f, **metrics_row(llr[ev.index].to_numpy(), ev["member"])})
    return pd.DataFrame(rows)


def forum_logodds(data: pd.DataFrame, cols, C: float = 1.0) -> pd.DataFrame:
    """Calibrated log-odds of all forum posts (id, logodds): known out of fold (folds of seed 0), test fit on all known.

    The regression output is the log-odds with the known member rate of forum posts as the prior.
    """
    from .photographer import baseline_folds, fit_llr, logit, oof_llr
    part = data[data["source_type"] == FORUM]
    known, test = part[part["split"] == "known"], part[part["split"] == "test"]
    prior = logit(known["member"].mean())
    lo_known = oof_llr(known, cols, baseline_folds(data, 0), C) + prior
    lo_test = pd.Series(fit_llr(known, test, cols, C) + prior, index=test.index)
    return pd.DataFrame({"id": part["id"], "logodds": pd.concat([lo_known, lo_test]).loc[part.index]})


def explore(out_dir: str, variants: dict, extra=()) -> None:
    """Forum record variants: single features, CV with the baseline folds, and the log-odds file of the first variant.

    variants maps a name to (columns, C). extra lists the feature functions to add (token_features, forum_shadow_features).
    """
    from .attacks import v6_columns_of
    from .photographer import baseline_folds, load_data, metrics_row, summary
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    data = load_data()
    data = pd.concat([data] + [fn(data) for fn in extra], axis=1)
    folds = {seed: baseline_folds(data, seed) for seed in range(5)}
    known = data[(data["source_type"] == FORUM) & (data["split"] == "known")].copy()
    known["member"] = known["member"].astype(int)
    new = [c for c in data.columns if c in LENGTH + FSHADOW]
    single = pd.DataFrame([{"feature": c, **metrics_row(known[c].to_numpy(), known["member"])}
                           for c in ["lira_gain"] + new]).round(3)
    rows = pd.concat([forum_cv(known, cols(v6_columns_of(FORUM)), folds, C).assign(variant=name)
                      for name, (cols, C) in variants.items()])
    cv = summary(rows)
    # The paired difference against the first variant on the same folds is more sensitive than two separate stds.
    first = rows[rows["variant"] == next(iter(variants))].set_index(["seed", "fold"])["TPR@5%FPR"]
    diff = rows.set_index(["seed", "fold"]).groupby("variant", sort=False)["TPR@5%FPR"].apply(lambda v: v - first)
    cv["paired diff mean"] = diff.groupby("variant", sort=False).mean().round(3).to_numpy()
    cv["paired diff std"] = diff.groupby("variant", sort=False).std().round(3).to_numpy()
    for file, table in [("single", single), ("cv", cv)]:
        table.to_csv(out / f"{file}.csv", index=False)
        print(file, "\n", table.to_string(), flush=True)
    name, (cols, C) = next(iter(variants.items()))
    forum_logodds(data, cols(v6_columns_of(FORUM)), C).to_csv(out / "submission_forum_logodds.csv", index=False)


# Variants of the job forum-v1: one change each against v6 (C 1).
V1 = {
    "v6": (lambda base: base, 1.0),
    "v6 C=0.1": (lambda base: base, 0.1),
    "v6 + length": (lambda base: base + LENGTH, 1.0),
    "v6 + length C=0.1": (lambda base: base + LENGTH, 0.1),
}


def explore_v1(out_dir: str) -> None:
    explore(out_dir, V1, extra=(token_features,))
