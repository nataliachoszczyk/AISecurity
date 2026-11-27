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


def forum_shadow_features(records: pd.DataFrame, max_splits: int | None = None, suffix: str = "") -> pd.DataFrame:
    """LiRA features of the forum shadow models: IN and OUT means and variances of each token.

    online: the log-likelihood ratio of two Gaussians (IN and OUT) for each token, with one variance of the record.
    offline: the z-score of the target against the OUT models only.

    offline_tok: as offline, with a variance for each token (shrunk to the variance of the record).

    The splits can come from more than one job output. max_splits keeps only the first splits.
    suffix goes after each column name, so that two split counts can be in one table.
    """
    from .attacks import find_passes, load_pass
    folders = {int(f.stem.split("_s")[1]): f.parent for f in Path("/kaggle/input").rglob("forum_halves_s*.csv")}
    splits = sorted(folders)[:max_splits]
    print("forum shadow splits", splits, flush=True)
    ids, models = None, {}
    for s in splits:
        for h in (0, 1):
            m_ids, models[s, h] = load_pass(folders[s], f"fshadow_s{s}_h{h}_text")
            assert ids is None or ids.equals(m_ids)
            ids = m_ids
    halves = {s: pd.read_csv(folders[s] / f"forum_halves_s{s}.csv").set_index("id")["half"] for s in splits}
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
                     "f_offline_tok": float((g / np.sqrt(lp_out.var(0, ddof=1) + var_out)).mean()),
                     "f_slope": float((g * m).sum() / max((m * m).sum(), 1e-6)),
                     "f_low20": float(np.sort(g)[:max(1, len(g) // 5)].mean())})
    table = pd.DataFrame(rows, index=ids).add_suffix(suffix)
    return table.reindex(records["id"]).set_axis(records.index)


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
    new = [c for c in data.columns if c in LENGTH + RECALL or c.startswith("f_")]
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


PREFIX_POSTS = 4  # known non-member forum posts in the ReCaLL prefix


def prefix_ids(records: pd.DataFrame) -> list[str]:
    """Ids of the known non-member forum posts of the ReCaLL prefix (a fixed random choice)."""
    pool = records[(records["source_type"] == FORUM) & (records["split"] == "known") & (records["member"] == 0)]
    return pool.sample(PREFIX_POSTS, random_state=0)["id"].tolist()


def run_recall(model_name: str, device: str, out_dir: str, limit: int | None = None) -> None:
    """ReCaLL (Xie et al. 2024): log p of each forum post after a prefix of known non-member forum posts.

    model_name: 'bumas' (the LLM of BUMAS-mini) or 'qwen' (the reference model, for the calibration).
    The tokens of the post have the same format as in src/logprobs.py, with the prefix and EOT before them.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .logprobs import import_openbumas, reference_path, save, token_stats
    torch.cuda.set_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        root = find_root()
        records = load_records(root)
        forum = records[records["source_type"] == FORUM]
        tok = AutoTokenizer.from_pretrained(root / "model" / "llm")
        if model_name == "bumas":
            llm = import_openbumas(root).BUMASMini.load(root / "model", dtype=torch.float32).llm.to(device).eval()
        else:
            llm = AutoModelForCausalLM.from_pretrained(reference_path(), dtype=torch.float32).to(device).eval()
        chosen = prefix_ids(records)
        pd.Series(chosen, name="id").to_csv(out / "recall_prefix.csv", index=False)
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        prefix = [eot]
        for text in forum.set_index("id").loc[chosen, "content"]:
            prefix += tok(text.strip() + "<|endoftext|>", add_special_tokens=False)["input_ids"]
        score = forum.head(limit) if limit else forum
        rows, t0 = [], time.time()
        with torch.no_grad():
            for i, text in enumerate(score["content"]):
                post = tok(text.strip() + "<|endoftext|>", add_special_tokens=False)["input_ids"][:2048 - len(prefix)]
                ids = torch.tensor(prefix + post, device=device)
                logits = llm(input_ids=ids[None]).logits[0, len(prefix) - 1:-1]
                rows.append(token_stats(logits, ids[len(prefix):]))
                if i % 500 == 0:
                    print(f"recall {model_name}: {i}/{len(score)} records, {time.time() - t0:.0f} s", flush=True)
        save(out / f"recall_{model_name}_text.npz", score["id"], rows)
    except Exception:
        traceback.print_exc()
        (out / f"error_{model_name}.txt").write_text(traceback.format_exc())


def recall_features(records: pd.DataFrame) -> pd.DataFrame:
    """ReCaLL features: the change of the mean log p of a post after the non-member prefix, for BUMAS-mini and Qwen.

    The posts of the prefix get the median value, because the prefix contains them.
    """
    from .attacks import find_folder, find_passes, load_pass
    folder = find_folder("recall_bumas_text.npz")
    out = {}
    for key, cond_name, plain_name in [("t", "recall_bumas_text", "bumas_text"), ("q", "recall_qwen_text", "qwen_text")]:
        ids, cond = load_pass(folder, cond_name)
        p_ids, plain = load_pass(find_passes(), plain_name)
        plain = dict(zip(p_ids, plain))
        c = np.array([r["lp"].mean() for r in cond])
        u = np.array([plain[rid]["lp"][:len(r["lp"])].mean() for rid, r in zip(ids, cond)])
        out[f"recall_{key}"] = pd.Series(c - u, index=ids)
        out[f"recall_ratio_{key}"] = pd.Series(c / u, index=ids)
    table = pd.DataFrame(out)
    table["recall_diff"] = table["recall_t"] - table["recall_q"]
    prefix = pd.read_csv(folder / "recall_prefix.csv")["id"]
    is_prefix = table.index.isin(prefix)
    for col in table.columns:
        table.loc[is_prefix, col] = table.loc[~is_prefix, col].median()
    return table.reindex(records["id"]).set_axis(records.index)


RECALL = ["recall_t", "recall_ratio_t", "recall_q", "recall_ratio_q", "recall_diff"]


# Variants of the job forum-v2: one change each against v6 (C 1).
V2 = {
    "v6": (lambda base: base, 1.0),
    "v6 + forum shadow": (lambda base: base + FSHADOW, 1.0),
    "v6 + recall": (lambda base: base + RECALL, 1.0),
    "v6 + length": (lambda base: base + LENGTH, 1.0),
    "v6 + forum shadow + length": (lambda base: base + FSHADOW + LENGTH, 1.0),
}


def explore_v2(out_dir: str) -> None:
    explore(out_dir, V2, extra=(token_features, forum_shadow_features, recall_features))


def feature_smoke(out_dir: str) -> None:
    """Check the feature functions of the forum shadow models and ReCaLL on the outputs of the smoke jobs."""
    from .photographer import load_data
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_data()
    for fn in (forum_shadow_features, recall_features):
        table = fn(data).dropna(how="all")
        print(fn.__name__, len(table), "\n", table.describe().round(3).to_string(), flush=True)
        table.describe().round(3).to_csv(out / f"{fn.__name__}.csv")


def explore_v2a(out_dir: str) -> None:
    """The variants of forum-v2 without the forum shadow models (their job runs longer)."""
    variants = {k: v for k, v in V2.items() if "forum shadow" not in k}
    explore(out_dir, variants, extra=(token_features, recall_features))


def load_forum_data(shadow_names=("shadow",), extra=()) -> pd.DataFrame:
    """All records with the features of attacks.py (v6 with the given shadow pairs) and the extra feature functions."""
    from .attacks import all_features, find_passes, owner_groups, v6_features
    records = load_records(find_root())
    data = pd.concat([records, all_features(find_passes(), records), v6_features(records, shadow_names)], axis=1)
    data["group"] = owner_groups(data)
    data = data.drop(columns=["text", "caption", "title", "content", "image"], errors="ignore")
    return pd.concat([data] + [fn(data) for fn in extra], axis=1)


# Variants of the job forum-v3: the best model of forum-v2 first, then one change each.
STRONG = ["f_gain", "f_gain_t", "f_offline"]
V3 = {
    "v6 + forum shadow + length": (lambda base: base + FSHADOW + LENGTH, 1.0),
    "v6 + forum shadow + length C=0.1": (lambda base: base + FSHADOW + LENGTH, 0.1),
    "v6 + strong forum shadow + length": (lambda base: base + STRONG + LENGTH, 1.0),
    "compact": (lambda base: ["lira_gain", "dom", "ref", "ref_minkpp", "minkpp20"] + FSHADOW + LENGTH, 1.0),
}


def explore_v3(out_dir: str) -> None:
    """The V3 variants, then the global simulation of explore_v10 without and with the forum file of the first variant."""
    from .attacks import explore_v10
    explore(out_dir, V3, extra=(token_features, forum_shadow_features))
    for name, files in [("global-v6", None), ("global-forum", {FORUM: f"{out_dir}/submission_forum_logodds.csv"})]:
        explore_v10(f"{out_dir}/{name}", ("shadow",), single_files=files, single_C={"web photo": 0.01})


# Variants of the job forum-v4 with the 8 splits of forum-shadow and forum-shadow-b. The suffix _4 marks the first 4 splits.
V4 = {
    "v6 + strong (8 splits) + length": (lambda base: base + STRONG + LENGTH, 1.0),
    "v6 + strong (4 splits) + length": (lambda base: base + [c + "_4" for c in STRONG] + LENGTH, 1.0),
    "v6 + strong + offline per token (8 splits) + length": (lambda base: base + STRONG + ["f_offline_tok"] + LENGTH, 1.0),
    "v6 + all forum shadow (8 splits) + length": (lambda base: base + FSHADOW + ["f_offline_tok"] + LENGTH, 1.0),
}


def explore_v4(out_dir: str) -> None:
    """The V4 variants, then the global simulation of explore_v10 with the forum file of the first variant."""
    from functools import partial
    from .attacks import explore_v10
    four = partial(forum_shadow_features, max_splits=4, suffix="_4")
    explore(out_dir, V4, extra=(token_features, forum_shadow_features, four))
    explore_v10(f"{out_dir}/global-forum", ("shadow",), single_files={FORUM: f"{out_dir}/submission_forum_logodds.csv"},
                single_C={"web photo": 0.01})


FINAL_EXTRA = (token_features, forum_shadow_features)


def final_columns(base: list[str]) -> list[str]:
    """The forum model: the v6 columns, the three strong forum shadow features and the length features (job forum-v3)."""
    return base + STRONG + LENGTH


def forum_module(out_dir: str, shadow_names=("shadow",)) -> pd.DataFrame:
    """The final forum module: calibrated log-odds of all forum posts (id, split, logodds) and its CV.

    It needs the passes of attacks.py, the shadow pairs of shadow_names and the outputs of the jobs forum-shadow and
    forum-shadow-b (account nataliachoszczyk). Known: out of fold (folds of seed 0). Test: fit on all known.
    """
    from .attacks import v6_columns_of
    from .photographer import baseline_folds, summary
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_forum_data(shadow_names, FINAL_EXTRA)
    cols = final_columns(v6_columns_of(FORUM))
    known = data[(data["source_type"] == FORUM) & (data["split"] == "known")].copy()
    known["member"] = known["member"].astype(int)
    cv = summary(forum_cv(known, cols, {seed: baseline_folds(data, seed) for seed in range(5)}).assign(variant="forum module"))
    cv.to_csv(out / "cv.csv", index=False)
    print(cv.to_string(), flush=True)
    result = forum_logodds(data, cols).merge(data[["id", "split"]], on="id")[["id", "split", "logodds"]]
    assert len(result) == (data["source_type"] == FORUM).sum() and np.isfinite(result["logodds"]).all()
    result[["id", "logodds"]].to_csv(out / "submission_forum_logodds.csv", index=False)
    return result
