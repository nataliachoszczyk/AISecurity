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



def owner_groups(data: pd.DataFrame) -> pd.Series:
    """Group of each record: single, victim, full (known records as a victim, no member) or other."""
    known = data[data["split"] == "known"]
    members = known.groupby("source")["member"].sum()
    n_known = known.groupby("source").size()
    full = n_known.groupby(known.groupby("source")["source_type"].first()).transform("max")
    group = pd.Series("other", index=members.index).mask(n_known == full, "full").mask(members > 0, "victim")
    return data["source"].map(group).where(data["source_type"].isin(SET_TYPES), "single")



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


def v6_features(records: pd.DataFrame, shadow_names=("shadow",)) -> pd.DataFrame:
    """Features from the domain reference models (job refmodel) and the shadow models (jobs shadow and shadow-b).

    With more than one pair of shadow models, the IN and the OUT log p are the means over the pairs.
    """
    from .validation import make_folds
    base, ref_dir = find_passes(), find_folder("ref0_text.npz")
    shadow_dirs = {n: find_folder(f"{n}0_text.npz") for n in shadow_names}
    by_id = records.set_index("id")
    known = records[records["split"] == "known"]
    ref_fold = pd.Series(make_folds(known, 2, seed=0), index=known["id"])  # the folds of the job refmodel
    halves = {n: pd.read_csv(d / f"{n}_halves.csv").set_index("id")["half"] for n, d in shadow_dirs.items()}
    tables = []
    for kind, target_names in [("text", ["bumas_text"]), ("caption", ["bumas_caption_text", "bumas_photo_own"])]:
        suffix = "text" if kind == "text" else "caption_text"
        ids, ref = zip(*[load_pass(ref_dir, f"ref{f}_{suffix}") for f in (0, 1)])
        shadows = {n: [load_pass(d, f"{n}{h}_{suffix}")[1] for h in (0, 1)] for n, d in shadow_dirs.items()}
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
            lp_in = np.mean([shadows[n][halves[n][rid]][i]["lp"] for n in shadow_dirs], axis=0)
            lp_out = np.mean([shadows[n][1 - halves[n][rid]][i]["lp"] for n in shadow_dirs], axis=0)
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


def explore_v8(out_dir: str, shadow_names=("shadow",)) -> None:
    """Record score with the v6 features (means over the shadow pairs): validation, expected members in test, submission."""
    from .data import find_root, load_records
    from .validation import cross_validate, summarize
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    records = load_records(find_root())
    data = pd.concat([records, all_features(find_passes(), records), v6_features(records, shadow_names)], axis=1)
    known = data[data["split"] == "known"].copy()
    known["member"] = known["member"].astype(int)
    results = cross_validate(known, logistic_score(v6_columns_of))
    table = summarize(results)
    table.to_csv(out / "cv_v8.csv", index=False)
    print(table.round(3).to_string(), flush=True)
    score = record_logodds(data, data.index[data["split"] == "known"], v6_columns_of)
    test = data["split"] == "test"
    expected = data.loc[test].assign(group=owner_groups(data)[test], p=1 / (1 + np.exp(-score[test])))
    expected = expected.groupby(["source_type", "group"])["p"].agg(records="size", expected_members="sum").round(1)
    expected.to_csv(out / "test_expected_v8.csv")
    print(expected.to_string(), flush=True)
    pd.DataFrame({"id": data.loc[test, "id"], "score": score[test]}).to_csv(out / "submission_v8.csv", index=False)
