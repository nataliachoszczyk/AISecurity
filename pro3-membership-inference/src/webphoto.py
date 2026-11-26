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


GROUP = 32  # photos of one ranking group: each caption is scored with each photo of its group
PAIR_BATCH = 32
OWN_BATCH = 8  # the pass with all logits and hidden states needs more memory
ALPHAS = (0.5, 1.0, 2.0, np.inf)  # orders of the Renyi entropy (MaxRenyi-K%, Li et al. 2024)
LAYERS = (14, 28)


def ranking_groups(n: int, seed: int = 0) -> list[np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, max(1, int(np.ceil(n / GROUP))))


def batch_inputs(model, img, encoded, device):
    """Input embeddings with the image tokens in place, the attention mask and the caption labels."""
    import torch
    pad = model.tok.pad_token_id
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in encoded], device=device)
    labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in encoded], device=device)
    mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in encoded], device=device)
    emb = model.llm.get_input_embeddings()(input_ids)
    emb = emb.masked_scatter((input_ids == model.image_pad_id).unsqueeze(-1), img.to(emb.dtype).reshape(-1, emb.shape[-1]))
    return input_ids, emb, mask, labels


def caption_lp(model, img, encoded, device):
    """Sum of log p of the caption tokens for each (image, caption) pair. The lm head runs only on the caption positions."""
    import torch
    _, emb, mask, labels = batch_inputs(model, img, encoded, device)
    h = model.llm.model(inputs_embeds=emb, attention_mask=mask).last_hidden_state[:, :-1]
    target = labels[:, 1:]
    sel = target != -100
    logp = torch.log_softmax(model.llm.lm_head(h[sel]).float(), -1).gather(-1, target[sel][:, None])[:, 0]
    rows = sel.nonzero()[:, 0]
    return torch.zeros(len(encoded), device=device).index_add_(0, rows, logp).cpu().numpy()


def renyi(logp, alpha: float):
    import torch
    if alpha == 1.0:
        return -(logp.exp() * logp).sum(-1)
    if np.isinf(alpha):
        return -logp.max(-1).values
    return torch.logsumexp(alpha * logp, -1) / (1 - alpha)


def own_pass(model, img, encoded, device) -> tuple[list[dict], dict]:
    """Target-free entropies of the image positions and of the caption positions, and the mean hidden states."""
    import torch
    input_ids, emb, mask, labels = batch_inputs(model, img, encoded, device)
    out = model.llm(inputs_embeds=emb, attention_mask=mask, output_hidden_states=True)
    image_pos = input_ids == model.image_pad_id
    caption_pos = labels != -100
    rows, hidden = [], {f"{part}_l{layer}": [] for part in ("img", "cap") for layer in LAYERS}
    for b in range(len(encoded)):
        n = int(mask[b].sum())
        logp = torch.log_softmax(out.logits[b, :n].float(), -1)
        predicts_caption = torch.zeros(n, dtype=torch.bool, device=logp.device)
        predicts_caption[:n - 1] = caption_pos[b, 1:n]
        regions = {"img": image_pos[b, :n], "cap": predicts_caption}
        row = {}
        for region, sel in regions.items():
            for alpha in ALPHAS:
                h = renyi(logp[sel], alpha)
                row[f"renyi_{region}_{alpha}"] = h.mean().item()
                row[f"renyi_{region}_{alpha}_max"] = h.max().item()
        rows.append(row)
        for layer in LAYERS:
            hs = out.hidden_states[layer][b].float()
            hidden[f"img_l{layer}"].append(hs[image_pos[b]].mean(0).cpu().numpy())
            hidden[f"cap_l{layer}"].append(hs[caption_pos[b]].mean(0).cpu().numpy())
    return rows, hidden


def run_matrix(source_type: str, device: str, out_dir: str, limit: int | None = None) -> None:
    """For each ranking group: log p of each caption with each photo, the own caption with the flipped photo,
    and the target-free features of the own pairs. One output set for each source type."""
    import time
    import traceback

    import torch
    from .data import find_root, load_records
    from .logprobs import import_openbumas
    torch.cuda.set_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = source_type.replace(" ", "_")
    try:
        root = find_root()
        ob = import_openbumas(root)
        records = load_records(root)
        photos = records[records["source_type"] == source_type].reset_index(drop=True)
        if limit:
            photos = photos.head(limit)
        model = ob.BUMASMini.load(root / "model", dtype=torch.float32).to(device).eval()
        encoded = [ob.caption_example(model.tok, c) for c in photos["caption"].fillna("")]
        pairs, own_rows, conn = [], [], []
        hidden = None
        t0 = time.time()
        with torch.no_grad():
            for g, members in enumerate(ranking_groups(len(photos))):
                pixels = torch.stack([ob.load_image(p) for p in photos["image_path"].iloc[members]]).to(device)
                img = model.encode_images(pixels)
                img_flip = model.encode_images(pixels.flip(-1))
                ii, jj = [a.ravel() for a in np.meshgrid(np.arange(len(members)), np.arange(len(members)), indexing="ij")]
                for s in range(0, len(ii), PAIR_BATCH):
                    i, j = ii[s:s + PAIR_BATCH], jj[s:s + PAIR_BATCH]
                    lp = caption_lp(model, img[i], [encoded[members[k]] for k in j], device)
                    pairs += [(g, members[a], members[b], v) for a, b, v in zip(i, j, lp)]
                own_enc = [encoded[m] for m in members]
                flip = np.concatenate([caption_lp(model, img_flip[s:s + PAIR_BATCH], own_enc[s:s + PAIR_BATCH], device)
                                       for s in range(0, len(members), PAIR_BATCH)])
                for s in range(0, len(members), OWN_BATCH):
                    rows, h = own_pass(model, img[s:s + OWN_BATCH], own_enc[s:s + OWN_BATCH], device)
                    own_rows += rows
                    hidden = h if hidden is None else {k: hidden[k] + h[k] for k in h}
                conn += list(img.float().mean(1).cpu().numpy())
                for a, m in enumerate(members):
                    own_rows[-len(members) + a].update(id=photos["id"].iloc[m], group=g, n_tokens=sum(x != -100 for x in encoded[m][1]),
                                                       lp_flip=float(flip[a]))
                if g % 10 == 0:
                    print(f"{source_type}: group {g}, {time.time() - t0:.0f} s", flush=True)
        ids = photos["id"].to_numpy()
        pd.DataFrame([(ids[i], ids[j], g, v) for g, i, j, v in pairs], columns=["image_id", "caption_id", "group", "lp"]).to_csv(
            out / f"pairs_{tag}.csv", index=False)
        pd.DataFrame(own_rows).to_csv(out / f"own_{tag}.csv", index=False)
        order = [r["id"] for r in own_rows]
        np.savez(out / f"hidden_{tag}.npz", ids=np.asarray(order), conn=np.asarray(conn, dtype=np.float16),
                 **{k: np.asarray(v, dtype=np.float16) for k, v in hidden.items()})
        print(f"{source_type}: done in {time.time() - t0:.0f} s", flush=True)
    except Exception:
        traceback.print_exc()
        (out / f"error_{tag}.txt").write_text(traceback.format_exc())


def matrix_features(folder: Path, tag: str) -> pd.DataFrame:
    """Ranking features of each photo record from the pairs of its group (index: record id).

    L[i, j] is the sum of log p of caption j with photo i. For the own pair (a, a):
    - rank_img_*: the own photo against the other photos of the group, for the own caption,
    - rank_cap_*: the own caption against the other captions, for the own photo, after the subtraction of the mean of
      each caption over all photos (an image-caption PMI),
    - flip_drop: the loss of log p with the flipped own photo.
    """
    pairs = pd.read_csv(folder / f"pairs_{tag}.csv")
    own = pd.read_csv(folder / f"own_{tag}.csv").set_index("id")
    rows = {}
    for _, part in pairs.groupby("group"):
        L = part.pivot(index="image_id", columns="caption_id", values="lp")
        ids = L.index.intersection(L.columns)
        L = L.loc[ids, ids].to_numpy()
        P = L - L.mean(0, keepdims=True)
        for a, rid in enumerate(ids):
            n = own.loc[rid, "n_tokens"]
            others = np.delete(L[:, a], a)
            cap = np.delete(P[a], a)
            rows[rid] = {"rank_own": L[a, a] / n,
                         "rank_img_gain": (L[a, a] - others.mean()) / n,
                         "rank_img_z": (L[a, a] - others.mean()) / (others.std() + 1e-6),
                         "rank_img_pct": float((others < L[a, a]).mean()),
                         "rank_cap_z": (P[a, a] - cap.mean()) / (cap.std() + 1e-6),
                         "rank_cap_pct": float((cap < P[a, a]).mean()),
                         "flip_drop": (L[a, a] - own.loc[rid, "lp_flip"]) / n}
    feats = pd.DataFrame.from_dict(rows, orient="index")
    renyi_cols = [c for c in own.columns if c.startswith("renyi_")]
    return feats.join(-own[renyi_cols])  # a lower entropy suggests a member


RANK = ["rank_img_gain", "rank_img_z", "rank_img_pct", "rank_cap_z", "rank_cap_pct", "flip_drop"]


def load_hidden(folder: Path, tag: str) -> dict:
    z = np.load(folder / f"hidden_{tag}.npz", allow_pickle=True)  # the ids are strings; the file comes from our own job
    ids = pd.Index(z["ids"])
    return {k: pd.DataFrame(z[k].astype(np.float32), index=ids) for k in z.files if k != "ids"}


def probe_score(vectors: pd.DataFrame, C: float = 0.01, components: int = 64, extra_fit=None):
    """Linear probe on a hidden vector of each record (Revisiting Data Auditing in LVLMs, 2025)."""
    from sklearn.decomposition import PCA

    def score(fit, evaluate):
        train = fit if extra_fit is None else pd.concat([fit, extra_fit])
        model = make_pipeline(StandardScaler(), PCA(components, random_state=0), LogisticRegression(C=C, max_iter=3000))
        model.fit(vectors.loc[train["id"]].to_numpy(), train["member"])
        return model.decision_function(vectors.loc[evaluate["id"]].to_numpy())
    return score


def stacked(columns, inner_fn, C: float = 1.0, name: str = "stack"):
    """Logistic regression on the columns and on the out-of-fold score of a second model (inner_fn)."""
    def score(fit, evaluate):
        fit, evaluate = fit.copy(), evaluate.copy()
        inner = make_folds(fit, 2, seed=123)
        fit[name] = 0.0
        for f in range(2):
            fit.loc[fit.index[inner == f], name] = inner_fn(fit[inner != f], fit[inner == f])
        evaluate[name] = inner_fn(fit, evaluate)
        return lr_score(columns + [name], C)(fit, evaluate)
    return score


def matrix_data(records: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    from .attacks import find_folder
    folder = find_folder("pairs_web_photo.csv")
    data = photo_data(records)
    data = pd.concat([data, duplicate_features(data)], axis=1)
    feats = pd.concat([matrix_features(folder, t) for t in ("web_photo", "photographer")])
    return data.join(feats, on="id"), folder


def explore_v2(out_dir: str) -> None:
    """The ranking features, the target-free entropies and the probes on the hidden states."""
    from .data import find_root, load_records
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    data, folder = matrix_data(load_records(find_root()))
    known_all = data[data["split"] == "known"].assign(member=lambda d: d["member"].astype(int))
    known = known_all[known_all["source_type"] == WEB]
    photographers = known_all[known_all["source_type"] == "photographer"]
    print("check: corr of rank_own and loss", np.corrcoef(known["rank_own"], known["loss"])[0, 1], flush=True)
    renyi_cols = [c for c in data.columns if c.startswith("renyi_")]
    rows = [single_table(part, RANK + renyi_cols).assign(source_type=t) for t, part in
            [(WEB, known), ("photographer", photographers)]]
    single = pd.concat(rows)
    single.to_csv(out / "single.csv", index=False)
    print(single.to_string(), flush=True)
    hidden = {k: pd.concat([v, load_hidden(folder, "photographer")[k]]) for k, v in load_hidden(folder, "web_photo").items()}
    variants = {
        "v8a base": lr_score(BASE),
        "base + rank": lr_score(BASE + RANK),
        "base + rank_img_z": lr_score(BASE + ["rank_img_z"]),
        "base + rank_cap_z": lr_score(BASE + ["rank_cap_z"]),
        "base + flip": lr_score(BASE + ["flip_drop"]),
        "base + renyi": lr_score(BASE + renyi_cols),
        "rank alone": lr_score(RANK),
        "renyi alone": lr_score(renyi_cols),
    }
    for key, vectors in hidden.items():
        variants[f"probe {key}"] = probe_score(vectors)
        variants[f"probe {key} pooled"] = probe_score(vectors, extra_fit=photographers)
    table = run_cv(known, variants)
    best = table[table["variant"].str.startswith("probe")].sort_values("AUC mean").iloc[-1]["variant"]
    key = best.split()[1]
    pooled = photographers if best.endswith("pooled") else None
    extra = run_cv(known, {f"base + {best}": stacked(BASE, probe_score(hidden[key], extra_fit=pooled))})
    pd.concat([table, extra]).to_csv(out / "cv.csv", index=False)
    data[["id", "source_type", "split", "member"] + RANK + renyi_cols].to_csv(out / "features.csv", index=False)
