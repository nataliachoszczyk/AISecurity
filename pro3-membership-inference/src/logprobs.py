"""Per-token log-probabilities of BUMAS-mini (target) and Qwen3-0.6B-Base (reference) for all records.

Each pass writes one npz file: the per-token arrays of all records one after the other, and `offsets` and `ids`.
Token values of record i are at offsets[i]:offsets[i+1]. All arrays are float32.

Per token, a pass keeps:
- lp: log p of the true token,
- mu, sigma: the mean and the standard deviation of log p over the vocabulary (for Min-K%++),
- kl: KL(p_own || p_noise) for the passes with a noised photo.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .data import find_root, load_records

REFERENCE = "qwen3-0.6b-base"  # the output folder of the job qwen-weights (Qwen/Qwen3-0.6B-Base)
WINDOW, STRIDE = 2048, 1024  # S1 used pieces of 2048 tokens. The stride gives each token at least 1024 tokens of context.
NOISE = {"noise02": 0.02, "noise10": 0.10}  # standard deviation of the pixel noise, in the [0, 1] pixel scale
PHOTO_BATCH = 16


def reference_path(base: str = "/kaggle/input") -> Path:
    matches = [p.parent for p in Path(base).rglob("config.json") if p.parent.name == REFERENCE]
    if len(matches) != 1:
        raise FileNotFoundError(f"need exactly one {REFERENCE} folder below {base}, found {matches}")
    return matches[0]


def import_openbumas(root: Path):
    sys.path.insert(0, str(root / "model"))
    import openbumas
    return openbumas


def token_stats(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 512) -> dict:
    """lp, mu and sigma for each position. Chunks keep the 152k-vocabulary tensors small."""
    out = {"lp": [], "mu": [], "sigma": []}
    for s in range(0, len(targets), chunk):
        logp = torch.log_softmax(logits[s:s + chunk].float(), dim=-1)
        p = logp.exp()
        mu = (p * logp).sum(-1)
        out["lp"].append(logp.gather(-1, targets[s:s + chunk, None])[:, 0])
        out["mu"].append(mu)
        out["sigma"].append(((p * logp.square()).sum(-1) - mu.square()).clamp_min(0).sqrt())
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


@torch.no_grad()
def text_stats(llm, ids: list[int], device) -> dict:
    """Stats for the tokens ids[1:]. A long text goes in windows of WINDOW tokens with a step of STRIDE."""
    ids_t = torch.tensor(ids, device=device)
    parts, start = [], 0
    while True:
        end = min(start + WINDOW, len(ids))
        logits = llm(input_ids=ids_t[None, start:end]).logits[0, :-1]
        stats = token_stats(logits, ids_t[start + 1:end])
        keep = 0 if start == 0 else WINDOW - STRIDE - 1  # only the tokens that the earlier window did not score
        parts.append({k: v[keep:] for k, v in stats.items()})
        if end == len(ids):
            break
        start += STRIDE
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def text_ids(tok, eot_id: int, text: str) -> list[int]:
    # In S1 the previous document of the packed piece ends with EOT. Thus EOT is the context of the first token.
    return [eot_id] + tok(text.strip() + "<|endoftext|>", add_special_tokens=False)["input_ids"]


def save(path: Path, ids, rows: list[dict]) -> None:
    lengths = [len(r["lp"]) for r in rows]
    arrays = {k: np.concatenate([r[k] for r in rows]).astype(np.float32) for k in rows[0]}
    np.savez(path, ids=np.asarray(ids), offsets=np.concatenate([[0], np.cumsum(lengths)]), **arrays)
    print(f"saved {path.name}: {len(rows)} records, {sum(lengths)} tokens", flush=True)


def text_pass(llm, tok, texts, device, out: Path, name: str) -> None:
    eot_id = tok.convert_tokens_to_ids("<|endoftext|>")
    rows, t0 = [], time.time()
    for i, text in enumerate(texts["content"]):
        rows.append(text_stats(llm, text_ids(tok, eot_id, text), device))
        if i % 500 == 0:
            print(f"{name}: {i}/{len(texts)} records, {time.time() - t0:.0f} s", flush=True)
    save(out / f"{name}.npz", texts["id"], rows)


def caption_text_pass(llm, tok, photos, device, out: Path, name: str) -> None:
    """The caption as plain text, without the image: the reference model, or the LLM of BUMAS-mini alone."""
    text_pass(llm, tok, photos.assign(content=photos["caption"].fillna("")), device, out, name)


@torch.no_grad()
def photo_pass(model, ob, photos, device, out: Path, name: str, image_of=None, noise: float = 0.0) -> None:
    """Caption stats of BUMAS-mini with a photo, in the format of S3 (caption_example).

    image_of gives the image path for each record (default: its own photo).
    With noise, the pass also gives KL(p_own || p_noise) on each caption token.
    """
    tok = model.tok
    paths = list(photos["image_path"] if image_of is None else image_of)
    captions = photos["caption"].fillna("").tolist()
    rows, gen = [], torch.Generator().manual_seed(0)
    for s in range(0, len(photos), PHOTO_BATCH):
        encoded = [ob.caption_example(tok, c) for c in captions[s:s + PHOTO_BATCH]]
        width = max(len(ids) for ids, _ in encoded)
        pad = tok.pad_token_id
        input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in encoded], device=device)
        labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in encoded], device=device)
        mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in encoded], device=device)
        pixels = torch.stack([ob.load_image(p) for p in paths[s:s + PHOTO_BATCH]])
        logits = model(input_ids=input_ids, attention_mask=mask, pixel_values=pixels.to(device)).logits
        if noise:
            # load_image scales to [-1, 1], so the noise doubles. The clamp keeps valid pixels.
            noisy = (pixels + 2 * noise * torch.randn(pixels.shape, generator=gen)).clamp(-1, 1)
            logits_noise = model(input_ids=input_ids, attention_mask=mask, pixel_values=noisy.to(device)).logits
        for b in range(len(encoded)):
            sel = labels[b, 1:] != -100
            stats = token_stats(logits[b, :-1][sel], labels[b, 1:][sel])
            if noise:
                ln = logits_noise[b, :-1][sel]
                lo = torch.log_softmax(logits[b, :-1][sel].float(), -1)
                stats = {"lp": token_stats(ln, labels[b, 1:][sel])["lp"],
                         "kl": (lo.exp() * (lo - torch.log_softmax(ln.float(), -1))).sum(-1).cpu().numpy()}
            rows.append(stats)
    save(out / f"{name}.npz", photos["id"], rows)


def swapped_images(photos) -> list:
    """Give each photo record the photo of a different record. Both records have the same source type."""
    paths = photos["image_path"].copy()
    for _, idx in photos.groupby("source_type").groups.items():
        paths.loc[idx] = np.roll(photos.loc[idx, "image_path"].to_numpy(), 7)
    return paths.tolist()


def run(model_name: str, device: str, out_dir: str, limit: int | None = None) -> None:
    """model_name: 'bumas' or 'qwen'. limit: records of each kind (for the smoke test)."""
    torch.cuda.set_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        root = find_root()
        records = load_records(root)
        texts = records[records["kind"] == "text"]
        photos = records[records["kind"] == "photo"]
        if limit:
            texts = texts.groupby("source_type").head(limit // 4)
            photos = photos.groupby("source_type").head(limit // 2)
        ob = import_openbumas(root)
        if model_name == "bumas":
            model = ob.BUMASMini.load(root / "model", dtype=torch.float32).to(device).eval()
            caption_text_pass(model.llm, model.tok, photos, device, out, "bumas_caption_text")
            photo_pass(model, ob, photos, device, out, "bumas_photo_own")
            photo_pass(model, ob, photos, device, out, "bumas_photo_swap", image_of=swapped_images(photos))
            for name, sigma in NOISE.items():
                photo_pass(model, ob, photos, device, out, f"bumas_photo_{name}", noise=sigma)
            text_pass(model.llm, model.tok, texts, device, out, "bumas_text")
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(root / "model" / "llm")
            ref_tok = AutoTokenizer.from_pretrained(reference_path())
            sample = texts["content"].head(50).tolist()
            assert all(tok(t)["input_ids"] == ref_tok(t)["input_ids"] for t in sample), "the tokenizers differ"
            llm = AutoModelForCausalLM.from_pretrained(reference_path(), dtype=torch.float32).to(device).eval()
            caption_text_pass(llm, tok, photos, device, out, "qwen_caption_text")
            text_pass(llm, tok, texts, device, out, "qwen_text")
    except Exception:
        # A failed kernel loses all its output. Thus log the error and keep the passes that are complete.
        traceback.print_exc()
        (out / f"error_{model_name}.txt").write_text(traceback.format_exc())
