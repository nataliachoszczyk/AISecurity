"""A reference model for the domain: Qwen3-0.6B-Base, trained on the known non-members of one fold.

The general reference model does not know the domain of the records. Thus the target gains on members and on non-members.
This reference model learns the domain from records that are surely not members. Thus the gain over it shows memorization.
Two models (fold 0 and fold 1) give a cross-fit: a known record gets the scores of the model of the other fold.
"""
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .data import find_root, load_records
from .logprobs import caption_text_pass, reference_path, text_pass
from .validation import make_folds

CHUNK = 2048  # S1 used pieces of 2048 tokens
LR = 3e-5
EPOCHS = 2
ACCUMULATION = 8  # pieces for each optimizer step


def training_texts(records, fold: int) -> list[str]:
    """Known non-members of this fold: the texts and the captions (as plain text)."""
    known = records[records["split"] == "known"].copy()
    known["fold"] = make_folds(known, 2, seed=0)
    chosen = known[(known["fold"] == fold) & (known["member"] == 0)]
    return chosen["content"].tolist()


def pieces(tok, texts: list[str], seed: int) -> list[torch.Tensor]:
    """Join the texts in a random order, each with EOT at the end, and cut the stream into pieces of CHUNK tokens."""
    order = np.random.default_rng(seed).permutation(len(texts))
    stream = []
    for i in order:
        stream += tok(texts[i].strip() + "<|endoftext|>", add_special_tokens=False)["input_ids"]
    return [torch.tensor(stream[s:s + CHUNK]) for s in range(0, len(stream) - 1, CHUNK)]


def train(llm, tok, texts: list[str], device, epochs: int = EPOCHS) -> None:
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    llm.train()
    optimizer = torch.optim.AdamW(llm.parameters(), lr=LR, weight_decay=0.0)
    all_pieces = [pieces(tok, texts, seed=epoch) for epoch in range(epochs)]
    total = sum(math.ceil(len(p) / ACCUMULATION) for p in all_pieces)
    warmup = max(1, total // 10)
    # A linear warmup, then a cosine decay to 10% of LR.
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, 0.1 + 0.45 * (1 + math.cos(math.pi * s / total))))
    step, t0 = 0, time.time()
    for epoch, epoch_pieces in enumerate(all_pieces):
        for s in range(0, len(epoch_pieces), ACCUMULATION):
            batch = epoch_pieces[s:s + ACCUMULATION]
            for piece in batch:
                ids = piece[None].to(device)
                loss = llm(input_ids=ids, labels=ids).loss / len(batch)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(llm.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0 or step == total:
                print(f"epoch {epoch} step {step}/{total} loss {loss.item() * len(batch):.3f} {time.time() - t0:.0f} s", flush=True)
    llm.eval()
    llm.gradient_checkpointing_disable()


def run(fold: int, device: str, out_dir: str, limit: int | None = None) -> None:
    """Train the reference model of one fold, then score all texts and all captions. limit: for the smoke test."""
    torch.cuda.set_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        root = find_root()
        records = load_records(root)
        texts = training_texts(records, fold)
        tok = AutoTokenizer.from_pretrained(root / "model" / "llm")
        llm = AutoModelForCausalLM.from_pretrained(reference_path(), dtype=torch.float32).to(device)
        if limit:
            texts = texts[:limit]
        print(f"fold {fold}: {len(texts)} training records", flush=True)
        train(llm, tok, texts, device)
        score = records
        if limit:
            score = records.groupby("source_type").head(limit // 4)
        with torch.no_grad():
            caption_text_pass(llm, tok, score[score["kind"] == "photo"], device, out, f"ref{fold}_caption_text")
            text_pass(llm, tok, score[score["kind"] == "text"], device, out, f"ref{fold}_text")
    except Exception:
        # A failed kernel loses all its output. Thus log the error and keep the passes that are complete.
        traceback.print_exc()
        (out / f"error_fold{fold}.txt").write_text(traceback.format_exc())


def shadow_halves(records, seed: int = 1) -> np.ndarray:
    """A random half (0 or 1) for each record. Each owner and each source type of single records has half of its records in each half."""
    rng = np.random.default_rng(seed)
    is_set = records.groupby("source")["source"].transform("size") > 1
    group = records["source"].where(is_set, records["source_type"])
    half = np.empty(len(records), dtype=int)
    for idx in records.groupby(group).indices.values():
        idx = rng.permutation(idx)
        half[idx] = (rng.integers(2) + np.arange(len(idx))) % 2
    return half


def run_shadow(half: int, device: str, out_dir: str, limit: int | None = None, seed: int = 1, name: str = "shadow") -> None:
    """Train a shadow model on one half of all records (known and test, no labels), one epoch as in S1. Then score all records.

    Each record is IN for the model of its half and OUT for the other model.
    """
    torch.cuda.set_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        root = find_root()
        records = load_records(root)
        records["half"] = shadow_halves(records, seed)
        if half == 0:
            records[["id", "half"]].to_csv(out / f"{name}_halves.csv", index=False)
        texts = records.loc[records["half"] == half, "content"].tolist()
        if limit:
            texts = texts[:limit]
        tok = AutoTokenizer.from_pretrained(root / "model" / "llm")
        llm = AutoModelForCausalLM.from_pretrained(reference_path(), dtype=torch.float32).to(device)
        print(f"half {half}: {len(texts)} training records", flush=True)
        train(llm, tok, texts, device, epochs=1)
        score = records.groupby("source_type").head(limit // 4) if limit else records
        with torch.no_grad():
            caption_text_pass(llm, tok, score[score["kind"] == "photo"], device, out, f"{name}{half}_caption_text")
            text_pass(llm, tok, score[score["kind"] == "text"], device, out, f"{name}{half}_text")
    except Exception:
        traceback.print_exc()
        (out / f"error_half{half}.txt").write_text(traceback.format_exc())


PHOTO_SPLITS = 4  # random half splits of the photos; each photo is IN for 4 shadow models and OUT for 4
PHOTO_BATCH = 8


def photo_shadow_model(ob, root: Path, device):
    """BUMAS-mini with the LLM of Qwen3-0.6B-Base. The connector of BUMAS-mini is the only one that exists."""
    model = ob.BUMASMini.from_components(reference_path(), root / "model" / "vision", dtype=torch.float32)
    state = torch.load(root / "model" / "connector.pt")
    model.connector.load_state_dict({k: v.float() for k, v in state.items()})
    return model.to(device)


def train_photos(model, ob, photos, device) -> None:
    """One epoch as in S3: the image block, then the caption. The loss is only on the caption. The vision encoder is frozen."""
    model.llm.gradient_checkpointing_enable()
    model.llm.config.use_cache = False
    model.train()
    params = list(model.llm.parameters()) + list(model.connector.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    order = np.random.default_rng(0).permutation(len(photos))
    total = math.ceil(len(order) / PHOTO_BATCH)
    warmup = max(1, total // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, 0.1 + 0.45 * (1 + math.cos(math.pi * s / total))))
    pad = model.tok.pad_token_id
    for step, s in enumerate(range(0, len(order), PHOTO_BATCH)):
        batch = photos.iloc[order[s:s + PHOTO_BATCH]]
        encoded = [ob.caption_example(model.tok, c) for c in batch["caption"].fillna("")]
        width = max(len(ids) for ids, _ in encoded)
        input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in encoded], device=device)
        labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in encoded], device=device)
        mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in encoded], device=device)
        pixels = torch.stack([ob.load_image(p) for p in batch["image_path"]]).to(device)
        loss = model(input_ids=input_ids, attention_mask=mask, labels=labels, pixel_values=pixels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == total - 1:
            print(f"step {step + 1}/{total} loss {loss.item():.3f}", flush=True)
    model.eval()
    model.llm.gradient_checkpointing_disable()


def run_photo_shadow(splits: list[int], device: str, out_dir: str, limit: int | None = None) -> None:
    """For each split: train one shadow model on each half of the photos, then score the captions of all photos with their own photo."""
    from .logprobs import import_openbumas, photo_pass
    torch.cuda.set_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        root = find_root()
        ob = import_openbumas(root)
        photos = load_records(root)
        photos = photos[photos["kind"] == "photo"].copy()
        if limit:
            photos = photos.groupby("source_type").head(limit // 2)
        for split in splits:
            photos["half"] = shadow_halves(photos, seed=10 + split)
            photos[["id", "half"]].to_csv(out / f"photo_halves_s{split}.csv", index=False)
            for half in (0, 1):
                model = photo_shadow_model(ob, root, device)
                train_photos(model, ob, photos[photos["half"] == half], device)
                photo_pass(model, ob, photos, device, out, f"pshadow_s{split}_h{half}")
                del model
                torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
        (out / f"error_{device.replace(':', '')}.txt").write_text(traceback.format_exc())
