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


def train(llm, tok, texts: list[str], device) -> None:
    llm.gradient_checkpointing_enable()
    llm.config.use_cache = False
    llm.train()
    optimizer = torch.optim.AdamW(llm.parameters(), lr=LR, weight_decay=0.0)
    all_pieces = [pieces(tok, texts, seed=epoch) for epoch in range(EPOCHS)]
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
