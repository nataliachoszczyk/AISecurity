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
