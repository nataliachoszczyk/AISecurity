"""Load the records of the task: known.jsonl, test.jsonl and sources.csv."""
from pathlib import Path

import pandas as pd


def find_root(base: str | Path = "/kaggle/input") -> Path:
    """Return the folder with known.jsonl. The path under /kaggle/input can change."""
    matches = sorted(Path(base).rglob("known.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"need exactly one known.jsonl below {base}, found {matches}")
    return matches[0].parent


def load_records(root: Path) -> pd.DataFrame:
    """Return known and test in one table. The column `split` tells the file, `member` is empty for test."""
    parts = []
    for split in ("known", "test"):
        part = pd.read_json(root / f"{split}.jsonl", lines=True)
        part["split"] = split
        parts.append(part)
    records = pd.concat(parts, ignore_index=True)
    for column in ("text", "caption", "image", "title"):
        if column not in records:
            records[column] = None
    # The model saw `text` (S1) or `caption` (S3), never both.
    records["content"] = records["text"].where(records["kind"] == "text", records["caption"]).fillna("")
    records["image_path"] = records["image"].map(lambda name: image_path(root, name))
    return records


def image_path(root: Path, name) -> Path | None:
    if not isinstance(name, str) or not name:
        return None
    for path in (root / name, root / "images" / Path(name).name):
        if path.exists():
            return path
    return None


def load_sources(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "sources.csv")
