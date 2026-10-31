import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
KAGGLE_DATASET = "bumas-p1-hidden-patch"
KAGGLE_INPUT = Path("/kaggle/input")


def data_dir() -> Path:
    """Return the data folder: BUMAS_P1_DATA, then the Kaggle input, then the local data/ folder."""
    if "BUMAS_P1_DATA" in os.environ:
        return Path(os.environ["BUMAS_P1_DATA"])
    if KAGGLE_INPUT.exists():
        # Kaggle can put the dataset files one folder deeper than the dataset name.
        matches = sorted(KAGGLE_INPUT.rglob("train_labels.csv"))
        if matches:
            return matches[0].parent
    return PROJECT_DIR / "data"


def output_dir() -> Path:
    """Return a writable folder for checkpoints and submissions."""
    if KAGGLE_INPUT.exists():
        return Path("/kaggle/working")
    return PROJECT_DIR / "submissions"
