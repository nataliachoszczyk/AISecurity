"""Find the task data, the output folder and the model code on Kaggle and on the local machine."""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")


def data_dir() -> Path:
    """Return the folder that contains example/ and test/."""
    if KAGGLE_INPUT.exists():
        # The path of a dataset under /kaggle/input can change, so search for a known file.
        matches = sorted(KAGGLE_INPUT.rglob("example/model.py"))
        assert matches, "no task data in /kaggle/input"
        return matches[0].parents[1]
    return PROJECT_DIR / "data"


def imagenet_dir() -> Path:
    """Return the folder of the ImageNet 64x64 dataset (only on Kaggle)."""
    matches = sorted(KAGGLE_INPUT.rglob("val_data"))
    assert matches, "no ImageNet 64x64 data in /kaggle/input"
    return matches[0].parents[1]


def output_dir() -> Path:
    """Return a writable folder for results and submissions."""
    if KAGGLE_INPUT.exists():
        return Path("/kaggle/working")
    return PROJECT_DIR / "submissions"


def load_model(path: Path, device: str = "cpu"):
    """Load one model (eval mode, float32) with the load function of the task."""
    # The file model.py of the task is in the dataset, not in the repository. Python must know its folder.
    folder = str(data_dir() / "example")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    from model import load
    return load(path, device)
