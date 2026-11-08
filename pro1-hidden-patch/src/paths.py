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


def generated_dir(name: str = "gen-v4") -> Path:
    """Return the folder with the generated set name (train.npy, val.npy and their CSV files).

    The notebook output of the generator has the zip <name>.zip (gen-v4) or <name>-set.zip (later sets).
    A zip is extracted to /kaggle/temp/<name>.
    """
    found = [p.parent for p in KAGGLE_INPUT.rglob("val.npy") if name in p.parts] if KAGGLE_INPUT.exists() else []
    if found:
        return sorted(found)[0]
    zips = sorted(p for p in KAGGLE_INPUT.rglob("*.zip") if p.name in (f"{name}.zip", f"{name}-set.zip"))         if KAGGLE_INPUT.exists() else []
    if not zips:
        raise FileNotFoundError(f"no generated set {name} in /kaggle/input")
    target = Path("/kaggle/temp") / name
    if not target.exists():
        import tempfile
        import zipfile
        # Two processes (one on each GPU) can extract at the same time. The rename makes the complete folder appear at once.
        Path("/kaggle/temp").mkdir(exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir="/kaggle/temp"))
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(tmp)
        try:
            tmp.rename(target)
        except OSError:
            import shutil
            shutil.rmtree(tmp)
    return target
