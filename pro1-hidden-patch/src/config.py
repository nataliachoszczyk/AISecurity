"""Global settings of the project. Change them only here."""
from pathlib import Path

import yaml

SEED = 0
VAL_FRACTION = 0.1

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_DIR / "splits" / "split.csv"
CONFIG_DIR = PROJECT_DIR / "configs"


def load_config(codename: str) -> dict:
    """Return the settings of one experiment from configs/YYYY-MM-DD_NN_<codename>.yaml.

    The date and NN are the same as in the name of the Kaggle job folder, so the files show the order of the runs.
    """
    matches = sorted(CONFIG_DIR.glob(f"????-??-??_??_{codename}.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"need exactly one config for {codename}, found {matches}")
    cfg = yaml.safe_load(matches[0].read_text())
    cfg["codename"] = codename
    return cfg
