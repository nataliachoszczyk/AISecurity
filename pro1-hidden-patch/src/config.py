"""Global settings of the project. Change them only here."""
from pathlib import Path

import yaml

SEED = 0
VAL_FRACTION = 0.1

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_DIR / "splits" / "split.csv"
CONFIG_DIR = PROJECT_DIR / "configs"


def load_config(codename: str) -> dict:
    """Return the settings of one experiment from configs/<codename>.yaml."""
    cfg = yaml.safe_load((CONFIG_DIR / f"{codename}.yaml").read_text())
    cfg["codename"] = codename
    return cfg
