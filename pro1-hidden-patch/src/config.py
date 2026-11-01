"""Global settings of the project. Change them only here."""
from pathlib import Path

import yaml

SEED = 0
VAL_FRACTION = 0.1

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPLIT_FILE = PROJECT_DIR / "splits" / "split.csv"
JOBS_DIR = PROJECT_DIR / "jobs"


def load_config(codename: str) -> dict:
    """Return the settings of one experiment from jobs/<job folder>/<codename>.yaml.

    The config is in the folder of the job that runs it. A job can run more than one codename.
    """
    matches = sorted(JOBS_DIR.glob(f"*/{codename}.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"need exactly one config for {codename}, found {matches}")
    cfg = yaml.safe_load(matches[0].read_text())
    cfg["codename"] = codename
    return cfg
