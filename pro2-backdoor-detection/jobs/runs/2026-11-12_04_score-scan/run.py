"""Scores of the probe-bank scan, from the features of the job scan: the report, the checks and the submission files."""
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments-v2"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"
CODENAME = "score-scan"
SUBMIT = ["combo4_rank", "combo4_z", "combo2_rank", "hit_max_top3"]

sys.path.insert(0, f"{CLONE}/{PROJECT}")

if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    import importlib
    importlib.invalidate_caches()
    from src import score
    from src.paths import output_dir

    source = next(Path("/kaggle/input").rglob("scan/features.npz")).parent
    score.run(output_dir() / CODENAME, SUBMIT, source)
