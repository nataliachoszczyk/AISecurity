"""New scores and the compact combiner on the features of detect-v2-a. No network, so a CPU job."""
import importlib
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"
CODENAME = "score-v2-a"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
sys.path.insert(0, f"{CLONE}/{PROJECT}")
importlib.invalidate_caches()
from src import evaluate
from src.paths import KAGGLE_INPUT, output_dir

source = next(KAGGLE_INPUT.rglob("detect-v2-a/features.npz")).parent
out = output_dir() / CODENAME
out.mkdir(parents=True, exist_ok=True)
models, feat = evaluate.load(source)
table = evaluate.report(models, feat, ["a"])
table.to_csv(out / "report.csv", index=False)
print(table.head(25).to_string(index=False))
loto = evaluate.loto_all(models, feat, "a")
loto.to_csv(out / "loto_a.csv", index=False)
print(loto.to_string(index=False))
