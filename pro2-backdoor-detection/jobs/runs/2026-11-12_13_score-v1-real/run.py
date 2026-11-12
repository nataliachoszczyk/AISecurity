"""Scores of detect-v1-real: the AUC on the 6 labeled train models and the submission files. No network, so a CPU job."""
import importlib
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"
CODENAME = "score-v1-real"
SUBMIT = ["mix_abs_1.0", "mix_low_1.0", "inv_top3_2nd"]  # The best scores on set A.

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
sys.path.insert(0, f"{CLONE}/{PROJECT}")
importlib.invalidate_caches()
from src import evaluate
from src.paths import KAGGLE_INPUT, output_dir

source = next(KAGGLE_INPUT.rglob("detect-v1-real/features.npz")).parent
out = output_dir() / CODENAME
(out / "submissions").mkdir(parents=True, exist_ok=True)
models, feat = evaluate.load(source)
table = evaluate.report(models, feat, ["real"])
table.to_csv(out / "train_report.csv", index=False)
print(table[table["score"].isin(SUBMIT + ["warp_max", "inv_max_max"])].to_string(index=False))
print(table.head(15).to_string(index=False))
for name in SUBMIT:
    sub = evaluate.submission(models, feat, name)
    assert len(sub) == 200 and sub.notna().all().all()
    sub.to_csv(out / "submissions" / f"v1_{name}.csv", index=False)
print("submissions:", SUBMIT)
