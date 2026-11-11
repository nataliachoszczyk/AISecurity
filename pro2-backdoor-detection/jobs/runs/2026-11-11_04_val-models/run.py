"""Make the 120 validation models with known backdoors. The output becomes the dataset jeronimoo/bumas-p2-val-models."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "val-models"
LIMIT = None

import torch

# A finetune on the CPU wastes hours, so stop immediately without a GPU.
assert torch.cuda.is_available(), "no GPU"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from src.paths import output_dir
from src.val_models import build

labels = build(output_dir() / CODENAME, "cuda", LIMIT)
print(labels.groupby(labels["targets"].str.count(";") + (labels["targets"] != "")).agg(models=("id", "size"), valid=("valid", "sum"), attempts=("attempts", "mean")))
print("mean clean accuracy:", labels["clean_acc"].mean())
