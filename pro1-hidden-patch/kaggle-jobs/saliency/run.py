"""Run the saliency experiments: tune on train, score train/val/val_flip, predict the test set."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1"
PROJECT = "pro1-hidden-patch"
CODENAMES = ["saliency-v1", "saliency-v2"]

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, "repo"], check=True)
subprocess.run(["git", "-C", "repo", "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", "repo", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"repo/{PROJECT}")

from src.config import load_config
from src.saliency import run_experiment

for codename in CODENAMES:
    run_experiment(load_config(codename))
