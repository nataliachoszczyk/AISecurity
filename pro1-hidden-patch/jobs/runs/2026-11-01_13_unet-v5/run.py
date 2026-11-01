"""Run the UNet experiments: train, tune on the tune part, score val, predict the test set."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in /kaggle/working, so the clone is not an output.
CODENAMES = ["unet-v5"]

import torch

# A long training on the CPU wastes hours, so stop at once without a GPU.
assert torch.cuda.is_available(), "no GPU: check the phone verification of the account"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from src.config import load_config
from src.unet import run_experiment

for codename in CODENAMES:
    run_experiment(load_config(codename))
