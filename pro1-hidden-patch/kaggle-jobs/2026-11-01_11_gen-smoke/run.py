"""Smoke test of gen-v4 on the CPU: the full flow of the generator with 4 bases for each part."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in /kaggle/working, so the clone is not an output.
CODENAMES = ["gen-smoke"]

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from src.config import load_config
from src.generate import run_generation as run_experiment

for codename in CODENAMES:
    run_experiment(load_config(codename))
