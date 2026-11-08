"""Run the diagnostics and the post-processing variants on the final ensembles of the unet-v8 to unet-v12 models. One GPU is enough."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1-experiments"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "post-v4"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")
from src.config import load_config
from src.post import run_experiment

# The smoke test runs in the job 2026-11-08_38_post-v4-smoke on the other account.
run_experiment(load_config(CODENAME))
