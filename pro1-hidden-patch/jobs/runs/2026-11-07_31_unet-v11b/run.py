"""Train one UNet variant on one GPU. Two variants with three generated sets need too much host memory."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1-experiments"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "unet-v11b"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")
from src.config import load_config
from src.unet import run_experiment

run_experiment(load_config(CODENAME), "cuda:0")
