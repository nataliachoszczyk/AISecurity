"""Run the diagnostics and the post-processing variants on ensembles of the unet-v8, unet-v9 and unet-v11 models. One GPU is enough."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1-experiments"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "post-v3"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")
from src.config import load_config
from src.post import run_experiment

# The smoke test on 40 images of each set finds errors in the new variants in minutes.
cfg = load_config(CODENAME)
run_experiment(cfg | {"smoke": 40, "codename": f"{CODENAME}-smoke"})
run_experiment(cfg)
