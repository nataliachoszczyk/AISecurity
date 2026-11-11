"""Smoke test of the job val-models: the first validation models with different finetune settings."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "val-models-smoke"
LIMIT = 3

import torch

# A finetune on the CPU wastes hours, so stop immediately without a GPU.
assert torch.cuda.is_available(), "no GPU"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from src import val_models as vm
from src.paths import output_dir

# Version 3 is a diagnosis: the finetune of version 2 gave one answer for all images.
vm.VERBOSE = True
vm.MAX_ATTEMPTS = 1
for name, bn_train, distill in [("bn-train-distill", True, 1.0), ("bn-eval-distill", False, 1.0), ("bn-eval", False, 0.0)]:
    print("=== setting", name, flush=True)
    vm.BN_TRAIN, vm.DISTILL = bn_train, distill
    vm.build(output_dir() / CODENAME / name, "cuda", LIMIT)
