"""Make the 120 validation models with known backdoors on 2 GPUs. The output becomes the dataset jeronimoo/bumas-p2-val-models."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "val-models"
SMOKE = False

# Each worker process imports this file again. It needs the path, but not the clone.
sys.path.insert(0, f"{CLONE}/{PROJECT}")

if __name__ == "__main__":
    import torch

    assert torch.cuda.device_count() == 2, "this job needs 2 GPUs (T4 x2)"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())

    # The path was added before the clone existed, so Python must forget its empty listing of the folder.
    import importlib
    importlib.invalidate_caches()
    from src.paths import output_dir
    from src.val_models import run

    run(output_dir() / CODENAME, SMOKE)
