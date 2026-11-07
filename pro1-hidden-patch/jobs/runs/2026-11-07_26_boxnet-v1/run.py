"""Train two BoxNet variants at the same time, one on each GPU. Each variant then refines the answers of unet-v8a and unet-v8b."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1-experiments"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAMES = ["boxnet-v1a", "boxnet-v1b"]


def run(codename: str, gpu: int) -> None:
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from src.config import load_config
    from src.boxnet import run_experiment
    run_experiment(load_config(codename), f"cuda:{gpu}")


# A new process imports this file again, so the clone and the start must be only in the main process.
if __name__ == "__main__":
    import torch

    # Each variant needs its own GPU.
    assert torch.cuda.device_count() == 2, "this job needs 2 GPUs (T4 x2)"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    with torch.multiprocessing.get_context("spawn").Pool(2) as pool:
        pool.starmap(run, [(codename, gpu) for gpu, codename in enumerate(CODENAMES)])
