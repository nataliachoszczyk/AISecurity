"""Generate attacked images with patches at any position. Score saliency-v1 on the generated val set."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1-experiments"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
# The smoke test runs first in the same session, so an error stops the job after a few minutes.
CODENAMES = ["gen-v5-smoke", "gen-v5"]

# The generator starts one process for each GPU. A new process imports this file again,
# so the clone and the run must be only in the main process.
if __name__ == "__main__":
    import torch

    # A long run on the CPU wastes hours, so stop immediately without a GPU.
    assert torch.cuda.is_available(), "no GPU: check the phone verification of the account"
    print(torch.cuda.device_count(), "GPU", flush=True)

    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    sys.path.insert(0, f"{CLONE}/{PROJECT}")

    from src.config import load_config
    from src.generate import run_generation

    for codename in CODENAMES:
        run_generation(load_config(codename))
        if codename.endswith("-smoke"):
            # Later jobs find the generated set by the zip name, so the smoke zip must not stay in the output.
            for path in __import__("pathlib").Path("/kaggle/working").glob(f"{codename}*.zip"):
                path.unlink()
