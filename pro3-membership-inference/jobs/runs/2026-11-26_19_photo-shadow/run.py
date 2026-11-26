"""Photo shadow models (LiRA): for 4 random half splits of the photos, train one model on each half (S3 format), then score all photos.

GPU 0 runs the splits 0 and 1, GPU 1 runs the splits 2 and 3.
"""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro3-experiments"
PROJECT = "pro3-membership-inference"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "photo-shadow"
LIMIT = None


def run(gpu: int) -> None:
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from src.refmodel import run_photo_shadow
    splits = [gpu] if LIMIT else [2 * gpu, 2 * gpu + 1]
    run_photo_shadow(splits, f"cuda:{gpu}", f"/kaggle/working/{CODENAME}", LIMIT)


# A new process imports this file again, so the clone and the start must be only in the main process.
if __name__ == "__main__":
    # The model was saved with transformers 5.23.0. An older version can read its config in a different way.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.23.0"], check=True)
    import torch
    import transformers

    print("torch", torch.__version__, "transformers", transformers.__version__, flush=True)
    assert torch.cuda.device_count() == 2, "this job needs 2 GPUs (T4 x2)"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    with torch.multiprocessing.get_context("spawn").Pool(2) as pool:
        pool.map(run, [0, 1])
