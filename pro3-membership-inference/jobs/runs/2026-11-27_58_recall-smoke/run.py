"""ReCaLL passes of the forum posts: BUMAS-mini on GPU 0, Qwen3-0.6B-Base on GPU 1. With LIMIT, only a few posts (smoke test)."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro3-forum"
PROJECT = "pro3-membership-inference"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "recall-smoke"
LIMIT = 20


def run(model_name: str, gpu: int) -> None:
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from src.forum import run_recall
    run_recall(model_name, f"cuda:{gpu}", f"/kaggle/working/{CODENAME}", LIMIT)


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
        pool.starmap(run, [("bumas", 0), ("qwen", 1)])
