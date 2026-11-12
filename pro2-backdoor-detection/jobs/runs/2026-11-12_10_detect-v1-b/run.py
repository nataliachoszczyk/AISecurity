"""Detector v1 on the sealed validation set B. This is the one look at set B. The score was selected before: mix_abs_1.0."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "detect-v1-b"
SETS = ["b"]
SMOKE = False
FINAL_SCORE = "mix_abs_1.0"  # Selected on set A and the 6 train models, before this run.

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
    from src import detect, evaluate
    from src.paths import output_dir

    out = output_dir() / CODENAME
    detect.run(out, SETS, SMOKE, families=detect.FAMILIES_V1)
    models, feat = evaluate.load(out)
    table = evaluate.report(models, feat, SETS)
    table.to_csv(out / "report.csv", index=False)
    y, _, _ = evaluate.pair_labels(models)
    s = evaluate.scores(models, feat)[FINAL_SCORE]
    print(f"FINAL {FINAL_SCORE} on set B: AUC {evaluate.auc(y, s):.4f}, 90% interval {evaluate.bootstrap(y, s)}")
    print(table[table["score"] == FINAL_SCORE].T.to_string())
