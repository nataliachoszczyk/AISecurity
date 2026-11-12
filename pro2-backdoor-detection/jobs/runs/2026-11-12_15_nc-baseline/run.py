"""Neural Cleanse, the baseline of the task, on the validation sets A and B. On the leaderboard (public), it gets 0.901."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "nc-baseline"
SETS = ["a", "b"]
SMOKE = True
SCORES = ["nc_anomaly", "nc_norm_z"]

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
    detect.run(out, SETS, SMOKE, families=["nc"])
    models, feat = evaluate.load(out)
    table = evaluate.report(models, feat, SETS)
    table.to_csv(out / "report.csv", index=False)
    y, _, _ = evaluate.pair_labels(models)
    all_scores = evaluate.scores(models, feat)
    for name in SETS:
        rows = (models["set"] == name).to_numpy()
        for score in SCORES:
            s = all_scores[score][rows]
            print(f"{score} on set {name}: AUC {evaluate.auc(y[rows], s):.4f}, 90% interval {evaluate.bootstrap(y[rows], s)}")
    print(table[table["score"].isin(SCORES)].T.to_string())
