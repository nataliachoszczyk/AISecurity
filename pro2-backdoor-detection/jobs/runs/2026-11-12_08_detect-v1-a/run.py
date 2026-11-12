"""Detector v1 on the validation set A: features, the AUC report and the combiner with leave-one-type-out."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "detect-v1-a"
SETS = ["a"]
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
    from src import detect, evaluate
    from src.paths import output_dir

    out = output_dir() / CODENAME
    detect.run(out, SETS, SMOKE)
    models, feat = evaluate.load(out)
    # Set B is the sealed set: no report for it here.
    report_sets = [s for s in SETS if s != "b"]
    if report_sets:
        table = evaluate.report(models, feat, report_sets)
        table.to_csv(out / "report.csv", index=False)
        print(table.drop(columns=[c for c in table.columns if c.startswith(("type_", "variant_"))]).head(40).to_string(index=False))
        print(table.head(10).to_string(index=False))
    if "a" in SETS:
        loto = evaluate.loto_report(models, feat, "a")
        loto.to_csv(out / "loto_a.csv", index=False)
        print("combiner, leave one trigger type out on set A:")
        print(loto.to_string(index=False))
