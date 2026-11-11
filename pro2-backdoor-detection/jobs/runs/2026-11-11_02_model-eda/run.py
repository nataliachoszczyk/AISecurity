"""EDA of the models: compare the weights and the answers of all 206 task models. Write small CSV and PNG files."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "model-eda"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

import traceback

import pandas as pd
import torch

from src import model_stats as ms
from src.paths import output_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
out = output_dir() / CODENAME
out.mkdir(parents=True, exist_ok=True)

results = ms.collect(device)
# Save the raw tables first. A failure of the analysis below must not delete them, so the job catches it.
for name in ("models", "classes", "layers"):
    results[name].to_csv(out / f"{name}.csv", index=False)
ids = results["models"]["id"]
pd.DataFrame(results["similarity"], index=ids, columns=ids).to_csv(out / "similarity.csv", float_format="%.4f")

try:
    stats = [c for c in results["classes"].columns if c not in ("id", "split", "cls", "is_target")]
    print("statistics with NaN:", results["classes"][stats].columns[results["classes"][stats].isna().any()].tolist())
    results["classes"] = ms.add_zscores(results["classes"].fillna(0), stats)
    results["classes"].to_csv(out / "classes.csv", index=False)
    auc = ms.train_auc(results["classes"], stats)
    auc.to_csv(out / "train_auc.csv", index=False)

    pd.set_option("display.width", 200)
    print(results["models"].groupby(["split", "n_targets"]).mean(numeric_only=True))
    print(auc.head(30).to_string(index=False))
    x, y = ms.clean_images()
    ms.save_plots(results, auc, x, y, out)
    print("done")
except Exception:
    traceback.print_exc()
