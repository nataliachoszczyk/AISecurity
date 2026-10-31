"""Estimate the JPEG quality of all train and test images. The task says that some test images are more difficult."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from multiprocessing import Pool

import pandas as pd

from src.eda import estimate_jpeg_quality, load_labels, load_test_ids


def part(args):
    split, ids = args
    return estimate_jpeg_quality(split, ids).assign(split=split)


labels = load_labels()
work = [("train", labels.id[i::4]) for i in range(4)] + [("test", load_test_ids()[i::4]) for i in range(4)]
with Pool(4) as pool:
    result = pd.concat(pool.map(part, work))
result = result.merge(labels[["id", "jpeg", "kind"]], on="id", how="left")
result.to_csv("/kaggle/working/jpeg_quality.csv", index=False)

train = result[result.split == "train"]
print("train: estimate equal to the label", (train.quality == train.jpeg).mean(), "within 1", ((train.quality - train.jpeg).abs() <= 1).mean())
print(pd.crosstab(result.split, result.quality).to_string())
