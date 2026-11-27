"""Merged score with both shadow pairs and C = 0.01 for web photos: the owner mixture for all set types, the record score for the single types, and its global simulation (CPU)."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro3-experiments"
PROJECT = "pro3-membership-inference"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "attacks-v10b"

if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from src.attacks import explore_v10
    explore_v10(f"/kaggle/working/{CODENAME}", ("shadow", "shadow-b"), single_C={"web photo": 0.01})
