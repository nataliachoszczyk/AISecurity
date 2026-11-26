"""Exploratory analysis of the task data (CPU): one HTML report with figures, tables and examples."""
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro3"
PROJECT = "pro3-membership-inference"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "eda"

if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    sys.path.insert(0, f"{CLONE}/{PROJECT}")

    from src.data import find_root
    from src.eda import make_report

    make_report(find_root(), Path("/kaggle/working") / CODENAME)
