"""Photographer module v2: record variants, scenarios of the honest owner check, global simulation, log-odds file (CPU)."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro3-photographer"
PROJECT = "pro3-membership-inference"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "photographer-v2"

if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from src.photographer import explore_v2
    explore_v2(f"/kaggle/working/{CODENAME}")
