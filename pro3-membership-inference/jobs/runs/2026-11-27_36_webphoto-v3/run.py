"""Web photo module: the choice of the regularization and of the feature set on the CV and on the global simulation, and the final log-odds (CPU)."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro3-webphoto"
PROJECT = "pro3-membership-inference"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "webphoto-v3"

if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from src.webphoto import explore_v3
    explore_v3(f"/kaggle/working/{CODENAME}")
