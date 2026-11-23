"""The figures of the README: probes, triggers of the validation sets, scan examples and results (CPU)."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2-experiments-v2"
PROJECT = "pro2-backdoor-detection"
CLONE = "/tmp/repo"
CODENAME = "readme-figures"

sys.path.insert(0, f"{CLONE}/{PROJECT}")

if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    import importlib
    importlib.invalidate_caches()
    from src import figures
    from src.paths import output_dir

    figures.run(output_dir() / CODENAME)
