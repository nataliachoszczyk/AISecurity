"""Run notebooks/01_eda.ipynb on the task data (CPU). The output is the notebook with all results."""
import subprocess
from pathlib import Path

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro3"
PROJECT = "pro3-membership-inference"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "eda-notebook"

if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())

    out = Path("/kaggle/working") / CODENAME
    out.mkdir(parents=True, exist_ok=True)
    # nbconvert runs the notebook in its own folder, so the notebook finds src/ in the folder above.
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=3600",
                    f"{CLONE}/{PROJECT}/notebooks/01_eda.ipynb", "--output-dir", str(out)], check=True)
