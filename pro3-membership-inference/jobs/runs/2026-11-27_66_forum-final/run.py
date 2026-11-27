"""Final forum module: CV, the log-odds file of all forum posts, and the global simulation of explore_v10 with this file (CPU)."""
import subprocess
import sys

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro3-forum"
PROJECT = "pro3-membership-inference"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
CODENAME = "forum-final"

if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    sys.path.insert(0, f"{CLONE}/{PROJECT}")
    from src.attacks import explore_v10
    from src.forum import FORUM, forum_module
    out = f"/kaggle/working/{CODENAME}"
    forum_module(out)
    explore_v10(f"{out}/global-forum", ("shadow",), single_files={FORUM: f"{out}/submission_forum_logodds.csv"},
                single_C={"web photo": 0.01})
