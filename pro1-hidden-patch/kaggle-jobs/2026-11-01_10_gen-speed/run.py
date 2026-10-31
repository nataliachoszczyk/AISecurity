"""Measure the speed of the generator on one batch of 300 images. gen-v3 shows no progress until its first batch of 2000 is complete."""
import subprocess
import sys
import time

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.

import torch

assert torch.cuda.is_available(), "no GPU: check the phone verification of the account"
print(torch.cuda.device_count(), "GPU", torch.cuda.get_device_name(0))

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

from src.generate import generate

start = time.time()
labels, images = generate("val", 3, True, 0, batch=300)
seconds = time.time() - start
print(f"{len(labels)} images in {seconds:.0f} s: {seconds / 300:.2f} s for each job")
print(f"gen-v3 train part (820 bases x 64 = 52480 jobs): approximately {52480 * seconds / 300 / 3600:.1f} hours")
