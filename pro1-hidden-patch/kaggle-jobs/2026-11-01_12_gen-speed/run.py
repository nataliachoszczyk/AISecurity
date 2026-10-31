"""Measure the speed of one attack (PGD, 100 steps) with fp32 or fp16, on 1 GPU or on 2 GPUs (DataParallel).

The success rate after JPEG shows if fp16 makes the attack weaker.
"""
import subprocess
import sys
import time

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1"
PROJECT = "pro1-hidden-patch"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.

import torch

assert torch.cuda.is_available(), "no GPU: check the phone verification of the account"
print(torch.cuda.device_count(), "GPU", torch.cuda.get_device_name(0), flush=True)

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
sys.path.insert(0, f"{CLONE}/{PROJECT}")

import numpy as np

import src.generate as gen
from src.eda import load_images, load_labels

torch.backends.cudnn.benchmark = True
labels = load_labels()
bases = labels[labels.kind == "CLEAN"].id[:256]
images = np.concatenate([load_images("train", bases)] * 4)  # 1024 images
rng = np.random.default_rng(0)
boxes = [gen.random_box(rng, np.zeros((64, 64), bool)) for _ in images]
quality = np.full(len(images), 90)

base_model = gen.load_model().cuda().requires_grad_(False)
x = torch.from_numpy(images).permute(0, 3, 1, 2).cuda()
mask = torch.from_numpy(np.stack([gen._rect(b) for b in boxes])[:, None].astype(np.float32)).cuda()
with torch.no_grad():
    c = base_model(x).argmax(1)
eps = torch.full((len(x),), 8 / 255, device="cuda")

for gpus in [1, 2]:
    model = torch.nn.DataParallel(base_model) if gpus == 2 else base_model
    for amp in [False, True]:
        for n in [512, 1024]:
            gen.AMP = amp
            gen.linf_attack(model, x[:64], mask[:64], c[:64], c[:64], eps[:64], "pgd_linf_u")  # warm-up
            torch.cuda.synchronize()
            t0 = time.time()
            adv = gen.linf_attack(model, x[:n], mask[:n], c[:n], c[:n], eps[:n], "pgd_linf_u")
            torch.cuda.synchronize()
            seconds = time.time() - t0
            with torch.no_grad():
                works = (base_model(gen.jpeg(adv, quality[:n])).argmax(1) != c[:n]).float().mean().item()
            print(f"gpus {gpus} amp {amp} batch {n}: {seconds:.1f} s, {1000 * seconds / n / 100:.2f} ms for each image step, "
                  f"works after JPEG {works:.3f}", flush=True)
