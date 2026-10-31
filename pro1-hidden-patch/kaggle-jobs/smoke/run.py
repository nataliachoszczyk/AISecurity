"""Smoke test: check the GPUs, the data, the repository clone and the output folder."""
import json
import subprocess
import sys
import time

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro1"
PROJECT = "pro1-hidden-patch"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, "repo"], check=True)
subprocess.run(["git", "-C", "repo", "sparse-checkout", "set", PROJECT], check=True)
commit = subprocess.run(["git", "-C", "repo", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
sys.path.insert(0, f"repo/{PROJECT}")

import torch

from src.config import load_config
from src.eda import load_images, load_labels, load_model, load_test_ids
from src.evaluate import eval_set
from src.paths import data_dir, output_dir

info = {"commit": commit, "data_dir": str(data_dir()), "output_dir": str(output_dir())}
info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
info["torch"] = torch.__version__
info["python"] = sys.version.split()[0]
device = "cuda" if info["gpus"] else "cpu"

labels = load_labels()
info["n_train"], info["n_test"] = len(labels), len(load_test_ids())
info["n_val"] = len(eval_set("val")[0])
info["config_smooth"] = load_config("saliency-v1")["smooth"]

model = load_model().to(device)
if len(info["gpus"]) > 1:
    model = torch.nn.DataParallel(model)
x = torch.from_numpy(load_images("train", labels.id[:512])).permute(0, 3, 1, 2).to(device)
start = time.time()
with torch.no_grad():
    for _ in range(10):
        model(x).argmax(1)
if device == "cuda":
    torch.cuda.synchronize()
info["images_per_second"] = round(5120 / (time.time() - start))

print(json.dumps(info, indent=2))
(output_dir() / "smoke.json").write_text(json.dumps(info, indent=2))
