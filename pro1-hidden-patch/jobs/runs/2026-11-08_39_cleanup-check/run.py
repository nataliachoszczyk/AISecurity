"""Compare the act2 steps of the old code (pro1-experiments) and the cleaned code (pro1-experiments-final) on a small sample."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
EQUIV = '"""Run the act2 steps of one code version on a small sample. Arguments: project folder, version (old/new), output .npz."""\nimport sys\n\nimport numpy as np\nimport torch\n\nroot, version, out = sys.argv[1:4]\nsys.path.insert(0, root)\nDEVICE = "cuda" if torch.cuda.is_available() else "cpu"\ntorch.backends.cudnn.deterministic = True\ntorch.backends.cudnn.benchmark = False\nfrom src import evaluate, generate, unet  # noqa: E402\n\ncfg = dict(seed=0, tune_fraction=0.1, width=8, ig_steps=2, epochs=2, batch=16, lr=0.05, weight_decay=0.0005,\n           candidates=6, batch_causal=50, chunk=600, thresholds=[0.0, 3.0, 61], fit_offset=0.3, ema=0.999,\n           checkpoint_every=1, train_seed=1)\nif version == "old":\n    cfg |= dict(use_ig=True, use_act=True, boxes="fit", amp=True, optimizer="sgd", best_epoch=False,\n                mask_threshold=0.5, min_area=30)\n\nres = {}\nlabels, images = evaluate.eval_set("train")\nx_in = unet.unet_inputs(unet.load_model().to(DEVICE).requires_grad_(False), images[:80], cfg, DEVICE)\nres["inputs"] = x_in\nmasks = unet.target_masks(labels[:80])\nresnet = unet.load_model().to(DEVICE).requires_grad_(False)\nmodel = unet.train(x_in[:64], masks[:64], cfg, DEVICE, (x_in[64:], masks[64:]))\nres["losses"] = np.array(model.losses + model.tune_losses)\nres["weights"] = torch.cat([v.flatten().float() for v in model.state_dict().values()]).cpu().numpy()\ncand = unet.analyze(resnet, model, images[64:80], cfg, DEVICE)\nfor k in ["boxes", "drops", "probs", "patch_prob"]:\n    res["cand_" + k] = cand[k]\nparams = unet.tune(labels[64:80].reset_index(drop=True), cand, np.linspace(*cfg["thresholds"]))\nres["tune"] = np.array([params["tune_score"], params["threshold"], params["rule"] == "unet"])\nres["choose"] = unet.choose(cand, params["rule"], params["threshold"])\n\n# The generator on 4 bases: the same subset of labels in both versions.\nfull = generate.load_labels\ngenerate.load_labels = lambda: full().head(120)\ngcfg = dict(seed=3, train_copies=1, val_copies=1, batch=8, tune_fraction=0.1, tune_seed=0, jpeg=[80, 95], base_aug=True)\nif version == "old":\n    lab, img = generate.generate("train", 1, True, 3, 8, None, evaluate.tune_ids(0.1, 0), None, (0, 1), DEVICE, (80, 95), True)\nelse:\n    generate_bases = generate.generate\n    lab, img = generate_bases("train", gcfg, None, (0, 1), DEVICE)\n    lab = lab.head(10**9)\nres["gen_images"] = img[: len(img)]\nres["gen_boxes"] = lab[evaluate.BOX].to_numpy()\nnp.savez(out, **res)\nprint(version, "done", len(lab), "generated")\n'

Path("/tmp/equiv.py").write_text(EQUIV)
for version, branch in [("old", "pro1-experiments"), ("new", "pro1-experiments-final")]:
    clone = f"/tmp/{version}"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, "--filter=blob:none", "--sparse", REPO, clone], check=True)
    subprocess.run(["git", "-C", clone, "sparse-checkout", "set", "pro1-hidden-patch"], check=True)
    print(version, subprocess.run(["git", "-C", clone, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(), flush=True)
    subprocess.run([sys.executable, "/tmp/equiv.py", f"{clone}/pro1-hidden-patch", version, f"/tmp/{version}.npz"], check=True)

old, new = np.load("/tmp/old.npz"), np.load("/tmp/new.npz")
report = {}
for key in old.files:
    a, b = old[key], new[key]
    same_shape = a.shape == b.shape
    diff = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if same_shape and a.size else None
    report[key] = {"shape_old": list(a.shape), "shape_new": list(b.shape), "max_abs_diff": diff}
report["all_equal"] = all(v["max_abs_diff"] == 0.0 for v in report.values())
Path("/kaggle/working/compare.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
