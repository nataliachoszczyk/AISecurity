"""Download the reference model Qwen3-0.6B-Base (the revision of job 05) into the output. Other jobs read it as a kernel source."""
from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"  # the revision of job 2026-11-26_05_qwen-weights
OUT = "/kaggle/working/qwen3-0.6b-base"

snapshot_download(REPO_ID, revision=REVISION, local_dir=OUT, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
with open(f"{OUT}/REVISION", "w") as f:
    f.write(REVISION + "\n")
print("revision", REVISION, flush=True)
