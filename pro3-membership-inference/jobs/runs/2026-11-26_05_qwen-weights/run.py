"""Download the reference model Qwen3-0.6B-Base from Hugging Face into the output. Other jobs read it as a kernel source."""
from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Qwen/Qwen3-0.6B-Base"
OUT = "/kaggle/working/qwen3-0.6b-base"

revision = HfApi().model_info(REPO_ID).sha
print("revision", revision, flush=True)
snapshot_download(REPO_ID, revision=revision, local_dir=OUT, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
with open(f"{OUT}/REVISION", "w") as f:
    f.write(revision + "\n")
