"""Copy the small model files (code, model card, configs) to the output, so the team can read the model API."""
import shutil
from pathlib import Path

OUT = Path("/kaggle/working/model-code")
OUT.mkdir(parents=True, exist_ok=True)
for name in ("openbumas.py", "MODEL_CARD.md", "openbumas.json", "config.json", "tokenizer_config.json", "generation_config.json"):
    for path in Path("/kaggle/input").rglob(name):
        shutil.copy(path, OUT / f"{path.parent.name}_{name}")
        print(path)
