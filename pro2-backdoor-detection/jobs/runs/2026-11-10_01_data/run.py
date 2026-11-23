"""Download the task data from Google Drive into the output folder. Check the clone of the repository."""
import html
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro2"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
FOLDER_ID = "1bMS02dzw9Yi4-TFTuvqvoqukaBbNj_Wn"
OUT = Path("/kaggle/working")
# The task text gives these numbers of files.
EXPECTED = {"test/models/*.pt": 200, "example/train/models/*.pt": 6, "example/clean/images/*": 400}
ENTRY = re.compile(r'id="entry-([\w-]+)".*?href="([^"]+)".*?flip-entry-title">([^<]*)<', re.S)


def list_files(folder_id: str, prefix: str = "") -> list[tuple[str, str]]:
    """Return (path, file id) for all files below a public Drive folder."""
    # This page shows all entries. The usual folder page shows only the first 50.
    page = requests.get("https://drive.google.com/embeddedfolderview", params={"id": folder_id}, timeout=60)
    page.raise_for_status()
    files = []
    for entry_id, link, name in ENTRY.findall(page.text):
        path = prefix + html.unescape(name)
        if "/folders/" in link:
            files += list_files(entry_id, path + "/")
        else:
            files.append((path, entry_id))
    return files


def download(item: tuple[str, str]) -> int:
    path, file_id = item
    target = OUT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    url = "https://drive.usercontent.google.com/download"
    for attempt in range(5):
        try:
            answer = requests.get(url, params={"id": file_id, "export": "download", "confirm": "t"}, timeout=300)
            # Drive sends an HTML page, not the file, when it refuses the download.
            if answer.ok and "text/html" not in answer.headers.get("content-type", ""):
                target.write_bytes(answer.content)
                return len(answer.content)
            error = f"status {answer.status_code}, {answer.headers.get('content-type')}"
        except requests.RequestException as exc:
            error = repr(exc)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"no download of {path}: {error}")


if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())

    files = list_files(FOLDER_ID)
    print(len(files), "files in the Drive folder", flush=True)
    # Stop before the long download if the list is not complete.
    for pattern, count in EXPECTED.items():
        found = sum(Path(path).match(pattern) for path, _ in files)
        assert found == count, f"{pattern}: {found} files in the list, expected {count}"

    with ThreadPoolExecutor(4) as pool:
        sizes = list(pool.map(download, files))
    print(f"{sum(sizes) / 1e9:.2f} GB in total")

    for folder in sorted({p.parent for p in OUT.rglob("*") if p.is_file()}):
        inside = [p for p in folder.iterdir() if p.is_file()]
        print(f"{folder.relative_to(OUT)}/: {len(inside)} files, {sum(p.stat().st_size for p in inside) / 1e6:.1f} MB")
