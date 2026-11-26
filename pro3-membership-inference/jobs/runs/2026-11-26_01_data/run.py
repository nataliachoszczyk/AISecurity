"""Download p3_student.zip from Google Drive into the output folder and check its content. Check the clone of the repository."""
import html
import json
import re
import subprocess
import zipfile
from pathlib import Path

import requests

REPO = "https://github.com/nataliachoszczyk/AISecurity.git"
BRANCH = "pro3"
PROJECT = "pro3-membership-inference"
CLONE = "/tmp/repo"  # Not in the output folder, so the clone is not an output.
FOLDER_ID = "18bjGNIoUNwg0i1Mkrf0xZTpkioMuMKek"
NAME = "p3_student.zip"
OUT = Path("/kaggle/working")
# The task text gives these numbers of records.
EXPECTED = {"known.jsonl": 5000, "test.jsonl": 5000}
ENTRY = re.compile(r'id="entry-([\w-]+)".*?href="([^"]+)".*?flip-entry-title">([^<]*)<', re.S)


def find_file(folder_id: str, name: str) -> str:
    """Return the Drive id of the file with this name in a public folder."""
    page = requests.get("https://drive.google.com/embeddedfolderview", params={"id": folder_id}, timeout=60)
    page.raise_for_status()
    entries = {html.unescape(title): entry_id for entry_id, _, title in ENTRY.findall(page.text)}
    print("Drive folder:", sorted(entries), flush=True)
    return entries[name]


def download(file_id: str, target: Path) -> None:
    url = "https://drive.usercontent.google.com/download"
    with requests.get(url, params={"id": file_id, "export": "download", "confirm": "t"}, stream=True, timeout=300) as answer:
        answer.raise_for_status()
        # Drive sends an HTML page, not the file, when it refuses the download.
        assert "text/html" not in answer.headers.get("content-type", ""), "Drive refused the download"
        with target.open("wb") as out:
            for chunk in answer.iter_content(1 << 20):
                out.write(chunk)


if __name__ == "__main__":
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, "--filter=blob:none", "--sparse", REPO, CLONE], check=True)
    subprocess.run(["git", "-C", CLONE, "sparse-checkout", "set", PROJECT], check=True)
    print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())

    target = OUT / NAME
    download(find_file(FOLDER_ID, NAME), target)
    print(f"{NAME}: {target.stat().st_size / 1e9:.2f} GB", flush=True)

    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        print(len(names), "entries in the zip")
        tops = {}
        for name in names:
            parts = name.split("/")
            key = "/".join(parts[:2]) if len(parts) > 2 else name
            tops[key] = tops.get(key, 0) + 1
        for key, count in sorted(tops.items()):
            print(f"  {key}: {count}")
        for file, count in EXPECTED.items():
            path = next(n for n in names if n.endswith(file))
            lines = archive.read(path).decode().splitlines()
            assert len(lines) == count, f"{file}: {len(lines)} records, expected {count}"
            print(file, len(lines), "records, fields:", sorted(json.loads(lines[0])))
