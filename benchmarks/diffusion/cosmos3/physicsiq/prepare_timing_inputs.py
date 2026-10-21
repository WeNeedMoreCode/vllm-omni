# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Download Physics-IQ prompts and switch frames only; no model or scoring run."""

import argparse
import base64
import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path.cwd())
ROOT = parser.parse_args().output_dir.resolve()
COMMIT = "c2bca8a5bf452d24e177e9975641af1888eb1492"
DATA = ROOT / "datasets/physics-IQ-benchmark/switch-frames"
DATA.mkdir(parents=True, exist_ok=True)


def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(url, timeout=120) as stream:
                return stream.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


csv_bytes = fetch(
    "https://raw.githubusercontent.com/google-deepmind/physics-IQ-benchmark/"
    + COMMIT
    + "/descriptions/descriptions_original.csv"
)
rows = [r for r in csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))) if "_take-1_" in r["scenario"]]
if len(rows) != 198:
    raise ValueError("Expected 198 take-1 prompts")
(ROOT / "descriptions_original.csv").write_bytes(csv_bytes)
(ROOT / "EDGE_IQVERIFIED_SUBMISSION.yaml").write_bytes(
    fetch("https://github.com/user-attachments/files/31664307/EDGE_IQVERIFIED_SUBMISSION.yaml")
)
items = []
token = None
while True:
    params = {"prefix": "switch-frames/", "maxResults": 1000}
    if token:
        params["pageToken"] = token
    listing = json.loads(
        fetch("https://storage.googleapis.com/storage/v1/b/physics-iq-benchmark/o?" + urlencode(params))
    )
    items.extend(o for o in listing.get("items", []) if o["name"].endswith(".jpg"))
    token = listing.get("nextPageToken")
    if not token:
        break
expected = {r["generated_video_name"][:4] for r in rows}
items = [o for o in items if Path(o["name"]).name[:4] in expected]
if len(items) != 198 or {Path(o["name"]).name[:4] for o in items} != expected:
    raise ValueError("Expected one image per case")


def download(obj):
    path = DATA / Path(obj["name"]).name

    def valid(payload):
        return (
            len(payload) == int(obj["size"])
            and base64.b64encode(hashlib.md5(payload).digest()).decode() == obj["md5Hash"]
        )

    if path.exists() and valid(path.read_bytes()):
        return path.name
    payload = fetch(
        "https://storage.googleapis.com/download/storage/v1/b/physics-iq-benchmark/o/"
        + quote(obj["name"], safe="")
        + "?alt=media"
    )
    if not valid(payload):
        raise ValueError("Downloaded image checksum mismatch: " + path.name)
    pending = path.with_suffix(".partial")
    pending.write_bytes(payload)
    pending.replace(path)
    return path.name


with ThreadPoolExecutor(max_workers=6) as pool:
    for count, name in enumerate(pool.map(download, sorted(items, key=lambda o: o["name"])), 1):
        print(json.dumps({"images_ready": count, "total": 198, "last": name}), flush=True)
print("TIMING_INPUTS_READY (no video generation or scoring performed)", flush=True)
