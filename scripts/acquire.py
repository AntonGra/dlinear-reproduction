"""Download ETTh1 and check its SHA-256 against the pinned manifest."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
ALLOWED = {"raw.githubusercontent.com"}


def fetch(url, limit=20_000_000):
    if urlparse(url).scheme != "https" or urlparse(url).hostname not in ALLOWED:
        raise ValueError("Unapproved download URL")
    with urlopen(url, timeout=60) as response:
        if urlparse(response.url).hostname not in ALLOWED:
            raise ValueError("Unexpected redirect")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Download exceeds size limit")
    return data


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, content):
    with path.open("xb") as dest:
        dest.write(content)


def data():
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "records").mkdir(exist_ok=True)
    manifest = ROOT / "records/data-source.json"
    expected = json.loads(manifest.read_text()) if manifest.exists() else None
    if expected and expected["commit"] != DATA_COMMIT:
        raise ValueError("Unexpected dataset commit in existing manifest")
    base = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{DATA_COMMIT}"
    records = []
    for remote, local in [("ETT-small/ETTh1.csv", "data/ETTh1.csv"),
                          ("LICENSE", "data/ETT_LICENSE"),
                          ("README.md", "data/ETT_README.md")]:
        blob = fetch(f"{base}/{remote}")
        if remote.endswith(".csv") and not blob.startswith(b"date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT"):
            raise ValueError("Unexpected CSV header")
        if expected:
            entry = next(x for x in expected["files"] if x["path"] == local)
            if sha256(blob) != entry["sha256"]:
                raise ValueError(f"Downloaded data differs from pinned manifest: {local}")
        target = ROOT / local
        if target.exists():
            if target.read_bytes() != blob:
                raise ValueError(f"Refusing to overwrite existing file: {local}")
        else:
            write_new(target, blob)
        records.append({"path": local, "url": f"{base}/{remote}",
                        "sha256": sha256(blob), "bytes": len(blob)})
    if not expected:
        write_new(manifest, json.dumps({
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": "https://github.com/zhouhaoyi/ETDataset",
            "commit": DATA_COMMIT, "files": records,
        }, indent=2).encode() + b"\n")
    print("Downloaded and recorded official ETT data, license and README.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["data"])
    parser.parse_args()
    data()
