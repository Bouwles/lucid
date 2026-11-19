"""Download dream pack zips, check their SHA-256 and unpack them (used by the release builds).

python scripts/fetch_dream_pack.py --list app/src-tauri/resources/models/packs.json
python scripts/fetch_dream_pack.py --url <release asset url> --sha256 <hex> --out app/src-tauri/resources/models/<id>

Standard library only, so it runs before anything else is installed.
"""

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RELEASES = "https://github.com/Bouwles/lucid/releases/download/"


def fetch(url: str, sha256: str, out: Path) -> None:
    if not url.startswith(RELEASES):
        sys.exit("dream packs only come from the LUCID GitHub Releases")
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            break
        except OSError as e:
            if attempt == 4:
                raise
            print(f"download failed ({e}), retrying")
            time.sleep(3 * (attempt + 1))
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        sys.exit(f"{url}: checksum mismatch, expected {sha256}, got {digest}")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                sys.exit(f"unsafe path in zip: {name}")
        z.extractall(out)
    if not (out / "dream.json").exists():
        sys.exit(f"{url} isn't a dream pack (no dream.json)")
    print(f"unpacked {len(data) / 1e6:.1f} MB dream pack into {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--list", help="JSON list of {id, url, sha256}; each pack unpacks next to the list"
    )
    p.add_argument("--url")
    p.add_argument("--sha256")
    p.add_argument("--out")
    args = p.parse_args()
    if args.list:
        lst = Path(args.list)
        for item in json.loads(lst.read_text()):
            fetch(item["url"], item["sha256"], lst.parent / item["id"])
    elif args.url and args.sha256 and args.out:
        fetch(args.url, args.sha256, Path(args.out))
    else:
        p.error("pass --list, or --url, --sha256 and --out")


if __name__ == "__main__":
    main()
