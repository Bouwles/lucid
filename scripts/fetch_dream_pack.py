"""Download a dream pack zip, check its SHA-256 and unpack it (used by the release builds).

python scripts/fetch_dream_pack.py --url <release asset url> --sha256 <hex> --out app/src-tauri/resources/models/test-track

Standard library only, so it runs before anything else is installed.
"""

import argparse
import hashlib
import io
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not args.url.startswith("https://github.com/Bouwles/lucid/releases/download/"):
        sys.exit("dream packs only come from the LUCID GitHub Releases")
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(args.url, timeout=60).read()
            break
        except OSError as e:
            if attempt == 4:
                raise
            print(f"download failed ({e}), retrying")
            time.sleep(3 * (attempt + 1))
    digest = hashlib.sha256(data).hexdigest()
    if digest != args.sha256:
        sys.exit(f"checksum mismatch: expected {args.sha256}, got {digest}")
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                sys.exit(f"unsafe path in zip: {name}")
        z.extractall(out)
    if not (out / "dream.json").exists():
        sys.exit("the zip isn't a dream pack (no dream.json)")
    print(f"unpacked {len(data) / 1e6:.1f} MB dream pack into {out}")


if __name__ == "__main__":
    main()
