"""Smoke-test a built desktop app: launch it, let it open a dream, check it drew frames, exit.

python scripts/smoke_app.py [--app <path to the app's executable>]

Finds the release build for this platform if --app isn't given. Uses xvfb-run on Linux when
there's no display. Standard library only.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "src-tauri" / "target" / "release"


def default_app() -> Path:
    system = platform.system()
    if system == "Darwin":
        return TARGET / "bundle" / "macos" / "LUCID.app" / "Contents" / "MacOS" / "lucid"
    if system == "Windows":
        return TARGET / "lucid.exe"
    return TARGET / "lucid"


def data_dir() -> Path:
    """Where Tauri's data_dir() points on each platform, plus LUCID/."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "LUCID"
    if system == "Windows":
        return Path(os.environ["APPDATA"]) / "LUCID"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "LUCID"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--app", default=None)
    p.add_argument("--timeout", type=float, default=240)
    args = p.parse_args()
    app = Path(args.app) if args.app else default_app()
    if not app.exists():
        sys.exit(f"no built app at {app}")
    result = data_dir() / "logs" / "smoke.json"
    result.unlink(missing_ok=True)
    cmd = [str(app)]
    if platform.system() == "Linux" and not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", *cmd]
    env = {**os.environ, "LUCID_SMOKE_TEST": "1"}
    try:
        code = subprocess.run(cmd, env=env, timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        code = None
    report = json.loads(result.read_text()) if result.exists() else None
    print(f"exit code: {code}")
    print(f"smoke.json: {json.dumps(report)}")
    log = data_dir() / "logs" / "engine.log"
    if log.exists():
        print("engine.log (last lines):")
        print("\n".join(log.read_text(errors="replace").splitlines()[-15:]))
    if code != 0 or not report or not report.get("ok"):
        sys.exit("smoke test failed")
    print("smoke test passed")


if __name__ == "__main__":
    main()
