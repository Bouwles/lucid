"""Build the engine as one standalone executable for the Tauri app (no Python needed to run it).

python engine/build_sidecar.py [--out app/src-tauri/binaries]

It uses a clean build environment with only engine/requirements.txt installed, so PyTorch never
ends up in the base app. The output is named for Tauri's sidecar convention:
lucid-engine-<target triple>[.exe].
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRIPLES = {
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
}


def target_triple() -> str:
    key = (platform.system(), platform.machine())
    if key not in TRIPLES:
        sys.exit(f"no Tauri target triple known for {key}")
    return TRIPLES[key]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(HERE.parent / "app" / "src-tauri" / "binaries"))
    p.add_argument("--venv", default=str(HERE / ".venv-build"))
    args = p.parse_args()

    env_dir = Path(args.venv)
    py = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    reqs = ["-r", str(HERE / "requirements.txt"), "pyinstaller>=6.0"]
    if shutil.which("uv"):  # faster, and works with uv-managed Pythons that lack ensurepip
        if not py.exists():
            subprocess.run(
                [
                    "uv",
                    "venv",
                    "--python",
                    f"{sys.version_info.major}.{sys.version_info.minor}",
                    str(env_dir),
                ],
                check=True,
            )
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(py), *reqs], check=True)
    else:
        if not py.exists():
            venv.create(env_dir, with_pip=True)
        subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
        subprocess.run([py, "-m", "pip", "install", "-q", *reqs], check=True)

    triple = target_triple()
    name = f"lucid-engine-{triple}"
    work = HERE / "build"
    subprocess.run(
        [
            py,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--name",
            name,
            "--paths",
            str(HERE),
            # The engine isn't pip-installed in the build env, so its data files are added by path.
            "--add-data",
            f"{HERE / 'lucid_engine' / 'catalog.json'}{os.pathsep}lucid_engine",
            "--add-data",
            f"{HERE / 'lucid_engine' / 'media'}{os.pathsep}lucid_engine/media",
            "--collect-submodules",
            "uvicorn",
            "--collect-submodules",
            "websockets",
            "--exclude-module",
            "torch",
            "--exclude-module",
            "tkinter",
            "--distpath",
            str(work / "dist"),
            "--workpath",
            str(work / "tmp"),
            "--specpath",
            str(work),
            str(HERE / "sidecar_entry.py"),
        ],
        check=True,
    )
    exe = name + (".exe" if os.name == "nt" else "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / "dist" / exe, out / exe)
    print(f"built {out / exe} ({(out / exe).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
