"""What this machine can do, in plain language. No PyTorch needed."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def detect() -> dict:
    system, arch = platform.system(), platform.machine().lower()
    info = {
        "os": f"{system} {platform.release()}",
        "arch": arch,
        "device": "cpu",
        "gpu": None,
        "memory_gb": None,
    }
    if system == "Darwin":
        info["os"] = f"macOS {platform.mac_ver()[0]}"
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_gb"] = round(int(mem) / 2**30) if mem.isdigit() else None
        if arch == "arm64":
            info["device"] = "mps"
            info["gpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
    else:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            info["memory_gb"] = round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
            )
        if shutil.which("nvidia-smi"):
            name = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
            if name:
                info["device"] = "cuda"
                info["gpu"] = name.splitlines()[0]
    try:
        import onnxruntime as ort

        info["onnx_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnx_providers"] = []
    info["can_train"] = info["device"] in ("mps", "cuda")
    return info


def summary(info: dict, dream_fps: float | None) -> str:
    machine = "This Mac" if info["os"].startswith("macOS") else "This computer"
    if dream_fps is None:
        dream = f"{machine} hasn't been tested on a dream yet."
    elif dream_fps >= 12:
        dream = f"{machine} can play dreams at about {dream_fps:.0f} FPS."
    else:
        dream = f"{machine} can play dreams, but slowly (about {dream_fps:.0f} FPS)."
    if not info["can_train"]:
        train = "Teaching LUCID a new game needs a recent Apple Silicon Mac or an NVIDIA GPU, so here it would only run in preview quality."
    elif not info.get("training_pack_installed"):
        train = "Teaching LUCID a new game will need the training pack, a one-time download."
    else:
        train = "It's ready to teach LUCID a new game."
    return f"{dream} {train}"
