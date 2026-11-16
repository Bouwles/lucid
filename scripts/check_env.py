"""Print the Python/PyTorch/device setup and write it to docs/hardware.md."""

import argparse
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from lucid.utils.device import autocast_dtype, pick_device


def _autocast_ok(device: torch.device, dtype: torch.dtype) -> bool:
    if device.type == "cpu":
        return False
    try:
        with torch.autocast(device.type, dtype=dtype):
            x = torch.randn(64, 64, device=device)
            (x @ x).sum().item()
        return True
    except (RuntimeError, TypeError):
        return False


def _matmul_tflops(
    device: torch.device, dtype: torch.dtype, n: int = 2048, reps: int = 10
) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    (a @ b).sum().item()
    t = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    c.sum().item()
    return 2 * n**3 * reps / (time.perf_counter() - t) / 1e12


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/hardware.md")
    args = p.parse_args()
    dev = pick_device("auto")
    lines = [
        f"- Date: {datetime.now(UTC):%Y-%m-%d}",
        f"- OS: {platform.platform()}",
        f"- CPU: {platform.processor() or platform.machine()}",
        f"- Python: {sys.version.split()[0]}",
        f"- PyTorch: {torch.__version__}",
        f"- Device: {dev.type}",
    ]
    if dev.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        lines += [
            f"- GPU: {prop.name}",
            f"- GPU memory: {prop.total_memory / 1e9:.1f} GB",
            f"- CUDA: {torch.version.cuda}",
        ]
    elif dev.type == "mps":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
            mem = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
                ).stdout.strip()
            )
            lines += [f"- GPU: Apple Silicon ({chip}), unified memory {mem / 2**30:.0f} GB"]
        except (OSError, ValueError):
            lines += ["- GPU: Apple Silicon"]
    lines += [
        f"- bf16 autocast works: {_autocast_ok(dev, torch.bfloat16)}",
        f"- fp16 autocast works: {_autocast_ok(dev, torch.float16)}",
        f"- Precision the training loop will use for `bf16`: {autocast_dtype(dev, 'bf16') or 'fp32'}",
        f"- fp32 matmul: {_matmul_tflops(dev, torch.float32):.2f} TFLOPS (2048x2048, rough)",
    ]
    if dev.type != "cpu":
        lines.append(
            f"- fp16 matmul: {_matmul_tflops(dev, torch.float16):.2f} TFLOPS (2048x2048, rough)"
        )
    text = (
        "# Hardware\n\nWritten by `python scripts/check_env.py`. Re-run it on any new machine.\n\n"
        + "\n".join(lines)
    )
    text += (
        "\n\n## Measured training times\n\n| Config | Machine | Steps | Wall time | Samples/s |\n"
        "|---|---|---|---|---|\n"
    )
    out = Path(args.out)
    if out.exists() and "## Measured training times" in out.read_text():
        # keep the measured-times table Paul has filled in
        text = (
            text.split("## Measured training times")[0]
            + "## Measured training times"
            + out.read_text().split("## Measured training times", 1)[1]
        )
    out.write_text(text)
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
