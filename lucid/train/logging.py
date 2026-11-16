"""Run directories and metric logging (TensorBoard plus a plain JSONL file)."""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from lucid.utils.config import save_config
from lucid.utils.gitinfo import git_commit


def make_run_dir(out_dir: str | Path, run_name: str, resume: str | Path | None = None) -> Path:
    if resume:
        return Path(resume).resolve().parent
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir) / f"{stamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_info(
    run_dir: Path, cfg, dataset_version: str, device: torch.device, extra: dict | None = None
) -> dict:
    """Save the resolved config, git commit, dataset version and hardware next to the run."""
    save_config(cfg, run_dir / "config.yaml")
    info = {
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "dataset_version": dataset_version,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        **(extra or {}),
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=1)
    return info


class RunLogger:
    def __init__(self, run_dir: Path, use_tensorboard: bool = True):
        self.run_dir = run_dir
        self.jsonl = open(run_dir / "metrics.jsonl", "a")
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(run_dir / "tb"))
            except ImportError:
                self.tb = None
        self._t = time.time()

    def scalars(self, step: int, values: dict, prefix: str = "") -> None:
        clean = {
            f"{prefix}{k}": float(v)
            for k, v in values.items()
            if isinstance(v, int | float) and not math.isnan(float(v))
        }
        self.jsonl.write(
            json.dumps({"step": step, "time": round(time.time() - self._t, 2), **clean}) + "\n"
        )
        self.jsonl.flush()
        if self.tb:
            for k, v in clean.items():
                self.tb.add_scalar(k, v, step)

    def image(self, step: int, tag: str, img: np.ndarray) -> None:
        """img: uint8 [H, W, 3]."""
        if self.tb:
            self.tb.add_image(tag, img, step, dataformats="HWC")

    def video(self, step: int, tag: str, frames: list[np.ndarray], fps: int = 15) -> None:
        """frames: list of uint8 [H, W, 3]."""
        # TensorBoard videos need moviepy; without it the GIFs in samples/ are the record.
        if self.tb and importlib.util.find_spec("moviepy") is not None:
            v = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None]
            self.tb.add_video(tag, v, step, fps=fps)

    def close(self) -> None:
        self.jsonl.close()
        if self.tb:
            self.tb.close()
