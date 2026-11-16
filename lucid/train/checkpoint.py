"""Checkpoints: model, EMA, optimiser, scheduler, step, RNG states and the resolved config."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from lucid.utils.seed import get_rng_state, set_rng_state


def save_checkpoint(
    path: str | Path,
    *,
    model,
    ema,
    optimizer,
    scheduler,
    step: int,
    config: dict,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "rng": get_rng_state(),
            "config": config,
            "extra": extra or {},
        },
        tmp,
    )
    tmp.replace(path)  # never leave a half-written checkpoint behind


def load_checkpoint(
    path: str | Path, *, model, ema=None, optimizer=None, scheduler=None, restore_rng: bool = True
) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and "rng" in ckpt:
        set_rng_state(ckpt["rng"])
    return ckpt


class BestKeeper:
    """Keep the best `k` checkpoints by a metric where lower is better. best.pt is the top one."""

    def __init__(self, run_dir: Path, k: int = 3):
        self.run_dir = run_dir
        self.k = k
        self.index_path = run_dir / "best_checkpoints.json"
        self.entries: list[dict] = (
            json.loads(self.index_path.read_text()) if self.index_path.exists() else []
        )

    def consider(self, metric: float, step: int, save_fn) -> bool:
        if len(self.entries) >= self.k and metric >= self.entries[-1]["metric"]:
            return False
        path = self.run_dir / f"best_step{step:07d}.pt"
        save_fn(path)
        self.entries.append({"metric": metric, "step": step, "path": str(path)})
        self.entries.sort(key=lambda e: e["metric"])
        for e in self.entries[self.k :]:
            Path(e["path"]).unlink(missing_ok=True)
        self.entries = self.entries[: self.k]
        best = self.run_dir / "best.pt"
        best.unlink(missing_ok=True)
        best.symlink_to(Path(self.entries[0]["path"]).name)
        self.index_path.write_text(json.dumps(self.entries, indent=1))
        return True
