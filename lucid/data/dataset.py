"""PyTorch datasets over a processed LUCID dataset.

Everything here returns uint8 frames on the CPU. Convert to float in [-1, 1] on the GPU with
`to_model_space`, which keeps the data loader cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

LABEL_KEYS = ["lane", "vertical", "alive", "speed", "distance", "x", "y"]


def to_model_space(x: torch.Tensor) -> torch.Tensor:
    """uint8 [..., 3, H, W] in 0..255 to float in [-1, 1]."""
    return x.float().div(127.5).sub(1.0)


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """float in [-1, 1] to uint8, clamping out-of-range values."""
    return x.add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)


def load_manifest(root: str | Path, version: str) -> dict:
    with open(Path(root) / version / "manifest.json") as f:
        return json.load(f)


@dataclass
class Episode:
    name: str
    game: str
    game_id: int
    frames: np.ndarray  # uint8 [T, H, W, 3], usually memory-mapped
    buttons: np.ndarray  # uint8 [T, 8]
    look: np.ndarray  # float32 [T, 2]
    labels: dict[str, np.ndarray]
    full_frames: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.frames)


def load_episodes(
    root: str | Path,
    version: str,
    split: str,
    games: list[str],
    preload: bool = False,
    with_full: bool = False,
) -> tuple[list[Episode], dict]:
    manifest = load_manifest(root, version)
    base = Path(root) / version / split
    episodes = []
    for g in games:
        if g not in manifest["games"]:
            raise KeyError(f"game {g!r} is not in dataset {version}")
        gid = manifest["games"][g]["id"]
        for npz in sorted((base / g).glob("*.npz")):
            stem = npz.with_suffix("")
            frames = np.load(stem.with_suffix(".frames.npy"), mmap_mode=None if preload else "r")
            full = None
            if with_full:
                fp = stem.with_suffix(".full.npy")
                if not fp.exists():
                    raise FileNotFoundError(f"{fp} missing: build with full_resolution set")
                full = np.load(fp, mmap_mode=None if preload else "r")
            with np.load(npz) as d:
                episodes.append(
                    Episode(
                        name=stem.name,
                        game=g,
                        game_id=gid,
                        frames=frames,
                        buttons=d["buttons"].astype(np.uint8),
                        look=d["look"].astype(np.float32),
                        labels={k: d[k] for k in LABEL_KEYS if k in d},
                        full_frames=full,
                    )
                )
    return episodes, manifest


def _chw(frames: np.ndarray) -> torch.Tensor:
    """uint8 [..., H, W, 3] numpy to uint8 [..., 3, H, W] tensor."""
    # Copy: memory-mapped arrays are read-only, and torch tensors must own writable memory.
    return torch.from_numpy(np.array(frames, copy=True)).movedim(-1, -3)


class WindowDataset(Dataset):
    """Windows of `k` context frames plus the next frame, never crossing an episode boundary.

    Sample t (the target index) uses context frames t-k .. t-1 and the action taken at t-1
    (whose effect first shows in frame t). Near an episode start, `pad_start="repeat"` repeats
    the first frame (and pads actions with "none"), which keeps restarts learnable; `"skip"`
    drops targets that have fewer than k real context frames.

    Returns a dict:
      context  uint8 [k, 3, H, W]
      target   uint8 [3, H, W]
      buttons  float [8], or [k, 8] with action_history
      look     float [2], or [k, 2] with action_history
      game_id  long []
      labels   dict of label values at the target step
    """

    def __init__(
        self,
        episodes: list[Episode],
        context_frames: int = 4,
        action_history: bool = True,
        pad_start: str = "repeat",
    ):
        if pad_start not in ("repeat", "skip"):
            raise ValueError("pad_start must be 'repeat' or 'skip'")
        self.episodes = episodes
        self.k = context_frames
        self.action_history = action_history
        first = 1 if pad_start == "repeat" else context_frames
        index = [(e, t) for e, ep in enumerate(episodes) for t in range(first, len(ep))]
        self.index = np.asarray(index, dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.index)

    def game_of(self, i: int) -> str:
        return self.episodes[self.index[i, 0]].game

    def __getitem__(self, i: int) -> dict:
        e, t = (int(v) for v in self.index[i])
        ep = self.episodes[e]
        ctx_idx = np.arange(t - self.k, t)
        valid = ctx_idx >= 0
        frame_idx = np.clip(ctx_idx, 0, None)
        context = ep.frames[frame_idx]
        buttons = ep.buttons[frame_idx].astype(np.float32) * valid[:, None]
        look = ep.look[frame_idx] * valid[:, None]
        if not self.action_history:
            buttons, look = buttons[-1], look[-1]
        return {
            "context": _chw(context),
            "target": _chw(ep.frames[t]),
            "buttons": torch.from_numpy(np.ascontiguousarray(buttons)),
            "look": torch.from_numpy(np.ascontiguousarray(look)).float(),
            "game_id": torch.tensor(ep.game_id, dtype=torch.long),
            "labels": {k: torch.as_tensor(v[t].item()) for k, v in ep.labels.items()},
        }


class SequenceDataset(Dataset):
    """Long sequences for rollout evaluation: k context frames plus `length` future steps.

    Returns frames uint8 [k + length, 3, H, W], buttons float [k + length, 8],
    look float [k + length, 2], game_id, and labels dict of [k + length] tensors. The action
    at index j leads to frame j + 1.
    """

    def __init__(
        self, episodes: list[Episode], context_frames: int, length: int, stride: int | None = None
    ):
        self.episodes = episodes
        self.k = context_frames
        self.length = length
        n = context_frames + length
        stride = stride or n
        self.index = [
            (e, s) for e, ep in enumerate(episodes) for s in range(0, len(ep) - n + 1, stride)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        e, s = self.index[i]
        ep = self.episodes[e]
        sl = slice(s, s + self.k + self.length)
        return {
            "frames": _chw(ep.frames[sl]),
            "buttons": torch.from_numpy(ep.buttons[sl].astype(np.float32)),
            "look": torch.from_numpy(ep.look[sl].copy()),
            "game_id": torch.tensor(ep.game_id, dtype=torch.long),
            "labels": {k: torch.from_numpy(np.asarray(v[sl])) for k, v in ep.labels.items()},
            "episode": ep.name,
            "start": s,
        }


class UpsampleDataset(Dataset):
    """Pairs of (low-res, high-res) real frames plus the previous high-res frame.

    Returns low uint8 [3, h, w], high uint8 [3, H, W], prev_high uint8 [3, H, W] (a copy of high's
    first frame at an episode start, with has_prev=False), game_id.
    """

    def __init__(self, episodes: list[Episode]):
        self.episodes = episodes
        self.index = [(e, t) for e, ep in enumerate(episodes) for t in range(len(ep))]

    def __len__(self) -> int:
        return len(self.index)

    def game_of(self, i: int) -> str:
        return self.episodes[self.index[i][0]].game

    def __getitem__(self, i: int) -> dict:
        e, t = self.index[i]
        ep = self.episodes[e]
        assert ep.full_frames is not None
        return {
            "low": _chw(ep.frames[t]),
            "high": _chw(ep.full_frames[t]),
            "prev_high": _chw(ep.full_frames[max(t - 1, 0)]),
            "has_prev": torch.tensor(t > 0),
            "game_id": torch.tensor(ep.game_id, dtype=torch.long),
        }


def mixture_sampler(
    dataset: WindowDataset | UpsampleDataset,
    mixture: str | dict[str, float],
    num_samples: int,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Sample games by game weight, not by frame count, so small games are not drowned out."""
    games = [dataset.game_of(i) for i in range(len(dataset))]
    names = sorted(set(games))
    if mixture == "equal_per_game":
        weights = {g: 1.0 for g in names}
    elif isinstance(mixture, dict):
        weights = {g: float(mixture.get(g, 0.0)) for g in names}
    else:
        raise ValueError(f"unknown mixture {mixture!r}")
    counts = {g: games.count(g) for g in names}
    w = torch.tensor([weights[g] / counts[g] for g in games], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=num_samples, replacement=True, generator=generator)


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        if k == "labels":
            out[k] = {lk: torch.stack([b[k][lk] for b in batch]) for lk in batch[0][k]}
        elif isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out
