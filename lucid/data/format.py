"""Reading and writing the raw recording format (docs/formats.md, format_version 1)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

FORMAT_VERSION = 1
BUTTONS = ["left", "right", "up", "down", "jump", "action_a", "action_b", "reserved"]
NUM_BUTTONS = len(BUTTONS)
VERTICAL_STATES = ["grounded", "jumping", "sliding", "falling", "none"]
STATE_KEYS = ["lane", "x", "y", "vertical", "alive", "speed", "distance"]


def button_bit(name: str) -> int:
    return 1 << BUTTONS.index(name)


def unpack_buttons(buttons: int | np.ndarray) -> np.ndarray:
    """Bitmask int(s) to multi-hot uint8. Shape [...] -> [..., 8]."""
    b = np.asarray(buttons, dtype=np.uint8)
    return ((b[..., None] >> np.arange(NUM_BUTTONS, dtype=np.uint8)) & 1).astype(np.uint8)


def pack_buttons(multi_hot: np.ndarray) -> np.ndarray:
    """Multi-hot [..., 8] to bitmask ints [...]."""
    m = np.asarray(multi_hot).astype(np.uint16)
    return (m << np.arange(NUM_BUTTONS, dtype=np.uint16)).sum(-1).astype(np.uint8)


def cancel_opposites(buttons: int) -> int:
    """Left+right together means neither; same for up+down (docs/formats.md)."""
    for a, b in (("left", "right"), ("up", "down")):
        mask = button_bit(a) | button_bit(b)
        if buttons & mask == mask:
            buttons &= ~mask
    return buttons


@dataclass
class SessionMeta:
    session_id: str
    game_id: str
    recording_resolution: tuple[int, int]  # (W, H)
    logic_timestep: float
    recorder_version: str = ""
    game_build_version: str = ""
    created_utc: str = ""
    unity_version: str | None = None
    frame_format: str = "png"
    format_version: int = FORMAT_VERSION

    @classmethod
    def from_json(cls, d: dict) -> SessionMeta:
        return cls(
            session_id=d["session_id"],
            game_id=d["game_id"],
            recording_resolution=tuple(d["recording_resolution"]),
            logic_timestep=float(d["logic_timestep"]),
            recorder_version=d.get("recorder_version", ""),
            game_build_version=d.get("game_build_version", ""),
            created_utc=d.get("created_utc", ""),
            unity_version=d.get("unity_version"),
            frame_format=d.get("frame_format", "png"),
            format_version=int(d.get("format_version", FORMAT_VERSION)),
        )

    def to_json(self) -> dict:
        return {
            "format_version": self.format_version,
            "session_id": self.session_id,
            "game_id": self.game_id,
            "created_utc": self.created_utc,
            "recorder_version": self.recorder_version,
            "game_build_version": self.game_build_version,
            "unity_version": self.unity_version,
            "recording_resolution": list(self.recording_resolution),
            "logic_timestep": self.logic_timestep,
            "frame_format": self.frame_format,
        }


@dataclass
class EpisodeMeta:
    episode_id: str
    session_id: str
    game_id: str
    seed: int
    num_steps: int
    action_source: dict = field(default_factory=lambda: {"name": "unknown", "settings": {}})
    recording_resolution: tuple[int, int] = (256, 256)
    logic_timestep: float = 1 / 60
    look_scale: list[float] | None = None
    corrupted: bool = False
    corruption_reason: str | None = None
    game_build_version: str = ""
    recorder_version: str = ""
    format_version: int = FORMAT_VERSION

    @property
    def is_human(self) -> bool:
        return self.action_source.get("name") == "human"

    @classmethod
    def from_json(cls, d: dict) -> EpisodeMeta:
        return cls(
            episode_id=d["episode_id"],
            session_id=d["session_id"],
            game_id=d["game_id"],
            seed=int(d["seed"]),
            num_steps=int(d["num_steps"]),
            action_source=d.get("action_source") or {"name": "unknown", "settings": {}},
            recording_resolution=tuple(d.get("recording_resolution", (256, 256))),
            logic_timestep=float(d.get("logic_timestep", 1 / 60)),
            look_scale=d.get("look_scale"),
            corrupted=bool(d.get("corrupted", False)),
            corruption_reason=d.get("corruption_reason"),
            game_build_version=d.get("game_build_version", ""),
            recorder_version=d.get("recorder_version", ""),
            format_version=int(d.get("format_version", FORMAT_VERSION)),
        )

    def to_json(self) -> dict:
        return {
            "format_version": self.format_version,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "game_id": self.game_id,
            "seed": self.seed,
            "game_build_version": self.game_build_version,
            "recorder_version": self.recorder_version,
            "action_source": self.action_source,
            "recording_resolution": list(self.recording_resolution),
            "logic_timestep": self.logic_timestep,
            "look_scale": self.look_scale,
            "num_steps": self.num_steps,
            "corrupted": self.corrupted,
            "corruption_reason": self.corruption_reason,
        }


@dataclass
class Step:
    step: int
    time: float
    buttons: int
    look: tuple[float, float]
    state: dict
    episode_end: bool

    @classmethod
    def from_json(cls, d: dict) -> Step:
        return cls(
            step=int(d["step"]),
            time=float(d["time"]),
            buttons=int(d["buttons"]),
            look=(float(d["look"][0]), float(d["look"][1])),
            state=d.get("state") or {},
            episode_end=bool(d["episode_end"]),
        )

    def to_json(self) -> dict:
        return {
            "step": self.step,
            "time": round(self.time, 6),
            "buttons": self.buttons,
            "look": [round(self.look[0], 6), round(self.look[1], 6)],
            "state": self.state,
            "episode_end": self.episode_end,
        }


# ---------------------------------------------------------------- reading


def read_session(session_dir: str | Path) -> SessionMeta:
    with open(Path(session_dir) / "session.json") as f:
        return SessionMeta.from_json(json.load(f))


def list_episodes(session_dir: str | Path) -> list[Path]:
    return sorted(p for p in (Path(session_dir) / "episodes").glob("ep_*") if p.is_dir())


def read_episode_meta(ep_dir: str | Path) -> EpisodeMeta:
    with open(Path(ep_dir) / "meta.json") as f:
        return EpisodeMeta.from_json(json.load(f))


def read_steps(ep_dir: str | Path) -> list[Step]:
    steps = []
    with open(Path(ep_dir) / "steps.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(Step.from_json(json.loads(line)))
    return steps


def frame_path(ep_dir: str | Path, step: int) -> Path:
    return Path(ep_dir) / "frames" / f"{step:06d}.png"


def list_frames(ep_dir: str | Path) -> list[Path]:
    return sorted((Path(ep_dir) / "frames").glob("*.png"))


def read_frame(path: str | Path) -> np.ndarray:
    """PNG to uint8 [H, W, 3]. Alpha, if any, is dropped."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def iter_frames(ep_dir: str | Path, n: int) -> Iterator[np.ndarray]:
    for i in range(n):
        yield read_frame(frame_path(ep_dir, i))


# ---------------------------------------------------------------- writing (fixtures and tools)


def write_session(session_dir: str | Path, meta: SessionMeta) -> None:
    session_dir = Path(session_dir)
    (session_dir / "episodes").mkdir(parents=True, exist_ok=True)
    with open(session_dir / "session.json", "w") as f:
        json.dump(meta.to_json(), f, indent=2)


def write_episode(
    session_dir: str | Path, meta: EpisodeMeta, steps: list[Step], frames: list[np.ndarray]
) -> Path:
    ep_dir = Path(session_dir) / "episodes" / meta.episode_id
    (ep_dir / "frames").mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(frame_path(ep_dir, i), optimize=False)
    with open(ep_dir / "steps.jsonl", "w") as f:
        for s in steps:
            f.write(json.dumps(s.to_json()) + "\n")
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta.to_json(), f, indent=2)
    return ep_dir
