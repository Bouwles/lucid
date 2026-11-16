"""YAML configs loaded into typed dataclasses.

A config file may start with `base: <path>` or `base: [<path>, ...]`. Bases are loaded first
(paths relative to the file) and the file's own keys are merged on top. Command-line overrides
use dotted keys: `--set train.lr=3e-4 --set data.games=[runner,platformer]`.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


# ---------------------------------------------------------------- training run config


@dataclass
class DataConfig:
    version: str = "runner_v1"
    root: str = "data/processed"
    games: list[str] = field(default_factory=lambda: ["runner"])
    # "equal_per_game" or a {game: weight} mapping. Weights are per game, not per frame.
    mixture: str | dict[str, float] = "equal_per_game"
    resolution: list[int] = field(default_factory=lambda: [64, 64])  # [W, H], must match manifest
    context_frames: int = 4
    action_history: bool = True
    pad_start: str = "repeat"  # "repeat" the first frame, or "skip" the first k targets
    frame_skip: int = 4  # must match the dataset manifest; checked at load time
    num_workers: int = 4
    preload: bool = False  # load all frames into RAM (only for small datasets)
    rollout_length: int = 64  # sequence length for rollout evaluation


@dataclass
class ModelConfig:
    type: str = "edm_unet"  # edm_unet | baseline | upsampler
    base_channels: int = 96
    channel_mults: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    res_blocks_per_level: int = 2
    attention_resolutions: list[int] = field(default_factory=lambda: [8])
    attention_heads: int = 4
    cond_dim: int = 256
    dropout: float = 0.0
    num_games: int = 1
    sigma_data: float = 0.5
    p_mean: float = -1.2  # EDM training noise distribution: ln(sigma) ~ N(p_mean, p_std^2)
    p_std: float = 1.2
    p_uncond_action: float = 0.1
    lpips_weight: float = 0.0  # baseline only
    motion_weight: float = 0.0  # extra loss weight on pixels that change (helps controls)
    # upsampler only
    low_resolution: list[int] = field(default_factory=lambda: [64, 64])
    use_prev_frame: bool = True
    lowres_noise_max: float = 0.3
    lowres_blur_max: float = 1.0


@dataclass
class ContextNoiseConfig:
    enabled: bool = True
    p_clean: float = 0.2
    max_sigma: float = 0.7


@dataclass
class TrainConfig:
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    schedule: str = "cosine"  # cosine | constant
    max_steps: int = 100_000
    ema_decay: float = 0.9995
    grad_clip: float = 1.0
    precision: str = "bf16"  # bf16 | fp16 | fp32 (falls back to fp32 if unsupported)
    device: str = "auto"
    log_every: int = 50
    val_every: int = 2000
    val_batches: int = 8
    val_rollout_steps: int = 64
    ckpt_every: int = 5000
    keep_best: int = 3
    out_dir: str = "runs"


@dataclass
class SamplerConfig:
    type: str = "euler"  # euler | heun
    steps: int = 3
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    inference_context_noise: float = 0.1
    refinement: bool = False
    refinement_sigma: float = 0.5
    cfg_scale: float | None = None


@dataclass
class RunConfig:
    run_name: str = "run"
    seed: int = 0
    games_file: str = "configs/games.yaml"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    context_noise: ContextNoiseConfig = field(default_factory=ContextNoiseConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


# ---------------------------------------------------------------- dataset build config


@dataclass
class SplitConfig:
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    human_holdout_fraction: float = 0.5  # share of human episodes kept only for test_human
    seed: int = 0


@dataclass
class BuildConfig:
    version: str = "runner_v1"
    raw: list[str] = field(default_factory=lambda: ["data/raw/*"])  # session dirs or globs
    out_root: str = "data/processed"
    games_file: str = "configs/games.yaml"
    games: list[str] = field(default_factory=lambda: ["runner"])
    frame_skip: int = 4
    action_rule: str = "first_non_none"  # first | first_non_none | union
    resolution: list[int] = field(default_factory=lambda: [64, 64])  # [W, H]
    full_resolution: list[int] | None = None  # e.g. [256, 256] to keep upsampler targets
    resample: str = "lanczos"  # lanczos | area
    max_identical_seconds: float = 3.0  # longer runs of identical frames (while alive) fail
    splits: SplitConfig = field(default_factory=SplitConfig)
    workers: int = 4


# ---------------------------------------------------------------- loading machinery


def _load_yaml_with_bases(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    bases = raw.pop("base", None)
    if bases is None:
        return raw
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, _load_yaml_with_bases((path.parent / b).resolve()))
    return deep_merge(merged, raw)


def deep_merge(a: dict, b: dict) -> dict:
    """Return a copy of `a` with `b` merged in. Nested dicts merge; everything else replaces."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not k == "mixture":
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _float_or_str(s: str):
    # YAML 1.1 reads "3e-4" (no dot) as a string; treat it as the number it obviously is.
    try:
        return float(s)
    except ValueError:
        return s


def _parse_value(s: str):
    v = yaml.safe_load(s)
    if isinstance(v, str):
        return _float_or_str(v) if v.strip().lower() not in ("nan", "inf", "-inf") else v
    if isinstance(v, list):
        return [
            _float_or_str(x) if isinstance(x, str) and any(c.isdigit() for c in x) else x for x in v
        ]
    return v


def parse_overrides(overrides: list[str] | None) -> dict:
    """Turn ["train.lr=3e-4", "data.games=[a,b]"] into a nested dict. Values are parsed as YAML."""
    out: dict = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return out


def from_dict(cls: type[T], d: dict, path: str = "") -> T:
    """Build dataclass `cls` from a dict, recursing into nested dataclasses.

    Unknown keys raise, so a typo in a YAML file fails loudly instead of being ignored.
    """
    if not isinstance(d, dict):
        raise TypeError(f"{path or cls.__name__}: expected a mapping, got {type(d).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown config keys in {path or cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for k, v in d.items():
        kwargs[k] = _coerce(hints[k], v, f"{path}.{k}" if path else k)
    return cls(**kwargs)


def _coerce(tp: Any, v: Any, path: str) -> Any:
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return from_dict(tp, v, path)
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        if v is None:
            return None
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        for a in args:
            try:
                return _coerce(a, v, path)
            except (TypeError, ValueError, KeyError):
                continue
        raise TypeError(f"{path}: value {v!r} does not match {tp}")
    if origin is list:
        if not isinstance(v, list | tuple):
            raise TypeError(f"{path}: expected a list, got {v!r}")
        (inner,) = typing.get_args(tp) or (Any,)
        return [_coerce(inner, x, path) for x in v]
    if origin is dict:
        if not isinstance(v, dict):
            raise TypeError(f"{path}: expected a mapping, got {v!r}")
        return dict(v)
    if tp is float and isinstance(v, str):
        v = _float_or_str(v)
    if tp is float and isinstance(v, int | float) and not isinstance(v, bool):
        return float(v)
    if tp is int and isinstance(v, int) and not isinstance(v, bool):
        return v
    if tp is int and isinstance(v, float) and v.is_integer():
        return int(v)
    if tp is bool and isinstance(v, bool):
        return v
    if tp is str and isinstance(v, str):
        return v
    if tp is Any:
        return v
    raise TypeError(f"{path}: expected {getattr(tp, '__name__', tp)}, got {v!r}")


def load_config(
    path: str | Path, overrides: list[str] | None = None, cls: type[T] = RunConfig
) -> T:
    raw = _load_yaml_with_bases(Path(path).resolve())
    raw = deep_merge(raw, parse_overrides(overrides))
    return from_dict(cls, raw)


def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


def save_config(cfg: Any, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)


def load_games(games_file: str | Path) -> dict[str, dict]:
    """Read the game registry. Returns {game_name: {"id": int, "actions": {...}, ...}}."""
    with open(games_file) as f:
        games = (yaml.safe_load(f) or {}).get("games") or {}
    ids = [g["id"] for g in games.values()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate game ids in {games_file}")
    return games
