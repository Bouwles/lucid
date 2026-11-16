"""Build a processed, versioned dataset from validated raw recordings (SPEC 12.2)."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from lucid.data.format import (
    VERTICAL_STATES,
    cancel_opposites,
    frame_path,
    read_episode_meta,
    read_session,
    read_steps,
    unpack_buttons,
)
from lucid.data.validate import EpisodeReport, validate_session, write_report
from lucid.utils.config import BuildConfig, load_games
from lucid.utils.gitinfo import git_commit

SPLITS = ["train", "val", "test", "test_human"]


# ---------------------------------------------------------------- pure helpers (unit tested)


def summarise_actions(
    buttons: np.ndarray, look: np.ndarray, frame_skip: int, rule: str = "first_non_none"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample an episode to every `frame_skip`-th logic step.

    The kept step at index i*s stands for logic steps i*s .. i*s+s-1. Because action N first
    shows up in frame N+1, all of those sub-step actions affect the next kept frame, so the kept
    action summarises them:
      - "first": the action at the first sub-step (fine if the game holds actions).
      - "first_non_none": the first sub-step action that presses anything, else none.
      - "union": every button pressed during the interval (then left+right cancel).
    Look is the mean per-logic-step delta over the interval, so it stays in [-1, 1].

    buttons: [T] uint8 bitmasks, look: [T, 2]. Returns (kept indices [T'], buttons [T'], look [T', 2]).
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1")
    T = len(buttons)
    kept = np.arange(0, T, frame_skip)
    out_b = np.zeros(len(kept), np.uint8)
    out_l = np.zeros((len(kept), 2), np.float32)
    for j, i in enumerate(kept):
        sub_b = buttons[i : i + frame_skip]
        if rule == "first":
            b = int(sub_b[0])
        elif rule == "first_non_none":
            nz = np.flatnonzero(sub_b)
            b = int(sub_b[nz[0]]) if len(nz) else 0
        elif rule == "union":
            b = cancel_opposites(int(np.bitwise_or.reduce(sub_b)))
        else:
            raise ValueError(f"unknown action rule {rule!r}")
        out_b[j] = b
        out_l[j] = look[i : i + frame_skip].mean(0)
    return kept, out_b, out_l


def letterbox_geometry(
    src: tuple[int, int], dst: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Fit a (W, H) source inside a (W, H) target without stretching.

    Returns the resized content size (w, h) and padding (left, top, right, bottom) in pixels.
    """
    sw, sh = src
    dw, dh = dst
    scale = min(dw / sw, dh / sh)
    w, h = max(1, round(sw * scale)), max(1, round(sh * scale))
    left, top = (dw - w) // 2, (dh - h) // 2
    return (w, h), (left, top, dw - w - left, dh - h - top)


def resize_frame(frame: np.ndarray, dst: tuple[int, int], resample: str = "lanczos") -> np.ndarray:
    """uint8 [H, W, 3] to uint8 [dst_H, dst_W, 3], letterboxed with black if aspect ratios differ."""
    src = (frame.shape[1], frame.shape[0])
    if src == tuple(dst):
        return frame
    (w, h), (left, top, _, _) = letterbox_geometry(src, dst)
    method = {"lanczos": Image.Resampling.LANCZOS, "area": Image.Resampling.BOX}[resample]
    img = Image.fromarray(frame).resize((w, h), method)
    out = np.zeros((dst[1], dst[0], 3), np.uint8)
    out[top : top + h, left : left + w] = np.asarray(img)
    return out


def assign_split(key: str, is_human: bool, splits_cfg) -> str:
    """Stable, hash-based split by episode. Adding new sessions never moves old episodes."""

    def u(salt: str) -> float:
        h = hashlib.sha1(f"{splits_cfg.seed}:{salt}:{key}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    if is_human and u("human") < splits_cfg.human_holdout_fraction:
        return "test_human"
    x = u("split")
    if x < splits_cfg.val_fraction:
        return "val"
    if x < splits_cfg.val_fraction + splits_cfg.test_fraction:
        return "test"
    return "train"


def encode_labels(states: list[dict]) -> dict[str, np.ndarray]:
    def get(k, default):
        return [s.get(k, default) if s.get(k) is not None else default for s in states]

    return {
        "lane": np.asarray(get("lane", -1), np.int16),
        "vertical": np.asarray(
            [
                VERTICAL_STATES.index(v) if v in VERTICAL_STATES else VERTICAL_STATES.index("none")
                for v in get("vertical", "none")
            ],
            np.int8,
        ),
        "alive": np.asarray(get("alive", True), bool),
        "speed": np.asarray(get("speed", 0.0), np.float32),
        "distance": np.asarray(get("distance", 0.0), np.float32),
        "x": np.asarray(get("x", 0.0), np.float32),
        "y": np.asarray(get("y", 0.0), np.float32),
    }


# ---------------------------------------------------------------- per-episode work


def _process_episode(job: dict) -> dict:
    ep_dir = Path(job["ep_dir"])
    cfg = job["cfg"]
    steps = read_steps(ep_dir)
    meta = read_episode_meta(ep_dir)
    buttons = np.asarray([s.buttons for s in steps], np.uint8)
    look = np.asarray([s.look for s in steps], np.float32)
    kept, kb, kl = summarise_actions(buttons, look, cfg["frame_skip"], cfg["action_rule"])
    states = [steps[i].state for i in kept]
    episode_end = np.zeros(len(kept), bool)
    episode_end[-1] = True

    res = tuple(cfg["resolution"])
    full = tuple(cfg["full_resolution"]) if cfg["full_resolution"] else None
    frames = np.empty((len(kept), res[1], res[0], 3), np.uint8)
    full_frames = np.empty((len(kept), full[1], full[0], 3), np.uint8) if full else None
    for j, i in enumerate(kept):
        with Image.open(frame_path(ep_dir, int(i))) as img:
            fr = np.asarray(img.convert("RGB"))
        frames[j] = resize_frame(fr, res, cfg["resample"])
        if full_frames is not None:
            full_frames[j] = resize_frame(fr, full, cfg["resample"])

    out = Path(job["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".frames.npy"), frames)
    if full_frames is not None:
        np.save(out.with_suffix(".full.npy"), full_frames)
    labels = encode_labels(states)
    np.savez(
        out.with_suffix(".npz"),
        buttons=unpack_buttons(kb),
        look=kl.astype(np.float16),
        episode_end=episode_end,
        **labels,
    )
    # A small sample of pixels for measuring sigma_data on the training split.
    rng = np.random.default_rng(len(kept))
    pix = frames.reshape(-1)[rng.integers(0, frames.size, size=min(20_000, frames.size))]
    return {
        "name": out.name,
        "split": job["split"],
        "game": job["game"],
        "session": meta.session_id,
        "episode": meta.episode_id,
        "source": meta.action_source.get("name", "unknown"),
        "frames": len(kept),
        "died": bool(not labels["alive"][-1]),
        "pixel_sample": pix.astype(np.float32) / 127.5 - 1.0,
    }


# ---------------------------------------------------------------- the build


def find_sessions(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        for m in sorted(glob.glob(p)):
            if (Path(m) / "session.json").exists():
                out.append(Path(m))
    return sorted(set(out))


def build_dataset(cfg: BuildConfig, log=print) -> dict:
    games = load_games(cfg.games_file)
    for g in cfg.games:
        if g not in games:
            raise KeyError(f"game {g!r} is not in {cfg.games_file}")
    out_dir = Path(cfg.out_root) / cfg.version
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{out_dir} already exists. Dataset versions are immutable: pick a new version name."
        )
    sessions = [s for s in find_sessions(cfg.raw) if read_session(s).game_id in cfg.games]
    if not sessions:
        raise FileNotFoundError(f"no sessions for games {cfg.games} matched {cfg.raw}")

    reports: list[EpisodeReport] = []
    jobs = []
    job_cfg = {
        "frame_skip": cfg.frame_skip,
        "action_rule": cfg.action_rule,
        "resolution": cfg.resolution,
        "full_resolution": cfg.full_resolution,
        "resample": cfg.resample,
    }
    padding, look_scales = {}, {}
    for s in sessions:
        meta = read_session(s)
        reps = validate_session(s, games, cfg.max_identical_seconds)
        reports += reps
        _, pad = letterbox_geometry(tuple(meta.recording_resolution), tuple(cfg.resolution))
        padding.setdefault(meta.game_id, pad)
        for r in reps:
            if not r.ok:
                continue
            key = f"{r.session_id}/{r.episode_id}"
            split = assign_split(key, r.is_human, cfg.splits)
            ep_meta = read_episode_meta(r.path)
            if ep_meta.look_scale is not None:
                look_scales.setdefault(meta.game_id, ep_meta.look_scale)
            jobs.append(
                {
                    "ep_dir": r.path,
                    "cfg": job_cfg,
                    "split": split,
                    "game": meta.game_id,
                    "out_path": str(
                        out_dir / split / meta.game_id / f"{r.session_id}__{r.episode_id}"
                    ),
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(reports, out_dir / "validation.md")
    n_failed = sum(not r.ok for r in reports)
    log(f"validated {len(reports)} episodes: {len(reports) - n_failed} passed, {n_failed} failed")
    if not jobs:
        raise RuntimeError("every episode failed validation; see validation.md")

    if cfg.workers > 1:
        with ProcessPoolExecutor(min(cfg.workers, os.cpu_count() or 1)) as ex:
            results = list(ex.map(_process_episode, jobs, chunksize=4))
    else:
        results = [_process_episode(j) for j in jobs]

    train_pix = [r.pop("pixel_sample") for r in results if r["split"] == "train"]
    for r in results:
        r.pop("pixel_sample", None)
    sigma_data = float(np.concatenate(train_pix).std()) if train_pix else None

    split_sizes: dict = {sp: {} for sp in SPLITS}
    for r in results:
        d = split_sizes[r["split"]].setdefault(r["game"], {"episodes": 0, "frames": 0})
        d["episodes"] += 1
        d["frames"] += r["frames"]

    manifest = {
        "version": cfg.version,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "build_config": asdict(cfg),
        "source_sessions": [str(s) for s in sessions],
        "resolution": list(cfg.resolution),
        "full_resolution": list(cfg.full_resolution) if cfg.full_resolution else None,
        "frame_skip": cfg.frame_skip,
        "action_rule": cfg.action_rule,
        "games": {
            g: {
                "id": games[g]["id"],
                "padding": padding.get(g),
                "look_scale": look_scales.get(g, games[g].get("look_scale")),
            }
            for g in cfg.games
        },
        "sigma_data": sigma_data,
        "splits": split_sizes,
        "episodes": sorted(results, key=lambda r: r["name"]),
        "validation": {"passed": len(reports) - n_failed, "failed": n_failed},
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"wrote {out_dir} ({sum(r['frames'] for r in results)} frames)")
    return manifest
