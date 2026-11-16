"""Validate raw recordings. Failed episodes are reported and excluded, never repaired."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from lucid.data.format import (
    VERTICAL_STATES,
    button_bit,
    frame_path,
    list_episodes,
    list_frames,
    read_episode_meta,
    read_session,
    read_steps,
)


@dataclass
class EpisodeReport:
    session_id: str
    episode_id: str
    path: str
    ok: bool
    num_steps: int = 0
    is_human: bool = False
    reasons: list[str] = field(default_factory=list)


def allowed_button_mask(game_cfg: dict | None) -> int:
    """Bits this game is allowed to press, from the `actions` mapping in configs/games.yaml."""
    if not game_cfg or not game_cfg.get("actions"):
        return 0xFF
    mask = 0
    for unified in game_cfg["actions"].values():
        names = unified if isinstance(unified, list) else [unified]
        for n in names:
            mask |= button_bit(n)
    return mask


def validate_episode(
    ep_dir: str | Path,
    expected_resolution: tuple[int, int],
    game_cfg: dict | None = None,
    logic_timestep: float = 1 / 60,
    max_identical_seconds: float = 3.0,
) -> EpisodeReport:
    ep_dir = Path(ep_dir)
    rep = EpisodeReport(
        session_id=ep_dir.parent.parent.name, episode_id=ep_dir.name, path=str(ep_dir), ok=False
    )
    reasons = rep.reasons
    try:
        meta = read_episode_meta(ep_dir)
        rep.is_human = meta.is_human
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
        reasons.append(f"meta.json unreadable: {e}")
        return rep
    if meta.corrupted:
        reasons.append(f"marked corrupted: {meta.corruption_reason}")
    try:
        steps = read_steps(ep_dir)
    except (OSError, KeyError, ValueError, IndexError, TypeError, json.JSONDecodeError) as e:
        reasons.append(f"steps.jsonl unreadable: {e}")
        return rep
    rep.num_steps = len(steps)
    if not steps:
        reasons.append("no steps")
        return rep

    frames = list_frames(ep_dir)
    if len(frames) != len(steps):
        reasons.append(f"{len(frames)} frames but {len(steps)} step lines")
    if meta.num_steps != len(steps):
        reasons.append(f"meta.num_steps={meta.num_steps} but {len(steps)} step lines")
    if [s.step for s in steps] != list(range(len(steps))):
        reasons.append("step indices are not contiguous from 0")

    ends = [i for i, s in enumerate(steps) if s.episode_end]
    if ends != [len(steps) - 1]:
        reasons.append(f"episode_end must be set exactly once on the last step, got {ends}")

    mask = allowed_button_mask(game_cfg)
    for s in steps:
        if not 0 <= s.buttons <= 255:
            reasons.append(f"step {s.step}: buttons {s.buttons} outside 0-255")
            break
        if s.buttons & ~mask:
            reasons.append(
                f"step {s.step}: buttons {s.buttons:08b} use bits this game does not map"
            )
            break
        if not all(-1.0 <= v <= 1.0 for v in s.look):
            reasons.append(f"step {s.step}: look {s.look} outside [-1, 1]")
            break
        v = s.state.get("vertical")
        if v is not None and v not in VERTICAL_STATES:
            reasons.append(f"step {s.step}: unknown vertical state {v!r}")
            break

    # Frame checks. Resolution comes from the PNG header (cheap). Identical frames are spotted
    # by hashing file bytes: the recorder's encoder is deterministic, so equal pixels give equal
    # bytes. That can miss duplicates, never invent them.
    max_run = max(1, int(round(max_identical_seconds / logic_timestep)))
    prev_hash, run = None, 0
    for i in range(min(len(frames), len(steps))):
        p = frame_path(ep_dir, i)
        if not p.exists():
            reasons.append(f"missing frame {p.name}")
            break
        try:
            with Image.open(p) as img:
                if img.size != tuple(expected_resolution):
                    reasons.append(
                        f"{p.name}: size {img.size}, expected {tuple(expected_resolution)}"
                    )
                    break
                if img.mode not in ("RGB", "RGBA"):
                    reasons.append(f"{p.name}: mode {img.mode}, expected RGB")
                    break
        except OSError as e:
            reasons.append(f"{p.name}: unreadable ({e})")
            break
        h = hashlib.blake2b(p.read_bytes(), digest_size=16).digest()
        alive = steps[i].state.get("alive", True)
        run = run + 1 if (h == prev_hash and alive) else 1
        prev_hash = h
        if run > max_run:
            reasons.append(
                f"{run} identical frames in a row while alive around step {i} (frozen game?)"
            )
            break

    rep.ok = not reasons
    return rep


def validate_session(
    session_dir: str | Path, games: dict[str, dict], max_identical_seconds: float = 3.0
) -> list[EpisodeReport]:
    session = read_session(session_dir)
    if session.game_id not in games:
        return [
            EpisodeReport(
                session.session_id,
                ep.name,
                str(ep),
                False,
                reasons=[f"game_id {session.game_id!r} not in the games registry"],
            )
            for ep in list_episodes(session_dir)
        ]
    game_cfg = games[session.game_id]
    return [
        validate_episode(
            ep,
            session.recording_resolution,
            game_cfg,
            session.logic_timestep,
            max_identical_seconds,
        )
        for ep in list_episodes(session_dir)
    ]


def write_report(reports: list[EpisodeReport], path: str | Path) -> None:
    """Write a Markdown report and a JSON file next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = [r for r in reports if r.ok]
    failed = [r for r in reports if not r.ok]
    lines = [
        "# Validation report",
        "",
        f"{len(passed)} passed, {len(failed)} failed, {len(reports)} episodes in total.",
        "",
    ]
    if failed:
        lines += ["| Session | Episode | Reasons |", "|---|---|---|"]
        for r in failed:
            lines.append(f"| {r.session_id} | {r.episode_id} | {'; '.join(r.reasons)} |")
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps([asdict(r) for r in reports], indent=1))
