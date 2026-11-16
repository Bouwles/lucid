"""Dataset statistics report and a sample grid for checking actions by eye."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lucid.data.dataset import WindowDataset, load_episodes
from lucid.data.format import BUTTONS, VERTICAL_STATES
from lucid.utils.video import grid, save_png, upscale


def _pct(n: int, total: int) -> str:
    return f"{100 * n / max(total, 1):.1f}%"


def action_name(multi_hot: np.ndarray) -> str:
    names = [BUTTONS[i] for i in np.flatnonzero(multi_hot)]
    return "+".join(names) if names else "none"


def dataset_report(root: str | Path, version: str, out_dir: str | Path | None = None) -> str:
    """Write stats.md, mean_frame.png and examples.png for a processed dataset."""
    out_dir = Path(out_dir or Path(root) / version)
    lines = [f"# Dataset statistics: {version}", ""]
    from lucid.data.dataset import load_manifest

    manifest = load_manifest(root, version)
    games = list(manifest["games"])
    lines += [
        f"Resolution {manifest['resolution']}, frame skip {manifest['frame_skip']}, "
        f"action rule `{manifest['action_rule']}`, measured sigma_data "
        f"{manifest['sigma_data'] if manifest['sigma_data'] is None else round(manifest['sigma_data'], 4)}.",
        "",
        "## Sizes",
        "",
        "| Split | Game | Episodes | Frames |",
        "|---|---|---|---|",
    ]
    for split, per_game in manifest["splits"].items():
        for g, d in per_game.items():
            lines.append(f"| {split} | {g} | {d['episodes']} | {d['frames']} |")

    src = Counter()
    for e in manifest["episodes"]:
        src[(e["split"], e["source"])] += e["frames"]
    lines += ["", "## Frames by action source", "", "| Split | Source | Frames |", "|---|---|---|"]
    for (split, s), n in sorted(src.items()):
        lines.append(f"| {split} | {s} | {n} |")

    train, _ = load_episodes(root, version, "train", games)
    if not train:
        lines += ["", "No training episodes."]
        text = "\n".join(lines) + "\n"
        (out_dir / "stats.md").write_text(text)
        return text

    total = sum(len(e) for e in train)
    actions = Counter()
    per_button = np.zeros(len(BUTTONS), np.int64)
    per_lane: dict[int, Counter] = defaultdict(Counter)
    vertical = Counter()
    lengths = [len(e) for e in train]
    deaths = 0
    speeds = []
    for e in train:
        per_button += e.buttons.sum(0).astype(np.int64)
        names = [action_name(b) for b in e.buttons]
        actions.update(names)
        lanes = e.labels.get("lane", np.full(len(e), -1))
        for ln, a in zip(lanes, names, strict=True):
            per_lane[int(ln)][a] += 1
        vertical.update(VERTICAL_STATES[int(v)] for v in e.labels.get("vertical", []))
        if "alive" in e.labels and not bool(e.labels["alive"][-1]):
            deaths += 1
        if "speed" in e.labels:
            speeds.append(e.labels["speed"])

    lines += ["", "## Actions (train)", "", "| Action | Frames | Share |", "|---|---|---|"]
    for a, n in actions.most_common():
        flag = "  (below 5%)" if n / total < 0.05 else ""
        lines.append(f"| {a} | {n} | {_pct(n, total)}{flag} |")
    lines += ["", "Each button, share of frames where it is held:", ""]
    lines += [
        f"- {b}: {_pct(int(n), total)}" for b, n in zip(BUTTONS, per_button, strict=True) if n
    ]

    if len(per_lane) > 1 or -1 not in per_lane:
        lines += ["", "## Actions per lane (train)", ""]
        for ln in sorted(per_lane):
            c = per_lane[ln]
            n = sum(c.values())
            top = ", ".join(f"{a} {_pct(k, n)}" for a, k in c.most_common(6))
            lines.append(f"- lane {ln} ({n} frames): {top}")

    lines += ["", "## Vertical states (train)", ""]
    lines += [f"- {v}: {_pct(n, total)}" for v, n in vertical.most_common()]
    q = np.percentile(lengths, [0, 25, 50, 75, 100]).astype(int)
    lines += [
        "",
        "## Episodes (train)",
        "",
        f"{len(train)} episodes, {deaths} end in a death ({_pct(deaths, len(train))}).",
        f"Length in frames: min {q[0]}, 25% {q[1]}, median {q[2]}, 75% {q[3]}, max {q[4]}.",
    ]
    if speeds:
        s = np.concatenate(speeds)
        sq = np.percentile(s, [0, 25, 50, 75, 100])
        lines.append(
            "Speed: "
            + ", ".join(f"{p}% {v:.2f}" for p, v in zip([0, 25, 50, 75, 100], sq, strict=True))
        )

    # Mean frame: stuck UI elements and letterbox bars show up as sharp shapes here.
    rng = np.random.default_rng(0)
    idx = rng.integers(len(train), size=min(2000, total))
    picks = [(train[j], int(rng.integers(len(train[j])))) for j in idx]
    mean = np.mean([e.frames[t] for e, t in picks], axis=0).astype(np.uint8)
    save_png(upscale(mean, 4), out_dir / "mean_frame.png")
    sample_grid(train, out_dir / "examples.png")
    lines += ["", "![mean frame](mean_frame.png)", "", "![examples](examples.png)", ""]
    text = "\n".join(lines) + "\n"
    (out_dir / "stats.md").write_text(text)
    return text


def sample_grid(episodes, path: str | Path, n: int = 12, k: int = 4, seed: int = 0) -> None:
    """Random training windows: k context frames, then the target, labelled with the action.

    Check by eye that the action visibly matches what changes between the last context frame
    and the target.
    """
    ds = WindowDataset(episodes, context_frames=k, action_history=False, pad_start="skip")
    rng = np.random.default_rng(seed)
    rows = []
    for i in rng.choice(len(ds), size=min(n, len(ds)), replace=False):
        s = ds[int(i)]
        frames = [f.permute(1, 2, 0).numpy() for f in s["context"]] + [
            s["target"].permute(1, 2, 0).numpy()
        ]
        rows.append([upscale(f, 2) for f in frames])
        h, w = rows[-1][0].shape[:2]
        lab = Image.new("RGB", (w, h), (255, 255, 255))
        name = action_name(s["buttons"].numpy().astype(np.uint8))
        ImageDraw.Draw(lab).text((2, 2), name.replace("+", "+\n"), fill=(0, 0, 0))
        rows[-1].append(np.asarray(lab))
    save_png(grid(rows, scale=1), path)
