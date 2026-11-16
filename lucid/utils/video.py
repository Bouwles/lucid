"""Saving frames as GIFs, MP4s and image grids. Frames are uint8 [H, W, 3] arrays."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour upscale, so small frames stay crisp instead of blurry."""
    if scale == 1:
        return frame
    return frame.repeat(scale, axis=0).repeat(scale, axis=1)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 6 * len(text) + 4, 12], fill=(0, 0, 0))
    draw.text((2, 0), text, fill=(255, 255, 255))
    return np.asarray(img)


def side_by_side(
    columns: Sequence[Sequence[np.ndarray]],
    labels: Sequence[str] | None = None,
    scale: int = 4,
    gap: int = 4,
) -> list[np.ndarray]:
    """Combine several equal-length frame sequences into one, left to right."""
    n = min(len(c) for c in columns)
    out = []
    for t in range(n):
        tiles = []
        for i, col in enumerate(columns):
            tile = upscale(col[t], scale)
            if labels:
                tile = label(tile, labels[i])
            tiles.append(tile)
            if i < len(columns) - 1:
                tiles.append(np.full((tile.shape[0], gap, 3), 255, np.uint8))
        out.append(np.concatenate(tiles, axis=1))
    return out


def grid(rows: Sequence[Sequence[np.ndarray]], scale: int = 2, gap: int = 2) -> np.ndarray:
    """Tile a 2D list of frames into one image."""
    h, w = rows[0][0].shape[:2]
    h, w = h * scale, w * scale
    ncols = max(len(r) for r in rows)
    img = np.full((len(rows) * (h + gap) - gap, ncols * (w + gap) - gap, 3), 255, dtype=np.uint8)
    for i, r in enumerate(rows):
        for j, f in enumerate(r):
            img[i * (h + gap) : i * (h + gap) + h, j * (w + gap) : j * (w + gap) + w] = upscale(
                f, scale
            )
    return img


def save_gif(frames: Sequence[np.ndarray], path: str | Path, fps: int = 15) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), duration=1000 / fps, loop=0)


def save_mp4(frames: Sequence[np.ndarray], path: str | Path, fps: int = 15) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)


def save_png(frame: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)
