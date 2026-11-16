"""Long-horizon stability without ground truth (SPEC 17.2)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def sharpness(frames: torch.Tensor) -> torch.Tensor:
    """Variance of the Laplacian of the grey image. frames: float [N, 3, H, W] in [-1, 1]. Returns [N].

    Blur makes it collapse, noise makes it explode.
    """
    grey = frames.float().mean(1, keepdim=True)
    lap = F.conv2d(grey, _LAPLACE.to(frames.device)[None, None], padding=1)
    return lap[..., 1:-1, 1:-1].flatten(1).var(1)


def colour_stats(frames: torch.Tensor) -> torch.Tensor:
    """Per-channel mean and std: float [N, 3, H, W] -> [N, 6]."""
    f = frames.float().flatten(2)
    return torch.cat([f.mean(-1), f.std(-1)], dim=1)


def frame_features(frames: torch.Tensor) -> torch.Tensor:
    """[N, 7]: sharpness then the six colour statistics."""
    return torch.cat([sharpness(frames)[:, None], colour_stats(frames)], dim=1)


def real_ranges(
    real_frames: torch.Tensor, lo: float = 0.5, hi: float = 99.5
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile range of each feature over real frames: ([7], [7])."""
    feats = frame_features(real_frames).cpu().numpy()
    return np.percentile(feats, lo, axis=0), np.percentile(feats, hi, axis=0)


def survival_time(
    features: np.ndarray, lo: np.ndarray, hi: np.ndarray, tolerance: float = 0.1, patience: int = 10
) -> int:
    """First step where any feature has been outside the real range (widened by `tolerance` of
    its width) for `patience` steps in a row. features: [T, 7]. Returns T if it never breaks.
    """
    width = hi - lo
    lo_t, hi_t = lo - tolerance * width, hi + tolerance * width
    out = ((features < lo_t) | (features > hi_t)).any(1)
    run = 0
    for t, bad in enumerate(out):
        run = run + 1 if bad else 0
        if run >= patience:
            return t - patience + 1
    return len(features)
