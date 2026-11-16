"""Image metrics (SPEC 13.5). Inputs are float images in [-1, 1], shape [B, 3, H, W]."""

from __future__ import annotations

import warnings

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample PSNR in dB, with data range 2 (images in [-1, 1]). Returns [B]."""
    mse = (pred.float() - target.float()).pow(2).flatten(1).mean(1).clamp(min=1e-10)
    return 10 * torch.log10(4.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample SSIM. Returns [B]."""
    return structural_similarity_index_measure(
        (pred.float() + 1) / 2, (target.float() + 1) / 2, data_range=1.0, reduction="none"
    )


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.float() - target.float()).pow(2).flatten(1).mean(1)


class LPIPS:
    """LPIPS (AlexNet) perceptual distance, lower is better. Returns [B].

    The first use downloads the AlexNet weights through torchvision. If that is not possible
    (offline CI), `LPIPS.try_create` returns None and callers skip the metric.
    """

    def __init__(self, device: torch.device | str = "cpu"):
        import lpips

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.module = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @staticmethod
    def try_create(device: torch.device | str = "cpu") -> LPIPS | None:
        try:
            return LPIPS(device)
        except Exception as e:  # network or weight-download failures
            warnings.warn(f"LPIPS unavailable ({e}); skipping it", stacklevel=2)
            return None

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.module(pred.float(), target.float()).flatten()


def frame_metrics(
    pred: torch.Tensor, target: torch.Tensor, lpips: LPIPS | None = None
) -> dict[str, torch.Tensor]:
    out = {"mse": mse(pred, target), "psnr": psnr(pred, target), "ssim": ssim(pred, target)}
    if lpips is not None:
        out["lpips"] = lpips(pred, target)
    return out


def curve_steps(n: int) -> list[int]:
    """1, 2, 4, 8, ... up to n (SPEC 13.5): the steps at which rollout curves are reported."""
    out, s = [], 1
    while s <= n:
        out.append(s)
        s *= 2
    return out
