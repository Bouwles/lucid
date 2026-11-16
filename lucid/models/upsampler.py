"""Diffusion upsampler: low-res frame (+ previous high-res frame) -> sharp high-res frame.

It never sees the action, so it cannot change game logic; it only adds detail.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lucid.models import sampler as samplers
from lucid.models.edm import _b, edm_coefficients, sample_training_sigma
from lucid.models.unet import ConditionalUNet


def gaussian_blur(x: torch.Tensor, sigma: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Per-sample Gaussian blur. x: [B, C, H, W], sigma: [B] in pixels (0 means no blur)."""
    B, C, H, W = x.shape
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    s = sigma.clamp(min=1e-3)[:, None].to(x.dtype)
    k = torch.exp(-(t[None] ** 2) / (2 * s**2))
    k = k / k.sum(1, keepdim=True)  # [B, 2r+1]
    k = k.repeat_interleave(C, 0)  # one kernel per (sample, channel)
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(
        F.pad(y, (radius, radius, 0, 0), mode="replicate"), k[:, None, None, :], groups=B * C
    )
    y = F.conv2d(
        F.pad(y, (0, 0, radius, radius), mode="replicate"), k[:, None, :, None], groups=B * C
    )
    return y.reshape(B, C, H, W)


class DiffusionUpsampler(nn.Module):
    """EDM denoiser over the high-res frame [B, 3, H, W], conditioned on the low-res frame
    [B, 3, h, w] (bilinearly upscaled and concatenated) and optionally the previous high-res frame.
    """

    kind = "upsampler"

    def __init__(
        self,
        unet: ConditionalUNet,
        high_resolution: tuple[int, int],
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        use_prev_frame: bool = True,
        lowres_noise_max: float = 0.3,
        lowres_blur_max: float = 1.0,
        prev_noise_max: float = 0.5,
    ):
        super().__init__()
        self.unet = unet
        self.high_resolution = tuple(high_resolution)  # (W, H)
        self.sigma_data = sigma_data
        self.p_mean, self.p_std = p_mean, p_std
        self.use_prev = use_prev_frame
        self.low_noise_max = lowres_noise_max
        self.low_blur_max = lowres_blur_max
        self.prev_noise_max = prev_noise_max

    def _cond(self, low_up, prev):
        return torch.cat([low_up, prev], 1) if self.use_prev else low_up

    def upscale(self, low: torch.Tensor) -> torch.Tensor:
        W, H = self.high_resolution
        return F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)

    def denoise(self, x, sigma, low_up, prev, game, s_low, s_prev) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = edm_coefficients(sigma.float(), self.sigma_data)
        f = self.unet(
            _b(c_in) * x,
            self._cond(low_up, prev),
            game,
            scalars={"noise": c_noise, "low_noise": s_low, "prev_noise": s_prev},
        )
        return _b(c_skip) * x + _b(c_out) * f.float()

    def training_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """batch: low [B, 3, h, w], high [B, 3, H, W], prev_high, has_prev [B], game_id (floats in [-1, 1])."""
        y, low = batch["high"], batch["low"]
        B, dev = y.shape[0], y.device
        # Degrade the low-res input so the upsampler copes with imperfect dynamics-model frames.
        low = gaussian_blur(low, torch.rand(B, device=dev) * self.low_blur_max)
        s_low = torch.rand(B, device=dev) * self.low_noise_max
        low = low + _b(s_low) * torch.randn_like(low)
        low_up = self.upscale(low)
        # At an episode start there is no previous frame; the upscaled low-res frame stands in.
        prev = torch.where(_b(batch["has_prev"]), batch["prev_high"], low_up)
        s_prev = torch.rand(B, device=dev) * self.prev_noise_max
        prev = prev + _b(s_prev) * torch.randn_like(prev)

        sigma = sample_training_sigma(B, self.p_mean, self.p_std, dev)
        x = y + _b(sigma) * torch.randn_like(y)
        c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, self.sigma_data)
        f = self.unet(
            _b(c_in) * x,
            self._cond(low_up, prev),
            batch["game_id"],
            scalars={"noise": c_noise, "low_noise": s_low, "prev_noise": s_prev},
        ).float()
        loss = F.mse_loss(f, (y - _b(c_skip) * x) / _b(c_out))
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def generate(
        self,
        low: torch.Tensor,
        game: torch.Tensor,
        prev_high: torch.Tensor | None = None,
        steps: int = 4,
        method: str = "euler",
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        prev_noise: float = 0.05,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """low [B, 3, h, w] -> high [B, 3, H, W] in [-1, 1]."""
        B, dev = low.shape[0], low.device
        low_up = self.upscale(low)
        prev = low_up if prev_high is None else prev_high
        s_low = torch.zeros(B, device=dev)
        s_prev = torch.full((B,), prev_noise, device=dev)
        if prev_noise > 0:
            prev = prev + prev_noise * torch.randn(prev.shape, generator=generator, device=dev)
        W, H = self.high_resolution
        noise = torch.randn((B, 3, H, W), generator=generator, device=dev)

        def denoise(x, s):
            return self.denoise(x, s, low_up, prev, game, s_low, s_prev)

        return samplers.sample(denoise, noise, steps, sigma_min, sigma_max, rho, method).clamp(
            -1, 1
        )
