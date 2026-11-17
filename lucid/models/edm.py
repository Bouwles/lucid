"""EDM diffusion world model, following Karras et al. 2022, "Elucidating the Design
Space of Diffusion-Based Generative Models", Table 1 (preconditioning, training noise
distribution and loss weighting).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from lucid.models import sampler as samplers
from lucid.models.unet import ConditionalUNet


def edm_coefficients(sigma: torch.Tensor, sigma_data: float):
    """EDM preconditioning for noise levels `sigma` [B]. Returns (c_skip, c_out, c_in, c_noise), each [B].

    c_skip = sd^2 / (sigma^2 + sd^2)
    c_out  = sigma * sd / sqrt(sigma^2 + sd^2)
    c_in   = 1 / sqrt(sigma^2 + sd^2)
    c_noise = ln(sigma) / 4
    """
    sd2 = sigma_data**2
    s2 = sigma**2
    c_skip = sd2 / (s2 + sd2)
    c_out = sigma * sigma_data / (s2 + sd2).sqrt()
    c_in = 1 / (s2 + sd2).sqrt()
    c_noise = sigma.log() / 4
    return c_skip, c_out, c_in, c_noise


def loss_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """lambda(sigma) = (sigma^2 + sd^2) / (sigma * sd)^2. Note lambda * c_out^2 = 1."""
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2


def sample_training_sigma(
    n: int, p_mean: float, p_std: float, device=None, generator: torch.Generator | None = None
) -> torch.Tensor:
    """ln(sigma) ~ N(p_mean, p_std^2). Returns [n]."""
    return (torch.randn(n, device=device, generator=generator) * p_std + p_mean).exp()


def _b(x: torch.Tensor) -> torch.Tensor:
    return x[:, None, None, None]


class DiffusionWorldModel(nn.Module):
    """Action-conditioned EDM denoiser over the next frame, given k context frames.

    All images are float in [-1, 1]: context [B, k, 3, H, W], target [B, 3, H, W].
    Actions: buttons [B, 8] or [B, k, 8], look [B, 2] or [B, k, 2]. game: [B] ids or [B, G] weights.
    """

    kind = "edm_unet"

    def __init__(
        self,
        unet: ConditionalUNet,
        context_frames: int,
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        p_uncond_action: float = 0.1,
        context_noise_enabled: bool = True,
        context_noise_p_clean: float = 0.2,
        context_noise_max: float = 0.7,
        motion_weight: float = 0.0,
        action_channels: bool = False,
    ):
        super().__init__()
        self.unet = unet
        self.k = context_frames
        self.sigma_data = sigma_data
        self.p_mean, self.p_std = p_mean, p_std
        self.p_uncond = p_uncond_action
        self.ctx_enabled = context_noise_enabled
        self.ctx_p_clean = context_noise_p_clean
        self.ctx_max = context_noise_max
        self.motion_weight = motion_weight
        self.action_channels = action_channels

    # --------------------------------------------------------------- the denoiser

    @staticmethod
    def action_planes(buttons, look, drop_action, H: int, W: int) -> torch.Tensor:
        """Actions as constant image channels: [B, A, H, W] with A = (history) * 10. Dropped actions are zeros."""
        a = torch.cat([buttons.float(), look.float()], dim=-1).flatten(1)  # [B, A]
        if drop_action is not None:
            a = a * (~drop_action)[:, None].to(a.dtype)
        return a[:, :, None, None].expand(-1, -1, H, W)

    def raw(self, x_in, c_noise, context, buttons, look, game, sigma_ctx, drop_action=None):
        cond = context.flatten(1, 2)
        if self.action_channels:
            # A direct path for the controls into every convolution, alongside the embedding.
            cond = torch.cat(
                [
                    cond,
                    self.action_planes(buttons, look, drop_action, *x_in.shape[-2:]).to(cond.dtype),
                ],
                1,
            )
        return self.unet(
            x_in,
            cond,
            game,
            buttons,
            look,
            {"noise": c_noise, "ctx_noise": sigma_ctx},
            drop_action,
        )

    def denoise(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        context: torch.Tensor,
        buttons: torch.Tensor,
        look: torch.Tensor,
        game: torch.Tensor,
        sigma_ctx: torch.Tensor,
        drop_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """D(x; sigma, cond) = c_skip * x + c_out * F(c_in * x; c_noise, cond). Returns [B, 3, H, W]."""
        c_skip, c_out, c_in, c_noise = edm_coefficients(sigma.float(), self.sigma_data)
        f = self.raw(_b(c_in) * x, c_noise, context, buttons, look, game, sigma_ctx, drop_action)
        return _b(c_skip) * x + _b(c_out) * f.float()

    # --------------------------------------------------------------- training

    def sample_context_noise(self, B: int, device) -> torch.Tensor:
        """Context noise level: sigma_ctx is 0 with probability p_clean, else uniform in [0, max]. [B]."""
        if not self.ctx_enabled:
            return torch.zeros(B, device=device)
        s = torch.rand(B, device=device) * self.ctx_max
        return torch.where(torch.rand(B, device=device) < self.ctx_p_clean, torch.zeros_like(s), s)

    def training_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """batch: context [B, k, 3, H, W], target [B, 3, H, W] (float, [-1, 1]), buttons, look, game_id."""
        y = batch["target"]
        context = batch["context"]
        B = y.shape[0]
        sigma = sample_training_sigma(B, self.p_mean, self.p_std, y.device)
        x = y + _b(sigma) * torch.randn_like(y)
        sigma_ctx = self.sample_context_noise(B, y.device)
        context = context + sigma_ctx[:, None, None, None, None] * torch.randn_like(context)
        drop = torch.rand(B, device=y.device) < self.p_uncond if self.p_uncond > 0 else None

        # Loss on the raw network output. Since lambda(sigma) * c_out^2 = 1, this equals the
        # paper's lambda-weighted loss on D, but avoids dividing by tiny numbers.
        c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, self.sigma_data)
        f = self.raw(
            _b(c_in) * x,
            c_noise,
            context,
            batch["buttons"],
            batch["look"],
            batch["game_id"],
            sigma_ctx,
            drop,
        ).float()
        f_target = (y - _b(c_skip) * x) / _b(c_out)
        err = F.mse_loss(f, f_target, reduction="none").mean(dim=1)  # [B, H, W]
        if self.motion_weight > 0:
            # Pixels that change between the last context frame and the target count more. That is
            # where the effect of an action shows up; without this the loss is dominated by the
            # static background and the model can do well while ignoring the controls.
            moving = ((y - batch["context"][:, -1]).abs().mean(dim=1) > 0.04).float()
            w = 1.0 + self.motion_weight * moving
            per_sample = (err * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2))
        else:
            per_sample = err.mean(dim=(1, 2))
        loss = per_sample.mean()
        with torch.no_grad():
            low = sigma < self.sigma_data
            logs = {
                "loss": loss.item(),
                "loss_low_sigma": per_sample[low].mean().item() if low.any() else math.nan,
                "loss_high_sigma": per_sample[~low].mean().item() if (~low).any() else math.nan,
            }
        return loss, logs

    # --------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        buttons: torch.Tensor,
        look: torch.Tensor,
        game: torch.Tensor,
        steps: int = 3,
        method: str = "euler",
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        context_noise: float = 0.1,
        cfg_scale: float | None = None,
        refinement: bool = False,
        refinement_sigma: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the next frame. Returns float [B, 3, H, W] clamped to [-1, 1]."""
        B = context.shape[0]
        dev = context.device
        sigma_ctx = torch.full((B,), float(context_noise), device=dev)
        if context_noise > 0:
            context = context + context_noise * torch.randn(
                context.shape, generator=generator, device=dev, dtype=context.dtype
            )

        def denoise(x, s):
            d = self.denoise(x, s, context, buttons, look, game, sigma_ctx)
            if cfg_scale is not None and cfg_scale != 1.0:
                drop = torch.ones(B, dtype=torch.bool, device=dev)
                d_u = self.denoise(x, s, context, buttons, look, game, sigma_ctx, drop)
                d = samplers.guided(d, d_u, cfg_scale)
            return d

        shape = (B, 3, *context.shape[-2:])
        noise = torch.randn(shape, generator=generator, device=dev)
        x = samplers.sample(denoise, noise, steps, sigma_min, sigma_max, rho, method)
        if refinement:
            x = samplers.refine(denoise, x.clamp(-1, 1), refinement_sigma, generator)
        return x.clamp(-1, 1)
