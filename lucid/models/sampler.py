"""EDM samplers (Karras et al. 2022, Algorithm 1 without stochastic churn)."""

from __future__ import annotations

from collections.abc import Callable

import torch

Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x [B,C,H,W], sigma [B]) -> D(x)


def sigma_schedule(
    steps: int, sigma_min: float, sigma_max: float, rho: float, device=None
) -> torch.Tensor:
    """EDM noise levels, from sigma_max down to sigma_min, with a final 0. Shape [steps + 1].

    sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if steps == 1:
        sig = torch.tensor([sigma_max], dtype=torch.float64)
    else:
        i = torch.arange(steps, dtype=torch.float64)
        a, b = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
        sig = (a + i / (steps - 1) * (b - a)) ** rho
    return torch.cat([sig, torch.zeros(1, dtype=torch.float64)]).float().to(device)


@torch.no_grad()
def sample(
    denoise: Denoiser,
    noise: torch.Tensor,
    steps: int = 3,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    method: str = "euler",
) -> torch.Tensor:
    """Deterministic EDM sampling from standard-normal `noise` [B, C, H, W] to a clean sample.

    Euler: x <- x + (sigma_next - sigma) * (x - D(x, sigma)) / sigma.
    Heun adds a second evaluation at sigma_next and averages the two slopes (except on the
    last step to sigma = 0).
    """
    if method not in ("euler", "heun"):
        raise ValueError(f"unknown sampler {method!r}")
    sigmas = sigma_schedule(steps, sigma_min, sigma_max, rho, noise.device)
    B = noise.shape[0]
    x = noise * sigmas[0]
    for i in range(steps):
        s, s_next = sigmas[i], sigmas[i + 1]
        d = (x - denoise(x, s.expand(B))) / s
        x_next = x + (s_next - s) * d
        if method == "heun" and s_next > 0:
            d2 = (x_next - denoise(x_next, s_next.expand(B))) / s_next
            x_next = x + (s_next - s) * 0.5 * (d + d2)
        x = x_next
    return x


@torch.no_grad()
def refine(
    denoise: Denoiser, x: torch.Tensor, sigma: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Refinement sampling (SPEC 3.9, the SDEdit idea): re-noise a sample a little, denoise once."""
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    s = torch.full((x.shape[0],), sigma, device=x.device, dtype=x.dtype)
    return denoise(x + sigma * noise, s)


def guided(d_cond: torch.Tensor, d_uncond: torch.Tensor, scale: float) -> torch.Tensor:
    """Classifier-free guidance: D_uncond + w * (D_cond - D_uncond)."""
    return d_uncond + scale * (d_cond - d_uncond)
