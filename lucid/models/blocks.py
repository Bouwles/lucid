"""Building blocks for the conditioned UNet (SPEC 13.1)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def num_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class FourierFeatures(nn.Module):
    """Random Fourier features of a scalar or small vector: [B, d_in] -> [B, dim]."""

    def __init__(self, dim: int, in_features: int = 1, scale: float = 16.0):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("freqs", torch.randn(in_features, dim // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x[:, None]
        f = 2 * math.pi * x.float() @ self.freqs
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ScalarEmbedding(nn.Module):
    """Embed a conditioning scalar (noise level, context noise level): [B] -> [B, dim]."""

    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.fourier = FourierFeatures(dim, 1, scale)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier(x))


class ActionEmbedding(nn.Module):
    """Embed unified actions (SPEC 6.1).

    buttons: [B, 8] or [B, K, 8] multi-hot floats; look: [B, 2] or [B, K, 2] in [-1, 1].
    drop: optional bool [B]; True replaces the whole action with a learned null embedding, which
    is what classifier-free guidance needs. Returns [B, dim].
    """

    def __init__(self, dim: int, num_buttons: int = 8, history: int = 1):
        super().__init__()
        self.buttons = nn.Sequential(nn.Linear(num_buttons, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.look_fourier = FourierFeatures(dim, 2, scale=1.0)
        self.look = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.history = history
        self.pos = nn.Parameter(torch.zeros(history, dim)) if history > 1 else None
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(
        self, buttons: torch.Tensor, look: torch.Tensor, drop: torch.Tensor | None = None
    ) -> torch.Tensor:
        if buttons.dim() == 2:
            buttons, look = buttons[:, None], look[:, None]
        B, K, _ = buttons.shape
        if K != self.history:
            raise ValueError(f"model expects an action history of {self.history}, got {K}")
        e = self.buttons(buttons.float()) + self.look(
            self.look_fourier(look.reshape(B * K, 2)).reshape(B, K, -1)
        )
        if self.pos is not None:
            e = e + self.pos
        e = self.out(e.sum(1))
        if drop is not None:
            e = torch.where(drop[:, None], self.null.expand_as(e), e)
        return e


class AdaGroupNorm(nn.Module):
    """GroupNorm whose scale and shift are predicted from the conditioning vector."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups(channels), channels, affine=False)
        # Default init, not zeros: with zero init the conditioning would have no effect at all
        # at step 0, and the "action changes the output" check could not run on a fresh model.
        self.proj = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(c)[:, :, None, None].chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift


class ResBlock(nn.Module):
    """x: [B, C_in, H, W], c: [B, cond_dim] -> [B, C_out, H, W]."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = AdaGroupNorm(c_in, cond_dim)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = AdaGroupNorm(c_out, cond_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, c)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h, c))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions: [B, C, H, W] -> same."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads if channels % heads == 0 else 1
        self.norm = nn.GroupNorm(num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        return x + self.proj(out.transpose(-1, -2).reshape(B, C, H, W))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))
