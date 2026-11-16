"""Small convolutional autoencoder for the optional latent-diffusion path.

Compresses frames [B, 3, H, W] to latents [B, latent_channels, H/f, W/f] with f = 2 ** len(mults).
Only the autoencoder exists so far; wiring the world model to train on latents is future work
(see docs/decisions.md).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lucid.models.blocks import num_groups


class _Res(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups(c_in), c_in),
            nn.SiLU(),
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(num_groups(c_out), c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
        )
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class Autoencoder(nn.Module):
    def __init__(
        self, base_channels: int = 64, mults: tuple[int, ...] = (1, 2, 4), latent_channels: int = 4
    ):
        super().__init__()
        chans = [base_channels * m for m in mults]
        enc: list[nn.Module] = [nn.Conv2d(3, chans[0], 3, padding=1)]
        c = chans[0]
        for ch in chans:
            enc += [_Res(c, ch), nn.Conv2d(ch, ch, 3, stride=2, padding=1)]
            c = ch
        enc += [
            _Res(c, c),
            nn.GroupNorm(num_groups(c), c),
            nn.SiLU(),
            nn.Conv2d(c, latent_channels, 1),
        ]
        self.encoder = nn.Sequential(*enc)
        dec: list[nn.Module] = [nn.Conv2d(latent_channels, c, 3, padding=1), _Res(c, c)]
        for ch in reversed(chans):
            dec += [
                _Res(c, ch),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, ch, 3, padding=1),
            ]
            c = ch
        dec += [nn.GroupNorm(num_groups(c), c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)
        self.factor = 2 ** len(mults)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def training_loss(self, x: torch.Tensor, lpips=None, lpips_weight: float = 0.1):
        recon, _ = self(x)
        loss = F.l1_loss(recon, x)
        if lpips is not None and lpips_weight > 0:
            loss = loss + lpips_weight * lpips(recon, x).mean()
        return loss, recon
