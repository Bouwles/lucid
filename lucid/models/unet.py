"""The conditioned UNet shared by the baseline, the diffusion world model and the upsampler."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lucid.models.blocks import (
    ActionEmbedding,
    Downsample,
    ResBlock,
    ScalarEmbedding,
    SelfAttention,
    Upsample,
    num_groups,
)


class _Level(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        n_blocks: int,
        cond_dim: int,
        attn: bool,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            ResBlock(c_in if i == 0 else c_out, c_out, cond_dim, dropout) for i in range(n_blocks)
        )
        self.attn = nn.ModuleList(
            SelfAttention(c_out, heads) if attn else nn.Identity() for _ in range(n_blocks)
        )

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for blk, att in zip(self.blocks, self.attn, strict=True):
            h = att(blk(h, c))
        return h


class ConditionalUNet(nn.Module):
    """UNet with adaptive-GroupNorm conditioning.

    Images: `x` [B, 3, H, W] (the frame being denoised or predicted) is concatenated with
    `cond_images` [B, C_cond, H, W] (context frames as channels, or an upscaled low-res frame).
    Vector conditioning, all summed into one vector c [B, cond_dim]:
      - actions: buttons [B, 8] / [B, K, 8] and look [B, 2] / [B, K, 2] (if use_actions)
      - game: integer ids [B], or soft weights [B, num_games] for blending games
      - scalars: one entry per name in `scalar_names`, each [B] (e.g. noise levels)
    H and W must be divisible by 2 ** (len(channel_mults) - 1).
    """

    def __init__(
        self,
        cond_channels: int,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 3, 4),
        res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (8,),
        input_resolution: tuple[int, int] = (64, 64),
        attention_heads: int = 4,
        cond_dim: int = 256,
        num_games: int = 1,
        use_actions: bool = True,
        action_history: int = 1,
        scalar_names: tuple[str, ...] = (),
        dropout: float = 0.0,
        out_channels: int = 3,
    ):
        super().__init__()
        self.scalar_names = tuple(scalar_names)
        self.num_games = num_games
        self.action_history = action_history
        self.down_factor = 2 ** (len(channel_mults) - 1)
        self.game_embed = nn.Embedding(num_games, cond_dim)
        self.action_embed = (
            ActionEmbedding(cond_dim, history=action_history) if use_actions else None
        )
        self.scalar_embed = nn.ModuleDict({n: ScalarEmbedding(cond_dim) for n in self.scalar_names})
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        chans = [base_channels * m for m in channel_mults]
        self.conv_in = nn.Conv2d(3 + cond_channels, chans[0], 3, padding=1)
        size = min(input_resolution)
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        c_prev = chans[0]
        for i, ch in enumerate(chans):
            s = size // 2**i
            self.down.append(
                _Level(
                    c_prev,
                    ch,
                    res_blocks,
                    cond_dim,
                    s in attention_resolutions,
                    attention_heads,
                    dropout,
                )
            )
            self.downsample.append(Downsample(ch) if i < len(chans) - 1 else nn.Identity())
            c_prev = ch
        self.mid1 = ResBlock(chans[-1], chans[-1], cond_dim, dropout)
        self.mid_attn = SelfAttention(chans[-1], attention_heads)
        self.mid2 = ResBlock(chans[-1], chans[-1], cond_dim, dropout)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        c_prev = chans[-1]
        for i in reversed(range(len(chans))):
            s = size // 2**i
            self.up.append(
                _Level(
                    c_prev + chans[i],
                    chans[i],
                    res_blocks,
                    cond_dim,
                    s in attention_resolutions,
                    attention_heads,
                    dropout,
                )
            )
            self.upsample.append(Upsample(chans[i]) if i > 0 else nn.Identity())
            c_prev = chans[i]
        self.norm_out = nn.GroupNorm(num_groups(chans[0]), chans[0])
        self.conv_out = nn.Conv2d(chans[0], out_channels, 3, padding=1)

    def embed(
        self,
        game: torch.Tensor,
        buttons: torch.Tensor | None = None,
        look: torch.Tensor | None = None,
        scalars: dict[str, torch.Tensor] | None = None,
        drop_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if game.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            c = self.game_embed(game)
        else:  # soft weights [B, num_games], used for blending games
            c = game.to(self.game_embed.weight.dtype) @ self.game_embed.weight
        if self.action_embed is not None:
            if buttons is None or look is None:
                raise ValueError("this UNet is action-conditioned: pass buttons and look")
            c = c + self.action_embed(buttons, look, drop_action)
        scalars = scalars or {}
        for name in self.scalar_names:
            c = c + self.scalar_embed[name](scalars[name])
        return self.cond_mlp(c)

    def forward(
        self,
        x: torch.Tensor,
        cond_images: torch.Tensor,
        game: torch.Tensor,
        buttons: torch.Tensor | None = None,
        look: torch.Tensor | None = None,
        scalars: dict[str, torch.Tensor] | None = None,
        drop_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns [B, out_channels, H, W]."""
        H, W = x.shape[-2:]
        if H % self.down_factor or W % self.down_factor:
            raise ValueError(f"H and W must be divisible by {self.down_factor}, got {H}x{W}")
        c = self.embed(game, buttons, look, scalars, drop_action)
        h = self.conv_in(torch.cat([x, cond_images], dim=1))
        skips = []
        for level, down in zip(self.down, self.downsample, strict=True):
            h = level(h, c)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid_attn(self.mid1(h, c)), c)
        for level, up in zip(self.up, self.upsample, strict=True):
            h = level(torch.cat([h, skips.pop()], dim=1), c)
            h = up(h)
        return self.conv_out(F.silu(self.norm_out(h)))
