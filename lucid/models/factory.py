"""Build models from a RunConfig, and load them back from checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from lucid.models.baseline import BaselineModel
from lucid.models.edm import DiffusionWorldModel
from lucid.models.unet import ConditionalUNet
from lucid.models.upsampler import DiffusionUpsampler
from lucid.utils.config import RunConfig, from_dict


def build_model(cfg: RunConfig, sigma_data: float | None = None) -> torch.nn.Module:
    """sigma_data: the measured value from the dataset manifest, if cfg.model.sigma_data <= 0."""
    m, d = cfg.model, cfg.data
    k = d.context_frames
    history = k if d.action_history else 1
    sd = m.sigma_data if m.sigma_data > 0 else (sigma_data or 0.5)
    common = dict(
        base_channels=m.base_channels,
        channel_mults=tuple(m.channel_mults),
        res_blocks=m.res_blocks_per_level,
        attention_resolutions=tuple(m.attention_resolutions),
        attention_heads=m.attention_heads,
        cond_dim=m.cond_dim,
        num_games=m.num_games,
        dropout=m.dropout,
        cond_norm=m.cond_norm,
    )
    if m.type == "edm_unet":
        unet = ConditionalUNet(
            cond_channels=3 * k + (history * 10 if m.action_channels else 0),
            input_resolution=tuple(d.resolution),
            action_history=history,
            scalar_names=("noise", "ctx_noise"),
            **common,
        )
        cn = cfg.context_noise
        return DiffusionWorldModel(
            unet,
            k,
            sd,
            m.p_mean,
            m.p_std,
            m.p_uncond_action,
            cn.enabled,
            cn.p_clean,
            cn.max_sigma,
            m.motion_weight,
            m.action_channels,
        )
    if m.type == "baseline":
        unet = ConditionalUNet(
            cond_channels=3 * k,
            input_resolution=tuple(d.resolution),
            action_history=history,
            **common,
        )
        return BaselineModel(unet, k, m.lpips_weight)
    if m.type == "upsampler":
        unet = ConditionalUNet(
            cond_channels=6 if m.use_prev_frame else 3,
            input_resolution=tuple(d.resolution),
            use_actions=False,
            scalar_names=("noise", "low_noise", "prev_noise"),
            **common,
        )
        return DiffusionUpsampler(
            unet,
            tuple(d.resolution),
            sd,
            m.p_mean,
            m.p_std,
            m.use_prev_frame,
            m.lowres_noise_max,
            m.lowres_blur_max,
        )
    raise ValueError(f"unknown model type {m.type!r}")


def load_model(path: str | Path, device: torch.device | str = "cpu", use_ema: bool = True):
    """Load a checkpoint written by lucid.train.checkpoint. Returns (model, cfg, extra)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(RunConfig, ckpt["config"])
    model = build_model(cfg, ckpt.get("extra", {}).get("sigma_data"))
    state = ckpt["ema"] if use_ema and ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), cfg, ckpt.get("extra", {})
