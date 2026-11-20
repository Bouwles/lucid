"""Export the world model's denoiser D(x; sigma, cond) to ONNX, for the desktop app's runtime
(ONNX Runtime) and for client-side inference (onnxruntime-web on WebGPU).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from lucid.models.edm import DiffusionWorldModel

INPUTS = ["x", "sigma", "context", "buttons", "look", "game_weights", "sigma_ctx", "drop_action"]


class DenoiserForExport(nn.Module):
    """Fixed-signature wrapper:
    (x, sigma, context, buttons, look, game_weights, sigma_ctx, drop_action) -> D.

    x [1, 3, H, W], sigma [1], context [1, k, 3, H, W], buttons [1, k, 8] (or [1, 8]),
    look [1, k, 2] (or [1, 2]), sigma_ctx [1].
    game_weights [1, num_games]: one-hot for a game, or a mix to blend games' embeddings.
    drop_action [1]: 1 swaps the action for the learned null action, which classifier-free
    guidance needs (run once with 0, once with 1, and push away from the second).
    """

    def __init__(self, model: DiffusionWorldModel):
        super().__init__()
        self.model = model

    def forward(self, x, sigma, context, buttons, look, game_weights, sigma_ctx, drop_action):
        return self.model.denoise(
            x, sigma, context, buttons, look, game_weights, sigma_ctx, drop_action > 0.5
        )


def export_denoiser(model: DiffusionWorldModel, cfg, path: str | Path, opset: int = 18) -> Path:
    if model.kind != "edm_unet":
        raise ValueError("only the diffusion world model is exported")
    model = model.float().cpu().eval()
    W, H = cfg.data.resolution
    k = cfg.data.context_frames
    a = k if cfg.data.action_history else None
    G = cfg.model.num_games
    args = (
        torch.zeros(1, 3, H, W),
        torch.ones(1),
        torch.zeros(1, k, 3, H, W),
        torch.zeros(1, a, 8) if a else torch.zeros(1, 8),
        torch.zeros(1, a, 2) if a else torch.zeros(1, 2),
        torch.nn.functional.one_hot(torch.zeros(1, dtype=torch.long), G).float(),
        torch.zeros(1),
        torch.zeros(1),
    )
    path = Path(path)
    torch.onnx.export(
        DenoiserForExport(model),
        args,
        str(path),
        input_names=INPUTS,
        output_names=["denoised"],
        opset_version=opset,
    )
    return path


class UpsamplerForExport(nn.Module):
    """(x [1, 3, H, W], sigma [1], low_up [1, 3, H, W], prev [1, 3, H, W], game_weights [1, G],
    s_low [1], s_prev [1]) -> D. low_up is the low-res frame already upscaled to H x W."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, sigma, low_up, prev, game_weights, s_low, s_prev):
        return self.model.denoise(x, sigma, low_up, prev, game_weights, s_low, s_prev)


def export_upsampler(model, cfg, path: str | Path, opset: int = 18) -> Path:
    if model.kind != "upsampler":
        raise ValueError("not an upsampler")
    model = model.float().cpu().eval()
    W, H = cfg.data.resolution
    G = cfg.model.num_games
    args = (
        torch.zeros(1, 3, H, W),
        torch.ones(1),
        torch.zeros(1, 3, H, W),
        torch.zeros(1, 3, H, W),
        torch.nn.functional.one_hot(torch.zeros(1, dtype=torch.long), G).float(),
        torch.zeros(1),
        torch.zeros(1),
    )
    path = Path(path)
    torch.onnx.export(
        UpsamplerForExport(model),
        args,
        str(path),
        input_names=["x", "sigma", "low_up", "prev", "game_weights", "s_low", "s_prev"],
        output_names=["denoised"],
        opset_version=opset,
    )
    return path
