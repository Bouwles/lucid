"""Export the world model's denoiser D(x; sigma, cond) to ONNX for client-side inference.
The sampling loop then runs in JavaScript with onnxruntime-web on WebGPU.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from lucid.models.edm import DiffusionWorldModel


class DenoiserForExport(nn.Module):
    """Fixed-signature wrapper: (x, sigma, context, buttons, look, game_id, sigma_ctx) -> D.

    x [1, 3, H, W], sigma [1], context [1, k, 3, H, W], buttons [1, k, 8] (or [1, 8]),
    look [1, k, 2] (or [1, 2]), game_id [1] int64, sigma_ctx [1].
    """

    def __init__(self, model: DiffusionWorldModel):
        super().__init__()
        self.model = model

    def forward(self, x, sigma, context, buttons, look, game_id, sigma_ctx):
        return self.model.denoise(x, sigma, context, buttons, look, game_id, sigma_ctx)


def export_denoiser(model: DiffusionWorldModel, cfg, path: str | Path, opset: int = 18) -> Path:
    if model.kind != "edm_unet":
        raise ValueError("only the diffusion world model is exported")
    model = model.float().cpu().eval()
    W, H = cfg.data.resolution
    k = cfg.data.context_frames
    a = k if cfg.data.action_history else None
    args = (
        torch.zeros(1, 3, H, W),
        torch.ones(1),
        torch.zeros(1, k, 3, H, W),
        torch.zeros(1, a, 8) if a else torch.zeros(1, 8),
        torch.zeros(1, a, 2) if a else torch.zeros(1, 2),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1),
    )
    path = Path(path)
    torch.onnx.export(
        DenoiserForExport(model),
        args,
        str(path),
        input_names=["x", "sigma", "context", "buttons", "look", "game_id", "sigma_ctx"],
        output_names=["denoised"],
        opset_version=opset,
    )
    return path
