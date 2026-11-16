"""The MSE baseline: predict the next frame directly. Expected to be blurry."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lucid.models.unet import ConditionalUNet


class BaselineModel(nn.Module):
    """context [B, k, 3, H, W] + action -> next frame [B, 3, H, W], all float in [-1, 1]."""

    kind = "baseline"

    def __init__(self, unet: ConditionalUNet, context_frames: int, lpips_weight: float = 0.0):
        super().__init__()
        self.unet = unet
        self.k = context_frames
        self.lpips_weight = lpips_weight
        self._lpips = None

    def forward(self, context, buttons, look, game) -> torch.Tensor:
        x = torch.zeros_like(context[:, -1])
        return self.unet(x, context.flatten(1, 2), game, buttons, look)

    def training_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        pred = self(batch["context"], batch["buttons"], batch["look"], batch["game_id"]).float()
        mse = F.mse_loss(pred, batch["target"])
        loss, logs = mse, {"mse": mse.item()}
        if self.lpips_weight > 0:
            if self._lpips is None:
                from lucid.eval.metrics import LPIPS

                self._lpips = LPIPS(pred.device)
            lp = self._lpips.module(pred, batch["target"]).mean()  # differentiable path
            loss = loss + self.lpips_weight * lp
            logs["lpips"] = lp.item()
        logs["loss"] = loss.item()
        return loss, logs

    @torch.no_grad()
    def generate(self, context, buttons, look, game, **_ignored) -> torch.Tensor:
        """Same signature as DiffusionWorldModel.generate; sampler settings are ignored."""
        return self(context, buttons, look, game).float().clamp(-1, 1)
