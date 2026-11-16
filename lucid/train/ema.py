"""Exponential moving average of model weights. Generation uses the EMA weights."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int | None = None) -> None:
        # Warm up the decay early on so the EMA is not dominated by random initial weights.
        d = self.decay if step is None else min(self.decay, (1 + step) / (10 + step))
        for e, p in zip(self.model.parameters(), model.parameters(), strict=True):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers(), strict=True):
            e.copy_(b)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)
