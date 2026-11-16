"""Rollouts: teacher-forced (always real context) and free-running (own outputs)."""

from __future__ import annotations

import torch

from lucid.utils.config import SamplerConfig


def sampler_kwargs(s: SamplerConfig, **overrides) -> dict:
    kw = dict(
        steps=s.steps,
        method=s.type,
        sigma_min=s.sigma_min,
        sigma_max=s.sigma_max,
        rho=s.rho,
        context_noise=s.inference_context_noise,
        cfg_scale=s.cfg_scale,
        refinement=s.refinement,
        refinement_sigma=s.refinement_sigma,
    )
    kw.update(overrides)
    return kw


def actions_for(buttons: torch.Tensor, look: torch.Tensor, t: int, k: int, history: bool):
    """The action input for generating frame t from a [B, T, ...] action timeline.

    The action at index j leads to frame j + 1, so frame t uses the action at t - 1, or the
    history t - k .. t - 1.
    """
    if history:
        return buttons[:, t - k : t], look[:, t - k : t]
    return buttons[:, t - 1], look[:, t - 1]


@torch.no_grad()
def rollout(
    model,
    frames: torch.Tensor,
    buttons: torch.Tensor,
    look: torch.Tensor,
    game: torch.Tensor,
    length: int,
    history: bool,
    teacher_forced: bool = False,
    generator: torch.Generator | None = None,
    **gen_kwargs,
) -> torch.Tensor:
    """Generate `length` frames after the first k.

    frames: float [B, >=k, 3, H, W] in [-1, 1]; only the first k are used unless teacher_forced
    (then [B, k + length, ...] real frames are needed). buttons [B, k + length - 1 (or more), 8],
    look [B, ..., 2]. game: [B] ids or [B, G] weights. Returns float [B, length, 3, H, W].
    """
    k = model.k
    if buttons.shape[1] < k + length - 1:
        raise ValueError(f"need at least {k + length - 1} actions, got {buttons.shape[1]}")
    if teacher_forced and frames.shape[1] < k + length:
        raise ValueError("teacher-forced rollouts need the real frames for every step")
    ctx = frames[:, :k]
    out = []
    for t in range(k, k + length):
        b, lk = actions_for(buttons, look, t, k, history)
        nxt = model.generate(ctx, b, lk, game, generator=generator, **gen_kwargs)
        out.append(nxt)
        ctx = (
            frames[:, t - k + 1 : t + 1]
            if teacher_forced
            else torch.cat([ctx[:, 1:], nxt[:, None]], 1)
        )
    return torch.stack(out, 1)


def hold_action(
    button_index: int | None, B: int, T: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """A scripted action timeline holding one button (or none) for T steps: ([B, T, 8], [B, T, 2])."""
    buttons = torch.zeros(B, T, 8, device=device)
    if button_index is not None:
        buttons[..., button_index] = 1
    return buttons, torch.zeros(B, T, 2, device=device)
