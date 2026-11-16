"""Does it look real?

1. A real-vs-dream classifier: its accuracy over rollout time is a realism curve, where 50%
   means it cannot tell them apart.
2. Helpers for the small blind test with people: anonymised clip pairs plus an answer key.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lucid.eval.probes import ProbeNet
from lucid.utils.video import save_gif


def train_real_vs_dream(
    real: torch.Tensor, dream: torch.Tensor, device, steps: int = 1000, batch_size: int = 128
) -> ProbeNet:
    """real, dream: float [N, 3, H, W] in [-1, 1] (training halves). Label 1 = real."""
    net = ProbeNet(2).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    net.train()
    for _ in range(steps):
        ri = torch.randint(len(real), (batch_size // 2,), generator=gen)
        di = torch.randint(len(dream), (batch_size // 2,), generator=gen)
        x = torch.cat([real[ri], dream[di]]).to(device)
        y = torch.cat([torch.ones(len(ri)), torch.zeros(len(di))]).long().to(device)
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


@torch.no_grad()
def realism_curve(
    net: ProbeNet, dream_rollouts: torch.Tensor, real_frames: torch.Tensor
) -> list[float]:
    """Classifier accuracy at each rollout step on held-out data, balanced with real frames.

    dream_rollouts: [B, T, 3, H, W]; real_frames: [N, 3, H, W]. Returns T accuracies.
    """
    p_real_on_real = (net(real_frames).argmax(-1) == 1).float().mean().item()
    out = []
    for t in range(dream_rollouts.shape[1]):
        p_dream_caught = (net(dream_rollouts[:, t]).argmax(-1) == 0).float().mean().item()
        out.append(0.5 * (p_real_on_real + p_dream_caught))
    return out


def make_blind_test(
    real_clips: list[list[np.ndarray]],
    dream_clips: list[list[np.ndarray]],
    out_dir: str | Path,
    seed: int = 0,
) -> Path:
    """Write pairs of clips named A/B in random order, plus answers.json (keep it from testers)."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    answers = {}
    for i, (r, d) in enumerate(zip(real_clips, dream_clips, strict=True)):
        real_is_a = bool(rng.random() < 0.5)
        save_gif(r if real_is_a else d, out_dir / f"pair_{i:02d}_A.gif")
        save_gif(d if real_is_a else r, out_dir / f"pair_{i:02d}_B.gif")
        answers[f"pair_{i:02d}"] = "A" if real_is_a else "B"
    (out_dir / "answers.json").write_text(json.dumps(answers, indent=1))
    return out_dir
