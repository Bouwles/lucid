"""Policies that play from pixels: the last k frames in, buttons (and look) out.

Used for dream mode (a policy playing inside the dream), and as the starting point for
learning inside the dream. Trained by behaviour cloning on the autopilot's recorded play.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lucid.data.dataset import Episode, to_model_space
from lucid.data.format import BUTTONS


class Policy(nn.Module):
    """frames: float [B, k, 3, H, W] in [-1, 1] -> (button logits [B, 8], look [B, 2] in [-1, 1])."""

    def __init__(self, k: int = 4, width: int = 32):
        super().__init__()
        c = width
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(3 * k, c, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(4 * c, 4 * c, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.buttons = nn.Linear(4 * c, len(BUTTONS))
        self.look = nn.Linear(4 * c, 2)
        self.value = nn.Linear(4 * c, 1)  # used when learning inside the dream

    def features(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames.flatten(1, 2))

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(frames)
        return self.buttons(h), torch.tanh(self.look(h))


def _samples(episodes: list[Episode], k: int, rng: np.random.Generator, n: int):
    xs, bs, ls = [], [], []
    eps = [e for e in episodes if len(e) > k + 1]
    while len(bs) < n:
        e = eps[int(rng.integers(len(eps)))]
        t = int(rng.integers(k - 1, len(e) - 1))
        if "alive" in e.labels and not bool(e.labels["alive"][t]):
            continue
        xs.append(np.asarray(e.frames[t - k + 1 : t + 1]))
        bs.append(e.buttons[t])  # the action chosen while seeing frame t
        ls.append(e.look[t])
    return np.stack(xs), np.stack(bs).astype(np.float32), np.stack(ls).astype(np.float32)


def _x(frames: np.ndarray, device) -> torch.Tensor:
    return to_model_space(torch.from_numpy(frames).to(device).permute(0, 1, 4, 2, 3))


def train_bc(
    train_eps, val_eps, device, k: int = 4, steps: int = 3000, batch: int = 128, log=print
):
    """Behaviour cloning. Returns (policy, per-button accuracy on held-out frames)."""
    rng = np.random.default_rng(0)
    x, b, lk = _samples(train_eps, k, rng, 40_000)
    used = b.mean(0) > 0.01
    pos = torch.tensor(np.clip((1 - b.mean(0)) / np.maximum(b.mean(0), 1e-3), 1, 10), device=device)
    pol = Policy(k).to(device)
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(steps):
        i = rng.integers(0, len(b), batch)
        logits, look = pol(_x(x[i], device))
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(b[i]).to(device), pos_weight=pos
        )
        loss = loss + F.mse_loss(look, torch.from_numpy(lk[i]).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            log(f"  policy step {step + 1} loss {loss.item():.4f}")
    vx, vb, _ = _samples(val_eps, k, np.random.default_rng(1), 4000)
    pol.eval()
    with torch.no_grad():
        pred = (
            torch.cat(
                [torch.sigmoid(pol(_x(vx[i : i + 256], device))[0]) for i in range(0, len(vx), 256)]
            )
            .cpu()
            .numpy()
            > 0.5
        )
    acc = {
        BUTTONS[j]: float((pred[:, j] == (vb[:, j] > 0.5)).mean())
        for j in range(len(BUTTONS))
        if used[j]
    }
    return pol, acc, used


class PolicyForExport(nn.Module):
    """ONNX wrapper: frames [1, k, 3, H, W] -> (button probabilities [1, 8], look [1, 2])."""

    def __init__(self, pol: Policy):
        super().__init__()
        self.pol = pol

    def forward(self, frames):
        logits, look = self.pol(frames)
        return torch.sigmoid(logits), look


def export_policy(pol: Policy, path: str | Path, k: int, H: int, W: int) -> Path:
    """Writes one self-contained .onnx file (weights embedded, so it can be renamed or moved)."""
    import onnx

    pol = pol.float().cpu().eval()
    torch.onnx.export(
        PolicyForExport(pol),
        (torch.zeros(1, k, 3, H, W),),
        str(path),
        input_names=["frames"],
        output_names=["buttons", "look"],
        opset_version=18,
    )
    model = onnx.load(str(path))  # pulls in any external weight file the exporter wrote
    onnx.save_model(model, str(path), save_as_external_data=False)
    Path(str(path) + ".data").unlink(missing_ok=True)
    return Path(path)
