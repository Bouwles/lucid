"""Inverse dynamics model: which buttons were held between two frames?

Trained on real recordings only. Used two ways:
1. Controllability for any game: roll the dream out under a held action, then ask the IDM which
   action the dreamed frames show. Agreement with the commanded action means the dream obeyed.
2. Later, labelling uploaded videos that have no key presses (the VPT approach).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lucid.data.dataset import Episode, to_model_space
from lucid.data.format import BUTTONS


class IDM(nn.Module):
    """frames before and after: float [B, 3, H, W] each, in [-1, 1]. Returns button logits [B, 8]."""

    def __init__(self, width: int = 32):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(9, c, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 4 * c, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * c, 4 * c, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * c, len(BUTTONS)),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        # The difference image makes motion easy to see; the raw frames give context.
        return self.net(torch.cat([before, after, after - before], dim=1))


def _pairs(episodes: list[Episode], gap: int, rng: np.random.Generator, n: int):
    """Real (frame t, frame t+gap, buttons held for all of t..t+gap-1) samples."""
    out_a, out_b, out_y = [], [], []
    eps = [e for e in episodes if len(e) > gap + 1]
    while len(out_y) < n:
        e = eps[int(rng.integers(len(eps)))]
        t = int(rng.integers(0, len(e) - gap))
        if "alive" in e.labels and not bool(e.labels["alive"][t : t + gap + 1].all()):
            continue
        held = e.buttons[t : t + gap].min(0)  # only count buttons held the whole time
        out_a.append(np.asarray(e.frames[t]))
        out_b.append(np.asarray(e.frames[t + gap]))
        out_y.append(held)
    return np.stack(out_a), np.stack(out_b), np.stack(out_y).astype(np.float32)


def _to(x: np.ndarray, device) -> torch.Tensor:
    return to_model_space(torch.from_numpy(x).to(device).permute(0, 3, 1, 2))


def train_idm(
    train_eps, val_eps, device, gap: int = 4, steps: int = 3000, batch: int = 128, log=print
):
    """Returns (model, per-button balanced accuracy on held-out real frames)."""
    rng = np.random.default_rng(0)
    a, b, y = _pairs(train_eps, gap, rng, 60_000)
    used = y.mean(0) > 0.01  # buttons this game actually uses
    # Positive weight per button so rarely held buttons are not ignored.
    pos = torch.tensor(np.clip((1 - y.mean(0)) / np.maximum(y.mean(0), 1e-3), 1, 20), device=device)
    model = IDM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(steps):
        i = rng.integers(0, len(y), batch)
        logits = model(_to(a[i], device), _to(b[i], device))
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(y[i]).to(device), pos_weight=pos
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            log(f"  idm step {step + 1} loss {loss.item():.4f}")
    va, vb, vy = _pairs(val_eps, gap, np.random.default_rng(1), 6000)
    acc = balanced_accuracy(predict(model, _to(va, device), _to(vb, device)), vy, used)
    return model.eval(), acc, used


@torch.no_grad()
def predict(model: IDM, before: torch.Tensor, after: torch.Tensor, batch: int = 512) -> np.ndarray:
    out = []
    for i in range(0, len(before), batch):
        out.append(torch.sigmoid(model(before[i : i + batch], after[i : i + batch])).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(BUTTONS)))


def balanced_accuracy(prob: np.ndarray, target: np.ndarray, used: np.ndarray) -> dict[str, float]:
    """Mean of true-positive and true-negative rates, per used button (0.5 = chance)."""
    pred = prob > 0.5
    out = {}
    for j, name in enumerate(BUTTONS):
        if not used[j]:
            continue
        p, t = pred[:, j], target[:, j] > 0.5
        if t.sum() == 0 or (~t).sum() == 0:
            continue
        out[name] = float(0.5 * ((p & t).sum() / t.sum() + (~p & ~t).sum() / (~t).sum()))
    return out


def save_idm(path: str | Path, model: IDM, gap: int, used: np.ndarray, acc: dict) -> None:
    torch.save(
        {"state": model.state_dict(), "gap": gap, "used": used.tolist(), "accuracy": acc}, path
    )


def load_idm(path: str | Path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = IDM()
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck["gap"], np.array(ck["used"], bool), ck["accuracy"]
