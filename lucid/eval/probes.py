"""State probes (SPEC 17.3): small CNNs trained on real frames only, to read game state from
frames. Running them on dreamed frames turns "does the dream make sense?" into numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lucid.data.dataset import Episode, to_model_space
from lucid.data.format import VERTICAL_STATES


@dataclass
class ProbeSpec:
    name: str  # label key: lane | vertical | alive
    num_classes: int


def probe_specs(episodes: list[Episode]) -> list[ProbeSpec]:
    """Probes that make sense for this data. Lane needs lane labels; others always exist."""
    specs = []
    lanes = np.concatenate([e.labels["lane"] for e in episodes])
    if (lanes >= 0).any():
        specs.append(ProbeSpec("lane", int(lanes.max()) + 1))
    specs.append(ProbeSpec("vertical", len(VERTICAL_STATES)))
    specs.append(ProbeSpec("alive", 2))
    return specs


class ProbeNet(nn.Module):
    """uint8-derived float frames [B, 3, H, W] in [-1, 1] -> logits [B, num_classes]."""

    def __init__(self, num_classes: int, width: int = 32):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1),
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
            nn.Linear(4 * c, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _arrays(episodes: list[Episode], key: str) -> tuple[np.ndarray, np.ndarray]:
    frames, labels = [], []
    for e in episodes:
        y = e.labels[key].astype(np.int64)
        keep = y >= 0
        frames.append(np.asarray(e.frames)[keep])
        labels.append(y[keep])
    return np.concatenate(frames), np.concatenate(labels)


def train_probe(
    spec: ProbeSpec,
    train_eps: list[Episode],
    val_eps: list[Episode],
    device: torch.device,
    steps: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    log=print,
) -> tuple[ProbeNet, float]:
    """Train one probe on real frames. Returns (probe, held-out accuracy)."""
    x, y = _arrays(train_eps, spec.name)
    xv, yv = _arrays(val_eps, spec.name)
    net = ProbeNet(spec.num_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    # Class-balanced sampling, so a rare state (e.g. dead) is not ignored.
    counts = np.bincount(y, minlength=spec.num_classes).astype(np.float64)
    w = 1.0 / np.maximum(counts[y], 1)
    w /= w.sum()
    rng = np.random.default_rng(0)
    net.train()
    for step in range(steps):
        idx = rng.choice(len(y), size=batch_size, p=w)
        xb = to_model_space(torch.from_numpy(x[idx]).to(device).permute(0, 3, 1, 2))
        # light augmentation: dreamed frames are a little noisy, real ones are not
        xb = xb + 0.05 * torch.rand(len(idx), 1, 1, 1, device=device) * torch.randn_like(xb)
        loss = F.cross_entropy(net(xb), torch.from_numpy(y[idx]).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            log(f"  probe {spec.name} step {step + 1} loss {loss.item():.4f}")
    acc = accuracy(net, xv, yv, device)
    return net.eval(), acc


@torch.no_grad()
def predict(net: ProbeNet, frames: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """frames: float [N, 3, H, W] in [-1, 1]. Returns class indices [N]."""
    out = []
    for i in range(0, len(frames), batch_size):
        out.append(net(frames[i : i + batch_size]).argmax(-1))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


@torch.no_grad()
def accuracy(net: ProbeNet, x: np.ndarray, y: np.ndarray, device) -> float:
    if len(y) == 0:
        return float("nan")
    frames = to_model_space(torch.from_numpy(x).to(device).permute(0, 3, 1, 2))
    return (predict(net, frames).cpu().numpy() == y).mean().item()


def save_probes(
    path: str | Path, probes: dict[str, ProbeNet], specs: list[ProbeSpec], accs: dict[str, float]
) -> None:
    torch.save(
        {
            "specs": [s.__dict__ for s in specs],
            "state": {k: v.state_dict() for k, v in probes.items()},
            "accuracy": accs,
        },
        path,
    )


def load_probes(path: str | Path, device) -> tuple[dict[str, ProbeNet], dict[str, float]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    probes = {}
    for s in ck["specs"]:
        net = ProbeNet(s["num_classes"])
        net.load_state_dict(ck["state"][s["name"]])
        probes[s["name"]] = net.to(device).eval()
    return probes, ck["accuracy"]
