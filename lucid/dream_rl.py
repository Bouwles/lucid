"""Learning to play inside the dream (the classic world-models experiment).

A reward model, trained on real footage, scores the dream: how far the player moved and whether
they're still alive. A policy then trains only on dreamed experience with an advantage
actor-critic update, and is tested afterwards in the real game.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lucid.data.dataset import Episode, to_model_space
from lucid.policy import Policy


class RewardModel(nn.Module):
    """Two consecutive frames [B, 2, 3, H, W] -> (progress made between them [B], alive logit [B])."""

    def __init__(self, width: int = 32):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(9, c, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * c, 2),
        )

    def forward(self, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = pair[:, 0], pair[:, 1]
        out = self.net(torch.cat([a, b, b - a], 1))
        return out[:, 0], out[:, 1]


def train_reward(episodes: list[Episode], device, steps: int = 2000, batch: int = 128, log=print):
    """Progress is the change in the 'distance' label between frames, alive is the 'alive' label."""
    rng = np.random.default_rng(0)
    pairs, dx, alive = [], [], []
    eps = [e for e in episodes if len(e) > 2]
    for _ in range(50_000):
        e = eps[int(rng.integers(len(eps)))]
        t = int(rng.integers(1, len(e)))
        pairs.append(np.asarray(e.frames[t - 1 : t + 1]))
        d = e.labels.get("distance")
        dx.append(float(d[t] - d[t - 1]) if d is not None else 0.0)
        alive.append(float(e.labels["alive"][t]))
    pairs, dx, alive = np.stack(pairs), np.asarray(dx, np.float32), np.asarray(alive, np.float32)
    scale = float(np.abs(dx).std() + 1e-6)
    m = RewardModel().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for step in range(steps):
        i = rng.integers(0, len(dx), batch)
        x = to_model_space(torch.from_numpy(pairs[i]).to(device).permute(0, 1, 4, 2, 3))
        p_dx, p_alive = m(x)
        loss = F.mse_loss(
            p_dx, torch.from_numpy(dx[i] / scale).to(device)
        ) + F.binary_cross_entropy_with_logits(p_alive, torch.from_numpy(alive[i]).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            log(f"  reward step {step + 1} loss {loss.item():.4f}")
    return m.eval(), scale


def sample_actions(pol: Policy, frames: torch.Tensor, used: torch.Tensor):
    """Independent Bernoulli buttons (only the game's own). Returns (bitmask-ready buttons [B, 8], log prob [B], value [B])."""
    h = pol.features(frames)
    logits = pol.buttons(h)
    dist = torch.distributions.Bernoulli(logits=logits)
    a = dist.sample() * used
    logp = (dist.log_prob(a) * used).sum(-1)
    return a, logp, pol.value(h).squeeze(-1), (dist.entropy() * used).sum(-1)


def train_in_dream(
    world,
    pol: Policy,
    reward: RewardModel,
    starts: dict,
    device,
    used_buttons: list[int],
    game_id: int,
    minutes: float = 30,
    envs: int = 16,
    horizon: int = 48,
    gamma: float = 0.97,
    lr: float = 3e-4,
    gen_kwargs: dict | None = None,
    log=print,
) -> dict:
    """Advantage actor-critic on dreamed rollouts. starts: {frames uint8 [N,k,H,W,3], buttons [N,k,8], look [N,k,2]}."""
    import time

    gen_kwargs = gen_kwargs or {}
    k = world.k
    used = torch.zeros(8, device=device)
    used[used_buttons] = 1
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    game = torch.full((envs,), game_id, device=device, dtype=torch.long)
    t_end = time.time() + minutes * 60
    history = []
    it = 0
    while time.time() < t_end:
        idx = np.random.default_rng(it).integers(0, len(starts["frames"]), envs)
        ctx = to_model_space(
            torch.from_numpy(starts["frames"][idx]).to(device).permute(0, 1, 4, 2, 3)
        )
        acts = torch.from_numpy(starts["buttons"][idx].astype(np.float32)).to(device)
        look = torch.zeros(envs, k, 2, device=device)
        logps, values, rewards, entropies, masks = [], [], [], [], []
        alive = torch.ones(envs, device=device)
        for _ in range(horizon):
            a, logp, v, ent = sample_actions(pol, ctx, used)
            acts = torch.cat([acts[:, 1:], a[:, None]], 1)
            with torch.no_grad():
                nxt = world.generate(ctx, acts, look, game, **gen_kwargs)
                d_x, alive_logit = reward(torch.stack([ctx[:, -1], nxt], 1))
            r = d_x.clamp(-3, 3) - 2.0 * (torch.sigmoid(alive_logit) < 0.5).float()
            logps.append(logp)
            values.append(v)
            entropies.append(ent)
            rewards.append(r * alive)
            masks.append(alive.clone())
            alive = (
                alive * (torch.sigmoid(alive_logit) >= 0.5).float()
            )  # a dreamed death ends the episode
            ctx = torch.cat([ctx[:, 1:], nxt[:, None]], 1)
        with torch.no_grad():
            ret = pol.value(pol.features(ctx)).squeeze(-1) * alive
        returns = []
        for r, m in zip(reversed(rewards), reversed(masks), strict=True):
            ret = r + gamma * ret * m
            returns.append(ret)
        returns = torch.stack(returns[::-1])
        values, logps, entropies, masks = (
            torch.stack(x) for x in (values, logps, entropies, masks)
        )
        adv = (returns - values).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = (
            -(logps * adv * masks).mean()
            + 0.5 * ((returns - values) ** 2 * masks).mean()
            - 0.01 * (entropies * masks).mean()
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        it += 1
        mean_return = float(torch.stack(rewards).sum(0).mean())
        history.append(mean_return)
        if it % 10 == 0:
            log(
                f"  dream iteration {it}: mean dreamed return {np.mean(history[-10:]):.2f}, alive at end {alive.mean():.2f}"
            )
    return {"iterations": it, "returns": history}
