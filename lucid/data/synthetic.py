"""A tiny procedural lane game that writes episodes in the raw recording format.

This exists only to test the pipeline (recorder format, validation, dataset building, training,
evaluation and serving) end to end without Paul's Unity games. It is not a stand-in for real
data and is never used to report results.

Top-down view: three lanes, obstacles scroll down towards the player at the bottom. Two obstacle
kinds: low blocks (jump over them) and high bars (slide under them). Hitting one shows a red
game-over screen for a fixed time, then the episode ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from lucid.data.format import (
    EpisodeMeta,
    SessionMeta,
    Step,
    button_bit,
    write_episode,
    write_session,
)

GAME_ID = "synthetic"
LANES = 3
LEFT, RIGHT, DOWN, JUMP = (button_bit(n) for n in ("left", "right", "down", "jump"))
COLORS = {
    "bg": (34, 40, 49),
    "lane": (57, 62, 70),
    "player": (0, 173, 181),
    "block": (238, 238, 238),
    "bar": (255, 196, 0),
    "dead": (180, 30, 30),
}


@dataclass
class _Obstacle:
    lane: int
    y: float
    kind: str  # "block" (jump) or "bar" (slide)


class SyntheticRunner:
    def __init__(self, seed: int, size: int = 64, jump_steps: int = 8, death_steps: int = 15):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.jump_steps = jump_steps
        self.death_steps = death_steps
        self.lane = 1
        self.vertical = "grounded"
        self.vtimer = 0
        self.alive = True
        self.dead_timer = 0
        self.speed = 1.0
        self.distance = 0.0
        self.obstacles: list[_Obstacle] = []
        self.spawn_timer = 10
        self.prev_buttons = 0

    @property
    def done(self) -> bool:
        return not self.alive and self.dead_timer >= self.death_steps

    def state(self) -> dict:
        return {
            "lane": self.lane,
            "x": float(self.lane),
            "y": 0.0,
            "vertical": self.vertical,
            "alive": self.alive,
            "speed": round(self.speed, 4),
            "distance": round(self.distance, 4),
        }

    def step(self, buttons: int) -> None:
        if not self.alive:
            self.dead_timer += 1
            return
        pressed = buttons & ~self.prev_buttons  # lane changes happen on key-down, like a runner
        self.prev_buttons = buttons
        if pressed & LEFT and not buttons & RIGHT:
            self.lane = max(0, self.lane - 1)
        if pressed & RIGHT and not buttons & LEFT:
            self.lane = min(LANES - 1, self.lane + 1)
        if self.vertical == "grounded":
            if pressed & JUMP:
                self.vertical, self.vtimer = "jumping", self.jump_steps
            elif pressed & DOWN:
                self.vertical, self.vtimer = "sliding", self.jump_steps
        else:
            self.vtimer -= 1
            if self.vtimer <= 0:
                self.vertical = "grounded"

        self.speed = min(2.5, 1.0 + self.distance / 400)
        self.distance += self.speed
        for o in self.obstacles:
            o.y += 0.03 * self.speed
        self.obstacles = [o for o in self.obstacles if o.y < 1.1]
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            kind = "block" if self.rng.random() < 0.5 else "bar"
            self.obstacles.append(_Obstacle(int(self.rng.integers(LANES)), -0.1, kind))
            self.spawn_timer = int(self.rng.integers(12, 30))

        for o in self.obstacles:
            if o.lane == self.lane and 0.82 < o.y < 0.92:
                safe = (o.kind == "block" and self.vertical == "jumping") or (
                    o.kind == "bar" and self.vertical == "sliding"
                )
                if not safe:
                    self.alive = False

    def render(self) -> np.ndarray:
        s = self.size
        img = np.empty((s, s, 3), np.uint8)
        if not self.alive:
            img[:] = COLORS["dead"]
            img[s // 2 - 2 : s // 2 + 2, s // 4 : 3 * s // 4] = 255
            return img
        img[:] = COLORS["bg"]
        lw = s // LANES
        for i in range(1, LANES):
            img[:, i * lw - 1 : i * lw + 1] = COLORS["lane"]
        scroll = int(self.distance * 2) % 8
        for y in range(-8 + scroll, s, 8):
            img[max(0, y) : max(0, y + 2), 0:2] = COLORS["lane"]
        for o in self.obstacles:
            y = int(o.y * s)
            h = s // 16 if o.kind == "block" else s // 32
            x0, x1 = o.lane * lw + 2, (o.lane + 1) * lw - 2
            img[max(0, y) : max(0, min(s, y + h)), x0:x1] = COLORS[o.kind]
        ps = {"grounded": s // 10, "jumping": s // 7, "sliding": s // 16}[self.vertical]
        cx, cy = self.lane * lw + lw // 2, int(0.87 * s)
        img[cy - ps // 2 : cy + ps // 2 + 1, cx - ps // 2 : cx + ps // 2 + 1] = COLORS["player"]
        return img


class _AutoPilot:
    """Mixes random-with-persistence, a look-ahead avoider, a noisy avoider and idling."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.mode = rng.choice(["random", "avoider", "noisy", "idle"], p=[0.3, 0.3, 0.3, 0.1])
        self.hold, self.held = 0, 0

    def act(self, game: SyntheticRunner) -> int:
        if self.rng.random() < 0.01:
            self.mode = self.rng.choice(["random", "avoider", "noisy", "idle"])
        if self.mode == "idle":
            return 0
        if self.mode == "random":
            if self.hold <= 0:
                self.held = int(self.rng.choice([0, LEFT, RIGHT, JUMP, DOWN, LEFT | JUMP]))
                self.hold = int(self.rng.integers(1, 8))
            self.hold -= 1
            return self.held
        if self.mode == "noisy" and self.rng.random() < 0.15:
            return int(self.rng.choice([0, LEFT, RIGHT, JUMP, DOWN]))
        ahead = [o for o in game.obstacles if o.lane == game.lane and 0.55 < o.y < 0.85]
        if not ahead:
            return 0
        o = ahead[0]
        if self.rng.random() < 0.5:
            free = [lane for lane in range(LANES) if lane != game.lane]
            target = free[int(self.rng.integers(len(free)))]
            return LEFT if target < game.lane else RIGHT
        return JUMP if o.kind == "block" else DOWN


def record_session(
    out_root: str | Path,
    num_episodes: int,
    seed: int = 0,
    size: int = 64,
    max_steps: int = 300,
    human_every: int = 0,
    session_id: str | None = None,
) -> Path:
    """Write a session of synthetic episodes. Every `human_every`-th episode is labelled human."""
    session_id = session_id or f"{GAME_ID}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    session_dir = Path(out_root) / session_id
    timestep = 1 / 15
    write_session(
        session_dir,
        SessionMeta(
            session_id=session_id,
            game_id=GAME_ID,
            recording_resolution=(size, size),
            logic_timestep=timestep,
            recorder_version="synthetic-0.1.0",
            game_build_version="synthetic",
            created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )
    rng = np.random.default_rng(seed)
    for e in range(num_episodes):
        ep_seed = int(rng.integers(2**31))
        game = SyntheticRunner(ep_seed, size=size)
        pilot = _AutoPilot(np.random.default_rng(ep_seed + 1))
        is_human = human_every > 0 and e % human_every == human_every - 1
        frames, steps = [], []
        t = 0
        while True:
            frames.append(game.render())
            buttons = pilot.act(game) if game.alive else 0
            state = game.state()
            game.step(buttons)
            end = game.done or t == max_steps - 1
            steps.append(Step(t, t * timestep, buttons, (0.0, 0.0), state, end))
            t += 1
            if end:
                break
        meta = EpisodeMeta(
            episode_id=f"ep_{e:06d}",
            session_id=session_id,
            game_id=GAME_ID,
            seed=ep_seed,
            num_steps=len(steps),
            action_source={
                "name": "human" if is_human else "autopilot",
                "settings": {} if is_human else {"mode": str(pilot.mode)},
            },
            recording_resolution=(size, size),
            logic_timestep=timestep,
            recorder_version="synthetic-0.1.0",
            game_build_version="synthetic",
        )
        write_episode(session_dir, meta, steps, frames)
    return session_dir
