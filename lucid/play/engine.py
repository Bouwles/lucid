"""DreamEngine: the world model wrapped as a game engine with a step(action) interface (SPEC 18)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lucid.data.dataset import load_episodes, to_model_space, to_uint8
from lucid.data.format import pack_buttons, unpack_buttons
from lucid.models.factory import build_model, load_model
from lucid.utils.config import RunConfig, load_config
from lucid.utils.device import autocast, autocast_dtype, pick_device


@dataclass
class UnifiedAction:
    buttons: int = 0  # bitmask, docs/formats.md
    look: tuple[float, float] = (0.0, 0.0)


def _frame_out(x: torch.Tensor) -> np.ndarray:
    """float [1, 3, H, W] in [-1, 1] -> uint8 [H, W, 3]."""
    return to_uint8(x[0].float()).permute(1, 2, 0).cpu().numpy()


class DreamEngine:
    """Keeps the context window and action history, and generates one frame per step.

    reset(game_id) -> uint8 [H, W, 3]; step(action) -> uint8 [H, W, 3].
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "auto",
        sampler_steps: int | None = None,
        context_noise: float | None = None,
        cfg_scale: float | None = None,
        *,
        model: torch.nn.Module | None = None,
        cfg: RunConfig | None = None,
        extra: dict | None = None,
        start_split: str = "test",
        upsampler_path: str | Path | None = None,
        precision: str = "bf16",
        seed: int = 0,
    ):
        self.device = pick_device(str(device)) if not isinstance(device, torch.device) else device
        if model is None:
            if checkpoint_path is None:
                raise ValueError("pass checkpoint_path, or model and cfg")
            model, cfg, extra = load_model(checkpoint_path, self.device)
        assert cfg is not None
        self.model = model.to(self.device).eval()
        self.cfg = cfg
        self.extra = extra or {}
        self.k = cfg.data.context_frames
        self.history = cfg.data.action_history
        W, H = cfg.data.resolution
        self.resolution = (W, H)
        self.games: dict[str, int] = self.extra.get("games") or {
            g: i for i, g in enumerate(cfg.data.games)
        }
        s = cfg.sampler
        self.gen_kwargs = dict(
            steps=sampler_steps or s.steps,
            method=s.type,
            sigma_min=s.sigma_min,
            sigma_max=s.sigma_max,
            rho=s.rho,
            context_noise=s.inference_context_noise if context_noise is None else context_noise,
            cfg_scale=cfg_scale if cfg_scale is not None else s.cfg_scale,
            refinement=s.refinement,
            refinement_sigma=s.refinement_sigma,
        )
        self.amp = autocast_dtype(self.device, precision)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.start_split = start_split
        self._starts: dict[str, list] | None = None
        self.upsampler = None
        if upsampler_path:
            self.upsampler, self.ups_cfg, _ = load_model(upsampler_path, self.device)
        self._times: list[float] = []
        self.game = next(iter(self.games))
        self._game_input: torch.Tensor = torch.tensor([self.games[self.game]], device=self.device)
        self.replay = None  # (episode, index) for side-by-side mode
        self.context = torch.zeros(1, self.k, 3, H, W, device=self.device)
        self.buttons = torch.zeros(1, self.k, 8, device=self.device)
        self.look = torch.zeros(1, self.k, 2, device=self.device)
        self.prev_high = None

    @classmethod
    def random_weights(cls, config_path: str | Path, device: str = "cpu", **kw) -> DreamEngine:
        """An untrained engine for UI and protocol smoke tests. Frames are meaningless."""
        cfg = load_config(config_path)
        model = build_model(cfg)
        games = {g: i for i, g in enumerate(cfg.data.games)}
        return cls(model=model, cfg=cfg, extra={"games": games}, device=device, **kw)

    # --------------------------------------------------------------- start contexts

    def _load_starts(self) -> dict[str, list]:
        if self._starts is None:
            self._starts = {}
            d = self.cfg.data
            for g in self.games:
                try:
                    eps, _ = load_episodes(d.root, d.version, self.start_split, [g])
                except (FileNotFoundError, KeyError):
                    eps = []
                self._starts[g] = [e for e in eps if len(e) > self.k + 1]
        return self._starts

    def _game_tensor(self, game_id: str) -> torch.Tensor:
        if game_id not in self.games:
            raise KeyError(f"unknown game {game_id!r}; known: {sorted(self.games)}")
        return torch.tensor([self.games[game_id]], device=self.device)

    # --------------------------------------------------------------- public interface

    def reset(
        self,
        game_id: str | None = None,
        start_context: np.ndarray | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Start a new dream from a real starting context (random from the test split if None).

        start_context: optional uint8 [k, H, W, 3]. Returns the current frame, uint8 [H, W, 3].
        """
        if seed is not None:
            self.generator.manual_seed(seed)
            self.rng = np.random.default_rng(seed)
        self.game = game_id or self.game
        self._game_input = self._game_tensor(self.game)
        self.buttons.zero_()
        self.look.zero_()
        self.replay = None
        self.prev_high = None
        if start_context is None:
            starts = self._load_starts().get(self.game) or []
            if starts:
                ep = starts[int(self.rng.integers(len(starts)))]
                t = int(self.rng.integers(0, len(ep) - self.k))
                start_context = np.asarray(ep.frames[t : t + self.k])
                self.buttons[0] = torch.from_numpy(ep.buttons[t : t + self.k].astype(np.float32))
                self.look[0] = torch.from_numpy(ep.look[t : t + self.k])
                self.replay = (ep, t + self.k)
            else:  # no dataset available (e.g. random-weights smoke test)
                W, H = self.resolution
                start_context = np.full((self.k, H, W, 3), 127, np.uint8)
        ctx = (
            torch.from_numpy(np.array(start_context, copy=True)).to(self.device).permute(0, 3, 1, 2)
        )
        self.context = to_model_space(ctx)[None]
        return self.current_frame()

    def current_frame(self) -> np.ndarray:
        return _frame_out(self.context[:, -1])

    def set_game(self, game_id: str) -> None:
        """Switch game mid-rollout without resetting the context (the dream morphs, SPEC 15.4)."""
        self.game = game_id
        self._game_input = self._game_tensor(game_id)

    def set_blend(self, game_a: str, game_b: str, alpha: float) -> None:
        """Interpolate two game embeddings: alpha 0 = all game_a, 1 = all game_b."""
        w = torch.zeros(1, self.model.unet.num_games, device=self.device)
        w[0, self.games[game_a]] += 1 - alpha
        w[0, self.games[game_b]] += alpha
        self._game_input = w

    def clear_blend(self) -> None:
        self._game_input = self._game_tensor(self.game)

    @torch.inference_mode()
    def step(self, action: UnifiedAction | int | dict) -> np.ndarray:
        """Generate the next frame given the action. Returns uint8 [H, W, 3]."""
        if isinstance(action, int):
            action = UnifiedAction(action)
        elif isinstance(action, dict):
            action = UnifiedAction(
                int(action.get("buttons", 0)), tuple(action.get("look", (0.0, 0.0)))
            )
        t0 = time.perf_counter()
        b = torch.from_numpy(unpack_buttons(action.buttons).astype(np.float32)).to(self.device)
        lk = torch.tensor(action.look, dtype=torch.float32, device=self.device).clamp(-1, 1)
        self.buttons = torch.cat([self.buttons[:, 1:], b[None, None]], 1)
        self.look = torch.cat([self.look[:, 1:], lk[None, None]], 1)
        bi, li = (
            (self.buttons, self.look) if self.history else (self.buttons[:, -1], self.look[:, -1])
        )
        with autocast(self.device, self.amp):
            nxt = self.model.generate(
                self.context, bi, li, self._game_input, generator=self.generator, **self.gen_kwargs
            )
        self.context = torch.cat([self.context[:, 1:], nxt[:, None].float()], 1)
        out = nxt
        if self.upsampler is not None:
            with autocast(self.device, self.amp):
                out = self.upsampler.generate(
                    nxt,
                    self._game_input,
                    prev_high=self.prev_high,
                    steps=self.ups_cfg.sampler.steps,
                    generator=self.generator,
                )
            self.prev_high = out.float()
        frame = _frame_out(out)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._times.append(time.perf_counter() - t0)
        self._times = self._times[-30:]
        return frame

    def step_replay(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Side-by-side mode: advance the recorded real episode one step and drive the dream with
        the same recorded action. Returns (real frame, dream frame), or None at the episode end."""
        if self.replay is None:
            return None
        ep, t = self.replay
        if t >= len(ep):
            return None
        a = UnifiedAction(
            int(pack_buttons(ep.buttons[t - 1])), tuple(float(v) for v in ep.look[t - 1])
        )
        dream = self.step(a)
        self.replay = (ep, t + 1)
        return np.asarray(ep.frames[t]), dream

    @property
    def fps_estimate(self) -> float:
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))

    @property
    def frame_ms(self) -> float:
        return 1000 * sum(self._times) / len(self._times) if self._times else 0.0

    @property
    def sampler_steps(self) -> int:
        return self.gen_kwargs["steps"]

    @sampler_steps.setter
    def sampler_steps(self, n: int) -> None:
        self.gen_kwargs["steps"] = int(max(1, min(64, n)))
