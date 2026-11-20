"""Dreams on ONNX Runtime: the same EDM sampling loop as lucid.models.edm, in numpy.

The base app never imports PyTorch. A dream pack holds the exported denoiser D(x; sigma, cond)
plus real starting contexts; this module runs the Euler sampler around it.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NUM_BUTTONS = 8


def unpack_buttons(buttons: int) -> np.ndarray:
    """Bitmask int -> float32 [8] (bit i = button i, see docs/formats.md)."""
    return ((int(buttons) >> np.arange(NUM_BUTTONS)) & 1).astype(np.float32)


def cancel_opposites(buttons: int) -> int:
    for a, b in ((0, 1), (2, 3)):  # left/right, up/down
        mask = (1 << a) | (1 << b)
        if buttons & mask == mask:
            buttons &= ~mask
    return buttons


def sigma_schedule(steps: int, sigma_min: float, sigma_max: float, rho: float) -> np.ndarray:
    """EDM noise levels from sigma_max down to sigma_min, then 0. Matches lucid.models.sampler."""
    if steps == 1:
        sig = np.array([sigma_max], np.float64)
    else:
        i = np.arange(steps, dtype=np.float64)
        a, b = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
        sig = (a + i / (steps - 1) * (b - a)) ** rho
    return np.concatenate([sig, [0.0]]).astype(np.float32)


@dataclass
class DreamPack:
    path: Path
    meta: dict
    bundled: bool
    _session: object = field(default=None, repr=False)
    _starts: dict | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def game(self) -> str:
        return self.meta.get("game", self.id)

    @property
    def games(self) -> dict[str, int]:
        """Every game this pack can dream, with its game id (one for most packs)."""
        return dict(self.meta.get("games") or {self.game: int(self.meta.get("game_id", 0))})

    @classmethod
    def load(cls, path: Path, bundled: bool = False) -> DreamPack:
        meta = json.loads((path / "dream.json").read_text())
        if meta.get("format") != 1:
            raise ValueError(f"{path}: unsupported dream pack format {meta.get('format')}")
        for f in ("denoiser.onnx", "starts.npz"):
            if not (path / f).exists():
                raise FileNotFoundError(f"{path}: missing {f}")
        return cls(path=path, meta=meta, bundled=bundled)

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())

    def session(self):
        """The ONNX Runtime session, created on first use and shared by every player."""
        with self._lock:
            if self._session is None:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                self._session = ort.InferenceSession(
                    str(self.path / "denoiser.onnx"), opts, providers=["CPUExecutionProvider"]
                )
            return self._session

    def policy_path(self, game: str) -> Path | None:
        for name in (f"policy_{game}.onnx", "policy.onnx"):
            if (self.path / name).exists():
                return self.path / name
        return None

    def has_policy_for(self, game: str) -> bool:
        return self.policy_path(game) is not None

    @property
    def has_policy(self) -> bool:
        return any(self.has_policy_for(g) for g in self.games)

    def policy(self, game: str):
        """The dream-mode policy (pixels to buttons) for this game, if the pack has one."""
        with self._lock:
            cache = self.__dict__.setdefault("_policies", {})
            path = self.policy_path(game)
            if path is not None and path not in cache:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                cache[path] = ort.InferenceSession(
                    str(path), opts, providers=["CPUExecutionProvider"]
                )
            return cache.get(path)

    def starts(self) -> dict:
        with self._lock:
            if self._starts is None:
                with np.load(self.path / "starts.npz") as d:
                    self._starts = {k: d[k] for k in d.files}
            return self._starts


def find_packs(*roots: tuple[Path, bool]) -> dict[str, DreamPack]:
    """Scan (folder, bundled) roots for dream packs. Later roots win on id clashes."""
    packs: dict[str, DreamPack] = {}
    for root, bundled in roots:
        if not root or not Path(root).exists():
            continue
        for d in sorted(Path(root).iterdir()):
            if (d / "dream.json").exists():
                try:
                    packs[DreamPack.load(d, bundled).id] = DreamPack.load(d, bundled)
                except (ValueError, FileNotFoundError, json.JSONDecodeError):
                    continue
    return packs


class Dream:
    """One player's dream: keeps the context window and action history.

    reset() and step() return the current frame as uint8 [H, W, 3].
    """

    def __init__(
        self,
        pack: DreamPack,
        steps: int | None = None,
        seed: int | None = None,
        game: str | None = None,
    ):
        m = pack.meta
        self.pack = pack
        self.k = int(m["context_frames"])
        self.history = bool(m.get("action_history", True))
        self.W, self.H = (int(v) for v in m["resolution"])
        s = m["sampler"]
        self.steps = int(steps or s["steps"])
        self.sigma_min, self.sigma_max, self.rho = (
            float(s["sigma_min"]),
            float(s["sigma_max"]),
            float(s["rho"]),
        )
        self.context_noise = float(s.get("context_noise", 0.1))
        # Classifier-free guidance strengthens how much the controls matter (packs exported with a
        # drop_action input only). 1.0 means off.
        self.cfg_base = float(s.get("cfg_scale") or 1.0)
        self.cfg_scale = self.cfg_base
        self.cfg_per_game = {k: float(v) for k, v in (s.get("cfg_scale_per_game") or {}).items()}
        self.num_games = int(m.get("num_games", 1))
        self.game = game if game in pack.games else next(iter(pack.games))
        self.game_id = pack.games[self.game]
        self.game_weights = self._one_hot(self.game_id)
        self.cfg_scale = self.cfg_per_game.get(self.game, self.cfg_base)
        self.rng = np.random.default_rng(seed)
        self.session = pack.session()
        self.context = np.zeros((1, self.k, 3, self.H, self.W), np.float32)
        self.buttons = np.zeros((1, self.k, NUM_BUTTONS), np.float32)
        self.look = np.zeros((1, self.k, 2), np.float32)
        self._times: list[float] = []
        self.sharpener: Sharpener | None = None  # set for "Sharp" quality

    # --------------------------------------------------------------- the denoiser and sampler

    def _one_hot(self, gid: int) -> np.ndarray:
        w = np.zeros((1, max(self.num_games, gid + 1)), np.float32)
        w[0, gid] = 1.0
        return w

    @property
    def inputs(self) -> set[str]:
        return {i.name for i in self.session.get_inputs()}

    def set_game(self, game: str) -> None:
        """Switch game without restarting: the dream morphs into the other game."""
        if game not in self.pack.games:
            raise KeyError(f"this dream can't do {game}")
        self.game, self.game_id = game, self.pack.games[game]
        self.game_weights = self._one_hot(self.game_id)
        self.cfg_scale = self.cfg_per_game.get(game, self.cfg_base)

    def set_blend(self, a: str, b: str, alpha: float) -> None:
        """Mix two games' embeddings: alpha 0 is all a, 1 is all b. Needs a multi-game pack."""
        if "game_weights" not in self.inputs:
            raise KeyError("this dream can't blend games")
        w = np.zeros_like(self.game_weights)
        w[0, self.pack.games[a]] += 1 - alpha
        w[0, self.pack.games[b]] += alpha
        self.game_weights = w

    @property
    def supports_guidance(self) -> bool:
        return "drop_action" in {i.name for i in self.session.get_inputs()}

    def denoise(
        self, x: np.ndarray, sigma: float, context: np.ndarray, sigma_ctx: float, drop: bool = False
    ) -> np.ndarray:
        b = self.buttons if self.history else self.buttons[:, -1]
        lk = self.look if self.history else self.look[:, -1]
        feeds = {
            "x": x.astype(np.float32),
            "sigma": np.array([sigma], np.float32),
            "context": context.astype(np.float32),
            "buttons": b,
            "look": lk,
            "sigma_ctx": np.array([sigma_ctx], np.float32),
        }
        if "game_weights" in self.inputs:
            feeds["game_weights"] = self.game_weights
        else:
            feeds["game_id"] = np.array([self.game_id], np.int64)
        if self.supports_guidance:
            feeds["drop_action"] = np.array([1.0 if drop else 0.0], np.float32)
        return self.session.run(None, feeds)[0]

    def guided_denoise(
        self, x: np.ndarray, sigma: float, context: np.ndarray, sigma_ctx: float
    ) -> np.ndarray:
        d = self.denoise(x, sigma, context, sigma_ctx)
        if self.cfg_scale != 1.0 and self.supports_guidance:
            d_null = self.denoise(x, sigma, context, sigma_ctx, drop=True)
            d = d_null + self.cfg_scale * (d - d_null)
        return d

    def generate(
        self, noise: np.ndarray | None = None, ctx_noise: np.ndarray | None = None
    ) -> np.ndarray:
        """Sample the next frame (float [1, 3, H, W] in [-1, 1]) from the current context and actions.

        `noise` and `ctx_noise` (standard normal, shapes [1, 3, H, W] and [1, k, 3, H, W]) can be passed
        in so tests can compare against the PyTorch sampler exactly.
        """
        cn = self.context_noise
        context = self.context
        if cn > 0:
            if ctx_noise is None:
                ctx_noise = self.rng.standard_normal(context.shape, dtype=np.float32)
            context = context + cn * ctx_noise
        sigmas = sigma_schedule(self.steps, self.sigma_min, self.sigma_max, self.rho)
        if noise is None:
            noise = self.rng.standard_normal((1, 3, self.H, self.W), dtype=np.float32)
        x = noise * sigmas[0]
        for i in range(self.steps):
            s, s_next = float(sigmas[i]), float(sigmas[i + 1])
            d = (x - self.guided_denoise(x, s, context, cn)) / s
            x = x + (s_next - s) * d
        return np.clip(x, -1.0, 1.0)

    # --------------------------------------------------------------- game interface

    @staticmethod
    def _to_uint8(x: np.ndarray) -> np.ndarray:
        return np.clip(np.rint((x + 1.0) * 127.5), 0, 255).astype(np.uint8).transpose(1, 2, 0)

    def current_frame(self) -> np.ndarray:
        return self._to_uint8(self.context[0, -1])

    def reset(self, start: int | None = None, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        st = self.pack.starts()
        idx = np.arange(len(st["frames"]))
        if "game_id" in st:  # multi-game packs: start from a real frame of the current game
            mine = idx[st["game_id"] == self.game_id]
            idx = mine if len(mine) else idx
        i = int(idx[self.rng.integers(len(idx))]) if start is None else int(idx[start % len(idx)])
        frames = st["frames"][i]  # uint8 [k, H, W, 3]
        self.context = (frames.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)[None]
        # In training, the action at slot j is the one taken while seeing context frame j, so the
        # last slot belongs to the player's next key. step() drops the oldest slot and appends
        # that key, so start one slot back (repeating the first) to keep the start's keys aligned.
        b, lk = st["buttons"][i].astype(np.float32), st["look"][i].astype(np.float32)
        self.buttons = np.concatenate([b[:1], b[:-1]])[None]
        self.look = np.concatenate([lk[:1], lk[:-1]])[None]
        return self.current_frame()

    def policy_action(self, threshold: float = 0.5) -> tuple[int, tuple[float, float]]:
        """What the dream-mode policy would press, looking at the dream's own last frames."""
        sess = self.pack.policy(self.game)
        if sess is None:
            return 0, (0.0, 0.0)
        k = int(sess.get_inputs()[0].shape[1])
        probs, look = sess.run(None, {"frames": self.context[:, -k:].astype(np.float32)})
        b = 0
        for i, pr in enumerate(probs[0]):
            if pr > threshold:
                b |= 1 << i
        return b, (float(look[0, 0]), float(look[0, 1]))

    def step(self, buttons: int = 0, look: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        t0 = time.perf_counter()
        b = unpack_buttons(cancel_opposites(int(buttons) & 0xFF))
        lk = np.clip(np.asarray(look, np.float32), -1, 1)
        self.buttons = np.concatenate([self.buttons[:, 1:], b[None, None]], 1)
        self.look = np.concatenate([self.look[:, 1:], lk[None, None]], 1)
        nxt = self.generate()
        self.context = np.concatenate([self.context[:, 1:], nxt[:, None]], 1)
        out = self.sharpener(nxt, self.game_weights) if self.sharpener is not None else nxt
        self._times = (self._times + [time.perf_counter() - t0])[-30:]
        return self._to_uint8(out[0])

    @property
    def frame_ms(self) -> float:
        return 1000 * sum(self._times) / len(self._times) if self._times else 0.0

    @property
    def fps(self) -> float:
        return 1000 / self.frame_ms if self._times else 0.0


def benchmark(pack: DreamPack, frames: int = 20) -> float:
    """Measured dream frames per second on this machine (generation only)."""
    d = Dream(pack, seed=0)
    d.reset(0)
    for _ in range(3):
        d.step(0)
    t0 = time.perf_counter()
    for _ in range(frames):
        d.step(0)
    return frames / (time.perf_counter() - t0)


def upscale2x(x: np.ndarray) -> np.ndarray:
    """Bilinear 2x upscale of [1, C, H, W], identical to PyTorch's align_corners=False."""

    def axis(a: np.ndarray, ax: int) -> np.ndarray:
        prev = np.concatenate([np.take(a, [0], ax), np.take(a, range(a.shape[ax] - 1), ax)], ax)
        nxt = np.concatenate([np.take(a, range(1, a.shape[ax]), ax), np.take(a, [-1], ax)], ax)
        even, odd = 0.75 * a + 0.25 * prev, 0.75 * a + 0.25 * nxt
        out = np.stack([even, odd], ax + 1)
        shape = list(a.shape)
        shape[ax] *= 2
        return out.reshape(shape)

    return axis(axis(x, 2), 3)


class Sharpener:
    """The "Sharp" quality mode: a diffusion upsampler turns each 64 px dream frame into 128 px,
    conditioned on the previous sharp frame so detail doesn't flicker."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        self.meta = json.loads((Path(path) / "sharp.json").read_text())
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(Path(path) / "upsampler.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        s = self.meta["sampler"]
        self.steps, self.prev_noise = int(s["steps"]), float(s.get("prev_noise", 0.05))
        self.sigmas = sigma_schedule(
            self.steps, float(s["sigma_min"]), float(s["sigma_max"]), float(s["rho"])
        )
        self.num_games = int(self.meta["num_games"])
        self.prev: np.ndarray | None = None
        self.rng = np.random.default_rng(0)

    def reset(self) -> None:
        self.prev = None

    def __call__(self, low: np.ndarray, game_weights: np.ndarray) -> np.ndarray:
        """low: float [1, 3, h, w] in [-1, 1] -> float [1, 3, 2h, 2w]."""
        low_up = upscale2x(low).astype(np.float32)
        prev = low_up if self.prev is None else self.prev
        prev = prev + self.prev_noise * self.rng.standard_normal(prev.shape, dtype=np.float32)
        gw = np.zeros((1, self.num_games), np.float32)
        gw[0, : game_weights.shape[1]] = game_weights[0, : self.num_games]
        x = self.rng.standard_normal(low_up.shape, dtype=np.float32) * self.sigmas[0]
        for i in range(self.steps):
            s, s_next = float(self.sigmas[i]), float(self.sigmas[i + 1])
            d = self.session.run(
                None,
                {
                    "x": x,
                    "sigma": np.array([s], np.float32),
                    "low_up": low_up,
                    "prev": prev.astype(np.float32),
                    "game_weights": gw,
                    "s_low": np.zeros(1, np.float32),
                    "s_prev": np.array([self.prev_noise], np.float32),
                },
            )[0]
            x = x + (s_next - s) * (x - d) / s
        out = np.clip(x, -1.0, 1.0)
        self.prev = out
        return out
