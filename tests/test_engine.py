import numpy as np

from lucid.models.factory import build_model
from lucid.play.engine import DreamEngine, UnifiedAction
from lucid.utils.config import RunConfig


def tiny_cfg() -> RunConfig:
    cfg = RunConfig()
    cfg.data.games = ["a", "b"]
    cfg.data.resolution = [16, 16]
    cfg.data.root = "/nonexistent"
    cfg.model.num_games = 2
    cfg.model.base_channels, cfg.model.channel_mults = 8, [1, 2]
    cfg.model.attention_resolutions, cfg.model.res_blocks_per_level, cfg.model.cond_dim = [], 1, 16
    cfg.sampler.steps = 2
    return cfg


def make_engine() -> DreamEngine:
    cfg = tiny_cfg()
    return DreamEngine(
        model=build_model(cfg), cfg=cfg, extra={"games": {"a": 0, "b": 1}}, device="cpu"
    )


def test_reset_and_step_shapes():
    e = make_engine()
    f = e.reset("a", seed=0)
    assert f.shape == (16, 16, 3) and f.dtype == np.uint8
    for a in [0, UnifiedAction(1 << 4), {"buttons": 3, "look": [0.2, -0.1]}]:
        f = e.step(a)
        assert f.shape == (16, 16, 3) and f.dtype == np.uint8
    assert e.fps_estimate > 0


def test_start_context_and_switching():
    e = make_engine()
    ctx = np.random.default_rng(0).integers(0, 255, (4, 16, 16, 3), dtype=np.uint8)
    assert np.array_equal(e.reset("a", start_context=ctx), ctx[-1])
    e.set_game("b")
    e.step(0)
    e.set_blend("a", "b", 0.5)
    assert e.step(0).shape == (16, 16, 3)
    e.clear_blend()
    e.sampler_steps = 1
    assert e.step(0).shape == (16, 16, 3)


def test_same_seed_same_dream():
    a, b = make_engine(), make_engine()
    b.model.load_state_dict(a.model.state_dict())
    a.reset("a", seed=3)
    b.reset("a", seed=3)
    assert np.array_equal(a.step(1), b.step(1))
