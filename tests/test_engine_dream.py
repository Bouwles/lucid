import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from lucid_engine.dream import Dream, DreamPack, sigma_schedule, unpack_buttons  # noqa: E402

from lucid.models.sampler import sample  # noqa: E402
from lucid.models.sampler import sigma_schedule as torch_schedule

from engine_fixtures import make_pack  # noqa: E402


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    path, model, cfg = make_pack(tmp_path_factory.mktemp("packs"))
    return DreamPack.load(path, bundled=True), model


def test_schedule_matches_pytorch():
    assert np.allclose(
        sigma_schedule(5, 0.002, 80, 7), torch_schedule(5, 0.002, 80, 7).numpy(), rtol=1e-5
    )


def test_onnx_sampler_matches_pytorch(pack):
    p, model = pack
    d = Dream(p, seed=0)
    d.reset(1)
    d.buttons[0, -1] = unpack_buttons(16)
    noise = np.random.default_rng(1).standard_normal((1, 3, 16, 16)).astype(np.float32)
    ctx_noise = np.random.default_rng(2).standard_normal(d.context.shape).astype(np.float32)
    out = d.generate(noise=noise, ctx_noise=ctx_noise)

    ctx = torch.from_numpy(d.context + 0.1 * ctx_noise)
    b, lk = torch.from_numpy(d.buttons), torch.from_numpy(d.look)
    game, sctx = torch.zeros(1, dtype=torch.long), torch.full((1,), 0.1)
    with torch.no_grad():
        ref = (
            sample(
                lambda x, s: model.denoise(x, s, ctx, b, lk, game, sctx),
                torch.from_numpy(noise),
                steps=2,
                sigma_min=0.002,
                sigma_max=80,
                rho=7,
            )
            .clamp(-1, 1)
            .numpy()
        )
    assert np.abs(out - ref).max() < 1e-3


def test_reset_and_step_shapes(pack):
    d = Dream(pack[0], seed=0)
    f = d.reset()
    assert f.shape == (16, 16, 3) and f.dtype == np.uint8
    for b in (0, 1, 3, 16):
        f = d.step(b)
        assert f.shape == (16, 16, 3)
    assert d.fps > 0


def test_guidance_input_changes_the_sample(pack):
    d = Dream(pack[0], seed=0)
    d.reset(0)
    assert d.supports_guidance
    noise = np.random.default_rng(3).standard_normal((1, 3, 16, 16)).astype(np.float32)
    ctx_noise = np.random.default_rng(4).standard_normal(d.context.shape).astype(np.float32)
    plain = d.generate(noise=noise, ctx_noise=ctx_noise)
    d.cfg_scale = 3.0
    guided = d.generate(noise=noise, ctx_noise=ctx_noise)
    assert np.abs(plain - guided).max() > 1e-4


def test_multi_game_pack_switches_and_blends(tmp_path):
    import json

    import torch as _torch

    from lucid.models.factory import build_model
    from lucid.play.export_onnx import export_denoiser

    from engine_fixtures import tiny_cfg

    cfg = tiny_cfg()
    cfg.model.num_games = 2
    _torch.manual_seed(1)
    model = build_model(cfg).eval()
    out = tmp_path / "both"
    out.mkdir()
    export_denoiser(model, cfg, out / "denoiser.onnx")
    rng = np.random.default_rng(0)
    np.savez(
        out / "starts.npz",
        frames=rng.integers(0, 255, (4, 4, 16, 16, 3), dtype=np.uint8),
        buttons=np.zeros((4, 4, 8), np.uint8),
        look=np.zeros((4, 4, 2), np.float32),
        game_id=np.array([0, 0, 1, 1]),
    )
    s = cfg.sampler
    (out / "dream.json").write_text(
        json.dumps(
            {
                "format": 1,
                "id": "both",
                "game": "both",
                "game_id": 0,
                "games": {"x": 0, "y": 1},
                "num_games": 2,
                "resolution": [16, 16],
                "context_frames": 4,
                "action_history": True,
                "sigma_data": 0.5,
                "sampler": {
                    "steps": 2,
                    "sigma_min": s.sigma_min,
                    "sigma_max": s.sigma_max,
                    "rho": s.rho,
                    "context_noise": 0.1,
                },
            }
        )
    )
    pack = DreamPack.load(out)
    assert pack.games == {"x": 0, "y": 1}
    d = Dream(pack, seed=0, game="y")
    d.reset(0)
    assert d.game == "y" and d.game_weights.tolist() == [[0.0, 1.0]]
    noise = np.random.default_rng(3).standard_normal((1, 3, 16, 16)).astype(np.float32)
    cn = np.random.default_rng(4).standard_normal(d.context.shape).astype(np.float32)
    as_y = d.generate(noise=noise, ctx_noise=cn)
    d.set_game("x")
    as_x = d.generate(noise=noise, ctx_noise=cn)
    d.set_blend("x", "y", 0.5)
    assert np.allclose(d.game_weights, [[0.5, 0.5]])
    blended = d.generate(noise=noise, ctx_noise=cn)
    assert np.abs(as_x - as_y).max() > 1e-4 and np.abs(blended - as_x).max() > 1e-5


def test_multi_game_reset_starts_from_the_right_game(tmp_path):
    import json

    from lucid.models.factory import build_model
    from lucid.play.export_onnx import export_denoiser

    from engine_fixtures import tiny_cfg

    cfg = tiny_cfg()
    cfg.model.num_games = 2
    out = tmp_path / "both"
    out.mkdir()
    export_denoiser(build_model(cfg).eval(), cfg, out / "denoiser.onnx")
    frames = np.zeros((6, 4, 16, 16, 3), np.uint8)
    frames[3:] = 200  # game 1's starts are bright, game 0's dark
    np.savez(
        out / "starts.npz",
        frames=frames,
        buttons=np.zeros((6, 4, 8), np.uint8),
        look=np.zeros((6, 4, 2), np.float32),
        game_id=np.array([0, 0, 0, 1, 1, 1]),
    )
    s = cfg.sampler
    (out / "dream.json").write_text(
        json.dumps(
            {
                "format": 1,
                "id": "both",
                "game": "both",
                "game_id": 0,
                "games": {"x": 0, "y": 1},
                "num_games": 2,
                "resolution": [16, 16],
                "context_frames": 4,
                "action_history": True,
                "sigma_data": 0.5,
                "sampler": {
                    "steps": 2,
                    "sigma_min": s.sigma_min,
                    "sigma_max": s.sigma_max,
                    "rho": s.rho,
                },
            }
        )
    )
    pack = DreamPack.load(out)
    for seed in range(5):
        assert Dream(pack, seed=seed, game="y").reset().mean() > 150
        assert Dream(pack, seed=seed, game="x").reset().mean() < 50


def test_first_step_keeps_the_start_keys_aligned(pack):
    """Training pairs context frame j with the action taken at frame j; the player's key is last."""
    p = DreamPack.load(pack[0].path)
    st = dict(p.starts())
    st["buttons"] = np.zeros_like(st["buttons"])
    for j in range(st["buttons"].shape[1]):
        st["buttons"][:, j, j] = 1  # a different key in every slot
    p._starts = st
    d = Dream(p, seed=0)
    d.reset(2)
    recorded = st["buttons"][2].astype(np.float32)
    seen = {}
    real_generate = d.generate
    d.generate = lambda **kw: (seen.setdefault("b", d.buttons.copy()), real_generate(**kw))[1]
    d.step(2)  # hold right
    assert np.array_equal(seen["b"][0, :-1], recorded[:-1])
    assert np.array_equal(seen["b"][0, -1], unpack_buttons(2))
