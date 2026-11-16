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
