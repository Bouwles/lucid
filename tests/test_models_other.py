import torch

from lucid.models.autoencoder import Autoencoder
from lucid.models.factory import build_model
from lucid.models.upsampler import gaussian_blur
from lucid.utils.config import RunConfig


def test_upsampler_shapes_and_loss():
    cfg = RunConfig()
    cfg.model.type = "upsampler"
    cfg.model.base_channels, cfg.model.channel_mults, cfg.model.attention_resolutions = (
        16,
        [1, 2],
        [],
    )
    cfg.model.res_blocks_per_level, cfg.model.cond_dim = 1, 32
    cfg.data.resolution = [32, 32]
    m = build_model(cfg)
    b = {
        "low": torch.rand(2, 3, 16, 16) * 2 - 1,
        "high": torch.rand(2, 3, 32, 32) * 2 - 1,
        "prev_high": torch.rand(2, 3, 32, 32) * 2 - 1,
        "has_prev": torch.tensor([True, False]),
        "game_id": torch.zeros(2, dtype=torch.long),
    }
    loss, _ = m.training_loss(b)
    loss.backward()
    assert torch.isfinite(loss)
    out = m.eval().generate(b["low"], b["game_id"], b["prev_high"], steps=2)
    assert out.shape == (2, 3, 32, 32)


def test_baseline_shapes():
    cfg = RunConfig()
    cfg.model.type = "baseline"
    cfg.model.base_channels, cfg.model.channel_mults, cfg.model.attention_resolutions = (
        16,
        [1, 2],
        [],
    )
    cfg.data.resolution = [16, 16]
    m = build_model(cfg)
    ctx = torch.zeros(2, 4, 3, 16, 16)
    out = m.generate(
        ctx, torch.zeros(2, 4, 8), torch.zeros(2, 4, 2), torch.zeros(2, dtype=torch.long)
    )
    assert out.shape == (2, 3, 16, 16)


def test_blur_keeps_shape_and_mean():
    x = torch.rand(3, 3, 16, 16)
    y = gaussian_blur(x, torch.tensor([0.0, 0.5, 2.0]))
    assert y.shape == x.shape
    assert torch.allclose(y[0], x[0], atol=1e-4)  # sigma 0 is (nearly) no blur
    assert torch.allclose(y.mean(dim=(1, 2, 3)), x.mean(dim=(1, 2, 3)), atol=0.02)


def test_autoencoder_shapes():
    ae = Autoencoder(base_channels=16, mults=(1, 2), latent_channels=4)
    recon, z = ae(torch.randn(2, 3, 32, 32))
    assert z.shape == (2, 4, 8, 8) and recon.shape == (2, 3, 32, 32)
