import math

import pytest
import torch

from lucid.models.edm import DiffusionWorldModel, edm_coefficients, loss_weight
from lucid.models.unet import ConditionalUNet


@pytest.mark.parametrize("sigma", [0.002, 0.5, 1.0, 80.0])
def test_preconditioning_matches_hand_computed(sigma):
    sd = 0.5
    c_skip, c_out, c_in, c_noise = edm_coefficients(torch.tensor([sigma]), sd)
    assert c_skip.item() == pytest.approx(sd**2 / (sigma**2 + sd**2), rel=1e-5)
    assert c_out.item() == pytest.approx(sigma * sd / math.sqrt(sigma**2 + sd**2), rel=1e-5)
    assert c_in.item() == pytest.approx(1 / math.sqrt(sigma**2 + sd**2), rel=1e-5)
    assert c_noise.item() == pytest.approx(math.log(sigma) / 4, rel=1e-5)


def test_known_values_at_sigma_data():
    c_skip, c_out, c_in, _ = edm_coefficients(torch.tensor([0.5]), 0.5)
    assert c_skip.item() == pytest.approx(0.5)
    assert c_out.item() == pytest.approx(0.5 / math.sqrt(2))
    assert c_in.item() == pytest.approx(1 / math.sqrt(0.5))


def test_weight_times_cout_squared_is_one():
    """Why the loss can be computed on F directly: lambda(sigma) * c_out(sigma)^2 = 1."""
    s = torch.logspace(-3, 2, 20)
    _, c_out, _, _ = edm_coefficients(s, 0.5)
    assert torch.allclose(loss_weight(s, 0.5) * c_out**2, torch.ones_like(s), rtol=1e-4)


def _model(**kw):
    torch.manual_seed(0)
    unet = ConditionalUNet(
        cond_channels=12,
        base_channels=16,
        channel_mults=(1, 2),
        res_blocks=1,
        attention_resolutions=(),
        input_resolution=(16, 16),
        cond_dim=32,
        action_history=4,
        scalar_names=("noise", "ctx_noise"),
    )
    return DiffusionWorldModel(unet, 4, **kw)


def _batch(B=4):
    torch.manual_seed(1)
    return {
        "context": torch.rand(B, 4, 3, 16, 16) * 2 - 1,
        "target": torch.rand(B, 3, 16, 16) * 2 - 1,
        "buttons": torch.zeros(B, 4, 8),
        "look": torch.zeros(B, 4, 2),
        "game_id": torch.zeros(B, dtype=torch.long),
    }


def test_f_space_loss_equals_weighted_d_loss():
    m = _model(context_noise_enabled=False, p_uncond_action=0.0)
    b = _batch()
    sigma = torch.tensor([0.01, 0.3, 2.0, 50.0])
    y = b["target"]
    x = y + sigma[:, None, None, None] * torch.randn_like(y)
    zero = torch.zeros(4)
    d = m.denoise(x, sigma, b["context"], b["buttons"], b["look"], b["game_id"], zero)
    weighted = (loss_weight(sigma, 0.5)[:, None, None, None] * (d - y) ** 2).mean(dim=(1, 2, 3))
    c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, 0.5)
    f = m.raw(
        c_in[:, None, None, None] * x,
        c_noise,
        b["context"],
        b["buttons"],
        b["look"],
        b["game_id"],
        zero,
    )
    f_loss = ((f - (y - c_skip[:, None, None, None] * x) / c_out[:, None, None, None]) ** 2).mean(
        dim=(1, 2, 3)
    )
    assert torch.allclose(weighted, f_loss, rtol=1e-3)


@pytest.mark.parametrize("p_mean", [-8.0, -1.2, 5.0])
def test_loss_is_finite_for_extreme_sigma(p_mean):
    m = _model(p_mean=p_mean, p_std=0.1)
    loss, logs = m.training_loss(_batch())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_context_noise_distribution():
    m = _model(context_noise_p_clean=0.3, context_noise_max=0.7)
    s = m.sample_context_noise(20000, "cpu")
    assert 0.25 < (s == 0).float().mean() < 0.35
    assert s.max() <= 0.7
    off = _model(context_noise_enabled=False)
    assert (off.sample_context_noise(100, "cpu") == 0).all()


def test_generate_shape_and_range():
    m = _model().eval()
    b = _batch(2)
    out = m.generate(
        b["context"][:2],
        b["buttons"][:2],
        b["look"][:2],
        b["game_id"][:2],
        steps=2,
        cfg_scale=1.5,
        refinement=True,
    )
    assert out.shape == (2, 3, 16, 16) and out.min() >= -1 and out.max() <= 1


def test_motion_weighted_loss_is_finite_and_differs():
    b = _batch()
    # Only a small patch changes between the last context frame and the target.
    b["target"] = b["context"][:, -1].clone()
    b["target"][:, :, 4:8, 4:8] += 0.5
    torch.manual_seed(5)
    plain, _ = _model(motion_weight=0.0).training_loss(b)
    torch.manual_seed(5)
    weighted, _ = _model(motion_weight=8.0).training_loss(b)
    assert torch.isfinite(weighted) and abs(plain.item() - weighted.item()) > 1e-6
