import numpy as np
import torch

from lucid.eval.metrics import curve_steps, psnr, ssim
from lucid.eval.rollout import actions_for, rollout
from lucid.eval.stability import real_ranges, sharpness, survival_time


def test_metrics_basics():
    x = torch.rand(2, 3, 16, 16) * 2 - 1
    assert torch.isinf(psnr(x, x)).logical_not().all() and psnr(x, x).min() > 90
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-4)
    assert curve_steps(64) == [1, 2, 4, 8, 16, 32, 64]


def test_sharpness_drops_with_blur():
    x = (torch.rand(1, 3, 32, 32) > 0.5).float() * 2 - 1
    blurred = torch.nn.functional.avg_pool2d(x, 5, 1, 2)
    assert sharpness(blurred) < sharpness(x) / 4


def test_survival_time():
    lo, hi = np.zeros(7), np.ones(7)
    good = np.full((100, 7), 0.5)
    assert survival_time(good, lo, hi) == 100
    bad = good.copy()
    bad[40:, 0] = 5.0
    assert survival_time(bad, lo, hi, patience=10) == 40
    blip = good.copy()
    blip[50:53, 0] = 5.0  # a short blip is not a breakdown
    assert survival_time(blip, lo, hi, patience=10) == 100


def test_real_ranges_shape():
    lo, hi = real_ranges(torch.rand(20, 3, 16, 16) * 2 - 1)
    assert lo.shape == hi.shape == (7,) and (hi >= lo).all()


class CopyModel:
    """Predicts the last context frame plus the button index: makes rollout bookkeeping checkable."""

    k = 3

    def generate(self, ctx, buttons, look, game, generator=None, **kw):
        a = buttons if buttons.dim() == 2 else buttons[:, -1]
        return ctx[:, -1] * 0 + a.argmax(-1).float()[:, None, None, None]


def test_rollout_uses_the_right_actions_and_feeds_back():
    B, L, k = 1, 5, 3
    frames = torch.zeros(B, k + L, 3, 2, 2)
    buttons = torch.zeros(B, k + L, 8)
    for j in range(k + L):
        buttons[0, j, j % 8] = 1
    out = rollout(
        CopyModel(), frames, buttons, torch.zeros(B, k + L, 2), torch.zeros(B), L, history=True
    )
    # frame t uses the action at t - 1
    assert out[0, :, 0, 0, 0].tolist() == [float((t - 1) % 8) for t in range(k, k + L)]
    b, _ = actions_for(buttons, torch.zeros(B, k + L, 2), 4, k, history=False)
    assert b.argmax(-1).item() == 3
