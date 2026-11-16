import pytest
import torch

from lucid.models.sampler import guided, refine, sample, sigma_schedule


def test_schedule_endpoints():
    s = sigma_schedule(5, 0.002, 80.0, 7.0)
    assert s.shape == (6,)
    assert (
        s[0].item() == pytest.approx(80.0) and s[-2].item() == pytest.approx(0.002) and s[-1] == 0
    )
    assert (s[:-1][1:] < s[:-1][:-1]).all()


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 6, 8, 16])
@pytest.mark.parametrize("method", ["euler", "heun"])
def test_perfect_denoiser_returns_target(steps, method):
    target = torch.rand(2, 3, 8, 8) * 2 - 1
    out = sample(lambda x, s: target, torch.randn(2, 3, 8, 8), steps=steps, method=method)
    assert torch.allclose(out, target, atol=1e-4)


def test_refine_with_perfect_denoiser():
    target = torch.zeros(1, 3, 4, 4)
    assert torch.equal(refine(lambda x, s: target, target + 0.3, 0.5), target)


def test_guidance():
    c, u = torch.ones(1), torch.zeros(1)
    assert guided(c, u, 1.0).item() == 1.0 and guided(c, u, 2.0).item() == 2.0
