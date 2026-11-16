import pytest
import torch

from lucid.models.unet import ConditionalUNet


def make(k=4, res=(32, 32), history=4, games=2):
    torch.manual_seed(0)
    return ConditionalUNet(
        cond_channels=3 * k,
        base_channels=16,
        channel_mults=(1, 2, 2),
        res_blocks=1,
        attention_resolutions=(8,),
        input_resolution=res,
        cond_dim=32,
        num_games=games,
        action_history=history,
        scalar_names=("noise",),
    )


@pytest.mark.parametrize("B,k,H,W", [(1, 4, 32, 32), (3, 2, 16, 32), (2, 4, 32, 48)])
def test_output_shape(B, k, H, W):
    net = make(k, (W, H))
    out = net(
        torch.randn(B, 3, H, W),
        torch.randn(B, 3 * k, H, W),
        torch.zeros(B, dtype=torch.long),
        torch.zeros(B, 4, 8),
        torch.zeros(B, 4, 2),
        {"noise": torch.randn(B)},
    )
    assert out.shape == (B, 3, H, W)


def _inputs(B=2, k=4):
    torch.manual_seed(1)
    return (
        torch.randn(B, 3, 32, 32),
        torch.randn(B, 3 * k, 32, 32),
        torch.zeros(B, dtype=torch.long),
    )


def test_action_changes_output():
    net = make().eval()
    x, ctx, g = _inputs()
    s = {"noise": torch.zeros(2)}
    b0 = torch.zeros(2, 4, 8)
    b1 = b0.clone()
    b1[:, -1, 0] = 1
    out0 = net(x, ctx, g, b0, torch.zeros(2, 4, 2), s)
    out1 = net(x, ctx, g, b1, torch.zeros(2, 4, 2), s)
    assert (out0 - out1).abs().max() > 1e-4


def test_game_id_and_blending():
    net = make().eval()
    x, ctx, _ = _inputs()
    args = (torch.zeros(2, 4, 8), torch.zeros(2, 4, 2), {"noise": torch.zeros(2)})
    a = net(x, ctx, torch.zeros(2, dtype=torch.long), *args)
    b = net(x, ctx, torch.ones(2, dtype=torch.long), *args)
    assert (a - b).abs().max() > 1e-4
    onehot = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert torch.allclose(net(x, ctx, onehot, *args), a, atol=1e-5)


def test_gradients_reach_action_embedding():
    net = make()
    x, ctx, g = _inputs()
    b = torch.zeros(2, 4, 8)
    b[:, :, 4] = 1
    net(x, ctx, g, b, torch.zeros(2, 4, 2), {"noise": torch.zeros(2)}).pow(2).mean().backward()
    grad = net.action_embed.buttons[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_null_action_drop():
    net = make().eval()
    x, ctx, g = _inputs()
    s = {"noise": torch.zeros(2)}
    drop = torch.tensor([True, True])
    b = torch.rand(2, 4, 8)
    a1 = net(x, ctx, g, b, torch.zeros(2, 4, 2), s, drop)
    a2 = net(x, ctx, g, torch.zeros(2, 4, 8), torch.zeros(2, 4, 2), s, drop)
    assert torch.allclose(a1, a2)  # a dropped action ignores what was pressed


def test_bad_resolution_raises():
    net = make()
    with pytest.raises(ValueError):
        net(
            torch.randn(1, 3, 30, 30),
            torch.randn(1, 12, 30, 30),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, 4, 8),
            torch.zeros(1, 4, 2),
            {"noise": torch.zeros(1)},
        )
