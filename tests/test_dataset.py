import numpy as np
import torch

from lucid.data.dataset import (
    Episode,
    SequenceDataset,
    WindowDataset,
    collate,
    mixture_sampler,
    to_model_space,
    to_uint8,
)


def make_episode(T: int, value_offset: int = 0, game: str = "a", gid: int = 0) -> Episode:
    # frame t is filled with value t, so windows can be checked exactly
    frames = (np.arange(T)[:, None, None, None] + value_offset).astype(np.uint8) * np.ones(
        (1, 4, 4, 3), np.uint8
    )
    buttons = np.zeros((T, 8), np.uint8)
    buttons[np.arange(T), np.arange(T) % 8] = 1
    return Episode(
        "e",
        game,
        gid,
        frames,
        buttons,
        np.zeros((T, 2), np.float32),
        {"lane": np.arange(T, dtype=np.int16), "alive": np.ones(T, bool)},
    )


def test_window_contents_and_action_alignment():
    ds = WindowDataset([make_episode(10)], context_frames=3, action_history=True, pad_start="skip")
    s = ds[0]  # target t=3, context 0,1,2, actions at 0,1,2
    assert s["context"][:, 0, 0, 0].tolist() == [0, 1, 2]
    assert s["target"][0, 0, 0].item() == 3
    assert s["buttons"].argmax(-1).tolist() == [0, 1, 2]
    assert s["labels"]["lane"].item() == 3


def test_windows_never_cross_episodes():
    eps = [make_episode(6, 0), make_episode(6, 100)]
    ds = WindowDataset(eps, context_frames=3, pad_start="skip")
    assert len(ds) == 2 * (6 - 3)
    for i in range(len(ds)):
        s = ds[i]
        vals = s["context"][:, 0, 0, 0].tolist() + [s["target"][0, 0, 0].item()]
        assert max(vals) - min(vals) == 3  # all from one episode, consecutive


def test_repeat_padding_at_episode_start():
    ds = WindowDataset([make_episode(6)], context_frames=4, action_history=True, pad_start="repeat")
    s = ds[0]  # target t=1
    assert s["context"][:, 0, 0, 0].tolist() == [0, 0, 0, 0]
    assert s["buttons"].sum(-1).tolist() == [0, 0, 0, 1]  # padded actions are "none"
    assert len(ds) == 5


def test_single_action_mode():
    ds = WindowDataset([make_episode(8)], context_frames=2, action_history=False, pad_start="skip")
    assert ds[0]["buttons"].shape == (8,) and ds[0]["buttons"].argmax().item() == 1


def test_sequences_and_collate():
    ds = SequenceDataset([make_episode(20)], context_frames=4, length=8)
    b = collate([ds[0], ds[0]])
    assert b["frames"].shape == (2, 12, 3, 4, 4) and b["labels"]["lane"].shape == (2, 12)


def test_model_space_roundtrip():
    x = torch.arange(256, dtype=torch.uint8)
    assert torch.equal(to_uint8(to_model_space(x)), x)
    assert to_model_space(x).min() == -1 and to_model_space(x).max() == 1


def test_mixture_is_equal_per_game():
    eps = [make_episode(100, game="big", gid=0), make_episode(10, game="small", gid=1)]
    ds = WindowDataset(eps, context_frames=2, pad_start="skip")
    idx = list(mixture_sampler(ds, "equal_per_game", 4000, torch.Generator().manual_seed(0)))
    share_small = np.mean([ds.game_of(i) == "small" for i in idx])
    assert 0.45 < share_small < 0.55
