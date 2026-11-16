import json

import numpy as np

from lucid.data.build import assign_split, letterbox_geometry, resize_frame, summarise_actions
from lucid.data.dataset import load_episodes
from lucid.data.format import button_bit
from lucid.utils.config import SplitConfig

L, R, J = button_bit("left"), button_bit("right"), button_bit("jump")


def test_frame_skip_keeps_every_nth_step():
    kept, b, lk = summarise_actions(np.zeros(10, np.uint8), np.zeros((10, 2)), 4)
    assert kept.tolist() == [0, 4, 8] and len(b) == 3


def test_action_rules():
    buttons = np.array([0, J, 0, 0, L, 0, R, 0], np.uint8)
    look = np.zeros((8, 2))
    assert summarise_actions(buttons, look, 4, "first")[1].tolist() == [0, L]
    assert summarise_actions(buttons, look, 4, "first_non_none")[1].tolist() == [J, L]
    # union: left and right in the same interval cancel out
    assert summarise_actions(buttons, look, 4, "union")[1].tolist() == [J, 0]


def test_look_is_averaged_and_stays_in_range():
    look = np.array([[1, -1]] * 4 + [[0.5, 0.5]] * 4, np.float32)
    _, _, lk = summarise_actions(np.zeros(8, np.uint8), look, 4)
    assert np.allclose(lk, [[1, -1], [0.5, 0.5]])


def test_letterbox_pads_instead_of_stretching():
    size, pad = letterbox_geometry((256, 144), (64, 64))
    assert size == (64, 36) and pad[1] + pad[3] == 28 and pad[0] == pad[2] == 0
    out = resize_frame(np.full((144, 256, 3), 200, np.uint8), (64, 64))
    assert out.shape == (64, 64, 3) and out[0].max() == 0 and out[32].min() == 200


def test_split_is_stable_and_by_episode():
    cfg = SplitConfig(val_fraction=0.1, test_fraction=0.1, human_holdout_fraction=1.0, seed=3)
    a = [assign_split(f"s/ep_{i:06d}", False, cfg) for i in range(500)]
    assert a == [assign_split(f"s/ep_{i:06d}", False, cfg) for i in range(500)]
    assert 0.05 < a.count("val") / 500 < 0.15
    assert assign_split("s/ep_000001", True, cfg) == "test_human"


def test_built_dataset_has_no_episode_in_two_splits(synthetic_dataset):
    root, version = synthetic_dataset
    manifest = json.loads((root / version / "manifest.json").read_text())
    keys = [(e["session"], e["episode"]) for e in manifest["episodes"]]
    assert len(keys) == len(set(keys))
    humans = [e for e in manifest["episodes"] if e["source"] == "human"]
    assert all(e["split"] in ("test_human", "train", "val", "test") for e in humans)
    assert manifest["sigma_data"] is not None and 0.1 < manifest["sigma_data"] < 1.5


def test_processed_shapes(synthetic_dataset):
    root, version = synthetic_dataset
    eps, _ = load_episodes(root, version, "train", ["synthetic"], with_full=True)
    e = eps[0]
    assert e.frames.shape[1:] == (32, 32, 3) and e.frames.dtype == np.uint8
    assert e.full_frames.shape[1:] == (64, 64, 3)
    assert e.buttons.shape == (len(e), 8)
    assert set(e.labels) >= {"lane", "vertical", "alive"}
