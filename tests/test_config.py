import pytest

from lucid.utils.config import BuildConfig, RunConfig, load_config, parse_overrides, save_config

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_overrides_are_typed():
    assert parse_overrides(["train.lr=3e-4", "data.games=[a,b]", "x.y=true"]) == {
        "train": {"lr": 3e-4},
        "data": {"games": ["a", "b"]},
        "x": {"y": True},
    }


def test_every_shipped_train_config_loads():
    for p in (ROOT / "configs/train").glob("*.yaml"):
        cfg = load_config(p)
        assert isinstance(cfg, RunConfig), p


def test_every_shipped_data_config_loads():
    for p in (ROOT / "configs/data").glob("*.yaml"):
        assert isinstance(load_config(p, cls=BuildConfig), BuildConfig)


def test_base_merging_and_overrides():
    cfg = load_config(ROOT / "configs/train/small_no_ctx_noise.yaml", ["train.lr=1e-4"])
    assert cfg.context_noise.enabled is False
    assert cfg.model.base_channels == 96  # inherited from small -> model/small
    assert cfg.train.lr == 1e-4


def test_unknown_key_fails_loudly(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("train: {lrr: 0.1}\n")
    with pytest.raises(KeyError):
        load_config(p)


def test_wrong_type_fails(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("train: {max_steps: lots}\n")
    with pytest.raises(TypeError):
        load_config(p)


def test_save_roundtrip(tmp_path):
    cfg = load_config(ROOT / "configs/train/tiny.yaml")
    save_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg
