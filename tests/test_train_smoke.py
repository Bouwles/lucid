"""Train for a few steps end to end on synthetic data, then reload and evaluate pieces."""

import json

import torch

from lucid.data.dataset import load_episodes
from lucid.eval.probes import probe_specs, train_probe
from lucid.models.factory import load_model
from lucid.train.loop import train
from lucid.utils.config import load_config

from conftest import ROOT


def test_tiny_training_runs_and_resumes(synthetic_dataset, tmp_path, monkeypatch):
    root, version = synthetic_dataset
    monkeypatch.setattr(
        "lucid.eval.metrics.LPIPS.try_create", staticmethod(lambda device="cpu": None)
    )
    overrides = [
        f"data.root={root}",
        f"data.version={version}",
        "data.resolution=[32, 32]",
        "model.channel_mults=[1, 2]",
        "model.base_channels=16",
        "model.cond_dim=32",
        "train.device=cpu",
        f"train.out_dir={tmp_path}",
        "train.max_steps=6",
        "train.val_every=3",
        "train.ckpt_every=3",
        "train.batch_size=4",
        "train.log_every=2",
        "train.val_batches=1",
        "train.val_rollout_steps=4",
    ]
    cfg = load_config(ROOT / "configs/train/smoke_synthetic.yaml", overrides)
    run = train(cfg, log=lambda *_: None)
    assert (run / "best.pt").exists() and (run / "latest.pt").exists()
    info = json.loads((run / "run_info.json").read_text())
    assert info["dataset_version"] == version and "git_commit" in info
    assert (run / "config.yaml").exists()
    assert list((run / "samples").glob("rollout_*.gif")) and list(
        (run / "samples").glob("action_grid_*.png")
    )
    model, cfg2, extra = load_model(run / "latest.pt")
    assert extra["games"] == {"synthetic": 0}

    cfg.train.max_steps = 8
    run2 = train(cfg, resume=str(run / "latest.pt"), log=lambda *_: None)
    assert run2 == run
    assert torch.load(run / "latest.pt", weights_only=False)["step"] == 8


def test_probe_trains(synthetic_dataset):
    root, version = synthetic_dataset
    eps, _ = load_episodes(root, version, "train", ["synthetic"])
    spec = next(s for s in probe_specs(eps) if s.name == "lane")
    net, acc = train_probe(
        spec, eps, eps, torch.device("cpu"), steps=60, batch_size=32, log=lambda *_: None
    )
    assert 0.0 <= acc <= 1.0
