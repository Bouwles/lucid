import os
from pathlib import Path

import pytest
import torch

from lucid.data.synthetic import record_session
from lucid.utils.config import BuildConfig, SplitConfig

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(min(4, os.cpu_count() or 1))


@pytest.fixture(scope="session")
def synthetic_session(tmp_path_factory) -> Path:
    raw = tmp_path_factory.mktemp("raw")
    return record_session(
        raw,
        num_episodes=12,
        seed=0,
        size=32,
        max_steps=80,
        human_every=4,
        session_id="synthetic_20260101_000000",
    )


@pytest.fixture(scope="session")
def synthetic_dataset(synthetic_session, tmp_path_factory) -> tuple[Path, str]:
    from lucid.data.build import build_dataset

    out = tmp_path_factory.mktemp("processed")
    cfg = BuildConfig(
        version="syn_test",
        raw=[str(synthetic_session)],
        out_root=str(out),
        games_file=str(ROOT / "configs/games_synthetic.yaml"),
        games=["synthetic"],
        frame_skip=1,
        resolution=[32, 32],
        full_resolution=[64, 64],
        splits=SplitConfig(val_fraction=0.2, test_fraction=0.2, human_holdout_fraction=0.5),
        workers=1,
    )
    build_dataset(cfg, log=lambda *_: None)
    return out, "syn_test"
