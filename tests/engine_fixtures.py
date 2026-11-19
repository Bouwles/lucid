"""A tiny random-weight dream pack exported to ONNX, for engine tests."""

import json
from pathlib import Path

import numpy as np

from lucid.models.factory import build_model
from lucid.play.export_onnx import export_denoiser
from lucid.utils.config import RunConfig


def tiny_cfg() -> RunConfig:
    cfg = RunConfig()
    cfg.data.resolution = [16, 16]
    cfg.model.base_channels, cfg.model.channel_mults = 8, [1, 2]
    cfg.model.attention_resolutions, cfg.model.res_blocks_per_level, cfg.model.cond_dim = [], 1, 16
    cfg.sampler.steps = 2
    return cfg


def make_pack(root: Path, pack_id: str = "bramble"):
    import torch

    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = build_model(cfg).eval()
    out = root / pack_id
    out.mkdir(parents=True)
    export_denoiser(model, cfg, out / "denoiser.onnx")
    rng = np.random.default_rng(0)
    np.savez(
        out / "starts.npz",
        frames=rng.integers(0, 255, (3, 4, 16, 16, 3), dtype=np.uint8),
        buttons=np.zeros((3, 4, 8), np.uint8),
        look=np.zeros((3, 4, 2), np.float32),
    )
    s = cfg.sampler
    (out / "dream.json").write_text(
        json.dumps(
            {
                "format": 1,
                "id": pack_id,
                "game": pack_id,
                "game_id": 0,
                "num_games": 1,
                "resolution": [16, 16],
                "context_frames": 4,
                "action_history": True,
                "sigma_data": 0.5,
                "sampler": {
                    "steps": 2,
                    "sigma_min": s.sigma_min,
                    "sigma_max": s.sigma_max,
                    "rho": s.rho,
                    "context_noise": 0.1,
                },
                "quality": "fast",
            }
        )
    )
    return out, model, cfg
