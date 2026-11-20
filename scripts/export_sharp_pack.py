"""Package a trained upsampler as the app's "Sharp" quality pack.

python scripts/export_sharp_pack.py --checkpoint runs/<upsampler run>/latest.pt --out app/src-tauri/resources/models

Writes <out>/sharp/{sharp.json, upsampler.onnx} and checks ONNX against PyTorch.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from lucid.models.factory import load_model
from lucid.play.export_onnx import export_upsampler


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()
    model, cfg, extra = load_model(args.checkpoint, "cpu")
    out = Path(args.out) / "sharp"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    export_upsampler(model, cfg, out / "upsampler.onnx")
    W, H = cfg.data.resolution
    lw, lh = cfg.model.low_resolution
    s = cfg.sampler
    meta = {
        "format": 1,
        "kind": "upsampler",
        "resolution": [W, H],
        "low_resolution": [lw, lh],
        "num_games": cfg.model.num_games,
        "games": extra.get("games"),
        "use_prev_frame": cfg.model.use_prev_frame,
        "sampler": {
            "steps": args.steps,
            "sigma_min": s.sigma_min,
            "sigma_max": s.sigma_max,
            "rho": s.rho,
            "prev_noise": 0.05,
        },
        "trained_from": Path(args.checkpoint).resolve().parent.name,
    }
    (out / "sharp.json").write_text(json.dumps(meta, indent=1))

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "upsampler.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    feeds = {
        "x": rng.standard_normal((1, 3, H, W)).astype(np.float32),
        "sigma": np.array([1.0], np.float32),
        "low_up": rng.uniform(-1, 1, (1, 3, H, W)).astype(np.float32),
        "prev": rng.uniform(-1, 1, (1, 3, H, W)).astype(np.float32),
        "game_weights": np.eye(cfg.model.num_games, dtype=np.float32)[1:2],
        "s_low": np.zeros(1, np.float32),
        "s_prev": np.array([0.05], np.float32),
    }
    o = sess.run(None, feeds)[0]
    with torch.no_grad():
        ref = (
            model.float()
            .denoise(
                *(
                    torch.from_numpy(feeds[n])
                    for n in ["x", "sigma", "low_up", "prev", "game_weights", "s_low", "s_prev"]
                )
            )
            .numpy()
        )
    diff = float(np.abs(o - ref).max())
    if diff > 1e-3:
        raise SystemExit(f"ONNX output differs from PyTorch by {diff}")
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"wrote {out} ({size / 1e6:.1f} MB, ONNX vs PyTorch max diff {diff:.2e})")


if __name__ == "__main__":
    main()
