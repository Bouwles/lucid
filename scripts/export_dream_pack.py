"""Package a trained world model as a dream pack for the desktop app (docs/engine-api.md).

python scripts/export_dream_pack.py --checkpoint runs/<run>/best.pt --id drift-kit --out app/src-tauri/resources/models

Writes <out>/<id>/{dream.json, denoiser.onnx, starts.npz} and checks the ONNX denoiser matches
PyTorch before finishing. With --zip it also writes <id>.zip and prints its SHA-256 for the
catalogue.
"""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from lucid.data.dataset import load_episodes
from lucid.models.factory import load_model
from lucid.play.export_onnx import export_denoiser
from lucid.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--id", required=True, help="catalogue id, e.g. test-track or drift-kit")
    p.add_argument(
        "--game", default=None, help="game name in the training data (default: the only one)"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--starts", type=int, default=64, help="real starting contexts to include")
    p.add_argument("--split", default="test")
    p.add_argument(
        "--steps", type=int, default=None, help="default sampler steps (from the config if unset)"
    )
    p.add_argument("--zip", action="store_true")
    args = p.parse_args()
    seed_everything(0)
    model, cfg, extra = load_model(args.checkpoint, "cpu")
    games = extra.get("games") or {g: i for i, g in enumerate(cfg.data.games)}
    game = args.game or next(iter(games))
    k, d = cfg.data.context_frames, cfg.data

    out = Path(args.out) / args.id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    export_denoiser(model, cfg, out / "denoiser.onnx")

    eps, _ = load_episodes(d.root, d.version, args.split, [game])
    rng = np.random.default_rng(0)
    frames, buttons, look = [], [], []
    candidates = [(e, t) for e in eps for t in range(0, len(e) - k, max(1, len(e) // 8))]
    for j in rng.choice(len(candidates), size=min(args.starts, len(candidates)), replace=False):
        e, t = candidates[int(j)]
        if not bool(e.labels["alive"][t : t + k].all()):
            continue
        frames.append(np.asarray(e.frames[t : t + k]))
        buttons.append(e.buttons[t : t + k])
        look.append(e.look[t : t + k])
    np.savez_compressed(
        out / "starts.npz",
        frames=np.stack(frames),
        buttons=np.stack(buttons).astype(np.uint8),
        look=np.stack(look).astype(np.float32),
    )

    s = cfg.sampler
    meta = {
        "format": 1,
        "id": args.id,
        "game": args.id,
        "game_id": int(games[game]),
        "num_games": cfg.model.num_games,
        "resolution": list(d.resolution),
        "context_frames": k,
        "action_history": d.action_history,
        "sigma_data": float(extra.get("sigma_data") or cfg.model.sigma_data),
        "sampler": {
            "steps": args.steps or s.steps,
            "sigma_min": s.sigma_min,
            "sigma_max": s.sigma_max,
            "rho": s.rho,
            "context_noise": s.inference_context_noise,
        },
        "quality": "fast",
        "trained_from": Path(args.checkpoint).resolve().parent.name,
        "dataset_version": d.version,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "dream.json").write_text(json.dumps(meta, indent=1))

    # Check the ONNX denoiser against PyTorch on a real context.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "denoiser.onnx"), providers=["CPUExecutionProvider"])
    st = np.load(out / "starts.npz")
    ctx = (st["frames"][:1].astype(np.float32) / 127.5 - 1).transpose(0, 1, 4, 2, 3)
    b = (
        st["buttons"][:1].astype(np.float32)
        if d.action_history
        else st["buttons"][:1, -1].astype(np.float32)
    )
    lk = st["look"][:1] if d.action_history else st["look"][:1, -1]
    x = np.random.default_rng(1).standard_normal((1, 3, *ctx.shape[-2:])).astype(np.float32)
    feeds = {
        "x": x,
        "sigma": np.array([1.0], np.float32),
        "context": ctx,
        "buttons": b,
        "look": lk,
        "game_id": np.array([games[game]], np.int64),
        "sigma_ctx": np.array([0.1], np.float32),
        "drop_action": np.array([0.0], np.float32),
    }
    onnx_out = sess.run(None, feeds)[0]
    with torch.no_grad():
        ref = (
            model.float()
            .denoise(
                *(
                    torch.from_numpy(feeds[n])
                    for n in ["x", "sigma", "context", "buttons", "look", "game_id", "sigma_ctx"]
                )
            )
            .numpy()
        )
    diff = float(np.abs(onnx_out - ref).max())
    if diff > 1e-3:
        raise SystemExit(f"ONNX output differs from PyTorch by {diff}; not shipping this pack")
    size = sum(f.stat().st_size for f in out.iterdir())
    print(
        f"wrote {out} ({size / 1e6:.1f} MB, {len(frames)} starts, ONNX vs PyTorch max diff {diff:.2e})"
    )

    if args.zip:
        z = Path(shutil.make_archive(str(Path(args.out) / args.id), "zip", root_dir=out))
        print(f"{z}  sha256 {hashlib.sha256(z.read_bytes()).hexdigest()}  {z.stat().st_size} bytes")


if __name__ == "__main__":
    main()
