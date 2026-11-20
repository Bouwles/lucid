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
    p.add_argument(
        "--id", required=True, help="pack id, e.g. drift-kit, or lucid-multi for a multi-game pack"
    )
    p.add_argument(
        "--game",
        default=None,
        help="export just this game (default: every game the model was trained on)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--starts", type=int, default=64, help="real starting contexts to include")
    p.add_argument("--split", default="test")
    p.add_argument(
        "--steps", type=int, default=None, help="default sampler steps (from the config if unset)"
    )
    p.add_argument(
        "--cfg-scale", type=float, default=1.0, help="classifier-free guidance strength (1 = off)"
    )
    p.add_argument("--zip", action="store_true")
    args = p.parse_args()
    seed_everything(0)
    model, cfg, extra = load_model(args.checkpoint, "cpu")
    games = extra.get("games") or {g: i for i, g in enumerate(cfg.data.games)}
    # A pack can hold one game or several (a multi-game model). --game picks one of them.
    pack_games = {args.game: games[args.game]} if args.game else dict(games)
    k, d = cfg.data.context_frames, cfg.data
    G = cfg.model.num_games

    out = Path(args.out) / args.id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    export_denoiser(model, cfg, out / "denoiser.onnx")

    rng = np.random.default_rng(0)
    frames, buttons, look, gids = [], [], [], []
    for name, gid in pack_games.items():
        eps, _ = load_episodes(d.root, d.version, args.split, [name])
        candidates = [(e, t) for e in eps for t in range(0, len(e) - k, max(1, len(e) // 8))]
        n = min(args.starts, len(candidates))
        for j in rng.choice(len(candidates), size=n, replace=False):
            e, t = candidates[int(j)]
            if not bool(e.labels["alive"][t : t + k].all()):
                continue
            frames.append(np.asarray(e.frames[t : t + k]))
            buttons.append(e.buttons[t : t + k])
            look.append(e.look[t : t + k])
            gids.append(gid)
    np.savez_compressed(
        out / "starts.npz",
        frames=np.stack(frames),
        buttons=np.stack(buttons).astype(np.uint8),
        look=np.stack(look).astype(np.float32),
        game_id=np.asarray(gids, np.int64),
    )

    s = cfg.sampler
    first = next(iter(pack_games))
    meta = {
        "format": 1,
        "id": args.id,
        "game": first if len(pack_games) == 1 else args.id,
        "game_id": int(pack_games[first]),
        "games": {name: int(gid) for name, gid in pack_games.items()},
        "num_games": G,
        "game_input": "weights",
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
            "cfg_scale": args.cfg_scale,
        },
        "quality": "fast",
        "trained_from": Path(args.checkpoint).resolve().parent.name,
        "dataset_version": d.version,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "dream.json").write_text(json.dumps(meta, indent=1))

    # Check the ONNX denoiser against PyTorch on a real context, for each game in the pack.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "denoiser.onnx"), providers=["CPUExecutionProvider"])
    st = np.load(out / "starts.npz")
    worst = 0.0
    for i in [int(np.flatnonzero(st["game_id"] == g)[0]) for g in pack_games.values()]:
        ctx = (st["frames"][i : i + 1].astype(np.float32) / 127.5 - 1).transpose(0, 1, 4, 2, 3)
        b = st["buttons"][i : i + 1].astype(np.float32)
        lk = st["look"][i : i + 1]
        if not d.action_history:
            b, lk = b[:, -1], lk[:, -1]
        gw = np.zeros((1, G), np.float32)
        gw[0, st["game_id"][i]] = 1.0
        x = np.random.default_rng(1).standard_normal((1, 3, *ctx.shape[-2:])).astype(np.float32)
        feeds = {
            "x": x,
            "sigma": np.array([1.0], np.float32),
            "context": ctx,
            "buttons": b,
            "look": lk,
            "game_weights": gw,
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
                        for n in [
                            "x",
                            "sigma",
                            "context",
                            "buttons",
                            "look",
                            "game_weights",
                            "sigma_ctx",
                        ]
                    )
                )
                .numpy()
            )
        worst = max(worst, float(np.abs(onnx_out - ref).max()))
    if worst > 1e-3:
        raise SystemExit(f"ONNX output differs from PyTorch by {worst}; not shipping this pack")
    size = sum(f.stat().st_size for f in out.iterdir())
    print(
        f"wrote {out} ({size / 1e6:.1f} MB, {len(frames)} starts, games {list(pack_games)}, "
        f"ONNX vs PyTorch max diff {worst:.2e})"
    )

    if args.zip:
        z = Path(shutil.make_archive(str(Path(args.out) / args.id), "zip", root_dir=out))
        print(f"{z}  sha256 {hashlib.sha256(z.read_bytes()).hexdigest()}  {z.stat().st_size} bytes")


if __name__ == "__main__":
    main()
