"""Serve the browser demo: python scripts/serve.py --checkpoint runs/<run>/best.pt, then open http://localhost:8000"""

import argparse
import logging

import uvicorn

from lucid.models.factory import load_model
from lucid.play.engine import DreamEngine
from lucid.play.server import create_app
from lucid.utils.config import load_config
from lucid.utils.device import pick_device
from lucid.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", help="world model checkpoint (EMA weights are used)")
    p.add_argument(
        "--random-weights", action="store_true", help="untrained model, for UI smoke tests"
    )
    p.add_argument(
        "--config", default="configs/train/tiny.yaml", help="config for --random-weights"
    )
    p.add_argument(
        "--upsampler", default=None, help="optional upsampler checkpoint (high-res mode)"
    )
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="sampler steps (default: from the checkpoint config)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-players", type=int, default=4)
    p.add_argument("--idle-timeout", type=float, default=120.0)
    p.add_argument("--fps", type=float, default=20.0, help="frame rate cap per player")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    seed_everything(0)
    device = pick_device(args.device)
    if args.random_weights:
        cfg = load_config(args.config)
        from lucid.models.factory import build_model

        model, extra = (
            build_model(cfg).to(device).eval(),
            {"games": {g: i for i, g in enumerate(cfg.data.games)}},
        )
    elif args.checkpoint:
        model, cfg, extra = load_model(args.checkpoint, device)
    else:
        p.error("pass --checkpoint or --random-weights")

    def factory() -> DreamEngine:
        return DreamEngine(
            model=model,
            cfg=cfg,
            extra=extra,
            device=device,
            sampler_steps=args.steps,
            upsampler_path=args.upsampler,
        )

    app = create_app(factory, args.max_players, args.idle_timeout, args.fps)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
