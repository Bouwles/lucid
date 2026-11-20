"""Train a LUCID model: python scripts/train.py --config configs/train/small.yaml [--set train.lr=1e-4]"""

import argparse

from lucid.train.loop import train
from lucid.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--resume", default=None, help="path to a latest.pt to resume from")
    p.add_argument("--init", default=None, help="start a new run from another checkpoint's weights")
    p.add_argument("--max-minutes", type=float, default=None, help="stop cleanly after this long")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    train(cfg, resume=args.resume, init=args.init, max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
