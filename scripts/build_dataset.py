"""Validate raw recordings and build a processed, versioned dataset plus its statistics report."""

import argparse

from lucid.data.build import build_dataset
from lucid.data.stats import dataset_report
from lucid.utils.config import BuildConfig, load_config
from lucid.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    cfg = load_config(args.config, args.set, cls=BuildConfig)
    seed_everything(cfg.splits.seed)
    manifest = build_dataset(cfg)
    dataset_report(cfg.out_root, cfg.version)
    out = f"{cfg.out_root}/{cfg.version}"
    print(f"stats: {out}/stats.md  |  sample grid to check by eye: {out}/examples.png")
    for split, games in manifest["splits"].items():
        for g, d in games.items():
            print(f"  {split:<10} {g:<12} {d['episodes']:>5} episodes {d['frames']:>8} frames")


if __name__ == "__main__":
    main()
