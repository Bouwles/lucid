"""Train the state probes on real frames only: python scripts/train_probes.py --config configs/train/small.yaml"""

import argparse
from pathlib import Path

from lucid.data.dataset import load_episodes
from lucid.eval.probes import probe_specs, save_probes, train_probe
from lucid.utils.config import load_config
from lucid.utils.device import pick_device
from lucid.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", required=True, help="a train config (for the dataset version and games)"
    )
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--out", default=None, help="default: data/processed/<version>/probes.pt")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.seed)
    device = pick_device(args.device)
    d = cfg.data
    train_eps, _ = load_episodes(d.root, d.version, "train", d.games)
    val_eps, _ = load_episodes(d.root, d.version, "val", d.games)
    if not val_eps:
        raise SystemExit("the val split is empty; probes need held-out real frames")
    probes, accs = {}, {}
    specs = probe_specs(train_eps)
    for spec in specs:
        net, acc = train_probe(spec, train_eps, val_eps, device, steps=args.steps)
        probes[spec.name], accs[spec.name] = net, acc
        print(f"probe {spec.name}: {acc:.1%} accuracy on real held-out frames")
    out = Path(args.out or Path(d.root) / d.version / "probes.pt")
    save_probes(out, probes, specs, accs)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
