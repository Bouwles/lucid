"""Train a pixels-to-buttons policy by copying the autopilot (behaviour cloning), for dream mode.

python scripts/train_policy.py --game bramble --dataset bramble_v1 [--out app/src-tauri/resources/models/bramble]

Writes policy.onnx (and policy.pt) into the game's dream pack folder, so the app's dream player
can let the policy play inside the dream.
"""

import argparse
import json
from pathlib import Path

import torch

from lucid.data.dataset import load_episodes
from lucid.policy import export_policy, train_bc
from lucid.utils.device import pick_device
from lucid.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--game", required=True)
    p.add_argument(
        "--dataset", required=True, help="a dataset of pure autopilot play, e.g. bramble_v1"
    )
    p.add_argument("--root", default="data/processed")
    p.add_argument("--out", default=None, help="default: app/src-tauri/resources/models/<game>")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    seed_everything(0)
    device = pick_device(args.device)
    train_eps, man = load_episodes(args.root, args.dataset, "train", [args.game])
    val_eps, _ = load_episodes(args.root, args.dataset, "val", [args.game])
    pol, acc, used = train_bc(train_eps, val_eps or train_eps, device, steps=args.steps)
    print(
        f"{args.game}: per-button agreement with the autopilot on held-out frames:",
        {b: round(v, 3) for b, v in acc.items()},
    )
    out = Path(args.out or f"app/src-tauri/resources/models/{args.game}")
    out.mkdir(parents=True, exist_ok=True)
    W, H = man["resolution"]
    export_policy(pol, out / "policy.onnx", pol.k, H, W)
    torch.save(
        {"state": pol.state_dict(), "k": pol.k, "accuracy": acc, "dataset": args.dataset},
        out / "policy.pt",
    )
    (out / "policy.json").write_text(
        json.dumps(
            {"k": pol.k, "accuracy": acc, "dataset": args.dataset, "kind": "behaviour cloning"},
            indent=1,
        )
    )
    print(f"wrote {out / 'policy.onnx'}")


if __name__ == "__main__":
    main()
