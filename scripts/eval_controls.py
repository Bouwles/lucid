"""Does the dream obey the controls? Measured with an inverse dynamics model (lucid/eval/idm.py).

python scripts/eval_controls.py --checkpoint runs/<run>/best.pt [--cfg 1.0 2.0 3.0]

1. Trains an IDM on the real training frames (or loads --idm), and reports how well it reads each
   button from real held-out frames. That is the ceiling.
2. From real starting contexts, holds one button at a time in the dream and asks the IDM which
   buttons the dreamed frames show. "Obeys" is the share of dreams where it sees the commanded button.
Optionally repeats step 2 at several classifier-free guidance scales.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lucid.data.dataset import SequenceDataset, collate, load_episodes
from lucid.data.format import BUTTONS
from lucid.eval.idm import balanced_accuracy, load_idm, predict, save_idm, train_idm
from lucid.eval.rollout import rollout, sampler_kwargs
from lucid.models.factory import load_model
from lucid.train.loop import prepare_batch
from lucid.utils.device import autocast, autocast_dtype, pick_device
from lucid.utils.seed import seed_everything


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--idm",
        default=None,
        help="a saved IDM (otherwise one is trained and saved next to the run)",
    )
    p.add_argument("--gap", type=int, default=4, help="steps between the IDM's two frames")
    p.add_argument("--starts", type=int, default=96)
    p.add_argument("--cfg", type=float, nargs="*", default=[1.0])
    p.add_argument("--device", default="auto")
    p.add_argument("--idm-steps", type=int, default=3000)
    p.add_argument("--game", default=None, help="for a multi-game model: evaluate this game only")
    p.add_argument(
        "--dataset", default=None, help="evaluate on this dataset version instead of the model's"
    )
    args = p.parse_args()
    seed_everything(0)
    device = pick_device(args.device)
    model, cfg, extra = load_model(args.checkpoint, device)
    amp = autocast_dtype(device, cfg.train.precision)
    d, k = cfg.data, cfg.data.context_frames
    if args.game:
        d.games = [args.game]
    if args.dataset:
        d.version = args.dataset
    train_eps, _ = load_episodes(d.root, d.version, "train", d.games)
    test_eps, _ = load_episodes(d.root, d.version, "test", d.games) or (train_eps, None)

    tag = f"_{args.game}" if args.game else ""
    idm_path = (
        Path(args.idm) if args.idm else Path(d.root) / d.version / f"idm_gap{args.gap}{tag}.pt"
    )
    if idm_path.exists():
        idm, gap, used, real_acc = load_idm(idm_path, device)
    else:
        idm, real_acc, used = train_idm(train_eps, test_eps, device, args.gap, args.idm_steps)
        gap = args.gap
        save_idm(idm_path, idm, gap, used, real_acc)
    print(
        "IDM on real held-out frames (balanced accuracy, 0.5 = chance):",
        {b: round(v, 3) for b, v in real_acc.items()},
    )

    seq = SequenceDataset(test_eps, k, gap, stride=5)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(seq), size=min(args.starts, len(seq)), replace=False)
    starts = [seq[int(i)] for i in idx]
    starts = (
        [s for s in starts if bool(s["labels"]["alive"].all())]
        if "alive" in starts[0]["labels"]
        else starts
    )
    b = prepare_batch(collate(starts), device)
    B = b["frames"].shape[0]
    buttons_to_test = [j for j, u in enumerate(used) if u]
    results = {"real_idm_accuracy": real_acc, "gap": gap, "n_starts": B, "by_cfg": {}}

    for scale in args.cfg:
        sk = sampler_kwargs(cfg.sampler, cfg_scale=None if scale == 1.0 else scale)
        obeys = {}
        for j in buttons_to_test:
            acts = b["buttons"].clone()
            acts[:, k - 1 :] = 0
            acts[:, k - 1 :, j] = 1
            gen = torch.Generator(device=device).manual_seed(1)
            with autocast(device, amp):
                pred = rollout(
                    model,
                    b["frames"],
                    acts,
                    b["look"],
                    b["game_id"],
                    gap,
                    d.action_history,
                    generator=gen,
                    **sk,
                )
            before, after = b["frames"][:, k - 1].float(), pred[:, -1].float()
            prob = predict(idm, before, after)
            target = np.zeros((B, len(BUTTONS)), np.float32)
            target[:, j] = 1
            obeys[BUTTONS[j]] = float((prob[:, j] > 0.5).mean())
            # how often the IDM also sees a *different* used button, i.e. the dream did something else
            others = [o for o in buttons_to_test if o != j]
            obeys[BUTTONS[j] + "_other_detected"] = (
                float((prob[:, others] > 0.5).any(1).mean()) if others else 0.0
            )
            _ = balanced_accuracy  # same scoring module; kept for the real-data ceiling
        results["by_cfg"][str(scale)] = obeys
        print(
            f"cfg {scale}: dream obeys",
            {b_: round(v, 3) for b_, v in obeys.items() if not b_.endswith("_other_detected")},
        )

    out = Path(args.checkpoint).resolve().parent / f"controls{tag}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
