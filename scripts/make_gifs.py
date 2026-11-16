"""README media (SPEC 20, 26.3): side-by-side real | dream clips driven by the same recorded
actions, an optional ablation comparison, a game-switching clip, and a folder of random samples.

python scripts/make_gifs.py --checkpoint runs/<run>/best.pt [--compare runs/<other>/best.pt] --out docs/media

Paul picks the showcase clips. If a clip is a best case, the README says so and links random/.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lucid.data.dataset import SequenceDataset, collate, load_episodes, to_uint8
from lucid.eval.rollout import rollout, sampler_kwargs
from lucid.models.factory import load_model
from lucid.train.loop import prepare_batch
from lucid.utils.device import autocast, autocast_dtype, pick_device
from lucid.utils.seed import seed_everything
from lucid.utils.video import save_gif, side_by_side

MAX_BYTES = 5 * 1024 * 1024


def frames_of(x: torch.Tensor) -> list[np.ndarray]:
    return [to_uint8(f.float()).permute(1, 2, 0).cpu().numpy() for f in x]


def save_capped(frames, path: Path, fps: int) -> None:
    """Save a GIF, dropping every other frame until it fits under 5 MB."""
    while True:
        save_gif(frames, path, fps)
        if path.stat().st_size <= MAX_BYTES or len(frames) < 8:
            return
        frames, fps = frames[::2], max(1, fps // 2)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--compare", default=None, help="second checkpoint, e.g. the no-context-noise ablation"
    )
    p.add_argument("--out", default="docs/media")
    p.add_argument("--length", type=int, default=150)
    p.add_argument(
        "--random", type=int, default=8, help="number of random (not hand-picked) samples"
    )
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    model, cfg, extra = load_model(args.checkpoint, device)
    amp = autocast_dtype(device, cfg.train.precision)
    sk = sampler_kwargs(cfg.sampler)
    k, d = cfg.data.context_frames, cfg.data
    out = Path(args.out)
    eps, _ = load_episodes(d.root, d.version, "test", d.games)
    seq = SequenceDataset(eps, k, args.length, stride=args.length // 2)
    if len(seq) == 0:
        raise SystemExit("no test episodes are long enough; lower --length")
    rng = np.random.default_rng(args.seed)

    def dream(m, items, mcfg, msk, seed):
        b = prepare_batch(collate(items), device)
        gen = torch.Generator(device=device).manual_seed(seed)
        with autocast(device, amp):
            pred = rollout(
                m,
                b["frames"],
                b["buttons"],
                b["look"],
                b["game_id"],
                args.length,
                mcfg.data.action_history,
                generator=gen,
                **msk,
            )
        return b["frames"][:, k:], pred

    picks = rng.choice(len(seq), size=min(args.random, len(seq)), replace=False)
    real, pred = dream(model, [seq[int(i)] for i in picks], cfg, sk, args.seed)
    for j in range(len(picks)):
        clip = side_by_side([frames_of(real[j]), frames_of(pred[j])], ["real", "dream"], args.scale)
        save_capped(clip, out / "random" / f"sample_{j:02d}.gif", args.fps)
    print(f"wrote {len(picks)} random samples to {out / 'random'}")

    if args.compare:
        m2, cfg2, _ = load_model(args.compare, device)
        item = [seq[int(picks[0])]]
        r, a = dream(model, item, cfg, sk, args.seed)
        _, b2 = dream(m2, item, cfg2, sampler_kwargs(cfg2.sampler), args.seed)
        clip = side_by_side(
            [frames_of(r[0]), frames_of(a[0]), frames_of(b2[0])],
            ["real", Path(args.checkpoint).parent.name[-14:], Path(args.compare).parent.name[-14:]],
            args.scale,
        )
        save_capped(clip, out / "ablation.gif", args.fps)
        print(f"wrote {out / 'ablation.gif'}")

    games = list(extra.get("games", {}))
    if len(games) > 1:
        item = prepare_batch(collate([seq[int(picks[0])]]), device)
        ids = [extra["games"][g] for g in games]
        ctx = item["frames"][:, :k]
        frames, per = [], args.length // len(games)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        for gi in ids:
            game = torch.tensor([gi], device=device)
            for _ in range(per):
                b = (
                    torch.zeros(1, k, 8, device=device)
                    if d.action_history
                    else torch.zeros(1, 8, device=device)
                )
                lk = torch.zeros(*b.shape[:-1], 2, device=device)
                with autocast(device, amp):
                    nxt = model.generate(ctx, b, lk, game, generator=gen, **sk)
                ctx = torch.cat([ctx[:, 1:], nxt[:, None].float()], 1)
                frames.append(frames_of(nxt)[0].repeat(args.scale, 0).repeat(args.scale, 1))
        save_capped(frames, out / "game_switch.gif", args.fps)
        print(f"wrote {out / 'game_switch.gif'}")


if __name__ == "__main__":
    main()
