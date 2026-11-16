"""Full evaluation of a world model checkpoint.

python scripts/evaluate.py --checkpoint runs/<run>/best.pt [--probes data/processed/<v>/probes.pt]
    [--compare runs/<baseline>/best.pt runs/<no_ctx_noise>/best.pt] [--to-docs]

Writes results.json, results.md, plots and GIFs to runs/<run>/eval_<step>/. With --to-docs the
Markdown section is also appended to docs/results.md, marked as not yet re-checked against the run logs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lucid.data.dataset import SequenceDataset, WindowDataset, collate, load_episodes, to_uint8
from lucid.data.format import BUTTONS
from lucid.eval.control import controllability, real_controllability
from lucid.eval.metrics import LPIPS, curve_steps, frame_metrics
from lucid.eval.probes import accuracy, load_probes
from lucid.eval.report import plot_band, plot_curves, results_section
from lucid.eval.rollout import actions_for, rollout, sampler_kwargs
from lucid.eval.stability import frame_features, real_ranges, survival_time
from lucid.eval.turing import realism_curve, train_real_vs_dream
from lucid.models.factory import load_model
from lucid.train.loop import prepare_batch
from lucid.utils.config import load_games
from lucid.utils.device import autocast, autocast_dtype, pick_device
from lucid.utils.gitinfo import git_commit
from lucid.utils.seed import seed_everything
from lucid.utils.video import save_gif, side_by_side


def img(x: torch.Tensor) -> np.ndarray:
    return to_uint8(x.float()).permute(1, 2, 0).cpu().numpy()


def used_buttons(cfg) -> list[str]:
    games = load_games(cfg.games_file)
    used = set()
    for g in cfg.data.games:
        for v in (games[g].get("actions") or {}).values():
            used.update(v if isinstance(v, list) else [v])
    return [b for b in BUTTONS if b in used]


def random_persistent_actions(
    B: int, T: int, buttons: list[str], rng: np.random.Generator, device
) -> torch.Tensor:
    """Scripted actions for long rollouts: a random action (or none) held for 1-8 steps. [B, T, 8]."""
    out = torch.zeros(B, T, 8)
    choices = [None] + [BUTTONS.index(b) for b in buttons]
    for b in range(B):
        t = 0
        while t < T:
            a, hold = choices[rng.integers(len(choices))], int(rng.integers(1, 9))
            if a is not None:
                out[b, t : t + hold, a] = 1
            t += hold
    return out.to(device)


@torch.no_grad()
def one_step(model, episodes, cfg, device, amp, lpips, sk, n: int, seed: int) -> dict:
    ds = WindowDataset(
        episodes, cfg.data.context_frames, cfg.data.action_history, cfg.data.pad_start
    )
    if len(ds) == 0:
        return {"n": 0}
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    gen = torch.Generator(device=device).manual_seed(seed)
    acc: dict[str, list] = {}
    for i in range(0, len(idx), 64):
        b = prepare_batch(collate([ds[int(j)] for j in idx[i : i + 64]]), device)
        with autocast(device, amp):
            pred = model.generate(
                b["context"], b["buttons"], b["look"], b["game_id"], generator=gen, **sk
            )
        for k, v in frame_metrics(pred, b["target"], lpips).items():
            acc.setdefault(k, []).append(v.float().cpu())
    return {"n": len(idx), **{k: torch.cat(v).mean().item() for k, v in acc.items()}}


@torch.no_grad()
def free_running(
    model, batch, cfg, device, amp, sk, length: int, seed: int, teacher_forced=False
) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    with autocast(device, amp):
        return rollout(
            model,
            batch["frames"],
            batch["buttons"],
            batch["look"],
            batch["game_id"],
            length,
            cfg.data.action_history,
            teacher_forced=teacher_forced,
            generator=gen,
            **sk,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--compare", nargs="*", default=[], help="other checkpoints for the same LPIPS plot"
    )
    p.add_argument("--probes", default=None, help="probes.pt from scripts/train_probes.py")
    p.add_argument("--split", default="test")
    p.add_argument("--one-step-samples", type=int, default=512)
    p.add_argument("--sequences", type=int, default=32)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--long-steps", type=int, default=2000)
    p.add_argument("--long-rollouts", type=int, default=8)
    p.add_argument("--hold-steps", type=int, default=8)
    p.add_argument("--steps", type=int, default=None, help="override sampler steps")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--to-docs", action="store_true", help="append the section to docs/results.md")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    model, cfg, extra = load_model(args.checkpoint, device)
    amp = autocast_dtype(device, cfg.train.precision)
    ckpt_step = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["step"]
    out = Path(args.out or Path(args.checkpoint).resolve().parent / f"eval_step{ckpt_step:07d}")
    out.mkdir(parents=True, exist_ok=True)
    sk = sampler_kwargs(cfg.sampler, **({"steps": args.steps} if args.steps else {}))
    d, k = cfg.data, cfg.data.context_frames
    lpips = LPIPS.try_create(device)
    print(f"evaluating {args.checkpoint} on {device} -> {out}")

    res = {
        "run": Path(args.checkpoint).resolve().parent.name,
        "checkpoint": str(args.checkpoint),
        "step": ckpt_step,
        "dataset_version": d.version,
        "git_commit": git_commit(short=True),
        "sampler": f"{sk['method']}, {sk['steps']} steps, context noise {sk['context_noise']}",
    }

    # 1. one-step metrics on test and test_human
    splits = {}
    for split in [args.split, "test_human"]:
        eps, _ = load_episodes(d.root, d.version, split, d.games)
        splits[split] = eps
    res["one_step"] = {
        s: one_step(model, e, cfg, device, amp, lpips, sk, args.one_step_samples, args.seed)
        for s, e in splits.items()
        if e
    }
    print("one-step:", json.dumps(res["one_step"]))

    # 2. free-running LPIPS curve on the same sequences for every compared model
    test_eps = splits[args.split]
    seq = SequenceDataset(test_eps, k, args.length)
    if len(seq) == 0:
        raise SystemExit(
            f"no {args.split} episodes are longer than {k + args.length} frames; lower --length"
        )
    pick = np.random.default_rng(args.seed).choice(
        len(seq), size=min(args.sequences, len(seq)), replace=False
    )
    batch = prepare_batch(collate([seq[int(i)] for i in pick]), device)
    real = batch["frames"][:, k:]
    steps_at = curve_steps(args.length)
    models = {"this model": (model, cfg, sk)}
    for c in args.compare:
        m2, cfg2, _ = load_model(c, device)
        models[Path(c).resolve().parent.name] = (m2, cfg2, sampler_kwargs(cfg2.sampler))
    curves, dreams = {}, {}
    for label, (m, c2, sk2) in models.items():
        if c2.data.context_frames != k:
            print(f"skipping {label}: different context length")
            continue
        pred = free_running(m, batch, c2, device, amp, sk2, args.length, args.seed)
        dreams[label] = pred
        metric = lpips if lpips is not None else None
        curve = {}
        for s in steps_at:
            fm = frame_metrics(pred[:, s - 1], real[:, s - 1], metric)
            curve[s] = fm["lpips" if metric else "mse"].mean().item()
        curves[label] = curve
    res["rollout_curves"] = {lab: {str(s): v for s, v in c.items()} for lab, c in curves.items()}
    plot_curves(curves, "LPIPS" if lpips else "MSE", out / "lpips_curve.png")
    pred = dreams["this model"]
    tf = free_running(
        model, batch, cfg, device, amp, sk, args.length, args.seed, teacher_forced=True
    )
    for i in range(min(4, len(pick))):
        cols = [[img(f) for f in real[i]], [img(f) for f in tf[i]], [img(f) for f in pred[i]]]
        save_gif(
            side_by_side(cols, ["real", "teacher-forced", "free-running"], scale=3),
            out / f"rollout_{i}.gif",
        )

    # 3. long-horizon stability with scripted actions
    rng = np.random.default_rng(args.seed)
    real_frames = torch.cat([batch["frames"].flatten(0, 1)])
    lo, hi = real_ranges(real_frames)
    B, T = min(args.long_rollouts, len(pick)), args.long_steps
    acts = random_persistent_actions(B, T + k, used_buttons(cfg), rng, device)
    look = torch.zeros(B, T + k, 2, device=device)
    acts[:, : k - 1] = batch["buttons"][:B, : k - 1]
    ctx = batch["frames"][:B, :k]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    feats, keep = [], []
    t0 = time.time()
    for t in range(k, k + T):
        bi, li = actions_for(acts, look, t, k, cfg.data.action_history)
        with autocast(device, amp):
            nxt = model.generate(ctx, bi, li, batch["game_id"][:B], generator=gen, **sk)
        ctx = torch.cat([ctx[:, 1:], nxt[:, None].float()], 1)
        feats.append(frame_features(nxt.float()).cpu().numpy())
        keep.append(to_uint8(nxt.float()).cpu())
    feats_arr = np.stack(feats, 1)  # [B, T, 7]
    surv = [survival_time(feats_arr[b], lo, hi) for b in range(B)]
    res["stability"] = {
        "n": B,
        "steps": T,
        "median_survival": int(np.median(surv)),
        "min_survival": int(min(surv)),
        "max_survival": int(max(surv)),
        "survival": surv,
        "seconds": round(time.time() - t0, 1),
    }
    plot_band(
        {f"rollout {b}": feats_arr[b, :, 0].tolist() for b in range(B)},
        lo[0],
        hi[0],
        "sharpness (Laplacian variance)",
        out / "sharpness.png",
    )
    frames_u8 = torch.stack(keep, 1)  # [B, T, 3, H, W]
    for rank, b in enumerate(np.argsort(surv)[: min(3, B)]):
        s0 = max(0, surv[b] - 60)
        clip = [f.permute(1, 2, 0).numpy() for f in frames_u8[b, s0 : surv[b] + 60]]
        save_gif(clip, out / "failures" / f"failure_{rank}_rollout{b}_step{surv[b]}.gif")
    print("stability:", {k_: v for k_, v in res["stability"].items() if k_ != "survival"})

    # 4. probes and controllability
    if args.probes:
        probes, _ = load_probes(args.probes, device)
        res["probe_accuracy"] = {}
        for name, net in probes.items():
            ys = np.concatenate([e.labels[name] for e in test_eps])
            xs = np.concatenate([np.asarray(e.frames) for e in test_eps])
            m = ys >= 0
            res["probe_accuracy"][name] = accuracy(net, xs[m], ys[m].astype(np.int64), device)
        starts_ds = SequenceDataset(test_eps, k, args.hold_steps, stride=8)
        idx = rng.choice(len(starts_ds), size=min(256, len(starts_ds)), replace=False)
        starts = [prepare_batch(starts_ds[int(i)], device) for i in idx]
        starts = [s for s in starts if bool(s["labels"]["alive"][k - 1])]
        with autocast(device, amp):
            res["controllability"] = controllability(
                model,
                probes,
                starts,
                set(used_buttons(cfg)),
                args.hold_steps,
                cfg.data.action_history,
                sk,
            )
        res["controllability_real"] = real_controllability(test_eps, hold_steps=args.hold_steps)
        print(
            "controllability:", {n: round(v["score"], 3) for n, v in res["controllability"].items()}
        )
    else:
        print(
            "no --probes given: skipping controllability (train them with scripts/train_probes.py)"
        )

    # 5. realism classifier: train on half the sequences, measure on the other half
    half = max(1, len(pick) // 2)
    if len(pick) >= 4:
        net = train_real_vs_dream(
            real[:half].flatten(0, 1).float(), pred[:half].flatten(0, 1).float(), device
        )
        res["realism"] = realism_curve(net, pred[half:].float(), real[half:].flatten(0, 1).float())

    # 6. speed (batch 1)
    b1 = {kk: v[:1] for kk, v in batch.items() if isinstance(v, torch.Tensor)}
    bi, li = actions_for(b1["buttons"], b1["look"], k, k, cfg.data.action_history)
    for i in range(25):
        if i == 5:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        with autocast(device, amp):
            x = model.generate(b1["frames"][:, :k], bi, li, b1["game_id"], **sk)
    x.sum().item()
    ms = 1000 * (time.perf_counter() - t0) / 20
    res["speed"] = {"device": str(device), "ms_per_frame": ms, "fps": 1000 / ms}

    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    md = results_section(res)
    (out / "results.md").write_text(md)
    print(f"\nwrote {out}/results.md")
    if args.to_docs:
        with open("docs/results.md", "a") as f:
            f.write("\n" + md)
        print("appended to docs/results.md (re-check it against the run logs before relying on it)")


if __name__ == "__main__":
    main()
