"""One training loop for every LUCID model (SPEC 13.3)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lucid.data.dataset import (
    SequenceDataset,
    UpsampleDataset,
    WindowDataset,
    collate,
    load_episodes,
    load_manifest,
    mixture_sampler,
    to_model_space,
    to_uint8,
)
from lucid.data.format import BUTTONS
from lucid.eval.metrics import LPIPS, frame_metrics
from lucid.eval.rollout import rollout, sampler_kwargs
from lucid.models.factory import build_model
from lucid.train.checkpoint import BestKeeper, load_checkpoint, save_checkpoint
from lucid.train.ema import EMA
from lucid.train.logging import RunLogger, make_run_dir, write_run_info
from lucid.utils.config import RunConfig, load_games, to_dict
from lucid.utils.device import autocast, autocast_dtype, pick_device
from lucid.utils.seed import seed_everything
from lucid.utils.video import grid, save_gif, save_png, side_by_side

IMAGE_KEYS = ("context", "target", "frames", "low", "high", "prev_high")


def prepare_batch(batch: dict, device: torch.device) -> dict:
    """Move a batch to the device and convert uint8 images to float in [-1, 1] there."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device, non_blocking=True)
            out[k] = to_model_space(v) if k in IMAGE_KEYS else v
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def lr_lambda(cfg: RunConfig):
    t = cfg.train

    def f(step: int) -> float:
        if step < t.warmup_steps:
            return (step + 1) / max(1, t.warmup_steps)
        if t.schedule == "constant":
            return 1.0
        p = (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return f


def check_dataset(cfg: RunConfig, manifest: dict) -> None:
    if list(manifest["resolution"]) != list(
        cfg.model.low_resolution if cfg.model.type == "upsampler" else cfg.data.resolution
    ):
        raise ValueError(
            f"config resolution {cfg.data.resolution} does not match dataset {manifest['resolution']}"
        )
    if manifest["frame_skip"] != cfg.data.frame_skip:
        raise ValueError(
            f"config frame_skip {cfg.data.frame_skip} does not match dataset {manifest['frame_skip']}"
        )
    games = load_games(cfg.games_file)
    for g in cfg.data.games:
        if games[g]["id"] >= cfg.model.num_games:
            raise ValueError(
                f"game {g} has id {games[g]['id']} but model.num_games={cfg.model.num_games}"
            )


def _loader(ds, cfg: RunConfig, device, seed: int) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    sampler = mixture_sampler(ds, cfg.data.mixture, num_samples=10**8, generator=gen)
    nw = cfg.data.num_workers
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=nw > 0,
        drop_last=True,
    )


def train(
    cfg: RunConfig,
    resume: str | None = None,
    max_minutes: float | None = None,
    log=print,
) -> Path:
    """Train until cfg.train.max_steps (or max_minutes, or Ctrl+C). Returns the run directory."""
    seed_everything(cfg.seed)
    device = pick_device(cfg.train.device)
    amp = autocast_dtype(device, cfg.train.precision)
    d = cfg.data
    manifest = load_manifest(d.root, d.version)
    check_dataset(cfg, manifest)
    is_ups = cfg.model.type == "upsampler"

    train_eps, _ = load_episodes(d.root, d.version, "train", d.games, d.preload, with_full=is_ups)
    val_eps, _ = load_episodes(d.root, d.version, "val", d.games, d.preload, with_full=is_ups)
    if not val_eps:
        log(
            "warning: the val split is empty, validating on train episodes (fine for smoke tests only)"
        )
        val_eps = train_eps
    if is_ups:
        train_ds, val_ds = UpsampleDataset(train_eps), UpsampleDataset(val_eps)
    else:
        train_ds = WindowDataset(train_eps, d.context_frames, d.action_history, d.pad_start)
        val_ds = WindowDataset(val_eps, d.context_frames, d.action_history, d.pad_start)
    loader = _loader(train_ds, cfg, device, cfg.seed)

    model = build_model(cfg, manifest.get("sigma_data")).to(device)
    ema = EMA(model, cfg.train.ema_decay)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.99)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp == torch.float16)

    run_dir = make_run_dir(cfg.train.out_dir, cfg.run_name, resume)
    extra = {
        "dataset_version": d.version,
        "sigma_data": model.sigma_data if hasattr(model, "sigma_data") else None,
        "games": {g: manifest["games"][g]["id"] for g in d.games},
        "resolution": manifest["full_resolution"] if is_ups else manifest["resolution"],
        "padding": {g: manifest["games"][g]["padding"] for g in d.games},
    }
    step = 0
    if resume:
        ckpt = load_checkpoint(resume, model=model, ema=ema, optimizer=opt, scheduler=sched)
        step = ckpt["step"]
        log(f"resumed from {resume} at step {step}")
    else:
        n_params = sum(p.numel() for p in model.parameters())
        write_run_info(
            run_dir,
            cfg,
            d.version,
            device,
            {"parameters": n_params, "train_samples": len(train_ds)},
        )
        log(
            f"run {run_dir}  |  {n_params / 1e6:.2f}M parameters  |  {len(train_ds)} training samples  |  {device}, autocast={amp}"
        )
    logger = RunLogger(run_dir)
    best = BestKeeper(run_dir, cfg.train.keep_best)
    lpips = LPIPS.try_create(device)

    def save(path):
        save_checkpoint(
            path,
            model=model,
            ema=ema.model,
            optimizer=opt,
            scheduler=sched,
            step=step,
            config=to_dict(cfg),
            extra=extra,
        )

    t0 = time.time()
    t_log, n_seen = time.time(), 0
    data_iter = iter(loader)
    model.train()
    try:
        while step < cfg.train.max_steps:
            opt.zero_grad(set_to_none=True)
            logs_acc: dict[str, float] = {}
            for _ in range(cfg.train.grad_accum):
                batch = prepare_batch(next(data_iter), device)
                with autocast(device, amp):
                    loss, logs = model.training_loss(batch)
                scaler.scale(loss / cfg.train.grad_accum).backward()
                for k, v in logs.items():
                    logs_acc[k] = logs_acc.get(k, 0.0) + v / cfg.train.grad_accum
                n_seen += cfg.train.batch_size
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            if not torch.isfinite(gnorm):
                log(f"step {step}: non-finite gradient norm, skipping update (see SPEC Appendix C)")
                scaler.update()
                step += 1
                continue
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model, step)
            step += 1

            if step % cfg.train.log_every == 0 or step == 1:
                dt = time.time() - t_log
                logs_acc.update(
                    lr=sched.get_last_lr()[0],
                    grad_norm=gnorm.item(),
                    samples_per_sec=n_seen / max(dt, 1e-6),
                    memory_gb=_memory_gb(device),
                )
                logger.scalars(step, logs_acc, "train/")
                log(
                    f"step {step:>7}  loss {logs_acc.get('loss', float('nan')):.4f}  "
                    f"lr {logs_acc['lr']:.2e}  {logs_acc['samples_per_sec']:.0f} samples/s"
                )
                t_log, n_seen = time.time(), 0

            if step % cfg.train.val_every == 0 or step == cfg.train.max_steps:
                metric = validate(
                    cfg, ema.model, val_ds, val_eps, device, amp, logger, step, run_dir, lpips, log
                )
                best.consider(metric, step, save)
                model.train()
            if step % cfg.train.ckpt_every == 0:
                save(run_dir / "latest.pt")
            if max_minutes is not None and time.time() - t0 > max_minutes * 60:
                log(f"time limit of {max_minutes} minutes reached at step {step}")
                break
    except KeyboardInterrupt:
        log("interrupted, saving latest.pt")
    finally:
        save(run_dir / "latest.pt")
        logger.close()
    log(f"done: {run_dir}  ({(time.time() - t0) / 60:.1f} min, step {step})")
    return run_dir


def _memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    return 0.0


def _img(x: torch.Tensor) -> np.ndarray:
    """float [3, H, W] in [-1, 1] -> uint8 [H, W, 3]."""
    return to_uint8(x).permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def validate(cfg, model, val_ds, val_eps, device, amp, logger, step, run_dir, lpips, log) -> float:
    """One-step metrics, an action grid and a free-running rollout GIF. Returns the metric to minimise."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    rng = np.random.default_rng(0)
    idx = rng.choice(
        len(val_ds),
        size=min(len(val_ds), cfg.train.val_batches * cfg.train.batch_size),
        replace=False,
    )
    sk = sampler_kwargs(cfg.sampler)
    totals: dict[str, list] = {}
    bs = cfg.train.batch_size
    for i in range(0, len(idx), bs):
        batch = prepare_batch(collate([val_ds[int(j)] for j in idx[i : i + bs]]), device)
        with autocast(device, amp):
            if model.kind == "upsampler":
                pred = model.generate(
                    batch["low"],
                    batch["game_id"],
                    prev_high=None,
                    steps=cfg.sampler.steps,
                    method=cfg.sampler.type,
                    sigma_max=cfg.sampler.sigma_max,
                    generator=gen,
                )
                target = batch["high"]
                bil = frame_metrics(model.upscale(batch["low"]), target, lpips)
                for k, v in bil.items():
                    totals.setdefault(f"bilinear_{k}", []).append(v.float().cpu())
            else:
                pred = model.generate(
                    batch["context"],
                    batch["buttons"],
                    batch["look"],
                    batch["game_id"],
                    generator=gen,
                    **sk,
                )
                target = batch["target"]
        for k, v in frame_metrics(pred, target, lpips).items():
            totals.setdefault(k, []).append(v.float().cpu())
    metrics = {k: torch.cat(v).mean().item() for k, v in totals.items()}
    logger.scalars(step, metrics, "val/")
    log("val  " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

    samples = run_dir / "samples"
    if model.kind != "upsampler":
        _action_grid(cfg, model, val_ds, device, amp, gen, logger, step, samples)
        _rollout_gif(cfg, model, val_eps, device, amp, gen, logger, step, samples)
    return metrics.get("lpips", metrics["mse"])


def _action_grid(cfg, model, val_ds, device, amp, gen, logger, step, out_dir):
    """Rows: fixed contexts. Columns: the last context frame, then the prediction for each action.
    Identical columns mean the model is ignoring the action (SPEC 14.7)."""
    games = load_games(cfg.games_file)
    used = set()
    for g in cfg.data.games:
        for v in (games[g].get("actions") or {}).values():
            used.update(v if isinstance(v, list) else [v])
    actions = [None] + [BUTTONS.index(b) for b in BUTTONS if b in used]
    rng = np.random.default_rng(7)
    rows = []
    sk = sampler_kwargs(cfg.sampler)
    for j in rng.choice(len(val_ds), size=min(4, len(val_ds)), replace=False):
        s = prepare_batch(collate([val_ds[int(j)]]), device)
        row = [_img(s["context"][0, -1])]
        for a in actions:
            # With an action history, keep the real past actions and change only the newest one.
            b = s["buttons"].clone()
            newest = b[:, -1] if b.dim() == 3 else b
            newest.zero_()
            if a is not None:
                newest[:, a] = 1
            with autocast(device, amp):
                pred = model.generate(s["context"], b, s["look"], s["game_id"], generator=gen, **sk)
            row.append(_img(pred[0]))
        rows.append(row)
    img = grid(rows, scale=2)
    save_png(img, out_dir / f"action_grid_{step:07d}.png")
    logger.image(step, "val/action_grid", img)


def _rollout_gif(cfg, model, val_eps, device, amp, gen, logger, step, out_dir):
    k = cfg.data.context_frames
    length = cfg.train.val_rollout_steps
    seq = SequenceDataset(val_eps, k, length)
    if len(seq) == 0:
        length = max(1, max(len(e) for e in val_eps) - k)
        seq = SequenceDataset(val_eps, k, length)
        if len(seq) == 0:
            return
    s = prepare_batch(collate([seq[0]]), device)
    with autocast(device, amp):
        pred = rollout(
            model,
            s["frames"],
            s["buttons"],
            s["look"],
            s["game_id"],
            length,
            cfg.data.action_history,
            generator=gen,
            **sampler_kwargs(cfg.sampler),
        )
    real = [_img(f) for f in s["frames"][0, k:]]
    dream = [_img(f) for f in pred[0]]
    frames = side_by_side([real, dream], ["real", "dream"], scale=3)
    save_gif(frames, out_dir / f"rollout_{step:07d}.gif")
    logger.video(step, "val/rollout", frames)
