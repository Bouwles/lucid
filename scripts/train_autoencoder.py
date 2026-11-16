"""Train the small frame autoencoder for the optional latent-diffusion path.

Check the saved reconstructions by eye: small details such as obstacles must survive.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from lucid.data.dataset import load_episodes, to_model_space, to_uint8
from lucid.eval.metrics import LPIPS, psnr
from lucid.models.autoencoder import Autoencoder
from lucid.utils.config import load_config
from lucid.utils.device import pick_device
from lucid.utils.seed import seed_everything
from lucid.utils.video import grid, save_png


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="a train config, for the dataset")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--lpips-weight", type=float, default=0.1)
    p.add_argument("--out", default="runs/autoencoder")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = pick_device(args.device)
    d = cfg.data
    train_eps, _ = load_episodes(d.root, d.version, "train", d.games)
    val_eps, _ = load_episodes(d.root, d.version, "val", d.games)
    frames = np.concatenate([np.asarray(e.frames) for e in train_eps])
    val = np.concatenate([np.asarray(e.frames) for e in (val_eps or train_eps)])[:64]
    ae = Autoencoder(latent_channels=args.latent_channels).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=2e-4)
    lpips = LPIPS.try_create(device) if args.lpips_weight > 0 else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for step in range(1, args.steps + 1):
        x = to_model_space(
            torch.from_numpy(frames[rng.integers(len(frames), size=args.batch_size)])
            .to(device)
            .permute(0, 3, 1, 2)
        )
        loss, _ = ae.training_loss(x, lpips.module if lpips else None, args.lpips_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == args.steps:
            with torch.no_grad():
                xv = to_model_space(torch.from_numpy(val).to(device).permute(0, 3, 1, 2))
                rv, z = ae(xv)
            print(
                f"step {step} loss {loss.item():.4f} val psnr {psnr(rv, xv).mean():.2f} latent {tuple(z.shape[1:])}"
            )
            pairs = [
                [
                    to_uint8(a).permute(1, 2, 0).cpu().numpy(),
                    to_uint8(b).permute(1, 2, 0).cpu().numpy(),
                ]
                for a, b in zip(xv[:8], rv[:8], strict=True)
            ]
            save_png(grid(pairs, scale=3), out / f"recon_{step:07d}.png")
            torch.save(
                {"model": ae.state_dict(), "latent_channels": args.latent_channels},
                out / "autoencoder.pt",
            )


if __name__ == "__main__":
    main()
