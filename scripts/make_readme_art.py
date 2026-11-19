"""Draw the README artwork (banner, how-it-works diagram, pipeline diagram) as SVG.

The pictures inside the banner and the diagram are real model output: the banner shows the
model's actual denoising steps from pure noise to a frame, and the diagram uses real context
frames and a real prediction. Rerun it on a new checkpoint to refresh the art.

python scripts/make_readme_art.py --checkpoint runs/<run>/best.pt --out docs/media
"""

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lucid.data.dataset import SequenceDataset, collate, load_episodes, to_uint8
from lucid.eval.rollout import actions_for
from lucid.models.factory import load_model
from lucid.models.sampler import sigma_schedule
from lucid.train.loop import prepare_batch
from lucid.utils.seed import seed_everything

SANS = "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
BG, PANEL, LINE, TEXT, MUTED = "#0a0c11", "#11141b", "#262c3a", "#e8eaf0", "#8b93a7"
VIOLET, CYAN = "#a78bfa", "#22d3ee"


def data_uri(frame: np.ndarray, size: int) -> str:
    """uint8 [H, W, 3] -> PNG data URI, upscaled with nearest neighbour so it stays crisp anywhere."""
    s = max(1, size // frame.shape[0])
    img = Image.fromarray(frame.repeat(s, 0).repeat(s, 1))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img(x: torch.Tensor) -> np.ndarray:
    return to_uint8(x.float().clamp(-1, 1)).permute(1, 2, 0).cpu().numpy()


def defs() -> str:
    return f"""<defs>
  <linearGradient id="brand" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{VIOLET}"/><stop offset="1" stop-color="{CYAN}"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.8" cy="0.1" r="0.7">
    <stop offset="0" stop-color="{VIOLET}" stop-opacity="0.22"/>
    <stop offset="0.5" stop-color="{CYAN}" stop-opacity="0.06"/>
    <stop offset="1" stop-color="{BG}" stop-opacity="0"/>
  </radialGradient>
  <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
    <path d="M32 0H0V32" fill="none" stroke="#ffffff" stroke-opacity="0.035"/>
  </pattern>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0 0L10 5L0 10z" fill="{MUTED}"/>
  </marker>
  <marker id="arrowv" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0 0L10 5L0 10z" fill="{VIOLET}"/>
  </marker>
</defs>"""


def background(w: int, h: int) -> str:
    return (
        f'<rect width="{w}" height="{h}" rx="20" fill="{BG}"/>'
        f'<rect width="{w}" height="{h}" rx="20" fill="url(#grid)"/>'
        f'<rect width="{w}" height="{h}" rx="20" fill="url(#glow)"/>'
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="20" fill="none" stroke="{LINE}"/>'
    )


def tile(x: float, y: float, size: float, frame: np.ndarray, rid: str, r: int = 10) -> str:
    return (
        f'<clipPath id="{rid}"><rect x="{x}" y="{y}" width="{size}" height="{size}" rx="{r}"/></clipPath>'
        f'<image href="{data_uri(frame, int(size) * 2)}" x="{x}" y="{y}" width="{size}" height="{size}" '
        f'clip-path="url(#{rid})" style="image-rendering:pixelated"/>'
        f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="{r}" fill="none" stroke="#ffffff" stroke-opacity="0.12"/>'
    )


def banner(trajectory: list[np.ndarray], sigmas: list[float]) -> str:
    W, H = 1280, 520
    n = len(trajectory)
    size, x0, y0 = 150, 72, 300
    gap = (W - 2 * x0 - n * size) / (n - 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        defs(),
        background(W, H),
        f'<text x="{x0 - 6}" y="176" font-family="{SANS}" font-size="128" font-weight="800" letter-spacing="20" fill="url(#brand)">LUCID</text>',
        f'<text x="{x0}" y="226" font-family="{SANS}" font-size="27" fill="{TEXT}">A neural game engine that learns to dream the games I make.</text>',
        f'<text x="{x0}" y="262" font-family="{MONO}" font-size="15" fill="{MUTED}" letter-spacing="1">DIFFUSION WORLD MODEL  /  PYTORCH  /  DESKTOP APP  /  THREE BUILT-IN GAMES</text>',
    ]
    for i, (fr, s) in enumerate(zip(trajectory, sigmas, strict=True)):
        x = x0 + i * (size + gap)
        parts.append(tile(x, y0, size, fr, f"b{i}"))
        if i < n - 1:
            cx = x + size + gap / 2
            parts.append(
                f'<path d="M{cx - 9} {y0 + size / 2 - 7}l8 7l-8 7" fill="none" stroke="{MUTED}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        label = "next frame" if s == 0 else f"σ = {s:.3g}"
        color = TEXT if s == 0 else MUTED
        parts.append(
            f'<text x="{x + size / 2}" y="{y0 + size + 30}" text-anchor="middle" font-family="{MONO}" font-size="14" fill="{color}">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def how_it_works(
    context: list[np.ndarray], mini: list[np.ndarray], nxt: np.ndarray, steps: int
) -> str:
    W, H = 1280, 470
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        defs(),
        background(W, H),
    ]
    # context frames, cascaded
    p.append(
        f'<text x="64" y="64" font-family="{MONO}" font-size="14" fill="{MUTED}" letter-spacing="1">LAST {len(context)} FRAMES</text>'
    )
    for i, fr in enumerate(context):
        p.append(tile(64 + i * 26, 84 + i * 18, 112, fr, f"c{i}", 8))
    # keys
    ky = 330
    p.append(
        f'<text x="64" y="{ky - 16}" font-family="{MONO}" font-size="14" fill="{MUTED}" letter-spacing="1">YOUR KEYS</text>'
    )
    for j, (k, on) in enumerate(
        [("LEFT", False), ("RIGHT", True), ("JUMP", False), ("DOWN", False)]
    ):
        x = 64 + j * 74
        fill = "url(#brand)" if on else PANEL
        col = BG if on else TEXT
        p.append(
            f'<rect x="{x}" y="{ky}" width="64" height="38" rx="8" fill="{fill}" stroke="{LINE}"/>'
            f'<text x="{x + 32}" y="{ky + 24}" text-anchor="middle" font-family="{MONO}" font-size="12" font-weight="700" fill="{col}">{k}</text>'
        )
    p.append(
        f'<text x="64" y="{ky + 76}" font-family="{MONO}" font-size="14" fill="{MUTED}" letter-spacing="1">WHICH GAME</text>'
        f'<rect x="190" y="{ky + 56}" width="92" height="30" rx="15" fill="none" stroke="{VIOLET}"/>'
        f'<text x="236" y="{ky + 76}" text-anchor="middle" font-family="{MONO}" font-size="13" fill="{VIOLET}">drift kit</text>'
    )
    # arrows into the denoiser
    bx, by, bw, bh = 500, 110, 330, 230
    for sx, sy in ((330, 170), (370, 349), (292, 401)):
        p.append(
            f'<path d="M{sx} {sy} C 420 {sy}, 420 {by + bh / 2}, {bx - 6} {by + bh / 2}" fill="none" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#arrow)"/>'
        )
    # the denoiser box
    p.append(
        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="18" fill="{PANEL}" stroke="url(#brand)" stroke-width="2"/>'
        f'<text x="{bx + 28}" y="{by + 46}" font-family="{SANS}" font-size="26" font-weight="700" fill="{TEXT}">Diffusion model</text>'
        f'<text x="{bx + 28}" y="{by + 74}" font-family="{MONO}" font-size="13" fill="{MUTED}">EDM UNet, conditioned on frames, keys and game</text>'
    )
    ms = 64
    for i, fr in enumerate(mini):
        x = bx + 28 + i * (ms + 34)
        p.append(tile(x, by + 104, ms, fr, f"m{i}", 6))
        if i < len(mini) - 1:
            p.append(
                f'<path d="M{x + ms + 12} {by + 128}l8 8l-8 8" fill="none" stroke="{MUTED}" stroke-width="2" stroke-linecap="round"/>'
            )
    p.append(
        f'<text x="{bx + 28}" y="{by + 202}" font-family="{MONO}" font-size="13" fill="{MUTED}">noise to frame in {steps} denoising steps</text>'
    )
    # next frame
    nx, ny, ns = 930, 125, 200
    p.append(
        f'<path d="M{bx + bw + 6} {by + bh / 2} L{nx - 10} {by + bh / 2}" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#arrow)"/>'
    )
    p.append(
        f'<text x="{nx}" y="{ny - 20}" font-family="{MONO}" font-size="14" fill="{TEXT}" letter-spacing="1">NEXT FRAME</text>'
    )
    p.append(tile(nx, ny, ns, nxt, "n0", 12))
    # feedback loop, routed under everything and up the left edge into the context stack
    fx = nx + ns / 2
    p.append(
        f'<path d="M{fx} {ny + ns + 8} V 436 Q {fx} 448 {fx - 12} 448 H 42 Q 30 448 30 436 V 152 Q 30 140 42 140 H 56" '
        f'fill="none" stroke="{VIOLET}" stroke-width="2" stroke-dasharray="7 6" marker-end="url(#arrowv)"/>'
        f'<rect x="470" y="436" width="340" height="24" fill="{BG}"/>'
        f'<text x="640" y="453" text-anchor="middle" font-family="{MONO}" font-size="14" fill="{VIOLET}">fed back in as context, 15 times a second</text>'
    )
    p.append("</svg>")
    return "\n".join(p)


def pipeline() -> str:
    W, H = 1280, 250
    stages = [
        ("Record", "Unity package or JS", "recorder, every logic step"),
        ("Validate", "broken episodes dropped,", "never quietly repaired"),
        ("Build", "frame skip, resize, split", "by episode, versioned"),
        ("Train", "EDM world model and", "upsampler, context noise"),
        ("Play", "DreamEngine over", "WebSocket, in a browser"),
    ]
    bw, gap, x0, y0, bh = 220, 28, 34, 50, 150
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        defs(),
        background(W, H),
    ]
    for i, (t, a, b) in enumerate(stages):
        x = x0 + i * (bw + gap)
        p.append(
            f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="14" fill="{PANEL}" stroke="{LINE}"/>'
            f'<circle cx="{x + 34}" cy="{y0 + 38}" r="15" fill="url(#brand)"/>'
            f'<text x="{x + 34}" y="{y0 + 43}" text-anchor="middle" font-family="{MONO}" font-size="14" font-weight="700" fill="{BG}">{i + 1}</text>'
            f'<text x="{x + 60}" y="{y0 + 45}" font-family="{SANS}" font-size="21" font-weight="700" fill="{TEXT}">{t}</text>'
            f'<text x="{x + 20}" y="{y0 + 92}" font-family="{MONO}" font-size="12" fill="{MUTED}">{a}</text>'
            f'<text x="{x + 20}" y="{y0 + 112}" font-family="{MONO}" font-size="12" fill="{MUTED}">{b}</text>'
        )
        if i < len(stages) - 1:
            p.append(
                f'<path d="M{x + bw + 5} {y0 + bh / 2} L{x + bw + gap - 5} {y0 + bh / 2}" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#arrow)"/>'
            )
    p.append("</svg>")
    return "\n".join(p)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="docs/media")
    p.add_argument("--steps", type=int, default=5, help="denoising steps shown in the banner")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="which test sequence to use")
    args = p.parse_args()
    seed_everything(args.seed)
    model, cfg, _ = load_model(args.checkpoint, "cpu")
    k, d, s = cfg.data.context_frames, cfg.data, cfg.sampler
    eps, _ = load_episodes(d.root, d.version, "test", d.games)
    seq = SequenceDataset(eps, k, 1, stride=7)
    b = prepare_batch(collate([seq[args.start]]), torch.device("cpu"))
    ctx = b["frames"][:, :k]
    bi, li = actions_for(b["buttons"], b["look"], k, k, d.action_history)
    sig_ctx = torch.zeros(1)

    def denoise(x, sigma):
        return model.denoise(x, sigma.expand(1), ctx, bi, li, b["game_id"], sig_ctx)

    sigmas = sigma_schedule(args.steps, s.sigma_min, s.sigma_max, s.rho)
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randn((1, 3, *ctx.shape[-2:]), generator=gen) * sigmas[0]
    trajectory = [img(x[0] / sigmas[0] * 0.6)]  # pure noise, scaled into view
    for i in range(args.steps):
        d_x = denoise(x, sigmas[i])
        x = x + (sigmas[i + 1] - sigmas[i]) * (x - d_x) / sigmas[i]
        trajectory.append(img(x[0]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "banner.svg").write_text(banner(trajectory, [float(v) for v in sigmas]))
    mini = [trajectory[0], trajectory[len(trajectory) // 2], trajectory[-1]]
    (out / "how-it-works.svg").write_text(
        how_it_works([img(f) for f in ctx[0]], mini, trajectory[-1], s.steps)
    )
    (out / "pipeline.svg").write_text(pipeline())
    print(f"wrote banner.svg, how-it-works.svg and pipeline.svg to {out}")


if __name__ == "__main__":
    main()
