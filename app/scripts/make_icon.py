"""Draw the LUCID app mark: a circle, smooth on the left, pixelated on the right.

The left half is the real frame and the right half is the dream of it.
Writes src-tauri/icons/source.png (1024 px, for `npx tauri icon`) and
src/assets/mark.svg (for the interface).

    ../.venv/bin/python scripts/make_icon.py && npx tauri icon src-tauri/icons/source.png
"""

from pathlib import Path

from PIL import Image, ImageDraw

APP = Path(__file__).resolve().parent.parent
BG = (23, 26, 32, 255)
ACCENT = (255, 107, 53, 255)
SIZE = 1024
SS = 4  # supersampling for smooth edges


def pixel_cells(cx: float, cy: float, r: float, cell: float) -> list[tuple[float, float]]:
    """Top-left corners of grid cells on the right half whose centres fall inside the circle."""
    cells = []
    n = int(r // cell) + 1
    for j in range(-n, n):
        for i in range(0, n):
            x0, y0 = cx + i * cell, cy + j * cell
            mx, my = x0 + cell / 2, y0 + cell / 2
            if (mx - cx) ** 2 + (my - cy) ** 2 <= r**2:
                cells.append((x0, y0))
    return cells


def draw_png() -> None:
    s = SIZE * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # macOS-style tile: 824 px body inside the 1024 canvas.
    pad = 100 * SS
    d.rounded_rectangle([pad, pad, s - pad, s - pad], radius=185 * SS, fill=BG)
    cx = cy = s / 2
    r = 270 * SS
    cell = r / 4
    # Smooth left half.
    d.pieslice([cx - r, cy - r, cx + r, cy + r], 90, 270, fill=ACCENT)
    # Pixel right half, with a small gap so the split reads at small sizes.
    gap = 18 * SS
    for x0, y0 in pixel_cells(cx, cy, r, cell):
        d.rectangle([x0 + gap, y0, x0 + gap + cell - 1, y0 + cell - 1], fill=ACCENT)
    out = APP / "src-tauri" / "icons" / "source.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize((SIZE, SIZE), Image.LANCZOS).save(out)
    print(f"wrote {out}")


def draw_svg() -> None:
    cx = cy = 16.0
    r = 12.0
    cell = r / 4
    gap = 0.9
    rects = "".join(
        f'<rect x="{x0 + gap:g}" y="{y0:g}" width="{cell:g}" height="{cell:g}"/>'
        for x0, y0 in pixel_cells(cx, cy, r, cell)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="currentColor" aria-hidden="true">'
        f'<path d="M{cx:g} {cy - r:g}A{r:g} {r:g} 0 0 0 {cx:g} {cy + r:g}Z"/>{rects}</svg>\n'
    )
    out = APP / "src" / "assets" / "mark.svg"
    out.write_text(svg)
    print(f"wrote {out}")


if __name__ == "__main__":
    draw_png()
    draw_svg()
