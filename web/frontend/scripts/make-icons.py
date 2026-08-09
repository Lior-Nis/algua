#!/usr/bin/env python3
"""Generate the PWA icon set into web/frontend/public/.

Solid --ink (#07090d) rounded tile with a gold (#e8b339) lowercase 'a' glyph.
Run from web/frontend/:  python3 scripts/make-icons.py
Requires Pillow and DejaVu Sans Mono Bold (stock on Debian/Ubuntu).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INK = (7, 9, 13, 255)  # --ink #07090d
GOLD = (232, 179, 57, 255)  # --gold #e8b339
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
OUT = Path(__file__).resolve().parent.parent / "public"

# Render at 4x and downsample for clean anti-aliased edges.
SS = 4


def tile(size: int, *, glyph_frac: float, radius_frac: float, opaque: bool) -> Image.Image:
    big = size * SS
    img = Image.new("RGBA", (big, big), INK if opaque else (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if not opaque:
        draw.rounded_rectangle([0, 0, big - 1, big - 1], radius=int(big * radius_frac), fill=INK)
    font = ImageFont.truetype(FONT, int(big * glyph_frac))
    draw.text((big / 2, big / 2), "a", font=font, fill=GOLD, anchor="mm")
    return img.resize((size, size), Image.LANCZOS)


def save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    img.save(path, "PNG")
    print(f"wrote {path} {img.size}")


def main() -> None:
    # Standard icons: rounded ink tile, transparent corners.
    save(tile(192, glyph_frac=0.62, radius_frac=0.22, opaque=False), "icon-192.png")
    save(tile(512, glyph_frac=0.62, radius_frac=0.22, opaque=False), "icon-512.png")
    # Maskable: full-bleed opaque square, glyph shrunk into the ~80% safe zone.
    save(tile(512, glyph_frac=0.62 * 0.8, radius_frac=0.0, opaque=True), "icon-maskable-512.png")
    # Apple touch icon: 180x180, NO transparency (iOS blackens alpha).
    save(tile(180, glyph_frac=0.62, radius_frac=0.0, opaque=True), "apple-touch-icon.png")


if __name__ == "__main__":
    main()
