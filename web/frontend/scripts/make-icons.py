#!/usr/bin/env python3
"""Generate the PWA icon set into web/frontend/public/.

The icons are rasterized FROM the brand master (docs/brand/favicon.svg) rather
than redrawn, so they can never drift from the approved Sea mark. The core
identity is fill-only straight-line geometry, so a tiny polygon parser is all
that is needed — anything curved in the master is a deliberate change and this
script refuses it loudly instead of rendering an approximation.

Run from web/frontend/:  python3 scripts/make-icons.py
Requires Pillow.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "public"
MASTER = HERE.parents[2] / "docs" / "brand" / "favicon.svg"

VIEWBOX = 512.0  # the brand master's grid
SS = 4  # supersample factor; downsampled for clean anti-aliased edges

_RECT_RE = re.compile(r"<rect[^>]*\brx=\"([\d.]+)\"[^>]*\bfill=\"(#[0-9A-Fa-f]{6})\"")
_PATH_RE = re.compile(r"<path[^>]*\bd=\"([^\"]+)\"[^>]*\bfill=\"(#[0-9A-Fa-f]{6})\"")
_TOKEN_RE = re.compile(r"([MLHVZmlhvz])|(-?[\d.]+)")


def parse_polygons(d: str) -> list[list[tuple[float, float]]]:
    """Split one path's `d` into closed polygons (absolute M/L/H/V/Z only)."""
    polys: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    cmd = ""
    nums: list[float] = []
    x = y = 0.0

    def flush() -> None:
        nonlocal cur
        if len(cur) >= 3:
            polys.append(cur)
        cur = []

    def emit() -> None:
        """Consume the pending operand run for `cmd`."""
        nonlocal x, y, nums
        if cmd in ("M", "L"):
            if len(nums) % 2:
                raise ValueError(f"odd coordinate count for {cmd!r} in {d!r}")
            for i in range(0, len(nums), 2):
                x, y = nums[i], nums[i + 1]
                cur.append((x, y))
        elif cmd == "H":
            for v in nums:
                x = v
                cur.append((x, y))
        elif cmd == "V":
            for v in nums:
                y = v
                cur.append((x, y))
        nums = []

    for m in _TOKEN_RE.finditer(d):
        letter, number = m.group(1), m.group(2)
        if number is not None:
            nums.append(float(number))
            continue
        emit()
        if letter.islower():
            raise ValueError(f"relative command {letter!r} unsupported; master changed")
        if letter == "Z":
            flush()
            cmd = ""
        else:
            if letter == "M":
                flush()
            cmd = letter
    emit()
    flush()
    if any(c in d for c in "CcSsQqTtAa"):
        raise ValueError("master contains curves — the icon renderer cannot rasterize it")
    return polys


def load_master() -> tuple[str, float, list[tuple[list[list[tuple[float, float]]], str]]]:
    svg = MASTER.read_text()
    rect = _RECT_RE.search(svg)
    if rect is None:
        raise ValueError(f"no tile rect in {MASTER}")
    radius_frac = float(rect.group(1)) / VIEWBOX
    shapes = [(parse_polygons(d), fill) for d, fill in _PATH_RE.findall(svg)]
    if not shapes:
        raise ValueError(f"no paths in {MASTER}")
    return rect.group(2), radius_frac, shapes


TILE, RADIUS_FRAC, SHAPES = load_master()


def tile(size: int, *, mark_frac: float, rounded: bool) -> Image.Image:
    """Render the mark centred on the brand tile at `mark_frac` of the edge."""
    big = size * SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = int(big * RADIUS_FRAC) if rounded else 0
    draw.rounded_rectangle([0, 0, big - 1, big - 1], radius=radius, fill=TILE)
    # The master already sits inside its own margins; mark_frac shrinks it
    # further for the maskable safe zone.
    scale = big / VIEWBOX * mark_frac
    offset = big * (1 - mark_frac) / 2
    for polys, fill in SHAPES:
        for poly in polys:
            draw.polygon([(px * scale + offset, py * scale + offset) for px, py in poly], fill=fill)
    return img.resize((size, size), Image.LANCZOS)


def save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    img.save(path, "PNG")
    print(f"wrote {path} {img.size}")


def main() -> None:
    # Standard icons: the brand tile with its own rounded corners.
    save(tile(192, mark_frac=1.0, rounded=True), "icon-192.png")
    save(tile(512, mark_frac=1.0, rounded=True), "icon-512.png")
    # Maskable: full-bleed square, mark shrunk into the ~80% safe zone.
    save(tile(512, mark_frac=0.8, rounded=False), "icon-maskable-512.png")
    # Apple touch icon: 180x180, square (iOS applies its own mask and blackens alpha).
    save(tile(180, mark_frac=1.0, rounded=False), "apple-touch-icon.png")


if __name__ == "__main__":
    main()
