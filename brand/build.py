#!/usr/bin/env python3
"""Generate the Algua brand kit (glyph, wordmark, lockups, favicon, tokens).

The glyph is the real Space Grotesk capital "A" letterform with its crossbar
recolored to the aqua "waterline"; the wordmark is the same typeface outlined.
Both are baked to <path> so the SVGs are font-independent and survive GitHub's
SVG sanitizer. Re-runnable: regenerates the whole kit.

Source of truth for the design: docs/superpowers/specs/2026-06-20-algua-brand-identity-design.md

Usage:  uv run --with fonttools python brand/build.py
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

BRAND = Path(__file__).parent
FONT_SRC = BRAND / ".cache" / "Outfit.ttf"

# ---------------------------------------------------------------------------
# §6 color tokens
# ---------------------------------------------------------------------------
TOK = {
    "ink": "#000000",          # true black
    "paper": "#F3F6F6",
    "aqua": "#13C2CE",         # the water blue — the only accent
    "aqua-deep": "#0A8E99",
    "mute": "#5C6B6E",
}

# Wordmark / glyph weights (Outfit axis, 100=thin .. 900). The glyph IS the
# wordmark's capital "A", so they share one weight — the icon is literally the A.
WORDMARK_WGHT = 160        # thin/hairline — slim and sleek
WORDMARK_TRACK = 16.0      # tracking in font units (airier to match the thin weight)
GLYPH_WGHT = WORDMARK_WGHT

# Glyph placement in the 100-unit master.
APEX_Y = 8.0               # master-y of the cap top
BASE_Y = 92.0             # master-y of the baseline (glyph feet)
WL_BLEED = 0.4             # vertical bleed (master units) so no black crossbar peeks
WL_BLEED_FONT = 4.0        # same, in font units (for the in-wordmark A)

# The aqua waterline is the OCEAN: a long, thin, tapered (trapezoid) horizontal
# line that runs well past the A's legs, so the sharp apex reads as a fin breaking
# the surface. All in font units.
WL_EXTEND = 150.0          # how far past each leg the water reaches (long)
WL_EXTEND_R = 120.0        # right side in the wordmark, capped to clear the 'l'
WL_THICK = 34.0            # the line's thickness (thin)
WL_TAPER = 60.0            # trapezoid: each end narrows this much from bottom to top

# Apex sharpening: Space Grotesk's A has a flat top. We collapse that flat to a
# true point; APEX_SHARPEN is how far (font units) the point rises above the cap
# line — larger = a taller, sharper peak. 0 = a clean point exactly at cap height.
APEX_SHARPEN = 70.0


def _extract_A() -> dict:
    """The real 'A' outline + its crossbar band, measured in font units."""
    font = TTFont(FONT_SRC)
    instantiateVariableFont(font, {"wght": GLYPH_WGHT}, inplace=True)
    gs = font.getGlyphSet()
    gname = font.getBestCmap()[ord("A")]
    adv = font["hmtx"][gname][0]

    rp = RecordingPen()
    gs[gname].draw(rp)
    contours: list[list] = []
    cur: list = []
    for cmd, args in rp.value:
        if cmd == "moveTo":
            if cur:
                contours.append(cur)
            cur = [args[0]]
        elif cmd == "lineTo":
            cur.append(args[0])
        elif cmd in ("qCurveTo", "curveTo"):
            cur += [p for p in args if p]
        elif cmd == "closePath" and cur:
            contours.append(cur)
            cur = []
    if cur:
        contours.append(cur)

    def bbox(c: list) -> tuple[float, float, float, float]:
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        return min(xs), min(ys), max(xs), max(ys)

    def area(c: list) -> float:
        x0, y0, x1, y1 = bbox(c)
        return (x1 - x0) * (y1 - y0)

    outer = max(contours, key=area)
    inner = min(contours, key=area)           # the counter (triangular hole)
    cap0 = bbox(outer)[3]                      # original cap height (flat-top y)

    def sharpen_apex(c: list, rise: float) -> list:
        """Collapse the flat top (two topmost points) into one point, raised."""
        top_y = max(p[1] for p in c)
        tops = [i for i, p in enumerate(c) if abs(p[1] - top_y) < 1.0]
        peak = (sum(c[i][0] for i in tops) / len(tops), top_y + rise)
        out: list = []
        placed = False
        for i, p in enumerate(c):
            if i in tops:
                if not placed:
                    out.append(peak)
                    placed = True
            else:
                out.append(p)
        return out

    # Outer apex rises by APEX_SHARPEN; the inner counter tip just closes to a
    # point (keeps the solid peak above it, which is what reads as "sharp").
    outer = sharpen_apex(outer, APEX_SHARPEN)
    inner = sharpen_apex(inner, 0.0)

    def to_path(c: list) -> str:
        head = f'M {c[0][0]:.1f} {c[0][1]:.1f}'
        rest = "".join(f' L {x:.1f} {y:.1f}' for x, y in c[1:])
        return head + rest + " Z"

    path_d = to_path(outer) + to_path(inner)
    oxmin, oymin, oxmax, oymax = bbox(outer)
    # `inner` is the crossbar bar; its bbox is the crossbar band.
    _, cb_lo, _, cb_hi = bbox(inner)
    cy = (cb_lo + cb_hi) / 2
    legs = _crossings(outer, cy)
    return {
        "path": path_d, "outer": outer, "adv": adv,
        "ymin": oymin, "ymax": oymax, "cap0": cap0, "cx": (oxmin + oxmax) / 2,
        "cb_lo": cb_lo, "cb_hi": cb_hi, "cy": cy,
        "leg_l": legs[0], "leg_r": legs[-1],
        "apex": ((oxmin + oxmax) / 2, oymax),     # the sharpened peak (fin tip)
    }


def _crossings(poly: list, y: float) -> list[float]:
    """x-coords where the polygon edges cross horizontal line y, sorted."""
    xs = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 <= y < y2) or (y2 <= y < y1):
            xs.append(x1 + (y - y1) / (y2 - y1) * (x2 - x1))
    return sorted(xs)


_GA = _extract_A()


def _a_transform() -> tuple[float, float, float]:
    """(scale, tx, ty) mapping the font 'A' into the 100-unit master (y-down)."""
    s = (BASE_Y - APEX_Y) / (_GA["ymax"] - _GA["ymin"])
    ty = BASE_Y + s * _GA["ymin"]
    tx = 50.0 - s * _GA["cx"]
    return s, tx, ty


def _waterline_rect(fill: str, ext_l: float = WL_EXTEND, ext_r: float = WL_EXTEND) -> str:
    """The aqua ocean line — a long, thin, tapered trapezoid — in font-unit (y-up)
    coords; place inside a scaled, y-flipped group. The bottom edge is the longer
    one; each end angles in by WL_TAPER toward the top, for a horizon-in-perspective
    feel. Runs ext_l / ext_r past the left / right legs (longer than the A)."""
    cy = _GA["cy"]
    yb, yt = cy - WL_THICK / 2, cy + WL_THICK / 2
    lb0, lb1 = _GA["leg_l"] - ext_l, _GA["leg_r"] + ext_r            # bottom (long)
    lt0, lt1 = lb0 + WL_TAPER, lb1 - WL_TAPER                        # top (shorter)
    pts = f"{lb0:.1f},{yb:.1f} {lb1:.1f},{yb:.1f} {lt1:.1f},{yt:.1f} {lt0:.1f},{yt:.1f}"
    return f'<polygon points="{pts}" fill="{fill}"/>'


def _fin(fill: str) -> str:
    """The solid 'fin': the A's silhouette above the waterline, filled (so the
    counter shows no white). Drawn under the waterline so the taper can't expose
    background. Font-unit (y-up) coords for the scaled, y-flipped group."""
    yb = _GA["cy"] - WL_THICK / 2
    edges = _crossings(_GA["outer"], yb)        # outer leg x's at the band bottom
    xl, xr = edges[0], edges[-1]
    ax, ay = _GA["apex"]
    return (f'<polygon points="{xl:.1f},{yb:.1f} {xr:.1f},{yb:.1f} '
            f'{ax:.1f},{ay:.1f}" fill="{fill}"/>')


def glyph_body(fg: str, *, mono_bg: str | None = None) -> str:
    """The 'A' letterform (a shark fin) over the long aqua waterline (the ocean).

    The waterline runs past both legs, so it reads as the sea surface and the
    sharp apex as a fin breaking it.
    """
    s, tx, ty = _a_transform()
    tfm = f'transform="translate({tx:.3f},{ty:.3f}) scale({s:.4f},{-s:.4f})"'
    if mono_bg is not None:
        # Mono: solid fin above the waterline, with a knockout band for the line.
        cy = _GA["cy"]
        band_y = ty - s * (cy + WL_THICK / 2)
        band_h = s * WL_THICK
        return (
            '<defs><mask id="wl-knockout">'
            '<rect x="0" y="0" width="100" height="100" fill="white"/>'
            f'<rect x="0" y="{band_y:.2f}" width="100" height="{band_h:.2f}" '
            'fill="black"/></mask></defs>'
            f'<g mask="url(#wl-knockout)"><g {tfm}>'
            f'<path d="{_GA["path"]}" fill="{fg}"/>{_fin(fg)}</g></g>'
        )
    # A (legs) → solid fin above the surface → the aqua waterline on top.
    return (
        f'<g {tfm}><path d="{_GA["path"]}" fill="{fg}"/>'
        f'{_fin(fg)}{_waterline_rect(TOK["aqua"])}</g>'
    )


def svg(width: float, height: float, body: str, *, bg: str | None = None,
        view: str | None = None) -> str:
    vb = view or f"0 0 {width} {height}"
    rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>' if bg else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="{vb}" role="img" aria-label="Algua">{rect}{body}</svg>\n'
    )


# ---------------------------------------------------------------------------
# Wordmark — real Space Grotesk outlines, baked to <path>
# ---------------------------------------------------------------------------
@cache
def build_wordmark_paths(word: str = "Algua"):
    """Return (path_d, advance_width, cap_height, descent) in font units (y-up)."""
    font = TTFont(FONT_SRC)
    instantiateVariableFont(font, {"wght": WORDMARK_WGHT}, inplace=True)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]

    # Cap line is the original (flat-top) A height, so other letters align and the
    # sharpened apex overshoots it the way a pointed cap optically should.
    cap_height = _GA["cap0"]

    d_parts: list[str] = []
    x = 0.0
    descent = 0.0
    for ch in word:
        gname = cmap[ord(ch)]
        if ch == "A":
            cmds = _GA["path"]                 # the sharpened A
        else:
            pen = SVGPathPen(glyph_set)
            glyph_set[gname].draw(pen)
            cmds = pen.getCommands()
            bp = ControlBoundsPen(glyph_set)
            glyph_set[gname].draw(bp)
            if bp.bounds:
                descent = max(descent, -bp.bounds[1])   # depth below baseline
        if cmds:
            d_parts.append(f'<path d="{cmds}" transform="translate({x:.1f},0)"/>')
        x += hmtx[gname][0] + WORDMARK_TRACK
    advance = x - WORDMARK_TRACK
    return "".join(d_parts), advance, cap_height, descent


def _wordmark_group(fg: str, tx: float, ty: float, s: float) -> str:
    """The full 'Algua' wordmark, with the aqua ocean line on its leading A.

    The waterline is drawn at the leading A's crossbar height and runs past its
    legs (the ocean), so the A's sharp apex reads as a fin above the surface.
    """
    paths, _, _, _ = build_wordmark_paths()
    return (
        f'<g transform="translate({tx:.2f},{ty:.2f}) scale({s:.5f},{-s:.5f})" fill="{fg}">'
        f'{paths}{_fin(fg)}{_waterline_rect(TOK["aqua"], ext_r=WL_EXTEND_R)}'
        f'</g>'
    )


def wordmark_svg(fg: str, *, bg: str | None = None, target_cap: float = 64.0,
                 pad: float = 16.0):
    _, advance, cap, descent = build_wordmark_paths()
    s = target_cap / cap
    word_w = advance * s
    baseline_y = pad + target_cap
    total_w = word_w + 2 * pad
    # Canvas must clear the descender (e.g. 'g') below the baseline, plus pad.
    total_h = baseline_y + descent * s + pad
    body = _wordmark_group(fg, pad, baseline_y, s)
    return svg(round(total_w, 1), round(total_h, 1), body, bg=bg), total_w, total_h, word_w


def banner_dark():
    """README header: black field, the 'Algua' wordmark centred."""
    W, H = 1280.0, 360.0
    cap = 124.0
    _, advance, fcap, descent = build_wordmark_paths()
    s = cap / fcap
    word_w = advance * s
    tx = (W - word_w) / 2
    # Centre the full cap-to-descender ink block vertically.
    ty = (H + cap) / 2 - descent * s / 2
    return svg(W, H, _wordmark_group(TOK["paper"], tx, ty, s), bg=TOK["ink"])


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def main() -> None:
    out = {
        # The logo = the wordmark, with the waterline on its own A.
        "algua-wordmark.svg": wordmark_svg(TOK["ink"])[0],
        "algua-wordmark-dark.svg": wordmark_svg(TOK["paper"], bg=TOK["ink"])[0],
        # The icon = that same A, extracted (favicon / avatar).
        "algua-glyph.svg": svg(100, 100, glyph_body(TOK["ink"])),
        "algua-glyph-dark.svg": svg(100, 100, glyph_body(TOK["paper"]), bg=TOK["ink"]),
        "algua-glyph-mono.svg": svg(100, 100, glyph_body(TOK["ink"], mono_bg="x")),
        "favicon.svg": svg(100, 100, glyph_body(TOK["ink"]), bg=TOK["paper"]),
        "banner-dark.svg": banner_dark(),
    }
    for name, content in out.items():
        (BRAND / name).write_text(content)
        print("wrote", name)

    (BRAND / "tokens.json").write_text(json.dumps(
        {"color": {k: {"value": v} for k, v in TOK.items()}}, indent=2) + "\n")
    print("wrote tokens.json")


if __name__ == "__main__":
    main()
