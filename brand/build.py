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

from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

BRAND = Path(__file__).parent
FONT_SRC = BRAND / ".cache" / "SpaceGrotesk-var.ttf"

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

# Wordmark / glyph weights (Space Grotesk axis). The glyph IS the wordmark's
# capital "A", so they share one weight — the icon is literally the A of Algua.
WORDMARK_WGHT = 300        # Light — slim, sleek
WORDMARK_TRACK = 12.0      # tracking in font units
GLYPH_WGHT = WORDMARK_WGHT

# Glyph placement in the 100-unit master.
APEX_Y = 8.0               # master-y of the cap top
BASE_Y = 92.0             # master-y of the baseline (glyph feet)
WL_BLEED = 0.4             # vertical bleed (master units) so no black crossbar peeks
WL_BLEED_FONT = 4.0        # same, in font units (for the in-wordmark A)

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
    cb_top = bbox(inner)[1]                    # counter bottom = crossbar top edge

    def crossings(poly: list, y: float) -> list[float]:
        xs = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 <= y < y2) or (y2 <= y < y1):
                xs.append(x1 + (y - y1) / (y2 - y1) * (x2 - x1))
        return sorted(xs)

    # Walk down from the crossbar top until the legs split (>2 crossings).
    cb_bot = cb_top
    y = cb_top
    while y > oymin:
        if len(crossings(outer, y)) > 2:
            cb_bot = y
            break
        y -= 2
    legs = crossings(outer, (cb_top + cb_bot) / 2)
    return {
        "path": path_d, "adv": adv, "ymin": oymin, "ymax": oymax, "cap0": cap0,
        "cx": (oxmin + oxmax) / 2, "cb_top": cb_top, "cb_bot": cb_bot,
        "leg_l": legs[0], "leg_r": legs[-1],
    }


_GA = _extract_A()


def _a_transform() -> tuple[float, float, float]:
    """(scale, tx, ty) mapping the font 'A' into the 100-unit master (y-down)."""
    s = (BASE_Y - APEX_Y) / (_GA["ymax"] - _GA["ymin"])
    ty = BASE_Y + s * _GA["ymin"]
    tx = 50.0 - s * _GA["cx"]
    return s, tx, ty


def glyph_body(fg: str, *, mono_bg: str | None = None) -> str:
    """The 'A' letterform with its own crossbar recolored to the aqua waterline.

    The aqua is a band-clipped copy of the letterform — so it inherits the A's
    slanted leg edges instead of being a rectangle laid on top.
    """
    s, tx, ty = _a_transform()
    tfm = f'transform="translate({tx:.3f},{ty:.3f}) scale({s:.4f},{-s:.4f})"'
    band_y = ty - s * _GA["cb_top"] - WL_BLEED
    band_h = s * (_GA["cb_top"] - _GA["cb_bot"]) + 2 * WL_BLEED
    if mono_bg is not None:
        # Mono: knock the waterline band out of the letterform (transparent).
        return (
            '<defs><mask id="wl-knockout">'
            '<rect x="0" y="0" width="100" height="100" fill="white"/>'
            f'<rect x="0" y="{band_y:.2f}" width="100" height="{band_h:.2f}" '
            'fill="black"/></mask></defs>'
            f'<g mask="url(#wl-knockout)"><g {tfm}>'
            f'<path d="{_GA["path"]}" fill="{fg}"/></g></g>'
        )
    return (
        f'<defs><clipPath id="wl"><rect x="0" y="{band_y:.2f}" width="100" '
        f'height="{band_h:.2f}"/></clipPath></defs>'
        f'<g {tfm}><path d="{_GA["path"]}" fill="{fg}"/></g>'
        f'<g clip-path="url(#wl)"><g {tfm}>'
        f'<path d="{_GA["path"]}" fill="{TOK["aqua"]}"/></g></g>'
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
    """Return (path_d, advance_width, cap_height) in font units (y-up)."""
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
    for ch in word:
        gname = cmap[ord(ch)]
        if ch == "A":
            cmds = _GA["path"]                 # the sharpened A
        else:
            pen = SVGPathPen(glyph_set)
            glyph_set[gname].draw(pen)
            cmds = pen.getCommands()
        if cmds:
            d_parts.append(f'<path d="{cmds}" transform="translate({x:.1f},0)"/>')
        x += hmtx[gname][0] + WORDMARK_TRACK
    advance = x - WORDMARK_TRACK
    return "".join(d_parts), advance, cap_height


def _wordmark_group(fg: str, tx: float, ty: float, s: float) -> str:
    """The full 'Algua' wordmark, with the aqua waterline on its own capital A.

    The accent is a band-clipped copy of the leading 'A' glyph (drawn at x=0 in
    font units, same as the wordmark's first letter), so the aqua is the A's own
    crossbar — no duplicated glyph, no rectangle on top.
    """
    paths, _, _ = build_wordmark_paths()
    band_y = _GA["cb_bot"] - WL_BLEED_FONT
    band_h = (_GA["cb_top"] - _GA["cb_bot"]) + 2 * WL_BLEED_FONT
    clip = (f'<clipPath id="wlw"><rect x="0" y="{band_y:.1f}" '
            f'width="{_GA["adv"]:.1f}" height="{band_h:.1f}"/></clipPath>')
    return (
        f'<defs>{clip}</defs>'
        f'<g transform="translate({tx:.2f},{ty:.2f}) scale({s:.5f},{-s:.5f})" fill="{fg}">'
        f'{paths}'
        f'<g clip-path="url(#wlw)"><path d="{_GA["path"]}" fill="{TOK["aqua"]}"/></g>'
        f'</g>'
    )


def wordmark_svg(fg: str, *, bg: str | None = None, target_cap: float = 64.0,
                 pad: float = 16.0):
    _, advance, cap = build_wordmark_paths()
    s = target_cap / cap
    word_w = advance * s
    baseline_y = pad + target_cap
    total_w = word_w + 2 * pad
    total_h = baseline_y + pad
    body = _wordmark_group(fg, pad, baseline_y, s)
    return svg(round(total_w, 1), round(total_h, 1), body, bg=bg), total_w, total_h, word_w


def banner_dark():
    """README header: black field, the 'Algua' wordmark centred."""
    W, H = 1280.0, 360.0
    cap = 124.0
    _, advance, fcap = build_wordmark_paths()
    s = cap / fcap
    word_w = advance * s
    tx = (W - word_w) / 2
    ty = H / 2 + cap * 0.42        # baseline; descenders sit just below centre
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
