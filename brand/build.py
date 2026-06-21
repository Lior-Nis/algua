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
from pathlib import Path

from fontTools.pens.boundsPen import ControlBoundsPen
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

# Wordmark / glyph weights (Space Grotesk axis).
WORDMARK_WGHT = 300        # Light — slim, sleek
WORDMARK_TRACK = 12.0      # tracking in font units
GLYPH_WGHT = 400           # the A letterform — a touch of presence over the text

# Glyph placement in the 100-unit master.
APEX_Y = 8.0               # master-y of the cap top
BASE_Y = 92.0             # master-y of the baseline (glyph feet)
WL_BLEED = 0.4             # tiny vertical bleed so no black crossbar peeks


def _extract_A() -> dict:
    """The real 'A' outline + its crossbar band, measured in font units."""
    font = TTFont(FONT_SRC)
    instantiateVariableFont(font, {"wght": GLYPH_WGHT}, inplace=True)
    gs = font.getGlyphSet()
    gname = font.getBestCmap()[ord("A")]

    pen = SVGPathPen(gs)
    gs[gname].draw(pen)
    path_d = pen.getCommands()

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
        "path": path_d, "ymin": oymin, "ymax": oymax,
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


def glyph_size_for_cap(cap: float) -> float:
    """Return a 100-unit glyph box size whose apex-to-baseline height equals cap."""
    return cap * 100 / (BASE_Y - APEX_Y)


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
def build_wordmark_paths(word: str = "Algua"):
    """Return (path_d, advance_width, cap_height) in font units (y-up)."""
    font = TTFont(FONT_SRC)
    instantiateVariableFont(font, {"wght": WORDMARK_WGHT}, inplace=True)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]

    bp = ControlBoundsPen(glyph_set)
    glyph_set[cmap[ord("A")]].draw(bp)
    cap_height = bp.bounds[3]

    d_parts: list[str] = []
    x = 0.0
    for ch in word:
        gname = cmap[ord(ch)]
        pen = SVGPathPen(glyph_set)
        glyph_set[gname].draw(pen)
        cmds = pen.getCommands()
        if cmds:
            d_parts.append(f'<path d="{cmds}" transform="translate({x:.1f},0)"/>')
        x += hmtx[gname][0] + WORDMARK_TRACK
    advance = x - WORDMARK_TRACK
    return "".join(d_parts), advance, cap_height


def wordmark_svg(fg: str, *, bg: str | None = None, target_cap: float = 64.0,
                 pad: float = 16.0):
    paths, advance, cap = build_wordmark_paths()
    s = target_cap / cap
    word_w = advance * s
    baseline_y = pad + target_cap
    total_w = word_w + 2 * pad
    total_h = baseline_y + pad
    body = (
        f'<g transform="translate({pad:.1f},{baseline_y:.1f}) scale({s:.5f},{-s:.5f})" '
        f'fill="{fg}">{paths}</g>'
    )
    return svg(round(total_w, 1), round(total_h, 1), body, bg=bg), total_w, total_h, word_w


# ---------------------------------------------------------------------------
# Lockups
# ---------------------------------------------------------------------------
def lockup_h(fg: str, *, bg: str | None = None, **glyph_kw):
    """Horizontal: glyph + wordmark, glyph feet aligned to the text baseline."""
    cap = 64.0
    pad = 18.0
    gsize = glyph_size_for_cap(cap)
    paths, advance, fcap = build_wordmark_paths()
    s = cap / fcap
    word_w = advance * s
    gap = cap * 0.34
    glyph_x = pad
    glyph_y_top = pad
    baseline_y = glyph_y_top + gsize * (BASE_Y / 100)
    text_x = glyph_x + gsize + gap
    total_w = text_x + word_w + pad
    total_h = glyph_y_top + gsize + pad
    body = (
        f'<g transform="translate({glyph_x},{glyph_y_top})">'
        f'<g transform="scale({gsize / 100})">{glyph_body(fg, **glyph_kw)}</g></g>'
        f'<g transform="translate({text_x:.1f},{baseline_y:.1f}) scale({s:.5f},{-s:.5f})" '
        f'fill="{fg}">{paths}</g>'
    )
    return svg(round(total_w, 1), round(total_h, 1), body, bg=bg)


def lockup_stacked(fg: str, *, bg: str | None = None, **glyph_kw):
    pad = 20.0
    gsize = glyph_size_for_cap(72.0)
    cap = 46.0
    paths, advance, fcap = build_wordmark_paths()
    s = cap / fcap
    word_w = advance * s
    total_w = max(gsize, word_w) + 2 * pad
    glyph_x = (total_w - gsize) / 2
    gap = 24.0
    baseline_y = pad + gsize + gap + cap
    text_x = (total_w - word_w) / 2
    total_h = baseline_y + pad
    body = (
        f'<g transform="translate({glyph_x:.1f},{pad})">'
        f'<g transform="scale({gsize / 100})">{glyph_body(fg, **glyph_kw)}</g></g>'
        f'<g transform="translate({text_x:.1f},{baseline_y:.1f}) scale({s:.5f},{-s:.5f})" '
        f'fill="{fg}">{paths}</g>'
    )
    return svg(round(total_w, 1), round(total_h, 1), body, bg=bg)


def banner_dark():
    """README header: black field, horizontal lockup centred with breathing room."""
    W, H = 1280.0, 360.0
    cap = 96.0
    gsize = glyph_size_for_cap(cap)
    paths, advance, fcap = build_wordmark_paths()
    s = cap / fcap
    word_w = advance * s
    gap = cap * 0.34
    block_w = gsize + gap + word_w
    x0 = (W - block_w) / 2
    gy = (H - gsize) / 2
    foot_y = gy + gsize * (BASE_Y / 100)
    text_x = x0 + gsize + gap
    body = (
        f'<g transform="translate({x0:.1f},{gy:.1f})">'
        f'<g transform="scale({gsize / 100})">{glyph_body(TOK["paper"])}</g></g>'
        f'<g transform="translate({text_x:.1f},{foot_y:.1f}) scale({s:.5f},{-s:.5f})" '
        f'fill="{TOK["paper"]}">{paths}</g>'
    )
    return svg(W, H, body, bg=TOK["ink"])


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def main() -> None:
    out = {
        "algua-glyph.svg": svg(100, 100, glyph_body(TOK["ink"])),
        "algua-glyph-dark.svg": svg(100, 100, glyph_body(TOK["paper"]), bg=TOK["ink"]),
        "algua-glyph-mono.svg": svg(100, 100, glyph_body(TOK["ink"], mono_bg="x")),
        "algua-wordmark.svg": wordmark_svg(TOK["ink"])[0],
        "algua-wordmark-dark.svg": wordmark_svg(TOK["paper"], bg=TOK["ink"])[0],
        "algua-lockup-h.svg": lockup_h(TOK["ink"]),
        "algua-lockup-h-dark.svg": lockup_h(TOK["paper"], bg=TOK["ink"]),
        "algua-lockup-h-mono.svg": lockup_h(TOK["ink"], mono_bg="x"),
        "algua-lockup-stacked.svg": lockup_stacked(TOK["ink"]),
        "algua-lockup-stacked-dark.svg": lockup_stacked(TOK["paper"], bg=TOK["ink"]),
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
