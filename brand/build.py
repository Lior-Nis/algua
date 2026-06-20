#!/usr/bin/env python3
"""Generate the Algua brand kit (glyph, wordmark, lockups, favicon, tokens).

The glyph is hand-built geometry (the "waterline A"); the wordmark is the real
Space Grotesk Medium outline, baked to <path> so the SVGs are font-independent
and survive GitHub's SVG sanitizer. Re-runnable: regenerates the whole kit.

Source of truth for the design: docs/superpowers/specs/2026-06-20-algua-brand-identity-design.md

Usage:  uv run --with fonttools python brand/build.py
"""
from __future__ import annotations

import json
from pathlib import Path

from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

BRAND = Path(__file__).parent
FONT_SRC = BRAND / ".cache" / "SpaceGrotesk-var.ttf"

# ---------------------------------------------------------------------------
# §6 color tokens
# ---------------------------------------------------------------------------
TOK = {
    "ink": "#0C1618",
    "ink-submerged": "#16282B",
    "paper": "#F3F6F6",
    "aqua": "#13C2CE",
    "aqua-deep": "#0A8E99",
    "mute": "#5C6B6E",
    "paper-submerged": "#93A3A3",
}

# Submerged tint is context-aware: a deeper ink on light fields, a dimmed paper on
# dark fields (a dark tint would vanish into a dark background).
SUB_ON_LIGHT = TOK["ink-submerged"]
SUB_ON_DARK = TOK["paper-submerged"]

# ---------------------------------------------------------------------------
# §8 glyph geometry — 100-unit master ("the waterline A")
# ---------------------------------------------------------------------------
APEX = (50.0, 10.0)
LFOOT = (14.0, 90.0)
RFOOT = (86.0, 90.0)
STROKE = 15.0          # leg + waterline weight
WL_CY = 62.0           # waterline centre (≈ lower third → smaller submerged part)
WL_H = STROKE
WL_X0, WL_X1 = 15.0, 85.0   # waterline span (overshoots the legs at this height)


def glyph_body(fg: str, *, two_tone: str | None = None, mono_bg: str | None = None) -> str:
    """Inner SVG for the glyph on a 100x100 canvas.

    fg        — letterform colour (ink or paper).
    two_tone  — if set, the part below the waterline uses this colour.
    mono_bg   — if set, the waterline is a true knockout (transparent band).
    """
    caret = f"M {LFOOT[0]} {LFOOT[1]} L {APEX[0]} {APEX[1]} L {RFOOT[0]} {RFOOT[1]}"
    stroke_attrs = (
        f'fill="none" stroke-width="{STROKE}" '
        'stroke-linejoin="miter" stroke-miterlimit="20" stroke-linecap="butt"'
    )
    parts: list[str] = []

    if mono_bg is not None:
        # Mono: caret masked so the waterline band is cut out (knockout).
        parts.append(
            '<defs><mask id="wl-knockout">'
            '<rect x="0" y="0" width="100" height="100" fill="white"/>'
            f'<rect x="0" y="{WL_CY - WL_H / 2}" width="100" height="{WL_H}" fill="black"/>'
            "</mask></defs>"
        )
        parts.append(f'<path d="{caret}" stroke="{fg}" {stroke_attrs} mask="url(#wl-knockout)"/>')
        return "".join(parts)

    # Base caret in fg.
    parts.append(f'<path d="{caret}" stroke="{fg}" {stroke_attrs}/>')

    if two_tone is not None:
        # Re-draw the caret below the waterline centre in the submerged tint.
        parts.append(
            f'<clipPath id="below">'
            f'<rect x="0" y="{WL_CY}" width="100" height="{100 - WL_CY}"/></clipPath>'
        )
        parts.append(
            f'<path d="{caret}" stroke="{two_tone}" {stroke_attrs} clip-path="url(#below)"/>'
        )

    # Aqua waterline on top (the only accent).
    parts.append(
        f'<rect x="{WL_X0}" y="{WL_CY - WL_H / 2}" width="{WL_X1 - WL_X0}" '
        f'height="{WL_H}" fill="{TOK["aqua"]}"/>'
    )
    return "".join(parts)


def svg(width: float, height: float, body: str, *, bg: str | None = None,
        view: str | None = None) -> str:
    vb = view or f"0 0 {width} {height}"
    rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>' if bg else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="{vb}" role="img" aria-label="Algua">{rect}{body}</svg>\n'
    )


# ---------------------------------------------------------------------------
# Wordmark — real Space Grotesk Medium outlines, baked to <path>
# ---------------------------------------------------------------------------
def build_wordmark_paths(word: str = "Algua", tracking: float = 12.0):
    """Return (path_d, advance_width, cap_height) in font units (y-up)."""
    font = TTFont(FONT_SRC)
    instantiateVariableFont(font, {"wght": 500}, inplace=True)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]

    # Cap height from the 'A' glyph bbox.
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
        x += hmtx[gname][0] + tracking
    advance = x - tracking
    return "".join(d_parts), advance, cap_height


def wordmark_svg(fg: str, *, bg: str | None = None, target_cap: float = 64.0,
                 pad: float = 16.0):
    paths, advance, cap = build_wordmark_paths()
    s = target_cap / cap
    word_w = advance * s
    # Baseline waterline under the letters, overshooting the final glyph (§4).
    wl_h = STROKE * (target_cap / 84.0) * 0.55   # scaled, lighter than the glyph bar
    over = word_w * 0.04
    baseline_y = pad + target_cap
    total_w = word_w + 2 * pad
    total_h = baseline_y + pad + wl_h + 4
    # Flip y-up font space to y-down screen; place baseline.
    g = (
        f'<g transform="translate({pad:.1f},{baseline_y:.1f}) scale({s:.5f},{-s:.5f})" '
        f'fill="{fg}">{paths}</g>'
    )
    bar = (
        f'<rect x="{pad - over:.1f}" y="{baseline_y + 6:.1f}" '
        f'width="{word_w + 2 * over:.1f}" height="{wl_h:.1f}" fill="{TOK["aqua"]}"/>'
    )
    return svg(round(total_w, 1), round(total_h, 1), g + bar, bg=bg), total_w, total_h, word_w


# ---------------------------------------------------------------------------
# Lockups
# ---------------------------------------------------------------------------
def lockup_h(fg: str, *, bg: str | None = None, **glyph_kw):
    """Horizontal: glyph + wordmark, baselines aligned."""
    cap = 64.0
    pad = 18.0
    gsize = cap + 26          # glyph box edge
    paths, advance, fcap = build_wordmark_paths()
    s = cap / fcap
    word_w = advance * s
    gap = STROKE / 100 * gsize  # one glyph stroke-width
    # Vertical: align glyph feet (y=90/100 of its box) with text baseline.
    glyph_x = pad
    glyph_y_top = pad
    # text baseline sits at glyph foot line.
    foot_y = glyph_y_top + gsize * 0.90
    baseline_y = foot_y
    text_x = glyph_x + gsize + gap
    total_w = text_x + word_w + pad
    total_h = glyph_y_top + gsize + pad
    body = (
        f'<g transform="translate({glyph_x},{glyph_y_top})">'
        f'<g transform="scale({gsize / 100})">{glyph_body(fg, **glyph_kw)}</g></g>'
        f'<g transform="translate({text_x:.1f},{baseline_y:.1f}) scale({s:.5f},{-s:.5f})" '
        f'fill="{fg}">{paths}</g>'
    )
    # wordmark baseline waterline, lined up with glyph waterline height? keep simple: under text
    wl_h = STROKE * (cap / 84.0) * 0.55
    over = word_w * 0.04
    body += (
        f'<rect x="{text_x - over:.1f}" y="{baseline_y + 6:.1f}" '
        f'width="{word_w + 2 * over:.1f}" height="{wl_h:.1f}" fill="{TOK["aqua"]}"/>'
    )
    return svg(round(total_w, 1), round(total_h, 1), body, bg=bg)


def lockup_stacked(fg: str, *, bg: str | None = None, **glyph_kw):
    pad = 20.0
    gsize = 96.0
    cap = 46.0
    paths, advance, fcap = build_wordmark_paths()
    s = cap / fcap
    word_w = advance * s
    total_w = max(gsize, word_w) + 2 * pad
    glyph_x = (total_w - gsize) / 2
    gap = 22.0
    baseline_y = pad + gsize + gap + cap
    text_x = (total_w - word_w) / 2
    total_h = baseline_y + pad
    body = (
        f'<g transform="translate({glyph_x:.1f},{pad})">'
        f'<g transform="scale({gsize / 100})">{glyph_body(fg, **glyph_kw)}</g></g>'
        f'<g transform="translate({text_x:.1f},{baseline_y:.1f}) scale({s:.5f},{-s:.5f})" '
        f'fill="{fg}">{paths}</g>'
    )
    wl_h = STROKE * (cap / 84.0) * 0.55
    over = word_w * 0.04
    body += (
        f'<rect x="{text_x - over:.1f}" y="{baseline_y + 5:.1f}" '
        f'width="{word_w + 2 * over:.1f}" height="{wl_h:.1f}" fill="{TOK["aqua"]}"/>'
    )
    return svg(round(total_w, 1), round(total_h, 1), body, bg=bg)


def banner_dark():
    """README header: dark field, horizontal lockup centred with breathing room."""
    W, H = 1280.0, 360.0
    # Re-build lockup content scaled & centred.
    cap = 96.0
    gsize = cap + 40
    paths, advance, fcap = build_wordmark_paths()
    s = cap / fcap
    word_w = advance * s
    gap = STROKE / 100 * gsize
    block_w = gsize + gap + word_w
    x0 = (W - block_w) / 2
    gy = (H - gsize) / 2
    foot_y = gy + gsize * 0.90
    text_x = x0 + gsize + gap
    body = (
        f'<g transform="translate({x0:.1f},{gy:.1f})">'
        f'<g transform="scale({gsize / 100})">'
        f'{glyph_body(TOK["paper"], two_tone=SUB_ON_DARK)}</g></g>'
        f'<g transform="translate({text_x:.1f},{foot_y:.1f}) scale({s:.5f},{-s:.5f})" '
        f'fill="{TOK["paper"]}">{paths}</g>'
    )
    wl_h = STROKE * (cap / 84.0) * 0.55
    over = word_w * 0.04
    body += (
        f'<rect x="{text_x - over:.1f}" y="{foot_y + 9:.1f}" '
        f'width="{word_w + 2 * over:.1f}" height="{wl_h:.1f}" fill="{TOK["aqua"]}"/>'
    )
    return svg(W, H, body, bg=TOK["ink"])


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def main() -> None:
    out = {}

    # Glyph variants
    out["algua-glyph.svg"] = svg(100, 100, glyph_body(TOK["ink"]))
    out["algua-glyph-dark.svg"] = svg(100, 100, glyph_body(TOK["paper"]), bg=TOK["ink"])
    out["algua-glyph-twotone.svg"] = svg(
        100, 100, glyph_body(TOK["ink"], two_tone=SUB_ON_LIGHT))
    out["algua-glyph-twotone-dark.svg"] = svg(
        100, 100, glyph_body(TOK["paper"], two_tone=SUB_ON_DARK), bg=TOK["ink"])
    out["algua-glyph-mono.svg"] = svg(100, 100, glyph_body(TOK["ink"], mono_bg="x"))

    # Wordmark
    wm_light, *_ = wordmark_svg(TOK["ink"])
    out["algua-wordmark.svg"] = wm_light
    wm_dark, *_ = wordmark_svg(TOK["paper"], bg=TOK["ink"])
    out["algua-wordmark-dark.svg"] = wm_dark

    # Lockups
    out["algua-lockup-h.svg"] = lockup_h(TOK["ink"], two_tone=SUB_ON_LIGHT)
    out["algua-lockup-h-dark.svg"] = lockup_h(
        TOK["paper"], bg=TOK["ink"], two_tone=SUB_ON_DARK)
    out["algua-lockup-h-mono.svg"] = lockup_h(TOK["ink"], mono_bg="x")
    out["algua-lockup-stacked.svg"] = lockup_stacked(TOK["ink"], two_tone=SUB_ON_LIGHT)
    out["algua-lockup-stacked-dark.svg"] = lockup_stacked(
        TOK["paper"], bg=TOK["ink"], two_tone=SUB_ON_DARK)

    # Favicon (flat glyph, on paper so it reads on any tab bar)
    out["favicon.svg"] = svg(100, 100, glyph_body(TOK["ink"]), bg=TOK["paper"])
    out["banner-dark.svg"] = banner_dark()

    for name, content in out.items():
        (BRAND / name).write_text(content)
        print("wrote", name)

    (BRAND / "tokens.json").write_text(json.dumps(
        {"color": {k: {"value": v} for k, v in TOK.items()}}, indent=2) + "\n")
    print("wrote tokens.json")


if __name__ == "__main__":
    main()
