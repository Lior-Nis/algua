# Algua — Brand Identity Design Spec

**Date:** 2026-06-20
**Status:** Approved design (pre-implementation)
**Scope:** Full brand identity — primary mark (wordmark + glyph), color system,
typography, construction geometry, usage rules, and the asset deliverable set.
**Out of scope:** A website/docs-site theme, marketing collateral, animation.

---

## 1. Brand context

Algua is an **agent-first algotrading research and lifecycle platform**. It is
CLI-driven (no GUI), disciplined (every promotion is gate-token-enforced), and
deliberately un-hyped — serious engineering, not fintech marketing. The identity
must read that way: restrained, technical, terminal-adjacent, and legible at the
sizes a developer tool actually appears (favicon, GitHub avatar, README header,
terminal banners).

The name **Algua** carries *aqua*. The identity leans on that echo — but quietly.

---

## 2. Concept: the waterline

The entire identity rests on **one device: a single horizontal aqua "waterline."**
We do not draw literal waves — that fights the chosen aesthetic (minimal & sharp).
Instead, one flat horizontal stroke does all the work, reading three ways at once:

- **water** — the *aqua* in Algua;
- a **price / equity level** — a market line;
- a **threshold / gate** — the lifecycle's disciplined checkpoints
  (`idea → backtested → … → live`).

One line, three meanings, zero decoration. **The waterline is the only place the
aqua accent ever appears.** Everything else is monochrome ink. This makes the
"single accent, used sparingly" rule literal and self-enforcing.

---

## 3. The glyph — "the waterline A"

A geometric capital **A**, built from straight, **slim** sharp strokes, with **no
crossbar of its own**. The aqua waterline *is* the crossbar. Slim strokes keep the
mark sleek; the waterline carries the only weight that needs to register.

```
   /\          apex: a sharp point (never rounded)
  /  \
 /====\   <-  aqua waterline = crossbar; overshoots both legs
/      \
```

Requirements:

- **Crossbar-less A.** The A's own structure is the two legs + apex only. The
  horizontal member is the aqua waterline, a separate flat stroke slotted between
  the legs.
- **Overshoot.** The waterline extends slightly **past both legs** (the tide
  reaching beyond the form) — a modest, symmetric overshoot, reused at every scale.
- **Sharp apex.** The top is a true point. No rounding, no ink trap softening.
- **Two colors only.** The mark is **black + the aqua waterline** — nothing else.
  (An earlier two-tone "submerged" lower-leg tint was tried and dropped; it added
  a third tone and muddied the black-and-blue clarity.)

Legibility intent: at 16px the glyph reads as a crisp black **A** bisected by one
bright aqua line; the overshoot keeps the waterline distinguishable even when the
letterform itself blurs. Slim strokes thin out at 16px but the A + bar still read.

---

## 4. Wordmark

The wordmark is `Algua` — **capital A, lowercase `lgua`** (title case). The
capitalized initial gives the mark a name-like, proper-noun presence while the
lowercase tail keeps it quiet and liquid, not all-caps loud.

- **Skeleton:** drawn from a **geometric grotesk (Space Grotesk Light, ~300
  weight)** for a slim, sleek line, then **converted to outlines** in the final
  asset so the wordmark is font-independent (no dependency on the font being
  installed/licensed at use).
- **Letterforms:** airy tracking to match the light weight; flat terminals; Space
  Grotesk's native geometric lowercase (two-story `a`) is kept — it already reads
  sharp and geometric, and altering it would abandon the stated skeleton.
- **The capital "A":** the font's native sharp-apex capital A (with its own ink
  crossbar). It is **type, not the glyph** — it deliberately does *not* try to
  mimic the crossbar-less glyph, which stays the distinct mark.
- **No underline.** The wordmark is set plain, in black, with **no aqua baseline
  rule** (an underline was tried and dropped — it competed with the glyph's
  waterline and cluttered the lockup). The aqua waterline lives in the glyph only.
- **Accent discipline:** the wordmark carries **no aqua at all**; aqua appears
  solely as the glyph's waterline crossbar.

---

## 5. Lockups

| Lockup | Composition | Use |
|---|---|---|
| **Horizontal** (primary) | glyph + `Algua`, **baselines aligned**; gap = one glyph stroke-width | README header, docs nav, default |
| **Stacked** | glyph above, wordmark below, centered | square-ish placements, social cards |
| **Glyph-alone** | the waterline A | favicon, avatar, app icon, terminal banner |
| **Wordmark-alone** | `Algua`, plain black | inline text, footers, where a glyph is redundant |

In the horizontal lockup the glyph's feet align to the wordmark's baseline. The
glyph supplies the only aqua (its waterline crossbar); the wordmark sits beside it
in plain black, so the eye lands on the one blue accent.

---

## 6. Color system

Near-monochrome with exactly **one accent**.

**Black and the water blue — that's it.**

| Token | Hex | Use |
|---|---|---|
| `ink` | `#000000` | the mark (letterforms) on light backgrounds; true black |
| `paper` | `#F3F6F6` | the mark on dark backgrounds; off-white |
| `aqua` | `#13C2CE` | **the waterline only** — the single accent (the water blue) |
| `aqua-deep` | `#0A8E99` | aqua used on light where WCAG AA contrast is required (small lines/text) |
| `mute` | `#5C6B6E` | dividers, captions, supporting UI — **never** the mark |

Rules:

- **Aqua is reserved for the glyph's waterline.** Letterforms (glyph legs and the
  whole wordmark) are never aqua.
- **No gradients. No second accent. No third tone.** Black + aqua only.
- **One-color fallback** (single-ink print, laser, embroidery, stamp): render the
  whole mark in `ink` *or* `paper`, with the waterline as a **knockout/outline**
  so the device still reads without color.
- **Contrast:** when aqua sits on `paper` or another light field at small sizes,
  use `aqua-deep` to hold AA.

**Required variants** (each lockup ships all that apply):

- `light` — black mark on paper, aqua waterline
- `dark` — paper mark on black, aqua waterline
- `mono` — single-ink, knockout waterline

---

## 7. Typography

| Role | Typeface | Notes |
|---|---|---|
| **Logotype** | Space Grotesk **Light (~300)**, outlined | slim/sleek; baked to paths, not a live font |
| **Display / headings** (README, docs) | **Space Grotesk** | sharp, technical; matches the logotype skeleton |
| **Body** | **Inter** (or system sans stack) | neutral, highly legible |
| **Code / CLI** | **JetBrains Mono** | terminal-adjacent; fits the agent-first, CLI character |

All four are open-licensed (OFL/Apache) — appropriate for an open repo with no
licensing friction.

---

## 8. Construction geometry

Masters (the spec produces measured drawings, not eyeballed art):

- **100-unit glyph master** — all proportions defined as ratios of this unit so
  the mark scales exactly.
- **24×24 favicon master** — the glyph is **redrawn (hinted) at favicon size**,
  not naively downscaled, so the apex and waterline stay crisp on the pixel grid.

Fixed parameters (exact values finalized during implementation, then frozen here):

As built (100-unit master): apex `(50,8)`, feet `(24,90)`/`(76,90)` (tall, narrow
A), **stroke width 5.75** (slim), waterline centred at `y=60`, spanning `x=30..70`.
The glyph is sized from the wordmark's cap-height in lockups (not a fixed box), so
it sits in balance with the text rather than overpowering it.

- **Stroke width** of the A legs — slim (5.75/100).
- **Waterline thickness** = **glyph stroke width** (same weight as the legs).
- **Overshoot.** The waterline extends past the legs at its height — a modest,
  symmetric overshoot reused at every scale.
- **Waterline vertical position** ≈ lower third of cap-height.

---

## 9. Usage rules

**Clear space:** minimum margin on all sides = the **cap-height of the glyph**.

**Minimum sizes:**

- Glyph: **16px**.
- Horizontal lockup: **96px** wide; below that, switch to glyph-alone.

**Don'ts:**

- Don't recolor the letterforms aqua (aqua = the glyph's waterline only).
- Don't add gradients, drop shadows, glows, a second accent, or a third tone.
- Don't round the apex.
- Don't rotate, tilt, or curve the waterline off horizontal.
- Don't add an aqua underline / baseline rule to the wordmark.
- Don't thicken the strokes — slim is the point; or stretch/condense the wordmark.
- Don't place the mark on a low-contrast field that breaks the variant rules.

---

## 10. Deliverables

Asset tree (target location `brand/` at repo root):

```
brand/
  algua-glyph.svg            # light; + -dark, -mono
  algua-wordmark.svg         # light; + -dark
  algua-lockup-h.svg         # horizontal (primary); + -dark, -mono
  algua-lockup-stacked.svg   # stacked; + -dark
  favicon.svg                # glyph on paper
  favicon-16.png
  favicon-32.png
  apple-touch-icon-180.png
  icon-512.png
  tokens.json                # the §6 color table as design tokens
  README.md                  # this spec's usage rules, condensed
  banner-dark.svg            # README header (black-bg horizontal lockup)
  build.py                   # regenerates the whole kit
```

Notes:

- Each variant in §6 is a separate file (suffix `-dark` / `-mono`).
- PNG favicons are currently exported by scaling the glyph master; a hand-hinted
  small master is a future refinement for extra crispness at 16px.
- `tokens.json` is the single source of truth for color; the SVGs reference the
  same hex values.

---

## 11. Acceptance criteria

The identity is complete when:

1. The glyph reads as **A** + aqua waterline at **16px**.
2. The mark is **black + aqua only** — no third tone, no gradient, no underline.
3. Aqua appears **only** on the glyph's waterline in every asset.
4. All §6 variants (light / dark / mono) export.
5. The full favicon/icon set renders at 16/32/180/512.
6. `tokens.json` matches the §6 table and the SVGs reference those values.
7. The mono one-color fallback preserves the waterline as a knockout.
