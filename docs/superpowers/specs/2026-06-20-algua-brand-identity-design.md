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

A geometric capital **A**, built from straight, sharp strokes, with **no crossbar
of its own**. The aqua waterline *is* the crossbar.

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
  reaching beyond the form). Overshoot = **≈12% of the leg's horizontal span on
  each side** — one ratio, reused at every scale.
- **Sharp apex.** The top is a true point. No rounding, no ink trap softening.
- **Optional two-tone ("submerged").** The lower triangle of the A — the region
  *below* the waterline — may drop to `ink-submerged`, suggesting "below water
  level." This is the **only** allowed tonal move. The spec defines **two glyph
  variants**: `flat` (single ink) and `two-tone` (submerged lower triangle).
  Favicon and any sub-32px use **must** use `flat`.

Legibility intent: at 16px the glyph reads as a crisp monochrome **A** bisected by
one bright aqua line; the overshoot keeps the waterline distinguishable even when
the letterform itself blurs.

---

## 4. Wordmark

The wordmark is `Algua` — **capital A, lowercase `lgua`** (title case). The
capitalized initial gives the mark a name-like, proper-noun presence while the
lowercase tail keeps it quiet and liquid, not all-caps loud.

- **Skeleton:** drawn from a **geometric grotesk (Space Grotesk Medium)**, then
  **converted to outlines** in the final asset so the wordmark is
  font-independent (no dependency on the font being installed/licensed at use).
- **Letterforms:** tight-but-open tracking; flat terminals; the lowercase `a`
  uses the **single-story round-bowl** form to echo the glyph's geometry.
- **The capital "A":** a standard sharp-apex capital A with a **normal ink
  crossbar** (this is type, not the glyph). To make it *rhyme* with the glyph
  without breaking accent discipline, its crossbar sits at the **same height
  ratio** as the glyph's waterline — same line, lower down, rendered in ink.
- **Baseline waterline:** the waterline device reappears as a **thin aqua baseline
  rule** that the letters sit on, overshooting the final `a` by the **same
  overshoot ratio** (≈12%) as the glyph. This is the single visual hinge that
  unifies glyph and wordmark — same device, same accent, scaled down.
- **Accent discipline:** the letterforms are **never** aqua — including the
  capital A's crossbar (ink). Only the baseline rule is aqua.

---

## 5. Lockups

| Lockup | Composition | Use |
|---|---|---|
| **Horizontal** (primary) | glyph + `Algua`, **baselines aligned**; gap = one glyph stroke-width | README header, docs nav, default |
| **Stacked** | glyph above, wordmark below, centered | square-ish placements, social cards |
| **Glyph-alone** | the waterline A | favicon, avatar, app icon, terminal banner |
| **Wordmark-alone** | `Algua` + baseline rule | inline text, footers, where a glyph is redundant |

In the horizontal lockup the glyph's waterline and the wordmark's baseline rule
share the **same vertical position** (both are baselines), reinforcing that
they're the same line.

---

## 6. Color system

Near-monochrome with exactly **one accent**.

| Token | Hex | Use |
|---|---|---|
| `ink` | `#0C1618` | primary mark on light backgrounds; deep slate-black |
| `ink-submerged` | `#16282B` | the optional "below waterline" lower-triangle tint |
| `paper` | `#F3F6F6` | the mark on dark backgrounds; off-white |
| `aqua` | `#13C2CE` | **the waterline only** — the primary accent |
| `aqua-deep` | `#0A8E99` | aqua used on light where WCAG AA contrast is required (small lines/text) |
| `mute` | `#5C6B6E` | dividers, captions, supporting UI — **never** the mark |

Rules:

- **Aqua is reserved for the waterline.** Letterforms are never aqua.
- **No gradients. No second accent.** If a future surface needs more color, it
  comes from neutrals, not new hues.
- **One-color fallback** (single-ink print, laser, embroidery, stamp): render the
  whole mark in `ink` *or* `paper`, with the waterline as a **knockout/outline**
  so the device still reads without color.
- **Contrast:** when aqua sits on `paper` or another light field at small sizes,
  use `aqua-deep` to hold AA.

**Required variants** (each lockup ships all that apply):

- `light` — ink mark on paper, aqua waterline
- `dark` — paper mark on ink, aqua waterline
- `mono` — single-ink, knockout waterline (flat only)

`flat` vs `two-tone` applies on top of `light`/`dark` for the glyph and any lockup
containing it.

---

## 7. Typography

| Role | Typeface | Notes |
|---|---|---|
| **Logotype** | custom-drawn (Space Grotesk skeleton) | outlined in the final asset; not a live font |
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

- **Stroke width** of the A legs (in master units).
- **Apex angle** / leg spread.
- **Waterline thickness** = **glyph stroke width** (the waterline is visually the
  same weight as the legs).
- **Overshoot ratio** = **≈12%** of leg horizontal span per side — reused for the
  glyph crossbar *and* the wordmark baseline rule.
- **Waterline vertical position** as a fraction of cap-height (sets how much of
  the A is "submerged").

---

## 9. Usage rules

**Clear space:** minimum margin on all sides = the **cap-height of the glyph**.

**Minimum sizes:**

- Glyph: **16px**.
- Horizontal lockup: **96px** wide; below that, switch to glyph-alone.

**Don'ts:**

- Don't recolor the letterforms aqua (aqua = waterline only).
- Don't add gradients, drop shadows, glows, or a second accent color.
- Don't round the apex.
- Don't rotate, tilt, or curve the waterline off horizontal.
- Don't use the `two-tone` glyph below 32px (use `flat`).
- Don't stretch, condense, or re-track the wordmark.
- Don't place the mark on a low-contrast field that breaks the variant rules.

---

## 10. Deliverables

Asset tree (target location `brand/` at repo root):

```
brand/
  algua-glyph.svg            # flat + two-tone, light/dark/mono
  algua-wordmark.svg         # light/dark/mono
  algua-lockup-h.svg         # horizontal (primary), light/dark/mono
  algua-lockup-stacked.svg   # stacked, light/dark/mono
  favicon.svg                # flat glyph, redrawn at small size
  favicon-16.png
  favicon-32.png
  apple-touch-icon-180.png
  icon-512.png
  tokens.json                # the §6 color table as design tokens
  README.md                  # this spec's usage rules, condensed
  banner-dark.svg            # README header (dark-bg horizontal lockup)
```

Notes:

- Variant packaging (light/dark/mono, flat/two-tone) may be separate files or a
  single SVG with `id`-addressable layers — decided at implementation; either way
  every variant in §6 must be exportable.
- PNG favicons are exported from the **24×24 hinted master**, not the 100-unit
  master.
- `tokens.json` is the single source of truth for color; the SVGs reference the
  same hex values.

---

## 11. Acceptance criteria

The identity is complete when:

1. The glyph reads unambiguously as **A** + waterline at **16px** (flat variant).
2. Glyph and wordmark visibly share the **one waterline device** at the **same
   overshoot ratio**.
3. Aqua appears **only** on the waterline in every asset.
4. All §6 variants (light/dark/mono; flat/two-tone where applicable) export.
5. The full favicon/icon set renders crisp at 16/32/180/512.
6. `tokens.json` matches the §6 table and the SVGs reference those values.
7. The mono one-color fallback preserves the waterline as a knockout.
