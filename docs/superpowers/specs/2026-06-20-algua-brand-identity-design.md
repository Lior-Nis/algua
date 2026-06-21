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

The glyph is the **real Space Grotesk capital `A`** (the same typeface as the
wordmark) with **its crossbar recolored to the aqua waterline**. Using a true
letterform — not hand-drawn strokes — is what gives the mark professional,
type-quality craft and ties it directly to the wordmark (the mark *is* the "A" in
"Algua," accented). The aqua bar sits exactly on the letterform's crossbar and
**overshoots both legs** as the "tide."

```
   /\          apex: a sharp point (never rounded)
  /  \
 /====\   <-  aqua waterline = crossbar; overshoots both legs
/      \
```

Requirements:

- **Real letterform.** The black shape is the actual Space Grotesk `A` outline,
  at the **same weight as the wordmark (~300)** — it is the wordmark's own `A`
  extracted — baked to a `<path>`. It keeps the typeface's optical corrections; it
  is never a polyline.
- **Crossbar = waterline.** The aqua bar is positioned and sized to the
  letterform's own crossbar band, replacing it. The result reads as a crossbar-less
  A whose crossbar is the waterline.
- **Overshoot.** The waterline extends past both legs (the tide reaching beyond
  the form) — a modest, symmetric overshoot.
- **Two colors only.** The mark is **black + the aqua waterline** — nothing else.
  (An earlier two-tone "submerged" tint, and earlier still a hand-drawn polyline
  caret, were both tried and dropped — the polyline read as crude SVG lines.)

Legibility intent: as a filled letterform the glyph reads as a solid black **A**
crossed by one bright aqua line, and stays legible down to 16px.

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
- **The capital "A" carries the waterline.** The wordmark's *own* leading `A` is
  the mark: its crossbar is recolored to the aqua waterline (a band-clipped copy
  of that `A`, so the aqua keeps the letterform's slanted edges). There is **no
  separate glyph placed beside the wordmark** — that would duplicate the A. The
  logo is simply `Algua` with its A accented.
- **No underline.** The wordmark is otherwise plain black — no aqua baseline rule.
- **Accent discipline:** the only aqua anywhere is that one crossbar.

---

## 5. The logo and the icon

There is **no glyph-beside-wordmark lockup** — the wordmark already contains the
accented `A`, so adding a separate glyph would show the letter twice. Instead:

| Asset | What | Use |
|---|---|---|
| **Logo** (primary) | the `Algua` wordmark, A accented with the waterline | README header, docs, default |
| **Icon** | that same `A`, extracted standalone | favicon, avatar, app icon, terminal banner |

The icon is literally the wordmark's `A` on its own — same weight, same waterline
— so the two never disagree.

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

The glyph is **not eyeballed geometry** — it is the Space Grotesk `A` outline
placed and measured programmatically (`brand/build.py`):

- **Letterform source.** The `A` glyph (weight ~400) is extracted from the font,
  baked to a `<path>`, and scaled into a **100-unit master**: cap top at `y=8`,
  baseline at `y=92`, centered horizontally.
- **Crossbar band, measured — not guessed.** The script finds the crossbar by
  geometry: the inner counter's lower edge is the crossbar *top*; a downward
  scanline of the outline finds where the legs split, giving the crossbar *bottom*.
- **Waterline = recolored crossbar.** A band-clipped copy of the letterform,
  filled aqua over that crossbar band, recolors the crossbar (and the leg slices
  it crosses). Because it is clipped to the letterform, the aqua **inherits the
  A's slanted leg edges** — it is never a rectangle laid on top. A small vertical
  bleed (`WL_BLEED`) ensures no black crossbar peeks above/below the aqua.
- **Lockup sizing.** The glyph is sized from the wordmark's cap-height (not a
  fixed box), so it balances the text rather than overpowering it.

---

## 9. Usage rules

**Clear space:** minimum margin on all sides = the **cap-height of the glyph**.

**Minimum sizes:**

- Glyph: **16px**.
- Horizontal lockup: **96px** wide; below that, switch to glyph-alone.

**Don'ts:**

- Don't recolor the letterforms aqua (aqua = the glyph's waterline band only).
- Don't add gradients, drop shadows, glows, a second accent, or a third tone.
- Don't lay the waterline back as a rectangle on top — it must be the letterform's
  own crossbar recolored, keeping the slanted leg edges.
- Don't rotate, tilt, or curve the waterline off horizontal.
- Don't add an aqua underline / baseline rule to the wordmark.
- Don't substitute a hand-drawn / polyline "A" for the real letterform.
- Don't stretch, condense, or re-track the wordmark.
- Don't place the mark on a low-contrast field that breaks the variant rules.

---

## 10. Deliverables

Asset tree (target location `brand/` at repo root):

```
brand/
  algua-wordmark.svg         # the logo (A accented); + -dark
  algua-glyph.svg            # the icon (the A alone); + -dark, -mono
  favicon.svg                # the icon on paper
  favicon-16.png
  favicon-32.png
  apple-touch-icon-180.png
  icon-512.png
  tokens.json                # the §6 color table as design tokens
  README.md                  # this spec's usage rules, condensed
  banner-dark.svg            # README header (wordmark on black)
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
