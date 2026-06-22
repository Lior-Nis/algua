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

## 2. Concept: the ocean and the fin

The identity rests on **one device: a single horizontal aqua "waterline"** — the
**ocean surface**. It runs *past* the A's legs, longer than the letter itself, so
it reads as open water. The A's **sharp apex breaks the surface like a shark's
fin** cutting through the sea. The line reads several ways at once:

- **water** — the *aqua* in Algua, the sea the fin slices through;
- a **price / equity level** — a market line;
- a **threshold / gate** — the lifecycle's disciplined checkpoints
  (`idea → backtested → … → live`).

One line, zero decoration. **The waterline is the only place the aqua accent ever
appears.** Everything else is monochrome ink.

---

## 3. The glyph — "the fin"

The glyph is the **real Outfit capital `A`** (the same typeface as the
wordmark) sitting in the **aqua waterline**. Using a true letterform — not
hand-drawn strokes — gives the mark type-quality craft and ties it to the wordmark
(the mark *is* the "A" in "Algua," accented). The aqua bar sits at the
letterform's crossbar height but **runs well past both legs** as the ocean
surface; the sharp apex above it is the fin.

```
      /\          apex: a sharp point — the fin
     /  \
 ===/====\===   <- aqua waterline = the OCEAN, longer than the A
   /      \
```

Requirements:

- **Real letterform.** The black shape is the actual **Outfit `A`** outline, at
  the **same thin weight as the wordmark (~160)** — it is the wordmark's own `A`
  extracted — baked to a `<path>`. It keeps the typeface's optical corrections; it
  is never a polyline.
- **Sharpened apex.** Whatever flat the font's `A` has on top, both the outer apex
  and the inner counter tip are collapsed to **true points**, and the outer point
  is **raised above the cap line** (`APEX_SHARPEN`, font units) for a crisp,
  pointed peak — the fin. The same sharpened `A` is used in the wordmark.
- **Waterline = the ocean.** A **long, thin, tapered trapezoid** (not a rect): it
  sits at the `A`'s crossbar height, runs `WL_EXTEND` font units past the legs
  (longer than the letter), and its top edge is shorter than the bottom
  (`WL_TAPER`) for a horizon-in-perspective feel. In the wordmark the right side
  uses a smaller `WL_EXTEND_R` so the line clears the following `l`.
- **Solid fin above the surface.** The part of the `A` above the waterline is
  **filled solid** (the `_fin` triangle), so the emerged fin is a clean black
  shape with no open counter — this is what removes the white triangle the open
  `A` counter used to show on the line. The fin is drawn *under* the waterline so
  the trapezoid taper can never expose background. The legs stay slim below the
  surface (submerged body).
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

- **Skeleton:** drawn from the **geometric sans Outfit at a thin weight (~160)**
  for a slim, sleek hairline, then **converted to outlines** in the final asset so
  the wordmark is font-independent (no dependency on the font being installed at
  use). Outfit replaced Space Grotesk, whose thinnest weight (300) wasn't slim
  enough.
- **Letterforms:** airy tracking to match the thin weight; Outfit's native
  geometric lowercase is kept as-is (its **double-story `g`** included). **The
  canvas must clear the descender** — the `g` drops ~200 font units below the
  baseline, so the SVG height is computed as `baseline + descent·scale + pad` (a
  fixed height once clipped the bottom of the `g`).
- **The leading "A" carries the waterline.** The aqua trapezoid is drawn at the
  wordmark's *own* leading `A` (at its crossbar height, running past its legs).
  There is **no separate glyph placed beside the wordmark** — that would duplicate
  the A. The logo is simply `Algua` with its A sitting in the ocean line.
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
| **Logotype** | Outfit **Thin (~160)**, outlined | slim/sleek hairline; baked to paths, not a live font |
| **Display / headings** (README, docs) | **Outfit** | clean geometric; matches the logotype |
| **Body** | **Inter** (or system sans stack) | neutral, highly legible |
| **Code / CLI** | **JetBrains Mono** | terminal-adjacent; fits the agent-first, CLI character |

All four are open-licensed (OFL/Apache) — appropriate for an open repo with no
licensing friction.

---

## 8. Construction geometry

The glyph is **not eyeballed geometry** — it is the Outfit `A` outline placed and
measured programmatically (`brand/build.py`):

- **Letterform source.** The `A` glyph (thin weight ~160) is extracted from the
  font, baked to a `<path>`, and scaled into a **100-unit master**: cap top at
  `y=8`, baseline at `y=92`, centered horizontally.
- **Crossbar height, measured.** The script finds the crossbar from the inner
  counter / leg-split geometry; the waterline is centered on it.
- **Waterline = a long thin trapezoid.** A separate aqua polygon at the crossbar
  height, `WL_THICK` tall, running `WL_EXTEND` past each leg, with its top edge
  shorter than the bottom by `WL_TAPER` (the trapezoid). In the wordmark the right
  reach is capped (`WL_EXTEND_R`) to clear the `l`. (Earlier iterations recolored
  the A's own crossbar to keep slanted edges; the explicit long trapezoid better
  serves the ocean/fin reading.)
- **Lockup sizing.** The glyph is sized from the wordmark's cap-height (not a
  fixed box), so it balances the text rather than overpowering it.

---

## 9. Usage rules

**Clear space:** minimum margin on all sides = the **cap-height of the glyph**.

**Minimum sizes:**

- Glyph: **16px**.
- Horizontal lockup: **96px** wide; below that, switch to glyph-alone.

**Don'ts:**

- Don't recolor the letterforms aqua (aqua = the waterline only).
- Don't add gradients, drop shadows, glows, a second accent, or a third tone.
- Don't make the waterline a plain rect — keep it the long thin tapered trapezoid.
- Don't rotate or curve the waterline off horizontal.
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
