# Algua brand kit

Algua's identity combines a narrow geometric capital **A** with a restrained water plane. The A's
own apex carries the upward direction—there is no separate arrow. A recessed horizon sits behind
its legs; a wider trapezoidal foreground horizon crosses the A and follows its outer slope. The
space between them uses the same ink as the A—Obsidian on light backgrounds and Ice on dark ones—
creating a solid water plane. The result
is calm and assured without borrowing the usual trading vocabulary of charts, candlesticks, coins,
or market animals.

## Identity

**Sea** is the single approved treatment. Two layered horizons express calm intelligence emerging
from uncertainty, while the steep A preserves precision and small-size legibility. Do not create or
use alternate water treatments.

| Asset | Use |
|---|---|
| `logo-horizontal.svg` | Default lockup on light backgrounds |
| `logo-horizontal-reversed.svg` | Default lockup on dark backgrounds |
| `logo-mark.svg` | Standalone mark on light backgrounds |
| `logo-mark-reversed.svg` | Standalone mark on dark backgrounds |
| `favicon.svg` | Browser icon and small square avatar |
| `social-avatar.svg` | Social profile and community avatar |
| `readme-banner.svg` | Repository and documentation banner |
| `tokens.css` | Copy-ready web color and typography tokens |
| `build_assets.py` | Deterministic source for every production SVG |
| `exports/` | Ready-to-use PNG exports |
| `mockups/` | Illustrative premium applications; never use as logo source files |

## Color

| Token | Hex | Role |
|---|---:|---|
| Obsidian | `#0B0D12` | Primary ink and dark background |
| Electric | `#1267FF` | Both sea horizons and brand signal on light backgrounds |
| Electric Light | `#3982FF` | Both sea horizons and brand signal on Obsidian backgrounds |
| Ice | `#F7F9FC` | Primary light surface and reversed ink |
| Mist | `#DCE4F0` | Dividers and quiet surfaces |
| Slate | `#5D6675` | Secondary text on light surfaces |
| Fog | `#A9B4C5` | Secondary text on dark surfaces |

Keep the palette mostly neutral. Electric blue should behave like a signal: concentrated, decisive,
and rare. Never use blue for the A legs; the A itself is the upward form. Both horizons use the same
blue. The recessed horizon is drawn behind the A, while the trapezoidal foreground horizon is drawn
in front. Preserve this layer order: it creates the mark's restrained depth without effects.

## Typography

- **Primary:** Inter, weights 300, 400, and 500. Use 300 for display typography.
- **Code and data:** IBM Plex Mono, weight 400 or 500.
- **Wordmark:** A custom-tracked Fira Sans ExtraLight drawing, supplied entirely as vector outlines.
  It has no runtime font dependency and must not be retyped or substituted.
- Use sentence case for product language. Reserve all-caps with generous tracking for short labels.

## Master construction

Every approved asset is generated from one fill-only SVG master on a 512-unit grid. The two sea
boundaries share one blue; their angled terminals follow the A's outer slope. The upper boundary
sits behind the A, the lower trapezoid sits in front, and the channel between them is cut in the A's
ink color. There are no strokes, live text, filters, masks, bevels, or raster effects in the core
identity.

Rebuild the SVG source after a deliberate master change:

```bash
uv run python docs/brand/build_assets.py
```

PNG files in `exports/` are delivery derivatives. The SVG files remain authoritative.

The images in `mockups/` demonstrate material and digital applications of the same identity. They
are presentation imagery, not additional logo variants, and must never be used to reconstruct the
mark or wordmark.

## Spacing and size

Use the A leg width as the clear-space unit **x**. Keep at least **x** clear on every side of
the mark or lockup.

- Minimum digital mark: **24 px** high.
- Minimum horizontal lockup: **120 px** wide.
- Below 32 px, use `favicon.svg`; it preserves the master geometry on a high-contrast tile.

## Usage rules

Do:

- Use Sea as the only identity.
- Use the reversed assets on Obsidian or similarly dark fields.
- Preserve the supplied proportions and clear space.
- Use the supplied light or dark production asset without altering its construction.

Do not:

- Add a separate arrow or icon above the A.
- Stretch, outline, bevel, shadow, or add gradients to the core mark.
- Flatten or reverse the sea/A layer order in full-color assets.
- Animate the water as a price chart.
- Place the full-color mark on busy imagery or low-contrast blue surfaces.
- Modify or substitute the approved Sea treatment.

## Voice

Algua speaks with calm certainty. Prefer short, factual language grounded in evidence.

- **Precise:** say what the system measured and under which conditions.
- **Reliable:** make constraints and safety gates visible.
- **Bold:** state decisions directly; avoid hype, swagger, and guaranteed-outcome language.

The line in `readme-banner.svg` — “Agent-first algorithmic research” — is a product descriptor,
not a permanent tagline.
