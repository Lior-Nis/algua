# FirstRate import: raw/adjusted directory-role guard (#148)

**Status:** design approved (reshaped after GATE-1 design review)
**Issue:** #148 (re-scoped). Numeric/economic `adj_close` validation relocated to #149.
**Date:** 2026-06-08

## Summary

Add a fail-closed guard to `FirstRateImporter` that rejects a `data import-bars` run whose
`--raw-dir` / `--adjusted-dir` are **swapped or mislabeled**, by checking FirstRate's filename role
marker. This catches the realistic operator error that would otherwise silently transpose `close`
and `adj_close` in the consolidated snapshot and corrupt every downstream backtest that uses
adjusted prices.

## Why this, and not the numeric "plausibility wall" #148 originally proposed

#148 was originally scoped as a numeric economic-plausibility wall on `adj_close` (e.g. adj/raw
ratio `r ≤ 1`, `r` monotone in time). A GATE-1 adversarial design review (Codex + Gemini, both
independently) found that scope **unsound on a hard-reject path**:

- **Reverse splits break it (CRITICAL, both models).** FirstRate back-adjustment scales *pre-event*
  prices **up** for a reverse split, so `adj_close > close` (`r > 1`) before the event — and by the
  same mechanism `r` steps *down* going forward across a reverse split. So **both** `r ≤ 1` and
  monotonicity false-reject legitimate reverse-split histories (common in distressed names and
  leveraged/inverse ETFs). A sound economic check needs the corporate-action event list — which is
  exactly #149's scope.
- **The misalignment it would catch is already impossible.** `_merge_symbol` joins raw and adjusted
  **by timestamp** (`raw.merge(adj, on="ts")`) with key-set parity and duplicate-`ts` already
  enforced, so row-shuffle / row-misalignment cannot occur. The monotonicity check had little value
  even before the reverse-split problem.
- **Tolerance is numerically fragile (HIGH, both models).** Rounding noise in `r = adj/close` scales
  as `¢/price`, so a single global ratio tolerance is simultaneously too tight for low-priced bars
  and too loose for high-priced ones.

The reviewers' one constructive, low-false-reject, ships-now recommendation was the **filename
role guard** below — reverse-split-safe (non-numeric) and aimed at the one realistic failure the
key-based merge does *not* already prevent: swapped/mislabeled directories.

**Decision:** build the role guard now; relocate all economic/numeric `adj_close` validation to
#149 (reverse-split-aware, event-based), where it is soundly buildable.

## The guard

FirstRate daily files are named by role: unadjusted files carry the token `UNADJUSTED`
(e.g. `AAPL_full_1day_UNADJUSTED.txt`); adjusted files do not (e.g. `AAPL_full_1day_adjsplitdiv.txt`).
The single discriminator — presence/absence of the case-insensitive substring `unadjusted` — catches
the swap from both directions with minimal naming assumptions:

1. **Every file in `--raw-dir` must look unadjusted** (filename contains `unadjusted`). Otherwise
   raise: the raw dir holds adjusted files — dirs are likely swapped.
2. **No file in `--adjusted-dir` may look unadjusted** (filename must not contain `unadjusted`).
   Otherwise raise: the adjusted dir holds unadjusted files — dirs are likely swapped, or both
   point at the raw dir.

Together these also catch "both dirs point at the same directory" (one side fails whichever way).

We deliberately do **not** require a positive *adjusted* marker (FirstRate's adjusted token varies:
`adjsplitdiv`, `adjsplit`, …) — requiring it would raise false-reject risk for no extra swap-
detection value. The `unadjusted` token alone is the robust discriminator.

## Placement

A small module-level helper in `algua/data/importers/firstrate.py` (FirstRate-naming-specific, so
**not** a shared module — Databento #150 has different naming), called from `import_bars`
**immediately after** the two `_discover(...)` calls, before the symbol-set parity check. Per-symbol
seam, fail-closed `ValueError`, consistent with the importer's existing guards.

## Out of scope (→ #149)

- Numeric/economic `adj_close` validation (reverse-split-aware, event-aligned).
- Latest-bar anchor (`r → 1` at the as-of date) — needs provenance the importer can't assert.
- Positive adjusted-marker enforcement / configurable markers.

## Testing

TDD. New cases in `tests/test_firstrate_importer.py`:

- `--raw-dir` holds adjusted-named files → raises (rule 1), message names the swap.
- `--adjusted-dir` holds an unadjusted-named file → raises (rule 2).
- Correctly-named dirs → import proceeds (regression guard; existing `_write_pair` tests already
  use correct names and must stay green).
- Case-insensitivity (`unadjusted` lower-case still matched).

Gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
