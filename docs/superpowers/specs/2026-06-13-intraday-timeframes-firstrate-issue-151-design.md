# Design — Data import slice 5: intraday timeframes (FirstRate) — issue #151

**Date:** 2026-06-13
**Issue:** #151 (follow-up slice of #129; slice 1 = FirstRate import, merged in #144)
**Status:** Approved for planning.

## Goal

Lift the `1d`-only restriction in the FirstRate import path so intraday vendor bars
(`1m`/`5m`/`30m`/`1h`) import to the bar-schema correctly, with timezone-aware,
DST-correct, never-silently-shifted timestamps.

`import_bars` in `algua/data/importers/firstrate.py` currently raises
`"intraday import not yet supported (1d only)"`. This is not just a guard deletion: the daily
parser localizes a bare date to UTC midnight, whereas intraday `datetime` carries time-of-day in
vendor-local time (FirstRate US-equity intraday is **US/Eastern**) and must be localized to the
exchange tz and converted to UTC, with DST handled.

## Scope decisions (settled)

- **FirstRate only.** Databento keeps its `1d`-only guard. Databento is UTC-native and its CA
  back-adjustment is ex-date-keyed; applying a per-session adjustment factor across many intraday
  bars is a separate, harder problem deserving its own slice.
- **Exchange tz is hardcoded** `America/New_York` (a module constant). FirstRate US-equity intraday
  is ET; no current consumer needs a configurable per-feed tz (YAGNI).
- **Closed timeframe vocabulary, no spacing validation.** One canonical set lives in a new shared
  module. Unknown tokens are rejected. Observed inter-bar spacing is NOT validated against the
  declared timeframe (brittle against legitimate gaps, half-days, and extended-hours sessions).
- **DST ambiguous/nonexistent timestamps fail closed** (raise a clean `ValueError` naming the
  offending timestamp). Aligns with the contract's "never silently shifted." US equity hours
  (including extended 04:00–20:00 ET) never fall in the 02:00–03:00 ET DST-transition window, so
  this is a guard, not a normal path.

## Architecture

### New module: `algua/data/timeframes.py`

The single source of truth for the timeframe vocabulary — the "shared with any other importer"
factoring the issue asks for.

```python
DAILY = "1d"
INTRADAY = frozenset({"1m", "5m", "30m", "1h"})   # FirstRate's actual intraday offerings
KNOWN = frozenset({DAILY, *INTRADAY})

def validate_timeframe(tf: str) -> str:   # return tf; raise ValueError on unknown token
def is_intraday(tf: str) -> bool:         # validate then report DAILY vs intraday
```

Pure (stdlib only) — no pandas, no cross-module imports beyond none. Lives in `algua/data` (not
`algua/contracts`, which must stay pure of vendor specifics).

### `algua/data/importers/firstrate.py`

- Add module constant `_EXCHANGE_TZ = "America/New_York"`.
- `parse_firstrate_file(path: Path, timeframe: str) -> pd.DataFrame` — add the `timeframe`
  parameter and branch:
  - **Daily** (`is_intraday` false): unchanged behavior — bare date; naive → UTC midnight,
    tz-aware → `tz_convert("UTC")`.
  - **Intraday**: the parsed `datetime` must be **naive** (FirstRate intraday is naive ET
    wall-clock). Localize `America/New_York` with `ambiguous="raise", nonexistent="raise"`, then
    `tz_convert("UTC")`. A tz-aware intraday input raises `ValueError` (its wall-clock tz is
    unknowable — fail closed, mirroring the Databento "refusing to shift" stance). A
    DST-ambiguous/nonexistent local time raises a clean `ValueError` naming the timestamp (wrap
    pandas' `AmbiguousTimeError`/`NonExistentTimeError`).
- `FirstRateImporter.import_bars`: replace `if request.timeframe != "1d": raise ...` with
  `validate_timeframe(request.timeframe)`; thread `request.timeframe` into both
  `parse_firstrate_file` calls inside `_merge_symbol`.
- `_merge_symbol`: the three diagnostic messages currently render `.dt.date` / `.date()`, which
  collapse intraday timestamps so two bars in the same session look like one "date". Render full
  ISO timestamps instead (correct for daily too). Detection logic is unchanged (already on full
  `ts`).

### No change to the write/read path

`to_bar_schema`, `validate_bars`, `DataStore.ingest_bars_streamed` (symbol-major partitioned
write), and `StoreBackedProvider.get_bars` / `read_bars` pushdown are already timeframe-agnostic
and intraday-ready (they key off tz-aware UTC timestamps, not daily-ness). The end-to-end round-trip
is proven in tests but no code there changes.

### Documentation

Reconcile `docs/contracts/bar-schema.md`: its `timeframe` vocabulary line currently lists
`"1h", "15m", "1m"`. Align it to `1m`/`5m`/`30m`/`1h` (matching the new canonical set) and note
that ingest validates the token against the closed vocabulary.

## Data flow (intraday import)

1. Operator runs `uv run algua data import-bars --vendor firstrate --raw-dir DIR
   --adjusted-dir DIR --timeframe 1m --as-of TS`.
2. CLI builds a `FirstRateImportRequest(timeframe="1m", ...)`.
3. `import_bars` validates the timeframe token, discovers raw/adjusted files, checks dir roles,
   and yields one `ProviderBars` per symbol.
4. Per symbol: `parse_firstrate_file(..., "1m")` localizes naive ET → UTC for both raw and
   adjusted files; `_merge_symbol` merges on exact UTC `ts`, validates prices, sorts.
5. `ingest_bars_streamed` normalizes each chunk via `to_bar_schema`, writes the
   `symbol=<SYM>/` partition, composes the content hash, commits the snapshot with
   `timeframe="1m"` in its metadata.
6. `StoreBackedProvider.get_bars(..., "1m")` matches the snapshot timeframe and serves the
   half-open `[start, end)` window with time-of-day preserved.

## Error handling

- Unknown timeframe token → `ValueError` from `validate_timeframe` (at the top of `import_bars`,
  before any file I/O).
- tz-aware intraday input → `ValueError` (cannot infer wall-clock tz).
- DST ambiguous/nonexistent local time → `ValueError` naming the offending timestamp.
- Existing guards unchanged: malformed file, missing columns, alias/symbol collisions, raw/adjusted
  key-set mismatch, NaN/nonpositive prices, raw/adjusted symbol-set mismatch.

## Testing

- **ET→UTC correctness:** 09:30 ET → 13:30 UTC (EDT summer); 09:30 ET → 14:30 UTC (EST winter).
- **DST-boundary day:** a file spanning 2024-03-10 (spring-forward) and one spanning 2024-11-03
  (fall-back) — regular-hours bars convert correctly across the boundary within a single file.
- **DST fail-closed:** synthetic 02:30 ET on 2024-03-10 (nonexistent) → `ValueError`; synthetic
  01:30 ET on 2024-11-03 (ambiguous) → `ValueError`.
- **Multiple timeframes:** 1m / 5m / 1h parse and import.
- **Ordering/uniqueness:** output ascending by `(timestamp, symbol)`, no duplicate
  `(timestamp, symbol)`; an intraday duplicate raises with full timestamps in the message.
- **tz-aware intraday input** → `ValueError`. **Unknown timeframe** → `ValueError`.
- **End-to-end:** `import-bars --timeframe 1m` → snapshot → `read_bars` preserves time-of-day and
  honors the half-open `[start, end)` window.
- **Daily regression:** existing daily import behavior unchanged.

## Assumptions (flagged)

- **FirstRate intraday timestamps are bar-open in ET.** We preserve them as-is (no open/close
  shift). Look-ahead safety comes from the serving half-open window plus the engine's `t→t+1`
  shift, not from the timestamp label — consistent with `docs/contracts/bar-schema.md`.

## Known limitations / deferred

- `read_bars` materializes the *filtered* window into a pandas frame (not the whole snapshot).
  True constant-memory / chunked serving remains the deferred #130 read-path-trio item — not a
  blocker for this slice, since intraday backtests read a bounded window.
- Databento intraday (intraday CA back-adjustment) — separate future slice.
- Intraday import of any other vendor (yfinance ingest path) — not in scope.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
