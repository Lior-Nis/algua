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
- **The declared timeframe is operator-asserted metadata, recorded as-is** — exactly like the
  `adjustment` flavor. The system does not infer or cross-check bar granularity from the data or
  the filename (e.g. it will not detect 1m bars imported under `--timeframe 5m`); that mislabel is
  operator error, parallel to pointing `--adjustment` at the wrong flavor. What the system DOES
  guard, fail-closed, is the daily↔intraday confusion that would corrupt timestamps (see the two
  midnight guards below) — because that crosses the tz-localization fork and silently shifts data.
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

The vocabulary is enforced fail-closed at three points so an unknown token can neither enter a
snapshot nor confuse a read:
- `FirstRateImporter.import_bars` (early, before any file I/O — best operator UX).
- **Both bars-snapshot creation chokepoints** — `DataStore.ingest_bars_streamed` AND the
  non-streamed `DataStore.ingest_bars` — so a snapshot with an unknown timeframe can never be
  written by either path (otherwise it would later be unservable once `get_bars` validates).
  `timeframe` is always a real string at both (default `"1d"`).
- `StoreBackedProvider.get_bars` (the serving seam — validate the *requested* token before the
  snapshot-mismatch check, so `get_bars(..., "15m")` returns `"unknown timeframe '15m'..."` rather
  than a confusing "snapshot is '1m', not '15m'"). Validating the request token is safe even for
  legacy snapshots whose stored `timeframe` may be `None`.

(Not enforced at the generic `_metadata` helper / `data ingest --from-file` path, which may carry a
`None` timeframe — that path is out of scope and validating there would false-reject it.)

### `algua/data/importers/firstrate.py`

- Add module constant `_EXCHANGE_TZ = "America/New_York"`.
- `parse_firstrate_file(path: Path, timeframe: str = "1d") -> pd.DataFrame` — add the `timeframe`
  parameter (defaulted to `"1d"` so existing direct/daily callers and tests are unaffected) and
  branch:
  - **Daily** (`is_intraday` false): bare date; naive → UTC midnight, tz-aware → `tz_convert("UTC")`
    — then **assert every timestamp is UTC midnight** (`ts == ts.normalize()`), raising a clean
    `ValueError` naming the offender otherwise. This closes a pre-existing hole (a `1d`-labelled
    file carrying time-of-day would otherwise store non-midnight `1d` bars, violating the contract;
    `validate_bars` does not catch it) and gives parity with `databento.py`'s daily guard. Real
    FirstRate daily files are bare dates, so no legitimate input is rejected.
  - **Intraday**: the parsed `datetime` must be **naive** (FirstRate intraday is naive ET
    wall-clock). A tz-aware intraday input raises an actionable `ValueError`, e.g. _"FirstRate
    intraday timestamps must be naive (wall-clock US/Eastern); found a tz-aware column — strip
    timezone offsets before importing."_ (its wall-clock tz is unknowable — fail closed, mirroring
    the Databento "refusing to shift" stance). **Reject any naive value at local (ET) midnight**
    before localization — a 00:00 bar is never a valid FirstRate US-equity intraday bar (extended
    hours start 04:00 ET), so a midnight value means a date-only / daily-shaped file was fed to the
    intraday path; raise a clean `ValueError` naming it. Then localize `America/New_York` with
    `ambiguous="raise", nonexistent="raise"` and `tz_convert("UTC")`. A DST-ambiguous/nonexistent
    local time raises a clean `ValueError` naming the timestamp (wrap pandas'
    `AmbiguousTimeError`/`NonExistentTimeError`).
- `FirstRateImporter.import_bars`: replace `if request.timeframe != "1d": raise ...` with
  `validate_timeframe(request.timeframe)`; thread `request.timeframe` into both
  `parse_firstrate_file` calls inside `_merge_symbol`.
- `_merge_symbol`: the three diagnostic messages currently render `.dt.date` / `.date()`, which
  collapse intraday timestamps so two bars in the same session look like one "date". Render full
  ISO timestamps instead (correct for daily too). Detection logic is unchanged (already on full
  `ts`).

### Minimal write/read-path touch (vocab enforcement only)

`to_bar_schema`, `validate_bars`, `DataStore.ingest_bars_streamed` (symbol-major partitioned
write), and `StoreBackedProvider.get_bars` / `read_bars` pushdown are already timeframe-agnostic
and intraday-ready (they key off tz-aware UTC timestamps, not daily-ness). The end-to-end round-trip
is proven in tests. The ONLY production change here is the three one-line `validate_timeframe(...)`
guards described above (`ingest_bars` + `ingest_bars_streamed` entries and `get_bars` entry); the
partitioned write/read logic itself is unchanged.

### Documentation

Reconcile `docs/contracts/bar-schema.md`:
- Its `timeframe` vocabulary line currently lists `"1h", "15m", "1m"`. Align it to
  `1m`/`5m`/`30m`/`1h` (the new canonical set) and note that ingest + serve validate the token
  against the closed vocabulary.
- That same line calls intraday boundaries **"UTC-aligned bar boundaries"** — this is wrong for
  exchange-local-sourced data. Rewrite it: intraday timestamps preserve the **exchange/vendor bar
  label converted to a UTC instant**; they are NOT required to fall on UTC-clock-aligned boundaries
  (e.g. 09:30 ET → 13:30 UTC in EDT, 14:30 UTC in EST). Look-ahead safety comes from the half-open
  serving window + the engine `t→t+1` shift, not from clock alignment.

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

- Unknown timeframe token → `ValueError` from `validate_timeframe` (importer entry before any file
  I/O; also at the `ingest_bars_streamed` and `get_bars` chokepoints).
- tz-aware intraday input → actionable `ValueError` ("strip timezone offsets before importing").
- Intraday timestamp at local (ET) midnight → `ValueError` (date-only/daily file fed to intraday).
- Daily timestamp not at UTC midnight → `ValueError` (intraday file fed to the daily path).
- DST ambiguous/nonexistent local time → `ValueError` naming the offending timestamp.
- Existing guards unchanged: malformed file, missing columns, alias/symbol collisions, raw/adjusted
  key-set mismatch, NaN/nonpositive prices, raw/adjusted symbol-set mismatch.

## Testing

- **ET→UTC correctness:** 09:30 ET → 13:30 UTC (EDT summer); 09:30 ET → 14:30 UTC (EST winter).
- **DST-boundary day:** a file spanning 2024-03-10 (spring-forward) and one spanning 2024-11-03
  (fall-back) — regular-hours bars convert correctly across the boundary within a single file.
- **DST-adjacent valid timestamps** (the most likely real edge values): 03:00 ET on 2024-03-10
  (just after the spring-forward gap) → 07:00 UTC; 00:30 ET on 2024-11-03 (occurs only once,
  unambiguous) → 04:30 UTC.
- **DST fail-closed:** synthetic 02:30 ET on 2024-03-10 (nonexistent) → `ValueError`; synthetic
  01:30 ET on 2024-11-03 (ambiguous) → `ValueError`.
- **Multiple timeframes:** 1m / 5m / 30m / 1h parse and import.
- **Ordering/uniqueness:** output ascending by `(timestamp, symbol)`, no duplicate
  `(timestamp, symbol)`; an intraday duplicate raises with full timestamps in the message.
- **tz-aware intraday input** → `ValueError`. **Unknown timeframe** → `ValueError` (importer,
  ingest, and serve).
- **Daily midnight guard:** a `1d`-labelled file carrying time-of-day → `ValueError`.
- **Intraday midnight guard:** a date-only / local-midnight value under an intraday timeframe →
  `ValueError`.
- **End-to-end:** `import-bars --timeframe 1m` → snapshot → `read_bars` preserves time-of-day and
  honors the half-open `[start, end)` window.
- **Daily regression:** existing daily import behavior unchanged (bare-date files still import).

## Assumptions (flagged)

- **FirstRate intraday timestamps are bar-open in ET.** We preserve them as-is (no open/close
  shift). Look-ahead safety comes from the serving half-open window plus the engine's `t→t+1`
  shift, not from the timestamp label — consistent with `docs/contracts/bar-schema.md`.

## Known limitations / deferred

- `read_bars` materializes the *filtered* window into a pandas frame (not the whole snapshot).
  True constant-memory / chunked serving remains the deferred #130 read-path-trio item — not a
  blocker for this slice, since intraday backtests read a bounded window.
- **Intraday coverage metadata is date-granularity.** `ingest_bars_streamed` records
  `observed_start`/`observed_end` (and the span-coverage check) at `.date()` granularity. For
  intraday this is conservative-but-correct for the span check; it loses time-of-day in the
  recorded metadata and cannot detect interior intraday gaps (a known limitation of the existing
  span-only check for daily too). Promoting these to full ISO timestamps entangles the streamed-
  ingest coverage comparison (mixed date/datetime string ordering) and the snapshot-identity
  metadata format, so it is deferred — it relates to the #130 read-path / coverage-fidelity trio,
  not to this slice's tz-localization goal.
- Databento intraday (intraday CA back-adjustment) — separate future slice.
- Intraday import of any other vendor (yfinance ingest path) — not in scope.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
