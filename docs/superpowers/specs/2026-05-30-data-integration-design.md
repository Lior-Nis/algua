# Data Integration — Real Bars into Backtests — Design

**Date:** 2026-05-30
**Branch:** `data-integration`. **Status:** Approved (design); plan to follow.

## 1. Goal

Connect the data layer to the backtest engine so a real, point-in-time backtest runs end to end:

```
algua data ingest-bars --provider yfinance --symbols AAPL,MSFT --start D --end D   # -> snapshot_id
algua backtest run cross_sectional_momentum --snapshot <snapshot_id>               # real backtest
```

Today the engine only runs on the synthetic provider (`--demo`). The data layer ingests bar
snapshots to parquet but exposes **no read-back path**, and its stored frame does not match the
engine's bar-schema. This sub-project adds the **serve/read layer** that bridges them.

## 2. Context: the two roles (not a duplicate interface)

- `algua/data/contracts.py::BarProvider` — **fetch** role: pulls bars from a vendor
  (`get_bars(BarRequest) -> ProviderBars`); feeds `DataStore.ingest_bars` which writes a parquet
  snapshot + provenance manifest entry.
- `algua/contracts/types.py::DataProvider` — **serve** role: hands point-in-time bars to consumers
  (`get_bars(symbols, start, end, timeframe) -> DataFrame`, bar-schema shaped); what the engine
  consumes.

These are different layers and both stay. What is missing is the serve path: reading a stored
snapshot back as bar-schema data and presenting it through `DataProvider`.

## 3. Bar-schema contract update

`docs/contracts/bar-schema.md` is updated: **a daily (`1d`) bar's `timestamp` is the session date
as a tz-aware UTC midnight** (e.g. `2024-07-01 00:00:00+00:00`), matching what real daily sources
(yfinance/Alpaca daily) provide. Intraday timeframes retain time-of-day. The `t→t+1` rule (engine
shift) remains the anti-look-ahead guarantee; timestamp time-of-day is not relied upon for daily.

This reverses the earlier "session close" choice. Consequences (the "undo"):
- `algua/calendar/market_calendar.py`: remove `session_closes()` (added for the reverted reason;
  unused now — YAGNI) and its test.
- `algua/backtest/_sample.py`: emit session **dates** (calendar `sessions_in_range`, holiday-aware,
  as tz-aware UTC midnight) instead of session-close times.
- `tests/test_backtest_sample.py`: the `test_timestamps_*` test asserts UTC session-date (midnight)
  semantics, not "not midnight".

## 4. Components

### 4.1 `validate_bars(df)` — the conformance check
A small validator (the one the bar-schema doc promised) that asserts a DataFrame matches the
frozen schema: index named `timestamp`, tz-aware UTC, monotonic non-decreasing; columns exactly
`["symbol","open","high","low","close","adj_close","volume"]`; no NaN in OHLC/adj_close; sorted by
`(timestamp, symbol)`. Raises a descriptive `ValueError` on violation; returns the frame on success.
Used by **both** `store.read_bars` and the synthetic provider's tests so real and synthetic data
are guaranteed identical in shape. Location: `algua/data/schema.py` (data-access concern; importable
by data + tests).

### 4.2 `DataStore.read_bars(snapshot_id) -> DataFrame`
Reads the snapshot's parquet (via `SnapshotRecord.data_path`), reshapes the stored frame into the
bar-schema: rename `ts → timestamp`, coerce to tz-aware UTC (localize/convert; daily → UTC midnight),
set as the index, order columns, sort by `(timestamp, symbol)`. Passes the result through
`validate_bars` before returning. Raises `SnapshotNotFound` for an unknown id and a clear error if
the snapshot's `dataset` is not `bars`.

### 4.3 `StoreBackedProvider` — a `DataProvider` over a snapshot
`StoreBackedProvider(store: DataStore, snapshot_id: str)` implements the `DataProvider` protocol.
`get_bars(symbols, start, end, timeframe)` calls `store.read_bars(snapshot_id)`, filters to
`symbols` (intersection with the snapshot) and the `[start, end]` window, and returns the bar-schema
frame. Exposes `.snapshot_id` so the engine can stamp it into `BacktestResult`. Lives in
`algua/data/` (it imports the store); the engine remains decoupled — it only sees `DataProvider`.

### 4.4 Engine snapshot stamp
`algua/backtest/engine.py::run` already records `seed`; extend it to also record
`snapshot_id = getattr(provider, "snapshot_id", None)` so real-data runs carry the provenance
stamp in `BacktestResult`. (No other engine change; it stays pure of the data lane.)

### 4.5 CLI: `backtest run --snapshot`
`algua/cli/backtest_cmd.py` gains `--snapshot <id>` (mutually exclusive with `--demo`):
- `--demo` → `SyntheticProvider` (unchanged).
- `--snapshot <id>` → `StoreBackedProvider(DataStore(settings.data_dir), id)`.
- neither → JSON error instructing the user to pass one.
Emits the same `BacktestResult` JSON; with `--snapshot`, `snapshot_id` is populated. `--register`
continues to work.

## 5. Data flow

```
ingest:  vendor --BarProvider.get_bars--> ProviderBars --store.ingest_bars--> parquet snapshot (+manifest)
serve:   snapshot parquet --store.read_bars--> bar-schema DataFrame --StoreBackedProvider--> DataProvider
run:     algua backtest run <strategy> --snapshot <id>
           -> StoreBackedProvider serves bars -> engine per-bar loop + t->t+1 -> BacktestResult(snapshot_id=...)
```

## 6. Error handling
- Unknown snapshot id → `SnapshotNotFound` → CLI renders `{"ok": false, "error": ...}` exit 1.
- Snapshot is not a `bars` dataset → clear `ValueError`.
- Snapshot lacks requested universe symbols / empty window → `BacktestError` from the engine
  (existing empty-bars guard), surfaced as JSON.
- `--snapshot` and `--demo` both/neither given → `ValueError` with guidance.

## 7. Testing
- `validate_bars`: accepts a conformant frame; rejects each violation (missing column, wrong index
  name, tz-naive index, unsorted, NaN OHLC).
- `store.read_bars`: ingest a known bars frame via `ingest_bars`, read it back, assert it equals the
  expected bar-schema frame and passes `validate_bars`; unknown id → `SnapshotNotFound`.
- `StoreBackedProvider`: serves the snapshot filtered by symbols + window; output passes
  `validate_bars`; `.snapshot_id` exposed.
- Synthetic provider: still passes `validate_bars`; timestamps are UTC session dates (midnight).
- CLI end-to-end: ingest a small bars frame (or fixture parquet) → `backtest run <strategy>
  --snapshot <id>` → metrics JSON with `snapshot_id` populated; `--demo` still works; neither/both
  flags → JSON error.
- Full gate stays green: `pytest`, `ruff`, `mypy`, `lint-imports` (4 contracts; engine still must
  not import `algua.data`).

## 8. Out of scope (later)
- Auto-selecting / stitching multiple snapshots for a universe+range (v1 is one explicit snapshot).
- Intraday timeframes and their time-of-day timestamps.
- As-of/point-in-time reads across data revisions (the `as_of` parameter on `get_bars`).
- Live vendor fetch directly from `backtest run` (always go through an ingested snapshot for
  reproducibility).
