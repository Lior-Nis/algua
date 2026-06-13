# Data Contract — `get_bars` Bar Schema

**Status:** Authoritative. **Date:** 2026-05-30 (last updated when the data layer was integrated).

## Why this exists
This pins the **shape of the data crossing the seam between the data layer and the
research/backtest layer**:
- **Data layer** (`algua/data/*`): `DataProvider` adapters (Alpaca, yfinance) + the snapshot
  store + the `get_bars` read path (`StoreBackedProvider`).
- **Research/backtest layer** (`algua/features|strategies|backtest/*`): consumes bars through
  this exact shape; the engine depends only on the `DataProvider` protocol.

Pinning the shape keeps the seam mechanical: the backtest engine works unchanged whether it is
fed the synthetic fixture provider or a real store-backed snapshot. This contract is enforced in
code by `algua/data/schema.py::validate_bars`.

> Historical note: this document originated when the data layer and research layer were built in
> parallel by two agents; it now lives in a single-owner repo, but the seam — and the value of
> pinning it — is unchanged.

## The contract

```python
DataProvider.get_bars(
    symbols: list[str],
    start: datetime,
    end: datetime,
    timeframe: str,
) -> pandas.DataFrame
```

### Returned DataFrame — exact shape (tidy / long format)

- **One row per `(symbol, bar)`.**
- **Index:** a name=`timestamp`, **tz-aware** `DatetimeIndex` in **UTC**, monotonic non-decreasing.
  For daily (`1d`) bars the timestamp is the **session date at UTC midnight** (e.g.
  `2024-07-01 00:00:00+00:00`), matching what real daily sources (yfinance/Alpaca daily) provide.
  Intraday timeframes carry the bar's time-of-day. The `t→t+1` rule (engine shift) — not the
  timestamp's time-of-day — is what guarantees no look-ahead.
- **Columns (exact names, this order):**

  | column | dtype | meaning |
  |---|---|---|
  | `symbol` | `str` | e.g. `"AAPL"` |
  | `open` | `float64` | raw (unadjusted) |
  | `high` | `float64` | raw |
  | `low` | `float64` | raw |
  | `close` | `float64` | raw |
  | `adj_close` | `float64` | split- **and** dividend-adjusted close |
  | `volume` | `float64` | raw share volume |

- **Sort:** ascending by (`timestamp`, `symbol`).
- **Uniqueness:** no duplicate `(timestamp, symbol)` pairs.
- **Missing data:** represented by an **absent row**, never NaN-filled. When a row exists, OHLC
  values are non-NaN. (`adj_close` non-NaN; `volume` may be `0.0` but not NaN.)
- **Empty result:** an empty DataFrame **with these columns and a tz-aware empty index** (not
  `None`, not a bare `DataFrame()`), so consumers can rely on the schema unconditionally.

### Window boundary — half-open `[start, end)`
The serving `get_bars` window is **half-open**: `start` inclusive, `end` **exclusive**. A bar
timestamped exactly at `end` is **not** returned. This is the look-ahead-safe reading of "bars up
to time T" — asking for data as of T must never hand back the bar that prints at T. The store-read
path (`StoreBackedProvider.get_bars`) enforces this.

Ingestion (`BarRequest` / vendor adapters) shares this canonical convention but vendors differ at
the wire: **yfinance** treats `end` exclusive (matches); **Alpaca** treats `end` inclusive, so a
raw Alpaca pull may include the `end` bar. Adapters do not re-clip at ingestion — the boundary that
guards against look-ahead is the serving read path above.

### `timeframe` vocabulary
- `"1d"` — daily session bars. The research lane targets `"1d"` initially.
- `"1m"`, `"5m"`, `"30m"`, `"1h"` — intraday bars. The timestamp preserves the **exchange/vendor bar
  label converted to a UTC instant**; it is NOT required to fall on a UTC-clock-aligned boundary
  (e.g. an 09:30 US/Eastern open is `13:30Z` in EDT, `14:30Z` in EST). Look-ahead safety comes from
  the half-open serving window plus the engine `t→t+1` shift, not from clock alignment.
- The vocabulary is closed: `algua/data/timeframes.py::validate_timeframe` is the single source of
  truth, enforced at both bars-ingest chokepoints (`DataStore.ingest_bars` /
  `ingest_bars_streamed`) and at the serving seam (`StoreBackedProvider.get_bars`). Any other value
  → `ValueError`.
- The declared timeframe is operator-asserted metadata, recorded as-is (like the `adjustment`
  flavor): the system does not infer bar granularity from the data, so importing `1m` bars under
  `--timeframe 5m` is operator error, not a detected condition. What IS guarded fail-closed is the
  daily↔intraday confusion that would corrupt timestamps (non-midnight `1d`, or a local-midnight
  intraday bar).

### Adjustment semantics (the one judgment call — flag if you disagree)
`close` is **raw**; `adj_close` is **split + dividend adjusted**. Backtests compute returns from
`adj_close`; `close` is kept for reference and notional/position sizing. This mirrors common
vendor output (yfinance `Adj Close`, Alpaca `adjustment=all`) and avoids split-induced phantom
gaps in return series. Raw OHLC is preserved so corporate-action handling stays auditable.

### Point-in-time / `as_of` (coordination note)
The snapshot store already records `as_of` provenance. The `get_bars` signature above has **no
`as_of` parameter yet** — for now it returns the latest snapshot's bars. If/when we need
as-of-date reads (for leak-free walk-forward across data revisions), we add `as_of` to the
signature (see Change control) and update `validate_bars` + consumers together. Until then,
consumers must not assume as-of reads exist.

## Conformance
- `algua/data/schema.py::validate_bars(df)` is the canonical encoding of this schema; it raises
  on the first violation (index name/tz/monotonicity, exact column list, symbol/numeric dtypes,
  non-null OHLC+volume, uniqueness and sort of `(timestamp, symbol)`). `to_bar_schema` reshapes
  a stored frame into it.
- Data layer: `DataStore.read_bars` / `StoreBackedProvider.get_bars` output must satisfy
  `validate_bars` (it is called on the read path).
- Research/backtest layer: the synthetic fixture `DataProvider` emits this same shape, so
  swapping in a real store-backed snapshot is a no-op at the seam.
- **Streamed bulk imports** (`data import-bars`) store the consolidated `bars.parquet` in
  **symbol-major `(symbol, timestamp)`** order to keep ingest RAM bounded. This is contract-safe
  because `read_bars` re-sorts to canonical `(timestamp, symbol)` via `to_bar_schema` on read — the
  contract governs returned order, not on-disk layout. A future scaled read path (#130) must treat
  these datasets as symbol-major (per-symbol pruning / streaming merge), not time-major.

## Change control
This schema and `algua/contracts/types.py::DataProvider` are the **authoritative interface**
between the data layer and the research/backtest layer. Changing either requires updating this
doc **and** `validate_bars` together, plus the consumers, in the same change — keep the three in
sync. The backtest engine must continue to depend only on the `DataProvider` protocol (it never
imports `algua.data`; import-linter enforces this).
