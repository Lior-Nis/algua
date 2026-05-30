# Data Contract — `get_bars` Bar Schema (FROZEN)

**Status:** Frozen integration contract. **Date:** 2026-05-30.

## Why this exists
This is the **boundary between two parallel work lanes**:
- **Data lane (Codex):** implements `DataProvider` adapters (Alpaca, yfinance) + the `get_bars`
  read API over the point-in-time snapshot store.
- **Research lane (Claude):** builds `features/`, `strategies/`, and the `backtest/` engine,
  consuming bars through this exact shape (via a fixture provider until the real one lands).

Because the two lanes are built independently, the **shape of the data that crosses this seam
must be pinned**. Both lanes build to this document; integration is then mechanical.

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

### `timeframe` vocabulary
- `"1d"` — daily session bars. **Required first; the research lane targets `"1d"` initially.**
- `"1h"`, `"15m"`, `"1m"` — intraday, UTC-aligned bar boundaries (reserved; build later).
- Any other value → `ValueError`.

### Adjustment semantics (the one judgment call — flag if you disagree)
`close` is **raw**; `adj_close` is **split + dividend adjusted**. Backtests compute returns from
`adj_close`; `close` is kept for reference and notional/position sizing. This mirrors common
vendor output (yfinance `Adj Close`, Alpaca `adjustment=all`) and avoids split-induced phantom
gaps in return series. Raw OHLC is preserved so corporate-action handling stays auditable.

### Point-in-time / `as_of` (coordination note)
The snapshot store already records `as_of` provenance. The `get_bars` signature above has **no
`as_of` parameter yet** — for now it returns the latest snapshot's bars. If/when we need
as-of-date reads (for leak-free walk-forward across data revisions), we add `as_of` to the
signature **by agreement of both lanes** (see Change control). Until then, the research lane must
not assume as-of reads exist.

## Conformance
- A shared canonical fixture/validator (`validate_bars(df)`) will encode this schema so both
  lanes test against the identical definition rather than two drifting interpretations. It is a
  small follow-up; until it exists, this document is authoritative.
- Data lane: `get_bars` output must satisfy this shape exactly.
- Research lane: the fixture `DataProvider` used in backtest tests must emit this shape, so
  swapping in the real provider is a no-op at the seam.

## Change control
This schema and `algua/contracts/types.py::DataProvider` are **shared, frozen interfaces**.
Changing either requires agreement from **both** lanes: update this doc, and coordinate before
editing `contracts/types.py`. Neither lane edits the other's modules (`algua/data/*` vs
`algua/strategies|features|backtest|tracking/*`); they meet only at this contract.
