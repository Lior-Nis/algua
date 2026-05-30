# Data Integration (Real Bars into Backtests) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a real, point-in-time backtest end to end: ingest bars → `algua backtest run <strategy> --snapshot <id>` reads the stored snapshot through the bar-schema and produces metrics stamped with the snapshot id.

**Architecture:** Add the missing **serve/read layer** in the data lane — a `validate_bars` conformance check + `to_bar_schema` reshaper, a `DataStore.read_bars(snapshot_id)` that returns bar-schema data, and a `StoreBackedProvider` implementing the shared `DataProvider` protocol over a snapshot. Wire it into `backtest run --snapshot`. The engine stays decoupled (only knows `DataProvider`). Daily bar timestamps are session **dates** at tz-aware UTC midnight (revising the earlier session-close choice).

**Tech Stack:** Python 3.12, pandas 2.3.3, the existing `algua/data` store (parquet snapshots + manifest), Typer CLI, pytest.

**Key existing APIs (already on `main`):**
- `algua/data/store.py::DataStore(data_dir)` — `ingest_bars(*, provider, symbols, start, end, as_of, source, frame, timeframe="1d", adjustment="auto", source_metadata=None) -> SnapshotRecord`; `get_snapshot(id) -> SnapshotRecord` (raises `SnapshotNotFound`); `list_snapshots(dataset=None)`.
- `SnapshotRecord`: `.snapshot_id`, `.data_path` (Path, relative to `data_dir`), `.dataset` (property), `.symbols`. Parquet is written with `index=False`; a bars frame has columns `ts, symbol, open, high, low, close, adj_close, volume` (stored alphabetically sorted).
- `algua/contracts/types.py::DataProvider` protocol: `get_bars(symbols, start, end, timeframe) -> pd.DataFrame`.
- `algua/backtest/engine.py::run(strategy, provider, start, end, *, seed=None)`; `BacktestResult` has `seed` and `snapshot_id` fields (both default `None`).
- `algua/config/settings.py::get_settings().data_dir`.

---

### Task 1: Undo session-close; daily timestamps = session date (UTC midnight)

**Files:**
- Modify: `docs/contracts/bar-schema.md` (timestamp semantics)
- Modify: `algua/calendar/market_calendar.py` (remove `session_closes`)
- Modify: `tests/test_calendar.py` (remove its test)
- Modify: `algua/backtest/_sample.py` (emit session dates, not closes)
- Modify: `tests/test_backtest_sample.py` (assert UTC midnight)

- [ ] **Step 1: Update `docs/contracts/bar-schema.md`**

Find the index description (the row/line stating the daily timestamp is the session close) and replace the daily semantics with:

```markdown
- **Index:** a name=`timestamp`, **tz-aware** `DatetimeIndex` in **UTC**, monotonic non-decreasing.
  For daily (`1d`) bars the timestamp is the **session date at UTC midnight** (e.g.
  `2024-07-01 00:00:00+00:00`), matching what real daily sources (yfinance/Alpaca daily) provide.
  Intraday timeframes carry the bar's time-of-day. The `t→t+1` rule (engine shift) — not the
  timestamp's time-of-day — is what guarantees no look-ahead.
```

- [ ] **Step 2: Remove `session_closes` from `algua/calendar/market_calendar.py`**

Delete the entire `session_closes` method (the method added earlier returning session-close timestamps). Leave `is_session`, `next_session`, `previous_session`, `sessions_in_range` unchanged.

- [ ] **Step 3: Remove its test from `tests/test_calendar.py`**

Delete the `test_session_closes_are_utc_close_times` function entirely.

- [ ] **Step 4: Update `algua/backtest/_sample.py` to emit session dates**

Replace the `sessions = ...` line (currently using `MarketCalendar(...).session_closes(...)`) with calendar **session dates** at UTC midnight:

```python
        # Daily bars are timestamped at the session date (tz-aware UTC midnight), per the bar
        # schema and what real daily sources provide. Calendar-based so holidays are skipped.
        session_dates = MarketCalendar("XNYS").sessions_in_range(start.date(), end.date())
        sessions = pd.DatetimeIndex(
            [pd.Timestamp(d, tz="UTC") for d in session_dates], name="timestamp"
        )
```

(Keep the `from algua.calendar.market_calendar import MarketCalendar` import.)

- [ ] **Step 5: Fix the timestamp test in `tests/test_backtest_sample.py`**

Replace `test_timestamps_are_session_closes_not_midnight` with:

```python
def test_timestamps_are_utc_session_dates():
    df = SyntheticProvider(seed=1).get_bars(["AAA"], START, END, "1d")
    # daily bars are session dates at UTC midnight
    assert str(df.index.tz) == "UTC"
    assert (df.index.hour == 0).all()
    assert (df.index.normalize() == df.index).all()  # exactly midnight
```

- [ ] **Step 6: Run tests + gate**

Run: `uv run pytest tests/test_backtest_sample.py tests/test_calendar.py -q && uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass; ruff/mypy clean; 4 import contracts kept.

- [ ] **Step 7: Commit**

```bash
git add docs/contracts/bar-schema.md algua/calendar/market_calendar.py tests/test_calendar.py algua/backtest/_sample.py tests/test_backtest_sample.py
git commit -m "refactor: daily bar timestamps are session dates (UTC midnight); drop session_closes"
```

---

### Task 2: `validate_bars` + `to_bar_schema` (the conformance check & reshaper)

**Files:**
- Create: `algua/data/schema.py`
- Test: `tests/test_data_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_schema.py
import pandas as pd
import pytest
from algua.data.schema import BAR_COLUMNS, validate_bars, to_bar_schema


def _good() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2024-07-01", "2024-07-01", "2024-07-02"], utc=True), name="timestamp"
    )
    return pd.DataFrame(
        {"symbol": ["AAA", "BBB", "AAA"], "open": [1.0, 2.0, 1.1], "high": [1.0, 2.0, 1.1],
         "low": [1.0, 2.0, 1.1], "close": [1.0, 2.0, 1.1], "adj_close": [1.0, 2.0, 1.1],
         "volume": [10.0, 20.0, 11.0]},
        index=idx,
    )[BAR_COLUMNS.__iter__() and BAR_COLUMNS]


def test_validate_accepts_conformant_frame():
    df = _good()
    assert validate_bars(df) is df


def test_validate_rejects_missing_column():
    with pytest.raises(ValueError):
        validate_bars(_good().drop(columns=["adj_close"]))


def test_validate_rejects_tz_naive_index():
    df = _good()
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError):
        validate_bars(df)


def test_validate_rejects_wrong_index_name():
    df = _good()
    df.index = df.index.rename("ts")
    with pytest.raises(ValueError):
        validate_bars(df)


def test_validate_rejects_nan_ohlc():
    df = _good()
    df.loc[df.index[0], "close"] = float("nan")
    with pytest.raises(ValueError):
        validate_bars(df)


def test_validate_rejects_unsorted():
    df = _good().iloc[::-1]  # reverse -> not sorted by (timestamp, symbol)
    with pytest.raises(ValueError):
        validate_bars(df)


def test_to_bar_schema_reshapes_ts_column_frame():
    # stored frame: 'ts' column (string), columns in arbitrary (alphabetical) order, no index
    raw = pd.DataFrame(
        {"adj_close": [1.0, 2.0], "close": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
         "open": [1.0, 2.0], "symbol": ["BBB", "AAA"], "ts": ["2024-07-01", "2024-07-01"],
         "volume": [20.0, 10.0]}
    )
    out = to_bar_schema(raw)
    assert list(out.columns) == BAR_COLUMNS
    assert out.index.name == "timestamp"
    assert str(out.index.tz) == "UTC"
    assert list(out["symbol"]) == ["AAA", "BBB"]  # sorted by (timestamp, symbol)
    validate_bars(out)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.data.schema'`.

- [ ] **Step 3: Write the implementation**

```python
# algua/data/schema.py
from __future__ import annotations

import pandas as pd

BAR_COLUMNS = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]
_OHLC = ["open", "high", "low", "close", "adj_close"]


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Assert `df` matches the frozen bar schema; return it unchanged on success.

    Raises ValueError describing the first violation.
    """
    if df.index.name != "timestamp":
        raise ValueError(f"bars index must be named 'timestamp', got {df.index.name!r}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("bars index must be a tz-aware UTC DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("bars index must be monotonic non-decreasing")
    if list(df.columns) != BAR_COLUMNS:
        raise ValueError(f"bars columns must be {BAR_COLUMNS}, got {list(df.columns)}")
    if df[_OHLC].isna().any().any():
        raise ValueError("bars OHLC/adj_close must not contain NaN")
    keys = df.reset_index()[["timestamp", "symbol"]]
    if not keys.equals(keys.sort_values(["timestamp", "symbol"]).reset_index(drop=True)):
        raise ValueError("bars must be sorted by (timestamp, symbol)")
    return df


def to_bar_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape a stored bars frame (a `ts` column + OHLCV + symbol, any column order) into the
    bar schema: tz-aware UTC `timestamp` index, ordered columns, sorted, validated."""
    out = frame.copy()
    if "ts" in out.columns:
        out = out.rename(columns={"ts": "timestamp"})
    if "timestamp" not in out.columns:
        raise ValueError("frame must have a 'ts' or 'timestamp' column")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    missing = [c for c in BAR_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"frame missing bar columns: {missing}")
    out = out[["timestamp", *BAR_COLUMNS]]
    out = out.sort_values(["timestamp", "symbol"]).set_index("timestamp")
    return validate_bars(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_schema.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/data/schema.py tests/test_data_schema.py
git commit -m "feat: add bar-schema validator and reshaper"
```

---

### Task 3: `DataStore.read_bars(snapshot_id)`

**Files:**
- Modify: `algua/data/store.py` (add `read_bars` method + imports)
- Test: `tests/test_data_read_bars.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_read_bars.py
import pandas as pd
import pytest
from algua.data.store import DataStore, SnapshotNotFound
from algua.data.schema import BAR_COLUMNS, validate_bars


def _ingest(store: DataStore):
    frame = pd.DataFrame({
        "ts": ["2024-07-01", "2024-07-01", "2024-07-02", "2024-07-02"],
        "symbol": ["AAA", "BBB", "AAA", "BBB"],
        "open": [10.0, 20.0, 11.0, 21.0], "high": [10.0, 20.0, 11.0, 21.0],
        "low": [10.0, 20.0, 11.0, 21.0], "close": [10.0, 20.0, 11.0, 21.0],
        "adj_close": [10.0, 20.0, 11.0, 21.0], "volume": [100.0, 200.0, 110.0, 210.0],
    })
    return store.ingest_bars(
        provider="test", symbols=["AAA", "BBB"], start="2024-07-01", end="2024-07-02",
        as_of="2024-07-03", source="unit-test", frame=frame, timeframe="1d", adjustment="none",
    )


def test_read_bars_returns_bar_schema(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)
    out = store.read_bars(rec.snapshot_id)
    validate_bars(out)
    assert list(out.columns) == BAR_COLUMNS
    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2024-07-01", tz="UTC")
    assert len(out) == 4


def test_read_bars_unknown_id_raises(tmp_path):
    with pytest.raises(SnapshotNotFound):
        DataStore(tmp_path).read_bars("does-not-exist")


def test_read_bars_rejects_non_bars_dataset(tmp_path):
    store = DataStore(tmp_path)
    csv = tmp_path / "u.csv"
    csv.write_text("symbol\nAAA\n")
    rec = store.ingest_file(
        dataset="universe", provider="test", symbols=["AAA"], start="2024-07-01",
        end="2024-07-02", as_of="2024-07-03", source="unit-test", path=csv,
    )
    with pytest.raises(ValueError):
        store.read_bars(rec.snapshot_id)
```

Note: confirm `ingest_file`'s exact keyword signature by reading `algua/data/store.py` before writing this test; if a parameter name differs, adjust the `ingest_file(...)` call (its purpose here is only to create a non-`bars` snapshot). If `ingest_file` is awkward to call, instead assert the non-bars guard by constructing the situation through `ingest_universe` (whichever is simplest) — the requirement is just that `read_bars` rejects a dataset that isn't `bars`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_read_bars.py -v`
Expected: FAIL with `AttributeError: 'DataStore' object has no attribute 'read_bars'`.

- [ ] **Step 3: Add `read_bars` to `algua/data/store.py`**

Add this import near the top (with the other `algua.data` imports):

```python
from algua.data.schema import to_bar_schema
```

Add this method to the `DataStore` class (e.g. right after `get_snapshot`):

```python
    def read_bars(self, snapshot_id: str) -> pd.DataFrame:
        """Read a bars snapshot back as a bar-schema DataFrame (tz-aware UTC timestamp index)."""
        rec = self.get_snapshot(snapshot_id)  # raises SnapshotNotFound
        if rec.dataset != "bars":
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset!r}, not 'bars'"
            )
        frame = pd.read_parquet(self.data_dir / rec.data_path)
        return to_bar_schema(frame)
```

(`pandas` is already imported in `store.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_read_bars.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/data/store.py tests/test_data_read_bars.py
git commit -m "feat: add DataStore.read_bars returning bar-schema data"
```

---

### Task 4: `StoreBackedProvider` (a `DataProvider` over a snapshot)

**Files:**
- Create: `algua/data/serve.py`
- Test: `tests/test_data_serve.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_serve.py
from datetime import UTC, datetime

import pandas as pd
from algua.contracts.types import DataProvider
from algua.data.schema import BAR_COLUMNS, validate_bars
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore


def _ingest(store: DataStore):
    frame = pd.DataFrame({
        "ts": ["2024-07-01", "2024-07-01", "2024-07-02", "2024-07-02", "2024-07-03"],
        "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA"],
        "open": [10.0, 20.0, 11.0, 21.0, 12.0], "high": [10.0, 20.0, 11.0, 21.0, 12.0],
        "low": [10.0, 20.0, 11.0, 21.0, 12.0], "close": [10.0, 20.0, 11.0, 21.0, 12.0],
        "adj_close": [10.0, 20.0, 11.0, 21.0, 12.0], "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    return store.ingest_bars(
        provider="test", symbols=["AAA", "BBB"], start="2024-07-01", end="2024-07-03",
        as_of="2024-07-04", source="unit-test", frame=frame, timeframe="1d", adjustment="none",
    )


def test_provider_satisfies_protocol_and_filters(tmp_path):
    store = DataStore(tmp_path)
    rec = _ingest(store)
    provider = StoreBackedProvider(store, rec.snapshot_id)
    assert isinstance(provider, DataProvider)
    assert provider.snapshot_id == rec.snapshot_id

    out = provider.get_bars(
        ["AAA"], datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 7, 2, tzinfo=UTC), "1d"
    )
    validate_bars(out)
    assert list(out.columns) == BAR_COLUMNS
    assert set(out["symbol"]) == {"AAA"}                    # symbol filter
    assert out.index.max() == pd.Timestamp("2024-07-02", tz="UTC")  # window filter (end inclusive)
    assert out.index.min() == pd.Timestamp("2024-07-01", tz="UTC")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_serve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.data.serve'`.

- [ ] **Step 3: Write the implementation**

```python
# algua/data/serve.py
from __future__ import annotations

from datetime import datetime

import pandas as pd

from algua.data.store import DataStore


class StoreBackedProvider:
    """Serves a single ingested bars snapshot through the DataProvider protocol.

    Point-in-time and reproducible: a backtest run against this provider is pinned to exactly
    one snapshot, whose id is exposed for stamping into the result.
    """

    def __init__(self, store: DataStore, snapshot_id: str) -> None:
        self.store = store
        self.snapshot_id = snapshot_id

    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        bars = self.store.read_bars(self.snapshot_id)  # bar-schema, validated
        bars = bars[bars["symbol"].isin(set(symbols))]
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        return bars[(bars.index >= start_ts) & (bars.index <= end_ts)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_serve.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/data/serve.py tests/test_data_serve.py
git commit -m "feat: add StoreBackedProvider (DataProvider over a snapshot)"
```

---

### Task 5: Engine stamps the snapshot id

**Files:**
- Modify: `algua/backtest/engine.py` (one line in `run`)
- Test: `tests/test_backtest_engine.py` (append one test)

- [ ] **Step 1: Append the failing test to `tests/test_backtest_engine.py`**

```python
def test_run_stamps_snapshot_id_when_provider_exposes_it():
    class StampedProvider:
        snapshot_id = "snap-123"

        def get_bars(self, symbols, start, end, timeframe):
            return SyntheticProvider(seed=1).get_bars(symbols, start, end, timeframe)

    cfg = StrategyConfig(
        name="ew", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1), params={},
    )
    strat = LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series(
        1.0 / len(v["symbol"].unique()), index=sorted(v["symbol"].unique())))
    res = run(strat, StampedProvider(), START, END)
    assert res.snapshot_id == "snap-123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_engine.py::test_run_stamps_snapshot_id_when_provider_exposes_it -v`
Expected: FAIL — `res.snapshot_id` is `None` (engine doesn't read it yet).

- [ ] **Step 3: Edit `algua/backtest/engine.py`**

In `run`, the `BacktestResult(...)` construction currently ends with:

```python
        seed=getattr(provider, "seed", seed),
    )
```

Change it to also stamp the snapshot id:

```python
        seed=getattr(provider, "seed", seed),
        snapshot_id=getattr(provider, "snapshot_id", None),
    )
```

(This uses `getattr`, so the engine still does not import `algua.data` — the import contract stays intact.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backtest_engine.py -q`
Expected: PASS (all engine tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: stamp provider snapshot_id into BacktestResult"
```

---

### Task 6: CLI `backtest run --snapshot`

**Files:**
- Modify: `algua/cli/backtest_cmd.py`
- Test: `tests/test_cli_backtest.py` (append)

- [ ] **Step 1: Append the failing tests to `tests/test_cli_backtest.py`**

```python
def _ingest_snapshot(tmp_path):
    """Ingest synthetic momentum-universe bars and return (snapshot_id)."""
    from datetime import UTC, datetime
    from algua.backtest._sample import SyntheticProvider
    from algua.data.store import DataStore

    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    start, end = datetime(2022, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)
    bars = SyntheticProvider(seed=0).get_bars(symbols, start, end, "1d")
    frame = bars.reset_index().rename(columns={"timestamp": "ts"})
    rec = DataStore(tmp_path).ingest_bars(
        provider="synthetic", symbols=symbols, start="2022-01-01", end="2023-12-31",
        as_of="2024-01-01", source="test", frame=frame, timeframe="1d", adjustment="none",
    )
    return rec.snapshot_id


def test_backtest_run_on_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--snapshot", snap, "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["snapshot_id"] == snap
    assert "sharpe" in payload["metrics"]


def test_backtest_run_requires_a_data_source(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum"])  # neither flag
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_backtest_run_rejects_both_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--demo", "--snapshot", "x"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
```

(The existing `test_backtest_run_demo_emits_metrics` etc. keep `--demo` and must still pass.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_backtest.py -v`
Expected: the three new tests fail (no `--snapshot` option; current code requires `--demo`).

- [ ] **Step 3: Edit `algua/cli/backtest_cmd.py`**

Add an import for the data-lane serve layer near the other imports:

```python
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
```

Replace the `run` command's options + provider-selection block. The command signature gains a `--snapshot` option, and the body replaces the current `if not demo: raise ValueError(...)` with explicit source selection:

```python
@backtest_app.command("run")
@json_errors()
def run(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    register: bool = typer.Option(False, "--register", help="advance registry idea->backtested"),
) -> None:
    """Backtest a strategy and emit metrics JSON."""
    strategy = load_strategy(name)
    if demo and snapshot:
        raise ValueError("pass only one of --demo or --snapshot")
    if demo:
        provider = SyntheticProvider(seed=0)
    elif snapshot:
        provider = StoreBackedProvider(DataStore(get_settings().data_dir), snapshot)
    else:
        raise ValueError("pass one of --demo (synthetic) or --snapshot <id> (real data)")
    result = run_backtest(strategy, provider, _utc(start), _utc(end))

    if register:
        with closing(connect(get_settings().db_path)) as conn:
            migrate(conn)
            existing = {s.name for s in store.list_strategies(conn)}
            if name not in existing:
                store.add_strategy(conn, name)
            reason = (
                f"backtest sharpe={result.metrics['sharpe']:.2f} "
                f"ret={result.metrics['total_return']:.2%}"
            )
            # code_hash == config_hash for now; real source-code hashing comes before the live gate.
            store.transition(conn, name, Stage.BACKTESTED, Actor.AGENT, reason,
                             code_hash=result.config_hash, config_hash=result.config_hash)

    emit(result.to_dict())
```

(Keep the existing imports: `SyntheticProvider`, `run as run_backtest`, `get_settings`, `closing`, `connect`/`migrate`, `store`, `Stage`/`Actor`, `_utc`. `StoreBackedProvider` raising `SnapshotNotFound`/`ValueError` is rendered as JSON by `@json_errors()`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_backtest.py -q`
Expected: PASS (existing `--demo` tests + the three new ones).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/backtest_cmd.py tests/test_cli_backtest.py
git commit -m "feat: backtest run --snapshot for real point-in-time data"
```

---

### Task 7: Full verification & end-to-end smoke

**Files:** none (verification only)

- [ ] **Step 1: Full quality gate**

Run:
```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```
Expected: all tests pass; ruff clean; mypy `Success`; import-linter `4 kept, 0 broken` (the engine still must not import `algua.data` — `serve.py`/`schema.py`/`store.py` are in the data lane; the CLI is allowed to import them).

- [ ] **Step 2: End-to-end CLI smoke (real ingested data)**

Run:
```bash
export ALGUA_DATA_DIR="$(mktemp -d)/data"
export ALGUA_DB_PATH="$ALGUA_DATA_DIR/r.db"
uv run algua data ingest-bars --provider yfinance --symbols AAPL,MSFT,NVDA,AMZN,GOOGL --start 2022-01-01 --end 2023-12-31
# capture the snapshot id from the JSON output, then:
SNAP=$(uv run algua data inspect --dataset bars | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[-1]['snapshot_id'] if isinstance(d,list) else d['snapshots'][-1]['snapshot_id'])")
uv run algua backtest run cross_sectional_momentum --snapshot "$SNAP" --start 2022-01-01 --end 2023-12-31 --register
uv run algua registry show cross_sectional_momentum
```
Expected: `backtest run` emits metrics JSON with `"snapshot_id"` set to `$SNAP`; `registry show` reports stage `backtested`.

If `yfinance` network access is unavailable in this environment, substitute the smoke test by ingesting synthetic bars in a short Python snippet (mirroring `_ingest_snapshot` from Task 6) and running `backtest run --snapshot <id>` against it. Note in your report which path you used. The exact flag names for `data ingest-bars` / `data inspect` come from `algua/cli/data_cmd.py`; confirm them there before running and adjust if needed.

- [ ] **Step 3: Final commit (if any verification fixes were needed)**

```bash
git add -A
git commit -m "test: verify data integration end to end" --allow-empty
```

---

## Self-Review Notes

- **Spec coverage:** bar-schema update + undo (Task 1), `validate_bars`/`to_bar_schema` (Task 2),
  `DataStore.read_bars` (Task 3), `StoreBackedProvider` (Task 4), engine snapshot stamp (Task 5),
  CLI `--snapshot` (Task 6), verification + e2e (Task 7). Out-of-scope items (snapshot
  auto-selection, intraday, as-of reads, live fetch from `backtest run`) are intentionally absent.
- **Boundary:** new runtime code lives in `algua/data/*` (schema, serve, store) and the CLI; the
  engine change is a `getattr` only, so `backtest` still never imports `algua.data` (lint-imports
  stays green). `validate_bars` is imported by `store`/`serve` (data lane) and by tests — never by
  `algua/backtest`.
- **Type consistency:** `BAR_COLUMNS`, `validate_bars(df)->df`, `to_bar_schema(frame)->df`,
  `DataStore.read_bars(id)->df`, `StoreBackedProvider(store, snapshot_id)` with `.snapshot_id` and
  `get_bars(symbols,start,end,timeframe)`, and `BacktestResult.snapshot_id` are used consistently.
- **External caution:** Task 3/7 note confirming `ingest_file`/`data ingest-bars`/`data inspect`
  exact signatures from the existing code before relying on them.
