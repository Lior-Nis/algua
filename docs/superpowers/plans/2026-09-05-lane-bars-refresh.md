# Lane Bars Refresh (#556) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The paper/live daily tick refreshes its own bars (resolve-or-ingest behind a per-symbol coverage wall) instead of trading against a static snapshot named by an env var, and decision-data staleness becomes a fleet-health verdict.

**Architecture:** A pure data-layer primitive `algua.data.refresh.refresh_bars` (lease → resolve+re-validate → fetch → canonicalize → clip → coverage wall → ingest) is consumed by a cli-layer orchestrator `algua.cli.lane_refresh` (cycle plan with per-symbol history floors + lane symbol set) that both `paper run-all --refresh` and `live run-all --refresh` call after broker connect and fill ingest, before the account reconcile. Tick rows persist the `snapshot_id` they traded on (schema v45); `fleet_health` derives a `now`-relative `decision_stale_sessions`; the operator job template drops the `{snapshot}` placeholder and requires snapshot provenance to mark a session complete.

**Tech Stack:** Python 3.12, typer CLI, pandas, SQLite registry, `algua.primitives.flock`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-lane-bars-refresh-design.md`

## Global Constraints

- Quality gate at every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. Run the targeted test file first, the full suite before each commit.
- Layering: `algua.execution` / `algua.live` / `algua.backtest` / `algua.strategies` / `algua.contracts` / `algua.features` must NOT import `algua.data` (AST-enforced by `tests/test_data_wall.py`). `algua.cli` composes everything. `algua.calendar` must not import `cli`/`registry`.
- Snapshots are immutable; a refresh is always a NEW snapshot id via `DataStore.ingest_bars`.
- Fail closed: a refresh failure aborts the cycle (`ok:false`, exit 1); never fall back to an older snapshot.
- `tests/test_module_size_ratchet.py` is a shrink-only ratchet: put new code in NEW modules; if `paper_cmd.py` / `live_cmd.py` outgrow their pins from the wiring below, trim duplicated docstring prose in the same file — never raise a pin.
- `tests/test_lane_parity.py` reads the tick helpers' source: `_run_paper_strategy_tick` / `_run_strategy_tick` must keep calling `resolve_operational_universe`, `trip_for_breach`, `engage`. This plan does not touch those call sites.
- CODEOWNERS-protected paths touched (human merge required, do NOT auto-merge the PR): `algua/cli/paper_cmd.py`, `algua/registry/db/`.
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01EruxEBNzfSnaJjC8wtBZqN`.
- Never `git add -A`; add the exact files each task touches.

---

### Task 1: `algua.data.refresh.refresh_bars` — resolve-or-ingest behind a coverage wall

**Files:**
- Create: `algua/data/refresh.py`
- Test: `tests/test_data_refresh.py`

**Interfaces:**
- Consumes: `DataStore(data_dir)` — `.data_dir`, `.list_snapshots(Dataset.BARS)`, `.read_bars(snapshot_id)` (bar-schema frame: tz-aware `timestamp` index + `symbol` + OHLCV), `.ingest_bars(...)`; `normalize_symbols` from `algua.data.store`; `to_bar_schema(frame, *, timeframe)` from `algua.data.schema`; `logical_bars_hash(canon)` from `algua.data.files` (canon = bar-schema frame `.reset_index().rename(columns={"timestamp": "ts"})`, exactly as `store/bars.py` builds it at ingest); `BarProvider` (`.name`, `.get_bars(BarRequest) -> ProviderBars(frame, source_metadata)`) from `algua.data.contracts`; `algua.primitives.flock.acquire(path, verify_inode=True)` / `release(fd)`.
- Produces: `refresh_bars(store, provider, *, symbols, start, end, require_bar_on, min_rows=None, timeframe="1d", adjustment="none") -> tuple[SnapshotRecord, bool]`; `class RefreshError(RuntimeError)` with `kind` (`"missing"|"stale"|"misdated"|"short_history"`), `symbols: list[str]`, `require_bar_on: str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_refresh.py
"""refresh_bars (#556): resolve-or-ingest behind a per-symbol coverage wall."""
from __future__ import annotations

import pandas as pd
import pytest

from algua.data.contracts import ProviderBars
from algua.data.models import Dataset
from algua.data.refresh import RefreshError, refresh_bars
from algua.data.store import DataStore

_START, _END, _REQ = "2026-01-05", "2026-01-09", "2026-01-08"


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows = [(symbol, iso_date), ...] -> one bar per row, daily UTC-midnight stamps."""
    n = len(rows)
    return pd.DataFrame({
        "ts": [f"{d}T00:00:00+00:00" for _, d in rows],
        "symbol": [s for s, _ in rows],
        "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n, "close": [1.0] * n,
        "adj_close": [1.0] * n, "volume": [1.0] * n,
    })


class _Provider:
    name = "fake"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame, self.calls = frame, 0

    def get_bars(self, request):
        self.calls += 1
        return ProviderBars(frame=self.frame.copy(), source_metadata={"provider": "fake"})


_FULL = _frame([("AAPL", "2026-01-07"), ("AAPL", "2026-01-08"),
                ("MSFT", "2026-01-07"), ("MSFT", "2026-01-08")])


def _call(store, provider, **kw):
    args = dict(symbols=["msft", "AAPL"], start=_START, end=_END, require_bar_on=_REQ)
    args.update(kw)
    return refresh_bars(store, provider, **args)


def test_miss_ingests_and_reports_refreshed(tmp_path):
    store, provider = DataStore(tmp_path), _Provider(_FULL)
    rec, refreshed = _call(store, provider)
    assert refreshed is True and provider.calls == 1
    assert rec.dataset is Dataset.BARS and tuple(rec.symbols) == ("AAPL", "MSFT")
    assert rec.provider == "fake"
    assert (rec.metadata.start, rec.metadata.end) == (_START, _END)


def test_hit_reuses_without_provider_call(tmp_path):
    store, provider = DataStore(tmp_path), _Provider(_FULL)
    first, _ = _call(store, provider)
    second, refreshed = _call(store, provider, symbols=["AAPL", "MSFT"])  # order-insensitive
    assert refreshed is False and provider.calls == 1
    assert second.snapshot_id == first.snapshot_id


def test_different_end_is_a_miss(tmp_path):
    store, provider = DataStore(tmp_path), _Provider(_FULL)
    first, _ = _call(store, provider)
    second, refreshed = _call(store, provider, end="2026-01-10", require_bar_on="2026-01-08")
    assert refreshed is True and second.snapshot_id != first.snapshot_id


def test_same_key_candidate_failing_current_coverage_is_not_reused(tmp_path):
    """A lax earlier refresh (accepted for 01-07) under the SAME key must not satisfy a later
    01-08 requirement: the candidate is re-validated and a fresh snapshot minted."""
    store = DataStore(tmp_path)
    lax = _Provider(_frame([("AAPL", "2026-01-07"), ("MSFT", "2026-01-07")]))
    first, _ = _call(store, lax, require_bar_on="2026-01-07")
    fresh = _Provider(_FULL)
    second, refreshed = _call(store, fresh, require_bar_on="2026-01-08")
    assert refreshed is True and fresh.calls == 1
    assert second.snapshot_id != first.snapshot_id


def test_same_key_candidate_with_short_history_is_not_reused(tmp_path):
    store = DataStore(tmp_path)
    first, _ = _call(store, _Provider(_FULL))  # 2 rows per symbol
    fresh = _Provider(_frame([("AAPL", "2026-01-06"), ("AAPL", "2026-01-07"),
                              ("AAPL", "2026-01-08"), ("MSFT", "2026-01-06"),
                              ("MSFT", "2026-01-07"), ("MSFT", "2026-01-08")]))
    second, refreshed = _call(store, fresh, min_rows={"AAPL": 3, "MSFT": 3})
    assert refreshed is True and second.snapshot_id != first.snapshot_id


def test_tampered_candidate_with_equal_row_count_is_not_reused(tmp_path):
    store, provider = DataStore(tmp_path), _Provider(_FULL)
    first, _ = _call(store, provider)
    # Rewrite every partition with the same row count but different closes.
    for p in first.data_path.rglob("*.parquet") if first.data_path.is_absolute() else \
            (tmp_path / first.data_path).rglob("*.parquet"):
        df = pd.read_parquet(p)
        df["close"] = df["close"] + 1.0
        df.to_parquet(p, index=False)
    second, refreshed = _call(store, provider)
    assert refreshed is True and second.snapshot_id != first.snapshot_id


def test_missing_symbol_fails_closed_and_mints_nothing(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(RefreshError) as exc:
        _call(store, _Provider(_frame([("AAPL", "2026-01-08")])))
    assert exc.value.kind == "missing" and exc.value.symbols == ["MSFT"]
    assert store.list_snapshots(Dataset.BARS) == []


def test_stale_symbol_fails_closed_and_mints_nothing(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(RefreshError) as exc:
        _call(store, _Provider(_frame([("AAPL", "2026-01-08"), ("MSFT", "2026-01-07")])))
    assert exc.value.kind == "stale" and exc.value.symbols == ["MSFT"]
    assert store.list_snapshots(Dataset.BARS) == []


def test_misdated_newer_bar_fails_closed(tmp_path):
    # A row dated AFTER require_bar_on but before end (a weekend row from a bad vendor).
    store = DataStore(tmp_path)
    with pytest.raises(RefreshError) as exc:
        _call(store, _Provider(_frame([("AAPL", "2026-01-08"), ("MSFT", "2026-01-08"),
                                       ("MSFT", "2026-01-08T12")])),
              require_bar_on="2026-01-08", end="2026-01-10", timeframe="1d")
    assert exc.value.kind in {"misdated", "stale"}  # to_bar_schema may reject the intraday stamp first
    assert store.list_snapshots(Dataset.BARS) == []


def test_short_history_fails_closed(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(RefreshError) as exc:
        _call(store, _Provider(_FULL), min_rows={"AAPL": 5})
    assert exc.value.kind == "short_history" and exc.value.symbols == ["AAPL"]
    assert store.list_snapshots(Dataset.BARS) == []


def test_rows_at_or_after_end_are_clipped(tmp_path):
    store = DataStore(tmp_path)
    provider = _Provider(_frame([("AAPL", "2026-01-08"), ("AAPL", "2026-01-09"),
                                 ("MSFT", "2026-01-08"), ("MSFT", "2026-01-09")]))
    rec, _ = _call(store, provider)
    assert store.read_bars(rec.snapshot_id).index.max().date().isoformat() == "2026-01-08"


def test_provider_error_propagates_and_mints_nothing(tmp_path):
    class _Boom:
        name = "boom"

        def get_bars(self, request):
            raise ConnectionError("vendor down")

    store = DataStore(tmp_path)
    with pytest.raises(ConnectionError):
        _call(store, _Boom())
    assert store.list_snapshots(Dataset.BARS) == []
```

If `test_misdated_newer_bar_fails_closed` cannot construct a misdated daily row because `to_bar_schema` rejects non-midnight stamps, replace the intraday stamp with a legitimately dated later day inside the window (`("MSFT", "2026-01-09")` with `end="2026-01-10"`, `require_bar_on="2026-01-08"`) and assert `kind == "misdated"` exactly.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data_refresh.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'algua.data.refresh'`

- [ ] **Step 3: Write the implementation**

```python
# algua/data/refresh.py
"""Resolve-or-ingest bars for a lane tick (#556).

The paper/live daily tick must decide on bars that are (a) present for every symbol it needs,
(b) dated exactly the session it decides on, and (c) deep enough for each strategy's history
need. A static snapshot promises none of these, so the lane calls :func:`refresh_bars` each cycle.

ONE coverage predicate is applied both to a candidate snapshot considered for reuse and to a fresh
provider response: a same-key snapshot that no longer satisfies the CURRENT requirement (an older
accepted session, truncated history, or altered content) is never reused. The sequence
resolve -> fetch -> validate -> ingest runs under a request-keyed flock so two concurrent callers
(the 20-minute timer re-fire racing a manual cycle) cannot both miss and mint.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pandas as pd

from algua.data.contracts import BarProvider, BarRequest
from algua.data.files import logical_bars_hash
from algua.data.models import Dataset, SnapshotRecord
from algua.data.schema import to_bar_schema
from algua.data.store import DataStore, normalize_symbols
from algua.primitives import flock

__all__ = ["RefreshError", "refresh_bars"]

#: Never swept by any staging/GC sweep: a lock file here is only ever held or idle.
LOCK_DIRNAME = "refresh-locks"


class RefreshError(RuntimeError):
    """The bars cannot back a tick. ``kind``: ``"missing"`` (no rows), ``"stale"`` (newest bar
    older than ``require_bar_on``), ``"misdated"`` (newest bar NEWER than ``require_bar_on`` — a
    non-session row), ``"short_history"`` (fewer rows than the symbol's floor). Raised BEFORE any
    ingest, so nothing is minted and the next cycle re-queries the provider."""

    def __init__(self, kind: str, symbols: list[str], *, require_bar_on: str) -> None:
        self.kind = kind
        self.symbols = symbols
        self.require_bar_on = require_bar_on
        super().__init__(f"bars refresh: {kind} on {require_bar_on} for {symbols}")


def _to_canon(bars: pd.DataFrame) -> pd.DataFrame:
    """Bar-schema frame (timestamp index) -> the ``ts``-column canon the store hashes."""
    return bars.reset_index().rename(columns={"timestamp": "ts"})


def _coverage_error(
    canon: pd.DataFrame, symbols: list[str], require_bar_on: str, min_rows: Mapping[str, int],
) -> RefreshError | None:
    """The single coverage predicate (spec §1). ``canon`` has a tz-aware ``ts`` column."""
    want = pd.Timestamp(require_bar_on, tz="UTC").date()
    present: set[str] = set(canon["symbol"]) if not canon.empty else set()
    missing = [s for s in symbols if s not in present]
    if missing:
        return RefreshError("missing", missing, require_bar_on=require_bar_on)
    dates = pd.to_datetime(canon["ts"], utc=True).dt.date
    stale, misdated, short = [], [], []
    for s in symbols:
        mask = canon["symbol"] == s
        newest = dates[mask].max()
        if newest < want:
            stale.append(s)
        elif newest > want:
            misdated.append(s)
        if int(mask.sum()) < int(min_rows.get(s, 0)):
            short.append(s)
    if stale:
        return RefreshError("stale", stale, require_bar_on=require_bar_on)
    if misdated:
        return RefreshError("misdated", misdated, require_bar_on=require_bar_on)
    if short:
        return RefreshError("short_history", short, require_bar_on=require_bar_on)
    return None


def _matches(rec: SnapshotRecord, *, provider_name: str, symbols: list[str], start: str,
             end: str, timeframe: str, adjustment: str) -> bool:
    m = rec.metadata
    return (
        rec.dataset is Dataset.BARS
        and m.provider == provider_name
        and m.timeframe == timeframe
        and m.adjustment == adjustment
        and list(m.symbols) == symbols
        and m.start == start
        and m.end == end
    )


def _reusable(
    store: DataStore, rec: SnapshotRecord, symbols: list[str], require_bar_on: str,
    min_rows: Mapping[str, int],
) -> bool:
    """Re-validate a same-key candidate: content hash, then the CURRENT coverage predicate."""
    try:
        canon = _to_canon(store.read_bars(rec.snapshot_id))
    except Exception:  # noqa: BLE001 — unreadable payload: never reuse
        return False
    if logical_bars_hash(canon) != rec.content_hash:
        return False
    return _coverage_error(canon, symbols, require_bar_on, min_rows) is None


def refresh_bars(
    store: DataStore,
    provider: BarProvider,
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    require_bar_on: str,
    min_rows: Mapping[str, int] | None = None,
    timeframe: str = "1d",
    adjustment: str = "none",
) -> tuple[SnapshotRecord, bool]:
    """Return ``(snapshot, refreshed)``. ``refreshed`` is True iff the provider was queried and a
    new snapshot minted; False means the NEWEST same-key snapshot passed re-validation and was
    reused (no network). Provider identity is ``provider.name``.

    ``require_bar_on`` is the ISO date every symbol must have its newest bar on — the session the
    tick decides on. ``min_rows`` floors the in-window row count per symbol (absent = 0). The
    caller owns the calendar and the strategy contracts; this layer is free of both."""
    syms = normalize_symbols(list(symbols))
    floors: Mapping[str, int] = {k.upper(): int(v) for k, v in (min_rows or {}).items()}
    key = dict(provider_name=provider.name, symbols=syms, start=start, end=end,
               timeframe=timeframe, adjustment=adjustment)
    digest = hashlib.sha256(json.dumps(list(key.values())).encode()).hexdigest()[:24]
    lock_dir = store.data_dir / LOCK_DIRNAME
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = flock.acquire(lock_dir / f"{digest}.lock", verify_inode=True)
    try:
        newest = next((r for r in reversed(store.list_snapshots(Dataset.BARS))
                       if _matches(r, **key)), None)
        if newest is not None and _reusable(store, newest, syms, require_bar_on, floors):
            return newest, False
        result = provider.get_bars(BarRequest(symbols=tuple(syms), start=start, end=end,
                                              timeframe=timeframe, adjustment=adjustment))
        bars = to_bar_schema(result.frame, timeframe=timeframe)
        lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        bars = bars[(bars.index >= lo) & (bars.index < hi)]
        canon = _to_canon(bars)
        err = _coverage_error(canon, syms, require_bar_on, floors)
        if err is not None:
            raise err
        rec = store.ingest_bars(
            provider=provider.name, symbols=syms, start=start, end=end,
            as_of=datetime.now(UTC).isoformat(), source=provider.name, frame=canon,
            timeframe=timeframe, adjustment=adjustment,
            source_metadata=result.source_metadata,
        )
        return rec, True
    finally:
        flock.release(fd)
```

If `to_bar_schema` on an EMPTY provider frame raises, guard it: `bars = to_bar_schema(...) if not result.frame.empty else empty_bars()` (import `empty_bars` from `algua.data.schema`) so an empty response reaches the coverage wall as `missing`, not as a schema error.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data_refresh.py -q`
Expected: 13 passed.

- [ ] **Step 5: Gate and commit**

Run: `uv run ruff check algua/data/refresh.py tests/test_data_refresh.py && uv run mypy algua && uv run lint-imports && uv run pytest -q`

```bash
git add algua/data/refresh.py tests/test_data_refresh.py
git commit -m "feat(data): refresh_bars — resolve-or-ingest behind a per-symbol coverage wall (#556)"
```

---

### Task 2: `algua data refresh-bars` CLI, `bars_refresh_provider` setting, `refresh_failed` error code

**Files:**
- Modify: `algua/config/settings.py` (after `tracking_backend`, ~line 42)
- Modify: `algua/cli/errors.py` (`_registry`, RuntimeError family ~line 53)
- Modify: `algua/cli/data_cmd.py` (after the `ingest_bars` command, ~line 119)
- Test: `tests/test_cli_data.py`

**Interfaces:**
- Consumes: Task 1.
- Produces: `Settings.bars_refresh_provider: str = "yfinance"` (env `ALGUA_BARS_REFRESH_PROVIDER`); error code `"refresh_failed"` for `RefreshError`; CLI `algua data refresh-bars --symbols A,B --start D --end D --require-bar-on D [--min-rows N] [--provider P] [--timeframe 1d]` → `{"ok": true, "snapshot": {...}, "refreshed": bool}`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cli_data.py`, reusing `runner`, `_json`, `data_cmd`, `ProviderBars`, `pd` already imported there)

```python
def _fake_provider(dates: list[str]):
    class FakeProvider:
        name = "fake"

        def get_bars(self, _request):
            n = len(dates)
            return ProviderBars(
                frame=pd.DataFrame({
                    "ts": [f"{d}T00:00:00+00:00" for d in dates], "symbol": ["AAPL"] * n,
                    "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n,
                    "close": [1.0] * n, "adj_close": [1.0] * n, "volume": [1.0] * n,
                }),
                source_metadata={"provider": "fake"},
            )
    return FakeProvider()


_REFRESH_ARGS = ["data", "refresh-bars", "--provider", "fake", "--symbols", "AAPL",
                 "--start", "2026-01-05", "--end", "2026-01-09", "--require-bar-on", "2026-01-08"]


def test_refresh_bars_cli_emits_snapshot_and_refreshed_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_cmd, "_bar_provider",
                        lambda _name: _fake_provider(["2026-01-07", "2026-01-08"]))
    first = _json(runner.invoke(app, _REFRESH_ARGS))
    assert first["ok"] is True and first["refreshed"] is True
    assert first["snapshot"]["dataset"] == "bars"
    second = _json(runner.invoke(app, _REFRESH_ARGS))
    assert second["refreshed"] is False
    assert second["snapshot"]["snapshot_id"] == first["snapshot"]["snapshot_id"]


def test_refresh_bars_cli_stale_symbol_is_refresh_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_cmd, "_bar_provider", lambda _name: _fake_provider(["2026-01-07"]))
    r = runner.invoke(app, _REFRESH_ARGS)
    assert r.exit_code == 1
    out = _json(r)
    assert out["ok"] is False and out["code"] == "refresh_failed" and "stale" in out["error"]


def test_refresh_bars_cli_min_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_cmd, "_bar_provider",
                        lambda _name: _fake_provider(["2026-01-07", "2026-01-08"]))
    r = runner.invoke(app, [*_REFRESH_ARGS, "--min-rows", "5"])
    assert r.exit_code == 1 and _json(r)["code"] == "refresh_failed"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli_data.py -q -k refresh_bars`
Expected: FAIL — `No such command 'refresh-bars'`

- [ ] **Step 3: Implement**

`algua/config/settings.py`, after `tracking_backend`:
```python
    # Bars provider the PAPER lane refreshes through (`paper run-all --refresh`, #556). Any name
    # registered in algua.data.providers. env: ALGUA_BARS_REFRESH_PROVIDER.
    bars_refresh_provider: str = "yfinance"
    # Bars provider the LIVE lane refreshes through (`live run-all --refresh`). NO default on
    # purpose: real-money decision data is a human choice made explicitly, never inherited from
    # the research convenience default above. Unset => live refresh fails closed
    # (`live_refresh_provider_unset`). env: ALGUA_BARS_REFRESH_PROVIDER_LIVE.
    bars_refresh_provider_live: str | None = None
```

`algua/cli/errors.py` `_registry`: add `from algua.data.refresh import RefreshError` to the local imports and `(RefreshError, "refresh_failed"),` in the `# --- RuntimeError family ---` block (before the generic buckets).

`algua/cli/data_cmd.py`: add `from algua.data.refresh import refresh_bars` beside the other `algua.data` imports; ensure `from algua.config.settings import get_settings` is present; add after `ingest_bars`:

```python
@data_app.command("refresh-bars")
@json_errors
def refresh_bars_cmd(
    symbols: str = typer.Option(..., "--symbols", help="comma-separated symbols"),
    start: str = typer.Option(..., "--start", help="inclusive start date"),
    end: str = typer.Option(..., "--end", help="exclusive end date (half-open [start, end))"),
    require_bar_on: str = typer.Option(
        ..., "--require-bar-on",
        help="ISO date every symbol's NEWEST bar must fall on (the session a tick decides on); "
             "older = stale, newer = misdated, absent = missing — each fails closed, nothing minted"),
    min_rows: int = typer.Option(
        0, "--min-rows", help="minimum in-window rows per symbol (history floor); 0 = none"),
    provider: str = typer.Option(
        None, "--provider", help="bar provider name (default: settings.bars_refresh_provider)"),
    timeframe: str = typer.Option("1d", "--timeframe"),
) -> None:
    """Resolve-or-ingest bars for a lane tick (#556): reuse the newest same-request snapshot if
    it still passes the coverage wall, else fetch, clip, validate, and mint a new one."""
    name = provider or get_settings().bars_refresh_provider
    syms = symbols.split(",")
    rec, refreshed = refresh_bars(
        _store(), _bar_provider(name), symbols=syms, start=start, end=end,
        require_bar_on=require_bar_on,
        min_rows={s.strip().upper(): min_rows for s in syms} if min_rows > 0 else None,
        timeframe=timeframe,
    )
    emit(ok({"snapshot": rec.to_dict(), "refreshed": refreshed}))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli_data.py -q`

- [ ] **Step 5: Gate and commit**

Run: `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q`

```bash
git add algua/config/settings.py algua/cli/errors.py algua/cli/data_cmd.py tests/test_cli_data.py
git commit -m "feat(cli): data refresh-bars, bars_refresh_provider setting, refresh_failed code (#556)"
```

---

### Task 3: Tick rows persist `snapshot_id` (schema v45)

**Files:**
- Modify: `algua/registry/db/constants.py:26`
- Modify: `algua/registry/db/migrate.py` (before the final `PRAGMA user_version`, ~line 238)
- Modify: `algua/execution/order_state.py:198-260`
- Test: `tests/test_order_state.py` (has a `conn` fixture), `tests/test_registry_db.py:15,839`, `tests/test_family_registry.py:58,69`

**Interfaces:**
- Produces: `record_tick_snapshot(..., clock_source: str, snapshot_id: str | None = None)`; `latest_tick_snapshot(...)` dict gains `"snapshot_id"` (None for legacy rows).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_order_state.py`)

```python
def test_tick_snapshot_round_trips_snapshot_id(conn):
    record_tick_snapshot(
        conn, "s", tick_ts="2023-06-01T21:00:00+00:00", decision_ts="2023-05-31T00:00:00+00:00",
        equity=1.0, peak_equity=1.0, positions={}, n_submitted=0, reconcile_ok=True,
        lane="paper", strategy_id=1, code_hash="c", config_hash="cfg", dependency_hash="d",
        account_id="a", cash=1.0, clock_source="broker", snapshot_id="snap-abc")
    assert latest_tick_snapshot(conn, "s")["snapshot_id"] == "snap-abc"


def test_tick_snapshot_without_snapshot_id_is_none(conn):
    record_tick_snapshot(
        conn, "s2", tick_ts="2023-06-01T21:00:00+00:00", decision_ts=None,
        equity=1.0, peak_equity=1.0, positions={}, n_submitted=0, reconcile_ok=True,
        lane="paper", strategy_id=1, code_hash="c", config_hash="cfg", dependency_hash="d",
        account_id="a", cash=1.0, clock_source="broker")
    assert latest_tick_snapshot(conn, "s2")["snapshot_id"] is None
```

Change the four `== 44` pins to `== 45` (`tests/test_registry_db.py` lines 15 and 839; `tests/test_family_registry.py` lines 58 and 69).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_order_state.py tests/test_registry_db.py tests/test_family_registry.py -q`
Expected: FAIL — unexpected keyword `snapshot_id`; version pins.

- [ ] **Step 3: Implement**

`algua/registry/db/constants.py`:
```python
# v45 (#556): tick_snapshots.snapshot_id — which bars snapshot the tick decided on.
SCHEMA_VERSION = 45
```
`algua/registry/db/migrate.py`, immediately before `conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")`:
```python
    # v45 (#556): the bars snapshot a tick decided on. Additive nullable; legacy rows stay NULL
    # ("unknown"), never inferred — provenance is recorded at the tick, not reconstructed.
    _add_missing_columns(conn, "tick_snapshots", {"snapshot_id": "TEXT"})
```
`algua/execution/order_state.py`: `record_tick_snapshot` gains a last parameter `snapshot_id: str | None = None`; the INSERT column list gains `snapshot_id` (18 placeholders) and the values tuple appends `snapshot_id` after `datetime.now(UTC).isoformat()`. `latest_tick_snapshot`: add `snapshot_id` to the SELECT and `"snapshot_id": row["snapshot_id"],` to the dict. Docstring: "``snapshot_id`` is the bars snapshot the tick decided on (None for legacy rows)."

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_order_state.py tests/test_registry_db.py tests/test_family_registry.py -q`

- [ ] **Step 5: Gate and commit**

Run: `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q`

```bash
git add algua/registry/db/constants.py algua/registry/db/migrate.py algua/execution/order_state.py tests/test_order_state.py tests/test_registry_db.py tests/test_family_registry.py
git commit -m "feat(registry): v45 — tick_snapshots.snapshot_id, the bars a tick decided on (#556)"
```

---

### Task 4: `fleet_health` — decision-data staleness, measured against now

**Files:**
- Modify: `algua/contracts/types.py:384-393` (`SessionSpanCalendar`)
- Modify: `algua/execution/fleet_health.py` (constants; `strategy_health` 146-240)
- Test: `tests/test_fleet_health.py`

**Interfaces:**
- Consumes: `MarketCalendar.sessions_stale(latest_bar, now) -> int` (exists).
- Produces: `DECISION_STALE_AFTER_SESSIONS = 2`; health row gains `decision_stale_sessions` (vs now — the verdict), `decision_stale_at_tick` (vs tick_ts), `decision_stale_after_sessions`, `stale_detail`; protocol gains `sessions_stale`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_fleet_health.py`; add `from algua.execution.fleet_health import DECISION_STALE_AFTER_SESSIONS` and `from datetime import timedelta` if missing)

```python
def _tick2(conn, rec, *, tick_ts, decision_ts):
    """Like _tick but with an explicit decision_ts (None = a no-op tick that decided nothing)."""
    update_peak_equity(conn, rec.name, 100_000.0)
    record_tick_snapshot(
        conn, rec.name, tick_ts=tick_ts, decision_ts=decision_ts, equity=100_000.0,
        peak_equity=100_000.0, positions={}, n_submitted=0, reconcile_ok=True, lane="paper",
        strategy_id=rec.id, code_hash="c", config_hash="cfg", dependency_hash="d",
        account_id="acct", cash=100_000.0, clock_source="broker", snapshot_id="snap")


def _health_for(monkeypatch, tmp_path, *, decision_ts, now=None):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        # after the 2023-06-15 XNYS close; a normal tick decides on the prior session (06-14)
        _tick2(conn, rec, tick_ts="2023-06-15T20:30:00+00:00", decision_ts=decision_ts)
        return strategy_health(conn, rec, MarketCalendar("XNYS"), halted_globally=False,
                               now=now or (_now() + timedelta(hours=1)))


def test_decision_one_session_behind_is_ok(monkeypatch, tmp_path):
    row = _health_for(monkeypatch, tmp_path, decision_ts="2023-06-14T00:00:00+00:00")
    assert row["decision_stale_sessions"] == 1 and row["decision_stale_at_tick"] == 1
    assert row["health"] == "ok" and row["stale_detail"] is None


def test_decision_ages_against_now_not_the_tick(monkeypatch, tmp_path):
    """The tick is fresh, but no NEW tick has landed: by 06-21 the 06-14 decision bar is 4
    sessions behind the calendar (06-15, 16, 20, 21 — 06-19 is Juneteenth, closed) -> stale,
    even though at-tick it was 1."""
    row = _health_for(monkeypatch, tmp_path, decision_ts="2023-06-14T00:00:00+00:00",
                      now=datetime(2023, 6, 21, 15, 0, tzinfo=UTC))
    assert row["decision_stale_at_tick"] == 1
    assert row["decision_stale_sessions"] == 4
    assert row["health"] == "stale"
    assert row["stale_detail"] == "decision bars 4 sessions behind"


def test_decision_two_behind_is_tolerated(monkeypatch, tmp_path):
    # 06-16 (Fri) 15:00 UTC: session 06-16 open; 06-14 -> 06-16 = 2 completed sessions -> ok
    row = _health_for(monkeypatch, tmp_path, decision_ts="2023-06-14T00:00:00+00:00",
                      now=datetime(2023, 6, 16, 15, 0, tzinfo=UTC))
    assert row["decision_stale_sessions"] == 2 and row["health"] == "ok"


def test_decision_none_falls_through_to_tick_staleness(monkeypatch, tmp_path):
    row = _health_for(monkeypatch, tmp_path, decision_ts=None)
    assert row["decision_stale_sessions"] is None and row["health"] == "ok"


def test_unparseable_decision_fails_closed_to_stale(monkeypatch, tmp_path):
    row = _health_for(monkeypatch, tmp_path, decision_ts="not-a-timestamp")
    assert row["health"] == "stale"
    assert row["decision_stale_sessions"] == DECISION_STALE_AFTER_SESSIONS + 1


def test_decision_after_now_fails_closed_to_stale(monkeypatch, tmp_path):
    row = _health_for(monkeypatch, tmp_path, decision_ts="2023-06-20T00:00:00+00:00")
    assert row["health"] == "stale"


def test_fleet_health_cli_exits_nonzero_on_decision_stale(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "s")
        _tick2(conn, rec, tick_ts="2023-06-15T20:30:00+00:00",
               decision_ts="2023-06-14T00:00:00+00:00")
    # `fleet health` takes now=datetime.now(UTC) in algua/cli/fleet_cmd.py (~line 105).
    monkeypatch.setattr(
        "algua.cli.fleet_cmd.datetime",
        type("D", (), {"now": staticmethod(
            lambda tz=None: datetime(2023, 6, 20, 15, 0, tzinfo=UTC))}))
    r = CliRunner().invoke(app, ["fleet", "health"])
    assert r.exit_code != 0, r.stdout
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_fleet_health.py -q`
Expected: FAIL — `ImportError: DECISION_STALE_AFTER_SESSIONS`.

- [ ] **Step 3: Implement**

`algua/contracts/types.py` — add to the `SessionSpanCalendar` Protocol body:
```python
    def sessions_stale(self, latest_bar: datetime, now: datetime) -> int: ...
```

`algua/execution/fleet_health.py` — beside `STALE_AFTER_SESSIONS`:
```python
# Decision-DATA freshness (#556): completed sessions between the bar the newest tick DECIDED on
# and NOW. Measured against now (not the tick) so it AGES when the refresh fails and no new tick
# lands. A fresh after-close tick reads 1; the next session, before its tick, reads 2 (tolerated);
# a second consecutive miss reads 3 and trips. Aligned with the tick's own #452 mark wall
# (risk.limits.MAX_STALE_SESSIONS), NOT with STALE_AFTER_SESSIONS: "the loop is alive" and "the
# loop decided on fresh bars" are different questions with different tolerances.
DECISION_STALE_AFTER_SESSIONS = 2
```

In `strategy_health`, hoist `tick_dt` so it is available after the tick-staleness block (compute `tick_dt = _parse_utc(last["tick_ts"]) if last is not None else None` once; keep the existing branch logic byte-for-byte otherwise), then add after that block:

```python
    # Decision-data staleness (#556). NULL decision_ts is a legitimate no-op tick (nothing
    # decided) -> not evaluated. Unparseable, a calendar mapping failure, or a decision AFTER now
    # (clock skew / bad data) fails closed to stale — never a silent ok. The VERDICT is measured
    # against `now` so a lane whose refresh keeps failing (no new tick row) still trips; the
    # at-tick figure is kept for forensics ("did THAT tick trade on stale bars?").
    decision_stale_sessions: int | None = None
    decision_stale_at_tick: int | None = None
    stale_detail: str | None = None
    if last is not None and not has_unreadable_tick and last.get("decision_ts") is not None:
        decision_dt = _parse_utc(last["decision_ts"])
        if decision_dt is None:
            decision_stale_sessions = DECISION_STALE_AFTER_SESSIONS + 1
            stale_detail = "decision_ts unparseable"
        else:
            def _stale(vs: datetime | None) -> int | None:
                if vs is None:
                    return None
                try:
                    n = calendar.sessions_stale(decision_dt, vs)
                except Exception:  # noqa: BLE001 — any calendar failure fails closed -> stale
                    return DECISION_STALE_AFTER_SESSIONS + 1
                return DECISION_STALE_AFTER_SESSIONS + 1 if n < 0 else n
            decision_stale_sessions = _stale(now)
            decision_stale_at_tick = _stale(tick_dt)
            if (decision_stale_sessions is not None
                    and decision_stale_sessions > DECISION_STALE_AFTER_SESSIONS):
                stale_detail = f"decision bars {decision_stale_sessions} sessions behind"
```

Verdict chain: insert before the final `else`:
```python
    elif (decision_stale_sessions is not None
          and decision_stale_sessions > DECISION_STALE_AFTER_SESSIONS):
        health = "stale"
```
Returned dict: add
```python
        "decision_stale_sessions": decision_stale_sessions,
        "decision_stale_at_tick": decision_stale_at_tick,
        "decision_stale_after_sessions": DECISION_STALE_AFTER_SESSIONS,
        "stale_detail": stale_detail,
```
Docstring precedence sentence: `stale` now also covers "the newest tick's decision bar is more than ``DECISION_STALE_AFTER_SESSIONS`` sessions behind now".

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_fleet_health.py tests/test_cli_fleet_series.py -q`
(If a pre-existing test asserts an exact row dict, add the four new keys to its expectation.)

- [ ] **Step 5: Gate and commit**

Run: `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q`

```bash
git add algua/contracts/types.py algua/execution/fleet_health.py tests/test_fleet_health.py
git commit -m "feat(fleet): decision-data staleness is a now-relative health verdict (#556)"
```

---

### Task 5: `algua.cli.lane_refresh` — cycle plan with history floors + lane refresh

**Files:**
- Create: `algua/cli/lane_refresh.py`
- Test: `tests/test_lane_refresh.py`

**Interfaces:**
- Consumes: Task 1; `resolve_operational_universe(conn, data_dir, name, config_universe) -> (symbols, source)` (`algua.registry.universe_binding`); `load_tradable_strategy(name)` (`algua.strategies.loader` — the SAME admission path the tick helpers use; `.universe`, `.config.feature_lookback`, `.execution.warmup_bars`, `.execution.capacity` (None or a contract with `.adv_window_bars`)); `believed_positions(conn, strategy, kind)` (`algua.execution.live_ledger`); `get_calendar()` (`MarketCalendar.previous_session`, `.sessions_in_range(start, end) -> list[date]`); `get_provider(name, settings)`; `SYSTEMIC_SETUP_EXCEPTIONS` (`algua.cli._common`).
- Produces:
  - `@dataclass(frozen=True) class CyclePlan: universes: dict[str, list[str]]; held: dict[str, list[str]]; min_rows: dict[str, int]; skipped: list[dict]` + property `names`.
  - `build_cycle_plan(conn, *, names, kind, data_dir) -> CyclePlan`
  - `lane_symbols(plan, broker_net) -> list[str]`
  - `cycle_start(*, end: str, min_rows: dict[str, int]) -> str` — the earlier of the 400-day default start and the date that yields `max(min_rows) + 5` actual exchange sessions through `expected_session` (heuristic seed, then exact check with `sessions_in_range`, stepping back 30 days until satisfied).
  - `refresh_lane_snapshot(symbols, *, end, min_rows, kind: LedgerKind) -> dict` with keys `id, refreshed, symbols, require_bar_on, provider, start, end` (derives `start` via `cycle_start`; picks the provider per lane — `bars_refresh_provider` for PAPER, `bars_refresh_provider_live` for LIVE, raising `ValueError("live_refresh_provider_unset: ...")` when the live one is None; the caller uses the returned `start` as the cycle window the ticks read).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lane_refresh.py
"""Cycle plan + lane bars refresh for run-all --refresh (#556)."""
from __future__ import annotations

from contextlib import closing

import pandas as pd
import pytest
from typer.testing import CliRunner

import algua.cli.lane_refresh as lane_refresh
from algua.cli.lane_refresh import CyclePlan, build_cycle_plan, lane_symbols, refresh_lane_snapshot
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.data.contracts import ProviderBars
from algua.execution.live_ledger import LedgerKind
from algua.registry.db import connect, migrate
from tests._gate_row_helpers import seed_passing_gate

runner = CliRunner()
_S = "cross_sectional_momentum"
_CONFIG_UNIVERSE = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _register(name=_S):
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0


def _conn():
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


def test_plan_resolves_gate_bound_universe_held_and_history_floor(tmp_path):
    _register()
    seed_passing_gate(_S)
    with closing(_conn()) as conn:
        conn.execute("INSERT INTO paper_venue_fills(activity_id, strategy, symbol, qty, price, "
                     "fill_ts) VALUES (?,?,?,?,?,?)",
                     ("act-1", _S, "TSLA", 2.0, 100.0, "2023-01-01T00:00:00Z"))
        conn.commit()
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert plan.names == [_S] and plan.skipped == []
    assert set(plan.universes[_S]) == _CONFIG_UNIVERSE
    assert plan.held[_S] == ["TSLA"]
    # feature_lookback=60 for this strategy -> floor = max(60, warmup) + 1 on every universe name;
    # a held-only symbol carries no floor.
    assert all(plan.min_rows[s] >= 61 for s in _CONFIG_UNIVERSE)
    assert "TSLA" not in plan.min_rows


def test_plan_isolates_a_strategy_with_no_gate_row(tmp_path):
    _register()  # registered, NO passing gate row -> resolve_operational_universe raises
    with closing(_conn()) as conn:
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert plan.names == []
    assert plan.skipped[0]["strategy"] == _S and plan.skipped[0]["traded"] is False


def test_plan_skips_undeclared_feature_lookback_never_as_zero(monkeypatch, tmp_path):
    _register()
    seed_passing_gate(_S)
    real_load = lane_refresh.load_tradable_strategy

    def _undeclared(name):
        loaded = real_load(name)
        return type(loaded)(config=loaded.config.model_copy(update={"feature_lookback": None}),
                            **{k: getattr(loaded, k) for k in ("signal", "construct")
                               if hasattr(loaded, k)})
    monkeypatch.setattr(lane_refresh, "load_tradable_strategy", _undeclared)
    with closing(_conn()) as conn:
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert plan.names == [] and "undeclared_feature_lookback" in plan.skipped[0]["skipped"]


def test_cycle_start_widens_for_a_deep_lookback():
    from algua.calendar.market_calendar import MarketCalendar
    from algua.cli.lane_refresh import cycle_start
    default_start, _ = resolve_wall_clock_window(None, "2026-06-15")
    assert cycle_start(end="2026-06-15", min_rows={"AAPL": 61}) == default_start
    deep = cycle_start(end="2026-06-15", min_rows={"AAPL": 337})
    assert deep < default_start                       # earlier than 400 days back
    cal = MarketCalendar("XNYS")
    from datetime import date
    n = len(cal.sessions_in_range(date.fromisoformat(deep), cal.previous_session(date(2026, 6, 15))))
    assert n >= 337 + 5                                # exact session count, not a ratio


def test_plan_floor_covers_the_capacity_adv_window(monkeypatch, tmp_path):
    _register()
    seed_passing_gate(_S)
    from algua.contracts.types import CapacityContract  # the ExecutionContract.capacity type
    real_load = lane_refresh.load_tradable_strategy

    def _with_capacity(name):
        loaded = real_load(name)
        cap = CapacityContract(reference_aum=1_000_000.0, adv_window_bars=200,
                               **{})  # fill any other REQUIRED CapacityContract fields with valid values
        exec_ = loaded.execution.model_copy(update={"capacity": cap}) \
            if hasattr(loaded.execution, "model_copy") else \
            type(loaded.execution)(**{**vars(loaded.execution), "capacity": cap})
        return type(loaded)(config=loaded.config.model_copy(update={"execution": exec_}))
    monkeypatch.setattr(lane_refresh, "load_tradable_strategy", _with_capacity)
    with closing(_conn()) as conn:
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert all(plan.min_rows[s] >= 201 for s in _CONFIG_UNIVERSE)


def test_plan_rejects_a_non_tradable_strategy_like_the_tick_does(monkeypatch, tmp_path):
    _register()
    seed_passing_gate(_S)
    from algua.strategies.loader import StrategyNotFound  # any tradability error is isolated

    def _refuse(name):
        raise ValueError(f"{name}: needs_fundamentals has no paper/live lane")
    monkeypatch.setattr(lane_refresh, "load_tradable_strategy", _refuse)
    with closing(_conn()) as conn:
        plan = build_cycle_plan(conn, names=[_S], kind=LedgerKind.PAPER,
                                data_dir=get_settings().data_dir)
    assert plan.names == [] and "needs_fundamentals" in plan.skipped[0]["skipped"]


def test_lane_symbols_is_the_union_with_broker_net():
    plan = CyclePlan(universes={"a": ["MSFT", "AAPL"], "b": ["NVDA"]},
                     held={"a": ["TSLA"], "b": []}, min_rows={}, skipped=[])
    assert lane_symbols(plan, {"ORPH": 3.0, "ZERO": 0.0}) == ["AAPL", "MSFT", "NVDA", "ORPH",
                                                              "TSLA"]


def _provider_with(dates: list[str]):
    class _Provider:
        name = "fake"

        def __init__(self):
            self.request = None

        def get_bars(self, request):
            self.request = request
            n = len(dates)
            return ProviderBars(frame=pd.DataFrame({
                "ts": [f"{d}T00:00:00+00:00" for d in dates], "symbol": ["AAPL"] * n,
                "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n, "close": [1.0] * n,
                "adj_close": [1.0] * n, "volume": [1.0] * n}), source_metadata={})
    return _Provider()


def test_refresh_lane_snapshot_requires_previous_session_bar(monkeypatch, tmp_path):
    provider = _provider_with(["2023-06-13", "2023-06-14"])
    monkeypatch.setattr(lane_refresh, "get_provider", lambda name, settings: provider)
    info = refresh_lane_snapshot(["AAPL"], end="2023-06-15", min_rows={"AAPL": 2},
                                 kind=LedgerKind.PAPER)
    assert info["require_bar_on"] == "2023-06-14"   # previous XNYS session before 06-15
    assert info["refreshed"] is True and info["symbols"] == 1
    assert provider.request.end == "2023-06-15" and info["end"] == "2023-06-15"
    assert provider.request.start == info["start"] == resolve_wall_clock_window(None, "2023-06-15")[0]


def test_refresh_lane_snapshot_maps_weekend_to_friday(monkeypatch, tmp_path):
    provider = _provider_with(["2023-06-16"])
    monkeypatch.setattr(lane_refresh, "get_provider", lambda name, settings: provider)
    info = refresh_lane_snapshot(["AAPL"], end="2023-06-17", min_rows={}, kind=LedgerKind.PAPER)
    assert info["require_bar_on"] == "2023-06-16"


def test_live_refresh_provider_must_be_set_explicitly(monkeypatch, tmp_path):
    monkeypatch.delenv("ALGUA_BARS_REFRESH_PROVIDER_LIVE", raising=False)
    with pytest.raises(ValueError, match="live_refresh_provider_unset"):
        refresh_lane_snapshot(["AAPL"], end="2023-06-15", min_rows={}, kind=LedgerKind.LIVE)


def test_live_refresh_uses_the_live_provider_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_BARS_REFRESH_PROVIDER_LIVE", "fake-live")
    seen = {}
    provider = _provider_with(["2023-06-14"])

    def _get(name, settings):
        seen["name"] = name
        return provider
    monkeypatch.setattr(lane_refresh, "get_provider", _get)
    refresh_lane_snapshot(["AAPL"], end="2023-06-15", min_rows={}, kind=LedgerKind.LIVE)
    assert seen["name"] == "fake-live"
```

(`get_settings()` reads the environment per call; if it is cached, clear the cache the way other tests in the repo do — grep `get_settings.cache_clear` in `tests/`.)

Add `from algua.cli._common import resolve_wall_clock_window` to the test imports.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_lane_refresh.py -q`
Expected: FAIL — `ModuleNotFoundError: algua.cli.lane_refresh`

- [ ] **Step 3: Implement**

```python
# algua/cli/lane_refresh.py
"""Cycle plan + lane bars refresh for ``paper run-all --refresh`` / ``live run-all --refresh``
(#556).

Composition only — no policy the data layer or the registry doesn't already enforce. It answers
two questions for a run-all cycle: WHICH strategies are in this cycle and what bars they need
(the plan), and WHICH snapshot the cycle ticks against (the refresh).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from algua.calendar.factory import get_calendar
from algua.cli._common import SYSTEMIC_SETUP_EXCEPTIONS
from algua.config.settings import get_settings
from algua.data.providers import get_provider
from algua.data.refresh import refresh_bars
from algua.data.store import DataStore
from algua.execution.live_ledger import LedgerKind, believed_positions
from algua.registry.universe_binding import resolve_operational_universe
from algua.strategies.loader import load_tradable_strategy

__all__ = ["CyclePlan", "build_cycle_plan", "cycle_start", "lane_symbols",
           "refresh_lane_snapshot"]


@dataclass(frozen=True)
class CyclePlan:
    """The strategies this cycle will tick: each one's gate-bound operational universe, its
    ledger-believed held symbols, and a per-symbol history floor (rows the deepest strategy on
    that symbol needs); ``skipped`` carries per-tenant setup faults in the shape run-all emits."""

    universes: dict[str, list[str]]
    held: dict[str, list[str]]
    min_rows: dict[str, int]
    skipped: list[dict]

    @property
    def names(self) -> list[str]:
        return list(self.universes)


def build_cycle_plan(
    conn: sqlite3.Connection, *, names: list[str], kind: LedgerKind, data_dir: Path,
) -> CyclePlan:
    """Resolve each strategy's operational universe (#559: the gate-bound one, never the CONFIG
    template), held symbols, and history need. Admission is ``load_tradable_strategy`` — the SAME
    path the tick helpers use — so a tenant cannot pass planning and fail only at tick time. A
    per-strategy failure is ISOLATED (excluded from the plan, listed in ``skipped``) so one
    tenant's bad state never blocks its siblings; a systemic fault (``SYSTEMIC_SETUP_EXCEPTIONS``
    — a locked sqlite — or an ``OSError`` from the filesystem) propagates raw to abort the
    cycle."""
    universes: dict[str, list[str]] = {}
    held: dict[str, list[str]] = {}
    min_rows: dict[str, int] = {}
    skipped: list[dict] = []
    for name in names:
        try:
            strategy = load_tradable_strategy(name)
            symbols, _source = resolve_operational_universe(
                conn, data_dir, name, list(strategy.universe))
        except (KeyboardInterrupt, SystemExit):
            raise
        except (*SYSTEMIC_SETUP_EXCEPTIONS, OSError):
            raise
        except Exception as exc:  # noqa: BLE001 — a per-tenant setup fault, isolated by design
            skipped.append({"strategy": name, "traded": False, "skipped": f"cycle plan: {exc}"})
            continue
        lookback = strategy.config.feature_lookback
        if lookback is None:
            # UNDECLARED is not zero: the strategy cannot state its history need, so the wall
            # cannot be sized for it. Declare it (even 0) — the agent promote path already
            # requires this.
            skipped.append({"strategy": name, "traded": False,
                            "skipped": "cycle plan: undeclared_feature_lookback"})
            continue
        # Every bar-consuming contract sets the floor: the signal's lookback, the warm-up, AND
        # the capacity model's ADV window — a window too short for the ADV estimate silently
        # zeroes capacity and would force a held book flat.
        capacity = strategy.execution.capacity
        adv_window = int(capacity.adv_window_bars) if capacity is not None else 0
        need = max(int(lookback), int(strategy.execution.warmup_bars), adv_window) + 1
        universes[name] = sorted(symbols)
        for sym in universes[name]:
            min_rows[sym] = max(min_rows.get(sym, 0), need)
        held[name] = sorted(believed_positions(conn, name, kind))
    return CyclePlan(universes=universes, held=held, min_rows=min_rows, skipped=skipped)


def lane_symbols(plan: CyclePlan, broker_net: dict[str, float]) -> list[str]:
    """Every symbol the cycle can touch: plan universes ∪ ledger-held ∪ broker-truth net positions
    (orphan / residual / inherited holdings the book breakers and the mark wall value)."""
    out: set[str] = set()
    for syms in plan.universes.values():
        out.update(syms)
    for syms in plan.held.values():
        out.update(syms)
    out.update(s for s, q in broker_net.items() if q != 0.0)
    return sorted(out)


#: Heuristic seed for the start search (252 sessions/yr -> 1.45 days/session; 1.6 over-covers),
#: then an EXACT session count against the configured calendar decides.
_DAYS_PER_SESSION = 1.6
_START_PAD_DAYS = 10
_SESSION_PAD = 5          # extra sessions beyond the deepest floor (a vendor gap or two)
_WIDEN_STEP_DAYS = 30


def cycle_start(*, end: str, min_rows: dict[str, int]) -> str:
    """The cycle window start: the EARLIER of the default rolling start and the calendar date
    that yields ``max(min_rows) + _SESSION_PAD`` ACTUAL exchange sessions through the session the
    tick decides on. The default 400-day window holds ~275 sessions; a strategy declaring
    ``feature_lookback=336`` could never pass the wall — nor decide — without this. Exact, not a
    ratio: a holiday-dense span or another exchange cannot wedge the lane on the same
    insufficient start forever."""
    default_start, _ = resolve_wall_clock_window(None, end)
    deepest = max(min_rows.values(), default=0)
    if deepest <= 0:
        return default_start
    cal = get_calendar()
    expected = cal.previous_session(date.fromisoformat(end))
    need = deepest + _SESSION_PAD
    candidate = expected - timedelta(days=math.ceil(deepest * _DAYS_PER_SESSION) + _START_PAD_DAYS)
    while len(cal.sessions_in_range(candidate, expected)) < need:
        candidate -= timedelta(days=_WIDEN_STEP_DAYS)
    return min(default_start, candidate.isoformat())


def _lane_provider_name(kind: LedgerKind, settings) -> str:
    """PAPER uses the research-convenience default; LIVE must be named explicitly by a human —
    real-money decision data is never inherited from a default."""
    if kind is LedgerKind.LIVE:
        if not settings.bars_refresh_provider_live:
            raise ValueError(
                "live_refresh_provider_unset: set ALGUA_BARS_REFRESH_PROVIDER_LIVE to the "
                "approved live bars provider before `live run-all --refresh`")
        return settings.bars_refresh_provider_live
    return settings.bars_refresh_provider


def refresh_lane_snapshot(
    symbols: list[str], *, end: str, min_rows: dict[str, int], kind: LedgerKind,
) -> dict:
    """Resolve-or-ingest the lane's bars for ``[cycle_start, end)``, requiring every symbol's
    newest bar to fall on the session the tick decides on — the latest completed session
    strictly before ``end`` — and every decision-universe symbol to carry its history floor.
    Returns the window it used (``start``/``end``): the caller ticks over the SAME window.
    Raises (RefreshError / provider errors / an unset live provider) rather than returning a
    stale id; the caller fails the cycle closed."""
    settings = get_settings()
    provider = get_provider(_lane_provider_name(kind, settings), settings)
    start = cycle_start(end=end, min_rows=min_rows)
    require_bar_on = get_calendar().previous_session(date.fromisoformat(end)).isoformat()
    rec, refreshed = refresh_bars(
        DataStore(settings.data_dir), provider, symbols=symbols, start=start, end=end,
        require_bar_on=require_bar_on, min_rows=min_rows,
    )
    return {"id": rec.snapshot_id, "refreshed": refreshed, "symbols": len(symbols),
            "require_bar_on": require_bar_on, "provider": provider.name,
            "start": start, "end": end}
```

Add `import math` and `from datetime import date, timedelta` and `from algua.cli._common import SYSTEMIC_SETUP_EXCEPTIONS, resolve_wall_clock_window` to the module imports.

If `LoadedStrategy` exposes `feature_lookback` differently (check `algua/strategies/base.py:98-140`), read it from wherever `StrategyConfig.feature_lookback` is reachable; the floor must be `max(feature_lookback, warmup_bars) + 1`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_lane_refresh.py -q`

- [ ] **Step 5: Gate and commit**

Run: `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q`

```bash
git add algua/cli/lane_refresh.py tests/test_lane_refresh.py
git commit -m "feat(cli): lane_refresh — cycle plan with history floors + lane refresh (#556)"
```

---

### Task 6: `paper run-all --refresh`

**Files:**
- Modify: `algua/cli/paper_cmd.py` — `_run_paper_strategy_tick` (692-712; its `record_tick_snapshot` ~844), `trade_tick` (~898), `run_all` (968-1215)
- Test: `tests/test_paper_run_all.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 5.
- Produces: `paper run-all [--snapshot ID [--start D] [--end D] | --refresh]` (`--start`/`--end` both rejected with `--refresh`; the window is derived); envelope `"snapshot": {...}`; `_run_paper_strategy_tick(..., snapshot_id: str | None = None)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_paper_run_all.py`; add imports `from algua.data.refresh import RefreshError`, `from algua.execution.order_state import latest_tick_snapshot`)

```python
_DERIVED_START = "2024-12-01"  # what the stubbed refresh reports as the cycle window start


def _stub_refresh(monkeypatch, *, snapshot_id="fresh-1", seen=None):
    def _refresh(symbols, *, end, min_rows, kind):
        if seen is not None:
            seen["symbols"], seen["min_rows"], seen["end"], seen["kind"] = symbols, min_rows, end, kind
        return {"id": snapshot_id, "refreshed": True, "symbols": len(symbols),
                "require_bar_on": "2026-01-30", "provider": "fake",
                "start": _DERIVED_START, "end": end}
    monkeypatch.setattr("algua.cli.paper_cmd.refresh_lane_snapshot", _refresh)


def _stub_tick(monkeypatch, ticked=None, windows=None):
    def _rt(strategy, broker, provider, start, end, hooks=None, max_drawdown=None):
        if ticked is not None:
            ticked.append(strategy.name)
        if windows is not None:
            windows.append((start, end))
        return _success_result()
    monkeypatch.setattr("algua.cli.paper_cmd.run_tick", _rt)


def test_run_all_rejects_neither_both_and_window_flags_with_refresh():
    for args in ([], ["--snapshot", _SNAP, "--refresh"], ["--refresh", "--end", _END],
                 ["--refresh", "--start", _START]):
        r = runner.invoke(app, ["paper", "run-all", *args])
        assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False, args


def test_run_all_refresh_discovers_on_first_read_and_reconciles_on_second(monkeypatch):
    """Broker positions are read twice: the first sample only feeds symbol discovery; the
    reconcile runs on a FRESH read taken after the (slow) refresh — a fill that lands during the
    refresh must be judged by the reconcile, never traded against a stale sample."""
    _to_paper(_S1)
    _seed_allocation(_S1)
    broker = _RunAllBroker()  # flat at first read
    reads: list[dict] = []
    real_get_positions = broker.get_positions

    def _get_positions():
        reads.append({})
        if len(reads) == 2:               # a resting order filled during the refresh
            broker._positions = {"ORPH": 2.0}
        return real_get_positions()
    broker.get_positions = _get_positions
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    seen: dict = {}
    _stub_refresh(monkeypatch, seen=seen)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())
    ticked: list = []
    _stub_tick(monkeypatch, ticked)
    r = runner.invoke(app, ["paper", "run-all", "--refresh"])
    payload = json.loads(r.stdout)
    assert "ORPH" not in seen["symbols"] and "AAPL" in seen["symbols"]   # discovery saw flat
    assert seen["min_rows"]["AAPL"] >= 61
    assert payload.get("deferred") is True and ticked == [], r.stdout   # reconcile saw ORPH
    assert len(reads) >= 2


def test_run_all_refresh_clean_account_ticks_over_the_derived_window(monkeypatch):
    _to_paper(_S1)
    _seed_allocation(_S1)
    broker = _RunAllBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    _stub_refresh(monkeypatch)
    selected: dict = {}
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snap: selected.setdefault("snap", snap) or object())
    windows: list = []
    _stub_tick(monkeypatch, windows=windows)
    r = runner.invoke(app, ["paper", "run-all", "--refresh"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["snapshot"]["id"] == "fresh-1" and selected["snap"] == "fresh-1"
    assert payload["snapshot"]["start"] == _DERIVED_START
    assert windows == [(_DERIVED_START, payload["snapshot"]["end"])]   # ticks read the same window
    with closing(connect(get_settings().db_path)) as conn:
        assert latest_tick_snapshot(conn, _S1)["snapshot_id"] == "fresh-1"


def test_run_all_refresh_failure_fails_closed_before_any_tick(monkeypatch):
    _to_paper(_S1)
    _seed_allocation(_S1)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _RunAllBroker())

    def _boom(symbols, *, end, min_rows, kind):
        raise RefreshError("stale", ["AAPL"], require_bar_on="2026-01-30")
    monkeypatch.setattr("algua.cli.paper_cmd.refresh_lane_snapshot", _boom)
    ticked: list = []
    _stub_tick(monkeypatch, ticked)
    r = runner.invoke(app, ["paper", "run-all", "--refresh"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False and payload["code"] == "refresh_failed"
    assert "stale" in payload["error"] and ticked == []


def test_run_all_refresh_isolates_one_planless_strategy(monkeypatch):
    _to_paper(_S1)
    _seed_allocation(_S1)
    # _S2 reaches paper WITHOUT a gate row: the plan must skip it and still tick _S1.
    assert runner.invoke(app, ["backtest", "run", _S2, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    _force_stage(_S2, "paper")
    _seed_allocation(_S2)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _RunAllBroker())
    _stub_refresh(monkeypatch)
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snap: object())
    ticked: list = []
    _stub_tick(monkeypatch, ticked)
    r = runner.invoke(app, ["paper", "run-all", "--refresh"])
    assert r.exit_code == 0, r.stdout
    by_name = {s["strategy"]: s for s in json.loads(r.stdout)["strategies"]}
    assert by_name[_S1].get("ok") is True and ticked == [_S1]
    assert by_name[_S2]["traded"] is False and "cycle plan" in by_name[_S2]["skipped"]


def test_run_all_refresh_all_planless_is_a_failed_cycle(monkeypatch):
    # Registered + allocated + paper stage, but NO gate row: every tenant fails planning.
    assert runner.invoke(app, ["backtest", "run", _S1, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    _force_stage(_S1, "paper")
    _seed_allocation(_S1)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _RunAllBroker())
    called: list = []
    monkeypatch.setattr("algua.cli.paper_cmd.refresh_lane_snapshot",
                        lambda *a, **k: called.append(1))
    r = runner.invoke(app, ["paper", "run-all", "--refresh"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False and payload["code"] == "cycle_plan_failed"
    assert payload["strategies"][0]["strategy"] == _S1 and called == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_paper_run_all.py -q`
Expected: the 6 new tests FAIL; existing tests pass.

- [ ] **Step 3: Implement** (in `algua/cli/paper_cmd.py`)

1. Imports: `from algua.cli.lane_refresh import build_cycle_plan, lane_symbols, refresh_lane_snapshot`. (`LedgerKind`, `get_settings` are already imported.)
2. `_run_paper_strategy_tick`: add `snapshot_id: str | None = None` after `end: str` (keyword-only group); pass `snapshot_id=snapshot_id` to its `record_tick_snapshot(...)`.
3. `trade_tick`: pass `snapshot_id=snapshot` to its `_run_paper_strategy_tick(...)` call.
4. `run_all` options — replace `snapshot` and add `refresh`:
```python
    snapshot: str | None = typer.Option(
        None, "--snapshot", help="tick against this ingested bars snapshot id (explicit/replay)"),
    refresh: bool = typer.Option(
        False, "--refresh",
        help="resolve-or-ingest the lane's bars for the cycle window — the union of every "
             "tickable strategy's gate-bound universe, ledger-held and broker-held symbols — "
             "requiring each symbol's newest bar on the session the tick decides on and each "
             "universe symbol's history floor; the always-on operator path. Exactly one of "
             "--snapshot/--refresh is required; --end is derived (not accepted) under --refresh "
             "(#556)"),
```
   and as the first statements of the body (BEFORE `resolve_wall_clock_window`):
```python
    if bool(snapshot) == refresh:
        raise ValueError("pass exactly one of --snapshot <id> or --refresh")
    if refresh and (start is not None or end is not None):
        raise ValueError("--refresh derives the cycle window (end = today, start from the "
                         "strategies' history need); --start/--end are not accepted")
```
5. Delete `provider = _select_provider(False, snapshot)` from its current position (right after `broker = _alpaca_broker_from_settings()`).
6. Hoist `results: list[dict] = []` above the `for prec in tickable:` loop to just AFTER the venue-ingest `try/except` block and BEFORE `cycle = paper_reconcile.next_cycle(conn)`, and insert the refresh block there. The existing `_paper_broker_net(broker)` call inside the reconcile line is LEFT WHERE IT IS — that second, post-refresh read is what the reconcile judges:

```python
                results: list[dict] = []
                # Lane bars refresh (#556): AFTER fill ingest (so ledger-held symbols are current),
                # BEFORE the reconcile. Broker positions are read TWICE on purpose: this first
                # read only feeds symbol DISCOVERY (orphan/residual marks get fetched too); the
                # reconcile below takes its own fresh read AFTER the network round-trip, so a fill
                # that lands during the refresh is judged by the reconcile (defer), never traded
                # against a stale sample. A per-tenant plan fault is isolated like any setup
                # error; EVERY tenant failing to plan is a failed cycle, not a benign no-op; a
                # refresh failure fails the WHOLE cycle closed — the operator alerts, leaves the
                # session marker unwritten, and the next fire retries. Never fall back to an older
                # snapshot. The derived window (start from the deepest history floor) is the one
                # the ticks read too.
                snapshot_info: dict = {"id": snapshot, "refreshed": False,
                                       "start": start, "end": end}
                if refresh:
                    plan = build_cycle_plan(
                        conn, names=[prec.name for prec in tickable], kind=LedgerKind.PAPER,
                        data_dir=get_settings().data_dir)
                    results.extend(plan.skipped)
                    if tickable and not plan.universes:
                        audit_append(conn, actor="system", action="cycle_plan_failed",
                                     reason=f"{len(plan.skipped)} tenant(s) failed planning",
                                     strategy=None)
                        emit({"ok": False, "code": "cycle_plan_failed",
                              "error": "every tickable strategy failed the cycle plan",
                              "strategies": results,
                              "skipped_unallocated": skipped_unallocated})
                        raise typer.Exit(1)
                    tickable = [prec for prec in tickable if prec.name in plan.universes]
                    if tickable:
                        try:
                            snapshot_info = refresh_lane_snapshot(
                                lane_symbols(plan, _paper_broker_net(broker)), end=end,
                                min_rows=plan.min_rows, kind=LedgerKind.PAPER)
                        except Exception as exc:  # noqa: BLE001 — any refresh fault fails closed
                            audit_append(conn, actor="system", action="bars_refresh_failed",
                                         reason=str(exc), strategy=None)
                            log.error("bars_refresh_failed",
                                      extra={"fields": {"lane": "paper"}}, exc_info=True)
                            emit({"ok": False, "code": "refresh_failed", "error": str(exc),
                                  "strategies": results,
                                  "skipped_unallocated": skipped_unallocated})
                            raise typer.Exit(1) from exc
                        snapshot, start = snapshot_info["id"], snapshot_info["start"]
                        log.info("bars_refreshed",
                                 extra={"fields": {"lane": "paper", **snapshot_info}})
                provider = _select_provider(False, snapshot)
```
   (`start`/`end` are the values `resolve_wall_clock_window(start, end)` produced earlier in the body; under `--refresh` both flags were None, so `end` is today and `start` is then overwritten with the derived one.)
7. Loop: pass `snapshot_id=snapshot` to `_run_paper_strategy_tick(...)`.
8. Envelope: add `"snapshot": snapshot_info,`.
9. `cycle_start` log: `"snapshot": snapshot or "refresh"`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_paper_run_all.py tests/test_cli_paper.py tests/test_lane_parity.py -q`

- [ ] **Step 5: Gate and commit**

Run: `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q`

```bash
git add algua/cli/paper_cmd.py tests/test_paper_run_all.py
git commit -m "feat(paper): run-all --refresh — lane bars refresh before the reconcile (#556)"
```

---

### Task 7: `live run-all --refresh` (lane parity)

**Files:**
- Modify: `algua/cli/live_cmd.py` — `_run_strategy_tick` (145-165; `record_tick_snapshot` ~296), `run_all` (349-640; `_select_provider` at 449; `net_positions` at ~457)
- Test: `tests/test_cli_live.py`

**Interfaces:**
- Consumes: as Task 6; `_broker_net_positions(broker)` (already computed as `net_positions` for the reconcile).
- Produces: `live run-all [--snapshot ID | --refresh]`; envelope `"snapshot"`; `_run_strategy_tick(..., snapshot_id: str | None = None)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cli_live.py`; `_to_live`, `_auth`, `_permissive_book`, `runner`, `app`, `json` exist there; add `from algua.data.refresh import RefreshError`)

```python
def _one_live_strategy(monkeypatch, *, broker_net=None):
    """ONE verified + allocated live strategy over a clean account — the stub set
    `test_run_all_skips_only_the_unauthorized_strategy` uses, minus the phantom sibling."""
    _permissive_book(monkeypatch)
    _to_live("cross_sectional_momentum")
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd.ingest_activities", lambda conn, acts, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd.fill_cursor", lambda conn, kind: None)
    monkeypatch.setattr("algua.cli.live_cmd._broker_account_activities", lambda broker, after: [])
    monkeypatch.setattr("algua.cli.live_cmd._broker_net_positions",
                        lambda broker: dict(broker_net or {}))
    monkeypatch.setattr("algua.cli.live_cmd._broker_buying_power", lambda broker: 100_000.0)


def _refresh_stub(seen: dict | None = None, snapshot_id="live-fresh"):
    def _refresh(symbols, *, end, min_rows, kind):
        if seen is not None:
            seen["symbols"], seen["min_rows"], seen["kind"] = symbols, min_rows, kind
        return {"id": snapshot_id, "refreshed": True, "symbols": len(symbols),
                "require_bar_on": "2026-01-30", "provider": "fake",
                "start": "2024-12-01", "end": end}
    return _refresh


def test_live_run_all_rejects_neither_both_and_window_flags_with_refresh():
    for args in ([], ["--snapshot", "x", "--refresh"], ["--refresh", "--end", "2026-02-01"],
                 ["--refresh", "--start", "2025-01-01"]):
        r = runner.invoke(app, ["live", "run-all", *args])
        assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False, args


def test_live_run_all_refresh_ticks_on_the_resolved_snapshot(monkeypatch):
    _one_live_strategy(monkeypatch)
    seen: dict = {}
    monkeypatch.setattr("algua.cli.live_cmd.refresh_lane_snapshot", _refresh_stub(seen))
    selected: dict = {}
    monkeypatch.setattr("algua.cli.live_cmd._select_provider",
                        lambda demo, snap: selected.setdefault("snap", snap) or object())
    ticked: dict = {}

    def _tick(conn, name, authorization, broker, provider, max_drawdown, start, end,
              reserve_buy=None, cancel=None, snapshot_id=None):
        ticked["snapshot_id"], ticked["window"] = snapshot_id, (start, end)
        return {"strategy": name, "submitted": []}
    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick", _tick)
    r = runner.invoke(app, ["live", "run-all", "--refresh"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["snapshot"]["id"] == "live-fresh" and selected["snap"] == "live-fresh"
    assert ticked["snapshot_id"] == "live-fresh"
    assert ticked["window"] == ("2024-12-01", payload["snapshot"]["end"])   # derived window
    assert set(seen["symbols"]) == {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}
    assert seen["min_rows"]["AAPL"] >= 61
    assert seen["kind"] is LedgerKind.LIVE   # the LIVE provider setting is consulted, not paper's


def test_live_run_all_refresh_includes_broker_orphans(monkeypatch):
    # A broker-truth holding no strategy owns is still fetched (the book breakers mark it). The
    # live reconcile tolerates the mismatch for --grace-cycles, so the cycle proceeds.
    _one_live_strategy(monkeypatch, broker_net={"ORPH": 2.0})
    seen: dict = {}
    monkeypatch.setattr("algua.cli.live_cmd.refresh_lane_snapshot", _refresh_stub(seen))
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snap: object())
    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick",
                        lambda *a, **k: {"strategy": "cross_sectional_momentum", "submitted": []})
    r = runner.invoke(app, ["live", "run-all", "--refresh"])
    assert "ORPH" in seen["symbols"], r.stdout


def test_live_run_all_refresh_failure_fails_closed(monkeypatch):
    _one_live_strategy(monkeypatch)

    def _boom(symbols, *, end, min_rows, kind):
        raise RefreshError("missing", ["AAPL"], require_bar_on="2026-01-30")
    monkeypatch.setattr("algua.cli.live_cmd.refresh_lane_snapshot", _boom)
    ticked: list = []
    monkeypatch.setattr("algua.cli.live_cmd._run_strategy_tick",
                        lambda *a, **k: ticked.append(1) or {"strategy": "x", "submitted": []})
    r = runner.invoke(app, ["live", "run-all", "--refresh"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["code"] == "refresh_failed" and "missing" in payload["error"]
    assert ticked == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli_live.py -q -k "refresh or rejects"`

- [ ] **Step 3: Implement** — mirror Task 6 in `live_cmd.py`:

1. Import `from algua.cli.lane_refresh import build_cycle_plan, lane_symbols, refresh_lane_snapshot`.
2. `_run_strategy_tick`: add `snapshot_id: str | None = None` after `cancel=None`; pass it to `record_tick_snapshot`.
3. `run_all` options + the two `ValueError` guards, exactly as Task 6 step 4.
4. Delete `provider = _select_provider(False, snapshot)` (line 449).
5. After `_recover_live_stranded(conn, broker)` and BEFORE the existing `cycle = live_reconcile.next_cycle(conn)` / `net_positions = _broker_net_positions(broker)` lines, insert the refresh block from Task 6 step 6 with these substitutions: `kind=LedgerKind.LIVE` in BOTH `build_cycle_plan(...)` and `refresh_lane_snapshot(...)` (add `from algua.execution.live_ledger import LedgerKind` to the test file too); the tickable set is `verified` (a list of `(name, authorization)`) — `names=[n for n, _a in verified]`, filter `verified = [(n, a) for n, a in verified if n in plan.universes]`, `tickable` → `verified`; the discovery read is `_broker_net_positions(broker)` (a FIRST read — the existing `net_positions = _broker_net_positions(broker)` line stays where it is, AFTER the refresh, and feeds the reconcile + book exposure as today); `"lane": "live"`; add `"skipped": skipped` to both failure envelopes. Then `provider = _select_provider(False, snapshot)`.
6. Delete the later `results = []` (~line 583) so the hoisted list is the one the loop appends to.
7. Pass `snapshot_id=snapshot` to `_run_strategy_tick(...)` in the loop (and in single-strategy `live run` if it calls the same helper — `grep -n "_run_strategy_tick(" algua/cli/live_cmd.py`).
8. Envelope `"snapshot": snapshot_info,`; `cycle_start` logs `snapshot or "refresh"`. The reconcile-halt / reconcile-defer / book-breach early-return envelopes also carry `"strategies": results` and `"snapshot": snapshot_info` (Task 6 review finding: an early return must not discard `plan.skipped` rows or the refreshed id).
9. Guard learned from Task 6's review: only select a provider when there is something to tick — `provider = _select_provider(False, snapshot) if verified else None` — so a cycle that planned nothing can never hit `_select_provider(False, None)`'s `ValueError`. (Live already returns before the broker when nothing is allocated, and the plan block returns when every tenant fails planning, so this is belt-and-braces parity with paper.) Help text must say `--start`/`--end` are derived (not accepted) under `--refresh`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli_live.py tests/test_lane_parity.py -q`

- [ ] **Step 5: Gate and commit**

Run: `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q`

```bash
git add algua/cli/live_cmd.py tests/test_cli_live.py
git commit -m "feat(live): run-all --refresh — lane parity with paper (#556)"
```

---

### Task 8: Operator job template + completion provenance, marker, deployment files, docs

**Files:**
- Modify: `algua/operator/jobs.py:78-95`
- Modify: `algua/operator/schedule.py:186-215` (`SessionMarker.record`)
- Modify: `algua/operator/session_runner.py` (~line 224, the `marker.record(...)` success site)
- Modify: `deploy/systemd/algua-paper.service:16-22`, `deploy/systemd/algua.env.example:9-13`, `deploy/systemd/README.md:41-47`, `deploy/systemd/install-user-units.sh:99`
- Modify: `CLAUDE.md` (command surface)
- Test: `tests/test_operator_jobs.py`, `tests/test_cli_operator.py`, `tests/test_operator_schedule.py`

**Interfaces:**
- Produces: `OPERATOR_JOBS["paper"].argv_template == ("algua", "paper", "run-all", "--refresh")`; `is_completed` requires a non-empty `payload["snapshot"]["id"]` whenever `payload["strategies"]` is non-empty; `SessionMarker.record(..., pid, snapshot_id: str | None = None)`; operator success envelope gains `"snapshot_id"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_operator_jobs.py`: `_PAPER_ARGV = ("algua", "paper", "run-all", "--refresh")`; replace the placeholder-specific tests:
```python
def test_bind_accepts_canonical_argv_and_captures_nothing() -> None:
    assert _paper().bind(_PAPER_ARGV) == {}


def test_bind_rejects_trailing_snapshot_flag() -> None:
    with pytest.raises(CommandMismatch):
        _paper().bind(("algua", "paper", "run-all", "--refresh", "--snapshot", "X"))


def test_bind_rejects_legacy_snapshot_argv() -> None:
    with pytest.raises(CommandMismatch):
        _paper().bind(("algua", "paper", "run-all", "--snapshot", "SNAP"))


def test_is_completed_requires_snapshot_id_when_strategies_ticked() -> None:
    ok = {"ok": True, "strategies": [{"strategy": "s", "ok": True}]}
    assert _paper().is_completed(0, {**ok, "snapshot": {"id": "snap-1"}}) is True
    assert _paper().is_completed(0, ok) is False
    assert _paper().is_completed(0, {**ok, "snapshot": {"id": ""}}) is False
    assert _paper().is_completed(0, {**ok, "snapshot": {"id": None}}) is False


def test_is_completed_requires_at_least_one_successful_tick() -> None:
    # Every tenant failed tick-time setup: ok:true at the top, a valid snapshot, zero ticks.
    all_setup_errors = {"ok": True, "snapshot": {"id": "snap-1"},
                        "strategies": [{"ok": False, "strategy": "s", "kind": "setup_error"},
                                       {"strategy": "t", "traded": False, "skipped": "x"}]}
    assert _paper().is_completed(0, all_setup_errors) is False
    one_ok = {**all_setup_errors,
              "strategies": [*all_setup_errors["strategies"], {"strategy": "u", "ok": True}]}
    assert _paper().is_completed(0, one_ok) is True


def test_is_completed_no_work_needs_no_snapshot() -> None:
    assert _paper().is_completed(0, {"ok": True, "strategies": []}) is True
```
Delete `test_bind_rejects_missing_snapshot_value` and `test_bind_rejects_empty_placeholder_value`; update the argv literals in the remaining short-arity / swapped-flag / wrong-head tests to the new template.

`tests/test_cli_operator.py`: `_CMD = ["algua", "paper", "run-all", "--refresh"]`; in the mismatch parametrization replace the `--snapshot`-based junk case with `["algua", "paper", "run-all", "--refresh", "--evil"]`; extend `test_paper_systemd_units_present_and_shaped`:
```python
    assert "paper run-all --refresh" in svc
    assert "ALGUA_PAPER_SNAPSHOT" not in svc
    env = (_SYSTEMD / "algua.env.example").read_text()
    assert "ALGUA_PAPER_SNAPSHOT=" not in env
    installer = (_SYSTEMD / "install-user-units.sh").read_text()
    assert "ALGUA_PAPER_SNAPSHOT" not in installer
```
and add (mirroring `test_due_clean_run_records_full_argv` for the driver seam and marker read):
```python
def test_due_clean_run_records_snapshot_id_from_driver_payload(db_dir, monkeypatch):
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(
        0, json.dumps({"ok": True, "strategies": [{"strategy": "s", "ok": True}],
                       "snapshot": {"id": "snap-9", "refreshed": True}})))
    r = _invoke()
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["snapshot_id"] == "snap-9"
    entry = json.loads((db_dir / "operator_sessions.json").read_text())["paper"]
    assert entry["snapshot_id"] == "snap-9" and entry["command"] == _CMD


def test_due_ticked_without_snapshot_is_completion_unconfirmed(db_dir, monkeypatch):
    alerts = _spy_alerts(monkeypatch)
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(
        0, json.dumps({"ok": True, "strategies": [{"strategy": "s", "ok": True}]})))
    r = _invoke()
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["recorded"] is False
    assert not (db_dir / "operator_sessions.json").exists() or \
        "paper" not in json.loads((db_dir / "operator_sessions.json").read_text())
    assert any(kind == "completion_unconfirmed" for kind, _ in alerts)
```

`tests/test_operator_schedule.py`:
```python
def test_marker_record_round_trips_snapshot_id(tmp_path) -> None:
    m = SessionMarker(tmp_path)
    m.record("paper", date(2023, 6, 1), command=["algua", "paper", "run-all", "--refresh"],
             rc=0, host="h", pid=1, snapshot_id="s9")
    m.record("research", date(2023, 6, 1), command=["algua"], rc=0, host="h", pid=1)
    data = json.loads((tmp_path / "operator_sessions.json").read_text())
    assert data["paper"]["snapshot_id"] == "s9"
    assert data["research"]["snapshot_id"] is None
```
(Confirm the marker filename against the existing `_record`-based tests in that file.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_operator_jobs.py tests/test_cli_operator.py tests/test_operator_schedule.py -q`

- [ ] **Step 3: Implement**

`algua/operator/jobs.py` — the paper job:
```python
        # #556: no variable token. The lane refreshes its own bars; `--refresh` is a fixed flag,
        # so the exact-arity match still rejects any appended/altered token (incl. `--snapshot X`),
        # and the env-var-left-empty failure mode of the old {snapshot} placeholder cannot recur.
        argv_template=("algua", "paper", "run-all", "--refresh"),
        expected_duration_seconds=900.0,
        # ... (keep the existing comment) ... PLUS (#556): a cycle that TICKED strategies must
        # carry the snapshot it decided on (`snapshot.id`) — provenance is part of "completed";
        # a no-work cycle (no strategies) needs none.
        is_completed=lambda rc, payload: _paper_completed(rc, payload),
```
with, above `OPERATOR_JOBS`:
```python
def _paper_completed(rc: int, payload: dict | None) -> bool:
    """A cycle COMPLETED the session iff the driver asserted success, did not defer, and — when
    it had tenants — decided on a named snapshot AND at least one tenant actually ticked. A cycle
    whose every tenant failed tick-time setup, or that ticked without provenance, is NOT
    completed: the marker stays unwritten and the next fire retries (#556)."""
    p = payload or {}
    if not (rc == 0 and p.get("ok") is True and not p.get("deferred")):
        return False
    strategies = p.get("strategies") or []
    if not strategies:
        return True
    snap = p.get("snapshot")
    has_snapshot = isinstance(snap, dict) and isinstance(snap.get("id"), str) and bool(snap["id"])
    any_ticked = any(isinstance(s, dict) and s.get("ok") is True for s in strategies)
    return has_snapshot and any_ticked
```

`algua/operator/schedule.py` `record`: add `snapshot_id: str | None = None` after `pid`; add `"snapshot_id": snapshot_id,` to the entry; docstring: "``snapshot_id`` = the bars snapshot the driver reported ticking on (#556), so a session's completion is attributable to a concrete artifact even though the argv is now fixed."

`algua/operator/session_runner.py` success site:
```python
        snap = payload.get("snapshot") if isinstance(payload, dict) else None
        snapshot_id = snap.get("id") if isinstance(snap, dict) else None
        marker.record(job, decision.session, command=list(command), rc=rc, host=host, pid=pid,
                      snapshot_id=snapshot_id)
        emit(ok({"job": job, "ran": True, "recorded": True, "session": sess_iso, "rc": rc,
                 "snapshot_id": snapshot_id}))
```
(The "ticked without snapshot" case is already routed to `completion_unconfirmed` by the existing `not op_job.is_completed(...)` branch — no new branch needed.)

`deploy/systemd/algua-paper.service` lines 16-22:
```
# The command after `--` MUST exactly match the `paper` job's canonical argv template
# (`algua paper run-all --refresh`) or `operator run` fail-closes with command_mismatch. The lane
# refreshes its own bars each session (#556) — there is no snapshot variable to configure. `algua`
# must be on PATH for the operator's subprocess (a `.venv/bin` entry or a symlink); do NOT append
# ad-hoc flags — a drawdown/window override is a manual human path OUTSIDE the timer (see README).
ExecStart=/opt/algua/.venv/bin/algua operator run --job paper -- algua paper run-all --refresh
```
`deploy/systemd/algua.env.example` lines 9-13:
```
# Bars provider the paper lane refreshes through each session (#556). Optional; default yfinance.
# (ALGUA_PAPER_SNAPSHOT is GONE: the lane resolves-or-ingests its own snapshot per session, so a
# static snapshot can no longer go stale and an empty variable can no longer break the unit.)
# ALGUA_BARS_REFRESH_PROVIDER=yfinance
```
`deploy/systemd/README.md` step 2:
```
2. Nothing to set for data (#556): the paper unit runs `algua paper run-all --refresh`, which
   resolves-or-ingests the lane's bars each session — the union of every tickable strategy's
   gate-bound universe plus ledger-held and broker-held symbols, each required to carry its newest
   bar on the session the tick decides on and each universe symbol its history floor; a
   missing/lagging/short symbol fails the cycle closed and the next fire retries. The trailing
   command in `ExecStart` **must exactly match** the `paper` job's argv template
   (`algua paper run-all --refresh`) — an exact-arity structural match — or the wrapper fail-closes
   with `command_mismatch`. `ALGUA_BARS_REFRESH_PROVIDER` (default `yfinance`) picks the provider.
   For a replay or incident forensics run `algua paper run-all --snapshot <id>` by hand, outside
   the timer.
```
`deploy/systemd/install-user-units.sh:99`: replace `algua-paper.service needs ALGUA_PAPER_SNAPSHOT.` with `algua-paper.service needs the ALGUA_ALPACA_* paper credentials.`

`CLAUDE.md` — after the `fleet health` bullet:
```
- `uv run algua paper run-all --refresh` / `uv run algua live run-all --refresh` — the always-on
  lane cycle resolves-or-ingests its OWN bars each session (#556): union of every tickable
  strategy's gate-bound universe ∪ ledger-held ∪ broker-held symbols over the cycle window; each
  symbol's newest bar must fall on the session the tick decides on (`previous_session(today)`)
  and each universe symbol must carry its strategies' history floor; a same-request snapshot is
  reused only if it still passes that wall (content hash re-checked). Missing/stale/misdated/
  short → `refresh_failed`, nothing minted, no tick, the timer retries; every tenant failing to
  plan → `cycle_plan_failed`. `--snapshot <id>` is the explicit replay path (exactly one of the
  two; `--end` is derived under `--refresh`). Tick rows record the `snapshot_id` they decided on
  (v45) and the operator marks a session complete only with one; `fleet status` reports
  `decision_stale_sessions` (vs now) and flags `stale` past 2 (`DECISION_STALE_AFTER_SESSIONS`).
  `uv run algua data refresh-bars --symbols … --start D --end D --require-bar-on D [--min-rows N]`
  is the manual primitive.
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_operator_jobs.py tests/test_cli_operator.py tests/test_operator_schedule.py -q`

- [ ] **Step 5: Gate and commit**

Run: `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q`

```bash
git add algua/operator/jobs.py algua/operator/schedule.py algua/operator/session_runner.py deploy/systemd/algua-paper.service deploy/systemd/algua.env.example deploy/systemd/README.md deploy/systemd/install-user-units.sh CLAUDE.md tests/test_operator_jobs.py tests/test_cli_operator.py tests/test_operator_schedule.py
git commit -m "feat(operator): paper job runs run-all --refresh; completion requires snapshot provenance (#556)"
```

---

### Task 9: Spec + plan land with the branch

- [ ] **Step 1: Commit the design record**

```bash
git add docs/superpowers/specs/2026-09-05-lane-bars-refresh-design.md docs/superpowers/plans/2026-09-05-lane-bars-refresh.md
git commit -m "docs: lane bars refresh design + plan (#556)"
```

- [ ] **Step 2: Full gate one last time**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. The orchestrator then pushes and opens the PR (touches CODEOWNERS paths → human merge).
