# Holdout Interval Matching (#192) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the walk-forward holdout single-use guard match on the *actual OOS bar-interval* instead of `(full-period overlap, exact holdout_frac)`, closing the #192 re-burn exploit (varying `--holdout-frac` or re-framing the period re-peeks the same OOS bars).

**Architecture:** A new pure-ish backtest helper `holdout_window()` reproduces `build_portfolio`'s grid index from the bar date-index (no strategy run) and returns the exact `(holdout_start, holdout_end)` dates `walk_forward` would burn. The CLI computes this *before* reserving and passes the interval into `reserve_holdout`, which now matches on interval-overlap (NULL-interval rows fail closed). Schema 22→23 adds `holdout_start/holdout_end`; legacy rows backfill to the conservative full period. Registry stays pure (receives date strings).

**Tech Stack:** Python, pandas, vectorbt, SQLite, Typer CLI, pytest. Base off `origin/main` (has #161's `reserve_holdout`/`finalize`/`release` + the reserve-only Protocol).

**Spec:** `docs/superpowers/specs/2026-06-13-holdout-interval-match-issue-192-design.md`

**Quality gate (run between tasks):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

- `algua/backtest/engine.py` — extract `_adj_grid(bars)` (shared by `build_portfolio`); add `holdout_window()`.
- `tests/test_holdout_window.py` — **new**: unit + cross-check tests for `holdout_window`.
- `algua/registry/db.py` — `SCHEMA_VERSION` 22→23; add `holdout_start/holdout_end` to CREATE TABLE + `_add_missing_columns`; add `_backfill_holdout_intervals`; update the table comment.
- `tests/test_db_migrations.py` — **add**: v23 backfill migration test.
- `algua/registry/store.py` — `reserve_holdout` gains `holdout_start/holdout_end`; interval-overlap matcher (NULL ⇒ fail-closed); INSERT populates the columns.
- `algua/registry/repository.py` — update the `reserve_holdout` Protocol signature + docstring.
- `algua/cli/research_cmd.py` — call `holdout_window` before reserve; pass the interval in.
- `tests/test_registry_store.py` — rewrite the holdout-reservation tests for the new signature + semantics (invert the different-frac test).
- `tests/test_concurrency.py` + `tests/_concurrency_worker.py` — thread `holdout_start/holdout_end` through the reserve race.

---

## Task 1: `holdout_window` helper (exact OOS interval from the bar index)

**Files:**
- Modify: `algua/backtest/engine.py` (extract `_adj_grid`; add `holdout_window`)
- Test: `tests/test_holdout_window.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_holdout_window.py`:

```python
from datetime import UTC, datetime

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import holdout_window
from algua.backtest.walkforward import walk_forward
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _equal_weight():
    from algua.portfolio.construction import get_construction_policy

    cfg = StrategyConfig(
        name="ew", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1), params={},
        construction="equal_weight_positive",
    )
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(1.0, index=sorted(v["symbol"].unique())),
        construct_fn=get_construction_policy(cfg.construction),
    )


def test_holdout_window_matches_walk_forward_burned_tail():
    # The linchpin: the reserved interval must equal what walk_forward actually burns.
    strat, prov = _equal_weight(), SyntheticProvider(seed=3)
    for frac in (0.2, 0.35):
        wf = walk_forward(strat, prov, START, END, windows=4, holdout_frac=frac)
        hs, he = holdout_window(strat, prov, START, END, holdout_frac=frac)
        assert (hs, he) == (wf.holdout_metrics["start"], wf.holdout_metrics["end"])


def test_holdout_end_is_last_actual_bar_not_period_end():
    # Period end is a weekend/holiday-heavy boundary; holdout_end is the last real session.
    strat, prov = _equal_weight(), SyntheticProvider(seed=1)
    hs, he = holdout_window(strat, prov, START, END, holdout_frac=0.2)
    bars = prov.get_bars(["AAA", "BBB"], START, END, "1d")
    last_session = bars.index.max().date().isoformat()
    assert he == last_session
    assert hs < he


def test_holdout_window_empty_bars_returns_conservative_period():
    strat = _equal_weight()

    class _Empty:
        def get_bars(self, symbols, start, end, timeframe):
            return SyntheticProvider().get_bars([], start, end, timeframe)

    hs, he = holdout_window(strat, _Empty(), START, END, holdout_frac=0.2)
    assert (hs, he) == (START.date().isoformat(), END.date().isoformat())


def test_holdout_window_tiny_frac_rounds_to_zero_returns_full_grid():
    # int(n * frac) == 0 -> degenerate; return the full grid (no IndexError). walk_forward will
    # later raise + the reservation is released, so the exact value is immaterial but must not crash.
    strat, prov = _equal_weight(), SyntheticProvider(seed=2)
    bars = prov.get_bars(["AAA", "BBB"], START, END, "1d")
    hs, he = holdout_window(strat, prov, START, END, holdout_frac=1e-6)
    assert hs == bars.index.min().date().isoformat()
    assert he == bars.index.max().date().isoformat()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_holdout_window.py -q`
Expected: FAIL — `ImportError: cannot import name 'holdout_window'`.

- [ ] **Step 3: Extract `_adj_grid` and add `holdout_window` in `algua/backtest/engine.py`**

Find the inline pivot inside `simulate`/`build_portfolio` (after the `bars.empty` guard):

```python
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()
```

Replace it with a call to a new module-level helper, and define the helper + `holdout_window` near the top of the module (after the imports / `_fetch_symbols` is available — `holdout_window` must be defined after `_fetch_symbols`). Add this helper:

```python
def _adj_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """The simulation grid: adj_close pivoted to (timestamp index x symbol columns), sorted by
    time. This index IS the bar date-index `vectorbt` simulates on and `pf.returns()` carries, so
    it is the single source of truth for both `build_portfolio` and `holdout_window`."""
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return adj.sort_index()
```

In `simulate`, replace the two-line inline pivot with:

```python
    adj = _adj_grid(bars)
```

Then add `holdout_window` (place it after `_fetch_symbols`, which it calls):

```python
def holdout_window(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    holdout_frac: float,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
) -> tuple[str, str]:
    """The exact OOS holdout interval [start, end] (ISO dates) `walk_forward` would carve as the
    last `holdout_frac` of the simulation grid — computed from the bar date-index WITHOUT running
    the strategy. Reproduces `build_portfolio`'s grid (identical `n`), so the boundary is identical
    to `walk_forward`'s `holdout_metrics`. Computed at reserve time so the single-use guard can
    match on the bars that will actually be burned (issue #192).

    Degenerate inputs (no bars, or holdout rounds to <1 bar) return the conservative full
    grid/period: the subsequent `walk_forward` raises and the reservation is released, so the value
    is immaterial but stays fail-closed (a superset of any real tail)."""
    bars = provider.get_bars(_fetch_symbols(strategy, universe_by_date), start, end, "1d")
    if bars.empty:
        return start.date().isoformat(), end.date().isoformat()
    idx = _adj_grid(bars).index
    n = len(idx)
    holdout_n = int(n * holdout_frac)            # floor — mirrors _segment_bounds' int(n*frac)
    if holdout_n < 1:
        return idx[0].date().isoformat(), idx[-1].date().isoformat()
    train_n = n - holdout_n                       # 1 <= train_n <= n-1 for frac in (0, 1)
    return idx[train_n].date().isoformat(), idx[-1].date().isoformat()
```

Note: `LoadedStrategy`, `DataProvider`, `Mapping`, `Collection`, `date`, `datetime` are already imported in `engine.py` (used by `build_portfolio`); verify and add any missing import. `verify_signal_panel_parity` has the same inline pivot — optionally switch it to `_adj_grid(bars)` too for consistency (no behavior change).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_holdout_window.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (additive change; existing `build_portfolio` behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add algua/backtest/engine.py tests/test_holdout_window.py
git commit -m "feat(#192): holdout_window — exact OOS interval from the bar index"
```

---

## Task 2: Schema 22→23 — interval columns + conservative backfill

**Files:**
- Modify: `algua/registry/db.py`
- Test: `tests/test_db_migrations.py` (add a v23 test)

- [ ] **Step 1: Write the failing migration test**

Add to `tests/test_db_migrations.py` (model it on the existing pre-v22 `committed_at` test near line 59):

```python
def test_pre_v23_holdout_rows_backfill_to_full_period(tmp_path):
    """A holdout_evaluations row created WITHOUT holdout_start/holdout_end (pre-v23) gains them via
    migrate, backfilled to the conservative full period [period_start, period_end]."""
    import sqlite3

    from algua.registry.db import connect, migrate

    db = tmp_path / "r.db"
    conn = connect(db)
    # Minimal strategies row for the FK, then a pre-v23 holdout row lacking the interval columns.
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS strategies (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);"
        "INSERT INTO strategies(id, name) VALUES (1, 's');"
        "CREATE TABLE holdout_evaluations ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id INTEGER NOT NULL,"
        " data_source TEXT NOT NULL, snapshot_id TEXT, period_start TEXT NOT NULL,"
        " period_end TEXT NOT NULL, holdout_frac REAL NOT NULL, config_hash TEXT NOT NULL,"
        " reused INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, committed_at TEXT);"
    )
    conn.execute(
        "INSERT INTO holdout_evaluations"
        "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
        " config_hash, reused, created_at, committed_at)"
        " VALUES (1,'demo',NULL,'2022-01-01','2023-12-31',0.2,'h',0,'2022-01-01T00:00:00+00:00',"
        " '2022-02-01T00:00:00+00:00')",
    )
    conn.commit()

    migrate(conn)

    cols = {row["name"] for row in conn.execute("PRAGMA table_info(holdout_evaluations)")}
    assert {"holdout_start", "holdout_end"} <= cols
    row = conn.execute(
        "SELECT holdout_start, holdout_end FROM holdout_evaluations"
    ).fetchone()
    assert (row["holdout_start"], row["holdout_end"]) == ("2022-01-01", "2023-12-31")
    # No NULL-interval rows remain.
    n_null = conn.execute(
        "SELECT COUNT(*) AS c FROM holdout_evaluations"
        " WHERE holdout_start IS NULL OR holdout_end IS NULL"
    ).fetchone()["c"]
    assert n_null == 0
    # Idempotent: a second migrate is a no-op.
    migrate(conn)
    conn.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_db_migrations.py::test_pre_v23_holdout_rows_backfill_to_full_period -q`
Expected: FAIL — `holdout_start` column missing (KeyError / assertion on `cols`).

- [ ] **Step 3: Update `algua/registry/db.py`**

(a) Bump the version:

```python
SCHEMA_VERSION = 23
```

(b) In the `_SCHEMA` `CREATE TABLE holdout_evaluations`, add the two columns after `committed_at` and update the table-doc comment. Add these column lines:

```sql
    holdout_start TEXT,          -- ISO date; OOS tail start (the matched single-use window, #192)
    holdout_end TEXT,            -- ISO date; OOS tail end (last actual bar date)
```

Update the comment block above the table so it describes interval matching (replace the
"OVERLAPPING period, same holdout_frac" sentence):

```
-- A later promote whose (strategy, data identity, OVERLAPPING out-of-sample INTERVAL
-- [holdout_start, holdout_end]) collides with a recorded row is REFUSED unless the operator passes
-- --allow-holdout-reuse. The interval is the exact bars walk_forward burns (issue #192); period_*
-- and holdout_frac are retained as evidence only, no longer part of identity. A NULL interval (a
-- legacy/old-writer row) matches unconditionally (fail closed).
```

(c) In `migrate()`, extend the existing `holdout_evaluations` `_add_missing_columns` call to add the new columns, then call the backfill. Replace:

```python
    _add_missing_columns(conn, "holdout_evaluations", {"committed_at": "TEXT"})
```

with:

```python
    _add_missing_columns(
        conn,
        "holdout_evaluations",
        {"committed_at": "TEXT", "holdout_start": "TEXT", "holdout_end": "TEXT"},
    )
    _backfill_holdout_intervals(conn)
```

(d) Add the backfill function (near the other `_migrate_*` / `_add_missing_columns` helpers):

```python
def _backfill_holdout_intervals(conn: sqlite3.Connection) -> None:
    """Backfill v23 holdout_start/holdout_end on legacy rows to the CONSERVATIVE full period
    [period_start, period_end]. The exact OOS tail cannot be recomputed at migration time (no data
    provider here), and the full period is a guaranteed superset of any real tail -> fail closed
    (may over-block a new run overlapping a legacy burn's period, the acceptable direction). Only
    touches rows missing an interval, so a row written by the new reserve path (interval already
    set) is never overwritten; deterministic, so concurrent/repeat runs converge. Idempotent."""
    conn.execute(
        "UPDATE holdout_evaluations SET holdout_start = period_start, holdout_end = period_end"
        " WHERE holdout_start IS NULL OR holdout_end IS NULL"
    )
    leftover = conn.execute(
        "SELECT COUNT(*) AS c FROM holdout_evaluations"
        " WHERE holdout_start IS NULL OR holdout_end IS NULL"
    ).fetchone()["c"]
    if leftover:
        raise RuntimeError(
            f"holdout interval backfill left {leftover} NULL-interval row(s); refusing to stamp v23"
        )
```

- [ ] **Step 4: Run the migration test + the existing migration suite**

Run: `uv run pytest tests/test_db_migrations.py tests/test_registry_db.py -q`
Expected: PASS (new test green; existing migration tests still green — additive columns).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: green. (New rows still insert NULL into the new columns at this point — the matcher is updated in Task 3 — but no test asserts the interval columns yet, so the old behavior holds.)

- [ ] **Step 6: Commit**

```bash
git add algua/registry/db.py tests/test_db_migrations.py
git commit -m "feat(#192): schema v23 holdout interval columns + conservative backfill"
```

---

## Task 3: Interval matcher in `reserve_holdout` + Protocol + CLI wiring (atomic)

This task changes the `reserve_holdout` signature, so it MUST update every caller (production CLI + direct-call tests) in the same commit to keep the gate green.

**Files:**
- Modify: `algua/registry/store.py` (`reserve_holdout`)
- Modify: `algua/registry/repository.py` (Protocol signature + docstring)
- Modify: `algua/cli/research_cmd.py` (compute + pass the interval)
- Modify: `tests/test_registry_store.py` (rewrite holdout tests)
- Modify: `tests/test_concurrency.py` + `tests/_concurrency_worker.py` (thread the interval)

- [ ] **Step 1: Rewrite the store holdout tests for the new signature + semantics**

In `tests/test_registry_store.py`, the holdout-reservation block (currently ~lines 126–280) drives `reserve_holdout` with `period_start/period_end/holdout_frac` and asserts the OLD semantics. Replace that whole block with tests that pass the explicit interval and assert interval-overlap semantics. Key changes:
- Every `reserve_holdout(...)` call gains `holdout_start=..., holdout_end=...` (keep `period_start/period_end/holdout_frac` as evidence args — they remain required params).
- **Invert** the old "different `holdout_frac` ⇒ distinct rows, no block" test into "overlapping interval ⇒ blocks regardless of `holdout_frac`".
- Add a "non-overlapping interval ⇒ allowed" test and a "NULL-interval legacy row ⇒ fail-closed match" test.

Use this block (adjust the helper/fixture names to match the file's existing `repo`/strategy setup — reuse whatever fixture the surrounding tests use to get a `repo` + a registered strategy id):

```python
# --- holdout reservation: interval matching (#192) --------------------------

def _reserve(repo, sid, *, hs, he, frac=0.2, ps="2022-01-01", pe="2023-12-31",
             ds="demo", snap=None, allow_reuse=False):
    return repo.reserve_holdout(
        sid, data_source=ds, snapshot_id=snap, period_start=ps, period_end=pe,
        holdout_frac=frac, holdout_start=hs, holdout_end=he, allow_reuse=allow_reuse)


def test_overlapping_interval_blocks(repo_with_strategy):
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2023-09-01", he="2024-03-01")


def test_different_holdout_frac_same_interval_still_blocks(repo_with_strategy):
    # The #192 exploit: a different --holdout-frac must NOT escape the guard when the OOS bars
    # overlap. Identity is the interval, not the frac.
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2023-06-01", he="2023-12-31", frac=0.2)
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2023-06-01", he="2023-12-31", frac=0.4)


def test_non_overlapping_interval_allowed(repo_with_strategy):
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2021-06-01", he="2021-12-31", ps="2020-01-01", pe="2021-12-31")
    rid, reused = _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")
    assert rid and reused is False


def test_allow_reuse_overrides_overlap(repo_with_strategy):
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")
    rid, reused = _reserve(repo, sid, hs="2023-06-01", he="2023-12-31", allow_reuse=True)
    assert rid and reused is True


def test_null_interval_row_fails_closed(repo_with_strategy):
    # An old-code/legacy row with a NULL interval must match unconditionally (fail closed).
    repo, sid = repo_with_strategy
    repo._conn.execute(
        "INSERT INTO holdout_evaluations"
        "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
        " config_hash, reused, created_at, committed_at, holdout_start, holdout_end)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (sid, "demo", None, "2022-01-01", "2023-12-31", 0.2, "h", 0,
         "2022-01-01T00:00:00+00:00", "2022-02-01T00:00:00+00:00", None, None),
    )
    repo._conn.commit()
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2023-06-01", he="2023-12-31")


def test_finalize_and_release_unchanged(repo_with_strategy):
    repo, sid = repo_with_strategy
    rid, _ = _reserve(repo, sid, hs="2022-06-01", he="2022-12-31")
    repo.finalize_holdout_reservation(rid, config_hash="real-evidence-hash")
    row = repo._conn.execute(
        "SELECT committed_at, config_hash FROM holdout_evaluations WHERE id = ?", (rid,)
    ).fetchone()
    assert row["committed_at"] and row["config_hash"] == "real-evidence-hash"

    rid2, _ = _reserve(repo, sid, hs="2024-06-01", he="2024-12-31",
                       ps="2024-01-01", pe="2024-12-31")
    repo.release_holdout_reservation(rid2)
    gone = repo._conn.execute(
        "SELECT 1 FROM holdout_evaluations WHERE id = ?", (rid2,)
    ).fetchone()
    assert gone is None
```

If the file lacks a `repo_with_strategy` fixture, add one near the top of the holdout block built from the surrounding tests' existing setup (a `SqliteStrategyRepository` over a migrated in-memory/temp connection + one registered strategy returning `(repo, sid)`). Reuse the file's existing connection/migrate helper rather than inventing a new one.

- [ ] **Step 2: Run the rewritten store tests to verify they fail**

Run: `uv run pytest tests/test_registry_store.py -k holdout -q`
Expected: FAIL — `reserve_holdout() got an unexpected keyword argument 'holdout_start'`.

- [ ] **Step 3: Update `reserve_holdout` in `algua/registry/store.py`**

Add the two params and change the SELECT + INSERT. The new signature and body:

```python
    def reserve_holdout(
        self,
        strategy_id: int,
        *,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        holdout_start: str,
        holdout_end: str,
        allow_reuse: bool,
    ) -> tuple[int, bool]:
```

Keep the top-level-transaction guard and the data-identity branch (`snapshot_id` vs `data_source`) exactly as they are. Replace the SELECT with interval-overlap + NULL fail-closed:

```python
            row = self._conn.execute(
                f"SELECT 1 FROM holdout_evaluations WHERE strategy_id = ?"
                f" AND {data_match}"
                f" AND (holdout_start IS NULL OR holdout_end IS NULL"
                f"      OR (holdout_start <= ? AND ? <= holdout_end)) LIMIT 1",
                (strategy_id, data_param, holdout_end, holdout_start),
            ).fetchone()
```

(Note the bind order: `data_param`, then `holdout_end` (compared `holdout_start <= ?`), then `holdout_start` (compared `? <= holdout_end`).)

Update the INSERT to populate the interval columns:

```python
            cur = self._conn.execute(
                "INSERT INTO holdout_evaluations"
                "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
                " config_hash, reused, created_at, committed_at, holdout_start, holdout_end)"
                " VALUES (?,?,?,?,?,?,?,?,?,NULL,?,?)",
                (strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,
                 "", int(reused), _now(), holdout_start, holdout_end),
            )
```

Update the method's docstring/comment to say matching is on the interval (not the period+frac). The `ValueError` message stays as-is.

- [ ] **Step 4: Update the Protocol in `algua/registry/repository.py`**

Add `holdout_start: str` and `holdout_end: str` to the `reserve_holdout` signature (after `holdout_frac`, before `allow_reuse`) and revise the docstring to describe interval matching:

```python
    def reserve_holdout(
        self,
        strategy_id: int,
        *,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        holdout_start: str,
        holdout_end: str,
        allow_reuse: bool,
    ) -> tuple[int, bool]:
        """Atomically claim the holdout window; return ``(reservation_id, reused)``.

        Under ``BEGIN IMMEDIATE`` (write lock held): re-check overlap against ALL rows (pending
        reservation OR committed burn) for this strategy + data identity whose stored OOS interval
        ``[holdout_start, holdout_end]`` overlaps ``[holdout_start, holdout_end]`` (a NULL-interval
        legacy row matches unconditionally — fail closed), then INSERT a pending row
        (``committed_at=NULL``, placeholder ``config_hash=''``). Match is on the OOS INTERVAL (the
        exact bars walk_forward burns, #192), never config; ``period_*`` and ``holdout_frac`` are
        evidence only.

        Raises ``ValueError`` (fail closed) if an overlapping row exists and not ``allow_reuse``.
        ``reused`` is True iff an overlapping row existed and the human override let it proceed.

        TOP-LEVEL ONLY: must not be called inside an open transaction / ``with self._conn:`` block
        (raises ``RuntimeError`` if ``self._conn.in_transaction``)."""
        ...
```

- [ ] **Step 5: Wire the CLI in `algua/cli/research_cmd.py`**

Add the import (with the other backtest imports):

```python
from algua.backtest.engine import holdout_window
```

Immediately before the `repo.reserve_holdout(` call, compute the interval, and add the two args to the call:

```python
        holdout_start, holdout_end = holdout_window(
            strategy, provider, start_dt, end_dt,
            holdout_frac=holdout_frac, universe_by_date=universe_by_date)
        reservation_id, reused = repo.reserve_holdout(
            repo.get(name).id, data_source=data_source, snapshot_id=snapshot_id,
            period_start=period_start, period_end=period_end, holdout_frac=holdout_frac,
            holdout_start=holdout_start, holdout_end=holdout_end,
            allow_reuse=allow_holdout_reuse)
```

Confirm `provider`, `universe_by_date`, `start_dt`, `end_dt`, `strategy`, `holdout_frac` are all in scope at that point (they are — they're built earlier in `promote`).

- [ ] **Step 6: Thread the interval through the concurrency reserve race**

In `tests/_concurrency_worker.py`, update `op_reserve_holdout` to accept and forward the interval:

```python
def op_reserve_holdout(db_path, barrier, wid, name, period_start, period_end, holdout_frac,
                       holdout_start, holdout_end):
    """Race the atomic holdout reservation (BEGIN IMMEDIATE re-check + insert)."""
    conn = connect(Path(db_path))
    repo = SqliteStrategyRepository(conn)
    sid = _strategy_id(conn, name)
    touch(barrier, f"ready-{wid}")
    wait_for(barrier, "go")
    touch(barrier, f"attempting-{wid}")
    try:
        rid, reused = repo.reserve_holdout(
            sid, data_source="demo", snapshot_id=None,
            period_start=period_start, period_end=period_end,
            holdout_frac=float(holdout_frac),
            holdout_start=holdout_start, holdout_end=holdout_end, allow_reuse=False)
        _emit({"ok": True, "wid": wid, "rid": rid, "reused": reused})
    except ValueError as exc:
        _emit({"ok": False, "wid": wid, "error": "ValueError", "msg": str(exc)})
    except Exception as exc:
        _emit({"ok": False, "wid": wid, "error": type(exc).__name__, "msg": str(exc)})
```

In `tests/test_concurrency.py`, update `test_concurrent_reserve_holdout_single_burn`:
- The worker args list now must include the interval. Both racers should claim the SAME overlapping interval so exactly one wins (the test's existing assertion). Pass `holdout_start="2023-06-01", holdout_end="2023-12-31"` (or matching kwargs) to both worker invocations.
- The direct `SqliteStrategyRepository(c1).reserve_holdout(...)` / `c2` calls near line 390–395 (if the test pre-seeds or asserts directly) gain `holdout_start=...`, `holdout_end=...` too.

Read the test body and add the two values everywhere `reserve_holdout` / the worker arg-tuple is constructed, keeping both racers on the same interval.

- [ ] **Step 7: Run the affected suites**

Run: `uv run pytest tests/test_registry_store.py tests/test_concurrency.py tests/test_cli_research.py tests/test_promotion.py -q`
Expected: PASS. (`test_cli_research`/`test_promotion` exercise `promote` end-to-end — they confirm the CLI wiring + new signature integrate.)

- [ ] **Step 8: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. `lint-imports` confirms `registry` did not gain a forbidden import (it received date strings; `holdout_window` lives in `backtest`, called by the CLI).

- [ ] **Step 9: Commit**

```bash
git add algua/registry/store.py algua/registry/repository.py algua/cli/research_cmd.py \
        tests/test_registry_store.py tests/test_concurrency.py tests/_concurrency_worker.py
git commit -m "feat(#192): match holdout single-use on the OOS interval, not period+frac"
```

---

## Task 4: End-to-end exploit regression via the CLI + final verification

**Files:**
- Test: `tests/test_cli_research.py` (add one end-to-end test)

- [ ] **Step 1: Write the failing end-to-end exploit test**

Add a test to `tests/test_cli_research.py` that drives `promote` twice on the same period with different `--holdout-frac` and asserts the second is refused as a re-burn. Model it on the file's existing `promote` tests (reuse their CLI runner, demo/snapshot provider, and a registered `backtested` strategy). Shape:

```python
def test_promote_different_holdout_frac_is_refused_as_reburn(<existing fixtures>):
    # First promote at frac=0.2 burns the OOS tail.
    first = <invoke promote with --holdout-frac 0.2>
    # Re-running at a different frac whose OOS tail overlaps must now be refused (issue #192),
    # not silently re-peek the holdout.
    second = <invoke promote with --holdout-frac 0.4 on the same/overlapping period>
    assert second["ok"] is False
    assert "holdout already consumed" in second["error"]
```

Match the file's existing assertion style for the JSON error envelope (`ok=False` + `error`). If the existing tests use a helper to run promote and parse JSON, reuse it verbatim.

- [ ] **Step 2: Run it to verify it fails (then passes)**

Run: `uv run pytest tests/test_cli_research.py::test_promote_different_holdout_frac_is_refused_as_reburn -q`
Expected: PASS already (the behavior is implemented in Task 3) — this test LOCKS it in end-to-end. If it FAILS, the CLI wiring or interval computation is wrong; debug before proceeding. (If the test only fails because of fixture wiring, fix the fixtures, not the production code.)

- [ ] **Step 3: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cli_research.py
git commit -m "test(#192): end-to-end — varying --holdout-frac no longer re-burns the holdout"
```

---

## Self-review checklist (done while writing)

- **Spec coverage:** holdout_window/exact interval (T1) ✓; schema v23 + backfill + zero-NULL assert (T2) ✓; interval matcher + NULL fail-closed + Protocol + CLI seam (T3) ✓; cross-check test (T1) ✓; exploit regression unit (T3) + e2e (T4) ✓; conservative legacy backfill (T2) ✓; deferred snapshot-identity follow-up is a non-goal (file the issue at finish). 
- **Green between tasks:** T1/T2 additive; T3 changes the signature and updates ALL callers in one commit. ✓
- **No placeholders:** concrete code/edits given; the two test-fixture reuses (store `repo_with_strategy`, CLI promote runner) point at existing patterns rather than inventing infra. ✓
- **Type consistency:** `holdout_window(...) -> tuple[str, str]`; `reserve_holdout(..., holdout_start: str, holdout_end: str, allow_reuse: bool)` identical in store + Protocol + all callers. ✓
