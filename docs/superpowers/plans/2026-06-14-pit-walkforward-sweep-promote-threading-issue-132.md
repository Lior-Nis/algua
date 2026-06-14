# PIT walk_forward / sweep / promote threading (issue #132) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `needs_fundamentals` / `needs_news` strategy be walk-forwarded, swept, and `research promote`d from `backtested` to `candidate` — unblocking the research funnel for both PIT lanes.

**Architecture:** Thread `fundamentals_provider` / `news_provider` through `walk_forward`, `sweep` (incl. its parallel workers), and the three CLIs that drive them; remove the `_reject_pit_sidecar` guard AND the independent `promotion_preflight` PIT blockers, relying on the engine's existing "needs_X but no provider → fail closed"; stamp the PIT snapshot into `WalkForwardResult` / `SweepResult` / the `gate_evaluations` audit row.

**Tech Stack:** Python 3.12, pandas, vectorbt, typer, pytest, sqlite. Parallel sweep is `ProcessPoolExecutor` (#169) — provider must stay picklable.

**Spec:** `docs/superpowers/specs/2026-06-14-pit-walkforward-sweep-promote-threading-issue-132-design.md`

**Quality gate (run between tasks):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

**Note on base:** review/build against `origin/main` (local main is stale). The build worktree is created off `origin/main`.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `algua/backtest/walkforward.py` | modify | `walk_forward` gains provider kwargs + passes to `build_portfolio`; `WalkForwardResult` gains PIT snapshot fields; remove `_reject_pit_sidecar` call (Task 1) then the function (Task 2) |
| `algua/backtest/sweep.py` | modify | `sweep` + `_evaluate_combo` gain provider kwargs; `SweepResult` + meta gain PIT snapshot; drop `_reject_pit_sidecar` call/import |
| `algua/registry/promotion.py` | modify | remove the `needs_fundamentals`/`needs_news` blockers in `promotion_preflight`; `run_gate` passes `wf`'s PIT snapshot to `record_gate_evaluation` |
| `algua/registry/db.py` | modify | `gate_evaluations` gains `fundamentals_snapshot`/`news_snapshot` cols + migration; `SCHEMA_VERSION` 23→24 |
| `algua/registry/store.py`, `repository.py` | modify | `record_gate_evaluation` gains optional kw params + INSERT them |
| `algua/cli/backtest_cmd.py` | modify | `walk_forward_cmd` + `sweep_cmd` gain `--fundamentals-snapshot`/`--news-snapshot` + misuse guards + provider wiring |
| `algua/cli/research_cmd.py` | modify | `promote` gains the options + misuse + early fail-closed + provider wiring; `dormant_sweep` skip reason updated |
| `tests/...` | modify/create | rewrite the guard + promotion-block tests; add threading, provenance, CLI, e2e, picklability tests |

---

## Task 1: `walk_forward` — thread providers + PIT provenance + drop its guard call

**Files:**
- Modify: `algua/backtest/walkforward.py`
- Test: `tests/test_walkforward_pit.py` (create); `tests/test_wf_sweep_pit_guard.py` (update WF cases)

- [ ] **Step 1: Write failing tests** in `tests/test_walkforward_pit.py`

```python
from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import walk_forward
from algua.contracts.types import ExecutionContract
from algua.data.serve import StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import LoadedStrategy, StrategyConfig

START, END = datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)


def _news_strategy():
    return LoadedStrategy(
        config=StrategyConfig(
            name="news_wf", universe=["AAPL", "MSFT", "NVDA"],
            execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
            construction="equal_weight_positive", needs_news=True),
        news_signal_fn=lambda view, params, news: pd.Series(dtype="float64"),
        construct_fn=get_construction_policy("equal_weight_positive"))


def _news_provider(tmp_path):
    store = DataStore(tmp_path)
    raw = pd.DataFrame([
        {"source": "r", "article_id": "a1", "symbols": ["AAPL"],
         "published_at": "2023-02-01T00:00:00Z", "knowable_at": "2023-02-01T00:00:00Z",
         "headline": "h"},
    ])
    rec = store.ingest_news(provider="t", as_of="2023-03-01T00:00:00Z", frame=raw)
    return StoreBackedNewsProvider(store, rec.snapshot_id), rec.snapshot_id


def test_walk_forward_runs_with_news_provider_and_stamps_snapshot(tmp_path):
    prov, sid = _news_provider(tmp_path)
    wf = walk_forward(_news_strategy(), SyntheticProvider(seed=1), START, END,
                      news_provider=prov)
    assert wf.news_snapshot == sid
    assert wf.fundamentals_snapshot is None


def test_walk_forward_without_provider_fails_closed():
    with pytest.raises(BacktestError, match="needs_news"):
        walk_forward(_news_strategy(), SyntheticProvider(seed=1), START, END)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_walkforward_pit.py -q`
Expected: FAIL (`walk_forward` has no `news_provider` kwarg; `WalkForwardResult` has no `news_snapshot`).

- [ ] **Step 3: Implement in `algua/backtest/walkforward.py`**

Extend the contracts import:
```python
from algua.contracts.types import DataProvider, FundamentalsProvider, NewsProvider
```
Add two fields to the `WalkForwardResult` dataclass (after `universe_snapshots`):
```python
    # PIT sidecar snapshot provenance (issue #132); None unless the strategy is needs_*.
    fundamentals_snapshot: str | None = None
    news_snapshot: str | None = None
```
Add the provider kwargs to `walk_forward` (after `on_peek`):
```python
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
```
Delete the `_reject_pit_sidecar(strategy, "walk-forward")` line (keep the function for now — sweep still imports it; Task 2 removes it). Update the `build_portfolio(...)` call:
```python
    pf, _weights = build_portfolio(strategy, provider, start, end,
                                   universe_by_date=universe_by_date,
                                   fundamentals_provider=fundamentals_provider,
                                   news_provider=news_provider)
```
In the `return WalkForwardResult(...)`, add (only stamp news when the lane is active, mirroring engine.run):
```python
        fundamentals_snapshot=(
            getattr(fundamentals_provider, "snapshot_id", None)
            if strategy.config.needs_fundamentals else None),
        news_snapshot=(
            getattr(news_provider, "snapshot_id", None)
            if strategy.config.needs_news else None),
```

- [ ] **Step 4: Update the existing guard test's WF cases**

In `tests/test_wf_sweep_pit_guard.py`, the WF-rejection tests (`test_walk_forward_rejects_needs_news`, `test_walk_forward_rejects_needs_fundamentals`) now assert the WRONG behavior. Change them to the new contract: a PIT strategy walk-forwarded WITHOUT a provider fails closed with `match="needs_news"` / `"needs_fundamentals"` (the engine's message), not "not supported in walk-forward". (The with-provider success is covered in `test_walkforward_pit.py`.) Leave the sweep cases for Task 2.

- [ ] **Step 5: Run + gate**

Run: `uv run pytest tests/test_walkforward_pit.py tests/test_wf_sweep_pit_guard.py tests/test_tracking_walkforward.py tests/test_cli_walkforward.py -q && uv run mypy algua`
Expected: PASS (the WF CLI test still passes — plain strategies unaffected; CLI provider wiring is Task 5).

- [ ] **Step 6: Commit**

```bash
git add algua/backtest/walkforward.py tests/test_walkforward_pit.py tests/test_wf_sweep_pit_guard.py
git commit -m "feat(132): thread PIT providers + snapshot provenance through walk_forward"
```

---

## Task 2: `sweep` — thread providers (incl. parallel workers) + PIT provenance; delete the guard

**Files:**
- Modify: `algua/backtest/sweep.py`
- Test: `tests/test_sweep_pit.py` (create); `tests/test_wf_sweep_pit_guard.py` (update sweep cases)

- [ ] **Step 1: Write failing tests** in `tests/test_sweep_pit.py`

```python
import pickle
from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
from algua.backtest.sweep import sweep
from algua.contracts.types import ExecutionContract
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import LoadedStrategy, StrategyConfig

START, END = datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)


def _news_strategy():
    return LoadedStrategy(
        config=StrategyConfig(
            name="news_sweep", universe=["AAPL", "MSFT", "NVDA"],
            execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
            params={"window_days": 5}, construction="equal_weight_positive", needs_news=True),
        news_signal_fn=lambda view, params, news: pd.Series(dtype="float64"),
        construct_fn=get_construction_policy("equal_weight_positive"))


def _news_provider(tmp_path):
    store = DataStore(tmp_path)
    raw = pd.DataFrame([{"source": "r", "article_id": "a1", "symbols": ["AAPL"],
                         "published_at": "2023-02-01T00:00:00Z",
                         "knowable_at": "2023-02-01T00:00:00Z", "headline": "h"}])
    rec = store.ingest_news(provider="t", as_of="2023-03-01T00:00:00Z", frame=raw)
    return StoreBackedNewsProvider(store, rec.snapshot_id), rec.snapshot_id


def test_provider_is_picklable(tmp_path):
    prov, _ = _news_provider(tmp_path)
    pickle.loads(pickle.dumps(prov))  # must round-trip for the ProcessPoolExecutor path
    fprov = StoreBackedFundamentalsProvider(DataStore(tmp_path), "x")
    pickle.loads(pickle.dumps(fprov))


def test_sweep_parallel_with_news_provider_stamps_snapshot(tmp_path):
    prov, sid = _news_provider(tmp_path)
    # >1 combo -> exercises the worker pool (picklability of the bound provider)
    res = sweep(_news_strategy(), SyntheticProvider(seed=1), START, END,
                grid={"window_days": [3, 5]}, news_provider=prov)
    assert res.news_snapshot == sid
    assert res.n_combos == 2


def test_sweep_without_provider_fails_closed():
    with pytest.raises(BacktestError, match="needs_news"):
        sweep(_news_strategy(), SyntheticProvider(seed=1), START, END,
              grid={"window_days": [3, 5]})
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_sweep_pit.py -q`
Expected: FAIL (`sweep` has no `news_provider`; `SweepResult` has no `news_snapshot`).

- [ ] **Step 3: Implement in `algua/backtest/sweep.py`**

Add the contracts import:
```python
from algua.contracts.types import DataProvider, FundamentalsProvider, NewsProvider
```
Remove `_reject_pit_sidecar` from the walkforward import (leave `walk_forward`):
```python
from algua.backtest.walkforward import walk_forward
```
Add two fields to `SweepResult` (after `universe_snapshots`):
```python
    fundamentals_snapshot: str | None = None
    news_snapshot: str | None = None
```
Add provider params to `_evaluate_combo` (after `rank_by`):
```python
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
```
Pass them into `_evaluate_combo`'s `walk_forward(...)` call, and add to its returned `meta` dict:
```python
    wf = walk_forward(
        overridden, provider, start, end,
        windows=windows, holdout_frac=holdout_frac,
        universe_by_date=universe_by_date,
        universe_name=universe_name, universe_snapshots=universe_snapshots,
        fundamentals_provider=fundamentals_provider, news_provider=news_provider,
    )
    return {
        ...
        "meta": {
            ...
            "universe_snapshots": wf.universe_snapshots,
            "fundamentals_snapshot": wf.fundamentals_snapshot,
            "news_snapshot": wf.news_snapshot,
        },
    }
```
Add the provider params to `sweep` (after `universe_snapshots`), drop the
`_reject_pit_sidecar(strategy, "sweep")` line, and add the providers to `eval_kwargs`:
```python
    eval_kwargs: dict[str, Any] = dict(
        provider=provider, start=start, end=end,
        windows=windows, holdout_frac=holdout_frac,
        universe_by_date=universe_by_date,
        universe_name=universe_name, universe_snapshots=universe_snapshots,
        rank_by=rank_by,
        fundamentals_provider=fundamentals_provider, news_provider=news_provider,
    )
```
In the `return SweepResult(...)`, read from `meta`:
```python
        fundamentals_snapshot=meta["fundamentals_snapshot"],
        news_snapshot=meta["news_snapshot"],
```
Now `_reject_pit_sidecar` has no callers — **delete the function** from `algua/backtest/walkforward.py`.

- [ ] **Step 4: Update the guard test's sweep cases + delete now-obsolete guard tests**

In `tests/test_wf_sweep_pit_guard.py`: the sweep-rejection tests now assert the wrong behavior — change `test_sweep_rejects_needs_news` (and any fundamentals sweep-reject) to assert a sweep WITHOUT a provider fails closed (`match="needs_news"`). Keep `test_override_preserves_news_signal_fn` and ADD `test_override_preserves_fundamentals_signal_fn` (mirror it, asserting `_override(_fund_strategy(), {}).fundamentals_signal_fn is ...`). The file's `_news_strategy`/`_fund_strategy` helpers already exist.

- [ ] **Step 5: Run + gate**

Run: `uv run pytest tests/test_sweep_pit.py tests/test_wf_sweep_pit_guard.py tests/test_sweep.py tests/test_sweep_override_fundamentals.py tests/test_cli_sweep.py -q && uv run mypy algua`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add algua/backtest/sweep.py algua/backtest/walkforward.py tests/test_sweep_pit.py tests/test_wf_sweep_pit_guard.py
git commit -m "feat(132): thread PIT providers through sweep + workers; drop _reject_pit_sidecar"
```

---

## Task 3: Remove the `promotion_preflight` PIT blockers (the funnel enabler)

**Files:**
- Modify: `algua/registry/promotion.py`
- Test: `tests/test_promotion_needs_news.py` (rewrite); `tests/test_promotion_needs_fundamentals.py` (create)

- [ ] **Step 1: Rewrite the failing test** `tests/test_promotion_needs_news.py`

The current test asserts `promotion_preflight` REFUSES a `needs_news` strategy with "needs_news". That block is being removed. Rewrite so the test asserts the strategy now **passes the PIT check** and proceeds (it will then fail on the NEXT preflight check — measured breadth — with a DIFFERENT message, since the temp strategy has no recorded sweep). Keep the temp-module + registry setup; change the assertion:

```python
    # After #132 slice 4: needs_news no longer blocks promotion. Preflight now proceeds past the
    # (removed) PIT block and fails later on MISSING MEASURED BREADTH instead — proving the PIT
    # block is gone, not that promotion is free.
    with pytest.raises(ValueError, match="no recorded search breadth"):
        promotion_preflight(
            repo, name, actor=Actor.AGENT, declared_combos=None,
            allow_holdout_reuse=False, allow_non_pit=False,
            provider=SyntheticProvider(seed=0),
            start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2024, 6, 1, tzinfo=UTC))
```
(Confirm the exact breadth-refusal message in `promotion.py` — it is "no recorded search breadth for {name!r}" — and match a stable substring.)

Create `tests/test_promotion_needs_fundamentals.py` as the fundamentals twin: a temp `needs_fundamentals` module, same assertion (passes the PIT check, fails on breadth).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_promotion_needs_news.py tests/test_promotion_needs_fundamentals.py -q`
Expected: FAIL (the strategy is still refused with "needs_news"/"needs_fundamentals" because the block is still there).

- [ ] **Step 3: Remove the blocks** in `algua/registry/promotion.py`

Delete BOTH guard blocks in `promotion_preflight`:
```python
    if _loaded is not None and _loaded.config.needs_fundamentals:
        raise ValueError(... "needs_fundamentals" ...)
    if _loaded is not None and _loaded.config.needs_news:
        raise ValueError(... "needs_news" ...)
```
Keep the `load_strategy` try/except and `_loaded` (still used by the `verify_signal_panel_parity` line below). Do NOT touch `assert_tradable_without_*` in `strategies/base.py` (paper/live trading guards stay).

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_promotion_needs_news.py tests/test_promotion_needs_fundamentals.py tests/test_promotion.py tests/test_fundamentals_guards.py -q`
Expected: PASS. (`test_fundamentals_guards.py`'s paper/live helper tests are untouched and still pass.)

- [ ] **Step 5: Commit**

```bash
git add algua/registry/promotion.py tests/test_promotion_needs_news.py tests/test_promotion_needs_fundamentals.py
git commit -m "feat(132): allow backtested->candidate for PIT strategies (drop promote blockers)"
```

---

## Task 4: `gate_evaluations` PIT snapshot provenance

**Files:**
- Modify: `algua/registry/db.py`, `algua/registry/store.py`, `algua/registry/repository.py`, `algua/registry/promotion.py`
- Test: `tests/test_gate_evaluation_pit_provenance.py` (create)

- [ ] **Step 1: Write the failing test** `tests/test_gate_evaluation_pit_provenance.py`

```python
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository


def _repo(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_record_gate_evaluation_persists_pit_snapshots(tmp_path):
    repo = _repo(tmp_path)
    repo.add("s")
    sid = repo.get("s").id
    rid = repo.record_gate_evaluation(
        sid, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=30, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=63, min_holdout_observations=63, code_hash="c", config_hash="cfg",
        dependency_hash="d", data_source="SyntheticProvider", snapshot_id="bars1",
        period_start="2023-01-01", period_end="2023-12-31", holdout_frac=0.2, actor="agent",
        decision_json="{}", news_snapshot="news1", fundamentals_snapshot=None)
    row = repo._conn.execute(
        "SELECT news_snapshot, fundamentals_snapshot FROM gate_evaluations WHERE id=?",
        (rid,)).fetchone()
    assert row["news_snapshot"] == "news1"
    assert row["fundamentals_snapshot"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_gate_evaluation_pit_provenance.py -q`
Expected: FAIL (`record_gate_evaluation` has no `news_snapshot` kwarg / column absent).

- [ ] **Step 3: Schema + migration** in `algua/registry/db.py`

Bump `SCHEMA_VERSION = 24`. Add two columns to the `gate_evaluations` CREATE TABLE (after `snapshot_id TEXT,`):
```sql
    fundamentals_snapshot TEXT,
    news_snapshot TEXT,
```
In `migrate()`, add (next to the other `_add_missing_columns` calls):
```python
    # v24 (#132): PIT sidecar snapshot provenance on the gate audit row. Additive nullable — legacy
    # rows stay NULL (no backfill: pre-#132 promotions had no PIT snapshot).
    _add_missing_columns(
        conn, "gate_evaluations",
        {"fundamentals_snapshot": "TEXT", "news_snapshot": "TEXT"})
```

- [ ] **Step 4: Thread through the store + protocol**

In `algua/registry/repository.py` and `algua/registry/store.py`, add to `record_gate_evaluation` (as optional keyword-only, after `decision_json`):
```python
        fundamentals_snapshot: str | None = None,
        news_snapshot: str | None = None,
```
In the store INSERT, add the two columns + params (extend the column list, the `VALUES` placeholders, and the value tuple):
```python
                "(... holdout_frac, actor, decision_json, fundamentals_snapshot, news_snapshot,"
                " consumed, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
                (... holdout_frac, actor, decision_json, fundamentals_snapshot, news_snapshot,
                 _now()),
```
(Carefully match the placeholder count — there are now 23 `?` before `0` and one after for `created_at`.)

- [ ] **Step 5: `run_gate` passes the wf snapshots** in `algua/registry/promotion.py`

In `run_gate`, the `repo.record_gate_evaluation(...)` call gains:
```python
        fundamentals_snapshot=wf.fundamentals_snapshot, news_snapshot=wf.news_snapshot,
```
(`wf` is the `WalkForwardResult` param run_gate already receives; Task 1 added these fields.)

- [ ] **Step 6: Run + gate**

Run: `uv run pytest tests/test_gate_evaluation_pit_provenance.py tests/test_promotion.py tests/test_concurrency.py -q && uv run mypy algua`
Expected: PASS (concurrency/migration tests confirm the additive schema change is safe).

- [ ] **Step 7: Commit**

```bash
git add algua/registry/db.py algua/registry/store.py algua/registry/repository.py algua/registry/promotion.py tests/test_gate_evaluation_pit_provenance.py
git commit -m "feat(132): record PIT sidecar snapshot on the gate_evaluations audit row (schema v24)"
```

---

## Task 5: CLI — `backtest walk-forward` + `backtest sweep` PIT options

**Files:**
- Modify: `algua/cli/backtest_cmd.py`
- Test: `tests/test_cli_backtest_pit_eval.py` (create)

- [ ] **Step 1: Write failing tests** `tests/test_cli_backtest_pit_eval.py`

Mirror `tests/test_cli_backtest_news.py`'s `_seed` (ingest bars for AAPL/MSFT/NVDA + a news snapshot whose articles fall in the window). Then:
```python
def test_backtest_sweep_with_news_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, nid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "sweep", "news_coverage_tilt",
                              "--snapshot", bid, "--news-snapshot", nid,
                              "--param", "window_days=3,5",
                              "--start", "2025-01-01", "--end", "2025-01-10"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["news_snapshot"] == nid


def test_backtest_walkforward_with_news_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, nid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "walk-forward", "news_coverage_tilt",
                              "--snapshot", bid, "--news-snapshot", nid,
                              "--start", "2025-01-01", "--end", "2025-01-10"])
    assert res.exit_code == 0, res.output


def test_sweep_news_snapshot_on_non_news_strategy_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, nid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum",
                              "--snapshot", bid, "--news-snapshot", nid,
                              "--param", "lookback=2,3",
                              "--start", "2025-01-01", "--end", "2025-01-10"])
    assert res.exit_code != 0
    assert "needs_news" in res.output
```
(Confirm `walk-forward` is the registered command name; the function is `walk_forward_cmd`. Confirm the `cross_sectional_momentum` param key for the sweep grid — it is `lookback`.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cli_backtest_pit_eval.py -q`
Expected: FAIL (no `--news-snapshot` on these commands).

- [ ] **Step 3: Implement** in `algua/cli/backtest_cmd.py`

For BOTH `walk_forward_cmd` and `sweep_cmd`, mirror the block already in `run` (same file): add the two options, the two misuse guards, build the providers, and pass them through. Add options:
```python
    fundamentals_snapshot: str = typer.Option(
        None, "--fundamentals-snapshot",
        help="ingested fundamentals snapshot id (required for a needs_fundamentals strategy)"),
    news_snapshot: str = typer.Option(
        None, "--news-snapshot",
        help="ingested news snapshot id (required for a needs_news strategy)"),
```
After `resolve_eval_inputs`:
```python
    if fundamentals_snapshot and not strategy.config.needs_fundamentals:
        raise ValueError("--fundamentals-snapshot was given but the strategy does not declare "
                         "needs_fundamentals")
    if news_snapshot and not strategy.config.needs_news:
        raise ValueError("--news-snapshot was given but the strategy does not declare needs_news")
    fundamentals_provider = (
        StoreBackedFundamentalsProvider(DataStore(get_settings().data_dir), fundamentals_snapshot)
        if fundamentals_snapshot else None)
    news_provider = (
        StoreBackedNewsProvider(DataStore(get_settings().data_dir), news_snapshot)
        if news_snapshot else None)
```
Pass `fundamentals_provider=fundamentals_provider, news_provider=news_provider` into the
`walk_forward(...)` and `sweep(...)` calls respectively.

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_cli_backtest_pit_eval.py tests/test_cli_sweep.py tests/test_cli_walkforward.py tests/test_cli_backtest_news.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/backtest_cmd.py tests/test_cli_backtest_pit_eval.py
git commit -m "feat(132): backtest walk-forward/sweep gain --fundamentals/--news-snapshot"
```

---

## Task 6: CLI — `research promote` PIT options + end-to-end funnel unblock

**Files:**
- Modify: `algua/cli/research_cmd.py`
- Test: `tests/test_cli_research_promote_pit.py` (create)

- [ ] **Step 1: Write failing tests** `tests/test_cli_research_promote_pit.py`

Seed bars + a news snapshot, register `news_coverage_tilt`, advance to `backtested`, run `backtest sweep ... --news-snapshot` (to record breadth), then `research promote ... --news-snapshot` and assert it reaches `candidate`. Mirror an existing promote CLI test for the registry/seed mechanics. Also:
```python
def test_promote_news_snapshot_on_non_news_strategy_errors(...):
    # --news-snapshot for a plain strategy -> error mentioning needs_news
def test_promote_needs_news_without_snapshot_errors_before_reservation(...):
    # promote news_coverage_tilt WITHOUT --news-snapshot -> error mentioning needs_news;
    # assert NO holdout reservation row was created (repo holdout table empty).
def test_promote_news_end_to_end_reaches_candidate(...):
    # full path -> stage == candidate; gate_evaluations row has news_snapshot == nid
def test_promote_fundamentals_end_to_end_reaches_candidate(...):
    # the fundamentals twin (fundamentals_earnings_tilt + a fundamentals snapshot)
```
(For the e2e, the strategy must actually PASS the gate on the fixture data; if tuning thresholds is needed, pass relaxed human-only flags is NOT allowed for an agent — instead choose fixture data / criteria so the deterministic synthetic result passes, or assert the gate was REACHED and evaluated (decision present in payload) rather than necessarily passed. Prefer asserting `candidate` reached; if the synthetic fixture can't pass cleanly, assert the gate ran and produced a decision + the gate row carries the snapshot, and note it.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cli_research_promote_pit.py -q`
Expected: FAIL (no `--news-snapshot` on `promote`).

- [ ] **Step 3: Implement** in `algua/cli/research_cmd.py` `promote`

Add the two options (mirror Task 5). After `strategy` is resolved and BEFORE `reserve_holdout`/`holdout_window`, add the misuse guards AND the early fail-closed:
```python
    if fundamentals_snapshot and not strategy.config.needs_fundamentals:
        raise ValueError("--fundamentals-snapshot was given but the strategy does not declare "
                         "needs_fundamentals")
    if news_snapshot and not strategy.config.needs_news:
        raise ValueError("--news-snapshot was given but the strategy does not declare needs_news")
    if strategy.config.needs_fundamentals and not fundamentals_snapshot:
        raise ValueError("strategy declares needs_fundamentals; pass --fundamentals-snapshot")
    if strategy.config.needs_news and not news_snapshot:
        raise ValueError("strategy declares needs_news; pass --news-snapshot")
    fundamentals_provider = (
        StoreBackedFundamentalsProvider(DataStore(get_settings().data_dir), fundamentals_snapshot)
        if fundamentals_snapshot else None)
    news_provider = (
        StoreBackedNewsProvider(DataStore(get_settings().data_dir), news_snapshot)
        if news_snapshot else None)
```
(Place these after `promotion_preflight` is fine too, but BEFORE `reserve_holdout`. Simplest: right after `resolve_universe_inputs`, before `promotion_preflight`.) Add the imports if absent:
`from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider` and
`from algua.data.store import DataStore` / `from algua.config.settings import get_settings`
(check what `research_cmd.py` already imports; `resolve_eval_inputs` may already pull settings).
Pass the providers into the `walk_forward(...)` call only:
```python
            wf = walk_forward(
                strategy, provider, start_dt, end_dt, windows=windows,
                holdout_frac=holdout_frac, universe_by_date=universe_by_date,
                universe_name=universe, universe_snapshots=universe_prov,
                fundamentals_provider=fundamentals_provider, news_provider=news_provider,
                on_peek=lambda cfg: repo.finalize_holdout_reservation(reservation_id, config_hash=cfg),
            )
```

- [ ] **Step 4: Run + gate**

Run: `uv run pytest tests/test_cli_research_promote_pit.py -q && uv run pytest tests/test_forward_promotion.py tests/test_cli_research.py -q`
Expected: PASS (existing promote tests for plain strategies unaffected).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/research_cmd.py tests/test_cli_research_promote_pit.py
git commit -m "feat(132): research promote gains --fundamentals/--news-snapshot (funnel unblock)"
```

---

## Task 7: `research dormant-sweep` — honest skip reason

**Files:**
- Modify: `algua/cli/research_cmd.py` (`dormant_sweep`)
- Test: the existing dormant-sweep test (find via `grep -rln dormant_sweep tests/`)

- [ ] **Step 1: Update the failing test**

Find the dormant-sweep test that asserts a PIT strategy is skipped with reason "walk-forward lane not wired" (grep `grep -rn "lane not wired\|dormant_sweep\|dormant-sweep" tests/`). Change the expected `reason` substring to the new accurate text (Step 3).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest <that test file> -q`
Expected: FAIL (reason text mismatch).

- [ ] **Step 3: Update the skip reason** in `dormant_sweep`

```python
            if sidecar is not None:
                skipped.append({"strategy": rec.name,
                                "reason": f"{sidecar}: dormant-sweep takes no per-strategy PIT "
                                          f"snapshot; re-audition individually via "
                                          f"backtest walk-forward/research promote "
                                          f"--{'fundamentals' if sidecar == 'needs_fundamentals' "
                                          f"else 'news'}-snapshot"})
                continue
```
(Keep the rest of the skip logic — dormant-sweep still skips PIT strategies; only the reason changes.)

- [ ] **Step 4: Run + gate**

Run: `uv run pytest <that test file> -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/research_cmd.py tests/<that test file>
git commit -m "docs(132): accurate dormant-sweep PIT skip reason (lane now wired, no per-strategy snapshot)"
```

---

## Final verification

- [ ] Full gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- [ ] Confirm the headline behavior: a `needs_news` AND a `needs_fundamentals` strategy can be swept (breadth recorded) and `research promote`d to `candidate`; the `gate_evaluations` row carries the PIT snapshot.
- [ ] Confirm fail-closed: each of walk_forward / sweep / promote errors clearly when a PIT strategy is run without its provider; `promote` errors before creating a holdout reservation.
- [ ] Confirm paper/live trading guards still block PIT strategies (untouched).
