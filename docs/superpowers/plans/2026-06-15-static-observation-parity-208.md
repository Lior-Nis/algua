# Static-mode Observation Parity (issue #208) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In static-mode backtests, project the strategy-visible `bars`/`adj` down to the operating universe (`strategy.universe ∩ adj.columns`, order-preserving) so an undeclared symbol a misbehaving provider returns never enters the strategy's `view`, `signal_panel`, weights/grid, or fundamentals/news sidecars — closing the observation-parity gap left by #179.

**Architecture:** One private helper `_static_operating_view(strategy, bars, adj) -> (bars, adj)` in `algua/backtest/engine.py`, applied at the two static-mode sites that build the view from full fetched data (`simulate`'s `universe_by_date is None` branch and `verify_signal_panel_parity`). The helper absorbs the previously-duplicated empty-universe guard and fails closed on an empty operating universe. Column-only projection on `adj` (rows/index untouched) keeps `holdout_window`'s grid and the #192 single-use holdout identity unchanged. PIT and live paths are not touched.

**Tech Stack:** Python, pandas, vectorbt, pytest. Quality gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

**Design spec:** `docs/superpowers/specs/2026-06-15-static-observation-parity-208-design.md`

---

## File Structure

- **Modify** `algua/backtest/engine.py`:
  - Add `_static_operating_view(strategy, bars, adj)` (new private helper, near `_adj_grid`/`simulate`).
  - `simulate` (~L567): replace the `if universe_by_date is None:` empty-guard block with the helper call.
  - `verify_signal_panel_parity` (~L424–433): replace its empty-guard block with the helper call.
  - Add a one-line comment at the two `static_universe = set(strategy.universe) & set(columns)` sites (`_decision_weights` ~L170, `_fast_weights` ~L268) noting it is now redundant-but-kept defense-in-depth post-projection.
- **Create** `tests/test_static_observation_parity.py`: all new tests for this change.

No other production files change. `holdout_window` is deliberately NOT modified (see spec GATE-1 decisions: column-only projection preserves the grid index).

---

### Task 1: Helper + `simulate` wiring (loop-path observation leak closed)

**Files:**
- Modify: `algua/backtest/engine.py` (add `_static_operating_view`; rewire `simulate` static branch; add defense-in-depth comments)
- Test: `tests/test_static_observation_parity.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_static_observation_parity.py`:

```python
"""Issue #208: static-mode observation parity. A misbehaving provider that returns an UNDECLARED
symbol (one not in strategy.universe) must never have that symbol's data reach the strategy's view,
panel, weights/grid, or fundamentals/news sidecars. Mirror of #179, which closed the out-of-universe
WEIGHT path; this closes the OBSERVATION path."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError, run, simulate, verify_signal_panel_parity
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)


def _passthrough(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Identity construction: scores ARE the desired raw weights."""
    return scores


class _ExtraSymbolProvider:
    """A misbehaving provider: returns bars for the requested symbols PLUS an undeclared `extra`."""

    def __init__(self, extra: str = "ZZZ", seed: int = 0) -> None:
        self.extra = extra
        self.seed = seed

    def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
        requested = list(symbols)
        return SyntheticProvider(seed=self.seed).get_bars(
            requested + [self.extra], start, end, timeframe
        )


class _ViewRecorder:
    """A 2-arg signal that records every symbol it is shown and returns FLAT weights (so the
    observation check is isolated from the #179 weight-rejection path)."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        self.seen.update(view["symbol"].unique())
        return pd.Series(dtype="float64")


def _loop_strategy(signal) -> LoadedStrategy:  # noqa: ANN001
    cfg = StrategyConfig(
        name="obs_loop", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=_passthrough)


def test_loop_view_excludes_undeclared_symbol() -> None:
    recorder = _ViewRecorder()
    strat = _loop_strategy(recorder)
    run(strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END)
    assert recorder.seen, "signal was never invoked"
    assert "ZZZ" not in recorder.seen
    assert recorder.seen <= {"AAA", "BBB"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_static_observation_parity.py::test_loop_view_excludes_undeclared_symbol -v`
Expected: FAIL — `assert "ZZZ" not in recorder.seen` fails (today the full panel, including ZZZ, enters the view).

- [ ] **Step 3: Add the `_static_operating_view` helper**

In `algua/backtest/engine.py`, add this helper immediately after `_adj_grid` (the function ending ~L492, before `holdout_window`):

```python
def _static_operating_view(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project the strategy-visible STATIC view to the operating universe (declared AND available)
    so a misbehaving provider's undeclared symbols never reach the loop view, the fast-path
    signal_panel, the weights/grid, or the fundamentals/news sidecars (observation parity, #208).

    Fails closed when no declared symbol has fetched price data (empty operating universe) — this
    absorbs #179's empty-intersection guard AND the empty-declared-universe case. The projection on
    `adj` is COLUMN-ONLY (adj.index is untouched), so holdout_window's grid and the #192 single-use
    holdout identity are unaffected. No-op for a compliant provider (adj.columns subset of universe).
    """
    universe = set(strategy.universe)
    # Order-preserving intersection: keep adj's existing column order so a compliant provider is a
    # STRICT no-op (no reorder, no NaN/reindex-fill since operating is a subset of adj.columns).
    operating = [c for c in adj.columns if c in universe]
    if not operating:
        raise BacktestError(
            f"no fetched price data for any symbol in strategy {strategy.name!r} declared "
            f"universe {sorted(strategy.universe)} (fetched columns: "
            f"{sorted(map(str, adj.columns))})"
        )
    return bars[bars["symbol"].isin(operating)], adj.loc[:, operating]
```

- [ ] **Step 4: Rewire `simulate`'s static branch to use the helper**

In `simulate`, replace this block (currently ~L567–574):

```python
    if universe_by_date is None:
        operating_universe = set(strategy.universe) & set(adj.columns)
        if strategy.universe and not operating_universe:
            raise BacktestError(
                f"no fetched price data for any symbol in strategy {strategy.name!r} declared "
                f"universe {sorted(strategy.universe)} (fetched columns: "
                f"{sorted(map(str, adj.columns))})"
            )
```

with:

```python
    if universe_by_date is None:
        # Static mode: project the strategy-visible view + grid to the operating universe so an
        # undeclared symbol a misbehaving provider returned cannot influence in-universe decisions
        # (observation parity, #208). PIT keeps its per-bar as-of mask instead.
        bars, adj = _static_operating_view(strategy, bars, adj)
```

- [ ] **Step 5: Run the failing test to verify it now passes**

Run: `uv run pytest tests/test_static_observation_parity.py::test_loop_view_excludes_undeclared_symbol -v`
Expected: PASS.

- [ ] **Step 6: Add defense-in-depth comments at the two `static_universe` sites**

In `_decision_weights` (~L167–170) the comment block above `static_universe = set(strategy.universe) & set(columns)` already explains the intent. Append one sentence to that comment (and the identical one in `_fast_weights` ~L265–268):

```
    # ... (existing comment) ...
    # Post-#208 this equals set(columns) on the static path (adj is already projected in simulate),
    # but it is KEPT as defense-in-depth: it still fails closed if this private fn is ever called
    # with an unprojected adj (e.g. directly from a test).
```

- [ ] **Step 7: Run the gate, then commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

```bash
git add algua/backtest/engine.py tests/test_static_observation_parity.py
git commit -m "feat(208): project static backtest view to the operating universe (observation parity)"
```

---

### Task 2: Fast-path observation leak closed (`signal_panel` sees only declared)

**Files:**
- Test: `tests/test_static_observation_parity.py` (append)

- [ ] **Step 1: Write the test**

Append to `tests/test_static_observation_parity.py`:

```python
class _PanelRecorder:
    """A signal_panel that records the symbols it is handed and returns a FLAT (all-zero) scores
    matrix. Paired with a flat 2-arg signal so the fast-path parity guard (which compares the panel
    against the per-bar loop) holds."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(self, bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        self.seen.update(bars["symbol"].unique())
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.0, index=adj.index, columns=adj.columns)


def _fast_strategy(panel: _PanelRecorder) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="obs_fast", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    # Flat loop twin so the fast path's parity guard agrees with the panel (both produce 0.0).
    def flat_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(0.0, index=syms) if syms else pd.Series(dtype="float64")

    return LoadedStrategy(
        config=cfg, signal_fn=flat_loop, signal_panel_fn=panel, construct_fn=_passthrough
    )


def test_fast_path_panel_excludes_undeclared_symbol() -> None:
    panel = _PanelRecorder()
    strat = _fast_strategy(panel)
    # run() drives simulate(), which selects the fast path (signal_panel_fn set, static mode).
    run(strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END)
    assert panel.seen, "signal_panel was never invoked"
    assert "ZZZ" not in panel.seen
    assert panel.seen <= {"AAA", "BBB"}
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_static_observation_parity.py::test_fast_path_panel_excludes_undeclared_symbol -v`
Expected: PASS (the fast path goes through `simulate`, which now projects before `_fast_weights`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_static_observation_parity.py
git commit -m "test(208): fast-path signal_panel never sees an undeclared symbol"
```

---

### Task 3: `verify_signal_panel_parity` projection (promotion gate consistency)

**Files:**
- Modify: `algua/backtest/engine.py` (`verify_signal_panel_parity`)
- Test: `tests/test_static_observation_parity.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_static_observation_parity.py`:

```python
def test_verify_signal_panel_parity_panel_excludes_undeclared_symbol() -> None:
    panel = _PanelRecorder()
    strat = _fast_strategy(panel)
    # verify_signal_panel_parity fetches its own bars and runs both the panel and the loop.
    verify_signal_panel_parity(strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END)
    assert panel.seen, "signal_panel was never invoked"
    assert "ZZZ" not in panel.seen
    assert panel.seen <= {"AAA", "BBB"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_static_observation_parity.py::test_verify_signal_panel_parity_panel_excludes_undeclared_symbol -v`
Expected: FAIL — `verify_signal_panel_parity` still builds the panel from the full fetched bars (ZZZ leaks in).

- [ ] **Step 3: Rewire `verify_signal_panel_parity` to use the helper**

In `verify_signal_panel_parity`, replace this block (currently ~L424–433):

```python
        adj = _adj_grid(bars)
        # Same fail-closed guard as simulate(): an empty declared∩available universe would otherwise
        # surface as a confusing out-of-universe `(allowed: [])` breach instead of this clear cause.
        if strategy.universe and not (set(strategy.universe) & set(adj.columns)):
            raise BacktestError(
                f"no fetched price data for any symbol in strategy {strategy.name!r} declared "
                f"universe {sorted(strategy.universe)} (fetched columns: "
                f"{sorted(map(str, adj.columns))})"
            )
```

with:

```python
        adj = _adj_grid(bars)
        # Project to the operating universe identically to simulate()'s static path so the panel
        # under test sees exactly what the runtime fast path sees (observation parity, #208). Also
        # absorbs the empty-universe fail-closed guard.
        bars, adj = _static_operating_view(strategy, bars, adj)
```

- [ ] **Step 4: Run the test to verify it now passes**

Run: `uv run pytest tests/test_static_observation_parity.py::test_verify_signal_panel_parity_panel_excludes_undeclared_symbol -v`
Expected: PASS.

- [ ] **Step 5: Run the gate, then commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (existing `test_fast_path.py` parity tests still pass).

```bash
git add algua/backtest/engine.py tests/test_static_observation_parity.py
git commit -m "feat(208): project verify_signal_panel_parity to the operating universe"
```

---

### Task 4: Sidecar leak closed (fundamentals + news as-of frames)

**Files:**
- Test: `tests/test_static_observation_parity.py` (append)

- [ ] **Step 1: Write the test**

Append to `tests/test_static_observation_parity.py`:

```python
from algua.data.fundamentals_schema import to_fundamentals_schema
from algua.data.news_schema import to_news_schema


class _FundRecorder:
    """A 3-arg fundamentals signal that records the symbols present in the as-of fundamentals frame
    it is handed, and returns FLAT weights."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(
        self, view: pd.DataFrame, params: dict[str, Any], fundamentals: pd.DataFrame
    ) -> pd.Series:
        self.seen.update(fundamentals["symbol"].unique())
        return pd.Series(dtype="float64")


class _ExtraFundamentalsProvider:
    """Returns fundamentals for the requested symbols PLUS the undeclared `extra` (misbehaving)."""

    def __init__(self, extra: str = "ZZZ") -> None:
        self.extra = extra

    def get_fundamentals(self, symbols, end):  # noqa: ANN001
        rows = [
            [s, "2023-12-31", "eps_diluted", 1.0, "2023-12-31T00:00:00Z", "v"]
            for s in list(symbols) + [self.extra]
        ]
        raw = pd.DataFrame(rows, columns=[
            "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
        ])
        return to_fundamentals_schema(raw)


def test_fundamentals_sidecar_excludes_undeclared_symbol() -> None:
    recorder = _FundRecorder()
    cfg = StrategyConfig(
        name="obs_funds", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough", needs_fundamentals=True,
    )
    strat = LoadedStrategy(config=cfg, fundamentals_signal_fn=recorder, construct_fn=_passthrough)
    # Bars provider returns ZZZ too: without #208 projection, adj.columns would include ZZZ, so the
    # loop's `allowed = set(columns)` would NOT mask ZZZ out of the fundamentals frame.
    run(
        strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END,
        fundamentals_provider=_ExtraFundamentalsProvider(extra="ZZZ"),
    )
    assert recorder.seen, "fundamentals signal was never invoked"
    assert "ZZZ" not in recorder.seen


class _NewsRecorder:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def __call__(
        self, view: pd.DataFrame, params: dict[str, Any], news: pd.DataFrame
    ) -> pd.Series:
        self.seen.update(news["symbol"].unique())
        return pd.Series(dtype="float64")


class _ExtraNewsProvider:
    def __init__(self, extra: str = "ZZZ") -> None:
        self.extra = extra

    def get_news(self, symbols, end):  # noqa: ANN001
        rows = [
            [s, "2023-01-01T00:00:00Z", "src", f"art-{s}", "headline",
             "2023-01-01T00:00:00Z", False]
            for s in list(symbols) + [self.extra]
        ]
        raw = pd.DataFrame(rows, columns=[
            "symbol", "published_at", "source", "article_id", "headline",
            "knowable_at", "retracted",
        ])
        return to_news_schema(raw)


def test_news_sidecar_excludes_undeclared_symbol() -> None:
    recorder = _NewsRecorder()
    cfg = StrategyConfig(
        name="obs_news", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough", needs_news=True,
    )
    strat = LoadedStrategy(config=cfg, news_signal_fn=recorder, construct_fn=_passthrough)
    run(
        strat, _ExtraSymbolProvider(extra="ZZZ", seed=3), START, END,
        news_provider=_ExtraNewsProvider(extra="ZZZ"),
    )
    assert recorder.seen, "news signal was never invoked"
    assert "ZZZ" not in recorder.seen
```

- [ ] **Step 2: Confirm the news-schema column names**

Before running, verify the raw column names `to_news_schema` expects (it may differ from the list above). Run:

`uv run python -c "import inspect, algua.data.news_schema as m; print(inspect.getsource(m.to_news_schema))"`

If the expected raw columns differ, adjust `_ExtraNewsProvider.get_news`'s `columns=[...]` to match. (The fundamentals raw columns are confirmed from `tests/test_engine_symbol_mask.py`.)

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_static_observation_parity.py -k sidecar -v`
Expected: PASS — both fundamentals and news as-of frames exclude ZZZ (adj is projected, so `allowed = set(columns)` excludes ZZZ).

- [ ] **Step 4: Commit**

```bash
git add tests/test_static_observation_parity.py
git commit -m "test(208): fundamentals + news sidecars never see an undeclared symbol"
```

---

### Task 5: No-op for compliant providers + fail-closed regressions

**Files:**
- Test: `tests/test_static_observation_parity.py` (append)

- [ ] **Step 1: Write the tests**

Append to `tests/test_static_observation_parity.py`:

```python
def test_compliant_provider_is_a_noop() -> None:
    """A compliant provider (returns exactly the declared universe) must produce byte-identical
    results before/after projection: same metrics. Uses an equal-weight strategy and SyntheticProvider."""

    def ew(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(1.0 / len(syms), index=syms) if syms else pd.Series(dtype="float64")

    strat = _loop_strategy(ew)
    res = run(strat, SyntheticProvider(seed=3), START, END)
    # Compliant provider: every declared symbol is present, no extras → operating == adj.columns.
    # Sanity: the run completes and produces the standard metric keys (no projection-induced break).
    for key in ["total_return", "sharpe", "n_rebalances", "avg_gross_exposure"]:
        assert key in res.metrics


def test_compliant_provider_preserves_weight_columns_and_order() -> None:
    """Projection is order-preserving and a strict no-op for a compliant provider: the effective
    weights cover exactly the declared universe, in adj-column order."""

    def ew(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(1.0 / len(syms), index=syms) if syms else pd.Series(dtype="float64")

    strat = _loop_strategy(ew)
    _pf, weights_eff = simulate(strat, SyntheticProvider(seed=3), START, END)
    assert list(weights_eff.columns) == ["AAA", "BBB"]


def test_provider_returns_only_undeclared_fails_closed() -> None:
    """All declared symbols missing (provider returns only an undeclared symbol) → fail closed."""

    class _OnlyWrongProvider:
        def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
            return SyntheticProvider(seed=0).get_bars(["ZZZ"], start, end, timeframe)

    strat = _loop_strategy(_ViewRecorder())
    with pytest.raises(BacktestError, match="no fetched price data for any symbol"):
        run(strat, _OnlyWrongProvider(), START, END)


def test_empty_declared_universe_fails_closed_if_provider_returns_data() -> None:
    """Empty declared universe + a (contract-violating) provider that returns data for an empty
    request → fail closed (operating universe is empty). #208 reversed the prior 'show full panel'
    behavior; this is a no-op for compliant providers (empty request → no bars → earlier guard)."""

    class _DataForEmptyRequestProvider:
        def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
            # Ignore the (empty) request and return data anyway — a double contract violation.
            return SyntheticProvider(seed=0).get_bars(["AAA", "BBB"], start, end, timeframe)

    cfg = StrategyConfig(
        name="obs_empty", universe=[],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(config=cfg, signal_fn=_ViewRecorder(), construct_fn=_passthrough)
    with pytest.raises(BacktestError, match="no fetched price data for any symbol"):
        run(strat, _DataForEmptyRequestProvider(), START, END)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_static_observation_parity.py -v`
Expected: ALL pass. If `test_compliant_provider_preserves_weight_columns_and_order` fails on column order, confirm `_adj_grid` sorts columns (`AAA` before `BBB`) — the assertion matches pivot+sort order; adjust the expected list only if the synthetic provider yields a different order.

- [ ] **Step 3: Confirm the pre-existing empty-universe test still passes**

Run: `uv run pytest tests/test_backtest_engine.py::test_empty_universe_data_raises -v`
Expected: PASS (a compliant provider returns no bars for `[]`, so `simulate` still raises at the earlier "provider returned no bars" guard, before the helper).

- [ ] **Step 4: Run the full gate, then commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

```bash
git add tests/test_static_observation_parity.py
git commit -m "test(208): no-op for compliant providers + empty/undeclared fail-closed regressions"
```

---

## Self-Review

**Spec coverage:**
- Helper + simulate wiring → Task 1. ✓
- verify_signal_panel_parity wiring → Task 3. ✓
- Loop view leak closed → Task 1. ✓ Fast-path panel leak closed → Task 2. ✓
- Fundamentals + news sidecars → Task 4. ✓
- No-op for compliant providers → Task 5. ✓ Empty-intersection + empty-universe fail-closed → Task 5. ✓
- Order-preserving projection → helper (Task 1) + asserted in Task 5. ✓
- `holdout_window` deliberately untouched (column-only projection) → documented in spec; no task. ✓
- Defense-in-depth comment → Task 1 Step 6. ✓

**Placeholder scan:** No TBD/TODO. Every code step shows complete code. Task 4 Step 2 is a verification step (confirm `to_news_schema` raw columns) with an exact command, not a placeholder — included because the news raw-column names are the one schema not directly confirmed from an existing test.

**Type consistency:** `_static_operating_view(strategy, bars, adj) -> (bars, adj)` defined in Task 1, called identically in Tasks 1 & 3. `_ExtraSymbolProvider`, `_ViewRecorder`, `_PanelRecorder`, `_loop_strategy`, `_fast_strategy`, `_passthrough` defined once and reused. Test fixtures match real signatures verified against `tests/test_fast_path.py` and `tests/test_engine_symbol_mask.py`.
