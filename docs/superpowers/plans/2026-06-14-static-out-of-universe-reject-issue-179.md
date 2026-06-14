# Static backtest: reject out-of-universe target weights (#179) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every algua decision path reject a nonzero target weight for a symbol outside its operating universe through the single shared `validate_decision_weights` rail, so static backtest, PIT backtest, and live can never drift (today static silently drops, PIT rejects, live trades).

**Architecture:** Add one `check_universe_membership` to `algua/risk/limits.py` and fold it into `validate_decision_weights` via a new REQUIRED `allowed_symbols` param (order: finite → universe → short → cap → gross). Each caller passes its operating universe: live → `strategy.universe`; static backtest → `set(strategy.universe) & set(adj.columns)`; PIT → per-bar `members`. Delete the bespoke inline PIT non-member block; make `_canonical_row` a faithful loop-proxy by running the full rail; fail closed in `simulate` when the static operating universe is empty.

**Tech Stack:** Python, pandas, pytest. Spec: `docs/superpowers/specs/2026-06-13-static-out-of-universe-reject-issue-179-design.md`.

**Quality gate (run before each commit):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

- **Modify** `algua/risk/limits.py` — new `check_universe_membership`; `validate_decision_weights` gains required `allowed_symbols`.
- **Modify** `algua/backtest/engine.py` *(CODEOWNERS — covered by the signed-off design spec)* — `_decision_weights` (static + PIT) and `_fast_weights` pass `allowed_symbols`; delete the inline PIT `non_members` block; `_canonical_row` runs the full rail; `simulate` fails closed on an empty static operating universe.
- **Modify** `algua/live/paper_loop.py` — `decide()` passes `allowed_symbols=strategy.universe`.
- **Modify** `tests/test_risk_limits.py` — new `check_universe_membership` unit tests; the 5 existing `validate_decision_weights` calls gain `allowed_symbols`; universe-ordering assertions.
- **Modify** `tests/test_decision_parity.py` — out-of-universe breaches both paths (static loop + paper).
- **Modify** `tests/test_fast_path.py` — out-of-universe rejected via `_fast_weights` and via `_canonical_row`.
- **Modify** `tests/test_backtest_engine.py` — empty static operating universe fails closed.

---

## Task 1: `check_universe_membership` in the risk rail

**Files:**
- Modify: `algua/risk/limits.py`
- Test: `tests/test_risk_limits.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_risk_limits.py` (after `test_short_policy_allows_negatives_when_allow_short`):

```python
def test_universe_membership_passes_in_universe_and_zeros():
    from algua.risk.limits import check_universe_membership
    # in-universe nonzero weights pass
    check_universe_membership(pd.Series({"AAA": 0.6, "BBB": 0.4}), {"AAA", "BBB"}, "s")
    # a ZERO weight for an out-of-universe symbol passes (mirrors `!= 0.0`)
    check_universe_membership(pd.Series({"AAA": 0.5, "ZZZ": 0.0}), {"AAA"}, "s")
    # empty weights pass
    check_universe_membership(pd.Series(dtype="float64"), set(), "s")


def test_universe_membership_breaches_on_out_of_universe_nonzero():
    from algua.risk.limits import RiskBreach, check_universe_membership
    with pytest.raises(RiskBreach) as ei:
        check_universe_membership(pd.Series({"AAA": 0.5, "ZZZ": 0.5}), {"AAA", "BBB"}, "s")
    assert ei.value.kind == "out_of_universe"
    assert "ZZZ" in ei.value.detail and "s" in ei.value.detail


def test_universe_membership_empty_allowed_breaches_any_nonzero():
    from algua.risk.limits import RiskBreach, check_universe_membership
    with pytest.raises(RiskBreach) as ei:
        check_universe_membership(pd.Series({"AAA": 0.5}), set(), "s")
    assert ei.value.kind == "out_of_universe"


def test_universe_membership_non_string_label_does_not_raise_typeerror():
    from algua.risk.limits import RiskBreach, check_universe_membership
    # mixed/non-string offender labels must render via key=str, not raise a bare TypeError
    with pytest.raises(RiskBreach) as ei:
        check_universe_membership(pd.Series([0.5, 0.5], index=["AAA", 7]), {"AAA"}, "s")
    assert ei.value.kind == "out_of_universe"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_risk_limits.py -k universe_membership -q`
Expected: FAIL with `ImportError: cannot import name 'check_universe_membership'`.

- [ ] **Step 3: Add the `Collection` import**

In `algua/risk/limits.py`, change the `TYPE_CHECKING` block (top of file):

```python
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Collection

    from algua.contracts.types import ExecutionContract
```

- [ ] **Step 4: Implement `check_universe_membership`**

In `algua/risk/limits.py`, add immediately AFTER `check_finite_weights` (before `check_max_weight_per_symbol`):

```python
def check_universe_membership(
    weights: pd.Series, allowed_symbols: Collection[str], strategy_name: str
) -> None:
    """Reject any NONZERO target weight for a symbol outside the operating universe — the structural
    twin of the value checks. Mirrors the PIT loop's `w != 0.0` 'nonzero' semantics exactly: any
    nonzero weight for a non-member is a strategy bug (if numeric noise ever makes this too strict it
    can move to WEIGHT_TOL without changing the architecture). Offenders/allowed are rendered with
    `key=str` so a non-string symbol label cannot raise a bare TypeError that escapes the
    RiskBreach -> BacktestError / live-kill-switch contract. Empty `allowed_symbols` + any nonzero
    weight => every nonzero weight breaches (no allowed universe); a caller meaning "flat" must skip
    the call (as the PIT loop does via `if not members: continue`)."""
    if len(weights) == 0:
        return
    allowed = set(allowed_symbols)
    offenders = [s for s in weights.index[weights != 0.0] if s not in allowed]
    if offenders:
        raise RiskBreach(
            "out_of_universe",
            f"strategy '{strategy_name}' returned nonzero target weight(s) for out-of-universe "
            f"symbol(s) {sorted(offenders, key=str)} (allowed: {sorted(allowed, key=str)})",
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_risk_limits.py -k universe_membership -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the quality gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

```bash
git add algua/risk/limits.py tests/test_risk_limits.py
git commit -m "feat(179): add check_universe_membership risk rail"
```

---

## Task 2: Fold into `validate_decision_weights` + wire all callers (atomic)

This task changes `validate_decision_weights`'s signature to require `allowed_symbols`, so ALL callers (`decide`, `_decision_weights`, `_fast_weights`) and the 5 existing direct test calls must update in the same commit to keep the suite green.

**Files:**
- Modify: `algua/risk/limits.py` (`validate_decision_weights`)
- Modify: `algua/backtest/engine.py` (`_decision_weights` static+PIT; `_fast_weights`; delete PIT block)
- Modify: `algua/live/paper_loop.py` (`decide`)
- Test: `tests/test_risk_limits.py`, `tests/test_decision_parity.py`

- [ ] **Step 1: Write the failing integration test (both paths)**

Add to `tests/test_decision_parity.py` (after `test_named_symbol_nan_breaches_both_paths`):

```python
def test_out_of_universe_weight_breaches_both_paths() -> None:
    """A weight for a symbol OUTSIDE the declared universe must hard-fail backtest (silently dropped
    before #179) and be rejected in paper (silently traded before #179) — identical rejection."""
    cfg = StrategyConfig(
        name="cheat", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"ZZZ": 1.0}), construct_fn=_identity,
    )
    with pytest.raises(BacktestError) as ei:
        run(strat, SyntheticProvider(seed=0), START, END)
    assert isinstance(ei.value.__cause__, RiskBreach)
    assert ei.value.__cause__.kind == "out_of_universe"
    with pytest.raises(RiskBreach) as ei_paper:
        run_paper(strat, SimBroker(cash=1_000_000.0), SyntheticProvider(seed=0), START, END)
    assert ei_paper.value.kind == "out_of_universe"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_decision_parity.py::test_out_of_universe_weight_breaches_both_paths -q`
Expected: FAIL — backtest currently DROPS the ZZZ weight (no raise), so `run(...)` does not raise.

- [ ] **Step 3: Add required `allowed_symbols` to `validate_decision_weights`**

In `algua/risk/limits.py`, replace the function:

```python
def validate_decision_weights(
    weights: pd.Series,
    contract: ExecutionContract,
    strategy_name: str,
    allowed_symbols: Collection[str],
) -> None:
    """The ONE decision-weight validation every path calls (paper/live decide + backtest loop +
    fast-path), so the rails can never drift between research and live. Order: finite (fail-closed)
    -> universe membership -> short policy -> per-symbol cap -> gross exposure (#135, #179).
    `allowed_symbols` is the path's operating universe (live: strategy.universe; static backtest:
    strategy.universe & adj.columns; PIT: as-of members at t)."""
    check_finite_weights(weights, strategy_name)
    check_universe_membership(weights, allowed_symbols, strategy_name)
    check_short_policy(weights, contract.allow_short, strategy_name)
    check_max_weight_per_symbol(weights, contract.max_weight_per_symbol)
    check_gross_exposure(weights, contract.max_gross_exposure)
```

- [ ] **Step 4: Wire `paper_loop.decide()`**

In `algua/live/paper_loop.py`, change the call inside `decide` (currently line ~73):

```python
    weights = strategy.target_weights(view)
    validate_decision_weights(
        weights, strategy.execution, strategy.name, allowed_symbols=strategy.universe
    )
```

- [ ] **Step 5: Wire `_decision_weights` (static + PIT) and delete the inline PIT block**

In `algua/backtest/engine.py`, in `_decision_weights`: after `warmup = strategy.execution.warmup_bars` (line ~122), add the static operating universe:

```python
    columns = adj.columns
    warmup = strategy.execution.warmup_bars
    static_universe = set(strategy.universe) & set(columns)
```

Then DELETE the inline non-member block (currently lines ~150-156):

```python
        if universe_by_date is not None:
            non_members = [s for s in w.index[w != 0.0] if s not in members]
            if non_members:
                raise BacktestError(
                    f"strategy {strategy.name!r} returned weight for non-member symbol(s) "
                    f"{sorted(non_members)} at {t} (as-of members: {sorted(members)})"
                )
```

And change the validate call (currently lines ~159-162) to pass the per-mode allowed set:

```python
        try:
            allowed = members if universe_by_date is not None else static_universe
            validate_decision_weights(
                w, strategy.execution, strategy.name, allowed_symbols=allowed
            )
        except RiskBreach as breach:
            raise BacktestError(f"{breach.detail} at {t}") from breach
```

(`members` is in scope only when `universe_by_date is not None`, where it is computed at line ~135; the ternary only reads it in that branch.)

- [ ] **Step 6: Wire `_fast_weights`**

In `algua/backtest/engine.py`, in `_fast_weights`: after `columns = adj.columns` (line ~204), add:

```python
    columns = adj.columns
    static_universe = set(strategy.universe) & set(columns)
```

Change the validate call (currently lines ~221-224):

```python
        try:
            validate_decision_weights(
                w, strategy.execution, strategy.name, allowed_symbols=static_universe
            )
        except RiskBreach as breach:
            raise BacktestError(f"{breach.detail} at {t}") from breach
```

- [ ] **Step 7: Update the 5 existing direct calls in `test_risk_limits.py`**

In `tests/test_risk_limits.py::test_validate_decision_weights_runs_all_rails_in_order`, add `allowed_symbols={"AAA", "BBB"}` to each of the 5 `validate_decision_weights(...)` calls. Example (apply the same to all 5):

```python
    validate_decision_weights(
        pd.Series({"AAA": 0.6, "BBB": 0.4}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
    )
    ...
    with pytest.raises(RiskBreach) as ei_fin:
        validate_decision_weights(
            pd.Series({"AAA": np.nan}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
        )
    assert ei_fin.value.kind == "non_finite_weight"
    ...
    with pytest.raises(RiskBreach) as ei_short:
        validate_decision_weights(
            pd.Series({"AAA": -0.3}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
        )
    assert ei_short.value.kind == "long_only"
    ...
    with pytest.raises(RiskBreach) as ei_cap:
        validate_decision_weights(
            pd.Series({"AAA": 0.9}), _contract(max_weight_per_symbol=0.5), "s",
            allowed_symbols={"AAA", "BBB"},
        )
    assert ei_cap.value.kind == "max_weight_per_symbol"
    ...
    with pytest.raises(RiskBreach) as ei_gross:
        validate_decision_weights(
            pd.Series({"AAA": 0.7, "BBB": 0.7}), _contract(max_gross_exposure=1.0), "s",
            allowed_symbols={"AAA", "BBB"},
        )
    assert ei_gross.value.kind == "gross_exposure"
```

- [ ] **Step 8: Add a universe-ordering unit test**

Add to `tests/test_risk_limits.py`:

```python
def test_validate_decision_weights_universe_after_finite_before_value_checks():
    from algua.risk.limits import RiskBreach, validate_decision_weights

    # clean in-universe vector passes
    validate_decision_weights(
        pd.Series({"AAA": 0.6, "BBB": 0.4}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
    )
    # an out-of-universe nonzero weight breaches out_of_universe
    with pytest.raises(RiskBreach) as ei_u:
        validate_decision_weights(
            pd.Series({"AAA": 0.5, "ZZZ": 0.5}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
        )
    assert ei_u.value.kind == "out_of_universe"
    # finite runs BEFORE universe: a NaN on an out-of-universe symbol surfaces non_finite first
    import numpy as np
    with pytest.raises(RiskBreach) as ei_fin:
        validate_decision_weights(
            pd.Series({"ZZZ": np.nan}), _contract(), "s", allowed_symbols={"AAA", "BBB"}
        )
    assert ei_fin.value.kind == "non_finite_weight"
```

- [ ] **Step 9: Run the targeted tests**

Run: `uv run pytest tests/test_decision_parity.py::test_out_of_universe_weight_breaches_both_paths tests/test_risk_limits.py -q`
Expected: PASS. The existing PIT test still passes: `uv run pytest tests/test_backtest_engine.py::test_decision_weights_rejects_non_member_weight -q` → PASS (the unified rail raises `BacktestError` matching `"BBB"`).

- [ ] **Step 10: Run the quality gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

```bash
git add algua/risk/limits.py algua/backtest/engine.py algua/live/paper_loop.py tests/test_risk_limits.py tests/test_decision_parity.py
git commit -m "feat(179): enforce universe membership in the shared decision rail; delete inline PIT block"
```

---

## Task 3: Keep `_canonical_row` a faithful loop-proxy + fast-path `_fast_weights` coverage

The bounded runtime parity guard (`_assert_parity`) compares the fast path against `_canonical_row`, which today reindex-drops without validating. Now that the loop rejects out-of-universe, the proxy must too, or the bounded guard silently diverges for a strategy whose per-bar `signal` emits an out-of-universe weight while its `signal_panel` does not.

**Files:**
- Modify: `algua/backtest/engine.py` (`_canonical_row`)
- Test: `tests/test_fast_path.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fast_path.py`:

```python
def test_fast_weights_rejects_out_of_universe_construct_output() -> None:
    """_fast_weights validates the CONSTRUCT output against the static operating universe: a
    construct that emits an out-of-universe symbol hard-fails the fast path."""
    def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.5, index=adj.index, columns=adj.columns)

    def bad_construct(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        return pd.Series({"ZZZ": 1.0})  # out of the declared universe

    cfg = StrategyConfig(
        name="oob_construct", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 0.5, "BBB": 0.5}),
        signal_panel_fn=signal_panel, construct_fn=bad_construct,
    )
    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    with pytest.raises(BacktestError, match="out-of-universe"):
        _fast_weights(strat, bars, adj)


def test_canonical_row_rejects_per_bar_signal_out_of_universe() -> None:
    """The bounded parity guard's canonical proxy must reject what the loop rejects: a per-bar
    `signal` emitting an out-of-universe weight (with a clean panel) fails the fast-path run via
    _canonical_row, rather than slipping through as a mere parity mismatch."""
    def good_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        out = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)
        out["AAA"] = 0.5
        out["BBB"] = 0.5
        return out

    cfg = StrategyConfig(
        name="oob_signal", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"ZZZ": 1.0}),
        signal_panel_fn=good_panel, construct_fn=_passthrough,
    )
    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    with pytest.raises(BacktestError, match="out-of-universe"):
        _decision_weights_fast_or_loop(strat, bars, adj, universe_by_date=None)
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_fast_path.py -k "out_of_universe" -q`
Expected:
- `test_fast_weights_rejects_out_of_universe_construct_output` → PASS already (Task 2 wired `_fast_weights`).
- `test_canonical_row_rejects_per_bar_signal_out_of_universe` → FAIL: today the per-bar signal's `ZZZ` is reindex-dropped in `_canonical_row`, so the run raises a `"parity"` mismatch, not `"out-of-universe"`.

- [ ] **Step 3: Make `_canonical_row` run the full rail**

In `algua/backtest/engine.py`, replace the body of `_canonical_row`:

```python
def _canonical_row(
    strategy: LoadedStrategy, bars_sorted: pd.DataFrame, stop: int, columns: pd.Index
) -> pd.Series:
    """The canonical per-bar weights = construct(signal(view), view) over the expanding history
    slice ending at (and including) that bar, reindexed onto `columns` and zero-filled. This is the
    SAME computation the loop performs per bar — INCLUDING the shared risk rails — so the fast-path
    parity guard compares against the loop's own definition, not a re-derivation. Running the full
    `validate_decision_weights` here (not just one check) keeps the proxy a FAITHFUL loop-twin with
    identical check ordering, so e.g. an out-of-universe per-bar weight fails closed instead of being
    silently reindex-dropped before the comparison."""
    view = bars_sorted.iloc[:stop]
    w = strategy.target_weights(view)
    if len(w) == 0:
        return pd.Series(0.0, index=columns)
    try:
        validate_decision_weights(
            w, strategy.execution, strategy.name,
            allowed_symbols=set(strategy.universe) & set(columns),
        )
    except RiskBreach as breach:
        raise BacktestError(breach.detail) from breach
    return w.reindex(columns).fillna(0.0)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_fast_path.py -k "out_of_universe" -q`
Expected: both PASS. Then full fast-path file: `uv run pytest tests/test_fast_path.py -q` → PASS (well-formed strategies pass every rail; existing construct-output breach tests still raise in `_fast_weights` before `_assert_parity`).

- [ ] **Step 5: Run the quality gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

```bash
git add algua/backtest/engine.py tests/test_fast_path.py
git commit -m "feat(179): make _canonical_row a faithful loop-proxy; fast-path out-of-universe coverage"
```

---

## Task 4: Fail closed on an empty static operating universe

The `strategy.universe ∩ adj.columns` intersection introduces a new edge: if a provider returns bars but NONE for a declared symbol, the operating universe is empty, validation no-ops, and a flat strategy would run a silently-meaningless backtest over an all-undeclared panel. Fail closed.

**Files:**
- Modify: `algua/backtest/engine.py` (`simulate`)
- Test: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_backtest_engine.py` (after `test_empty_universe_data_raises`):

```python
def test_static_operating_universe_empty_raises() -> None:
    """A misbehaving provider that returns bars only for UNDECLARED symbols yields an empty
    strategy.universe & adj.columns intersection -> fail closed rather than run a flat backtest."""
    class _WrongSymbolProvider:
        def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
            return SyntheticProvider(seed=0).get_bars(["ZZZ"], start, end, timeframe)

    cfg = StrategyConfig(
        name="wrongdata", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={}, construction="passthrough",
    )
    strat = _strategy(cfg, lambda v, p: pd.Series(dtype="float64"))
    with pytest.raises(BacktestError, match="no fetched price data"):
        run(strat, _WrongSymbolProvider(), START, END)
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_backtest_engine.py::test_static_operating_universe_empty_raises -q`
Expected: FAIL — today the run completes (flat) instead of raising.

- [ ] **Step 3: Add the guard in `simulate`**

In `algua/backtest/engine.py`, in `simulate`, immediately AFTER `adj = adj.sort_index()` (line ~437) and BEFORE the fundamentals block:

```python
    adj = adj.sort_index()

    if universe_by_date is None:
        operating_universe = set(strategy.universe) & set(adj.columns)
        if strategy.universe and not operating_universe:
            raise BacktestError(
                f"no fetched price data for any symbol in strategy {strategy.name!r} declared "
                f"universe {sorted(strategy.universe)} (fetched columns: "
                f"{sorted(map(str, adj.columns))})"
            )
```

(Static mode only — PIT fetches the as-of union and masks per bar, so it must not be intersected with `strategy.universe`.)

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_backtest_engine.py::test_static_operating_universe_empty_raises -q`
Expected: PASS. Confirm the existing empty-universe test still passes: `uv run pytest tests/test_backtest_engine.py::test_empty_universe_data_raises -q` → PASS (universe=[] is falsy, so the new guard is skipped; the prior raise still fires).

- [ ] **Step 5: Run the quality gate and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

```bash
git add algua/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat(179): fail closed when the static operating universe is empty"
```

---

## Task 5: Final verification

- [ ] **Step 1: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 2: Confirm acceptance criteria**

- A static-mode strategy returning an out-of-universe weight hard-fails the backtest (loop AND fast path) the same way PIT mode does — `test_out_of_universe_weight_breaches_both_paths`, `test_fast_weights_rejects_out_of_universe_construct_output`, `test_canonical_row_rejects_per_bar_signal_out_of_universe`.
- Live rejects it identically — paper half of `test_out_of_universe_weight_breaches_both_paths`.
- Existing in-universe strategies/tests unaffected; PIT non-member test green via the unified rail.
- No raw `validate_decision_weights(` call remains without `allowed_symbols`: `grep -rn "validate_decision_weights(" algua tests` shows every call passing it.

---

## Self-Review notes

- **Spec coverage:** new check (T1), fold into rail + delete PIT block + all callers + live (T2), `_canonical_row` faithful proxy + fast-path coverage (T3), empty-universe fail-closed (T4), gate (T5). Exhaustive promotion gate (`verify_signal_panel_parity`) needs no change — it inherits via `_decision_weights` + `_fast_weights`.
- **Type consistency:** `check_universe_membership(weights, allowed_symbols, strategy_name)` and `validate_decision_weights(weights, contract, strategy_name, allowed_symbols)` signatures are used identically in every call site above; `out_of_universe` is the single breach kind throughout.
- **Declined (per GATE-1, do NOT implement):** live `decide()` does not intersect with latest-bar symbols (keeps `strategy.universe`); no fetch-time subset guard.
