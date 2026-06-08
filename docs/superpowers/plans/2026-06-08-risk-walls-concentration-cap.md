# Risk Walls: Concentration Cap + Explicit Long/Short — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a declared+hashed single-name concentration cap (`max_weight_per_symbol`) and an explicit `allow_short` field to `ExecutionContract`, enforced via one shared decision-weight validation bundle with backtest↔live parity.

**Architecture:** Two new frozen-dataclass fields fold automatically into `config_hash`. A single `validate_decision_weights(weights, contract, name)` bundle in `algua/risk/limits.py` runs finite-weight → short-policy → per-symbol-cap → gross checks; all three decision sites (`paper_loop.decide`, engine `_decision_weights`, engine `_decision_weights_fast`) call it, replacing today's scattered pair of checks. `RiskBreach` flows through existing kill-switch (live) / `BacktestError` (research) handling.

**Tech Stack:** Python, pandas, numpy, pytest, frozen dataclasses, import-linter.

**Spec:** `docs/superpowers/specs/2026-06-08-risk-walls-concentration-cap-design.md`

**Quality gate (run between tasks):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

### Task 1: ExecutionContract gains `max_weight_per_symbol` + `allow_short`

**Files:**
- Modify: `algua/contracts/types.py:26-36`
- Test: `tests/test_contracts.py` (create if absent; else append)

- [ ] **Step 1: Write the failing test**

Check first whether `tests/test_contracts.py` exists (`ls tests/test_contracts.py`). If it does, append; otherwise create it with the imports shown.

```python
import pytest

from algua.contracts.types import ExecutionContract


def test_execution_contract_new_fields_default_to_todays_behavior():
    c = ExecutionContract(rebalance_frequency="1d")
    assert c.max_weight_per_symbol == 1.0   # no cap by default
    assert c.allow_short is False           # long-only by default


def test_execution_contract_rejects_nonpositive_per_symbol_cap():
    with pytest.raises(ValueError, match="max_weight_per_symbol must be > 0"):
        ExecutionContract(rebalance_frequency="1d", max_weight_per_symbol=0.0)
    with pytest.raises(ValueError, match="max_weight_per_symbol must be > 0"):
        ExecutionContract(rebalance_frequency="1d", max_weight_per_symbol=-0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py -q`
Expected: FAIL — `TypeError: unexpected keyword argument 'max_weight_per_symbol'` (and the AttributeError on the default test).

- [ ] **Step 3: Add the fields + validation**

In `algua/contracts/types.py`, add the two fields after `max_gross_exposure` and the validation in `__post_init__`:

```python
    rebalance_frequency: str
    decision_lag_bars: int = 1
    allow_fractional: bool = True
    max_gross_exposure: float = 1.0
    max_weight_per_symbol: float = 1.0  # cap on |weight| per symbol; 1.0 = no cap
    allow_short: bool = False           # False = long-only (today's behavior)
    warmup_bars: int = 0

    def __post_init__(self) -> None:
        if self.decision_lag_bars < 1:
            raise ValueError("decision_lag_bars must be >= 1 (no same-bar fills)")
        if self.warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
        if self.max_weight_per_symbol <= 0:
            raise ValueError("max_weight_per_symbol must be > 0")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/contracts/types.py tests/test_contracts.py
git commit -m "feat(risk): declare max_weight_per_symbol + allow_short on ExecutionContract"
```

---

### Task 2: `check_max_weight_per_symbol` in risk/limits

**Files:**
- Modify: `algua/risk/limits.py` (add function after `check_gross_exposure`)
- Test: `tests/test_risk_limits.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_risk_limits.py`:

```python
def test_max_weight_per_symbol_passes_at_or_under_cap():
    from algua.risk.limits import check_max_weight_per_symbol
    check_max_weight_per_symbol(pd.Series({"AAA": 0.5, "BBB": 0.5}), 0.5)   # == cap, ok
    check_max_weight_per_symbol(pd.Series({"AAA": -0.5}), 0.5)              # short |w|==cap, ok
    check_max_weight_per_symbol(pd.Series(dtype="float64"), 0.5)           # empty, ok


def test_max_weight_per_symbol_breaches_over_cap_long_and_short():
    from algua.risk.limits import RiskBreach, check_max_weight_per_symbol
    with pytest.raises(RiskBreach) as ei_long:
        check_max_weight_per_symbol(pd.Series({"AAA": 0.6, "BBB": 0.4}), 0.5)
    assert ei_long.value.kind == "max_weight_per_symbol"
    with pytest.raises(RiskBreach) as ei_short:
        check_max_weight_per_symbol(pd.Series({"AAA": -0.6}), 0.5)
    assert ei_short.value.kind == "max_weight_per_symbol"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_risk_limits.py -q`
Expected: FAIL — `ImportError: cannot import name 'check_max_weight_per_symbol'`.

- [ ] **Step 3: Implement the check**

Add to `algua/risk/limits.py` after `check_gross_exposure`:

```python
def check_max_weight_per_symbol(weights: pd.Series, max_per_symbol: float) -> None:
    """Single-name concentration cap: reject any |weight| above the per-symbol limit. Caps the
    LARGEST position, where gross caps the sum — an agent can pass gross with 100% in one name, so
    this is the rail that stops it. Absolute value, so it holds for shorts too (#135)."""
    if len(weights) == 0:
        return
    over = weights[weights.abs() > max_per_symbol + WEIGHT_TOL]
    if len(over):
        worst = sorted(over.index)
        raise RiskBreach(
            "max_weight_per_symbol",
            f"single-name weight(s) for {worst} exceed max_weight_per_symbol "
            f"{max_per_symbol:.4f}",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_risk_limits.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/risk/limits.py tests/test_risk_limits.py
git commit -m "feat(risk): add check_max_weight_per_symbol (single-name concentration cap)"
```

---

### Task 3: `check_finite_weights` (fail-closed on non-finite)

**Files:**
- Modify: `algua/risk/limits.py` (add `import numpy as np` at top; add function)
- Test: `tests/test_risk_limits.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_risk_limits.py`:

```python
def test_finite_weights_passes_on_clean_series():
    from algua.risk.limits import check_finite_weights
    check_finite_weights(pd.Series({"AAA": 0.5, "BBB": -0.5}), "s")
    check_finite_weights(pd.Series(dtype="float64"), "s")


def test_finite_weights_breaches_on_nan_inf_dupes():
    import numpy as np

    from algua.risk.limits import RiskBreach, check_finite_weights
    for bad in (
        pd.Series({"AAA": np.nan}),
        pd.Series({"AAA": np.inf}),
        pd.Series({"AAA": -np.inf}),
    ):
        with pytest.raises(RiskBreach) as ei:
            check_finite_weights(bad, "s")
        assert ei.value.kind == "non_finite_weight"
    dupe = pd.Series([0.5, 0.5], index=["AAA", "AAA"])
    with pytest.raises(RiskBreach) as ei_dupe:
        check_finite_weights(dupe, "s")
    assert ei_dupe.value.kind == "non_finite_weight"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_risk_limits.py -q`
Expected: FAIL — `ImportError: cannot import name 'check_finite_weights'`.

- [ ] **Step 3: Implement the check**

At the top of `algua/risk/limits.py`, add `import numpy as np` below `import pandas as pd`. Then add:

```python
def check_finite_weights(weights: pd.Series, strategy_name: str) -> None:
    """Fail-closed guard against non-finite target weights. A strategy returning NaN/inf for a named
    symbol, a non-numeric weight, or a duplicated symbol index must HARD-BREACH, not be silently
    flattened by a downstream fillna(0.0) (NaN-skipping .sum() / `NaN < 0` would let it through). The
    panel fast-path's omitted-cell NaN is filled to flat BEFORE this runs, so its sparse-NaN-as-flat
    convention is preserved; only real non-finite VALUES reach here (#135)."""
    if len(weights) == 0:
        return
    if weights.index.has_duplicates:
        dups = sorted(set(weights.index[weights.index.duplicated(keep=False)]))
        raise RiskBreach(
            "non_finite_weight",
            f"strategy '{strategy_name}' returned duplicate symbol weight(s) for {dups}",
        )
    if not pd.api.types.is_numeric_dtype(weights):
        raise RiskBreach(
            "non_finite_weight",
            f"strategy '{strategy_name}' returned non-numeric target weights",
        )
    finite = np.isfinite(weights.to_numpy())
    if not bool(finite.all()):
        bad = sorted(weights.index[~finite])
        raise RiskBreach(
            "non_finite_weight",
            f"strategy '{strategy_name}' returned non-finite target weight(s) for {bad}",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_risk_limits.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/risk/limits.py tests/test_risk_limits.py
git commit -m "feat(risk): add fail-closed check_finite_weights guard"
```

---

### Task 4: `check_short_policy` (the declared long/short gate)

**Files:**
- Modify: `algua/risk/limits.py` (add `check_short_policy`; leave `check_long_only` for now)
- Test: `tests/test_risk_limits.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_risk_limits.py`:

```python
def test_short_policy_long_only_rejects_negatives():
    from algua.risk.limits import RiskBreach, check_short_policy
    check_short_policy(pd.Series({"AAA": 0.6, "BBB": 0.4}), allow_short=False, strategy_name="s")
    check_short_policy(pd.Series(dtype="float64"), allow_short=False, strategy_name="s")
    with pytest.raises(RiskBreach) as ei:
        check_short_policy(pd.Series({"AAA": -0.5}), allow_short=False, strategy_name="s")
    assert ei.value.kind == "long_only"


def test_short_policy_allows_negatives_when_allow_short():
    from algua.risk.limits import check_short_policy
    check_short_policy(pd.Series({"AAA": -0.5, "BBB": 0.5}), allow_short=True, strategy_name="s")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_risk_limits.py -q`
Expected: FAIL — `ImportError: cannot import name 'check_short_policy'`.

- [ ] **Step 3: Implement the check**

Add to `algua/risk/limits.py` (place it where `check_long_only` is; `check_long_only` is removed in Task 7):

```python
def check_short_policy(weights: pd.Series, allow_short: bool, strategy_name: str) -> None:
    """Declared long/short gate. When allow_short is False (the default, long-only), any negative
    target weight hard-breaches; when True, shorts are permitted (the per-symbol cap still bounds
    |weight|). Replaces the old undeclared check_long_only: the constraint is now a hashed contract
    field, not an invisible convention (#135)."""
    if not allow_short and len(weights) and bool((weights < 0).any()):
        negative = sorted(weights[weights < 0].index)
        raise RiskBreach(
            "long_only",
            f"long-only: strategy '{strategy_name}' returned negative target weight(s) "
            f"for {negative}",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_risk_limits.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/risk/limits.py tests/test_risk_limits.py
git commit -m "feat(risk): add check_short_policy (declared long/short gate)"
```

---

### Task 5: `validate_decision_weights` bundle

**Files:**
- Modify: `algua/risk/limits.py` (add `ExecutionContract` import under TYPE_CHECKING; add bundle)
- Test: `tests/test_risk_limits.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_risk_limits.py`:

```python
def _contract(**kw):
    from algua.contracts.types import ExecutionContract
    return ExecutionContract(rebalance_frequency="1d", **kw)


def test_validate_decision_weights_runs_all_rails_in_order():
    from algua.risk.limits import RiskBreach, validate_decision_weights

    # clean long-only vector passes
    validate_decision_weights(pd.Series({"AAA": 0.6, "BBB": 0.4}), _contract(), "s")

    # finite runs first: a NaN breaches as non_finite even though it also "looks" long-only-clean
    import numpy as np
    with pytest.raises(RiskBreach) as ei_fin:
        validate_decision_weights(pd.Series({"AAA": np.nan}), _contract(), "s")
    assert ei_fin.value.kind == "non_finite_weight"

    # short policy before cap/gross: a short under default long-only breaches long_only
    with pytest.raises(RiskBreach) as ei_short:
        validate_decision_weights(pd.Series({"AAA": -0.3}), _contract(), "s")
    assert ei_short.value.kind == "long_only"

    # per-symbol cap binds (allow_short so it isn't caught by long_only first)
    with pytest.raises(RiskBreach) as ei_cap:
        validate_decision_weights(
            pd.Series({"AAA": 0.9}), _contract(max_weight_per_symbol=0.5), "s"
        )
    assert ei_cap.value.kind == "max_weight_per_symbol"

    # gross still enforced last
    with pytest.raises(RiskBreach) as ei_gross:
        validate_decision_weights(
            pd.Series({"AAA": 0.7, "BBB": 0.7}), _contract(max_gross_exposure=1.0), "s"
        )
    assert ei_gross.value.kind == "gross_exposure"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_risk_limits.py -q`
Expected: FAIL — `ImportError: cannot import name 'validate_decision_weights'`.

- [ ] **Step 3: Implement the bundle**

At the top of `algua/risk/limits.py`, under the existing imports, add a TYPE_CHECKING import (keeps risk runtime-light):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algua.contracts.types import ExecutionContract
```

Then add the bundle (the single call surface for every decision path):

```python
def validate_decision_weights(
    weights: pd.Series, contract: ExecutionContract, strategy_name: str
) -> None:
    """The ONE decision-weight validation every path calls (paper/live decide + backtest loop +
    fast-path), so the rails can never drift between research and live. Order: finite (fail-closed)
    -> short policy -> per-symbol cap -> gross exposure (#135)."""
    check_finite_weights(weights, strategy_name)
    check_short_policy(weights, contract.allow_short, strategy_name)
    check_max_weight_per_symbol(weights, contract.max_weight_per_symbol)
    check_gross_exposure(weights, contract.max_gross_exposure)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_risk_limits.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/risk/limits.py tests/test_risk_limits.py
git commit -m "feat(risk): add validate_decision_weights bundle (single rail source of truth)"
```

---

### Task 6: Wire the bundle into the shared `decide()` (live + paper)

**Files:**
- Modify: `algua/live/paper_loop.py:12-17` (imports) and `:72-73` (decide body)
- Test: covered by Task 9 parity tests; smoke via existing `tests/test_paper_loop*.py`

- [ ] **Step 1: Replace the two checks with the bundle**

In `algua/live/paper_loop.py`, change the risk import block:

```python
from algua.risk.limits import (
    WEIGHT_TOL,
    check_drawdown,
    validate_decision_weights,
)
```

And in `decide()`, replace the two check lines:

```python
    weights = strategy.target_weights(view)
    validate_decision_weights(weights, strategy.execution, strategy.name)
    intents = build_intents(weights, current_weights, decision_ts)
    return weights, intents
```

(Update the decide docstring's "long-only + gross" phrasing to "the shared decision-weight rails".)

- [ ] **Step 2: Run the gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: PASS (existing paper-loop tests still green; `check_long_only` no longer imported here).

- [ ] **Step 3: Commit**

```bash
git add algua/live/paper_loop.py
git commit -m "refactor(live): decide() calls the shared validate_decision_weights bundle"
```

---

### Task 7: Wire the bundle into the backtest engine (loop + fast-path), remove `check_long_only`

**Files:**
- Modify: `algua/backtest/engine.py:14` (import), `:109-113` (loop), `:179-183` (fast-path)
- Modify: `algua/risk/limits.py` (delete now-unused `check_long_only`)
- Test: `tests/test_decision_parity.py:103-115` (retarget the long-only test message), `tests/test_risk_limits.py:36-42` (retarget to `check_short_policy`)

- [ ] **Step 1: Swap both engine call sites to the bundle**

In `algua/backtest/engine.py`, change the import:

```python
from algua.risk.limits import WEIGHT_TOL, RiskBreach, validate_decision_weights
```

In `_decision_weights` (the per-bar loop), replace the try-block body:

```python
        try:
            validate_decision_weights(w, strategy.execution, strategy.name)
        except RiskBreach as breach:
            raise BacktestError(f"{breach.detail} at {t}") from breach
        row = w.reindex(columns).fillna(0.0)
```

In `_decision_weights_fast`, replace the try-block body (runs on `nz`, AFTER the panel's `fillna` — so NaN-as-flat is preserved and any surviving `inf` value is caught by the finite check):

```python
        try:
            validate_decision_weights(nz, strategy.execution, strategy.name)
        except RiskBreach as breach:
            raise BacktestError(f"{breach.detail} at {t}") from breach
```

- [ ] **Step 2: Delete `check_long_only` from `algua/risk/limits.py`**

Remove the entire `check_long_only` function (now unreferenced — `check_short_policy` replaces it; no compat alias, per the no-cruft rule).

- [ ] **Step 3: Retarget the two tests that named `check_long_only`**

In `tests/test_risk_limits.py`, replace the old `test_check_long_only_passes_and_raises` body to import/use `check_short_policy` (it is superseded by Task 4's two short-policy tests — delete `test_check_long_only_passes_and_raises` entirely rather than keep a duplicate).

In `tests/test_decision_parity.py:103-115`, the test asserts `pytest.raises(BacktestError, match="long-only")`. The message is unchanged (`check_short_policy` keeps the `long-only:` detail), so only update the docstring reference from `check_long_only` to `check_short_policy`.

- [ ] **Step 4: Run the gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: PASS. `grep -rn check_long_only` returns nothing.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/engine.py algua/risk/limits.py tests/test_risk_limits.py tests/test_decision_parity.py
git commit -m "refactor(backtest): engine loop + fast-path call the shared rail bundle; drop check_long_only"
```

---

### Task 8: `config_hash` identity tests for the new fields

**Files:**
- Test: `tests/test_config_hash_fields.py` (create)

- [ ] **Step 1: Write the test**

```python
"""Changing a declared ExecutionContract rail changes config_hash, so existing live
authorizations correctly invalidate and must re-sign (#135)."""
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash


def _strategy(**execution_kw) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="s",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", **execution_kw),
        params={},
    )
    return LoadedStrategy(config=cfg, fn=lambda v, p: None)


def test_max_weight_per_symbol_changes_config_hash():
    base = config_hash(_strategy())
    tighter = config_hash(_strategy(max_weight_per_symbol=0.2))
    assert base != tighter


def test_allow_short_changes_config_hash():
    base = config_hash(_strategy())
    shortable = config_hash(_strategy(allow_short=True))
    assert base != shortable
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_config_hash_fields.py -q`
Expected: PASS (the fields fold into `asdict(execution)` automatically). If a no-arg `LoadedStrategy`/`StrategyConfig` field name differs, mirror the construction in `tests/test_decision_parity.py:48-55`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_config_hash_fields.py
git commit -m "test(identity): max_weight_per_symbol + allow_short change config_hash"
```

---

### Task 9: Parity tests — cap + short + finite hold identically in backtest and paper

**Files:**
- Test: `tests/test_decision_parity.py` (append)

- [ ] **Step 1: Write the tests**

Append to `tests/test_decision_parity.py`:

```python
def test_concentration_cap_fails_backtest_and_paper_identically() -> None:
    """A weight vector busting the per-symbol cap must FAIL the backtest the same way it fails
    paper/live — the cap is a shared decision-time rail, not a live-only afterthought (#135)."""
    cfg = StrategyConfig(
        name="concentrated", universe=["AAA", "BBB"],
        execution=ExecutionContract(
            rebalance_frequency="1d", decision_lag_bars=1, max_weight_per_symbol=0.5
        ),
        params={},
    )
    strat = LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series({"AAA": 0.9, "BBB": 0.1}))
    with pytest.raises(BacktestError) as ei:
        run(strat, SyntheticProvider(seed=0), START, END)
    assert isinstance(ei.value.__cause__, RiskBreach)
    assert ei.value.__cause__.kind == "max_weight_per_symbol"
    with pytest.raises(RiskBreach) as ei_paper:
        run_paper(strat, SimBroker(cash=1_000_000.0), SyntheticProvider(seed=0), START, END)
    assert ei_paper.value.kind == "max_weight_per_symbol"


def test_allow_short_lets_a_short_through_in_both_paths() -> None:
    """With allow_short=True a negative weight is permitted in BOTH backtest and paper; the default
    (False) rejects it in both (the latter pinned by the existing long-only parity test)."""
    cfg = StrategyConfig(
        name="ls", universe=["AAA", "BBB"],
        execution=ExecutionContract(
            rebalance_frequency="1d", decision_lag_bars=1, allow_short=True
        ),
        params={},
    )
    strat = LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series({"AAA": 0.5, "BBB": -0.5}))
    # Neither path raises (long/short now declared+permitted); a clean run is the assertion.
    run(strat, SyntheticProvider(seed=0), START, END)
    run_paper(strat, SimBroker(cash=1_000_000.0), SyntheticProvider(seed=0), START, END)


def test_named_symbol_nan_breaches_both_paths() -> None:
    """A strategy that names a symbol but returns NaN for it must hard-breach (not silently flatten)
    in backtest and paper — fail-closed finite guard (#135)."""
    import numpy as np
    cfg = StrategyConfig(
        name="nanny", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
    )
    strat = LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series({"AAA": np.nan}))
    with pytest.raises(BacktestError) as ei:
        run(strat, SyntheticProvider(seed=0), START, END)
    assert isinstance(ei.value.__cause__, RiskBreach)
    assert ei.value.__cause__.kind == "non_finite_weight"
    with pytest.raises(RiskBreach) as ei_paper:
        run_paper(strat, SimBroker(cash=1_000_000.0), SyntheticProvider(seed=0), START, END)
    assert ei_paper.value.kind == "non_finite_weight"
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_decision_parity.py -q`
Expected: PASS. If `run_paper`'s signature needs `on_decision`, it is optional (see `:86-89`); call without it. If a strategy that breaches on the very first decided bar needs enough bars, `START..END` (Jan–Apr daily) is ample.

- [ ] **Step 3: Commit**

```bash
git add tests/test_decision_parity.py
git commit -m "test(parity): concentration cap + allow_short + non-finite hold identically backtest<->paper"
```

---

### Task 10: Verify live `.kind` dispatch is generic + final full gate

**Files:**
- Read-only: `algua/cli/live_cmd.py` (or wherever `RiskBreach.kind` trips the kill-switch)

- [ ] **Step 1: Confirm the live kill-switch path is kind-agnostic**

Run: `grep -rn "\.kind\|RiskBreach\|kill" algua/cli/*.py algua/live/*.py`
Expected: the live tick handler trips the kill-switch/flatten on ANY `RiskBreach` (or dispatches on `.kind` without a hardcoded allowlist that would exclude `max_weight_per_symbol` / `non_finite_weight`). If an allowlist exists, add the two new kinds and note it in the PR; otherwise no change.

- [ ] **Step 2: Final full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 3: Commit any kind-allowlist fix (only if Step 1 found one)**

```bash
git add -A && git commit -m "fix(live): route new RiskBreach kinds through the kill-switch"
```

---

## Self-Review

- **Spec coverage:** fields + validation (T1); per-symbol cap (T2); finite guard (T3); short policy (T4); bundle (T5); three-site parity wiring (T6–T7); no compat shim / `check_long_only` removed (T7); hash identity (T8); parity + finite tests (T9); live `.kind` dispatch (T10). Deferred items (turnover, realized cap, full parity) are out of scope by design — no tasks, tracked in memory.
- **Placeholder scan:** none — every code step shows complete code.
- **Type consistency:** `validate_decision_weights(weights, contract, strategy_name)`, `check_short_policy(weights, allow_short, strategy_name)`, `check_max_weight_per_symbol(weights, max_per_symbol)`, `check_finite_weights(weights, strategy_name)` used consistently across tasks and call sites. `RiskBreach.kind` values: `max_weight_per_symbol`, `non_finite_weight`, `long_only`, `gross_exposure`.
