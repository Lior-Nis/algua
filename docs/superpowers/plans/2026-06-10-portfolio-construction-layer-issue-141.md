# Portfolio-Construction Layer (signal → target portfolio) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the strategy contract `compute_weights(view, params) → weights` into an authored `signal(view, params) → scores` plus a config-named, reusable portfolio-construction policy (`scores → weights`) resolved from a new pure `algua/portfolio/construction.py` library, with the #135 hard risk walls unchanged after construction.

**Architecture:** A strategy authors `signal` (+ optional `signal_panel` fast twin) and names a `construction` policy in `CONFIG`. `LoadedStrategy.target_weights(view) = construct(signal(view), view)` is the single pipeline both the backtest engine and the paper/live `decide()` call, so backtest↔live parity holds for free. Policy id + params fold into `config_hash`; the policy module folds into `code_hash`, so a policy change invalidates live approvals.

**Tech Stack:** Python 3.12, uv, pandas/numpy, pydantic (`StrategyConfig`), vectorbt (engine), Typer (CLI), pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-06-10-portfolio-construction-layer-issue-141-design.md`

**Quality gate (run after every task):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

**Create**
- `algua/portfolio/__init__.py` — empty package marker.
- `algua/portfolio/construction.py` — `ConstructFn` type, the three starter policies, shared `_finite_scores`/`_ranked` helpers, per-policy validators, `_POLICIES` dispatch + `CONSTRUCTION_POLICIES` (MappingProxyType), `get_construction_policy`, `validate_construction_params`, `ConstructionError`.
- `tests/test_portfolio_construction.py` — policy behavior, NaN/dtype handling, tie-break, validation.

**Modify**
- `algua/strategies/base.py` — rename signal type aliases; `StrategyConfig` (+`construction`, +`construction_params`); `LoadedStrategy` (store raw `construct_fn` + signal fns; `signal`/`signal_panel`/`construct`/`target_weights` methods; `authored_signal` property); `config_hash` (fold construction + `allow_nan=False`).
- `algua/strategies/loader.py` — require `signal`; resolve + validate the construction policy; arity checks; detect `signal_panel`.
- `algua/registry/approvals.py` — root the code-hash closure from the signal module **and** `algua.portfolio.construction`.
- `algua/backtest/engine.py` — per-bar loop + `_canonical_row` use `signal`/`target_weights`; `_decision_weights_fast` calls `signal_panel` then per-bar `construct`; weight-level parity guard kept.
- `algua/backtest/sweep.py` — `_override` rebuilds from config, routes `construction.<key>` → `construction_params`, re-validates, rejects unknown signal keys.
- `algua/cli/strategy_cmd.py` — `_TEMPLATE` authors `signal` + `construction`.
- `algua/contracts/types.py` — `Strategy` protocol docstring notes the composition.
- `algua/data/hindsight.py` — comment `compute_weights` → `signal`.
- `pyproject.toml` — import-linter contract: `algua.portfolio` is pure.
- `algua/strategies/examples/cross_sectional_momentum.py`, `algua/strategies/examples/fundamentals_earnings_tilt.py` — migrate to `signal` + `construction`.
- The contract test surface (Task 12).

---

## Task 1: Construction policy library

**Files:**
- Create: `algua/portfolio/__init__.py`
- Create: `algua/portfolio/construction.py`
- Test: `tests/test_portfolio_construction.py`

- [ ] **Step 1: Create the package marker**

Create `algua/portfolio/__init__.py` (empty file).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_portfolio_construction.py`:

```python
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from algua.portfolio.construction import (
    CONSTRUCTION_POLICIES,
    ConstructionError,
    equal_weight_positive,
    get_construction_policy,
    score_proportional_long,
    top_k_equal_weight,
    validate_construction_params,
)

_EMPTY = pd.DataFrame()  # starter policies ignore `view`


def test_top_k_equal_weight_selects_top_k_equal():
    scores = pd.Series({"A": 0.3, "B": 0.1, "C": 0.2, "D": -0.5})
    w = top_k_equal_weight(scores, _EMPTY, {"top_k": 2})
    assert set(w.index) == {"A", "C"}
    assert w.to_dict() == pytest.approx({"A": 0.5, "C": 0.5})


def test_top_k_tie_break_is_deterministic_by_symbol():
    # B and C tie at 0.2; with top_k=2 and A=0.3 highest, the tie must resolve to the
    # lexicographically-smaller symbol (B), regardless of input order.
    ordered = pd.Series({"A": 0.3, "B": 0.2, "C": 0.2})
    shuffled = pd.Series({"C": 0.2, "A": 0.3, "B": 0.2})
    wo = top_k_equal_weight(ordered, _EMPTY, {"top_k": 2})
    ws = top_k_equal_weight(shuffled, _EMPTY, {"top_k": 2})
    assert set(wo.index) == {"A", "B"}
    assert set(ws.index) == {"A", "B"}


def test_policies_drop_nonfinite_scores_not_zero_fill():
    scores = pd.Series({"A": 0.3, "B": np.nan, "C": 0.2})
    # B is dropped (no opinion), NOT treated as a 0.0 score that could be selected.
    w = top_k_equal_weight(scores, _EMPTY, {"top_k": 3})
    assert set(w.index) == {"A", "C"}


def test_policies_fail_closed_on_non_numeric_scores():
    scores = pd.Series({"A": "high", "B": "low"})
    with pytest.raises(ConstructionError):
        top_k_equal_weight(scores, _EMPTY, {"top_k": 1})


def test_equal_weight_positive():
    scores = pd.Series({"A": 1.0, "B": -1.0, "C": 0.0, "D": 2.0})
    w = equal_weight_positive(scores, _EMPTY, {})
    assert set(w.index) == {"A", "D"}
    assert w.to_dict() == pytest.approx({"A": 0.5, "D": 0.5})


def test_equal_weight_positive_all_nonpositive_is_flat():
    scores = pd.Series({"A": -1.0, "B": 0.0})
    assert equal_weight_positive(scores, _EMPTY, {}).empty


def test_score_proportional_long_normalizes_positives_to_gross_one():
    scores = pd.Series({"A": 3.0, "B": 1.0, "C": -5.0})
    w = score_proportional_long(scores, _EMPTY, {})
    assert w.to_dict() == pytest.approx({"A": 0.75, "B": 0.25})
    assert float(w.sum()) == pytest.approx(1.0)


def test_get_construction_policy_unknown_raises():
    with pytest.raises(ConstructionError):
        get_construction_policy("does_not_exist")


def test_validate_top_k_requires_positive_int():
    validate_construction_params("top_k_equal_weight", {"top_k": 3})
    for bad in ({}, {"top_k": 0}, {"top_k": -1}, {"top_k": 2.5}, {"top_k": True}, {"top_k": "3"}):
        with pytest.raises(ConstructionError):
            validate_construction_params("top_k_equal_weight", bad)


def test_validate_rejects_unknown_keys_and_nonfinite_values():
    with pytest.raises(ConstructionError):
        validate_construction_params("equal_weight_positive", {"surprise": 1})
    with pytest.raises(ConstructionError):
        validate_construction_params("top_k_equal_weight", {"top_k": 2, "x": float("nan")})


def test_dispatch_view_is_read_only():
    with pytest.raises(TypeError):
        CONSTRUCTION_POLICIES["new"] = None  # type: ignore[index]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_portfolio_construction.py -q`
Expected: FAIL (`ModuleNotFoundError: algua.portfolio.construction`).

- [ ] **Step 4: Implement `algua/portfolio/construction.py`**

```python
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

# A construction policy maps cross-sectional scores -> target weights under a risk convention.
# `view` is the same PIT bar-schema frame the signal saw (passed so a future vol-targeting policy
# can estimate vol from prices with no contract change); the starter policies ignore it.
ConstructFn = Callable[[pd.Series, pd.DataFrame, dict[str, Any]], pd.Series]


class ConstructionError(ValueError):
    """An invalid construction policy id, params, or score series. Subclasses ValueError so the
    CLI's json error contract still renders it."""


def _finite_scores(scores: pd.Series) -> pd.Series:
    """Fail closed on a non-numeric score series, then DROP missing/non-finite scores.

    A missing or NaN/inf score means 'no opinion - not selectable' and is removed; it is NEVER
    coerced to 0.0 (a real 0.0 score must stay distinct from 'no score'). Mirrors the fail-closed
    philosophy of risk.limits.check_finite_weights at the construction seam.
    """
    if pd.api.types.is_bool_dtype(scores) or not pd.api.types.is_numeric_dtype(scores):
        raise ConstructionError("signal returned a non-numeric score series")
    if scores.index.isnull().any():
        raise ConstructionError("signal returned a null symbol label")
    if scores.index.has_duplicates:
        raise ConstructionError("signal returned duplicate symbol score(s)")
    return scores[np.isfinite(scores.to_numpy())]


def _ranked(scores: pd.Series) -> pd.Series:
    """Finite scores ordered by (score descending, symbol ascending). Sorting by symbol first
    (stable) then by score makes ties resolve to the lexicographically-smaller symbol regardless of
    input order, so a per-bar `signal` Series and a `signal_panel` matrix row select IDENTICALLY."""
    finite = _finite_scores(scores)
    by_symbol = finite.sort_index(kind="stable")
    return by_symbol.sort_values(ascending=False, kind="stable")


def top_k_equal_weight(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Hold the top-`top_k` names by score, equal weight 1/k."""
    top_k = int(params["top_k"])
    winners = _ranked(scores).head(top_k).index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)


def equal_weight_positive(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Equal-weight every name with a strictly positive score."""
    finite = _finite_scores(scores)
    winners = sorted(finite[finite > 0.0].index)
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)


def score_proportional_long(
    scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]
) -> pd.Series:
    """Clip negatives to zero; weight the positives proportionally, normalized to gross 1.0."""
    finite = _finite_scores(scores)
    positive = finite[finite > 0.0]
    total = float(positive.sum())
    if total <= 0.0:
        return pd.Series(dtype="float64")
    return (positive / total).sort_index()


def _require_no_unknown_keys(params: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ConstructionError(f"unknown construction param(s): {sorted(unknown)}")


def _validate_top_k(params: dict[str, Any]) -> None:
    _require_no_unknown_keys(params, {"top_k"})
    if "top_k" not in params:
        raise ConstructionError("top_k_equal_weight requires 'top_k'")
    top_k = params["top_k"]
    # bool is an int subtype; reject it so True can't masquerade as 1.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ConstructionError(f"top_k must be a positive int, got {top_k!r}")


def _validate_no_params(params: dict[str, Any]) -> None:
    _require_no_unknown_keys(params, set())


def _assert_finite_json(value: Any, path: str = "construction_params") -> None:
    """Recursively reject non-finite floats, non-string dict keys, and non-JSON value types so
    config_hash (serialized with allow_nan=False) is canonical and meaningful."""
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ConstructionError(f"{path}: non-string key {k!r}")
            _assert_finite_json(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _assert_finite_json(v, f"{path}[{i}]")
    elif isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ConstructionError(f"{path}: non-finite float {value!r}")
    else:
        raise ConstructionError(f"{path}: non-JSON value of type {type(value).__name__}")


@dataclass(frozen=True)
class _Policy:
    fn: ConstructFn
    validate: Callable[[dict[str, Any]], None]


_POLICIES: dict[str, _Policy] = {
    "top_k_equal_weight": _Policy(top_k_equal_weight, _validate_top_k),
    "equal_weight_positive": _Policy(equal_weight_positive, _validate_no_params),
    "score_proportional_long": _Policy(score_proportional_long, _validate_no_params),
}
# Read-only public dispatch view. Identity rests on this module's STATIC source (approvals.py hashes
# the whole module), not on runtime dispatch state — there is no dynamic registration.
CONSTRUCTION_POLICIES = MappingProxyType(_POLICIES)


def get_construction_policy(policy_id: str) -> ConstructFn:
    try:
        return _POLICIES[policy_id].fn
    except KeyError:
        raise ConstructionError(
            f"unknown construction policy {policy_id!r}; available: {sorted(_POLICIES)}"
        ) from None


def validate_construction_params(policy_id: str, params: dict[str, Any]) -> None:
    """Per-policy load-time validation: unknown id, then finite/JSON values, then the policy's own
    type+domain checks (e.g. top_k positive int). Raises ConstructionError on any violation."""
    try:
        policy = _POLICIES[policy_id]
    except KeyError:
        raise ConstructionError(
            f"unknown construction policy {policy_id!r}; available: {sorted(_POLICIES)}"
        ) from None
    _assert_finite_json(params)
    policy.validate(params)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_portfolio_construction.py -q`
Expected: PASS (11 tests).

- [ ] **Step 6: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: the new module/tests are green (the import-linter contract for `algua.portfolio` is added in Task 11; existing contracts must still pass — `construction.py` imports only stdlib + numpy/pandas, so no boundary is crossed yet).

- [ ] **Step 7: Commit**

```bash
git add algua/portfolio/ tests/test_portfolio_construction.py
git commit -m "feat(portfolio): construction policy library (scores -> weights) + validators (#141)"
```

---

## Task 2: `StrategyConfig` fields + `config_hash` fold

**Files:**
- Modify: `algua/strategies/base.py` (`StrategyConfig`, `config_hash`)
- Test: `tests/test_config_hash_fields.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config_hash_fields.py` (import `StrategyConfig`, `LoadedStrategy`, `config_hash`, `ExecutionContract` as that file already does; if `LoadedStrategy` construction differs after Task 3, this test uses only `config_hash` over a minimally-built `LoadedStrategy` — keep it config-only):

```python
def _cfg(**over):
    from algua.contracts.types import ExecutionContract
    from algua.strategies.base import StrategyConfig
    base = dict(
        name="s", universe=["A", "B"], execution=ExecutionContract(rebalance_frequency="1d"),
        params={"lookback": 10}, construction="top_k_equal_weight",
        construction_params={"top_k": 2},
    )
    base.update(over)
    return StrategyConfig(**base)


def test_config_hash_changes_with_construction_id():
    from algua.strategies.base import config_hash
    from algua.strategies.loader import _loaded_for_test  # see note below
    a = _loaded_for_test(_cfg())
    b = _loaded_for_test(_cfg(construction="score_proportional_long", construction_params={}))
    assert config_hash(a) != config_hash(b)


def test_config_hash_changes_with_construction_params():
    from algua.strategies.base import config_hash
    from algua.strategies.loader import _loaded_for_test
    a = _loaded_for_test(_cfg(construction_params={"top_k": 2}))
    b = _loaded_for_test(_cfg(construction_params={"top_k": 3}))
    assert config_hash(a) != config_hash(b)
```

> Note: `config_hash` takes a `LoadedStrategy`. To avoid coupling this test to loader internals, add a tiny test helper `_loaded_for_test(config)` in `loader.py` (Task 4) that builds a `LoadedStrategy` from a config with a dummy `signal`/resolved policy. If you prefer, build the `LoadedStrategy` inline once Task 3 lands and reorder: implement Task 3 then return here. The two assertions only exercise `config_hash`.

- [ ] **Step 2: Add the fields to `StrategyConfig`**

In `algua/strategies/base.py`, extend `StrategyConfig`:

```python
class StrategyConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    universe: list[str]
    execution: ExecutionContract
    params: dict[str, Any] = {}
    # Portfolio-construction policy (issue #141): the id is resolved by the loader against
    # algua.portfolio.construction; construction_params are validated per-policy at load.
    construction: str
    construction_params: dict[str, Any] = {}
    needs_fundamentals: bool = False
```

- [ ] **Step 3: Fold construction into `config_hash` with `allow_nan=False`**

Replace the `payload = json.dumps({...})` block in `config_hash`:

```python
    payload = json.dumps(
        {
            "name": strategy.name,
            "universe": strategy.universe,
            "params": strategy.params,
            "execution": asdict(strategy.execution),
            "construction": strategy.config.construction,
            "construction_params": strategy.config.construction_params,
            "needs_fundamentals": strategy.config.needs_fundamentals,
        },
        sort_keys=True,
        allow_nan=False,
    )
```

Update the `config_hash` docstring to note construction id + params are part of the identity.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config_hash_fields.py -q`
Expected: PASS once Task 3/4 land the `LoadedStrategy`/helper. (If running before Task 3, expect a collection error on `_loaded_for_test`; that's fine — proceed to Task 3 and re-run.)

- [ ] **Step 5: Commit**

```bash
git add algua/strategies/base.py tests/test_config_hash_fields.py
git commit -m "feat(strategies): StrategyConfig construction fields + config_hash fold (allow_nan=False) (#141)"
```

---

## Task 3: `LoadedStrategy` composition (signal → construct)

**Files:**
- Modify: `algua/strategies/base.py`
- Test: `tests/test_strategies_base.py`, `tests/test_strategies_base_fundamentals.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_strategies_base.py`:

```python
def test_target_weights_composes_signal_then_construct():
    import pandas as pd
    from algua.contracts.types import ExecutionContract
    from algua.portfolio.construction import top_k_equal_weight
    from algua.strategies.base import LoadedStrategy, StrategyConfig

    def signal(view, params):
        return pd.Series({"A": 0.9, "B": 0.1, "C": 0.5})

    cfg = StrategyConfig(
        name="s", universe=["A", "B", "C"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        construction="top_k_equal_weight", construction_params={"top_k": 2},
    )
    loaded = LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=top_k_equal_weight)
    w = loaded.target_weights(pd.DataFrame())
    assert set(w.index) == {"A", "C"}
    assert w.to_dict() == {"A": 0.5, "C": 0.5}


def test_construct_reads_current_config_params_not_a_bound_partial():
    import pandas as pd
    from algua.contracts.types import ExecutionContract
    from algua.portfolio.construction import top_k_equal_weight
    from algua.strategies.base import LoadedStrategy, StrategyConfig

    def signal(view, params):
        return pd.Series({"A": 0.9, "B": 0.5, "C": 0.1})

    cfg = StrategyConfig(
        name="s", universe=["A", "B", "C"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    loaded = LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=top_k_equal_weight)
    # Rebuild with top_k=2 (what a sweep override does); behavior must follow the NEW config.
    loaded2 = LoadedStrategy(
        config=cfg.model_copy(update={"construction_params": {"top_k": 2}}),
        signal_fn=signal, construct_fn=top_k_equal_weight,
    )
    assert len(loaded.target_weights(pd.DataFrame())) == 1
    assert len(loaded2.target_weights(pd.DataFrame())) == 2
```

Add to `tests/test_strategies_base_fundamentals.py`:

```python
def test_target_weights_fundamentals_lane_composes():
    import pandas as pd
    from algua.contracts.types import ExecutionContract
    from algua.portfolio.construction import equal_weight_positive
    from algua.strategies.base import LoadedStrategy, StrategyConfig

    def signal(view, params, fundamentals):
        return pd.Series({"A": 1.0, "B": -1.0})

    cfg = StrategyConfig(
        name="f", universe=["A", "B"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        construction="equal_weight_positive", needs_fundamentals=True,
    )
    loaded = LoadedStrategy(
        config=cfg, fundamentals_signal_fn=signal, construct_fn=equal_weight_positive
    )
    w = loaded.target_weights(pd.DataFrame(), pd.DataFrame())
    assert w.to_dict() == {"A": 1.0}
    with pytest.raises(ValueError):
        loaded.target_weights(pd.DataFrame())  # needs_fundamentals but no frame -> fail closed
```

- [ ] **Step 2: Rewrite the type aliases + `LoadedStrategy` in `base.py`**

Replace the `ComputeWeightsFn` / `ComputeWeightsPanelFn` / `ComputeFundamentalsWeightsFn` aliases and the whole `LoadedStrategy` dataclass with:

```python
from algua.portfolio.construction import ConstructFn

# The AUTHORED signal: a pure module-level `signal(view, params) -> pd.Series` of cross-sectional
# scores (NOT weights). The protocol-level `Strategy.target_weights(features)` is exposed only by
# the LoadedStrategy adapter, which composes signal -> construction.
SignalFn = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]

# OPTIONAL vectorized acceleration: a pure `signal_panel(bars, params)` returning the FULL decision-
# time SCORES matrix (index=timestamp, columns=symbol; PRE-lag) in one shot. NOT a second signal
# definition: the engine uses it only behind a fail-closed WEIGHT-level parity guard.
SignalPanelFn = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]

# OPT-IN fundamentals signal (issue #132): `signal(view, params, fundamentals)`. Distinct type so
# the 2-arg and 3-arg forms never silently overload.
FundamentalsSignalFn = Callable[[pd.DataFrame, dict[str, Any], pd.DataFrame], pd.Series]


@dataclass
class LoadedStrategy:
    """Binds a StrategyConfig + the authored signal fn(s) + the RESOLVED construction policy into an
    object satisfying the Strategy protocol. Exactly one of (`signal_fn`, `fundamentals_signal_fn`)
    is active, selected by `config.needs_fundamentals`. The adapter is the ONLY place the
    protocol-level `target_weights` exists; it composes construct(signal(view), view).

    `construct_fn` is the RAW policy callable (never a params-bound partial): `construct` reads
    `config.construction_params` at call time, so a sweep that rebuilds the config takes effect and
    `inspect.getmodule(construct_fn)` resolves to the policy module for the identity hash.
    """

    config: StrategyConfig
    construct_fn: ConstructFn
    signal_fn: SignalFn | None = None
    signal_panel_fn: SignalPanelFn | None = None
    fundamentals_signal_fn: FundamentalsSignalFn | None = None

    def __post_init__(self) -> None:
        if self.config.needs_fundamentals:
            if self.fundamentals_signal_fn is None:
                raise ValueError(
                    "needs_fundamentals=True requires a 3-arg signal (fundamentals_signal_fn)"
                )
        elif self.signal_fn is None:
            raise ValueError("needs_fundamentals=False requires a 2-arg signal (signal_fn)")

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def universe(self) -> list[str]:
        return self.config.universe

    @property
    def execution(self) -> ExecutionContract:
        return self.config.execution

    @property
    def params(self) -> dict[str, Any]:
        return self.config.params

    @property
    def authored_signal(self) -> SignalFn | FundamentalsSignalFn:
        """The active authored signal fn — used wherever code needs the strategy's source module
        (e.g. code_hash), since `signal_fn` is None for a needs_fundamentals strategy."""
        fn = self.fundamentals_signal_fn if self.config.needs_fundamentals else self.signal_fn
        assert fn is not None  # __post_init__ guarantees the active fn is set
        return fn

    def signal(self, view: pd.DataFrame, fundamentals: pd.DataFrame | None = None) -> pd.Series:
        if self.config.needs_fundamentals:
            if fundamentals is None:
                raise ValueError(
                    f"strategy {self.name!r} needs fundamentals but signal was called without a "
                    f"fundamentals frame (fail closed)"
                )
            assert self.fundamentals_signal_fn is not None
            return self.fundamentals_signal_fn(view, self.config.params, fundamentals)
        assert self.signal_fn is not None
        return self.signal_fn(view, self.config.params)

    def signal_panel(self, bars: pd.DataFrame) -> pd.DataFrame | None:
        if self.signal_panel_fn is None:
            return None
        return self.signal_panel_fn(bars, self.config.params)

    def construct(self, scores: pd.Series, view: pd.DataFrame) -> pd.Series:
        return self.construct_fn(scores, view, self.config.construction_params)

    def target_weights(
        self, features: pd.DataFrame, fundamentals: pd.DataFrame | None = None
    ) -> pd.Series:
        return self.construct(self.signal(features, fundamentals), features)
```

> Keep the existing `assert_tradable_without_fundamentals` and `config_hash` functions in `base.py`. Remove the old `signal_fn` *property* (it's replaced by `authored_signal`); `signal_fn` is now a dataclass field. Ensure `from collections.abc import Callable` and `from typing import Any` imports remain.

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_strategies_base.py tests/test_strategies_base_fundamentals.py -q`
Expected: PASS for the new tests (other tests in these files migrate in Task 12).

- [ ] **Step 4: Commit**

```bash
git add algua/strategies/base.py tests/test_strategies_base.py tests/test_strategies_base_fundamentals.py
git commit -m "feat(strategies): LoadedStrategy composes signal -> construction; raw construct_fn (#141)"
```

---

## Task 4: Loader — require `signal`, resolve + validate the policy

**Files:**
- Modify: `algua/strategies/loader.py`
- Test: `tests/test_strategy_loader.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_strategy_loader.py`:

```python
def test_loader_resolves_and_binds_construction():
    from algua.strategies.loader import load_strategy
    s = load_strategy("cross_sectional_momentum")
    assert s.config.construction == "top_k_equal_weight"
    assert s.signal_fn is not None and s.signal_panel_fn is not None
    assert callable(s.construct_fn)


def test_loader_rejects_unknown_construction(tmp_path, monkeypatch):
    # A module whose CONFIG names a missing policy must fail at load.
    import textwrap
    import algua.strategies.examples as ex
    p = next(iter(ex.__path__)) + "/_tmp_bad_policy.py"
    with open(p, "w") as f:
        f.write(textwrap.dedent('''
            import pandas as pd
            from algua.contracts.types import ExecutionContract
            from algua.strategies.base import StrategyConfig
            CONFIG = StrategyConfig(name="_tmp_bad_policy", universe=["A"],
                execution=ExecutionContract(rebalance_frequency="1d"),
                construction="nope_not_real")
            def signal(view, params):
                return pd.Series(dtype="float64")
        '''))
    try:
        from algua.strategies.loader import StrategyNotFound, load_strategy
        with pytest.raises((StrategyNotFound, ValueError)):
            load_strategy("_tmp_bad_policy")
    finally:
        import os
        os.remove(p)
```

- [ ] **Step 2: Rewrite `load_strategy`**

Replace `load_strategy` in `algua/strategies/loader.py`:

```python
import importlib
import inspect
import pkgutil

from algua.portfolio.construction import (
    ConstructionError,
    get_construction_policy,
    validate_construction_params,
)
from algua.strategies import examples
from algua.strategies.base import LoadedStrategy


class StrategyNotFound(LookupError):
    pass


def load_strategy(name: str) -> LoadedStrategy:
    """Load a bundled strategy module by name; it must expose CONFIG + signal, and CONFIG must name
    a known construction policy with valid params. Optional `signal_panel` is the vectorized twin."""
    try:
        module = importlib.import_module(f"algua.strategies.examples.{name}")
    except ModuleNotFoundError as exc:
        raise StrategyNotFound(name) from exc
    if not hasattr(module, "CONFIG") or not hasattr(module, "signal"):
        raise StrategyNotFound(f"{name} is missing CONFIG or signal")

    config = module.CONFIG
    try:
        construct_fn = get_construction_policy(config.construction)
        validate_construction_params(config.construction, config.construction_params)
    except ConstructionError as exc:
        raise StrategyNotFound(f"{name}: {exc}") from exc

    panel_fn = getattr(module, "signal_panel", None)
    if panel_fn is not None and not callable(panel_fn):
        raise StrategyNotFound(
            f"{name}.signal_panel is not callable (got {type(panel_fn).__name__})"
        )

    needs_fundamentals = bool(getattr(config, "needs_fundamentals", False))
    n_params = len(inspect.signature(module.signal).parameters)
    if needs_fundamentals:
        if panel_fn is not None:
            raise StrategyNotFound(
                f"{name}: signal_panel is not supported with needs_fundamentals "
                f"(no vectorized fundamentals fast path yet)"
            )
        if n_params != 3:
            raise StrategyNotFound(
                f"{name}: needs_fundamentals=True requires signal(view, params, fundamentals); "
                f"got {n_params} params"
            )
        return LoadedStrategy(
            config=config, fundamentals_signal_fn=module.signal, construct_fn=construct_fn
        )

    if n_params != 2:
        raise StrategyNotFound(f"{name}: signal must take (view, params); got {n_params} params")
    return LoadedStrategy(
        config=config, signal_fn=module.signal, signal_panel_fn=panel_fn, construct_fn=construct_fn
    )


def list_strategies() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(examples.__path__) if not m.name.startswith("_")]
```

Optionally add the Task-2 test helper:

```python
def _loaded_for_test(config) -> LoadedStrategy:
    """Test-only: build a LoadedStrategy from a config with a trivial signal + its resolved policy.
    Used by config_hash tests that should not depend on a real example module."""
    import pandas as pd
    fn = get_construction_policy(config.construction)
    return LoadedStrategy(
        config=config, signal_fn=lambda view, params: pd.Series(dtype="float64"), construct_fn=fn
    )
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_strategy_loader.py -q`
Expected: the new tests PASS once the example migrates (Task 9). If running before Task 9, `test_loader_resolves_and_binds_construction` fails on the not-yet-migrated example — run it after Task 9, or temporarily point it at a migrated fixture. The `unknown construction` test passes immediately.

- [ ] **Step 4: Commit**

```bash
git add algua/strategies/loader.py tests/test_strategy_loader.py
git commit -m "feat(strategies): loader requires signal + resolves/validates construction policy (#141)"
```

---

## Task 5: `approvals.py` — root code-hash from the policy module too

**Files:**
- Modify: `algua/registry/approvals.py`
- Test: `tests/test_registry_approvals.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_registry_approvals.py`:

```python
def test_code_hash_covers_construction_module():
    # The construction module's source must be part of code_hash, so a policy edit invalidates a
    # prior approval. We assert the construction module's source contributes to the closure.
    import inspect

    import algua.portfolio.construction as construction
    from algua.registry.approvals import _first_party_closure

    import algua.strategies.examples.cross_sectional_momentum as mom
    closure = _first_party_closure(inspect.getmodule(mom.signal))
    # After Task 5 the approvals closure is rooted from BOTH the signal module and the construction
    # module; verify the construction module's source is present in the merged closure.
    merged = _merged_closure_for("cross_sectional_momentum")
    assert "algua.portfolio.construction" in merged
    assert merged["algua.portfolio.construction"] == inspect.getsource(construction)
```

> Add a small exported helper `_merged_closure_for(name)` in `approvals.py` that returns the merged closure dict (used by `compute_artifact_hashes`), so the test can assert membership without recomputing.

- [ ] **Step 2: Update `compute_artifact_hashes`**

In `algua/registry/approvals.py`, change the closure construction to root from both modules. Replace the body that builds `closure` in `compute_artifact_hashes`:

```python
import importlib

_CONSTRUCTION_MODULE = "algua.portfolio.construction"


def _merged_closure_for(name: str) -> dict[str, str]:
    """First-party source closure for a strategy's identity: the union of the closure reachable from
    its authored signal module AND the construction policy module (resolved by NAME, not via the
    bound callable — getmodule on a partial returns functools). The construction module holds every
    policy + the dispatch table, so a policy-body edit, a helper edit, or an id retarget invalidates
    a prior approval."""
    loaded = load_strategy(name)
    signal_root = inspect.getmodule(loaded.authored_signal)
    construction_root = importlib.import_module(_CONSTRUCTION_MODULE)
    merged: dict[str, str] = {}
    merged.update(_first_party_closure(signal_root))
    merged.update(_first_party_closure(construction_root))
    return merged


def compute_artifact_hashes(name: str) -> ArtifactIdentity:
    """... (keep the existing docstring; add a sentence noting the construction module is a second
    closure root so a portfolio-construction change invalidates a prior approval.)"""
    loaded = load_strategy(name)
    closure = _merged_closure_for(name)
    payload = "\n".join(
        f"# module: {mod_name}\n{source}" for mod_name, source in sorted(closure.items())
    )
    code_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return ArtifactIdentity(
        code_hash=code_hash,
        config_hash=config_hash(loaded),
        dependency_hash=lockfile.dependency_hash(),
    )
```

> Note: `compute_artifact_hashes` now calls `load_strategy` twice (once directly, once via `_merged_closure_for`). Either keep both for clarity or inline `_merged_closure_for` to reuse the single `loaded`. Inlining is cleaner — pass `loaded` into `_merged_closure_for(loaded)` and drop the redundant load. Update the test helper signature accordingly (`_merged_closure_for(load_strategy(name))`).

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_registry_approvals.py -q`
Expected: the construction-closure test PASSES after Task 9 migrates the example (it imports `cross_sectional_momentum`). Existing approval tests stay green (closure is a superset).

- [ ] **Step 4: Commit**

```bash
git add algua/registry/approvals.py tests/test_registry_approvals.py
git commit -m "feat(registry): code_hash roots from the construction policy module (#141)"
```

---

## Task 6: Engine per-bar path uses signal → construct

**Files:**
- Modify: `algua/backtest/engine.py` (`_decision_weights`, `_canonical_row`)
- Test: `tests/test_backtest_engine.py`, `tests/test_engine_symbol_mask.py`

- [ ] **Step 1: Confirm the existing per-bar tests fail only on the migrated example**

The per-bar loop already calls `strategy.target_weights(view[, f_asof])` — which now composes signal → construct. So `_decision_weights` needs **no change** to its decision call. Verify:

Run: `grep -n "target_weights" algua/backtest/engine.py`
Expected: `_decision_weights` calls `strategy.target_weights(view)` / `strategy.target_weights(view, f_asof)`; `_canonical_row` calls `strategy.target_weights(view)`. These remain correct.

- [ ] **Step 2: Update `_canonical_row` docstring only**

`_canonical_row` stays weight-level (it returns `target_weights`, i.e. `construct(signal(view), view)`) — this is exactly what the weight-level parity guard must compare against. Update its docstring to say "the canonical per-bar weights = construct(signal(view), view) over the expanding slice", no logic change.

- [ ] **Step 3: Run the engine tests**

Run: `uv run pytest tests/test_backtest_engine.py tests/test_engine_symbol_mask.py -q`
Expected: PASS after Task 9 migrates the example strategies these tests load. (No engine-logic change in this task; this task is a verification + docstring checkpoint that the composition flows through unchanged.)

- [ ] **Step 4: Commit**

```bash
git add algua/backtest/engine.py
git commit -m "docs(engine): per-bar loop composes signal->construct via target_weights (#141)"
```

---

## Task 7: Engine fast path — `signal_panel` + per-bar construct + weight-level guard

**Files:**
- Modify: `algua/backtest/engine.py` (`_decision_weights_fast`, `_assert_parity`, `_decision_weights_fast_or_loop`)
- Test: `tests/test_fast_path.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fast_path.py`:

```python
def test_fast_path_matches_loop_for_momentum(sample_provider):  # reuse the file's provider fixture
    from datetime import datetime
    from algua.backtest.engine import _decision_weights, _decision_weights_fast, simulate
    from algua.strategies.loader import load_strategy
    s = load_strategy("cross_sectional_momentum")
    # The public simulate() uses the fast path; an equivalent loop run must agree end-to-end.
    # (Reuse the existing fast-path test's construction of bars/adj; assert weights equality.)
    # See existing tests in this file for the bars/adj fixtures and the per-bar vs fast comparison.


def test_divergent_signal_panel_raises(sample_provider):
    import pandas as pd
    import pytest
    from algua.backtest.engine import BacktestError, _decision_weights_fast
    from algua.strategies.loader import load_strategy
    s = load_strategy("cross_sectional_momentum")
    # Monkeypatch signal_panel_fn to return a deliberately wrong scores matrix; the WEIGHT-level
    # parity guard must raise rather than trust it.
    s.signal_panel_fn = lambda bars, params: pd.DataFrame()  # empty -> all flat, disagrees with loop
    # ... build bars/adj as the other tests do, then:
    # with pytest.raises(BacktestError):
    #     _decision_weights_fast(s, bars, adj)
```

> The existing `tests/test_fast_path.py` already constructs `bars`/`adj` and compares the fast path to the loop. Adapt those fixtures: assert the migrated momentum strategy's `_decision_weights_fast` equals `_decision_weights`, and that a divergent `signal_panel` raises. Keep the existing parity test bodies, updated to the new attribute name `signal_panel_fn`.

- [ ] **Step 2: Rewrite `_decision_weights_fast`**

Replace `_decision_weights_fast` in `algua/backtest/engine.py`:

```python
def _decision_weights_fast(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> pd.DataFrame:
    """Vectorized fast path: call the strategy's `signal_panel` ONCE for the whole period to get the
    SCORES matrix, then apply the construction policy PER BAR (with the same expanding `view_t` the
    loop uses) + the shared risk walls. A fail-closed WEIGHT-level parity guard then confirms the
    result equals the canonical per-bar `construct(signal(view), view)` on a bounded sample. Static
    universe only; pre-lag, like the loop.

    The speedup is computing the signal once instead of recomputing it on the expanding view each
    bar; construction stays per-bar (cheap for the view-independent starter policies). The scores
    matrix is NOT NaN-filled before construction — a missing score means 'no opinion' and the policy
    drops it; only the FINAL weights are zero-filled to flat.
    """
    panel = strategy.signal_panel(bars)
    assert panel is not None  # caller guarantees signal_panel_fn is set
    if not isinstance(panel, pd.DataFrame):
        raise BacktestError(
            f"strategy {strategy.name!r} signal_panel returned "
            f"{type(panel).__name__}, expected a DataFrame"
        )
    columns = adj.columns
    warmup = strategy.execution.warmup_bars
    # Reindex the SCORES onto the simulation grid WITHOUT filling NaN (missing score != 0 score).
    scores = panel.reindex(index=adj.index, columns=columns)

    bars_sorted = bars.sort_index()
    end_pos = bars_sorted.index.searchsorted(adj.index, side="right")

    weights = pd.DataFrame(0.0, index=adj.index, columns=columns)
    for i, (t, stop) in enumerate(zip(adj.index, end_pos, strict=True)):
        if i < warmup:
            continue  # warm-up: held flat by SKIPPING construction (weights stay 0)
        view_t = bars_sorted.iloc[:stop]
        scores_row = scores.iloc[i].dropna()  # drop missing/NaN; policy also drops non-finite
        w = strategy.construct(scores_row, view_t)
        if len(w) == 0:
            continue
        try:
            validate_decision_weights(w, strategy.execution, strategy.name)
        except RiskBreach as breach:
            raise BacktestError(f"{breach.detail} at {t}") from breach
        row = w.reindex(columns).fillna(0.0)
        weights.loc[t, row.index] = row.to_numpy()

    _assert_parity(strategy, bars_sorted, end_pos, weights, warmup)
    return weights
```

- [ ] **Step 3: Keep `_assert_parity` weight-level (rename helper)**

`_assert_parity` already recomputes `_canonical_row` (= `construct(signal(view), view)`) and compares **weights** with `WEIGHT_TOL`. Keep it. Only verify `_canonical_row` calls `strategy.target_weights(view)` (the composed pipeline) — no change needed. Update its docstring to: "compares the fast-path weights row against the canonical per-bar construct(signal(view), view); a discontinuous policy near-tie that a signal-level check could miss is caught here because we compare final weights."

- [ ] **Step 4: `_decision_weights_fast_or_loop` selector**

Change the guard condition from `strategy.panel_fn is None` to `strategy.signal_panel_fn is None`:

```python
    if strategy.signal_panel_fn is None or universe_by_date is not None or fundamentals is not None:
        return _decision_weights(
            strategy, bars, adj, universe_by_date=universe_by_date, fundamentals=fundamentals
        )
    return _decision_weights_fast(strategy, bars, adj)
```

- [ ] **Step 5: Run the fast-path tests**

Run: `uv run pytest tests/test_fast_path.py -q`
Expected: PASS after Task 9 (the example migrates). The divergent-panel test raises `BacktestError`.

- [ ] **Step 6: Commit**

```bash
git add algua/backtest/engine.py tests/test_fast_path.py
git commit -m "feat(engine): fast path vectorizes signal, constructs per-bar, weight-level parity guard (#141)"
```

---

## Task 8: Sweep — construction namespace routing + re-validation

**Files:**
- Modify: `algua/backtest/sweep.py` (`_override`)
- Test: `tests/test_sweep.py`, `tests/test_sweep_override.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sweep_override.py`:

```python
def test_override_routes_construction_namespace():
    from algua.strategies.loader import load_strategy
    from algua.backtest.sweep import _override
    s = load_strategy("cross_sectional_momentum")  # construction top_k_equal_weight, top_k=3
    out = _override(s, {"construction.top_k": 5, "lookback": 30})
    assert out.config.construction_params["top_k"] == 5
    assert out.config.params["lookback"] == 30
    assert out.construct_fn is s.construct_fn
    assert out.signal_panel_fn is s.signal_panel_fn


def test_override_rejects_unknown_signal_key():
    import pytest
    from algua.strategies.loader import load_strategy
    from algua.backtest.sweep import _override
    s = load_strategy("cross_sectional_momentum")
    with pytest.raises(ValueError):
        _override(s, {"not_a_real_param": 1})  # non-prefixed key not in CONFIG.params


def test_override_revalidates_construction_params():
    import pytest
    from algua.strategies.loader import load_strategy
    from algua.backtest.sweep import _override
    s = load_strategy("cross_sectional_momentum")
    with pytest.raises(ValueError):
        _override(s, {"construction.top_k": 0})  # fails the policy validator
```

- [ ] **Step 2: Rewrite `_override`**

Replace `_override` in `algua/backtest/sweep.py`:

```python
from algua.portfolio.construction import ConstructionError, validate_construction_params

_CONSTRUCTION_PREFIX = "construction."


def _override(strategy: LoadedStrategy, combo: dict[str, Any]) -> LoadedStrategy:
    """Return a LoadedStrategy whose params/construction_params are the base merged with `combo`.

    A grid key prefixed `construction.` tunes `construction_params` (re-validated by the policy);
    any other key tunes signal `params` and MUST already exist in the base params (so a typo'd key
    is rejected, never a silent no-op). Preserves the resolved construction policy + signal_panel.
    Does not mutate the base strategy/config.
    """
    new_params = dict(strategy.config.params)
    new_cparams = dict(strategy.config.construction_params)
    for key, value in combo.items():
        if key.startswith(_CONSTRUCTION_PREFIX):
            new_cparams[key[len(_CONSTRUCTION_PREFIX):]] = value
        else:
            if key not in strategy.config.params:
                raise ValueError(
                    f"sweep key {key!r} is not a base signal param "
                    f"{sorted(strategy.config.params)}; prefix with 'construction.' to tune the "
                    f"construction policy"
                )
            new_params[key] = value
    try:
        validate_construction_params(strategy.config.construction, new_cparams)
    except ConstructionError as exc:
        raise ValueError(f"swept construction_params invalid: {exc}") from exc
    new_config = strategy.config.model_copy(
        update={"params": new_params, "construction_params": new_cparams}
    )
    return LoadedStrategy(
        config=new_config,
        construct_fn=strategy.construct_fn,
        signal_fn=strategy.signal_fn,
        signal_panel_fn=strategy.signal_panel_fn,
        fundamentals_signal_fn=strategy.fundamentals_signal_fn,
    )
```

- [ ] **Step 3: Run the sweep tests**

Run: `uv run pytest tests/test_sweep.py tests/test_sweep_override.py -q`
Expected: PASS after Task 9. The existing `test_sweep.py` cases that sweep `lookback` still work (it's a base param); any case that swept `top_k` must change its grid key to `construction.top_k` (handle in Task 12).

- [ ] **Step 4: Commit**

```bash
git add algua/backtest/sweep.py tests/test_sweep_override.py
git commit -m "feat(sweep): construction.<key> namespace routing + revalidation in _override (#141)"
```

---

## Task 9: Migrate the example strategies

**Files:**
- Modify: `algua/strategies/examples/cross_sectional_momentum.py`
- Modify: `algua/strategies/examples/fundamentals_earnings_tilt.py`
- Test: `tests/test_strategy_momentum.py`

- [ ] **Step 1: Rewrite `cross_sectional_momentum.py`**

```python
"""Cross-sectional momentum: SIGNAL = trailing return per symbol; CONSTRUCTION = top-k equal weight."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="cross_sectional_momentum",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Trailing `lookback`-bar return per symbol (the alpha score). Empty until enough history."""
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    return (wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0).dropna()


def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """OPTIONAL vectorized SCORES twin of `signal` (the canonical per-bar signal stays above). The
    full trailing-return matrix in one shot; rows without `lookback` history are all-NaN (the
    construction policy drops NaN, so those bars are flat — matching `signal` returning empty)."""
    lookback = int(params["lookback"])
    wide = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return wide / wide.shift(lookback) - 1.0
```

> Note: `top_k` moved from `params` to `construction_params`. `signal` no longer selects/weights — `top_k_equal_weight` does. `signal_panel` returns SCORES (the momentum matrix), not weights; rows with insufficient history are NaN and the policy drops them.

- [ ] **Step 2: Rewrite `fundamentals_earnings_tilt.py`**

```python
"""Earnings-yield tilt: SIGNAL = latest KNOWN diluted EPS per symbol; CONSTRUCTION = equal-weight
the positive-score names. A minimal demonstration of the as-of fundamentals lane (issue #132)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="fundamentals_earnings_tilt",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"metric": "eps_diluted"},
    construction="equal_weight_positive",
    needs_fundamentals=True,
)


def signal(view: pd.DataFrame, params: dict[str, Any], fundamentals: pd.DataFrame) -> pd.Series:
    """Latest-known value of `metric` per symbol (the score). equal_weight_positive then holds the
    names with a positive score, equal weight."""
    metric = str(params["metric"])
    rows = fundamentals[fundamentals["metric"] == metric]
    if rows.empty:
        return pd.Series(dtype="float64")
    return rows.groupby("symbol")["value"].last()
```

- [ ] **Step 3: Update `tests/test_strategy_momentum.py`**

Migrate the momentum test to call `signal` (scores) + the policy, or to load via `load_strategy` and assert `target_weights`. Replace any `compute_weights(view, params)` calls with `signal(view, params)` and assert it returns SCORES (a return Series), and that `top_k_equal_weight(signal(...), view, {"top_k": 3})` selects the top 3. Run:

Run: `uv run pytest tests/test_strategy_momentum.py -q`
Expected: PASS.

- [ ] **Step 4: Run the dependent tasks' tests now that examples exist**

Run: `uv run pytest tests/test_strategy_loader.py tests/test_backtest_engine.py tests/test_engine_symbol_mask.py tests/test_fast_path.py tests/test_registry_approvals.py tests/test_config_hash_fields.py tests/test_sweep.py tests/test_sweep_override.py -q`
Expected: PASS (these were waiting on the migrated examples).

- [ ] **Step 5: Commit**

```bash
git add algua/strategies/examples/ tests/test_strategy_momentum.py
git commit -m "feat(strategies): migrate example strategies to signal + construction (#141)"
```

---

## Task 10: CLI template + contracts/hindsight docstrings + `decide()` verify

**Files:**
- Modify: `algua/cli/strategy_cmd.py` (`_TEMPLATE`)
- Modify: `algua/contracts/types.py` (`Strategy` docstring)
- Modify: `algua/data/hindsight.py` (comment)
- Test: `tests/test_cli_*` (whatever currently asserts the scaffold), `tests/test_paper_loop.py`, `tests/test_live_loop.py`

- [ ] **Step 1: Replace the scaffold template**

In `algua/cli/strategy_cmd.py`, replace `_TEMPLATE`:

```python
_TEMPLATE = '''\
"""Strategy: {name}. Author `signal` (cross-sectional scores per symbol); the named construction
policy in CONFIG turns scores into target weights."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="{name}",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={{"lookback": 60}},
    construction="top_k_equal_weight",
    construction_params={{"top_k": 2}},
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Return a score per symbol (higher = more attractive). NOT weights — the construction policy
    maps scores to weights. See algua/portfolio/construction.py for available policies."""
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    return (wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0).dropna()
'''
```

- [ ] **Step 2: Update `Strategy` protocol docstring**

In `algua/contracts/types.py`, add to the `Strategy` protocol docstring: "`target_weights` is the composed pipeline `construct(signal(features), features)` — see `algua/strategies/base.py` and `algua/portfolio/construction.py` (issue #141)." (No signature change.)

- [ ] **Step 3: Update the stale hindsight comment**

In `algua/data/hindsight.py`, change the comment referencing `compute_weights` to `signal` (the decision-lane function hindsight must be structurally unable to reach).

- [ ] **Step 4: Verify `decide()` is unchanged**

`algua/live/paper_loop.py::decide` calls `strategy.target_weights(view)` — now the composed pipeline. No change needed. Confirm:

Run: `uv run pytest tests/test_paper_loop.py tests/test_live_loop.py tests/test_decision_parity.py -q`
Expected: PASS after Task 12 migrates any in-file strategy stubs (paper/live tests that define an inline `compute_weights` strategy must migrate to `signal` + `construction`).

- [ ] **Step 5: Update + run the CLI scaffold test**

Run: `uv run pytest tests/ -k "strategy_new or scaffold or cli_strategy" -q`
Expected: PASS — update any assertion that greps the scaffold for `compute_weights` to `signal`.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/strategy_cmd.py algua/contracts/types.py algua/data/hindsight.py tests/
git commit -m "feat(cli): strategy scaffold authors signal + construction; docstrings (#141)"
```

---

## Task 11: Import-linter contract for `algua.portfolio` purity

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the contract**

In `pyproject.toml`, after the `features` purity contract, add:

```toml
[[tool.importlinter.contracts]]
name = "portfolio construction layer is pure (no algua imports beyond contracts + features)"
type = "forbidden"
source_modules = ["algua.portfolio"]
forbidden_modules = [
    "algua.cli", "algua.registry", "algua.data", "algua.backtest", "algua.strategies",
    "algua.live", "algua.execution", "algua.tracking", "algua.research", "algua.knowledge",
]
```

- [ ] **Step 2: Run import-linter**

Run: `uv run lint-imports`
Expected: "Contracts: N kept, 0 broken." (`construction.py` imports only stdlib + numpy/pandas; `strategies`/`backtest` importing `portfolio` is allowed and does not break this forbidden contract, which only constrains what `algua.portfolio` itself imports.)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build(lint-imports): algua.portfolio is a pure layer (#141)"
```

---

## Task 12: Migrate the remaining contract test surface

**Files (each `compute_weights`/weights-`target_weights` usage migrates to `signal` + `construction`):**
- `tests/test_contracts.py`, `tests/test_strategies_target_weights_fail_closed.py`,
  `tests/test_decision_parity.py`, `tests/test_walkforward.py`, `tests/test_sweep.py`,
  `tests/test_backtest_metrics.py`, `tests/test_cli_paper.py`, `tests/test_paper_loop.py`,
  `tests/test_live_loop.py`, and any others surfaced by the gate.

- [ ] **Step 1: Find every remaining old-contract usage**

Run: `grep -rln "compute_weights\|panel_fn\|\.fn=\|fundamentals_fn\|ComputeWeights" tests/ algua/`
Expected: a list of files still on the old contract. For each, apply the **transform rules** below.

- [ ] **Step 2: Apply the transform rules**

For every inline/fixture strategy in a test:
1. Rename the authored fn `compute_weights(view, params)` → `signal(view, params)`; have it return **scores** (for selection strategies, return the raw score Series — drop any in-fn `top_k`/selection, which moves to the policy).
2. Add `construction=...` + `construction_params=...` to the `StrategyConfig`. Use `score_proportional_long` for a strategy whose old weights were proportional, `equal_weight_positive` for an equal-weight-of-positives, `top_k_equal_weight` for a top-k.
3. Build `LoadedStrategy` with the new fields: `LoadedStrategy(config=..., signal_fn=signal, construct_fn=get_construction_policy(cfg.construction))` (or `fundamentals_signal_fn=` / `signal_panel_fn=` as appropriate). Replace any `panel_fn=`/`fn=`/`fundamentals_fn=` kwargs.
4. Any test that asserts `compute_weights(...)` returned specific WEIGHTS now either asserts `signal(...)` returns the expected SCORES, or asserts `loaded.target_weights(view)` returns the expected weights.
5. Sweep grids that varied `top_k` change the grid key to `construction.top_k`.

Worked example — a fixture strategy that was:
```python
def compute_weights(view, params):
    scores = momentum(view)            # returns a Series of scores
    return _top(scores, params["top_k"])
CONFIG = StrategyConfig(name="x", universe=[...], execution=..., params={"top_k": 2})
```
becomes:
```python
def signal(view, params):
    return momentum(view)              # scores only
CONFIG = StrategyConfig(name="x", universe=[...], execution=...,
                        construction="top_k_equal_weight", construction_params={"top_k": 2})
```

- [ ] **Step 3: Run the full suite repeatedly until green**

Run: `uv run pytest -q`
Expected: iterate file-by-file until all pass. Each failure names a file still on the old contract — apply the transform rules.

- [ ] **Step 4: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: migrate contract test surface to signal + construction (#141)"
```

---

## Self-Review Checklist (run before opening the PR)

- [ ] **Spec coverage:** signal layer (Task 3,9) · construction library + validators (Task 1) · execution overlay unchanged (Task 6) · composition/parity (Task 3,6) · fast path weight-level guard (Task 7) · identity config_hash+code_hash (Task 2,5) · CONFIG/loader (Task 2,4) · sweep namespace (Task 8) · template/docstrings (Task 10) · import purity (Task 11) · examples + test migration (Task 9,12). No spec section unmapped.
- [ ] **No placeholders** remain in committed code.
- [ ] **Type/name consistency:** field names `signal_fn` / `signal_panel_fn` / `fundamentals_signal_fn` / `construct_fn`; methods `signal` / `signal_panel` / `construct` / `target_weights`; property `authored_signal`; policy ids `top_k_equal_weight` / `equal_weight_positive` / `score_proportional_long` — used identically across base.py, loader.py, engine.py, sweep.py, approvals.py, examples, tests.
- [ ] **Gate green** at the branch tip: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

---

## Deferred (NOT in this slice — do not build)

Vol-targeting/turnover/risk-budgeting policies + a `ConstructionContext` (#136); multi-output/long-short signals + the signal registry (#140); per-layer evaluation CLI (#137); formal per-policy param schemas; a `construct_panel` vectorization for view-dependent construction.
