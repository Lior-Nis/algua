# Standalone Factor Evaluation (#140 slice B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an agent evaluate a single catalogued factor on its own — a real PIT backtest via a 1-factor→weights adapter plus a construction-free IC/IR predictive-power block — emitted as ephemeral JSON, with factors staying off the live path.

**Architecture:** Introduce a uniform "standalone-evaluable" (signal-shaped) factor contract in the pure `algua.features` layer; a seed alpha (`xs_trailing_return`) that `cross_sectional_momentum` composes; a `algua.backtest.factor_eval` module that wraps a standalone factor as a synthetic `LoadedStrategy` (reusing the existing `engine.run`) and computes rank IC/IR from a PIT score panel; and an `algua factor eval` CLI mirroring `backtest run`.

**Tech Stack:** Python 3.12, pandas, numpy, typer, pytest. Spec: `docs/superpowers/specs/2026-06-15-factor-standalone-eval-140-design.md`.

---

## File Structure

- `algua/features/catalogue.py` (MODIFY) — add `FactorSpec.standalone`, `@factor(standalone=...)` shape validation, `load_factor_callable`.
- `algua/features/alphas.py` (CREATE) — seed standalone alpha `xs_trailing_return`.
- `algua/strategies/momentum/cross_sectional_momentum.py` (MODIFY) — `signal` delegates to the alpha (composition demo).
- `algua/backtest/factor_eval.py` (CREATE) — adapter (`build_factor_strategy`), IC math (`factor_ic`), PIT score panel + forward returns, orchestration (`evaluate_factor`, `FactorEvalResult`).
- `algua/cli/factor_cmd.py` (MODIFY) — add the `eval` command.
- Tests under `tests/` (CREATE/MODIFY as noted per task).
- `~/.claude/.../skills/interpret-results` note (MODIFY) — IC-not-FDR-corrected caveat (Task 10).

**Conventions to follow (already in the repo):**
- A factor is **standalone-evaluable** iff signal-shaped: `(view: pd.DataFrame, params: dict) -> pd.Series`.
- `view` is the long bar-schema frame: timestamp index (index name `"timestamp"`), columns include `"symbol"` and `"adj_close"`. Pivot with `view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")`.
- CLI success payloads use `ok({...})` then `emit(...)`; errors raise `ValueError`/`LookupError` under `@json_errors(...)`.
- The synthetic strategy name uses the reserved prefix `__factor__:` — never registered, never gate-tokened, never walk-forwarded.

---

## Task 1: `standalone` contract + shape validation

**Files:**
- Modify: `algua/features/catalogue.py`
- Test: `tests/test_factor_catalogue.py` (existing file — append; if absent, create)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_factor_catalogue.py` (create the file with the imports if it does not exist):

```python
import pytest

from algua.features.catalogue import FactorKind, factor


def test_standalone_factor_accepts_signal_shaped_fn():
    @factor(standalone=True, summary="ok", kind=FactorKind.MOMENTUM)
    def good(view, params):  # 2 positional-or-keyword args
        return view

    assert good.__factor_spec__.standalone is True


def test_factor_defaults_to_not_standalone():
    @factor(summary="ok")
    def helper(prices, lookback):
        return prices

    assert helper.__factor_spec__.standalone is False


@pytest.mark.parametrize(
    "bad",
    [
        lambda v: v,                       # 1 arg
        lambda v, p, x: v,                 # 3 args
        lambda *a, **k: a,                 # varargs
        lambda v, *a: v,                   # trailing *args
    ],
)
def test_standalone_rejects_non_signal_shape(bad):
    with pytest.raises(ValueError, match="standalone"):
        factor(standalone=True, summary="x")(bad)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_factor_catalogue.py -q`
Expected: FAIL — `factor()` has no `standalone` kwarg / `FactorSpec` has no `standalone`.

- [ ] **Step 3: Implement**

In `algua/features/catalogue.py`, add `standalone: bool` to `FactorSpec` (after `doc`):

```python
@dataclass(frozen=True)
class FactorSpec:
    name: str
    summary: str
    kind: FactorKind
    tags: tuple[str, ...]
    data_needs: tuple[DataCapability, ...]
    import_path: str
    module: str
    signature: str
    doc: str | None
    standalone: bool = False
```

Add a shape-check helper above `factor`:

```python
def _assert_signal_shaped(fn: Callable[..., Any], name: str) -> None:
    """A standalone-evaluable factor must be signal-shaped: exactly two POSITIONAL_OR_KEYWORD
    params (view, params) and no *args/**kwargs. Structural arity check only — it cannot verify
    semantics (it cannot tell (view, params) from (prices, lookback)); marking a factor standalone
    is a deliberate author assertion. Fails closed on the obvious mistakes (transforms, varargs)."""
    params = list(inspect.signature(fn).parameters.values())
    ok = len(params) == 2 and all(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params
    )
    if not ok:
        raise ValueError(
            f"factor {name!r} declares standalone=True but is not signal-shaped "
            f"(view, params); got signature {inspect.signature(fn)}"
        )
```

Extend `factor(...)` signature with `standalone: bool = False`, call the check, and pass it to the spec:

```python
def factor(
    *,
    name: str | None = None,
    summary: str | None = None,
    tags: Iterable[str] = (),
    kind: FactorKind = FactorKind.OTHER,
    data_needs: Iterable[DataCapability] | None = None,
    standalone: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = name or fn.__name__
        if standalone:
            _assert_signal_shaped(fn, resolved_name)
        doc = inspect.getdoc(fn)
        resolved_summary = summary or _first_nonempty_line(doc)
        if not resolved_summary:
            raise ValueError(
                f"factor {resolved_name!r} needs a summary (pass summary= or add a docstring)"
            )
        spec = FactorSpec(
            name=resolved_name,
            summary=resolved_summary,
            kind=kind,
            tags=tuple(tags),
            data_needs=tuple(data_needs) if data_needs is not None else (DataCapability.OHLCV,),
            import_path=f"{fn.__module__}:{fn.__qualname__}",
            module=fn.__module__,
            signature=str(inspect.signature(fn)),
            doc=doc,
            standalone=standalone,
        )
        setattr(fn, _SPEC_ATTR, spec)
        return fn

    return decorate
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_factor_catalogue.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/features/catalogue.py tests/test_factor_catalogue.py
git commit -m "feat(140): standalone-evaluable factor contract + signal-shape validation"
```

---

## Task 2: `load_factor_callable` resolver

**Files:**
- Modify: `algua/features/catalogue.py`
- Test: `tests/test_factor_catalogue.py`

- [ ] **Step 1: Write the failing test**

```python
from algua.features.catalogue import get_factor, load_factor_callable


def test_load_factor_callable_round_trips_to_the_function():
    spec = get_factor("momentum")
    fn = load_factor_callable(spec)
    assert callable(fn)
    assert getattr(fn, "__factor_spec__").name == "momentum"


def test_load_factor_callable_fails_closed_on_stamp_mismatch():
    import dataclasses
    from algua.features.catalogue import FactorNotFound

    spec = get_factor("momentum")
    # Point import_path at a real attribute that is NOT a stamped factor.
    bad = dataclasses.replace(spec, import_path="algua.features.catalogue:factor")
    with pytest.raises(FactorNotFound):
        load_factor_callable(bad)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_factor_catalogue.py -k load_factor_callable -q`
Expected: FAIL — `load_factor_callable` not defined.

- [ ] **Step 3: Implement**

Add to `algua/features/catalogue.py` (uses the already-imported `importlib`):

```python
def load_factor_callable(spec: FactorSpec) -> Callable[..., Any]:
    """Resolve a FactorSpec back to its function object via ``import_path`` ("module:qualname").
    The catalogue scan already imported the module, so this is import-safe. Fails closed
    (``FactorNotFound``) if the resolved object is not the matching stamped factor — guarding
    against a spec whose import_path drifted off its function."""
    module_name, _, qualname = spec.import_path.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise FactorNotFound(spec.name) from None
    resolved = getattr(obj, _SPEC_ATTR, None)
    if resolved is None or resolved.name != spec.name:
        raise FactorNotFound(spec.name)
    return obj  # type: ignore[no-any-return]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_factor_catalogue.py -k load_factor_callable -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/features/catalogue.py tests/test_factor_catalogue.py
git commit -m "feat(140): load_factor_callable — resolve a FactorSpec back to its function"
```

---

## Task 3: seed standalone alpha `xs_trailing_return`

**Files:**
- Create: `algua/features/alphas.py`
- Test: `tests/test_alphas.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_alphas.py`:

```python
import pandas as pd

from algua.features.alphas import xs_trailing_return
from algua.features.catalogue import load_all_factors


def _view(rows):
    # long bar-schema frame: timestamp index + symbol/adj_close columns
    df = pd.DataFrame(rows, columns=["timestamp", "symbol", "adj_close"])
    return df.set_index("timestamp")


def test_xs_trailing_return_is_trailing_return_per_symbol():
    view = _view([
        ("2023-01-01", "AAA", 100.0), ("2023-01-01", "BBB", 100.0),
        ("2023-01-02", "AAA", 110.0), ("2023-01-02", "BBB", 90.0),
    ])
    scores = xs_trailing_return(view, {"lookback": 1})
    assert scores["AAA"] == pytest.approx(0.10)
    assert scores["BBB"] == pytest.approx(-0.10)


def test_xs_trailing_return_empty_before_enough_history():
    view = _view([("2023-01-01", "AAA", 100.0)])
    assert xs_trailing_return(view, {"lookback": 5}).empty


def test_xs_trailing_return_is_catalogued_standalone():
    specs = load_all_factors()
    assert "xs_trailing_return" in specs
    assert specs["xs_trailing_return"].standalone is True


import pytest  # noqa: E402  (kept after fixtures for readability)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_alphas.py -q`
Expected: FAIL — `algua.features.alphas` does not exist.

- [ ] **Step 3: Implement**

Create `algua/features/alphas.py`:

```python
"""Standalone-evaluable alpha factors: signal-shaped (view, params) -> cross-sectional scores.

Distinct from indicators.py (arbitrary-signature building blocks): an alpha here IS reusable as a
strategy `signal`, and can be evaluated on its own (algua factor eval). Pure layer — pandas only.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.features.catalogue import FactorKind, factor
from algua.features.indicators import momentum


@factor(
    standalone=True,
    summary="Cross-sectional trailing return per symbol over `lookback` bars (the momentum alpha).",
    kind=FactorKind.MOMENTUM,
    tags=["momentum", "cross-sectional", "alpha"],
)
def xs_trailing_return(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Trailing `lookback`-bar return per symbol — the cross_sectional_momentum alpha as a
    reusable, individually-evaluable factor. Empty until `lookback`+1 bars of history exist."""
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    return momentum(wide.iloc[-1 - lookback :], lookback).iloc[-1].dropna()
```

Note: `momentum(prices, lookback)` returns `prices / prices.shift(lookback) - 1`; slicing the last `lookback+1` rows then taking `.iloc[-1]` yields the per-symbol trailing return at the final bar — identical math to the strategy's inline `wide.iloc[-1] / wide.iloc[-1 - lookback] - 1`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_alphas.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/features/alphas.py tests/test_alphas.py
git commit -m "feat(140): seed standalone alpha xs_trailing_return"
```

---

## Task 4: rewire `cross_sectional_momentum` to compose the alpha

**Files:**
- Modify: `algua/strategies/momentum/cross_sectional_momentum.py`
- Test: `tests/test_cross_sectional_momentum_rewire.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cross_sectional_momentum_rewire.py`:

```python
import pandas as pd

from algua.features.alphas import xs_trailing_return
from algua.registry.lineage import factors_used_by
from algua.strategies.momentum import cross_sectional_momentum as csm


def _view():
    rows = []
    for i, ts in enumerate(pd.date_range("2023-01-01", periods=70, freq="D")):
        rows.append((ts, "AAA", 100.0 + i))
        rows.append((ts, "BBB", 100.0 + 2 * i))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "adj_close"]).set_index("timestamp")


def test_signal_delegates_to_alpha_identically():
    view = _view()
    params = {"lookback": 60}
    pd.testing.assert_series_equal(csm.signal(view, params), xs_trailing_return(view, params))


def test_lineage_reports_the_composed_factor():
    used = {s.name for s in factors_used_by("cross_sectional_momentum")}
    assert "xs_trailing_return" in used
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cross_sectional_momentum_rewire.py -q`
Expected: FAIL — `factors_used_by` does not yet see `xs_trailing_return` (strategy does not import it).

- [ ] **Step 3: Implement**

In `algua/strategies/momentum/cross_sectional_momentum.py`, add the top-level import and delegate `signal` to the alpha (leave `CONFIG` and `signal_panel` unchanged — `signal_panel`'s matrix math is identical, so the engine's fast-path parity guard still holds):

```python
from algua.features.alphas import xs_trailing_return


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Trailing `lookback`-bar return per symbol (the alpha score), via the catalogued
    `xs_trailing_return` factor (issue #140 composition)."""
    return xs_trailing_return(view, params)
```

- [ ] **Step 4: Run to verify pass + parity gate**

Run: `uv run pytest tests/test_cross_sectional_momentum_rewire.py -q`
Expected: PASS.

Run the existing momentum/backtest tests to confirm byte-identical behavior (fast-path parity guard intact):
Run: `uv run pytest tests/ -k "cross_sectional or backtest or parity or signal_panel" -q`
Expected: PASS (no behavioral change).

- [ ] **Step 5: Commit**

```bash
git add algua/strategies/momentum/cross_sectional_momentum.py tests/test_cross_sectional_momentum_rewire.py
git commit -m "feat(140): cross_sectional_momentum composes the xs_trailing_return factor (lineage demo)"
```

---

## Task 5: `build_factor_strategy` adapter

**Files:**
- Create: `algua/backtest/factor_eval.py`
- Test: `tests/test_factor_eval_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_factor_eval_adapter.py`:

```python
import pytest

from algua.backtest.factor_eval import build_factor_strategy
from algua.features.catalogue import get_factor


def test_build_factor_strategy_wraps_a_standalone_factor():
    spec = get_factor("xs_trailing_return")
    strat = build_factor_strategy(
        spec,
        symbols=["AAA", "BBB"],
        params={"lookback": 5},
        construction="top_k_equal_weight",
        construction_params={"top_k": 1},
    )
    assert strat.name == "__factor__:xs_trailing_return"
    assert strat.universe == ["AAA", "BBB"]
    assert strat.signal_fn is not None
    assert strat.config.construction == "top_k_equal_weight"


def test_build_factor_strategy_rejects_non_standalone():
    spec = get_factor("momentum")  # standalone=False
    with pytest.raises(ValueError, match="not standalone-evaluable"):
        build_factor_strategy(
            spec, symbols=["AAA"], params={"lookback": 5},
            construction="equal_weight_positive", construction_params={},
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_factor_eval_adapter.py -q`
Expected: FAIL — `algua.backtest.factor_eval` does not exist.

- [ ] **Step 3: Implement**

Create `algua/backtest/factor_eval.py` (adapter portion):

```python
"""Standalone factor evaluation (issue #140 slice B): wrap a single catalogued, signal-shaped
factor as an ephemeral synthetic strategy, run it through the existing backtest engine, and
compute construction-free rank IC/IR. Factors are NEVER registered, gate-tokened, or live-pathed:
the synthetic name uses the reserved `__factor__:` prefix and nothing here touches the registry."""
from __future__ import annotations

from typing import Any

from algua.contracts.types import ExecutionContract
from algua.features.catalogue import FactorSpec, load_factor_callable
from algua.portfolio.construction import get_construction_policy, validate_construction_params
from algua.strategies.base import LoadedStrategy, StrategyConfig

SYNTHETIC_PREFIX = "__factor__:"


def build_factor_strategy(
    spec: FactorSpec,
    *,
    symbols: list[str],
    params: dict[str, Any],
    construction: str,
    construction_params: dict[str, Any],
    execution: ExecutionContract | None = None,
) -> LoadedStrategy:
    """Wrap a standalone-evaluable factor as a synthetic LoadedStrategy. Construction is required
    (no default) so factor eval imposes no hidden weighting bias. Rejects a non-standalone factor."""
    if not spec.standalone:
        raise ValueError(
            f"factor {spec.name!r} is not standalone-evaluable (not signal-shaped); "
            f"only standalone factors can be evaluated on their own"
        )
    validate_construction_params(construction, construction_params)
    fn = load_factor_callable(spec)
    config = StrategyConfig(
        name=f"{SYNTHETIC_PREFIX}{spec.name}",
        universe=symbols,
        execution=execution or ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params=params,
        construction=construction,
        construction_params=construction_params,
    )
    return LoadedStrategy(
        config=config,
        construct_fn=get_construction_policy(construction),
        signal_fn=fn,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_factor_eval_adapter.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/factor_eval.py tests/test_factor_eval_adapter.py
git commit -m "feat(140): factor->synthetic-strategy adapter (build_factor_strategy)"
```

---

## Task 6: `factor_ic` — pure rank-IC math

**Files:**
- Modify: `algua/backtest/factor_eval.py`
- Test: `tests/test_factor_ic.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_factor_ic.py`:

```python
import numpy as np
import pandas as pd
import pytest

from algua.backtest.factor_eval import factor_ic


def _panels(score_rows, return_rows):
    cols = ["AAA", "BBB", "CCC", "DDD"]
    idx = pd.date_range("2023-01-01", periods=len(score_rows), freq="D")
    return (pd.DataFrame(score_rows, index=idx, columns=cols),
            pd.DataFrame(return_rows, index=idx, columns=cols))


def test_perfectly_monotone_factor_has_ic_one():
    rows = [[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]]
    scores, rets = _panels(rows, rows)  # scores rank == return rank every bar
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["mean_ic"] == pytest.approx(1.0)
    assert ic["hit_rate"] == pytest.approx(1.0)
    assert ic["n_obs"] == 3


def test_sign_flipped_factor_has_ic_minus_one():
    rows = [[1, 2, 3, 4], [4, 3, 2, 1]]
    scores, rets = _panels(rows, [[-v for v in r] for r in rows])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["mean_ic"] == pytest.approx(-1.0)


def test_noise_factor_has_ic_near_zero():
    rng = np.random.default_rng(0)
    n = 200
    scores = pd.DataFrame(rng.normal(size=(n, 4)), columns=["AAA", "BBB", "CCC", "DDD"])
    rets = pd.DataFrame(rng.normal(size=(n, 4)), columns=["AAA", "BBB", "CCC", "DDD"])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert abs(ic["mean_ic"]) < 0.1
    assert ic["n_obs"] == n


def test_constant_cross_sections_are_skipped():
    rows = [[5, 5, 5, 5], [5, 5, 5, 5]]
    scores, rets = _panels(rows, [[1, 2, 3, 4], [4, 3, 2, 1]])
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["n_obs"] == 0
    assert ic["mean_ic"] is None


def test_too_few_observations_returns_none():
    rows = [[1, 2, 3, 4]]
    scores, rets = _panels(rows, rows)
    ic = factor_ic(scores, rets, min_cross_section=3)
    assert ic["n_obs"] == 1
    assert ic["ir"] is None and ic["t_stat"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_factor_ic.py -q`
Expected: FAIL — `factor_ic` not defined.

- [ ] **Step 3: Implement**

Add to `algua/backtest/factor_eval.py` (add `import math`, `import numpy as np`, `import pandas as pd` at top):

```python
def factor_ic(
    score_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    min_cross_section: int = 3,
) -> dict[str, Any]:
    """Cross-sectional rank (Spearman) Information Coefficient summary.

    Per timestamp: Spearman correlation between the factor scores and the forward returns over the
    symbols finite in both. Bars with a cross-section narrower than `min_cross_section`, or a
    degenerate (zero-variance -> NaN) correlation, are skipped. Aggregates: mean IC, sample IC std
    (ddof=1), IR = mean/std, t-stat = IR*sqrt(n), hit rate (share of IC>0), n_obs. A run with
    < 2 usable bars (or zero IC variance) returns explicit None rather than a misleading number.

    NOT multiple-testing corrected — the t-stat is raw (FDR accounting is #140 slice E)."""
    ics: list[float] = []
    common = score_panel.index.intersection(forward_returns.index)
    for t in common:
        pair = pd.DataFrame(
            {"s": score_panel.loc[t], "r": forward_returns.loc[t]}
        )
        pair = pair[np.isfinite(pair["s"]) & np.isfinite(pair["r"])]
        if len(pair) < min_cross_section:
            continue
        ic = pair["s"].corr(pair["r"], method="spearman")
        if pd.notna(ic):
            ics.append(float(ic))
    n = len(ics)
    base: dict[str, Any] = {
        "method": "spearman",
        "n_obs": n,
        "min_cross_section": min_cross_section,
        "fdr_corrected": False,
    }
    if n < 2:
        return {**base, "mean_ic": None, "ic_std": None, "ir": None,
                "t_stat": None, "hit_rate": None}
    arr = np.array(ics, dtype=float)
    mean_ic = float(arr.mean())
    ic_std = float(arr.std(ddof=1))
    ir = mean_ic / ic_std if ic_std > 0 else None
    t_stat = ir * math.sqrt(n) if ir is not None else None
    return {
        **base,
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ir": ir,
        "t_stat": t_stat,
        "hit_rate": float((arr > 0).mean()),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_factor_ic.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/factor_eval.py tests/test_factor_ic.py
git commit -m "feat(140): rank-IC/IR factor-quality metric (factor_ic)"
```

---

## Task 7: PIT score panel + forward returns

**Files:**
- Modify: `algua/backtest/factor_eval.py`
- Test: `tests/test_factor_eval_panel.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_factor_eval_panel.py`:

```python
import pandas as pd
import pytest

from algua.backtest.factor_eval import build_factor_strategy, score_panel, forward_returns
from algua.features.catalogue import get_factor


def _bars():
    rows = []
    for i, ts in enumerate(pd.date_range("2023-01-01", periods=10, freq="D")):
        rows.append((ts, "AAA", 100.0 + i))
        rows.append((ts, "BBB", 100.0 + 2.0 * i))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "adj_close"]).set_index("timestamp")


def _strategy():
    return build_factor_strategy(
        get_factor("xs_trailing_return"), symbols=["AAA", "BBB"],
        params={"lookback": 2}, construction="equal_weight_positive", construction_params={},
    )


def test_score_panel_is_pit_no_look_ahead():
    bars = _bars()
    full = score_panel(_strategy(), bars)
    truncated = score_panel(_strategy(), bars.loc[:full.index[5]])
    # the score at an early bar must not change when later bars are added
    pd.testing.assert_series_equal(full.loc[truncated.index[3]], truncated.loc[truncated.index[3]])


def test_forward_returns_offset_by_lag_and_horizon():
    bars = _bars()
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    fwd = forward_returns(adj, lag=1, horizon=1)
    t0 = adj.index[0]
    # entry at t0+lag=index[1], exit at index[2]: (102/101 - 1)
    assert fwd.loc[t0, "AAA"] == pytest.approx(adj["AAA"].iloc[2] / adj["AAA"].iloc[1] - 1)
    # the last (lag+horizon) rows have no future bar -> NaN
    assert fwd.iloc[-1].isna().all()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_factor_eval_panel.py -q`
Expected: FAIL — `score_panel` / `forward_returns` not defined.

- [ ] **Step 3: Implement**

Add to `algua/backtest/factor_eval.py`:

```python
def _adj_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """adj_close pivoted to (sorted unique timestamp index x symbol columns) — the decision grid."""
    return (
        bars.reset_index()
        .pivot(index="timestamp", columns="symbol", values="adj_close")
        .sort_index()
    )


def score_panel(strategy: LoadedStrategy, bars: pd.DataFrame) -> pd.DataFrame:
    """The factor's cross-sectional scores at every decision bar, PIT (data <= t only).

    For each timestamp t on the grid, calls the factor over the expanding window ending at t (the
    same expanding `view` the engine's per-bar loop uses), so a score at t can never see a bar
    after t. Returns a (timestamp x symbol) frame; bars before the factor has enough history
    contribute an all-NaN row."""
    bars_sorted = bars.sort_index()
    grid = _adj_grid(bars_sorted).index
    end_pos = bars_sorted.index.searchsorted(grid, side="right")
    rows: dict[pd.Timestamp, pd.Series] = {}
    for t, stop in zip(grid, end_pos, strict=True):
        rows[t] = strategy.signal(bars_sorted.iloc[:stop])
    panel = pd.DataFrame.from_dict(rows, orient="index")
    return panel.reindex(columns=_adj_grid(bars_sorted).columns)


def forward_returns(adj: pd.DataFrame, *, lag: int, horizon: int) -> pd.DataFrame:
    """Per-symbol forward return realized AFTER the decision lag: a score known at t is tradable at
    t+lag, so the label is adj_{t+lag+horizon} / adj_{t+lag} - 1. The trailing (lag+horizon) rows
    have no future bar and are NaN (skipped by the IC cross-section filter)."""
    entry = adj.shift(-lag)
    exit_ = adj.shift(-(lag + horizon))
    return exit_ / entry - 1.0
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_factor_eval_panel.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/factor_eval.py tests/test_factor_eval_panel.py
git commit -m "feat(140): PIT score panel + lag/horizon forward returns for IC"
```

---

## Task 8: `evaluate_factor` orchestration + `FactorEvalResult`

**Files:**
- Modify: `algua/backtest/factor_eval.py`
- Test: `tests/test_factor_eval_run.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_factor_eval_run.py`:

```python
from datetime import UTC, datetime

from algua.backtest._sample import SyntheticProvider
from algua.backtest.factor_eval import evaluate_factor
from algua.features.catalogue import get_factor


def test_evaluate_factor_returns_backtest_and_ic_blocks():
    spec = get_factor("xs_trailing_return")
    result = evaluate_factor(
        spec,
        SyntheticProvider(seed=0),
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 6, 30, tzinfo=UTC),
        symbols=["AAA", "BBB", "CCC"],
        params={"lookback": 10},
        construction="top_k_equal_weight",
        construction_params={"top_k": 1},
        horizon=1,
    )
    payload = result.to_dict()
    assert payload["factor"] == "xs_trailing_return"
    assert payload["standalone"] is True
    assert "metrics" in payload["backtest"]
    assert payload["backtest"]["strategy"] == "__factor__:xs_trailing_return"
    assert payload["ic"]["method"] == "spearman"
    assert payload["ic"]["fdr_corrected"] is False
    assert payload["horizon"] == 1
```

(`SyntheticProvider` synthesizes deterministic bars for any symbols requested — `AAA/BBB/CCC` are valid; confirmed in `algua/backtest/_sample.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_factor_eval_run.py -q`
Expected: FAIL — `evaluate_factor` not defined.

- [ ] **Step 3: Implement**

Add to `algua/backtest/factor_eval.py` (add imports: `from dataclasses import dataclass`; `from collections.abc import Collection, Mapping`; `from datetime import date, datetime`; `from algua.backtest.engine import run as run_backtest`; `from algua.contracts.types import DataProvider`):

```python
@dataclass
class FactorEvalResult:
    factor: str
    standalone: bool
    params: dict[str, Any]
    construction: str
    construction_params: dict[str, Any]
    horizon: int
    backtest: dict[str, Any]
    ic: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "standalone": self.standalone,
            "params": self.params,
            "construction": self.construction,
            "construction_params": self.construction_params,
            "horizon": self.horizon,
            "backtest": self.backtest,
            "ic": self.ic,
        }


def evaluate_factor(
    spec: FactorSpec,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    symbols: list[str],
    params: dict[str, Any],
    construction: str,
    construction_params: dict[str, Any],
    horizon: int = 1,
    execution: ExecutionContract | None = None,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    universe_name: str | None = None,
    universe_snapshots: list[dict[str, str]] | None = None,
) -> FactorEvalResult:
    """Evaluate one standalone factor: a real PIT backtest (existing engine) + rank IC/IR over the
    same fetched bars. IC is computed over the declared `symbols` (static); the backtest block
    honors `--universe` PIT membership via the engine. Touches no registry/holdout/gate."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    strategy = build_factor_strategy(
        spec, symbols=symbols, params=params, construction=construction,
        construction_params=construction_params, execution=execution,
    )
    bt = run_backtest(
        strategy, provider, start, end,
        universe_by_date=universe_by_date, universe_name=universe_name,
        universe_snapshots=universe_snapshots,
    )
    bars = provider.get_bars(sorted(set(symbols)), start, end, "1d")
    panel = score_panel(strategy, bars)
    fwd = forward_returns(
        _adj_grid(bars), lag=strategy.execution.decision_lag_bars, horizon=horizon
    )
    ic = factor_ic(panel, fwd)
    return FactorEvalResult(
        factor=spec.name,
        standalone=spec.standalone,
        params=params,
        construction=construction,
        construction_params=construction_params,
        horizon=horizon,
        backtest=bt.to_dict(),
        ic=ic,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_factor_eval_run.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/factor_eval.py tests/test_factor_eval_run.py
git commit -m "feat(140): evaluate_factor orchestration (backtest + IC) + FactorEvalResult"
```

---

## Task 9: CLI `algua factor eval`

**Files:**
- Modify: `algua/cli/factor_cmd.py`
- Test: `tests/test_cli_factor.py` (existing — append; reuse its `runner`, `_json`, and autouse `_tmp_db` fixture)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_factor.py` (it already defines `runner = CliRunner()`, the `_json(result)` helper that asserts exit 0 and parses `result.stdout`, and an autouse `_tmp_db` fixture):

```python
def test_factor_eval_emits_backtest_and_ic():
    payload = _json(runner.invoke(app, [
        "factor", "eval", "xs_trailing_return",
        "--demo", "--start", "2023-01-01", "--end", "2023-06-30",
        "--symbols", "AAA,BBB,CCC",
        "--construction", "top_k_equal_weight", "--construction-param", "top_k=1",
        "--param", "lookback=10",
    ]))
    assert payload["ok"] is True
    assert payload["factor"] == "xs_trailing_return"
    assert payload["ic"]["method"] == "spearman"
    assert payload["ic"]["fdr_corrected"] is False
    assert "metrics" in payload["backtest"]


def test_factor_eval_requires_construction():
    result = runner.invoke(app, [
        "factor", "eval", "xs_trailing_return", "--demo",
        "--symbols", "AAA,BBB", "--param", "lookback=10",
    ])
    assert result.exit_code != 0  # typer: missing required --construction


def test_factor_eval_rejects_non_standalone_factor():
    result = runner.invoke(app, [
        "factor", "eval", "momentum", "--demo", "--symbols", "AAA,BBB",
        "--construction", "equal_weight_positive",
    ])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "standalone" in payload["error"].lower()
```

(`SyntheticProvider` synthesizes deterministic bars for any symbols, so `AAA/BBB/CCC` are valid.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_factor_cmd_eval.py -q`
Expected: FAIL — no `eval` command.

- [ ] **Step 3: Implement**

In `algua/cli/factor_cmd.py`, add imports and the command. Add to the existing import block:

```python
from datetime import datetime

from algua.backtest.factor_eval import evaluate_factor
from algua.backtest.sweep import _coerce
from algua.cli._common import ok, registry_conn, resolve_universe_inputs, select_provider, utc
from algua.cli.app import app, emit
```

(Keep the existing `get_factor` import.) Add a small key=value parser and the command:

```python
def _parse_kv(items: list[str], flag: str) -> dict[str, object]:
    """Parse repeatable `KEY=value` flags into a single dict (values coerced int->float->str).
    Unlike the sweep grid, each key takes ONE value (a factor eval is a single point, not a grid)."""
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"malformed {flag} {item!r}: expected KEY=value")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key or raw.strip() == "":
            raise ValueError(f"malformed {flag} {item!r}: empty key or value")
        if key in out:
            raise ValueError(f"duplicate {flag} key {key!r}")
        out[key] = _coerce(raw.strip())
    return out


@factor_app.command("eval")
@json_errors(ValueError, LookupError)
def eval_factor(
    name: str = typer.Argument(..., help="standalone factor name"),
    symbols: str = typer.Option(..., "--symbols", help="comma-separated evaluation universe"),
    construction: str = typer.Option(..., "--construction", help="construction policy id"),
    construction_param: list[str] = typer.Option(
        [], "--construction-param", help="KEY=value for the construction policy (repeatable)"),
    param: list[str] = typer.Option(
        [], "--param", help="KEY=value factor param, e.g. lookback=60 (repeatable)"),
    horizon: int = typer.Option(1, "--horizon", help="forward-return horizon in bars"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="evaluate an ingested bars snapshot id"),
    universe: str = typer.Option(
        None, "--universe", help="point-in-time universe name (PIT membership for the backtest)"),
) -> None:
    """Evaluate ONE standalone factor on its own: a PIT backtest (via a 1-factor adapter) plus a
    construction-free rank IC/IR block. Ephemeral — writes nothing to the registry; the IC t-stat
    is NOT multiple-testing corrected (#140 slice E)."""
    spec = get_factor(name)
    provider = select_provider(demo, snapshot)
    start_dt, end_dt = utc(start), utc(end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise ValueError("--symbols must list at least one symbol")
    result = evaluate_factor(
        spec, provider, start_dt, end_dt,
        symbols=syms,
        params=_parse_kv(param, "--param"),
        construction=construction,
        construction_params=_parse_kv(construction_param, "--construction-param"),
        horizon=horizon,
        universe_by_date=universe_by_date,
        universe_name=universe,
        universe_snapshots=universe_prov,
    )
    emit(ok(result.to_dict()))
```

Note: `_coerce` is a module-private helper in `algua.backtest.sweep`; importing it from the CLI keeps value coercion identical to sweep. If a reviewer objects to importing a private name across modules, promote `_coerce` to a public `coerce_value` in `sweep.py` in this task and update the sweep call site + this import.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_factor_cmd_eval.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/factor_cmd.py tests/test_factor_cmd_eval.py
git commit -m "feat(140): algua factor eval CLI (1-factor backtest + IC/IR JSON)"
```

---

## Task 10: docs caveat + full gate

**Files:**
- Modify: the `interpret-results` skill markdown (locate via `grep -rl "promotion-gate\|holdout" ~/.claude` or the plugin skills dir; it documents how to read backtest/gate JSON).

- [ ] **Step 1: Add the IC caveat**

Add a short subsection to the `interpret-results` skill explaining the `factor eval` IC block: what `mean_ic`/`ir`/`t_stat`/`hit_rate`/`n_obs` mean, that Spearman rank IC ~0 means no monotone cross-sectional predictiveness, and the explicit caveat:

```markdown
### Reading `algua factor eval` IC (issue #140 slice B)

`ic.mean_ic` is the average per-bar cross-sectional rank (Spearman) correlation between the
factor's score and the forward return; `ir = mean_ic / ic_std`; `t_stat = ir * sqrt(n_obs)`.
Rules of thumb: |mean_ic| ~0 = no monotone predictiveness; a stable small-but-positive IC with
many observations is more credible than a large IC over few bars (check `n_obs`).

**Caveat — `fdr_corrected: false`.** The t-stat is RAW: it is not corrected for the multiple-
testing surface created by evaluating many factors. A factor passing on its own is a prior to
investigate, NOT evidence of edge. Funnel-wide FDR accounting for factor hypotheses is #140
slice E. A factor never goes live; only a strategy that composes it, through the normal gates.
```

- [ ] **Step 2: Commit the docs**

```bash
git add -A && git commit -m "docs(140): interpret-results — factor eval IC block + FDR caveat"
```

- [ ] **Step 3: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. In particular `lint-imports` must still pass — `features/alphas.py` imports only `features` + `contracts` (pure); `backtest/factor_eval.py` imports `engine`/`features`/`portfolio`/`strategies`/`contracts` (all permitted for `backtest`); the CLI imports `backtest` (permitted).

- [ ] **Step 4: Fix any gate failures, then final commit if needed**

```bash
git add -A && git commit -m "chore(140): satisfy quality gate"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** §1 evaluable contract → Task 1–2; §2 seed alpha + rewire → Task 3–4; §3 adapter+IC → Task 5–8; §4 CLI → Task 9; §5 non-goals are enforced by *omission* (no walk-forward/sweep/registry/persistence wired) + the `fdr_corrected:false`/`__factor__:` guards; §6 testing → every task is TDD.
- **Type/name consistency:** `build_factor_strategy(spec, *, symbols, params, construction, construction_params, execution=None)`, `score_panel(strategy, bars)`, `forward_returns(adj, *, lag, horizon)`, `factor_ic(score_panel, forward_returns, *, min_cross_section)`, `evaluate_factor(...) -> FactorEvalResult` with `.to_dict()` — used identically across Tasks 5–9.
- **Known slice-B limitation (documented, not a placeholder):** the IC block is computed over the static declared `--symbols`; PIT as-of membership masking of the IC panel is deferred (the *backtest* block still honors `--universe`). This is advisory metric scope, acceptable for slice B.
- **`SyntheticProvider` (confirmed):** synthesizes deterministic bars for *any* symbol list (`algua/backtest/_sample.py`), so the Task 8/9 tests can use `AAA/BBB/CCC` freely.
```
