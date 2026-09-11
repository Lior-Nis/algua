# Portfolio Overlays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ordered, stateless, tighten-only `overlays` stage to the strategy contract, applied inside `LoadedStrategy.construct()` after construction and before the capacity cap, with two first policies (`regime_gate`, `trailing_stop`) and the regime features they need.

**Architecture:** A new pure module `algua/portfolio/overlays.py` holds the `OverlaySpec` model, the `OverlayFn` interface, the static policy registry, per-policy validation, and the central tighten-only invariant check. `algua/features/regime.py` holds the pure universe-derived features. `strategies/base.py` grows one config field, one `LoadedStrategy` field, three lines in `construct()`, and an identity fold; the loader resolves and validates specs; sweeps get an `overlay.<i>.<key>` namespace; approvals root the code closure from the overlays module; CODEOWNERS protects it.

**Tech Stack:** Python 3.12, pandas, numpy, pydantic v2, pytest, ruff, mypy, import-linter, uv.

**Spec:** `docs/superpowers/specs/2026-09-10-portfolio-overlays-design.md`

## Global Constraints

- Every quality gate must pass before each commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. Run the targeted test file during the task, the full gate at the commit step.
- `algua/strategies/base.py` is ratchet-pinned at 380 lines and `algua/backtest/sweep.py` at 461 (`tests/test_module_size_ratchet.py`). Tasks 1 and 2 shrink them first; later tasks must keep them at or below those pins. No new module may reach 300 lines.
- `algua.portfolio` may import only `algua.contracts` and `algua.features` (import-linter contract "portfolio construction layer is pure"). `algua.features` may import only `algua.contracts`.
- Never `git add -A`. Stage the exact files each task names. The working tree carries unrelated dirty files (`approvers/allowed_signers`, `kb/`) that must not be swept in.
- Work on branch `feat/portfolio-overlays` (already created, holds the spec commit).
- Overlays never scale up, flip sign, add a symbol, or emit non-finite values. Freed weight is cash; nothing renormalises.
- All price math uses `adj_close`, never raw `close`.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01HupQYtfZet4Cefj7xyi7ZG
  ```

---

### Task 1: Carve the tradable-assert trio out of `strategies/base.py` (pure move)

**Files:**
- Create: `algua/strategies/tradable.py`
- Modify: `algua/strategies/base.py:309-338` (delete the three functions)
- Modify: `algua/strategies/loader.py` (import from the new module)
- Modify: `tests/test_fundamentals_guards.py:9`, `tests/test_news_guards.py:6-10`, `tests/test_strategies_base_news.py:6-11` (import path)

**Interfaces:**
- Produces: `algua.strategies.tradable.assert_tradable_without_fundamentals(strategy: LoadedStrategy) -> None`, `assert_tradable_without_news`, `assert_tradable_without_model`. Same bodies as today.

- [ ] **Step 1: Create the new module with the three functions moved verbatim**

```python
"""Paper/live tradability guards: a strategy that declares a PIT sidecar lane the lanes cannot
serve yet is refused at every trading load point (carved from strategies/base.py, overlays PR)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algua.strategies.base import LoadedStrategy


def assert_tradable_without_fundamentals(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_fundamentals strategy must NOT run paper/live yet — the as-of
    fundamentals lane is wired only into the backtest engine (issue #132). Called at every trading
    load point so no actor (agent promote OR human raw transition) can run it blind."""
    if strategy.config.needs_fundamentals:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_fundamentals; paper/live fundamentals "
            f"wiring is not built yet (#132 follow-up) — refusing to trade it blind"
        )


def assert_tradable_without_news(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_news strategy must NOT run paper/live yet — the as-of news lane is
    wired only into the backtest engine (issue #132). Called at every trading load point."""
    if strategy.config.needs_news:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_news; paper/live news wiring is not built "
            f"yet (#132 follow-up) — refusing to trade it blind"
        )


def assert_tradable_without_model(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_model strategy must NOT run paper/live yet — the model lane is wired
    only into the `backtest run` engine (issue #376). Called at every trading load point."""
    if strategy.config.needs_model:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_model; paper/live model wiring is not "
            f"built yet (#376 follow-up) — refusing to trade it blind"
        )
```

- [ ] **Step 2: Delete lines 309-338 of `algua/strategies/base.py`** (the three `assert_tradable_without_*` functions and the blank lines between them; `config_hash` must remain and now directly follow `target_weights`'s class body with two blank lines).

- [ ] **Step 3: Repoint the loader.** In `algua/strategies/loader.py`, find the `from algua.strategies.base import (...)` block, remove the three `assert_tradable_without_*` names from it, and add:

```python
from algua.strategies.tradable import (
    assert_tradable_without_fundamentals,
    assert_tradable_without_model,
    assert_tradable_without_news,
)
```

(`tests/test_strategy_loader.py:173` monkeypatches `loader.assert_tradable_without_fundamentals`; a from-import keeps that attribute on the loader module, so that test needs no change.)

- [ ] **Step 4: Repoint the three tests.**

`tests/test_fundamentals_guards.py` line 9:
```python
from algua.strategies.tradable import assert_tradable_without_fundamentals
```

`tests/test_news_guards.py` lines 6-10 become:
```python
from algua.strategies.base import (
    LoadedStrategy,
    StrategyConfig,
)
from algua.strategies.tradable import assert_tradable_without_news
```

`tests/test_strategies_base_news.py` lines 6-11 become:
```python
from algua.strategies.base import (
    LoadedStrategy,
    StrategyConfig,
    config_hash,
)
from algua.strategies.tradable import assert_tradable_without_news
```

- [ ] **Step 5: Run the affected tests and the ratchet**

Run: `uv run pytest tests/test_fundamentals_guards.py tests/test_news_guards.py tests/test_strategies_base_news.py tests/test_strategy_loader.py tests/test_module_size_ratchet.py -q`
Expected: all pass; `wc -l algua/strategies/base.py` prints 349 or fewer.

- [ ] **Step 6: Full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/strategies/tradable.py algua/strategies/base.py algua/strategies/loader.py tests/test_fundamentals_guards.py tests/test_news_guards.py tests/test_strategies_base_news.py
git commit -m "refactor: carve tradable-assert guards out of strategies/base.py (pure move)

Funds the overlays stage: base.py sits exactly on its 380-line ratchet pin."
```

---

### Task 2: Carve the grid parsers out of `backtest/sweep.py` (pure move)

**Files:**
- Create: `algua/backtest/sweep_grid.py`
- Modify: `algua/backtest/sweep.py:83-132` (delete `_coerce`, `_coerce_values`, `parse_grid`; keep `validate_sweep_grid`)
- Modify: `algua/evaluation/sweep_run.py:15`, `algua/registry/mergeback_intake.py:55`, `tests/test_sweep_parse.py:3`, `tests/test_sweep_coerce.py:2`

**Interfaces:**
- Produces: `algua.backtest.sweep_grid.parse_grid(params: list[str]) -> dict[str, list[Any]]`, `_coerce(value: str) -> Any`, `_coerce_values(values: list[Any]) -> list[Any]`. Same bodies as today.

- [ ] **Step 1: Create `algua/backtest/sweep_grid.py`** containing the module docstring below and the three functions moved verbatim from `sweep.py` lines 83-132:

```python
"""`--param KEY=v1,v2` grid parsing for `backtest sweep` (carved from backtest/sweep.py, overlays
PR). Pure string -> grid; the strategy-dependent half (`validate_sweep_grid`) stays in sweep.py."""
from __future__ import annotations

import math
from typing import Any
```

then `_coerce`, `_coerce_values`, `parse_grid` exactly as they are in `sweep.py`.

- [ ] **Step 2: Delete those three functions from `sweep.py`** and, if `math` is now unused there, remove `import math`. Add `from algua.backtest.sweep_grid import parse_grid` **only if** anything left in `sweep.py` references `parse_grid` (check with `grep -n parse_grid algua/backtest/sweep.py`; `validate_sweep_grid` does not, `sweep()` does not).

- [ ] **Step 3: Repoint importers**

`algua/evaluation/sweep_run.py` line 15:
```python
from algua.backtest.sweep import sweep
from algua.backtest.sweep_grid import parse_grid
```

`algua/registry/mergeback_intake.py` line 55:
```python
from algua.backtest.sweep import _RANK_KEYS, validate_sweep_grid
from algua.backtest.sweep_grid import parse_grid
```

`tests/test_sweep_parse.py` line 3: `from algua.backtest.sweep_grid import parse_grid`
`tests/test_sweep_coerce.py` line 2: `from algua.backtest.sweep_grid import _coerce_values`

Then `grep -rn "sweep import.*parse_grid\|sweep import.*_coerce" algua tests` must return nothing.

- [ ] **Step 4: Run the affected tests**

Run: `uv run pytest tests/test_sweep_parse.py tests/test_sweep_coerce.py tests/test_sweep_override.py tests/test_cli_sweep.py tests/test_mergeback_queue.py tests/test_module_size_ratchet.py -q`
Expected: pass; `wc -l algua/backtest/sweep.py` prints 415 or fewer.

- [ ] **Step 5: Full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/backtest/sweep_grid.py algua/backtest/sweep.py algua/evaluation/sweep_run.py algua/registry/mergeback_intake.py tests/test_sweep_parse.py tests/test_sweep_coerce.py
git commit -m "refactor: carve --param grid parsing out of backtest/sweep.py (pure move)

Funds the overlay.<i>.<key> sweep namespace: sweep.py sits exactly on its 461-line pin."
```

---

### Task 3: Regime features (`algua/features/regime.py`)

**Files:**
- Create: `algua/features/regime.py`
- Test: `tests/test_features_regime.py`

**Interfaces:**
- Produces:
  - `equal_weight_index(view: pd.DataFrame) -> pd.Series` (index level by timestamp, base 1.0)
  - `rolling_drawdown(level: pd.Series, window: int) -> pd.Series`
  - `turbulence(view: pd.DataFrame, window: int, *, last: int | None = None) -> pd.Series`
  - `robust_zscore(x: pd.Series, window: int) -> pd.Series`
  - `wide_adj_close(view: pd.DataFrame) -> pd.DataFrame` (helper: timestamp × symbol pivot of `adj_close`, sorted)
- `view` is the long bar-schema frame: `timestamp` index (tz-aware UTC), columns include `symbol` and `adj_close`.

- [ ] **Step 1: Write the failing tests**

```python
"""Universe-derived regime features (overlays spec §Features). Pure, adj_close-only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algua.features.regime import (
    equal_weight_index,
    robust_zscore,
    rolling_drawdown,
    turbulence,
    wide_adj_close,
)


def _view(prices: dict[str, list[float]]) -> pd.DataFrame:
    """prices = {symbol: [adj_close per bar]} -> long bar-schema view, all symbols same length."""
    n = len(next(iter(prices.values())))
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rows = []
    for sym, path in prices.items():
        for t, px in zip(ts, path, strict=True):
            rows.append({"timestamp": t, "symbol": sym, "open": px, "high": px, "low": px,
                         "close": px * 2.0, "adj_close": px, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_wide_adj_close_pivots_and_sorts():
    view = _view({"B": [1.0, 2.0], "A": [3.0, 4.0]})
    wide = wide_adj_close(view.iloc[::-1])  # reversed input must come back sorted
    assert list(wide.columns) == ["A", "B"]
    assert wide.index.is_monotonic_increasing
    assert wide["A"].tolist() == [3.0, 4.0]


def test_equal_weight_index_is_mean_of_member_returns_base_one():
    view = _view({"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 99.0]})
    level = equal_weight_index(view)
    # bar 0 -> 1.0; bar 1 -> mean(+10%, -10%) = 0 -> 1.0; bar 2 -> mean(+10%, +10%) = +10% -> 1.1
    assert level.iloc[0] == pytest.approx(1.0)
    assert level.iloc[1] == pytest.approx(1.0)
    assert level.iloc[2] == pytest.approx(1.1)


def test_equal_weight_index_ignores_a_member_missing_a_bar():
    view = _view({"A": [100.0, 110.0], "B": [100.0, 120.0]})
    view = view[~((view["symbol"] == "B") & (view.index == view.index[1]))]  # drop B's 2nd bar
    level = equal_weight_index(view)
    assert level.iloc[1] == pytest.approx(1.10)  # only A contributes


def test_equal_weight_index_uses_adj_close_not_close():
    view = _view({"A": [100.0, 110.0]})
    assert equal_weight_index(view).iloc[1] == pytest.approx(1.10)  # close is 2x and unused


def test_rolling_drawdown_nan_until_window_then_from_rolling_high():
    level = pd.Series([1.0, 1.2, 1.1, 0.9, 1.0])
    dd = rolling_drawdown(level, window=3)
    assert np.isnan(dd.iloc[0]) and np.isnan(dd.iloc[1])
    assert dd.iloc[2] == pytest.approx(1.1 / 1.2 - 1.0)
    assert dd.iloc[3] == pytest.approx(0.9 / 1.2 - 1.0)
    assert dd.iloc[4] == pytest.approx(1.0 / 1.1 - 1.0)  # 1.2 has rolled out of the window


def test_robust_zscore_nan_until_full_and_nan_on_zero_mad():
    z = robust_zscore(pd.Series([1.0] * 10), window=5)
    assert z.isna().all()  # constant -> zero MAD -> NaN, never inf
    x = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 50.0])
    z = robust_zscore(x, window=5)
    assert z.iloc[:4].isna().all()
    assert z.iloc[-1] > 10.0  # the planted outlier


def test_turbulence_nan_until_window_and_spikes_on_planted_outlier():
    rng = np.random.default_rng(0)
    n, window = 40, 10
    rets = rng.normal(0.0, 0.01, size=(n, 3))
    rets[-1] = 0.10  # every symbol jumps +10 sigma on the last bar
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    view = _view({s: prices[:, i].tolist() for i, s in enumerate(["A", "B", "C"])})
    t = turbulence(view, window)
    assert t.iloc[: window + 1].isna().all()  # first row has no return + `window` prior returns
    assert t.iloc[window + 1 :].notna().all()
    assert t.iloc[-1] > 10.0 * t.iloc[window + 1 : -1].median()


def test_turbulence_last_computes_only_the_tail_identically():
    rng = np.random.default_rng(1)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(30, 2)), axis=0)
    view = _view({"A": prices[:, 0].tolist(), "B": prices[:, 1].tolist()})
    full = turbulence(view, 5)
    tail = turbulence(view, 5, last=4)
    assert tail.iloc[:-4].isna().all()
    pd.testing.assert_series_equal(tail.iloc[-4:], full.iloc[-4:])


def test_turbulence_excludes_symbols_without_full_window_that_bar():
    rng = np.random.default_rng(2)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(20, 2)), axis=0)
    view = _view({"A": prices[:, 0].tolist(), "B": prices[:, 1].tolist()})
    view_a_only = view[view["symbol"] == "A"]
    # Drop B entirely before bar 15: at bar 19, B has only 4 prior returns -> excluded -> equals A-only.
    partial = pd.concat([view_a_only, view[(view["symbol"] == "B") & (view.index >= view.index[15])]])
    partial = partial.sort_index()
    assert turbulence(partial, 5).iloc[-1] == pytest.approx(turbulence(view_a_only, 5).iloc[-1])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_features_regime.py -q`
Expected: ImportError on `algua.features.regime`.

- [ ] **Step 3: Implement `algua/features/regime.py`**

```python
"""Universe-derived market-state features for the overlay stage (overlays spec §Features).

Pure (no I/O, no clock, no global state), `adj_close`-only. `view` is the long bar-schema frame the
signal saw (tz-aware `timestamp` index; `symbol`, `adj_close` columns). Everything here is
computable from the strategy's own universe, so a regime overlay needs no reference symbols.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from algua.features.catalogue import FactorKind, factor

# Consistency constant: 1 / Phi^{-1}(3/4). Scales a MAD to a normal-distribution sigma.
_MAD_TO_SIGMA = 1.4826


def wide_adj_close(view: pd.DataFrame) -> pd.DataFrame:
    """timestamp x symbol matrix of `adj_close`, sorted by timestamp (never raw `close`, #521)."""
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return wide.sort_index()


@factor(
    summary="Equal-weight index level (base 1.0) of the universe from adj_close simple returns.",
    kind=FactorKind.OTHER,
    tags=["regime", "universe", "index"],
)
def equal_weight_index(view: pd.DataFrame) -> pd.Series:
    """Each bar's index return is the mean of the member simple returns available that bar (a
    member missing a bar contributes nothing that bar); the level compounds from 1.0. A bar where
    no member has a return (the first bar) has index return 0."""
    rets = wide_adj_close(view).pct_change(fill_method=None)
    idx_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + idx_ret).cumprod()


@factor(
    summary="Drawdown of a level series from its trailing `window`-bar high.",
    kind=FactorKind.VOLATILITY,
    tags=["regime", "drawdown"],
)
def rolling_drawdown(level: pd.Series, window: int) -> pd.Series:
    """`level / rolling_max(window) - 1`; NaN until `window` observations exist."""
    return level / level.rolling(window, min_periods=window).max() - 1.0


@factor(
    summary="Robust z-score: (x - rolling median) / (1.4826 * rolling MAD).",
    kind=FactorKind.OTHER,
    tags=["normalization", "robust"],
)
def robust_zscore(x: pd.Series, window: int) -> pd.Series:
    """For each bar t: (x_t - med) / (1.4826 * MAD) over the window ending at t, where med is the
    window median and MAD the median of |x_i - med| over that SAME window (the exact windowed MAD).
    NaN until `window` observations exist, NaN for any window containing NaN, and NaN (never inf)
    when the MAD is zero (a constant window)."""
    v = x.to_numpy(dtype="float64")
    out = np.full(len(v), np.nan)
    if len(v) >= window:
        win = np.lib.stride_tricks.sliding_window_view(v, window)  # (n - window + 1, window)
        med = np.median(win, axis=1)
        scale = _MAD_TO_SIGMA * np.median(np.abs(win - med[:, None]), axis=1)
        ok = np.isfinite(scale) & (scale > 0.0)
        z = np.full(len(win), np.nan)
        z[ok] = (win[ok, -1] - med[ok]) / scale[ok]
        out[window - 1 :] = z
    return pd.Series(out, index=x.index, dtype="float64")


@factor(
    summary="Cross-sectional turbulence: Mahalanobis distance of a bar's return vector vs the trailing covariance.",
    kind=FactorKind.VOLATILITY,
    tags=["regime", "turbulence", "cross-sectional"],
)
def turbulence(view: pd.DataFrame, window: int, *, last: int | None = None) -> pd.Series:
    """For bar t: d' * pinv(cov) * d where d = r_t - mean(r over the `window` bars BEFORE t) and cov
    is that trailing window's covariance (pseudo-inverse, so a singular covariance is finite). A
    symbol enters bar t's vector only if it has a return on t AND on every one of the `window`
    prior bars. NaN until `window` prior returns exist (i.e. the first `window + 1` bars) or when no
    symbol qualifies. `last=k` computes only the final k bars (the rest NaN) — an overlay evaluated
    per decision bar only needs the tail, and this keeps that O(window) instead of O(history)."""
    rets = wide_adj_close(view).pct_change(fill_method=None)
    n = len(rets)
    out = pd.Series(np.nan, index=rets.index, dtype="float64")
    start = window + 1
    if last is not None:
        start = max(start, n - last)
    values = rets.to_numpy(dtype="float64")
    for i in range(start, n):
        hist = values[i - window : i]
        ok = np.isfinite(hist).all(axis=0) & np.isfinite(values[i])
        if not ok.any():
            continue
        h = hist[:, ok]
        d = values[i, ok] - h.mean(axis=0)
        cov = np.atleast_2d(np.cov(h, rowvar=False))
        out.iloc[i] = float(d @ np.linalg.pinv(cov) @ d)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_features_regime.py tests/test_alphas.py -q`
Expected: all pass (`test_alphas` exercises `load_all_factors`, which now imports `regime.py`).

- [ ] **Step 5: Full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/features/regime.py tests/test_features_regime.py
git commit -m "feat(features): universe-derived regime features — EW index, drawdown, turbulence, robust z"
```

---

### Task 4: The overlay seam and `trailing_stop` (`algua/portfolio/overlays.py`)

**Files:**
- Create: `algua/portfolio/overlays.py`
- Test: `tests/test_portfolio_overlays.py`

**Interfaces:**
- Produces (all in `algua.portfolio.overlays`):
  - `OverlayFn = Callable[[pd.Series, pd.DataFrame, dict[str, Any]], pd.Series]`
  - `class OverlayError(ValueError)`
  - `class OverlaySpec(BaseModel)`: `policy: str`, `params: dict[str, Any] = {}`
  - `OVERLAY_POLICIES: Mapping[str, _Overlay]` (read-only)
  - `get_overlay_policy(policy_id: str) -> OverlayFn`
  - `validate_overlay_params(policy_id: str, params: dict[str, Any]) -> None`
  - `overlay_lookback(spec: OverlaySpec) -> int`
  - `resolve_overlays(specs: Sequence[OverlaySpec], *, feature_lookback: int | None) -> tuple[OverlayFn, ...]`
  - `apply_overlays(weights: pd.Series, view: pd.DataFrame, specs: Sequence[OverlaySpec], fns: Sequence[OverlayFn]) -> pd.Series`
  - `trailing_stop(weights, view, params) -> pd.Series` with params `lookback: int>0`, `stop_pct: float in (0,1)`, `cooldown_bars: int>=0`
- The registry dict `_OVERLAYS` gains `"regime_gate"` in Task 6; leave the slot noted in a comment.

- [ ] **Step 1: Write the failing tests**

```python
"""The overlay seam: tighten-only invariants, validation, resolution, and trailing_stop."""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from algua.portfolio.overlays import (
    OVERLAY_POLICIES,
    OverlayError,
    OverlaySpec,
    apply_overlays,
    get_overlay_policy,
    overlay_lookback,
    resolve_overlays,
    trailing_stop,
    validate_overlay_params,
)


def _view(prices: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(prices.values())))
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rows = []
    for sym, path in prices.items():
        for t, px in zip(ts, path, strict=True):
            rows.append({"timestamp": t, "symbol": sym, "open": px, "high": px, "low": px,
                         "close": px, "adj_close": px, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


_W = pd.Series({"A": 0.5, "B": 0.5})
_V = _view({"A": [1.0, 1.0], "B": [1.0, 1.0]})


# --- invariants -------------------------------------------------------------------------------

def _spec(policy: str = "trailing_stop") -> OverlaySpec:
    return OverlaySpec(policy=policy, params={})


def test_apply_overlays_rejects_added_symbol():
    def adds(w: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        return pd.concat([w, pd.Series({"C": 0.1})])
    with pytest.raises(OverlayError, match=r"overlay\[0\] 'trailing_stop' added symbol"):
        apply_overlays(_W, _V, [_spec()], [adds])


def test_apply_overlays_rejects_scale_up():
    def up(w, view, params):
        return w * 1.5
    with pytest.raises(OverlayError, match=r"overlay\[0\].*increased"):
        apply_overlays(_W, _V, [_spec()], [up])


def test_apply_overlays_rejects_sign_flip():
    def flip(w, view, params):
        return -w
    with pytest.raises(OverlayError, match=r"overlay\[0\].*flipped"):
        apply_overlays(_W, _V, [_spec()], [flip])


def test_apply_overlays_rejects_non_finite():
    def nan(w, view, params):
        out = w.copy()
        out["A"] = float("nan")
        return out
    with pytest.raises(OverlayError, match=r"overlay\[0\].*non-finite"):
        apply_overlays(_W, _V, [_spec()], [nan])


def test_apply_overlays_names_the_failing_index_in_a_chain():
    ident = lambda w, view, params: w  # noqa: E731
    up = lambda w, view, params: w * 2.0  # noqa: E731
    with pytest.raises(OverlayError, match=r"overlay\[1\]"):
        apply_overlays(_W, _V, [_spec(), _spec()], [ident, up])


def test_apply_overlays_accepts_zeroing_and_dropping():
    def zero_a_drop_b(w, view, params):
        return pd.Series({"A": 0.0})
    out = apply_overlays(_W, _V, [_spec()], [zero_a_drop_b])
    assert out.to_dict() == {"A": 0.0}


def test_apply_overlays_empty_weights_short_circuits():
    def boom(w, view, params):
        raise AssertionError("must not be called on empty weights")
    empty = pd.Series(dtype="float64")
    assert apply_overlays(empty, _V, [_spec()], [boom]).empty


def test_apply_overlays_requires_one_fn_per_spec():
    with pytest.raises(OverlayError, match="one resolved fn per spec"):
        apply_overlays(_W, _V, [_spec()], [])


# --- registry + validation ---------------------------------------------------------------------

def test_registry_is_read_only_and_lists_trailing_stop():
    assert "trailing_stop" in OVERLAY_POLICIES
    with pytest.raises(TypeError):
        OVERLAY_POLICIES["x"] = None  # type: ignore[index]


def test_get_overlay_policy_unknown_id():
    with pytest.raises(OverlayError, match="unknown overlay policy 'nope'"):
        get_overlay_policy("nope")


@pytest.mark.parametrize(
    "params, msg",
    [
        ({"lookback": 20, "stop_pct": 0.1}, "missing"),
        ({"lookback": 20, "stop_pct": 0.1, "cooldown_bars": 0, "x": 1}, "unknown"),
        ({"lookback": 0, "stop_pct": 0.1, "cooldown_bars": 0}, "lookback"),
        ({"lookback": True, "stop_pct": 0.1, "cooldown_bars": 0}, "lookback"),
        ({"lookback": 20, "stop_pct": 1.0, "cooldown_bars": 0}, "stop_pct"),
        ({"lookback": 20, "stop_pct": 0.0, "cooldown_bars": 0}, "stop_pct"),
        ({"lookback": 20, "stop_pct": float("nan"), "cooldown_bars": 0}, "non-finite"),
        ({"lookback": 20, "stop_pct": 0.1, "cooldown_bars": -1}, "cooldown_bars"),
    ],
)
def test_validate_trailing_stop_params_fail_closed(params, msg):
    with pytest.raises(OverlayError, match=msg):
        validate_overlay_params("trailing_stop", params)


def test_validate_trailing_stop_params_ok():
    validate_overlay_params("trailing_stop", {"lookback": 20, "stop_pct": 0.1, "cooldown_bars": 3})


def test_overlay_lookback_trailing_stop():
    spec = OverlaySpec(policy="trailing_stop", params={"lookback": 20, "stop_pct": 0.1, "cooldown_bars": 3})
    assert overlay_lookback(spec) == 23


def test_resolve_overlays_binds_fns_and_checks_feature_lookback():
    spec = OverlaySpec(policy="trailing_stop", params={"lookback": 20, "stop_pct": 0.1, "cooldown_bars": 3})
    fns = resolve_overlays([spec], feature_lookback=None)
    assert fns == (trailing_stop,)
    assert resolve_overlays([spec], feature_lookback=23) == (trailing_stop,)
    with pytest.raises(OverlayError, match="feature_lookback 22 is smaller than the longest overlay window 23"):
        resolve_overlays([spec], feature_lookback=22)
    assert resolve_overlays([], feature_lookback=0) == ()


def test_resolve_overlays_validates_params():
    with pytest.raises(OverlayError, match="overlay\\[0\\] 'trailing_stop': missing"):
        resolve_overlays([OverlaySpec(policy="trailing_stop", params={})], feature_lookback=None)


# --- trailing_stop -----------------------------------------------------------------------------

_TS = {"lookback": 5, "stop_pct": 0.10, "cooldown_bars": 2}


def test_trailing_stop_zeroes_a_name_below_its_rolling_high():
    # A: peak 100 then 85 (-15% off the 5-bar high) -> stopped. B flat -> kept.
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 85.0], "B": [50.0] * 5})
    out = trailing_stop(pd.Series({"A": 0.5, "B": 0.5}), view, _TS)
    assert out.to_dict() == {"A": 0.0, "B": 0.5}


def test_trailing_stop_keeps_a_name_within_tolerance():
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 92.0]})  # -8% off the high < 10%
    out = trailing_stop(pd.Series({"A": 1.0}), view, _TS)
    assert out["A"] == 1.0


def test_trailing_stop_cooldown_keeps_it_out_after_recovery():
    # Breach at bar 4 (85 vs high 100), recovers to 99 by bar 6. cooldown_bars=2 -> bars 5,6 still out.
    path = [90.0, 100.0, 98.0, 95.0, 85.0, 99.0, 99.0]
    for end, expect in ((5, 0.0), (6, 0.0), (7, 0.0)):
        view = _view({"A": path[:end]})
        assert trailing_stop(pd.Series({"A": 1.0}), view, _TS)["A"] == expect
    # One more bar and the breach (bar 4) is outside the last cooldown_bars+1 bars; the 5-bar high no
    # longer holds 100 either -> back in.
    view = _view({"A": path + [99.0]})
    assert trailing_stop(pd.Series({"A": 1.0}), view, _TS)["A"] == 1.0


def test_trailing_stop_cooldown_zero_is_only_the_current_bar():
    path = [90.0, 100.0, 98.0, 95.0, 85.0, 99.0]
    params = {**_TS, "cooldown_bars": 0}
    # bar 5: price 99 vs 5-bar high 100 -> within tolerance -> kept, even though bar 4 breached.
    assert trailing_stop(pd.Series({"A": 1.0}), _view({"A": path}), params)["A"] == 1.0


def test_trailing_stop_short_history_uses_available_bars():
    view = _view({"A": [100.0, 85.0]})  # only 2 bars, lookback 5 -> high = 100 -> stopped
    assert trailing_stop(pd.Series({"A": 1.0}), view, _TS)["A"] == 0.0


def test_trailing_stop_passes_through_a_symbol_absent_from_view():
    view = _view({"A": [100.0] * 5})
    out = trailing_stop(pd.Series({"A": 0.5, "Z": 0.5}), view, _TS)
    assert out.to_dict() == {"A": 0.5, "Z": 0.5}


def test_trailing_stop_preserves_short_sign():
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 85.0]})
    out = trailing_stop(pd.Series({"A": -0.5}), view, _TS)
    assert out["A"] == 0.0  # a short is stopped the same way (weight -> 0, never flipped)


def test_trailing_stop_through_apply_overlays_satisfies_invariants():
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 85.0], "B": [50.0] * 5})
    spec = OverlaySpec(policy="trailing_stop", params=_TS)
    out = apply_overlays(pd.Series({"A": 0.5, "B": 0.5}), view, [spec], resolve_overlays([spec], feature_lookback=None))
    assert out.to_dict() == {"A": 0.0, "B": 0.5}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_portfolio_overlays.py -q`
Expected: ImportError on `algua.portfolio.overlays`.

- [ ] **Step 3: Implement `algua/portfolio/overlays.py`**

```python
"""Portfolio overlays: an ordered, stateless, TIGHTEN-ONLY stage applied to the construction
output before the capacity cap (spec: docs/superpowers/specs/2026-09-10-portfolio-overlays-design.md).

An overlay maps (weights, view, params) -> weights, reading only the PIT `view` the signal saw. It
may zero, drop, or scale DOWN a weight; it may never add a symbol, scale up, flip a side, or emit
a non-finite value — `apply_overlays` enforces that after every policy, so a vector that passed
the gross/per-symbol rails inside construction still passes after any chain. Freed weight is cash
(no renormalisation), the same cap-and-hold-cash rule as `apply_capacity_cap`.

Identity rests on this module's STATIC source (approvals hash the module); there is no dynamic
registration. Policies are pure: no I/O, no clock, no global state.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from algua.features.regime import wide_adj_close

OverlayFn = Callable[[pd.Series, pd.DataFrame, dict[str, Any]], pd.Series]

# Float slack for |out| <= |in|: a multiply by a factor <= 1 cannot exceed the input, but a policy
# that recomputes a weight through a different arithmetic path may differ by an ulp.
_TOL = 1e-12


class OverlayError(ValueError):
    """An invalid overlay policy id, params, or output. Subclasses ValueError so the CLI's json
    error contract still renders it."""


class OverlaySpec(BaseModel):
    """One declared overlay: a policy id + its params. Validated per-policy at load."""

    model_config = {"frozen": True}
    policy: str
    params: dict[str, Any] = {}


# --- invariants ---------------------------------------------------------------------------------


def _checked(index: int, policy: str, before: pd.Series, after: object) -> pd.Series:
    tag = f"overlay[{index}] {policy!r}"
    if not isinstance(after, pd.Series):
        raise OverlayError(f"{tag} returned {type(after).__name__}, not a pd.Series")
    if after.index.has_duplicates:
        raise OverlayError(f"{tag} returned duplicate symbol(s)")
    extra = after.index.difference(before.index)
    if len(extra):
        raise OverlayError(f"{tag} added symbol(s) {sorted(map(str, extra))}")
    try:
        a = after.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise OverlayError(f"{tag} returned non-numeric weights: {exc}") from None
    if not np.isfinite(a).all():
        raise OverlayError(f"{tag} returned non-finite weight(s)")
    b = before.reindex(after.index).to_numpy(dtype="float64")
    if np.any(np.abs(a) > np.abs(b) + _TOL):
        raise OverlayError(f"{tag} increased |weight| — overlays are tighten-only")
    if np.any((a != 0.0) & (np.sign(a) != np.sign(b))):
        raise OverlayError(f"{tag} flipped a weight's sign — overlays are tighten-only")
    return pd.Series(a, index=after.index, dtype="float64")


def apply_overlays(
    weights: pd.Series,
    view: pd.DataFrame,
    specs: Sequence[OverlaySpec],
    fns: Sequence[OverlayFn],
) -> pd.Series:
    """Run the declared chain in order, enforcing the tighten-only invariants after each policy.
    Empty weights short-circuit (nothing to tighten). `fns` must be the resolved callables for
    `specs`, one per spec, in order (see `resolve_overlays`)."""
    if len(specs) != len(fns):
        raise OverlayError(
            f"apply_overlays needs one resolved fn per spec; got {len(specs)} spec(s) and "
            f"{len(fns)} fn(s)"
        )
    if len(weights) == 0:
        return weights
    for i, (spec, fn) in enumerate(zip(specs, fns, strict=True)):
        weights = _checked(i, spec.policy, weights, fn(weights, view, spec.params))
    return weights


# --- param validation helpers -------------------------------------------------------------------


def _exact_keys(params: dict[str, Any], required: set[str]) -> None:
    missing = required - set(params)
    if missing:
        raise OverlayError(f"missing param(s): {sorted(missing)}")
    unknown = set(params) - required
    if unknown:
        raise OverlayError(f"unknown param(s): {sorted(unknown)}")


def _positive_int(params: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    v = params[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise OverlayError(f"{key} must be an int >= {minimum}, got {v!r}")
    return v


def _float_in(
    params: dict[str, Any], key: str, lo: float, hi: float, *, lo_open: bool, hi_open: bool
) -> float:
    v = params[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise OverlayError(f"{key} must be a number, got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise OverlayError(f"{key} is non-finite: {v!r}")
    below = f <= lo if lo_open else f < lo
    above = f >= hi if hi_open else f > hi
    if below or above:
        lb, rb = ("(" if lo_open else "["), (")" if hi_open else "]")
        raise OverlayError(f"{key} must be in {lb}{lo}, {hi}{rb}, got {v!r}")
    return f


# --- trailing_stop ------------------------------------------------------------------------------


def trailing_stop(weights: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Zero a name whose adj_close sits more than `stop_pct` below its `lookback`-bar rolling high
    (incl. the current bar), and keep it at zero while that breach fired within the last
    `cooldown_bars` bars — all read from the view, no position state. A name with fewer than
    `lookback` bars uses the bars it has; a name absent from `view` passes through unchanged.
    A short is stopped the same way (weight -> 0, never flipped)."""
    lookback = int(params["lookback"])
    stop_pct = float(params["stop_pct"])
    cooldown = int(params["cooldown_bars"])
    wide = wide_adj_close(view)
    present = [s for s in weights.index if s in wide.columns]
    if not present:
        return weights
    px = wide[present]
    high = px.rolling(lookback, min_periods=1).max()
    breached = px < (1.0 - stop_pct) * high  # NaN price -> False (no breach on a missing bar)
    stopped = breached.iloc[-(cooldown + 1):].any(axis=0)
    out = weights.astype("float64").copy()
    out[stopped.index[stopped.to_numpy()]] = 0.0
    return out


def _validate_trailing_stop(params: dict[str, Any]) -> None:
    _exact_keys(params, {"lookback", "stop_pct", "cooldown_bars"})
    _positive_int(params, "lookback")
    _float_in(params, "stop_pct", 0.0, 1.0, lo_open=True, hi_open=True)
    _positive_int(params, "cooldown_bars", minimum=0)


def _trailing_stop_lookback(params: dict[str, Any]) -> int:
    return int(params["lookback"]) + int(params["cooldown_bars"])


# --- registry -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Overlay:
    fn: OverlayFn
    validate: Callable[[dict[str, Any]], None]
    lookback: Callable[[dict[str, Any]], int]


_OVERLAYS: dict[str, _Overlay] = {
    "trailing_stop": _Overlay(trailing_stop, _validate_trailing_stop, _trailing_stop_lookback),
    # "regime_gate" is registered by the regime-gate task.
}
# Read-only public dispatch view (see module docstring: static source is the identity).
OVERLAY_POLICIES = MappingProxyType(_OVERLAYS)


def _policy(policy_id: str) -> _Overlay:
    try:
        return _OVERLAYS[policy_id]
    except KeyError:
        raise OverlayError(
            f"unknown overlay policy {policy_id!r}; available: {sorted(_OVERLAYS)}"
        ) from None


def get_overlay_policy(policy_id: str) -> OverlayFn:
    return _policy(policy_id).fn


def validate_overlay_params(policy_id: str, params: dict[str, Any]) -> None:
    """Per-policy load-time validation: unknown id, then the policy's own exact-key + type +
    domain checks (non-finite floats are rejected inside those checks). Raises OverlayError."""
    _policy(policy_id).validate(params)


def overlay_lookback(spec: OverlaySpec) -> int:
    """The longest trailing window (in bars) the policy reads for these params."""
    return _policy(spec.policy).lookback(spec.params)


def resolve_overlays(
    specs: Sequence[OverlaySpec], *, feature_lookback: int | None
) -> tuple[OverlayFn, ...]:
    """Resolve + validate a declared chain: each policy id and its params, then the #345 cross-
    check that a DECLARED `feature_lookback` covers the longest overlay window (an under-declared
    lookback would size the walk-forward embargo too small). Returns the callables in order."""
    fns: list[OverlayFn] = []
    for i, spec in enumerate(specs):
        try:
            validate_overlay_params(spec.policy, spec.params)
        except OverlayError as exc:
            raise OverlayError(f"overlay[{i}] {spec.policy!r}: {exc}") from None
        fns.append(get_overlay_policy(spec.policy))
    if feature_lookback is not None and specs:
        need = max(overlay_lookback(s) for s in specs)
        if feature_lookback < need:
            raise OverlayError(
                f"feature_lookback {feature_lookback} is smaller than the longest overlay window "
                f"{need}; declare feature_lookback >= {need}"
            )
    return tuple(fns)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_portfolio_overlays.py -q`
Expected: all pass.

- [ ] **Step 5: Full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/portfolio/overlays.py tests/test_portfolio_overlays.py
git commit -m "feat(portfolio): the overlay seam — tighten-only invariants, registry, validation, trailing_stop"
```

---

### Task 5: Wire overlays into the strategy contract, loader, identity, approvals, CODEOWNERS

**Files:**
- Modify: `algua/strategies/base.py` (`StrategyConfig`, `LoadedStrategy`, `construct`, `config_hash`)
- Modify: `algua/strategies/loader.py` (`load_strategy`, `_loaded_for_test`)
- Modify: `algua/registry/approvals.py:15-30`
- Modify: `CODEOWNERS`, `tests/test_repo_hygiene.py:INTEGRITY_CRITICAL_MODULES`
- Test: `tests/test_strategy_overlays_identity.py`

**Interfaces:**
- Consumes: everything from Task 4.
- Produces: `StrategyConfig.overlays: list[OverlaySpec]`; `LoadedStrategy.overlay_fns: tuple[OverlayFn, ...] = ()` (must have one fn per config overlay — `__post_init__` fails closed otherwise); `LoadedStrategy.construct()` applies the chain; `config_hash` folds `"overlays"` only when non-empty.

- [ ] **Step 1: Write the failing tests**

```python
"""Overlays in the strategy contract: identity fold, construct() chain, loader validation, closure."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import algua.strategies.momentum as momentum_pkg
from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.portfolio.overlays import OverlaySpec, trailing_stop
from algua.registry.approvals import closure_module_names
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash
from algua.strategies.loader import StrategyNotFound, _loaded_for_test, load_strategy

_TS = {"lookback": 5, "stop_pct": 0.10, "cooldown_bars": 2}


def _cfg(**over: Any) -> StrategyConfig:
    base: dict[str, Any] = dict(
        name="s", universe=["A", "B"], execution=ExecutionContract(rebalance_frequency="1d"),
        params={"lookback": 10}, construction="top_k_equal_weight",
        construction_params={"top_k": 2},
    )
    base.update(over)
    return StrategyConfig(**base)


# --- identity --------------------------------------------------------------------------------

def test_empty_overlays_leaves_config_hash_byte_identical():
    # Digest of _cfg() computed on main at 82b8ec7, BEFORE the overlays field existed. An
    # undeclared / empty overlays list must reproduce it exactly (no live-approval churn).
    assert config_hash(_loaded_for_test(_cfg())) == "ea29606c94cca1a5731a1fe4552c9ea5"
    assert config_hash(_loaded_for_test(_cfg(overlays=[]))) == "ea29606c94cca1a5731a1fe4552c9ea5"


def test_non_empty_overlays_change_config_hash_and_order_matters():
    a = OverlaySpec(policy="trailing_stop", params=_TS)
    b = OverlaySpec(policy="trailing_stop", params={**_TS, "stop_pct": 0.2})
    base = config_hash(_loaded_for_test(_cfg()))
    ab = config_hash(_loaded_for_test(_cfg(overlays=[a, b])))
    ba = config_hash(_loaded_for_test(_cfg(overlays=[b, a])))
    assert base != ab and ab != ba


def test_overlay_param_change_changes_config_hash():
    a = config_hash(_loaded_for_test(_cfg(overlays=[OverlaySpec(policy="trailing_stop", params=_TS)])))
    b = config_hash(_loaded_for_test(_cfg(overlays=[OverlaySpec(policy="trailing_stop", params={**_TS, "cooldown_bars": 3})])))
    assert a != b


# --- LoadedStrategy ---------------------------------------------------------------------------

def test_loaded_strategy_requires_one_fn_per_declared_overlay():
    cfg = _cfg(overlays=[OverlaySpec(policy="trailing_stop", params=_TS)])
    with pytest.raises(ValueError, match="overlay_fns must hold one resolved fn per config overlay"):
        LoadedStrategy(config=cfg, signal_fn=lambda v, p: pd.Series(dtype="float64"),
                       construct_fn=get_construction_policy("top_k_equal_weight"))


def _view(prices: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(prices.values())))
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rows = [{"timestamp": t, "symbol": s, "open": px, "high": px, "low": px, "close": px,
             "adj_close": px, "volume": 1.0}
            for s, path in prices.items() for t, px in zip(ts, path, strict=True)]
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_construct_applies_the_overlay_chain_after_construction():
    cfg = _cfg(overlays=[OverlaySpec(policy="trailing_stop", params=_TS)])
    strat = LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series({"A": 2.0, "B": 1.0}),
        construct_fn=get_construction_policy("top_k_equal_weight"),
        overlay_fns=(trailing_stop,),
    )
    view = _view({"A": [90.0, 100.0, 98.0, 95.0, 85.0], "B": [50.0] * 5})  # A breached
    w = strat.target_weights(view)
    assert w.to_dict() == {"A": 0.0, "B": 0.5}  # top-2 equal weight, then A stopped; B stays 0.5


def test_construct_without_overlays_is_unchanged():
    strat = LoadedStrategy(
        config=_cfg(),
        signal_fn=lambda v, p: pd.Series({"A": 2.0, "B": 1.0}),
        construct_fn=get_construction_policy("top_k_equal_weight"),
    )
    view = _view({"A": [1.0] * 3, "B": [1.0] * 3})
    assert strat.target_weights(view).to_dict() == {"A": 0.5, "B": 0.5}


# --- loader -----------------------------------------------------------------------------------

def _write(stem: str, overlays_src: str, feature_lookback: str = "None") -> Path:
    path = Path(momentum_pkg.__path__[0]) / f"{stem}.py"
    path.write_text(
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.portfolio.overlays import OverlaySpec\n"
        "from algua.strategies.base import StrategyConfig\n"
        f"CONFIG = StrategyConfig(name='{stem}', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive',\n"
        f"    feature_lookback={feature_lookback},\n"
        f"    overlays={overlays_src})\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    return pd.Series(dtype='float64')\n"
    )
    return path


@pytest.fixture
def tmp_strategy():
    import sys
    made: list[Path] = []

    def make(stem: str, overlays_src: str, feature_lookback: str = "None") -> str:
        made.append(_write(stem, overlays_src, feature_lookback))
        return stem

    yield make
    for p in made:
        p.unlink(missing_ok=True)
        sys.modules.pop(f"algua.strategies.momentum.{p.stem}", None)


def test_loader_binds_overlay_fns(tmp_strategy):
    name = tmp_strategy("tmp_ov_ok", f"[OverlaySpec(policy='trailing_stop', params={_TS!r})]", "7")
    strat = load_strategy(name)
    assert strat.overlay_fns == (trailing_stop,)


def test_loader_rejects_unknown_overlay_policy(tmp_strategy):
    name = tmp_strategy("tmp_ov_bad_id", "[OverlaySpec(policy='nope', params={})]")
    with pytest.raises(StrategyNotFound, match="unknown overlay policy 'nope'"):
        load_strategy(name)


def test_loader_rejects_bad_overlay_params(tmp_strategy):
    name = tmp_strategy("tmp_ov_bad_params", "[OverlaySpec(policy='trailing_stop', params={'lookback': 5})]")
    with pytest.raises(StrategyNotFound, match=r"overlay\[0\] 'trailing_stop': missing"):
        load_strategy(name)


def test_loader_rejects_feature_lookback_below_overlay_window(tmp_strategy):
    name = tmp_strategy("tmp_ov_short_lb", f"[OverlaySpec(policy='trailing_stop', params={_TS!r})]", "6")
    with pytest.raises(StrategyNotFound, match="feature_lookback 6 is smaller than the longest overlay window 7"):
        load_strategy(name)


# --- approvals closure ------------------------------------------------------------------------

def test_closure_includes_overlays_and_regime_modules():
    names = closure_module_names(load_strategy("cross_sectional_momentum"))
    assert "algua.portfolio.overlays" in names
    assert "algua.features.regime" in names
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_strategy_overlays_identity.py -q`
Expected: failures on `overlays` being an unknown `StrategyConfig` field / `overlay_fns` unknown kwarg.

- [ ] **Step 3: Modify `algua/strategies/base.py`**

Add to the imports (keep alphabetical within the `algua` group):
```python
from algua.portfolio.overlays import OverlayFn, OverlaySpec, apply_overlays
```

In `StrategyConfig`, after `construction_params: dict[str, Any] = {}`:
```python
    # Ordered, stateless, tighten-only overlays applied AFTER construction and BEFORE the capacity
    # cap (overlays spec). Each spec is resolved + validated by the loader against
    # algua.portfolio.overlays; empty = no overlay stage (every pre-existing strategy).
    overlays: list[OverlaySpec] = []
```

In `LoadedStrategy`, after `model_handle: ModelHandle | None = None`:
```python
    # The RESOLVED overlay callables, one per `config.overlays` entry, in order (raw policy fns —
    # params are read from the spec at call time, so a sweep that rebuilds the config takes effect).
    overlay_fns: tuple[OverlayFn, ...] = ()
```

At the top of `__post_init__` (before the three-way exclusivity block):
```python
        if len(self.overlay_fns) != len(cfg.overlays):
            raise ValueError(
                f"overlay_fns must hold one resolved fn per config overlay: got "
                f"{len(self.overlay_fns)} fn(s) for {len(cfg.overlays)} overlay(s)"
            )
```
(`cfg = self.config` is already the first line of `__post_init__`; place this after it.)

In `construct()`, between the `construct_fn` call and the capacity block:
```python
        weights = apply_overlays(weights, view, self.config.overlays, self.overlay_fns)
```
and extend the comment above the capacity block: `# ... The overlay chain runs BEFORE the cap: the cap is the hardest liquidity wall and must see the final vector.`

In `config_hash`, after the `if strategy.config.needs_model:` block:
```python
    # Overlays fold in ONLY when declared, so every pre-existing strategy's hash is byte-identical.
    # Policy ids, params AND order are identity: reordering two overlays is a different strategy.
    if strategy.config.overlays:
        identity["overlays"] = [spec.model_dump() for spec in strategy.config.overlays]
```

Check `wc -l algua/strategies/base.py` stays `<= 380`.

- [ ] **Step 4: Modify `algua/strategies/loader.py`**

Add import:
```python
from algua.portfolio.overlays import OverlayError, resolve_overlays
```

In `load_strategy`, change the construction try-block to:
```python
    try:
        construct_fn = get_construction_policy(config.construction)
        validate_construction_params(config.construction, config.construction_params)
        overlay_fns = resolve_overlays(config.overlays, feature_lookback=config.feature_lookback)
    except (ConstructionError, OverlayError) as exc:
        raise StrategyNotFound(f"{name}: {exc}") from exc
```
and pass `overlay_fns=overlay_fns` to **every** `LoadedStrategy(...)` construction in `load_strategy` (the model, fundamentals, news, and plain branches — four sites).

In `_loaded_for_test`:
```python
    fn = get_construction_policy(config.construction)
    return LoadedStrategy(
        config=config, signal_fn=lambda view, params: pd.Series(dtype="float64"), construct_fn=fn,
        overlay_fns=resolve_overlays(config.overlays, feature_lookback=config.feature_lookback),
    )
```

- [ ] **Step 5: Root the approvals closure from the overlays module**

In `algua/registry/approvals.py`, after `_CONSTRUCTION_MODULE = "algua.portfolio.construction"`:
```python
_OVERLAYS_MODULE = "algua.portfolio.overlays"
```
and in `_merged_closure_for`, after the construction merge:
```python
    merged.update(_first_party_closure(importlib.import_module(_OVERLAYS_MODULE)))
```
Update the docstring's last sentence to: `...AND the construction policy module AND the overlays module (resolved by NAME...)`.

- [ ] **Step 6: CODEOWNERS + hygiene set**

Append to `CODEOWNERS` under the "carved out" block:
```
/algua/portfolio/overlays.py           @Lior-Nis   # tighten-only overlay stage: invariants + policies, hashed into strategy identity
/algua/portfolio/construction.py       @Lior-Nis   # construction policies hashed into strategy identity (#141); was unprotected until the overlays PR
```
Add to `INTEGRITY_CRITICAL_MODULES` in `tests/test_repo_hygiene.py`:
```python
        "algua/portfolio/construction.py",
        "algua/portfolio/overlays.py",
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_strategy_overlays_identity.py tests/test_config_hash_fields.py tests/test_registry_approvals.py tests/test_repo_hygiene.py tests/test_strategy_loader.py tests/test_module_size_ratchet.py -q`
Expected: all pass. Note `test_closure_module_names_equals_source_closure_keys` and any test pinning a literal `code_hash` value: the closure grows for every strategy (overlays + regime modules), so a pinned literal must be regenerated; a test that only asserts equality between two derivations is unaffected.

- [ ] **Step 8: Full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/strategies/base.py algua/strategies/loader.py algua/registry/approvals.py CODEOWNERS tests/test_repo_hygiene.py tests/test_strategy_overlays_identity.py
git commit -m "feat(strategies): overlays on the strategy contract — construct() chain, loader resolution, identity fold, approvals closure, CODEOWNERS

Every strategy's code_hash changes (the closure now roots from portfolio/overlays);
config_hash is byte-identical for every strategy that declares no overlays."
```

---

### Task 6: The `regime_gate` policy

**Files:**
- Modify: `algua/portfolio/overlays.py` (add the policy + validator + lookback; register it)
- Test: `tests/test_portfolio_overlays.py` (append)

**Interfaces:**
- Consumes: `equal_weight_index`, `rolling_drawdown`, `turbulence`, `robust_zscore` from Task 3; helpers from Task 4.
- Produces: `regime_gate(weights, view, params) -> pd.Series`, registered as `"regime_gate"`. Params (all required): `trend_window, dd_window, turb_window, z_window, shock_window, fast_lookback` (int ≥ 1), `persistence` (int ≥ 1, ≤ `dd_window`), `dd_threshold, shock_return` (float in (0,1)), `turb_z, fast_turb_z` (float > 0), `neutral_exposure, risk_off_exposure, fast_exposure` (float in [0,1], `risk_off_exposure <= neutral_exposure`).

- [ ] **Step 1: Append the failing tests to `tests/test_portfolio_overlays.py`**

```python
# --- regime_gate --------------------------------------------------------------------------------

from algua.portfolio.overlays import regime_gate  # noqa: E402

# Small windows so a 60-bar synthetic path can traverse the states. Vol/turbulence stresses are
# disabled by absurd thresholds unless a test enables them.
_RG = {
    "trend_window": 10, "dd_window": 10, "dd_threshold": 0.05,
    "turb_window": 5, "z_window": 10, "turb_z": 1e9,
    "persistence": 2,
    "shock_window": 3, "shock_return": 0.9, "fast_turb_z": 1e9, "fast_lookback": 1,
    "neutral_exposure": 0.6, "risk_off_exposure": 0.2, "fast_exposure": 0.25,
}
_W2 = pd.Series({"A": 0.5, "B": 0.5})


def _uni(path: list[float]) -> pd.DataFrame:
    """A 3-symbol universe that all follow `path` (turbulence is then degenerate -> NaN -> off)."""
    return _view({"A": path, "B": [p * 2 for p in path], "C": [p * 3 for p in path]})


def _up(n: int, start: float = 100.0, step: float = 0.002) -> list[float]:
    return [start * (1 + step) ** i for i in range(n)]


def test_regime_gate_risk_on_leaves_weights_untouched():
    out = regime_gate(_W2, _uni(_up(40)), _RG)
    assert out.to_dict() == _W2.to_dict()


def test_regime_gate_risk_off_after_trend_and_drawdown_persist():
    path = _up(40) + [_up(40)[-1] * (0.98 ** i) for i in range(1, 11)]  # -18% over 10 bars
    out = regime_gate(_W2, _uni(path), _RG)  # trend below mean AND dd < -5% for >= 2 bars
    assert out.to_dict() == pytest.approx({"A": 0.1, "B": 0.1})


def test_regime_gate_neutral_on_a_single_stress():
    # 30 flat bars at 100, then 3 bars at 97: a -3% drawdown (under the 5% threshold, so NOT a
    # drawdown stress) but the level sits below its 10-bar mean (a trend stress) for 3 bars
    # >= persistence 2 -> exactly one stress -> neutral_exposure.
    path = [100.0] * 30 + [97.0] * 3
    out = regime_gate(_W2, _uni(path), _RG)
    assert out["A"] == pytest.approx(0.5 * 0.6)


def test_regime_gate_persistence_blocks_a_one_bar_state():
    # One bar of stress (last bar only) with persistence=2 -> the prior risk-on run still stands.
    path = _up(40) + [_up(40)[-1] * 0.90]
    out = regime_gate(_W2, _uni(path), _RG)
    assert out.to_dict() == _W2.to_dict()


def test_regime_gate_fast_shock_applies_fast_exposure():
    params = {**_RG, "shock_window": 1, "shock_return": 0.05, "fast_lookback": 2}
    path = _up(40) + [_up(40)[-1] * 0.90, _up(40)[-1] * 0.90]  # -10% shock, then flat: within last 2
    out = regime_gate(_W2, _uni(path), params)
    # slow state: trend+dd stressed for 2 bars -> 0.2; fast 0.25 -> min = 0.2
    assert out["A"] == pytest.approx(0.5 * 0.2)
    params2 = {**params, "risk_off_exposure": 0.6, "neutral_exposure": 0.6}
    out2 = regime_gate(_W2, _uni(path), params2)
    assert out2["A"] == pytest.approx(0.5 * 0.25)  # now fast is the binding one


def test_regime_gate_volatility_stress_via_turbulence():
    rng = np.random.default_rng(3)
    n = 60
    rets = rng.normal(0.0, 0.005, size=(n, 3))
    rets[-1] = 0.08  # a 16-sigma common shock on the LAST bar only (its trailing window is calm)
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    view = _view({s: prices[:, i].tolist() for i, s in enumerate(["A", "B", "C"])})
    params = {**_RG, "turb_z": 3.0, "persistence": 1, "trend_window": 3, "dd_window": 3,
              "dd_threshold": 0.99}  # trend can't be below a 3-bar mean while rising; dd disabled
    out = regime_gate(_W2, view, params)
    assert out["A"] == pytest.approx(0.5 * 0.6)  # exactly one stress (volatility) -> neutral


def test_regime_gate_no_op_on_short_history():
    out = regime_gate(_W2, _uni([100.0, 99.0, 98.0]), _RG)
    assert out.to_dict() == _W2.to_dict()


def test_regime_gate_lookback():
    spec = OverlaySpec(policy="regime_gate", params=_RG)
    # max(trend 10, dd 10, turb 5 + z 10, shock 3 + fast 1) + persistence 2 = 17
    assert overlay_lookback(spec) == 17


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"persistence": 11}, "persistence"),
        ({"risk_off_exposure": 0.7}, "risk_off_exposure"),
        ({"dd_threshold": 1.0}, "dd_threshold"),
        ({"turb_z": 0.0}, "turb_z"),
        ({"fast_exposure": 1.5}, "fast_exposure"),
        ({"trend_window": 0}, "trend_window"),
    ],
)
def test_validate_regime_gate_domains(over, msg):
    with pytest.raises(OverlayError, match=msg):
        validate_overlay_params("regime_gate", {**_RG, **over})


def test_validate_regime_gate_ok():
    validate_overlay_params("regime_gate", _RG)
```

Also add `import numpy as np` to the test module's imports.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_portfolio_overlays.py -q -k regime`
Expected: ImportError on `regime_gate`.

- [ ] **Step 3: Implement `regime_gate` in `algua/portfolio/overlays.py`**

Change the `algua.features.regime` import to:
```python
from algua.features.regime import (
    equal_weight_index,
    robust_zscore,
    rolling_drawdown,
    turbulence,
    wide_adj_close,
)
```

Insert before the `# --- registry` section:

```python
# --- regime_gate --------------------------------------------------------------------------------

_REGIME_INT_KEYS = (
    "trend_window", "dd_window", "turb_window", "z_window", "shock_window", "fast_lookback",
    "persistence",
)
_REGIME_KEYS = {
    *_REGIME_INT_KEYS, "dd_threshold", "shock_return", "turb_z", "fast_turb_z",
    "neutral_exposure", "risk_off_exposure", "fast_exposure",
}


def regime_gate(weights: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Two-speed exposure multiplier from the strategy's OWN universe (no reference symbols). The
    regime controls the risk budget, not asset selection: every weight is scaled by one factor.

    SLOW gate, per bar: count three stresses on the equal-weight universe index —
      trend       level < its `trend_window`-bar mean
      drawdown    rolling_drawdown(level, dd_window) < -dd_threshold
      volatility  robust_zscore(turbulence(view, turb_window), z_window) > turb_z
    score 0 -> risk-on (1.0), 1 -> neutral (`neutral_exposure`), >= 2 -> risk-off
    (`risk_off_exposure`). The state IN EFFECT is that of the most recent run of `persistence`
    consecutive bars sharing one state; none in the view -> risk-on.

    FAST overlay: if within the last `fast_lookback` bars the index's `shock_window`-bar return was
    below -`shock_return` OR the turbulence robust-z exceeded `fast_turb_z`, `fast_exposure` applies.

    Effective multiplier = min(slow, fast). A component without enough history is NOT stressed (a
    gate cannot fire on data it does not have), so on a short view this is a no-op."""
    p = params
    persistence = int(p["persistence"])
    level = equal_weight_index(view)
    tail = int(p["z_window"]) + persistence + int(p["fast_lookback"]) + 1
    turb_z = robust_zscore(turbulence(view, int(p["turb_window"]), last=tail), int(p["z_window"]))

    trend = level < level.rolling(int(p["trend_window"]), min_periods=int(p["trend_window"])).mean()
    dd = rolling_drawdown(level, int(p["dd_window"])) < -float(p["dd_threshold"])
    vol = turb_z > float(p["turb_z"])  # NaN -> False
    score = trend.astype(int) + dd.astype(int) + vol.astype(int)
    state = score.clip(upper=2)
    run_ok = state.rolling(persistence, min_periods=persistence).max() == state.rolling(
        persistence, min_periods=persistence
    ).min()
    in_effect = state.where(run_ok).ffill().fillna(0).iloc[-1]
    slow = {0: 1.0, 1: float(p["neutral_exposure"]), 2: float(p["risk_off_exposure"])}[int(in_effect)]

    shock = level.pct_change(int(p["shock_window"]), fill_method=None) < -float(p["shock_return"])
    fast_hit = bool((shock | (turb_z > float(p["fast_turb_z"]))).iloc[-int(p["fast_lookback"]):].any())
    fast = float(p["fast_exposure"]) if fast_hit else 1.0

    return weights.astype("float64") * min(slow, fast)


def _validate_regime_gate(params: dict[str, Any]) -> None:
    _exact_keys(params, _REGIME_KEYS)
    for key in _REGIME_INT_KEYS:
        _positive_int(params, key)
    if params["persistence"] > params["dd_window"]:
        raise OverlayError("persistence must be <= dd_window")
    for key in ("dd_threshold", "shock_return"):
        _float_in(params, key, 0.0, 1.0, lo_open=True, hi_open=True)
    for key in ("turb_z", "fast_turb_z"):
        _float_in(params, key, 0.0, math.inf, lo_open=True, hi_open=True)
    for key in ("neutral_exposure", "risk_off_exposure", "fast_exposure"):
        _float_in(params, key, 0.0, 1.0, lo_open=False, hi_open=False)
    if float(params["risk_off_exposure"]) > float(params["neutral_exposure"]):
        raise OverlayError("risk_off_exposure must be <= neutral_exposure")


def _regime_gate_lookback(params: dict[str, Any]) -> int:
    p = params
    return max(
        int(p["trend_window"]), int(p["dd_window"]),
        int(p["turb_window"]) + int(p["z_window"]),
        int(p["shock_window"]) + int(p["fast_lookback"]),
    ) + int(p["persistence"])
```

Register it (replace the placeholder comment in `_OVERLAYS`):
```python
    "regime_gate": _Overlay(regime_gate, _validate_regime_gate, _regime_gate_lookback),
```

Note on `_float_in` with `hi=math.inf, hi_open=True`: `f >= inf` is False for any finite f, so the upper bound is effectively open-ended; the message prints `(0.0, inf)`. That is intended.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_portfolio_overlays.py -q`
Expected: all pass. If `test_regime_gate_neutral_on_a_single_stress` derives `expected` as 0.2 or 1.0 rather than 0.6, that is fine: it asserts consistency with the features, not a fixed state.

- [ ] **Step 5: Full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/portfolio/overlays.py tests/test_portfolio_overlays.py
git commit -m "feat(portfolio): regime_gate overlay — two-speed universe-derived exposure multiplier"
```

---

### Task 7: Sweep namespace `overlay.<i>.<key>`

**Files:**
- Modify: `algua/backtest/sweep.py` (`_override`)
- Test: `tests/test_sweep_override.py` (append)

**Interfaces:**
- Consumes: `resolve_overlays`, `OverlayError`, `OverlaySpec` from Task 4; `LoadedStrategy.overlay_fns` from Task 5.
- Produces: `_override` accepts grid keys `overlay.<i>.<key>`; rebuilt strategies carry `overlay_fns`.

- [ ] **Step 1: Append the failing tests**

```python
from algua.portfolio.overlays import OverlaySpec, trailing_stop  # noqa: E402

_TS = {"lookback": 5, "stop_pct": 0.10, "cooldown_bars": 2}


def _with_overlay():
    cfg = StrategyConfig(
        name="m", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 60}, construction="top_k_equal_weight",
        construction_params={"top_k": 3},
        overlays=[OverlaySpec(policy="trailing_stop", params=_TS)],
    )
    return LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series(dtype="float64"),
        construct_fn=top_k_equal_weight, overlay_fns=(trailing_stop,),
    )


def test_override_tunes_overlay_params_and_keeps_fns():
    base = _with_overlay()
    out = _override(base, {"overlay.0.stop_pct": 0.2})
    assert out.config.overlays[0].params == {**_TS, "stop_pct": 0.2}
    assert out.overlay_fns == (trailing_stop,)
    assert base.config.overlays[0].params == _TS  # base untouched


def test_override_rejects_out_of_range_overlay_index():
    with pytest.raises(ValueError, match=r"overlay\.1\.stop_pct.*declares 1 overlay"):
        _override(_with_overlay(), {"overlay.1.stop_pct": 0.2})


def test_override_rejects_malformed_overlay_key():
    with pytest.raises(ValueError, match="expected overlay.<i>.<param>"):
        _override(_with_overlay(), {"overlay.stop_pct": 0.2})


def test_override_rejects_invalid_swept_overlay_param():
    with pytest.raises(ValueError, match="swept overlay params invalid"):
        _override(_with_overlay(), {"overlay.0.stop_pct": 1.5})


def test_override_rejects_overlay_window_exceeding_declared_lookback():
    base = _with_overlay()
    base = LoadedStrategy(
        config=base.config.model_copy(update={"feature_lookback": 7}),
        signal_fn=base.signal_fn, construct_fn=base.construct_fn, overlay_fns=base.overlay_fns,
    )
    with pytest.raises(ValueError, match="feature_lookback 7 is smaller"):
        _override(base, {"overlay.0.lookback": 10})
```

Add `import pytest` to the module's imports if absent.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sweep_override.py -q`
Expected: the new tests fail (`overlay.0.stop_pct` is treated as a signal param and rejected with the old message; `overlay_fns` missing on the rebuilt strategy).

- [ ] **Step 3: Modify `_override` in `algua/backtest/sweep.py`**

Add imports:
```python
from algua.portfolio.overlays import OverlayError, resolve_overlays
```

Add after `_CONSTRUCTION_PREFIX`:
```python
_OVERLAY_PREFIX = "overlay."


def _overlay_key(key: str, n_overlays: int) -> tuple[int, str]:
    """`overlay.<i>.<param>` -> (i, param); fails closed on shape and on an index the strategy
    does not declare."""
    idx_s, sep, param = key[len(_OVERLAY_PREFIX):].partition(".")
    if not sep or not idx_s.isdigit() or not param:
        raise ValueError(f"sweep key {key!r}: expected overlay.<i>.<param>")
    idx = int(idx_s)
    if idx >= n_overlays:
        raise ValueError(f"sweep key {key!r}: strategy declares {n_overlays} overlay(s)")
    return idx, param
```

Rewrite `_override`:
```python
def _override(strategy: LoadedStrategy, combo: dict[str, Any]) -> LoadedStrategy:
    """Return a LoadedStrategy whose params/construction_params/overlay params are the base merged
    with `combo`.

    A grid key prefixed `construction.` tunes `construction_params` (re-validated by the policy);
    `overlay.<i>.<key>` tunes `overlays[i].params` (re-validated by that policy, incl. the
    feature_lookback cover check); any other key tunes signal `params` and MUST already exist in the
    base params (so a typo'd key is rejected, never a silent no-op). Preserves the resolved
    construction policy, overlay fns and signal_panel. Does not mutate the base strategy/config.
    """
    cfg = strategy.config
    new_params = dict(cfg.params)
    new_cparams = dict(cfg.construction_params)
    new_oparams = [dict(spec.params) for spec in cfg.overlays]
    for key, value in combo.items():
        if key.startswith(_CONSTRUCTION_PREFIX):
            new_cparams[key[len(_CONSTRUCTION_PREFIX):]] = value
        elif key.startswith(_OVERLAY_PREFIX):
            idx, param = _overlay_key(key, len(cfg.overlays))
            new_oparams[idx][param] = value
        else:
            if key not in cfg.params:
                raise ValueError(
                    f"sweep key {key!r} is not a base signal param "
                    f"{sorted(cfg.params)}; prefix with 'construction.' to tune the "
                    f"construction policy or 'overlay.<i>.' to tune an overlay"
                )
            new_params[key] = value
    try:
        validate_construction_params(cfg.construction, new_cparams)
    except ConstructionError as exc:
        raise ValueError(f"swept construction_params invalid: {exc}") from exc
    new_overlays = [
        spec.model_copy(update={"params": p}) for spec, p in zip(cfg.overlays, new_oparams, strict=True)
    ]
    try:
        overlay_fns = resolve_overlays(new_overlays, feature_lookback=cfg.feature_lookback)
    except OverlayError as exc:
        raise ValueError(f"swept overlay params invalid: {exc}") from exc
    new_config = cfg.model_copy(
        update={"params": new_params, "construction_params": new_cparams, "overlays": new_overlays}
    )
    return LoadedStrategy(
        config=new_config,
        construct_fn=strategy.construct_fn,
        signal_fn=strategy.signal_fn,
        signal_panel_fn=strategy.signal_panel_fn,
        fundamentals_signal_fn=strategy.fundamentals_signal_fn,
        news_signal_fn=strategy.news_signal_fn,
        overlay_fns=overlay_fns,
    )
```

Update the `validate_sweep_grid` docstring's last paragraph: `Raises BacktestError (grid too large) or ValueError (unknown signal key, invalid construction or overlay params)`.

Check `wc -l algua/backtest/sweep.py` stays `<= 461`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sweep_override.py tests/test_cli_sweep.py tests/test_mergeback_queue.py tests/test_module_size_ratchet.py -q`
Expected: pass. One existing test may match the old "prefix with 'construction.'" message verbatim; if so, loosen its `match` to `"prefix with 'construction.'"` (the new message still contains that substring).

- [ ] **Step 5: Full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/backtest/sweep.py tests/test_sweep_override.py
git commit -m "feat(sweep): overlay.<i>.<key> grid namespace, re-validated per policy"
```

---

### Task 8: Example strategy, end-to-end + parity tests, docs and skills

**Files:**
- Create: `algua/strategies/momentum/momentum_regime_stop.py`
- Test: `tests/test_overlays_end_to_end.py`
- Modify: `docs/architecture.md` (after the "A strategy" subsection, and one bullet in "The walls")
- Modify: `.codex/skills/author-a-strategy/SKILL.md` (new section after "Construction policies"; one line in "Tuning construction in a sweep")
- Modify: `.codex/skills/interpret-results/SKILL.md` (one bullet under "Pitfalls to watch")

**Interfaces:**
- Consumes: everything above. `verify_signal_panel_parity(strategy, provider, start, end)` from `algua.backtest.decision_path`; `run(strategy, provider, start, end)` from `algua.backtest.engine`; `SyntheticProvider(seed=...)` from `algua.backtest._sample`.

- [ ] **Step 1: Write the example strategy**

```python
"""Cross-sectional momentum with a regime gate and a trailing stop — the bundled OVERLAYS example.

Same alpha as `cross_sectional_momentum` (trailing return, top-k equal weight); the difference is
the `overlays` chain: `regime_gate` scales the whole book by a universe-derived regime multiplier,
then `trailing_stop` zeroes any name more than 15% off its 60-bar high (with a 5-bar cooldown).
Both are stateless functions of the PIT view, enforced tighten-only inside construct(). It also
exposes `signal_panel`, so the exhaustive parity gate exercises the overlay chain on every bar."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.features.alphas import xs_trailing_return
from algua.portfolio.overlays import OverlaySpec
from algua.strategies.base import StrategyConfig

# Provenance marker (additions-only discipline): bundled examples are hand-authored.
GENERATED_BY = "human"

_REGIME = {
    "trend_window": 126, "dd_window": 63, "dd_threshold": 0.10,
    "turb_window": 63, "z_window": 126, "turb_z": 3.0,
    "persistence": 5,
    "shock_window": 3, "shock_return": 0.05, "fast_turb_z": 4.0, "fast_lookback": 5,
    "neutral_exposure": 0.6, "risk_off_exposure": 0.2, "fast_exposure": 0.25,
}
_STOP = {"lookback": 60, "stop_pct": 0.15, "cooldown_bars": 5}

CONFIG = StrategyConfig(
    name="momentum_regime_stop",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    overlays=[
        OverlaySpec(policy="regime_gate", params=_REGIME),
        OverlaySpec(policy="trailing_stop", params=_STOP),
    ],
    # max(signal 60, regime_gate max(126, 63, 63+126, 3+5)+5 = 194, trailing_stop 60+5 = 65) = 194
    feature_lookback=194,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Trailing `lookback`-bar return per symbol (the alpha score)."""
    return xs_trailing_return(view, params)


def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Vectorized SCORES twin of `signal`; the overlays run inside construct() per row either way."""
    lookback = int(params["lookback"])
    wide = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return wide / wide.shift(lookback) - 1.0
```

- [ ] **Step 2: Write the end-to-end tests**

```python
"""Overlays end to end: the example loads, backtests, and passes the exhaustive parity gate."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.decision_path import verify_signal_panel_parity
from algua.backtest.engine import run
from algua.portfolio.overlays import regime_gate, trailing_stop
from algua.strategies.base import config_hash
from algua.strategies.loader import load_strategy

START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime(2024, 6, 1, tzinfo=UTC)


def test_example_loads_with_resolved_overlay_fns():
    strat = load_strategy("momentum_regime_stop")
    assert strat.overlay_fns == (regime_gate, trailing_stop)
    assert strat.config.feature_lookback == 194


def test_example_backtests_and_never_exceeds_the_unoverlaid_gross():
    provider = SyntheticProvider(seed=7)
    with_overlays = run(load_strategy("momentum_regime_stop"), provider, START, END)
    without = run(load_strategy("cross_sectional_momentum"), provider, START, END)
    assert with_overlays.metrics["avg_gross_exposure"] <= without.metrics["avg_gross_exposure"] + 1e-9
    assert with_overlays.metrics["avg_gross_exposure"] > 0.0  # it did trade


def test_example_passes_the_exhaustive_parity_gate():
    assert verify_signal_panel_parity(
        load_strategy("momentum_regime_stop"), SyntheticProvider(seed=7), START, END
    ) is None


def test_example_identity_differs_from_the_plain_momentum():
    assert config_hash(load_strategy("momentum_regime_stop")) != config_hash(
        load_strategy("cross_sectional_momentum")
    )


def test_run_is_deterministic_with_overlays():
    a = run(load_strategy("momentum_regime_stop"), SyntheticProvider(seed=7), START, END)
    b = run(load_strategy("momentum_regime_stop"), SyntheticProvider(seed=7), START, END)
    assert a.metrics == b.metrics
    assert isinstance(a.metrics["sharpe"], float) or pd.isna(a.metrics["sharpe"])
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_overlays_end_to_end.py -q`
Expected: pass. If `avg_gross_exposure` is 0.0 for the overlaid run, the synthetic path spent the whole window risk-off or stopped; widen `START` to `2022-01-01` before touching the policies. `cross_sectional_momentum` shares the universe, params, and construction, so the gross comparison is apples to apples.

- [ ] **Step 4: Docs — `docs/architecture.md`**

After the "### A strategy" subsection (line ~124), insert:

```markdown
### An overlay policy
`algua/portfolio/overlays.py` — an overlay is `(weights, view, params) -> weights`, applied in
declared order inside `LoadedStrategy.construct()` AFTER the construction policy and BEFORE the
capacity cap. Add the function, its param validator and its `lookback(params)` to `_OVERLAYS`
(static dict, no runtime registration: the module source is hashed into strategy identity). It
must be **tighten-only** — never add a symbol, scale up, flip a side or emit NaN — and
`apply_overlays` enforces that after every policy. A strategy opts in with
`overlays=[OverlaySpec(policy=..., params=...)]`; sweeps tune it as `overlay.<i>.<key>`.
```

In "The walls", after the PIT bullet, add:

```markdown
- **Overlays are tighten-only** — `portfolio/overlays.py::apply_overlays` rejects any policy output
  that adds a symbol, increases `|weight|`, flips a sign or is non-finite. That is what lets the
  #135 risk rails run once, inside construction, and stay valid after any overlay chain.
```

- [ ] **Step 5: Docs — `.codex/skills/author-a-strategy/SKILL.md`**

Insert a new section after "## Construction policies" (before "## The bars you receive"):

````markdown
## Overlays (optional): regime gating and stops, after construction

`CONFIG.overlays` is an ordered list of `OverlaySpec(policy=<id>, params={...})` applied INSIDE the
loader's `construct()` — after the construction policy, before the capacity cap — in the backtest
loop, the fast path, and both lanes alike. Overlays are **stateless** functions of the PIT `view`
(no entry price, no position ledger) and **tighten-only**: they may zero, drop or scale down a
weight, never add a symbol, scale up, or flip a side. Freed weight is cash.

```python
from algua.portfolio.overlays import OverlaySpec

CONFIG = StrategyConfig(
    ...,
    overlays=[
        OverlaySpec(policy="regime_gate", params={...}),   # book-level exposure multiplier
        OverlaySpec(policy="trailing_stop", params={"lookback": 60, "stop_pct": 0.15, "cooldown_bars": 5}),
    ],
    feature_lookback=194,  # MUST cover the longest overlay window (the loader checks this)
)
```

| Policy id | params | What it does |
|---|---|---|
| `regime_gate` | `trend_window, dd_window, turb_window, z_window, shock_window, fast_lookback, persistence` (ints ≥1; `persistence <= dd_window`), `dd_threshold, shock_return` ∈ (0,1), `turb_z, fast_turb_z` > 0, `neutral_exposure, risk_off_exposure, fast_exposure` ∈ [0,1] (`risk_off <= neutral`) | Scales every weight by `min(slow, fast)`. Slow: counts trend / drawdown / turbulence stresses on the equal-weight index of YOUR universe (0 → 1.0, 1 → `neutral_exposure`, ≥2 → `risk_off_exposure`), a state binding only after `persistence` bars. Fast: a `shock_window`-bar index drop past `shock_return` or a turbulence spike within the last `fast_lookback` bars applies `fast_exposure`. Window = `max(trend, dd, turb+z, shock+fast_lookback) + persistence`. |
| `trailing_stop` | `lookback` ≥1, `stop_pct` ∈ (0,1), `cooldown_bars` ≥0 | Zeroes a name more than `stop_pct` below its `lookback`-bar rolling high, and keeps it at zero while that breach fired within the last `cooldown_bars`. Window = `lookback + cooldown_bars`. |

Rules: the regime reads only your universe (no SPY/VIX — see `algua/features/regime.py`); a
component without enough history is not stressed; a stop never zeroes a name for lack of data. Do
NOT smuggle regime or stop logic into `signal()` — it pollutes the alpha, escapes the sweep
namespace, and is invisible to identity. Bespoke overlay = a new policy in
`algua/portfolio/overlays.py` (additions-only, CODEOWNERS-protected), never inline math.
````

In "## Tuning construction in a sweep", append the sentence: `A key prefixed **\`overlay.<i>.\`** tunes \`overlays[i].params\` (re-validated by that policy, including the \`feature_lookback\` cover check), e.g. \`overlay.1.stop_pct=0.1,0.15,0.2\`.`

- [ ] **Step 6: Docs — `.codex/skills/interpret-results/SKILL.md`**

Under "## Pitfalls to watch", add a bullet:

```markdown
- **Overlays show up as time-varying gross exposure.** A strategy with a `regime_gate` overlay sits
  partly in cash through stressed regimes, so `avg_gross_exposure` < 1 and Sharpe is earned on less
  capital at risk; a `trailing_stop` shows as extra turnover around drawdowns. Compare against the
  same signal WITHOUT overlays before crediting the overlay — the gate does not treat overlaid
  strategies differently.
```

- [ ] **Step 7: Full gate, then commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/strategies/momentum/momentum_regime_stop.py tests/test_overlays_end_to_end.py docs/architecture.md .codex/skills/author-a-strategy/SKILL.md .codex/skills/interpret-results/SKILL.md
git commit -m "feat: momentum_regime_stop example + end-to-end/parity tests; overlays documented in the module map and the authoring skill"
```

- [ ] **Step 8: Open the PR** (do not push to `main`; push the branch and open a PR against `main`):

```bash
git push -u origin feat/portfolio-overlays
gh pr create --title "feat: portfolio overlays — a tighten-only stage after construction (regime_gate, trailing_stop)" --body-file - <<'EOF'
Implements docs/superpowers/specs/2026-09-10-portfolio-overlays-design.md.

- `StrategyConfig.overlays`: ordered, stateless, tighten-only overlay chain inside `construct()`,
  after the construction policy and before the capacity cap; enforced identically on the backtest
  loop, the fast path (parity-guarded), and both lanes with no engine/lane changes.
- `algua/portfolio/overlays.py`: seam + invariants + registry + `trailing_stop` + `regime_gate`.
- `algua/features/regime.py`: EW universe index, rolling drawdown, turbulence, robust z.
- Identity: `config_hash` folds overlays only when declared (existing hashes unchanged);
  approvals closure now roots from the overlays module (every strategy's `code_hash` changes).
- Sweeps: `overlay.<i>.<key>`. CODEOWNERS: `portfolio/overlays.py` AND the previously
  unprotected `portfolio/construction.py`.
- Two pure-move carves (`strategies/tradable.py`, `backtest/sweep_grid.py`) to stay under the
  size ratchet.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01HupQYtfZet4Cefj7xyi7ZG
EOF
```
