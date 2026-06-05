---
name: author-a-strategy
description: How to author a new algua strategy module — the CONFIG + compute_weights contract, the bar schema the function receives, available features, where the file goes, and the GENERATED_BY/additions-only discipline. Use when writing a strategy.
---

# Authoring an algua strategy

A strategy is a **single Python module** at `algua/strategies/examples/<name>.py` that exposes two
names: `CONFIG` and `compute_weights`. The loader imports it by name (`load_strategy("<name>")`),
so the filename stem **is** the strategy name and must match `CONFIG.name`.

`compute_weights(view, params)` is the **authored signal**. The loader wraps it in a
`LoadedStrategy` adapter that exposes the protocol-level `Strategy.target_weights(features)`
(1-arg) by injecting `params` — so authored modules never define `target_weights` themselves.

## The contract

```python
"""<one-line description of the strategy>."""
from __future__ import annotations  # MUST be the first statement (after the docstring)

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

GENERATED_BY = "agent"  # module-level marker; place it after the imports (not before __future__)

CONFIG = StrategyConfig(
    name="momentum_lb40",                                   # MUST equal the filename stem
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 40, "top_k": 3},
)


def compute_weights(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Pure, cross-sectional, per-bar. Given the point-in-time view of bars UP TO the
    current bar, return target weights indexed by symbol. The engine applies the
    t→t+1 decision lag centrally — do NOT shift or peek ahead yourself."""
    ...
```

Rules that matter:
- `compute_weights` is **pure** (no I/O, no network, no global state) and **cross-sectional**: it
  returns a `pd.Series` of weights indexed by symbol for the current bar. Return an empty
  `pd.Series(dtype="float64")` when you can't form a view (e.g. not enough history yet).
- **Never look ahead.** The `view` contains bars only up to the current timestamp, and the engine
  enforces the `t→t+1` execution lag (`decision_lag_bars=1`). Do not re-implement or weaken this.
- `rebalance_frequency` is `"1d"` (daily) for now.
- Keep weights sane (they're target portfolio weights; the engine handles gross-exposure limits).

## The bars you receive

`view` is **long-format** (see `docs/contracts/bar-schema.md`): a tz-aware UTC `timestamp` index
(session date at midnight for daily) and columns `symbol, open, high, low, close, adj_close,
volume`. Pivot to wide when you need a matrix:

```python
wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
```

## Available features

`algua/features/indicators.py` holds pure indicators (e.g. `momentum(wide, lookback=...)`). Reuse
them; add to that module only if a new indicator is genuinely needed (keep it pure — `features/`
imports nothing beyond contracts).

## Reuse vs. new logic

- **Parameter variant** of an existing idea: import its function and define a new `CONFIG` —
  `from algua.strategies.examples.cross_sectional_momentum import compute_weights`.
- **New signal**: write a new `compute_weights`. Look at
  `algua/strategies/examples/cross_sectional_momentum.py` as the reference implementation.

## Optional: `compute_weights_panel` (advanced acceleration hook)

`compute_weights` is the **canonical signal** — paper, live, and the backtest all run it per bar.
For the common "stateless" case (weights at `t` are a pure function of the trailing window ending
at `t`), the per-bar Python loop is slow. A module **MAY** additionally define a module-level:

```python
def compute_weights_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Decision-time (PRE-lag) weights for the WHOLE period at once, indexed timestamp × symbol."""
```

The loader detects it and binds it as `LoadedStrategy.panel_fn`; the backtest engine then computes
the whole weights matrix in one vectorized shot instead of looping. Rules:

- It is **NOT a second signal.** It MUST produce results **identical** to running `compute_weights`
  on the expanding view at each bar. The engine enforces this with a **fail-closed parity guard**:
  on every run it re-checks the panel output against the per-bar `compute_weights` on a bounded
  deterministic sample of bars, and **raises** on any disagreement (it never silently falls back).
- Same purity rules as `compute_weights`: pure, pandas-only, no I/O, no look-ahead. Return PRE-lag
  decision weights (the engine applies the `t→t+1` shift centrally — do not shift yourself).
- Rows without enough history must be flat, exactly as the per-bar function returns empty there.
- It is **optional and advanced** — only add it when the per-bar loop is a measured bottleneck. Most
  strategies should ship `compute_weights` alone. If you add a panel fn, you **must** add a parity
  test (`pd.testing.assert_frame_equal(..., check_exact=False, atol=WEIGHT_TOL, rtol=0)`) proving
  the two paths agree, mirroring `tests/test_fast_path.py`.
- **PIT (point-in-time universe) runs always use the per-bar loop**, even when a panel fn exists —
  the as-of masking can't be reproduced by a whole-period panel fn. The fast path is static-universe
  only. See `algua/strategies/examples/cross_sectional_momentum.py` for a worked example.

## Discipline

- **Additions only.** Create a NEW file under `algua/strategies/examples/`. Do **not** edit or
  overwrite an existing strategy (especially the curated `cross_sectional_momentum.py`).
- Include a module-level `GENERATED_BY = "agent"` (after the imports) so machine-authored
  strategies are identifiable. Do not place it before `from __future__ import annotations`.
- After authoring, verify it loads and runs: `uv run algua backtest run <name> --demo` should emit
  metrics JSON, not an error.
