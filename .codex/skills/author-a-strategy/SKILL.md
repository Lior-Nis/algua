---
name: author-a-strategy
description: How to author a new algua strategy module — the CONFIG + target_weights contract, the bar schema the function receives, available features, where the file goes, and the GENERATED_BY/additions-only discipline. Use when writing a strategy.
---

# Authoring an algua strategy

A strategy is a **single Python module** at `algua/strategies/examples/<name>.py` that exposes two
names: `CONFIG` and `target_weights`. The loader imports it by name (`load_strategy("<name>")`),
so the filename stem **is** the strategy name and must match `CONFIG.name`.

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


def target_weights(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Pure, cross-sectional, per-bar. Given the point-in-time view of bars UP TO the
    current bar, return target weights indexed by symbol. The engine applies the
    t→t+1 decision lag centrally — do NOT shift or peek ahead yourself."""
    ...
```

Rules that matter:
- `target_weights` is **pure** (no I/O, no network, no global state) and **cross-sectional**: it
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
  `from algua.strategies.examples.cross_sectional_momentum import target_weights`.
- **New signal**: write a new `target_weights`. Look at
  `algua/strategies/examples/cross_sectional_momentum.py` as the reference implementation.

## Discipline

- **Additions only.** Create a NEW file under `algua/strategies/examples/`. Do **not** edit or
  overwrite an existing strategy (especially the curated `cross_sectional_momentum.py`).
- Include a module-level `GENERATED_BY = "agent"` (after the imports) so machine-authored
  strategies are identifiable. Do not place it before `from __future__ import annotations`.
- After authoring, verify it loads and runs: `uv run algua backtest run <name> --demo` should emit
  metrics JSON, not an error.
