---
name: author-a-strategy
description: How to author a new algua strategy module — the CONFIG + signal + construction contract (signal scores, named construction policy), the bar schema the function receives, available features, where the file goes, and the GENERATED_BY/additions-only discipline. Use when writing a strategy.
---

# Authoring an algua strategy

A strategy is a **single Python module** at `algua/strategies/<family>/<name>.py` (created via
`uv run algua strategy new <name> --family <slug>`) that exposes two names: `CONFIG` and
`signal`. The loader imports it by bare name (`load_strategy("<name>")`), so the filename
stem **is** the strategy name and must match `CONFIG.name`. Family slugs may contain hyphens
(`mean-reversion`), which map to underscores on disk (`mean_reversion/`). Each family directory
must keep an **empty, side-effect-free `__init__.py`** — a family `__init__` must never import its
member strategies, because the loader relies on that for the single-import contract.

## signal → construction: the two halves

A strategy is split into two layers (issue #141):

1. **`signal(view, params)`** — the **authored** half. It returns a `pd.Series` of cross-sectional
   **scores** (higher = more attractive), NOT weights. This is your alpha.
2. **construction** — a **named, library-provided policy** (declared in `CONFIG.construction`) that
   maps scores → target weights under a risk convention. You pick a policy; you don't write
   weight math in the strategy.

The loader wraps both into a `LoadedStrategy` adapter that composes `construct(signal(view), view)`
and exposes the protocol-level `Strategy.target_weights(features)` — so authored modules never
define `target_weights` (or any weight logic) themselves. After construction, the **#135 hard risk
walls** (gross-exposure, concentration, short bounds — `algua/risk/limits.py::validate_decision_weights`)
run **centrally**; you neither re-implement nor weaken them.

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
    params={"lookback": 40},                                # signal params
    construction="top_k_equal_weight",                      # a policy id (see list below)
    construction_params={"top_k": 3},                       # validated per-policy at load
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Pure, cross-sectional, per-bar. Given the point-in-time view of bars UP TO the
    current bar, return a SCORE per symbol (higher = more attractive — NOT a weight). The
    construction policy named in CONFIG turns these scores into target weights, and the engine
    applies the t→t+1 decision lag centrally — do NOT shift or peek ahead yourself."""
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    return (wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0).dropna()
```

Rules that matter:
- `signal` is **pure** (no I/O, no network, no global state) and **cross-sectional**: it returns a
  `pd.Series` of **scores** indexed by symbol for the current bar. Return an empty
  `pd.Series(dtype="float64")` when you can't form a view (e.g. not enough history yet).
- **Scores are not weights.** Don't normalize, clip, or size — that's the construction policy's job.
  A missing or non-finite score means **"no opinion — not selectable"**: construction *drops* it
  (it is never coerced to `0.0`, so a real `0.0` score stays distinct from "no score").
- **Match the score to the policy.** `top_k_equal_weight` only *ranks* (the sign and scale of a
  score are irrelevant — only the order matters), whereas `equal_weight_positive` and
  `score_proportional_long` select on **sign** (only strictly-positive scores get weight) and the
  latter weights by **magnitude**. Produce scores whose sign/scale carry the meaning the policy reads.
- **Never look ahead.** The `view` contains bars only up to the current timestamp, and the engine
  enforces the `t→t+1` execution lag (`decision_lag_bars=1`). Do not re-implement or weaken this.
- `rebalance_frequency` is `"1d"` (daily) for now.
- **Prices: derive returns/momentum from `adj_close`, NEVER raw `close`.** Do **not** rationalize
  raw close/volume as "leak-safe" — raw is corporate-action **CONTAMINATED**: a split inside a
  lookback window fabricates fake momentum, and dividends distort returns. `close` is kept only for
  reference / notional sizing, not for return computation. The choice is a **three-way** trichotomy
  (raw contaminated / PIT-adjusted correct / future-restated leaky), not "raw safe vs adjusted
  leaky" — see `kb/principles/research-methodology.md` and `docs/contracts/bar-schema.md` as the
  source of truth.
- **Volume: no adjusted-volume column exists** — raw `volume` carries split discontinuities (the
  share count jumps at a split). If a signal uses volume across a window that may span a split,
  normalize / handle it deliberately and understand the limitation — do **not** treat raw volume as
  clean flow.
- **Authoring hygiene (lessons from #521):**
  - **Sort before positional indexing.** Call `.sort_index()` on the pivoted frame **before** any
    positional `.iloc[-1]` / `.iloc[-1 - lookback]` — never assume the view/pivot is already sorted.
  - **Guard the window count.** When averaging a trailing window (`.mean()` over the last N bars),
    require the expected number of **non-NaN** observations first — pandas `.mean()` skips NaN and
    silently shrinks a sparse window, so a thin window produces a mean off fewer bars than intended.
  - **Guard the denominator.** Guard zero / non-finite denominators (e.g. a volume-ratio surge) so a
    divide-by-zero doesn't emit `inf`/`NaN` scores.
- **Read the methodology AND the risk conventions before authoring.**
  `kb/principles/research-methodology.md` covers the leakage vectors no wall catches — full-sample
  fitting, target leakage inside a custom feature, the raw/PIT-adjusted/restated price-provenance
  trichotomy, and the `signal_panel` parity-vs-validity trap. `kb/principles/risk-conventions.md`
  covers weight-space risk — inverse-vol sizing, drawdown-based weight decay, conviction sizing, and
  the "R:R is the wrong yardstick" point. Note **where** that risk now lives: judgment that shapes
  the *score* (e.g. conviction sizing, slow-moving signals) stays in `signal`; judgment that shapes
  *weights from scores* (selection, weighting scheme, gross normalization) belongs in the
  **construction policy**. The rules here are the floor, not the whole job.

## Construction policies

The policy id in `CONFIG.construction` is resolved at load against `algua/portfolio/construction.py`.
The starter library (`construction.py::CONSTRUCTION_POLICIES`):

| Policy id | `construction_params` | What it does |
|---|---|---|
| `top_k_equal_weight` | `{"top_k": <positive int>}` | Hold the top-`top_k` names by score, equal weight `1/k`. Pure ranking — score sign/scale ignored. |
| `equal_weight_positive` | *(none)* | Equal-weight every name with a strictly-positive score. |
| `score_proportional_long` | *(none)* | Clip negatives to zero, weight the positives proportionally, normalized to gross `1.0`. |

`construction_params` are validated per-policy at load time — an unknown policy id, an unknown
param key, a missing required param (`top_k`), or a non-finite value all fail closed.

**Bespoke construction = add a named policy, not inline math.** If no library policy fits, add a new
`ConstructFn` to `algua/portfolio/construction.py` and register it in `_POLICIES` (additions-only;
don't edit an existing policy) — then reference it by id from your `CONFIG`. Keeping construction in
the library (not inside the strategy) is what lets the identity hash, the per-policy validation, and
the sweep `construction.<key>` namespace see it. A policy receives `(scores, view, params)`; `view`
is the same PIT bar-schema frame the signal saw (passed so a future vol-targeting policy can size
off prices with no contract change — the starter policies ignore it).

## The bars you receive

`view` is **long-format** (see `docs/contracts/bar-schema.md`): a tz-aware UTC `timestamp` index
(session date at midnight for daily) and columns `symbol, open, high, low, close, adj_close,
volume`. Pivot to wide when you need a matrix:

```python
wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
wide = wide.sort_index()  # never assume the pivot is already sorted before positional .iloc
```

## Available features — reuse before you reinvent (the factor catalogue)

`algua/features/` holds pure, composable **factors** (momentum, z-score, …). Before writing a
`signal`, discover what already exists instead of re-deriving it:

- `uv run algua factor list [--tag T] [--kind K]` — catalogued factors as JSON (name, summary,
  kind, tags, `data_needs`, `import_path`, `platform_supported`).
- `uv run algua factor show <name>` — one factor's full spec (summary, signature, `import_path`,
  data needs, docstring).

Prefer composing a catalogued factor over re-implementing it. Import it **at module top level via
its `import_path`** — e.g. `from algua.features.indicators import momentum`. Do **not** import a
factor lazily inside `signal` (a function-body or dynamic import escapes the live-gate `code_hash`
closure **and** the `algua factor dependents` lineage, so a later factor change would silently fail
to invalidate your strategy's backtest/live identity). A bespoke factor is fine when nothing fits —
keep it pure (`features/` imports nothing beyond `contracts`), and consider cataloguing it with
`@factor(...)` from `algua.features.catalogue` so the next strategy can reuse it.

After authoring, `uv run algua factor uses <strategy>` shows which catalogued factors your strategy
pulled in; `uv run algua factor dependents <factor>` shows every strategy a given factor reaches
(blast radius).

## Reuse vs. new logic

- **Parameter variant** of an existing idea: import its `signal` and define a new `CONFIG` —
  `from algua.strategies.momentum.cross_sectional_momentum import signal`. Vary `params`,
  `construction`, or `construction_params`.
- **New signal**: write a new `signal`. Look at
  `algua/strategies/momentum/cross_sectional_momentum.py` as the reference implementation (it pairs
  a trailing-return `signal` with `top_k_equal_weight`).

## Tuning construction in a sweep

`backtest sweep` tunes signal `params` and construction params through one grid. A grid key prefixed
**`construction.`** tunes `construction_params` (re-validated by the policy); any other key tunes a
signal `param` and **must already exist** in the base `params` (a typo'd key is rejected, never a
silent no-op). So `construction.top_k` sweeps the policy's `top_k`, while `lookback` sweeps the signal.

## Optional: `signal_panel` (advanced acceleration hook)

`signal` is the **canonical** definition — paper and live call it per bar, and every backtest path
must reproduce its per-bar output (a static-universe backtest may instead compute the whole scores
matrix via `signal_panel`, but only behind the parity guard below — never trusted blindly). For the
common "stateless" case (scores at `t` are a pure function of the trailing window ending at `t`), the
per-bar Python loop is slow. A module **MAY** additionally define a module-level:

```python
def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Decision-time (PRE-lag) SCORES for the WHOLE period at once, indexed timestamp × symbol."""
```

The loader detects it and binds it as `LoadedStrategy.signal_panel_fn`; the backtest engine then
computes the whole **scores** matrix in one vectorized shot, then applies the construction policy
per bar. Rules:

- It is **NOT a second signal.** It MUST produce scores that, once constructed, yield weights
  **identical** to running per-bar `construct(signal(view), view)` on the expanding view at each bar.
  The engine enforces this with a **fail-closed WEIGHT-level parity guard**: on every run it
  re-checks the panel path against the per-bar path on a bounded deterministic sample of bars, and
  **raises** on any disagreement (it never silently falls back).
- Same purity rules as `signal`: pure, pandas-only, no I/O, no look-ahead. Return PRE-lag decision
  scores (the engine applies the `t→t+1` shift centrally — do not shift yourself).
- Rows without enough history must be all-NaN, exactly as the per-bar `signal` returns empty there —
  construction drops NaN scores, so those bars are flat.
- It is **optional and advanced** — only add it when the per-bar loop is a measured bottleneck. Most
  strategies should ship `signal` alone. If you add a panel fn, you **must** add a parity test
  (`pd.testing.assert_frame_equal(..., check_exact=False, atol=WEIGHT_TOL, rtol=0)`) proving the two
  paths agree, mirroring `tests/test_fast_path.py`.
- **PIT (point-in-time universe) runs always use the per-bar loop**, even when a panel fn exists —
  the as-of masking can't be reproduced by a whole-period panel fn. The fast path is static-universe
  only. See `algua/strategies/momentum/cross_sectional_momentum.py` for a worked example.

## Optional: the fundamentals lane

To read point-in-time fundamentals, set `needs_fundamentals=True` in `CONFIG` and author the **3-arg**
form `signal(view, params, fundamentals) -> pd.Series` (the loader binds it as the active signal and
the engine injects the PIT-correct fundamentals frame per bar). Wiring today is **narrow**: only
`backtest run --fundamentals-snapshot <id>` supplies a fundamentals provider. **Everything else fails
closed** on a `needs_fundamentals` strategy — paper, live, and (critically) `walk-forward` and
`sweep`, which means a fundamentals strategy **cannot be promoted via `research promote` yet** (the
trading- and research-lane wiring is a #132 follow-up). Construction composes exactly the same — the
third arg only feeds the score.

## Discipline

- **Additions only.** Create a NEW file under `algua/strategies/<family>/` via
  `uv run algua strategy new <name> --family <slug>`. Do **not** edit or overwrite an existing
  strategy (especially the curated `cross_sectional_momentum.py`). Bespoke construction is also
  additions-only — a new policy in `construction.py`, never an edit to an existing one.
- Include a module-level `GENERATED_BY = "agent"` (after the imports) so machine-authored
  strategies are identifiable. Do not place it before `from __future__ import annotations`.
- After authoring, verify it loads and runs: `uv run algua backtest run <name> --demo` should emit
  metrics JSON, not an error. (A `needs_fundamentals` strategy needs a snapshot to run — verify it
  with `uv run algua backtest run <name> --demo --fundamentals-snapshot <id>` instead.)
