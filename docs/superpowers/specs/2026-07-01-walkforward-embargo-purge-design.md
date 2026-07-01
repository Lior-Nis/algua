# Walk-forward embargo / purge gap (issue #345)

## Problem

`walkforward._segment_bounds` makes the holdout the contiguous last `int(n*holdout_frac)`
bars with **gap = 0** between the last in-sample bar and the first holdout bar. Strategies carry
feature lookbacks up to ~60 bars (momentum `lookback=60`), so the in-sample window stats (which
drive sweep selection) and the holdout share overlapping feature/decision windows — a leakage door.

KB playbooks (Time-Series Leakage; Purged-Validation) require an embargo
`= max(feature_lookback, label_horizon)` with the assertion
`max(train_idx) < min(test_idx) - embargo`.

## Design (revised after GATE-1 / Codex)

### Carve the embargo from the TRAIN side

Holdout stays the last `int(n*holdout_frac)` bars, `[train_n, n)`, UNCHANGED. The embargo is the
last `E` training bars `[train_n - E, train_n)`. In-sample windows cover `[0, train_n - E)`.

**Invariant guaranteed (tightened per GATE-1):** no in-sample window sample index lies within
`E` bars of the holdout start — `max(train_idx) = train_n - E - 1 < train_n - E = min(holdout_idx) - E`.
We do NOT claim the holdout ignores pre-boundary bars: the holdout legitimately reads prior bars
as feature *history* (test set reading its own past inputs is not leakage). The embargo decorrelates
the in-sample *selection* statistics from the holdout by an `E`-bar index gap.

Because the holdout interval `[train_n, n)` is unchanged, `engine.holdout_window`
(the #192 single-use burn interval, computed independently as the last `holdout_n` bars) stays
byte-identical — the reservation/burn matching is untouched.

### Embargo size: `max(feature_lookback, decision_lag_bars)`

GATE-1 (HIGH): the `t -> t+1` execution shift means the first `decision_lag_bars` holdout returns
are driven by decisions made on the last in-sample bars, so the label-horizon term is real, not
dominated. Embargo = `max(feature_lookback, decision_lag_bars)`.

- Add `feature_lookback: int | None = None` to `StrategyConfig`. **`None` = undeclared**;
  an explicit value (including `0` for a strategy with no rolling feature window, e.g. a
  latest-value fundamentals tilt) = declared. This sentinel distinguishes "forgot to declare"
  from "declared zero" (GATE-1 CRITICAL: a `0` default would silently under-embargo the gated path).
- `decision_lag_bars` comes from the existing `ExecutionContract` (default 1, already `>= 1`).
- `walk_forward(..., embargo: int | None = None)`: explicit arg wins (validated `>= 0`); else if
  `config.feature_lookback is not None` -> `embargo = max(feature_lookback, decision_lag_bars)`;
  else (undeclared) -> `embargo = 0` (preserves legacy behavior for human-exploratory /
  synthetic-fixture runs that never declared a lookback).

### `config_hash` includes `feature_lookback` (GATE-1 HIGH)

`feature_lookback` is behavior-affecting (it changes the carved windows) but is neither a `params`
nor an `execution` field, so the hand-assembled `config_hash` would miss it. Add it to the hashed
payload + a regression test (different lookback => different hash) so it is part of artifact
identity (gate evidence, holdout reservation, live approval).

### Agent gated path fails closed on an undeclared lookback (GATE-1 CRITICAL)

In `promotion_preflight` (pre-peek): if `actor is AGENT` and the loaded strategy declares
`feature_lookback is None`, refuse. Forces every agent-promoted strategy to consciously declare its
lookback (it may declare `0`). Only enforced when the strategy is a bundled module (`load_strategy`
resolves); synthetic-name test strategies are unaffected (as today). This is the protected-file
change — the PR stays OPEN for human merge.

### `_segment_bounds(n, windows, holdout_frac, embargo=0)`

Pure, default `embargo=0` (existing unit tests unchanged). `embargo < 0` -> BacktestError
(GATE-1 CRITICAL: negative embargo reopens overlap and the naive `gap == embargo` assert would
pass). `usable_train = train_n - embargo`; split `[0, usable_train)` into `windows`. The
insufficient-bars guard uses `usable_train` and names the embargo. Postcondition asserts
`embargo >= 0` AND the carved gap `holdout[0] - bounds[-1][1] == embargo`.

### Auditability

- `WalkForwardResult.embargo: int = 0` (default keeps direct test fixtures constructible) +
  `to_dict` + mlflow `log_walk_forward` param.
- `--embargo` override flag on exploratory `backtest walk-forward` (validated `>= 0`). The gated
  agent paths (`research promote`, `sweep`) derive embargo from config — no lowering knob.

## Bundled strategies (declare `feature_lookback`)

- `cross_sectional_momentum` -> 60 (matches its `params.lookback`).
- `fundamentals_earnings_tilt` -> 0 (latest-value, no rolling price window; embargo = lag = 1).
- `news_coverage_tilt` -> 5 (its `window_days`).
- `strategy new` template -> 60 (matches the template `lookback`).

## Deferred (documented, not in this PR)

- **Sweep grid > declared lookback** (GATE-1 HIGH): a grid sweeping `params.lookback` above the
  declared `feature_lookback` under-embargoes those combos. Auto-enforcement needs a fragile
  param-name convention or static analysis; for now the author must declare `feature_lookback >=`
  the largest lookback they will sweep (same "declare the contract honestly" model as `universe`).
- **Embargo in the durable gate `decision_json`** (GATE-1 MEDIUM): recording the embargo next to
  `holdout_frac` in the registry gate row needs a protected `gates.py`/`store.py` change; the
  enforcement (preflight) + `WalkForwardResult.embargo` + MLflow cover protection and observability.

## Files

`algua/backtest/walkforward.py`, `algua/strategies/base.py` (field + `config_hash`),
`algua/registry/promotion.py` (agent preflight guard — protected),
`algua/strategies/momentum/cross_sectional_momentum.py`,
`algua/strategies/fundamentals/fundamentals_earnings_tilt.py`,
`algua/strategies/news/news_coverage_tilt.py`, `algua/cli/strategy_cmd.py` (template),
`algua/cli/backtest_cmd.py` (`--embargo`), `algua/tracking/mlflow_tracker.py`, tests.
