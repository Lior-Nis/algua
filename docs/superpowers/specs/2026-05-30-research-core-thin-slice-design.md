# Research Core (Thin Vertical Slice) — Design

**Date:** 2026-05-30
**Sub-project:** 3 (research core), first slice. **Branch:** `research-core`.
**Status:** Approved (design); plan to follow.

## 1. Goal

The smallest end-to-end research capability: an agent (or human) **authors one strategy,
backtests it on data conforming to the frozen bar-schema, gets structured JSON metrics, and the
registry advances `idea → backtested`** — proving the data↔research seam and the `t→t+1`
anti-look-ahead rule. Walk-forward, parameter sweeps, and MLflow tracking are explicitly deferred
to the next slice.

## 2. Constraints

- **Lane isolation:** this lane owns `algua/features|strategies|backtest/*` and its own CLI
  modules. It must NOT edit `algua/data/*` (Codex), `algua/registry/*` (called, not edited), or
  `algua/contracts/types.py` (frozen shared interface).
- **Data seam:** consumes bars only through the `DataProvider` protocol and the shape frozen in
  `docs/contracts/bar-schema.md`. Built against a synthetic provider until Codex's real provider
  is integrated.
- **Decision model:** cross-sectional, per-bar. The strategy is called once per rebalance bar `t`
  with the point-in-time view (rows with `timestamp <= t`) and returns target weights indexed by
  symbol. This is the same function live trading will call — parity by construction.

## 3. Module layout (new)

```
algua/
├── features/
│   └── indicators.py      # pure helpers: returns, momentum, zscore, rolling stats
├── strategies/
│   ├── base.py            # StrategyConfig (pydantic) + LoadedStrategy adapter
│   ├── loader.py          # load a strategy module by name -> LoadedStrategy
│   └── examples/cross_sectional_momentum.py   # bundled first strategy
├── backtest/
│   ├── engine.py          # run(strategy, provider, period) -> BacktestResult
│   ├── metrics.py         # Sharpe, CAGR, max-DD, turnover, vol, total return
│   ├── result.py          # BacktestResult dataclass + to_dict() (JSON)
│   └── _sample.py         # dev-only synthetic DataProvider (marked), for demos/tests
└── cli/
    ├── strategy_cmd.py    # `algua strategy new|list`
    └── backtest_cmd.py    # `algua backtest run`
```

## 4. Strategy authoring

A strategy is one module exposing:
- `CONFIG: StrategyConfig` — `name`, `universe: list[str]`, `execution: ExecutionContract`, and a
  typed `params` block (pydantic model or dict with declared keys).
- `def target_weights(view: pd.DataFrame, params) -> pd.Series` — **pure, cross-sectional**. Given
  the point-in-time view (bar-schema rows up to `t` for the universe), return target weights
  indexed by symbol with `sum(|w|) <= execution.max_gross_exposure`.

`strategies/loader.py` binds `CONFIG` + the function into a `LoadedStrategy` that satisfies the
existing `Strategy` protocol (`.name`, `.execution`, `.target_weights`) — honoring
`contracts/types.py` without editing it. `algua strategy new <name>` scaffolds this module from a
template containing a working momentum stub.

**Bundled first strategy — cross-sectional momentum:** rank the universe by trailing-`lookback`
return, hold the top-`k` equal-weight, rebalance on the contract cadence. Embodies "ride the
strongest names."

## 5. The backtest engine (`engine.py`)

Inputs: a `LoadedStrategy`, a `DataProvider`, a period `(start, end)`. Timeframe from the
execution contract.

1. **Fetch:** `provider.get_bars(universe, start, end, timeframe)` → bar-schema long frame.
2. **Rebalance schedule:** derive rebalance timestamps from the contract cadence (market calendar
   for `"1d"`).
3. **Per-bar decision loop:** for each rebalance `t`, slice the point-in-time view
   (`timestamp <= t`), call `strategy.target_weights(view, params)`; collect into a weights matrix
   `W` (time × symbol). The loop is the parity guarantee — the strategy never sees beyond `t`.
4. **Enforce `t→t+1`:** the engine shifts `W` forward by `execution.decision_lag_bars` before
   simulating, so a decision at `t` fills no earlier than `t+1`. Anti-look-ahead is enforced
   centrally, never trusted to the author.
5. **Simulate:** `vbt.Portfolio.from_orders(close=adj_close_wide, size=W_shifted,
   size_type="targetpercent", fees=<commission>, slippage=<bps>)`. vectorbt performs only the
   portfolio accounting; strategy logic stays in the reusable per-bar function. Returns use
   `adj_close` per the schema.
6. **Metrics → `BacktestResult`.**

**Fees/slippage:** a simple model for the slice — fixed commission + fixed bps slippage. The
slippage stress grid is deferred.

**Engine purity:** the engine imports only `contracts`, `calendar`, `features`, and vectorbt. It
does NOT import `registry`, `cli`, or `data` — it depends on the `DataProvider` protocol, so it is
decoupled from Codex's lane and trivially testable. Registry wiring lives only in the CLI layer.

## 6. Results & reproducibility (`result.py`, `metrics.py`)

`BacktestResult.to_dict()` emits a **stable JSON schema** (not a raw vectorbt stats dump):
- Metrics: `total_return`, `cagr`, `ann_volatility`, `sharpe`, `max_drawdown`, `turnover`,
  `avg_gross_exposure`, `n_rebalances`.
- **Reproducibility stamps:** strategy `name`, `config_hash` (hash of strategy code + params +
  universe + execution), `data_source`/`snapshot_id` (when available), `seed` (synthetic),
  `timeframe`, `period`. These carry forward the reproducibility principle even before MLflow.

## 7. Synthetic provider (`backtest/_sample.py`)

A deterministic, seeded `DataProvider` emitting **bar-schema-conformant** data for a small
universe, enabling `algua backtest run --demo` and the full test suite to run without Codex's data
layer. Clearly marked dev-only. Real-data wiring is the integration step after merging `main`.

## 8. CLI

- `algua strategy new <name>` — scaffold a strategy module; emit created path as JSON.
- `algua strategy list` — list available strategies.
- `algua backtest run <name> [--start D --end D] [--demo] [--register]` — load strategy, get bars
  (`--demo` → synthetic), run engine, emit `BacktestResult` JSON.
  - `--register`: auto-add the strategy at `idea` if missing, then transition `idea → backtested`
    via `registry.store.transition` (actor `agent`, reason summarizing key metrics, recording the
    `config_hash`). Full agent flow: `algua backtest run momentum --demo --register`.

Command modules self-register on import (existing pattern); `cli/main.py` gains two import lines.
(At merge with `main`, these combine trivially with Codex's `data_cmd` import.)

## 9. Testing

- `features/indicators` — pure unit tests (momentum/zscore correctness).
- Example strategy — engineered fixture (≈3 symbols, known returns) → expected top-`k` weights.
- Engine:
  - **`t→t+1` test:** a deliberately "cheating" strategy keyed on the current bar still cannot
    realize same-bar P&L (the central shift defeats it).
  - **End-to-end determinism:** same seed → identical `BacktestResult`.
  - **Schema conformance:** the synthetic provider output validates against the bar schema.
- CLI — `strategy new` scaffolds a loadable module; `backtest run --demo` returns valid metrics
  JSON with the documented keys; `--register` advances the registry to `backtested`.
- **Quality gate stays green:** `pytest`, `ruff`, `mypy`, `lint-imports`.

**New import-linter contracts:** `features/` stays pure (no algua imports beyond `contracts`);
`backtest/` may import `contracts`, `calendar`, `features` but NOT `cli`, `registry`, or `data`.

## 10. Deferred (next research slices)

Walk-forward validation, parameter-sweep runner, MLflow tracking + run taxonomy, slippage stress
grid, statistical promotion gates (untouched holdout, search-breadth penalty), and the autonomous
research loop. Real-data integration (swap synthetic provider for Codex's `get_bars`) happens once
this slice is green and `main` is merged in.
