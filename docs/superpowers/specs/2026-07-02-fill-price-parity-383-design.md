# Pin the fill-price basis in ExecutionContract (#383)

**Date:** 2026-07-02
**Issue:** #383 — ExecutionContract pins t→t+1 but NOT the fill price; backtest fills at `adj_close`, paper/live fill at the next-bar open → broken backtest↔live semantic parity.
**Lane:** qf. **Severity:** high.

## Problem

The `ExecutionContract` pins the decision→execution *lag* (`decision_lag_bars >= 1`, the t→t+1 anti-look-ahead rule) but NOT the intra-bar *fill reference price*. As a result the two execution paths price the SAME lagged decision differently:

- **Backtest** (`engine.simulate`): `vbt.Portfolio.from_orders(close=adj_exec, ...)` uses the `close=` series as the fill price → fills at **`adj_close[t+1]`**.
- **Paper/live** (`paper_loop.run_paper` → `SimBroker.fill_pending(opens.loc[t_next], ...)`): fills at the **raw `open[t+1]`**.

For a next-bar-return signal, close-fill captures none of day t+1's move while open-fill captures the whole day — a systematically unmodeled intra-day bar of drift that can correlate with signal direction. The frictionless backtest that gates candidate promotion and the paper loop that mints the forward-test certificate therefore measure different execution economics; the gap widens with turnover (ML/DL strategies most exposed).

## Goal

Pin the intra-bar fill reference in the `ExecutionContract` and thread it into BOTH paths so a single pinned assumption drives every fill. Default to next-bar **OPEN** (what the sim/paper loop already does, and the anti-look-ahead-correct choice: a decision on the close of bar t can realistically execute at the open of bar t+1). Scope note (accepted GATE-1 narrowing): this pins the **sim/backtest** fill *reference price*. The real Alpaca live broker sends day market orders (`type=market, tif=day`), which are open-ish only when the loop is scheduled pre-open — changing the live order type to an explicit MOO is out of scope; this issue does not claim to enforce a live MOO order, only to align the backtest fill *basis* with the sim/paper loop's next-bar-open reference. This is a *parity* fix — it deliberately shifts backtest numbers to match live economics; golden tests are updated to the correct new values, no gate is weakened.

## Design

### 1. Contract field — `ExecutionContract.fill_price`

Add a field to `algua/contracts/types.py`:

```python
fill_price: str = "open"   # Literal["open", "close"]; the intra-bar execution reference
```

- Placed as the LAST field (after `capacity`) — the codebase builds `ExecutionContract` with keyword args, so appending is safe; `asdict`-based `config_hash` folds it in automatically (a change invalidates a prior live approval — correct).
- `__post_init__` validation: `fill_price` must be exactly `"open"` or `"close"` (fail closed on anything else; reject non-str). No `"vwap"` — YAGNI (no vwap column in the bar schema, live cannot fill vwap trivially; deferred).
- **Default `"open"`**: matches the live/paper loop's existing behavior and is the semantically correct anti-look-ahead reference. `"close"` is retained as an explicit **legacy adjusted-close backtest mode** (reproduces today's pre-#383 backtest numbers) — it is NOT a claim that any live broker fills at adjusted close; in the sim/paper loop `"close"` simply fills against the adj_close grid as the in-sim analog. Honest naming: `"close"` = "value on the adj_close basis", not "market-on-close live order".

### 2. Shared, pure resolver — one source of truth

Both the engine and the paper loop already import `algua.contracts`. The import-linter forbids the backtest engine from importing the execution/live lanes, so the shared helper lives in **`algua/contracts/types.py`** (the one module both sides can import):

```python
def fill_reference_column(execution: ExecutionContract) -> str:
    """The bar-schema column both paths fill against for this contract:
    'open' (raw open, the default) or 'adj_close'. The engine maps the raw
    reference into its ADJUSTED grid (see engine); the live/paper loop fills
    at the raw column directly. Single source of truth so the two paths can
    never silently pick different intra-bar references."""
```

Returns the *raw schema column name* the fill references: `"open"` for `fill_price=="open"`, `"adj_close"` for `fill_price=="close"` (the backtest already treats adj_close as the close basis; paper's `closes` grid is built from `adj_close`). Pure — no I/O, no cross-module import.

### 3. Backtest engine — build the execution-price grid from the contract

The backtest runs on the **adjusted** grid for multi-year corporate-action correctness, so it cannot fill at the *raw* open. It fills at the **adjusted open**, derived per bar from the same adjustment ratio that maps raw→adjusted close:

```
adj_open[t] = open[t] * (adj_close[t] / close[t])
```

Splits/dividends scale the whole bar uniformly, so `adj_open` is to `adj_close` exactly what raw `open` is to raw `close` — the SAME intra-bar reference point, each expressed in its own frame. This is the true semantic parity: the live loop fills at raw open in a locally-unadjusted day; the backtest fills at adj_open in the adjusted frame; day-locally adj≈raw so they represent the identical economic fill.

Implementation in `engine.simulate`:
- New pure helper `adj_open_grid(bars)` → `open * adj_close / close` pivoted to the (timestamp × symbol) grid, mirroring `adj_grid`. **Validation (fail-safe cells):** where `close <= 0`, `close` NaN, `open <= 0`, `open` NaN, or the ratio is non-finite (`inf`/`-inf`/NaN), the `adj_open` cell is set to NaN — an untradeable bar, handled downstream exactly like a missing bar (the sim/`from_orders` treats a NaN price as a no-fill). Tested for zero, negative, `inf`, `-inf`, NaN.
- Choose the ordinary execution grid by the contract: `exec_grid = adj_open_grid(bars) if fill_price=="open" else adj` (the existing `adj` = adj_close grid).
- **Delisting terminal valuation stays on `adj_close` (accepted GATE-1 fix #2).** `apply_delisting_exits` gets a new keyword `terminal_price_grid=adj` (always the adj_close grid) used SOLELY for the `assume_terminal_last_close` human-only fallback (`col.loc[T]` → `terminal_price_grid.loc[T, c]`), so that relaxation realizes at the last *close*, never the last *open*, regardless of `fill_price`. The record-backed terminal price is already an explicit `adj_close`-unit value and is unaffected. The ordinary (non-terminal) fill cells come from `exec_grid`. This keeps terminal proceeds economically correct (a delisting is a close-of-book event, not an open fill).
- Pass `exec_grid` (not `adj`) as the price grid into `apply_delisting_exits(...)` → `from_orders(close=exec_grid_exec, ...)`.
- **The simulation date-index is unchanged.** `adj_open_grid` shares `adj_grid`'s index/columns, so `holdout_window`, the #192 single-use holdout identity, PIT masking, and the returns index are all untouched — only the fill *price values* change.

**#325 composition (slippage/fees):** `from_orders` applies `fees` on `|trade notional|` and `slippage` as an adverse per-side move on the fill price. With `close=adj_open_exec`, both frictions now apply to the OPEN base price — the live basis. No double count, no reintroduced look-ahead (the grid is still the t→t+1-shifted `weights_exec`; only the *reference price* moved from close to open, both on bar t+1). The sim/live fee-slippage model itself is out of scope here (#325 owns the backtest cost model; the sim broker's no-friction fill is a separate concern) — this issue pins only the fill *reference price*.

### 4. Paper/live — honor the contract

`paper_loop.run_paper` already fills at the next-bar open. Make that an explicit consequence of the contract rather than a hardcoded assumption:
- Resolve the fill column via `fill_reference_column(strategy.execution)` and fill `broker.fill_pending(fill_grid.loc[t_next], ...)` where `fill_grid` is `opens` for `"open"` / `closes` (adj_close) for `"close"`. Default path (`"open"`) is byte-unchanged from today.
- `SimBroker.fill_pending` is unchanged — it fills against whatever price series it is handed; the *choice* of series is the loop's, driven by the contract. Its docstring is updated from "at the next bar's open" to "at the next bar's contract-pinned reference price".

### 5. Load-time parity assertion

A single pinned assumption must resolve identically in both paths. `fill_reference_column` is the ONE function both the engine grid-selection and the paper loop consult for the intra-bar reference, so neither side can silently pick a different reference. (GATE-1 caveat #6 accepted: the engine additionally maps `"open"` into its *adjusted* frame via `adj_open_grid` while paper fills the raw `open` column — the helper guarantees both target the same *intra-bar reference*, and the parity test below proves the resulting fill prices coincide on an adj==raw panel; the helper alone does not make the adjusted-frame mapping self-proving, so the test is load-bearing.) A parity test asserts:
- backtest with `fill_price="open"` fills at adj_open, `"close"` fills at adj_close;
- the paper loop fills at raw open (`"open"`) / adj_close (`"close"`) via the same resolver;
- on a split/dividend-free synthetic panel (adj==raw), the per-bar fill *price* the two paths use is identical → true end-to-end parity, closing the gap the issue names.

## Files touched

- `algua/contracts/types.py` — `fill_price` field + validation + `fill_reference_column`.
- `algua/backtest/engine.py` (**CODEOWNERS-protected** → PR stays open for human merge) — `adj_open_grid` + contract-driven exec grid.
- `algua/backtest/delisting.py` — `terminal_price_grid` keyword so the `assume_terminal_last_close` fallback stays on adj_close regardless of the fill grid.
- `algua/execution/sim_broker.py` — docstring only (fills against the handed series).
- `algua/live/paper_loop.py` — resolve fill grid via the contract.
- Golden/parity tests — update backtest numbers that shift from close→open fill (to correct new values, assertions kept); add the fill-price parity test.

## Non-goals / deferred

- `"vwap"` fill (no vwap column; live can't fill it trivially).
- Per-symbol / per-order fill overrides — one contract-level reference is enough.
- Modeling the open→close intra-bar path or partial-day fills.
- Changing the Alpaca live broker's real order type (it sends day market orders, open-ish only when the loop is scheduled pre-open); only the sim/backtest fill *reference* is pinned here.

## Testing

- `fill_price` validation (accept open/close, reject junk/non-str).
- `adj_open_grid` correctness incl. a split bar (ratio scales the open) and a `close<=0` NaN cell.
- Backtest fills at adj_open for `"open"`, adj_close for `"close"` (numbers differ, both finite).
- End-to-end backtest↔paper fill-price parity on an adj==raw panel.
- `config_hash` changes when `fill_price` changes (identity fold-in).
- Existing decision-parity + golden backtests updated to the new open-fill numbers.
- Full gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
