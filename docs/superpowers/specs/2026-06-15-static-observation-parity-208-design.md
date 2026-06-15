# Static-mode observation parity (issue #208)

**Date:** 2026-06-15
**Status:** design — GATE-1 pending
**Touches:** CODEOWNERS-protected `algua/backtest/engine.py` → human design call.

## Problem

#179 / PR #207 closed the out-of-universe **weight** gap: a nonzero target for a symbol
outside the operating universe is now rejected identically across static backtest, PIT, and
live (`validate_decision_weights(..., allowed_symbols=...)`).

A related **observation** gap remains in static mode. `_decision_weights` / `_fast_weights`
build the strategy-visible `view` / `signal_panel` from the FULL fetched `bars` / `adj`. If a
misbehaving provider returns bars for an **undeclared** symbol (one not in `strategy.universe`),
that symbol's DATA enters the view, so the strategy can compute its (in-universe) signals using
out-of-universe data — a static-vs-live decision drift, since live only ever feeds the declared
universe.

The static operating universe is `set(strategy.universe) & set(adj.columns)`; #179 uses it for
weight validation but does not project the observable data to it. The empty-intersection case
already fails closed (in `simulate` and `verify_signal_panel_parity`); this is the **partial**
case (some declared + some undeclared columns).

This is triggered only by a provider returning undeclared symbols — a data-layer contract
violation. `_fetch_symbols` requests exactly `strategy.universe` in static mode, so a compliant
provider never trips it. PIT mode already filters the view to as-of members; live only feeds the
declared universe. So this is a static-mode defense-in-depth fix.

## Approach

Add ONE private helper to `algua/backtest/engine.py` and call it from the two static-mode sites
that build the view from full fetched data. Both sites already carry a **verbatim-identical**
empty-universe guard, which the helper absorbs (removing the duplication and guaranteeing the two
can't drift).

```python
def _static_operating_view(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project the strategy-visible STATIC view to the operating universe (declared AND available)
    so a misbehaving provider's undeclared symbols never enter the loop view, the fast-path
    signal_panel, the weights/grid, or the fundamentals/news sidecars (observation parity, #208).

    Fails closed when a declared universe has no available column (existing #179 behavior). No-op
    when the strategy declares no universe (already forced flat — every weight rejected with
    allowed=empty) or for a compliant provider (adj.columns subset of strategy.universe)."""
    if not strategy.universe:
        return bars, adj
    operating = sorted(set(strategy.universe) & set(adj.columns))
    if not operating:
        raise BacktestError(
            f"no fetched price data for any symbol in strategy {strategy.name!r} declared "
            f"universe {sorted(strategy.universe)} (fetched columns: "
            f"{sorted(map(str, adj.columns))})"
        )
    return bars[bars["symbol"].isin(operating)], adj.loc[:, operating]
```

- `adj.loc[:, operating]` — label column-select, order-stable, preserves `columns.name`;
  `operating ⊆ adj.columns` by construction so no NaN is introduced and no reindex fill occurs.
- `bars` is long-format (a `symbol` column); filter by membership.
- `operating` is non-empty whenever we project, so the projected `bars` is non-empty.

### Call sites (static mode only)

- **`simulate`** — replace the `if universe_by_date is None:` empty-guard block (currently lines
  ~567–574) with `bars, adj = _static_operating_view(strategy, bars, adj)`. Stays gated on
  `universe_by_date is None` (PIT keeps its per-bar as-of mask; no projection there). Sidecars are
  fetched *after* this point, and the loop's `allowed = set(columns)` (= projected `adj.columns`)
  already filters the as-of fundamentals/news frame — so projecting `adj` closes the sidecar leak
  with no extra code.
- **`verify_signal_panel_parity`** — replace its own empty-guard block (currently lines ~428–433)
  with the same call. This function fetches its own bars and runs both `_fast_weights` and
  `_decision_weights` in static mode, so it must project identically.

### Why one chokepoint covers everything

- Loop `view` derives from the projected `bars` (`bars_sorted.iloc[:stop]`).
- Fast-path `signal_panel(bars)` and per-bar `view_t` derive from the projected `bars`.
- The weights matrix and the vectorbt grid (`close=adj`, `size=weights_eff`) build on the
  projected `adj.columns`.
- Fundamentals/news as-of frames are filtered by `allowed = set(adj.columns)` in the loop.

## Scope / non-goals

- **Static mode only.** PIT (`universe_by_date is not None`) already masks per-bar to as-of
  members; live only feeds `strategy.universe`. No change to either.
- **Empty intersection** still fails closed (unchanged message).
- **Empty declared universe** (`universe == []`) is NOT projected — preserves today's behavior
  (full panel visible but every nonzero weight already rejected with `allowed=∅`, i.e. a
  forced-flat strategy). Matches the existing guard condition `if strategy.universe and not ...`.
- **Compliant provider** (`adj.columns ⊆ strategy.universe`) → `operating == adj.columns` → pure
  no-op (acceptance criterion).

## Behavior change

Only for a **non-compliant provider** (the bug case): undeclared columns are dropped from the
strategy-visible view, the panel, the sidecars, the weights matrix, and the simulated portfolio.
This is the intended defense — undeclared symbols can no longer influence in-universe decisions.

## Testing (additive)

Against a spy strategy whose `signal` / `signal_panel` / sidecar callback records the symbols it
is shown:

1. **Loop path** — provider returns a declared symbol + an undeclared one → the recorded `view`
   never contains the undeclared symbol; resulting weights cover only declared columns.
2. **Fast path** — a `signal_panel`-bearing strategy → the panel fn receives only declared
   symbols (capture the `bars` it is handed).
3. **`verify_signal_panel_parity`** — provider returns an undeclared symbol → the panel sees only
   declared symbols; parity still holds.
4. **Sidecars** — a `needs_fundamentals` and a `needs_news` strategy whose provider returns rows
   for an undeclared symbol → the undeclared rows are absent from the as-of frame the signal sees.
5. **No-op** — compliant provider → weights/columns identical to the pre-change result (same
   shape, same values).
6. **Fail-closed** — declared universe but provider returns only undeclared symbols →
   `BacktestError` (empty-intersection guard, unchanged). Empty declared universe → not projected,
   no crash, forced flat.

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

## Acceptance (from the issue)

- A static-mode strategy whose provider returns a declared symbol plus an undeclared one never
  sees the undeclared symbol in its `view` / panel / sidecars.
- No behavior change for compliant providers (`adj.columns ⊆ strategy.universe`).
- `pytest && ruff check . && mypy algua && lint-imports` green.
