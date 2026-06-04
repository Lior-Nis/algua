# Live Drawdown Breaker (Sub-project 5, Slice B2b-2 core)

**Date:** 2026-06-03
**Status:** Accepted (pending implementation)
**Scope:** An automatic equity-based circuit breaker for live (paper) trading. On each
`trade-live` tick, if the strategy's drawdown from its peak equity exceeds an operator policy
limit, trip the kill-switch and auto-flatten — *before* deciding or submitting any orders.

---

## 1. Context & non-goals

B2a's live tick submits real orders; B2b-1 made a kill (manual `paper flatten` or a per-tick
`RiskBreach` on gross/long-only) both halt **and** flatten. The remaining gap: a strategy bleeding
money *within* its weight limits never trips anything — nothing watches equity across ticks. This
slice adds the missing **automatic** trigger.

The pure check already exists: `check_drawdown(equity, peak, max_drawdown)` in
`algua/risk/limits.py` raises `RiskBreach(kind="drawdown")` (slice A, used by `run_paper` over an
in-memory sim equity curve). For live, the only new problem is **persisting the peak across ticks**
(each `trade-live` invocation is a separate process), then running that same check as a pre-flight
gate.

**Non-goals (later B2b slices / sub-project 6):** the global ("halt-all") kill-switch; intra-session
order idempotency (`client_order_id`); the per-tick account/positions snapshot optimization;
`paper → live`. **Out of scope:** a strategy-scoped equity attribution that splits one Alpaca
account across multiple strategies (see §3).

---

## 2. Design decisions (settled in brainstorming)

- **Peak reference = lifetime high-water mark.** A monotonically-rising peak per strategy; drawdown
  is the decline from the all-time high. Standard circuit-breaker semantics; matches
  `check_drawdown` (which already assumes a monotonic `peak`).
- **Reset on resume-after-trip.** `paper resume` clears the strategy's stored peak, so a
  flattened-and-resumed strategy re-bases its high-water mark to current equity rather than
  instantly re-tripping against the old (pre-loss) high.
- **Equity source = account equity, one-strategy-per-account.** Use `broker.account().equity`
  directly as the strategy's equity. The operating model is **one live strategy per Alpaca
  account** (already implicit in `trade-live` reading account equity and B2b-1 scoping flatten to
  the universe). Revisit only if multi-strategy-per-account is ever built.
- **Limit source = `--max-drawdown` on `trade-live`.** Operator policy, per run, mirroring
  `run_paper` exactly. Default `1.0` (disabled / opt-in).

---

## 3. Components

| Module | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | New table `live_equity_peak(strategy PK, peak_equity, updated_ts)`; bump `SCHEMA_VERSION` 3 → 4. |
| `algua/registry/store.py` (modify) | `get_equity_peak(conn, name) -> float \| None`; `set_equity_peak(conn, name, peak)` (UPSERT); `clear_equity_peak(conn, name)`. |
| `algua/cli/paper_cmd.py` (modify) | `trade-live` gains `--max-drawdown` + the pre-flight breaker (read peak → read equity → `check_drawdown` → run tick); `paper resume` clears the peak. |

**Reuses:** `check_drawdown` and `RiskBreach` (unchanged), the existing `except RiskBreach`
trip→`cancel_open_orders`→`close_positions(universe)` handler (B2b-1), `kill_switch`. **`run_tick`
is unchanged** — the breaker is a pre-flight gate in the CLI, not part of the decide-and-submit
engine. No new import contract (`cli`/`registry`/`risk` already bound).

---

## 4. Persistence

```sql
CREATE TABLE IF NOT EXISTS live_equity_peak (
    strategy    TEXT PRIMARY KEY,
    peak_equity REAL NOT NULL,
    updated_ts  TEXT NOT NULL
);
```
- `get_equity_peak` returns the stored peak or `None` (no row yet → first tick).
- `set_equity_peak` UPSERTs (`INSERT ... ON CONFLICT(strategy) DO UPDATE`).
- `clear_equity_peak` deletes the row (called by `paper resume`).
- Schema bump 3 → 4; the foundation test asserting `user_version` must be updated to 4.

---

## 5. Control flow — `trade-live` pre-flight breaker

`trade-live` gains `--max-drawdown FLOAT` (default `1.0` = disabled, identical to `run_paper`).
After the stage / kill-switch gates and building `broker`/`provider`:

```python
peak = store.get_equity_peak(conn, name)            # None on the first tick
try:
    equity   = broker.account().equity
    new_peak = max(peak, equity) if peak is not None else equity
    check_drawdown(equity, new_peak, max_drawdown)  # pre-flight circuit breaker
    result = run_tick(strategy, broker, provider, _utc(start), _utc(end))
except RiskBreach as exc:
    # EXISTING handler (B2b-1): trip + audit + cancel_open_orders + close_positions(strategy.universe)
    # + emit {ok:false, kind, kill_switch:"tripped", liquidation_submitted, [flatten_error]}
    ...
    raise typer.Exit(1) from exc
store.set_equity_peak(conn, name, new_peak)         # persist HWM — success path only
# ... persist submitted orders, audit, emit success payload
```

- **One `except RiskBreach`** now covers both the drawdown breaker and `run_tick`'s existing
  gross/long-only breaches. A drawdown breach emits `kind:"drawdown"` and auto-flattens via the
  same path.
- **First tick:** `peak is None` → `new_peak = equity` → `check_drawdown(equity, equity, dd)` never
  breaches → peak persisted.
- **New high:** `new_peak = equity` (≥ old peak) → drawdown 0 → higher peak persisted.
- **Disabled (`--max-drawdown 1.0`):** `check_drawdown` returns immediately (its `>= 1.0` guard);
  the peak is still tracked, so enabling the breaker later already has history.
- **Breach path** halts the strategy, so peak is intentionally **not** updated on breach (the
  success-path `set_equity_peak` is unreachable after `typer.Exit`).

## 6. Reset on resume

`paper resume <name>` (which resets the kill-switch) additionally calls
`store.clear_equity_peak(conn, name)` in the same transaction. The next tick sees `peak is None`
and re-bases the high-water mark to current equity.

---

## 7. Testing

- **Store** — `set_equity_peak` then `get_equity_peak` round-trips; UPSERT overwrites; `get` on an
  unknown strategy returns `None`; `clear_equity_peak` removes the row.
- **`trade-live` breaker (mocked broker + `run_tick`)** — equity below `peak*(1-dd)` →
  `RiskBreach(kind="drawdown")` → switch tripped + `close_positions` called + payload
  `kind:"drawdown"`, `liquidation_submitted` present, exit 1; `run_tick` **not** invoked.
- **New high** — a tick with equity above the stored peak persists the higher peak (no breach).
- **Disabled** — `--max-drawdown 1.0` never breaches even when equity < peak.
- **Resume clears peak** — after `paper resume`, `get_equity_peak` returns `None`.
- **Schema** — `migrate` yields `user_version == 4`; the foundation version-assertion test updated.
- **Gate** — `pytest · ruff · mypy · lint-imports` (contracts stay `10 kept, 0 broken`).
- **Live acceptance (manual, documented, NOT CI)** — run `trade-live` with a tight
  `--max-drawdown` against the paper account after a small simulated loss → breaker trips +
  flattens; `paper resume` clears the peak.

---

## 8. Consequences

- The kill→flat loop is now **automatic**: a live strategy that bleeds past the policy limit halts
  and flattens itself on the next tick, with no human in the loop — completing the safety intent of
  B2b.
- The breaker is a thin pre-flight gate reusing the existing pure check and the B2b-1 flatten
  handler; `run_tick` stays a focused decide-and-submit engine.
- High-water-mark persistence is the first cross-tick live state in the registry (schema v4),
  establishing the pattern the global kill-switch and snapshot work (later B2b) will build on.
- The one-strategy-per-account assumption is now documented; if multi-strategy-per-account is ever
  built, per-strategy equity attribution becomes the follow-up.
