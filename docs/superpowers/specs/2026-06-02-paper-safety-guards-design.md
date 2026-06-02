# Paper Safety Guards (Sub-project 5, Slice A)

**Date:** 2026-06-02
**Status:** Accepted (pending implementation)
**Scope:** Add the safety layer the spec wants *validated in paper* before a real broker:
risk limits (gross exposure), a drawdown circuit-breaker, a per-strategy **kill-switch**, and a
**warm-up gate** — all enforced in the existing deterministic sim-broker replay loop.

---

## 1. Context & non-goals

Slice 1 (PR #13) built the paper replay loop: `algua paper run <name>` replays a `paper`-stage
strategy through a local `SimBroker` (fills at `t+1` open, long-only, cash-safe), persisting
orders/fills/audit in the registry DB. This slice hardens that loop with the guards the
architecture spec lists under `risk/` ("exposure limits + kill-switch, validated in paper").

**Decisions locked in brainstorming:**
- A risk-limit breach is **fail-closed**: halt the run and **trip a persisted kill-switch** that
  won't auto-resume until a human resets it.
- Risk checks: **gross exposure** (contract violation) + a **drawdown circuit-breaker** (realized
  loss). Per-symbol caps / max-position-count deferred (YAGNI).
- Kill-switch is **per-strategy**, persisted in the registry DB, **human-only reset**, and
  **halt-only** (no auto-flatten — that lands with the wall-clock loop that needs it).
- The warm-up bar count lives on `ExecutionContract` (`warmup_bars`), pinned per strategy.

**Non-goals (later slices):** Alpaca paper adapter + wall-clock scheduler; auto-flatten on kill;
a global ("halt all") kill-switch; per-symbol / max-position limits; deeper status/health.

---

## 2. The contract/CLI split

- **`ExecutionContract`** holds *signal-execution* properties pinned per strategy:
  `max_gross_exposure` (exists), `decision_lag_bars` (exists), and **`warmup_bars: int = 0`** (new).
- **CLI `paper run --max-drawdown`** holds the *operator risk policy* (default `1.0` = off), since
  a drawdown threshold is a risk-management choice that can vary per run, not a property of the
  signal. Passed into `run_paper`.

---

## 3. Components

| Module | Responsibility |
|---|---|
| `algua/contracts/types.py` (modify) | Add `warmup_bars: int = 0` to `ExecutionContract`; `__post_init__` validates `warmup_bars >= 0`. Backward-compatible (default 0); the backtest engine ignores it. |
| `algua/risk/__init__.py` (new) | Package marker. |
| `algua/risk/limits.py` (new, pure) | `RiskBreach(ValueError)` with `.kind` + `.detail`; `check_gross_exposure(weights, max_gross)`; `check_drawdown(equity, peak, max_drawdown)`. No I/O. |
| `algua/risk/kill_switch.py` (new, DB) | Per-strategy state: `trip(conn, strategy, reason, actor)`, `is_tripped(conn, strategy)`, `reset(conn, strategy, actor)`, `get(conn, strategy)`. A row in `kill_switches` = tripped; `reset` deletes it. |
| `algua/registry/db.py` (modify) | Schema **v3**: add `kill_switches (strategy TEXT UNIQUE, reason TEXT, actor TEXT, created_at TEXT)`. |
| `algua/live/paper_loop.py` (modify) | `run_paper(..., max_drawdown=1.0)`: per-tick warm-up gate → gross/long-only checks → drawdown check; raises `RiskBreach` on a hard breach. Stays pure (no DB). |
| `algua/cli/paper_cmd.py` (modify+extend) | `paper run` refuses if tripped, catches `RiskBreach` → trips switch + `{ok:false}`; new `paper kill` / `paper resume`; `paper show` reports kill-switch state. |
| `pyproject.toml` (modify) | Import contracts: `algua.risk` off `algua.cli`; `algua.backtest` off `algua.risk` (extends the existing backtest-off-execution/live contract). |

`RiskBreach` subclasses `ValueError` so the existing long-only test still passes, `json_errors`
still renders it, and the CLI can `isinstance`-catch it to trip the switch.

---

## 4. Per-tick data flow (`run_paper`)

Setup: `peak = broker.equity(closes[ts[0]])` (initial); `bars_seen = 0`.

For each bar `t` with a successor `t_next`:
1. `bars_seen += 1`.
2. `weights = target_weights(view≤t)`. **Long-only** check (negative weight → `RiskBreach("long_only")`)
   and **gross-exposure** check (`Σ|w| > max_gross_exposure + 1e-9` → `RiskBreach("gross_exposure")`).
3. `equity = broker.equity(closes[t])`; `peak = max(peak, equity)`; **drawdown** check
   (`max_drawdown < 1.0 and equity < peak·(1 − max_drawdown)` → `RiskBreach("drawdown")`).
4. **Warm-up gate:** only if `bars_seen >= warmup_bars` do we `build_intents → submit → fill at
   t_next open`. During warm-up the loop observes bars (updates `bars_seen`/`peak`) but submits no
   orders.

On clean completion returns `PaperRunResult` (unchanged shape). A `RiskBreach` propagates out of
`run_paper` — the CLI handles the DB side.

---

## 5. Error handling (CLI owns the DB side)

- **Tripped at start.** `paper run` calls `kill_switch.is_tripped(conn, name)` first; if tripped →
  `{ok:false, error:"kill-switch tripped for <name>; reset with 'algua paper resume <name>'"}`,
  exit 1; `run_paper` is never called.
- **Breach mid-run.** `run_paper` raises `RiskBreach`; the CLI catches it, calls
  `kill_switch.trip(conn, name, reason=str(exc), actor="system")`, audits the trip, and emits
  `{ok:false, kind:<exc.kind>, kill_switch:"tripped", error:<detail>}`, exit 1. **Partial orders/
  fills are NOT persisted** (the run aborted), so a breach can't overwrite a prior good paper book.
- **Clean run.** Persist + audit + emit summary, exactly as slice 1.
- **Manual control.** `paper kill <name> --reason R [--actor agent|human]` trips; `paper resume
  <name>` resets (records an audit row, `actor=human`); `paper show <name>` includes
  `kill_switch: {tripped, reason, actor}`.

`--max-drawdown` defaults to `1.0` (off); `warmup_bars` defaults to `0` (no extra gating).

---

## 6. Testing (offline, deterministic)

- **Unit `risk/limits`:** gross within/over; drawdown within/over; `RiskBreach` is a `ValueError`.
- **Unit `risk/kill_switch`:** `trip`→`is_tripped` true; `reset`→false; `get` returns reason/actor;
  re-trip updates the row.
- **Unit `db`:** schema v3 creates `kill_switches`; migrate idempotent → `user_version == 3`.
- **Unit `paper_loop`:** a weights-sum-> >1 strategy raises `RiskBreach("gross_exposure")`; a
  losing strategy (synthetic falling prices) with a tight `--max-drawdown` raises
  `RiskBreach("drawdown")`; with `warmup_bars=N`, no fills occur until after `N` bars; existing
  slice-1 tests stay green (long-only now raises `RiskBreach`, still a `ValueError` matching
  "long-only").
- **e2e `paper_cmd`:** `paper kill` then `paper run` is refused; `paper resume` then `paper run`
  works; a breaching strategy trips the switch, persists no paper state, and `paper show` reports
  `tripped`.
- **Gates:** `pytest · ruff · mypy · lint-imports` (incl. the new `risk` boundary contracts).

---

## 7. Consequences

- The kill-switch is the single safety state on the road to live: a contract violation or a
  drawdown breach trips it, and only a human resets it. The future live wall-clock loop reuses the
  same `kill_switch` + `risk/limits` modules (and will add auto-flatten + a global switch).
- `run_paper` stays pure (raises `RiskBreach`); the DB/kill-switch side stays in the CLI, so the
  loop and the limit checks remain unit-testable without a database.
- `warmup_bars` on `ExecutionContract` makes "ready to trade" an explicit, pinned, inspectable
  property — the seam the live loop (which boots with limited history) will depend on.
