# Global ("Halt-All") Kill-Switch (Sub-project 5, final slice)

**Date:** 2026-06-05
**Status:** Accepted (pending implementation)
**Scope:** One operator command that halts **every** paper strategy at once and flattens the whole
Alpaca account, the gating that makes `trade-tick`/`paper run` respect it, and the command that
clears it. Completes sub-project 5.

---

## 1. Context & non-goals

Per-strategy safety is complete: each strategy has its own kill-switch, drawdown breaker, and
scoped flatten. What is missing is an **account-level** panic button — a single action that stops
all trading and liquidates the entire Alpaca account, independent of any individual strategy. The
account-wide `DELETE /v2/positions` was deliberately reserved for exactly this when B2b-1 scoped
per-strategy flatten to a strategy's universe.

**Non-goals:** auto-expiry / scheduled clearing of the halt; selective peak restore on resume-all
(all peaks are wiped wholesale); a fleet status dashboard; touching the `paper → live` gate
(sub-project 6). **Out of scope:** halting non-paper activity (research/backtest) — the global halt
governs the live/paper trading surface only.

---

## 2. Design decisions (settled in brainstorming)

- **State = a dedicated single-row `global_halt` table** (schema v6), separate from per-strategy
  `kill_switches`. Presence of the row = halted. Gates check `global_halt.is_set()` in addition to
  the per-strategy switch.
- **`resume-all` clears the flag AND re-bases all peaks** (`DELETE FROM strategy_peaks`): the halt
  flattened the whole account to cash, so every strategy's drawdown peak is stale — the same
  re-base logic proven for per-strategy resume (#109), applied account-wide. Per-strategy
  kill-switches are left untouched (those are independent operator decisions).

---

## 3. Components

| Module | Responsibility |
|---|---|
| `algua/registry/db.py` (modify) | `global_halt` table; `SCHEMA_VERSION` 5 → 6. |
| `algua/risk/global_halt.py` (new) | `set` / `is_set` / `clear` / `get` — mirrors `kill_switch.py`. |
| `algua/execution/alpaca_broker.py` (modify) | re-add `close_all_positions()` (account-wide liquidate + cancel). |
| `algua/cli/paper_cmd.py` (modify) | `paper halt-all` + `paper resume-all`; gate `trade-tick`/`paper run`; `show` reflects the global halt. |

**Reuses:** the `_multistatus_failures` 207 parser, `_alpaca_broker_from_settings`, `audit_append`,
`json_errors`, the trip-before-close fail-safe pattern from `paper flatten`. No new import contract.

---

## 4. State — `global_halt` table + module

```sql
CREATE TABLE IF NOT EXISTS global_halt (
    id         INTEGER PRIMARY KEY CHECK (id = 1),   -- single row; presence = halted
    reason     TEXT,
    actor      TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```
`algua/risk/global_halt.py` (mirrors `kill_switch.py`):
- `set(conn, *, reason, actor)` — `INSERT INTO global_halt(id, ...) VALUES (1, ...) ON CONFLICT(id) DO UPDATE ...` + commit.
- `is_set(conn) -> bool` — a row with id=1 exists.
- `clear(conn) -> bool` — `DELETE FROM global_halt`; returns whether a row was removed.
- `get(conn) -> dict | None` — `{reason, actor, created_at}` or None.

`SCHEMA_VERSION` 5 → 6 (bootstrap `migrate()` adds the table). The version-assertion test asserts
against the `SCHEMA_VERSION` constant, so no number edit is needed.

## 5. Broker — `close_all_positions()`

Re-add the account-wide primitive (reserved in B2b-1):
```python
def close_all_positions(self) -> None:
    """Liquidate the ENTIRE account: DELETE /v2/positions?cancel_orders=true — Alpaca cancels all
    open orders then market-closes every position, returning a 207 multi-status; raise if any
    per-position close failed. Account-wide — used ONLY by the global halt (per-strategy flatten
    uses close_positions(universe))."""
    results = self._read(
        self._delete("/v2/positions?cancel_orders=true"), "/v2/positions", ok=(200, 207)
    )
    if isinstance(results, list) and _multistatus_failures(results):
        raise BrokerError(f"alpaca failed to close some positions: {results}")
```
`cancel_orders=true` makes this both cancel orders and liquidate, so no separate
`cancel_open_orders` call is needed. Empty account → empty list → no-op.

## 6. `paper halt-all` / `paper resume-all`

**`paper halt-all --reason <str> [--actor agent|human]`** — trip-before-close fail-safe (mirrors
`paper flatten`):
1. `global_halt.set(conn, reason=reason, actor=actor)` + `audit_append(action="halt_all")`.
   **Halt first** — even if the close then fails, all trading is already stopped.
2. `broker = _alpaca_broker_from_settings()`; `try: broker.close_all_positions()`.
   - on `BrokerError` → `emit({"ok": False, "global_halt": "set", "liquidation_submitted": False, "error": <str>})` + `Exit(1)` (audited `flatten_failed`).
   - else → `emit({"global_halt": "set", "liquidation_submitted": True})`.

   Decorated `@json_errors(ValueError, LookupError, BrokerError)`.

**`paper resume-all [--actor human]`** — clears the halt and re-bases all peaks:
1. `was_set = global_halt.is_set(conn)`.
2. if `was_set`: `audit_append(action="resume_all", reason="re-bases all drawdown peaks")`;
   `clear_all_peaks(conn)` (re-base first — the un-halt is the final write, per #109's fail-safe
   ordering); `global_halt.clear(conn)`.
3. `emit({"global_halt": "reset" if was_set else "not_set"})`.

   `clear_all_peaks(conn)` is a new `order_state` helper: `DELETE FROM strategy_peaks` (all rows).

## 7. Gating

- **`trade-tick`** and **`paper run`**: after building the connection, before any work, raise
  `ValueError("global halt active; clear with 'algua paper resume-all'")` when
  `global_halt.is_set(conn)` (rendered `{ok:false}` by `json_errors`). This sits alongside the
  existing per-strategy kill-switch gate.
- **`trade-tick` pre-submit hook**: the `should_halt` callback becomes
  `kill_switch.is_tripped(conn, name) or global_halt.is_set(conn)`, so a `halt-all` fired *mid-tick*
  (between cancel and submit) aborts before any order goes out — reusing the existing #21 re-check.

## 8. `paper show` reflection

So the operator sees *why* a strategy is halted:
- `health` is `"halted"` when `global_halt.is_set()` **or** the per-strategy switch is tripped.
- the emitted `kill_switch` block gains `"global_halt": <bool>`.

---

## 9. Testing

- **`global_halt` module** — `set`→`is_set` True; `get` returns reason/actor; `clear` removes it
  (and returns True), `is_set` False after; `set` twice (upsert) stays single-row.
- **Schema** — `migrate` yields `user_version == 6`; `global_halt` in the table set.
- **`close_all_positions`** — 207 all-success no-op; 207 with a per-position failure → `BrokerError`;
  empty list → no-op (assert the request path carries `cancel_orders=true`).
- **`halt-all`** — sets the flag + calls `close_all_positions` (mocked broker) + `liquidation_submitted:true`; a `BrokerError` from the close → `{ok:false, global_halt:"set", liquidation_submitted:false}` exit 1, flag still set.
- **`resume-all`** — after a halt + a seeded peak: clears the flag, `strategy_peaks` emptied, a pre-existing per-strategy kill-switch still tripped; resume-all when not set → `{global_halt:"not_set"}`.
- **Gating** — `trade-tick` and `paper run` refused (`{ok:false}`) while globally halted.
- **`show`** — globally halted → `health=="halted"` and `kill_switch.global_halt==true`.
- **Gate** — `pytest · ruff · mypy · lint-imports` (contracts stay kept, 0 broken).
- **Live acceptance (manual, documented, NOT CI)** — `algua paper halt-all --reason test` against the
  real paper account → all positions liquidated + orders cancelled; `trade-tick` refused; `paper
  resume-all` clears it.

---

## 10. Consequences

- The account gains a true panic button: one command halts all paper trading and liquidates the
  whole account, independent of per-strategy state — completing sub-project 5's safety surface.
- `close_all_positions` is the account-wide counterpart to the per-strategy `close_positions`; the
  two flatten scopes are now both present and clearly separated.
- The global halt + per-strategy switch compose cleanly: a strategy is blocked if **either** is
  active, and clearing one never silently clears the other.
- Next: sub-project 6 (live hardening + the human TOTP gate for `paper → live`).
