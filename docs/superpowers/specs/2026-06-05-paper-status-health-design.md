# Paper Status / Health + Per-Tick Snapshot (Sub-project 5, Slice C)

**Date:** 2026-06-05
**Status:** Accepted (pending implementation)
**Scope:** Persist the per-tick account/positions snapshot (currently in-memory only) and grow
`paper show <name>` into a consolidated per-strategy operability view: stage, kill-switch, drawdown
state, last-tick snapshot, recent orders, and a health rollup.

---

## 1. Context & non-goals

After the hardening sweep + the drawdown breaker (#27), per-strategy paper trading is robust but
hard to *observe*: `paper show` reports only `{n_orders, positions, kill_switch}`, and the per-tick
`TickSnapshot` (equity + positions, captured in `run_tick` for sizing/drawdown) is never persisted,
so there is no record of account equity over time. This slice adds that record and a single
consolidated view an operator (or agent) reads to answer "what is this strategy doing right now?".

**Non-goals:** the global ("halt-all") kill-switch (separate slice); a fleet/all-strategies
summary (this stays per-strategy — `registry list` already enumerates); the web dashboard (later
observability thread; this slice is the CLI/JSON backbone it will read); any equity-curve analytics
(the history table just stores rows). **Out of scope:** snapshotting breach/halt ticks (see §3).

---

## 2. Design decisions (settled in brainstorming)

- **Consolidated view = enrich the existing `paper show <name>`** (per-strategy). One command, no
  overlap with a second `status`/`health` command.
- **Snapshot persistence = append-only history** — one `tick_snapshots` row per completed tick.
  "Last tick" is the most recent row; the full history also feeds the future equity curve. Barely
  more code than a latest-only upsert.

---

## 3. Persistence — `tick_snapshots` (schema v5)

```sql
CREATE TABLE IF NOT EXISTS tick_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy     TEXT NOT NULL,           -- denormalized by name (matches paper_orders/audit_log)
    tick_ts      TEXT NOT NULL,           -- wall-clock ISO-UTC when the tick ran
    decision_ts  TEXT,                    -- bar the decision was made on (NULL on a no-bar tick)
    equity       REAL NOT NULL,           -- account equity snapshot at tick start
    peak_equity  REAL,                    -- persisted high-water mark after this tick
    positions    TEXT NOT NULL,           -- JSON {symbol: qty} from the tick snapshot
    n_submitted  INTEGER NOT NULL,        -- orders submitted this tick
    reconcile_ok INTEGER NOT NULL         -- 1/0
);
CREATE INDEX IF NOT EXISTS ix_tick_snapshots_strategy_ts ON tick_snapshots(strategy, tick_ts);
```
- `SCHEMA_VERSION` 4 → 5 (bootstrap `migrate()` adds the table via `CREATE TABLE IF NOT EXISTS`);
  update the version-assertion test.
- **`TickResult` gains `equity: float`** (default `0.0`), set from `snap.equity` in `run_tick`, so
  the CLI can persist the snapshot equity without a second broker call.
- **Only completed ticks are snapshotted.** A breach/halt raises inside `run_tick` before returning
  a `TickResult`, so no equity is available on that path; the `audit_log` + kill-switch already
  record the halt. Snapshots therefore represent the equity curve of completed ticks — a clean,
  consistent series. (Enhancing this to snapshot breach ticks is deferred.)

### `order_state.py` helpers (alongside the peak helpers)
- `record_tick_snapshot(conn, strategy, *, tick_ts, decision_ts, equity, peak_equity, positions, n_submitted, reconcile_ok)` — INSERT one row (`positions` dict → `json.dumps`).
- `latest_tick_snapshot(conn, strategy) -> dict | None` — most recent row by `id` (positions JSON parsed back to a dict), or `None`.
- `recent_orders(conn, strategy, limit=10) -> list[dict]` — most recent `paper_orders` rows (symbol, side, status, broker_order_id, submitted_ts), newest first.

### `trade-tick` wiring
On the **success path** (after `update_peak_equity`), call `record_tick_snapshot(...)` with
`tick_ts = now`, the result's `decision_ts`/`equity`/`peak_equity`/`positions_before`/
`len(result.submitted)`/`reconcile_ok`. No change to the breach/halt paths.

---

## 4. Enriched `paper show <name>`

Reads only persisted state (no broker call), emitting:
```json
{
  "strategy": "<name>",
  "stage": "<registry stage>",
  "kill_switch": {"tripped": <bool>, "reason": <str|null>},
  "drawdown": {"peak_equity": <float|null>, "last_equity": <float|null>, "drawdown": <float|null>},
  "last_tick": {"tick_ts","decision_ts","equity","positions","n_submitted","reconcile_ok"} | null,
  "positions": {<symbol>: <qty>},        // current, derived from fills (unchanged)
  "n_orders": <int>,
  "recent_orders": [ {"symbol","side","status","broker_order_id","submitted_ts"}, ... up to 10 ],
  "health": "ok" | "halted" | "drift" | "idle"
}
```
- `stage` from `SqliteStrategyRepository.get(name).stage.value`.
- `drawdown`: `peak_equity` from `get_peak_equity`; `last_equity` from the latest snapshot's
  `equity`; `drawdown = 1 - last_equity/peak_equity` when both present and `peak_equity > 0`, else
  `null`.
- `last_tick` from `latest_tick_snapshot` (or `null` if none).
- Unknown strategy → the existing `LookupError`/`ValueError` JSON-error path (`{ok:false}`).

## 5. Health rollup (derived)

Single field, evaluated in order:
- `"halted"` — kill-switch tripped.
- `"drift"` — latest snapshot exists and its `reconcile_ok` is false (DB vs broker book diverged).
- `"idle"` — no snapshots yet (never ticked).
- `"ok"` — otherwise.

---

## 6. Testing

- **Store** — `record_tick_snapshot` then `latest_tick_snapshot` round-trips (positions dict
  preserved, most-recent wins across two rows); `latest_tick_snapshot` on an unticked strategy is
  `None`; `recent_orders` returns newest-first and respects `limit`.
- **Schema** — `migrate` yields `user_version == 5`; `tick_snapshots` in the table set; the
  version-assertion test updated.
- **`trade-tick` persists a snapshot** — after a mocked successful tick, exactly one
  `tick_snapshots` row with the expected equity/positions/n_submitted; a breach tick writes none.
- **Enriched `show`** — surfaces `stage`, `drawdown` (peak/last/ratio), `last_tick`,
  `recent_orders`; unknown strategy → `{ok:false}`.
- **Health** — one test per branch: tripped → `halted`; reconcile-false snapshot → `drift`; no
  snapshot → `idle`; clean → `ok`.
- **Gate** — `pytest · ruff · mypy · lint-imports` (contracts stay `10 kept, 0 broken`).

---

## 7. Consequences

- The system gains a persistent equity/positions record per strategy and a single command that
  answers "what is this strategy doing, and is it healthy?" — completing sub-project 5's
  operability.
- `tick_snapshots` is the first time-series the web dashboard's PnL/equity view can read directly,
  with zero further schema work.
- `paper show` stays a pure read of persisted state (no broker round-trip), so it is safe to call
  anytime, including when the venue is unreachable.
