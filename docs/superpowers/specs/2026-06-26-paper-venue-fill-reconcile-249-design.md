# Paper-venue fill reconcile (issue #249) — design

## Problem

The wall-clock `paper trade-tick` lane reconciles a DB-derived position belief against the
Alpaca paper broker before each tick (`run_tick`, `live_loop.py:171-179`: raise
`RiskBreach("reconcile")` when `derived != positions_before`). But the belief it passes,
`derive_positions(conn, name)` (`order_state.py:89`), sums **`paper_fills`** — a table written
**only** by `persist_run`, the SimBroker `paper run` *replay*. The wall-clock lane's sole DB
write is `record_submitted_order` → `paper_orders` (`paper_cmd.py:319`). It never writes a fill.

So once a real Alpaca-paper order fills:

- Tick 1: `derive_positions` = `{}`, broker = `{}` → reconcile passes, orders submit.
- Orders fill at the paper venue.
- Tick 2: `derive_positions` = `{}` (still — the lane writes no fills) vs broker = `{AAPL: 10}`
  → `RiskBreach("reconcile")` → kill-switch trip + flatten of a **healthy** strategy.

The lane's central drift defense is non-functional and actively harmful (phantom flatten).

### Why "just drop the reconcile" is unsafe (PR #279, closed as superseded)

The wall-clock `trade-tick` lane is the **broker-clocked forward-evidence lane**. The forward
gate requires `reconcile_ok = 1` on every admissible tick (`forward_promotion.py:249` counts
`reconcile_ok = 0` rows; `forward_gates.py:208` fails the gate if `n_reconcile_failures != 0`).
Dropping the reconcile makes that integrity gate pass **vacuously** — a hole on the path to live.

### The real fix

Give the lane a real per-strategy position belief sourced from the Alpaca paper account's own
fill stream — a `paper_venue_fills` ledger ingested through the #250-hardened activity path —
and reconcile that belief against the broker's whole-account book before each tick.

## Design decision: single-tenant reconcile (GATE-1, 2026-06-26)

The issue's original GATE-1 note prescribed an **attributed-net** reconcile (a paper analog of
`attributed_live_net`: this strategy's belief = broker-net minus *other* paper strategies'
attributed net), on the premise that the paper account is shared/multi-strategy.

GATE-1 review (Codex) surfaced — and the code confirms — that the **forward gate already
mandates single tenancy** for any window that produces promotable evidence:

- `forward_promotion.py:278-283`: `single_tenant_ok = (n_siblings == 0)` — fails if *any other*
  strategy recorded a `lane='paper'` tick on the same `account_id` during the evidence window.
- `forward_promotion.py:139-158` `_classify_activities` + `forward_gates.py:235`
  `no_unattributable_fills`: every FILL on the account in the window must reconcile to *this*
  strategy's orders by `(strategy_id, broker_order_id)`; an unattributable fill fails the gate.
- `forward_gates.py:232` `no_external_cash_flows`: zero external capital movements.

So a strategy generating valid forward evidence is the **sole paper-lane tenant** on its account
by construction. The attributed-net machinery would therefore exist only to avoid phantom-tripping
during genuinely co-tenant *audition* trading — a configuration that can never promote anyway.

**Decision: drop attributed-net. The paper reconcile is single-tenant.** This is simpler, is
consistent with the platform's own gate invariant, and removes the unsound account-level
machinery the attributed model dragged in (account-level sizing contamination; an innocent
strategy flattened for a sibling's drift; stage-scoped attribution that breaks when a strategy
leaves the lane while holding positions). A co-tenant or orphan/manual holding the strategy
cannot explain leaves an unexplained residual and **fails closed** (refuses to trade) — the
operator's correct response is one strategy per paper account.

## Architecture

### Reconcile, inside `run_tick`'s snapshot boundary (fed by the venue belief)

`run_tick`'s reconcile branch is set by **exactly one caller** — paper `trade-tick`
(`paper_cmd.py:330`); the live lane reconciles in its own CLI loop and never sets it (verified:
`derived_positions` has one setter). We keep the reconcile **inside `run_tick`** rather than
moving it to the lane, because moving it would create a pre-trade TOCTOU: a lane-level
`get_positions()` reconcile followed by `run_tick`'s *separate* `broker.snapshot()` for sizing
lets a fill/manual change between the two calls trade on unreconciled state (Codex GATE-1-r2
HIGH). Keeping it in `run_tick` holds reconcile + sizing within one pre-submit critical section.

The trade-tick flow:

1. **Lane ingests** the paper venue's activities into `paper_venue_fills` via the paginated,
   fail-closed `account_activities_window` path (see *Cursor* below) **before** calling
   `run_tick`. A transport/pagination failure raises → the tick aborts with **no snapshot
   recorded** (never stamps `reconcile_ok = True`); a single malformed activity is quarantined
   (#250), not fatal.
2. **`run_tick` reconciles** the lane-supplied **venue belief** — a hook
   `venue_belief: Callable[[], dict] | None` that returns `paper_believed_positions(conn, name)`
   (Σ this strategy's own `paper_venue_fills`, signed, nonzero) — against the broker's
   **whole-account** net (`broker.get_positions()` — `/v2/positions`, read inside `run_tick`
   immediately before its sizing snapshot, so a held-but-dropped symbol out of the universe is
   still compared). Comparison is **per symbol over the union, with a `1e-6` tolerance**
   (fractional Alpaca fills summed as float must not exact-compare). A residual beyond tolerance →
   `RiskBreach("reconcile")` → trip + strategy-scoped flatten (S4). Result carries
   `reconcile_ok`.

The old exact-equality `derived_positions` branch is **replaced** by this tolerance comparison
against the whole-account book; `TickResult.reconcile_ok` is **kept** (live still stamps it from
its own reconcile — Codex r2 MEDIUM). When `venue_belief` is not supplied (live, sim), the branch
is skipped exactly as `derived_positions=None` is today. The sim-fed `derive_positions` is
**retained** for the SimBroker `paper show` view (Codex r2 MEDIUM); only `clear_derived_positions`
(the #163 band-aid) is removed — the offset flatten replaces it.

> Tolerance lives in one place: `_RECONCILE_TOL = 1e-6` already exists in `paper_cmd.py` (used by
> the live resume reconcile) — promote it to where `run_tick` can reuse it.

### Cursor (paper ingest is broker-time high-water, not activity-id)

Live `ingest_activities` advances the cursor to `max(activity_id)` and re-fetches via the
single-page `account_activities(after=id)`. Paper needs the **exhaustive** paginated
`account_activities_window(after, until)` (which is *time*-bounded, not id-bounded). So the paper
cursor is a **broker-time high-water**: the lane fetches
`account_activities_window(after=cursor_ts, until=broker.clock())`, and on success persists
`until` (the broker clock) as the new cursor. The window deliberately **overlaps** the previous
`until`; dedup by `activity_id` (UNIQUE) makes the replay idempotent (Codex r2 MEDIUM — cursor
semantics defined). `ingest_activities` therefore takes an explicit `cursor_value` (the paper
timestamp) instead of deriving `max(activity_id)`; the live call passes `None` and keeps the
id-based advance. The cursor is initialized fail-closed (first ingest fetches from a far-past
bound / the strategy's first venue order ts), so no fill is skipped.

### Fail-closed evidence (invariant)

Per-tick `reconcile_ok` is a live tripwire; the **authoritative** integrity check is the forward
gate's *promote-time* re-verification, which already re-fetches the entire window exhaustively
and fails closed (`forward_promotion.py:316-323` `activities_ok`, `_classify_activities`,
`single_tenant`). The only new obligation here is invariant #7: a tick whose ingestion failed
records **no** snapshot (handled by aborting in step 1) rather than a `reconcile_ok = True` row.
No new evidence ledger is required — the gate's promote-time re-classification is the backstop.

### Data model (SCHEMA_VERSION 29 → 30)

Five new tables in `algua/registry/db.py`, mirroring the live ledger. The wall-clock order
ledger is **separate from `paper_orders`** (which stays the SimBroker `paper run` replay table):
mixing pre-submit intents, rejected attempts, and offset orders into the replay table would break
its readers (`count_orders`, `recent_orders`, `_strategy_held_symbols`) and its
`(strategy, broker_order_id)` idempotency (Codex C1/M9/M10).

```sql
CREATE TABLE IF NOT EXISTS paper_venue_orders (        -- crash-safe intent + attribution (≈ live_orders)
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy          TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    intended_notional REAL,
    client_order_id   TEXT NOT NULL UNIQUE,             -- durable identity; idempotent re-submit
    broker_order_id   TEXT,                             -- backfilled on broker accept
    strategy_id       INTEGER NOT NULL,                 -- attribution for the forward gate
    status            TEXT NOT NULL,
    submitted_ts      TEXT NOT NULL
);
-- exactly one order may own a broker id (the fill-attribution key); many pre-backfill NULLs allowed
CREATE UNIQUE INDEX IF NOT EXISTS ux_paper_venue_orders_broker_order_id
    ON paper_venue_orders(broker_order_id) WHERE broker_order_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS paper_venue_fills (          -- signed fills (≈ live_fills)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id TEXT NOT NULL UNIQUE,
    broker_order_id TEXT,
    strategy TEXT,                                       -- nullable: orphan / pre-backfill
    symbol TEXT NOT NULL,
    qty REAL NOT NULL CHECK(qty != 0),
    price REAL NOT NULL CHECK(price > 0),
    fill_ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_paper_venue_fills_strategy_symbol
    ON paper_venue_fills(strategy, symbol);

CREATE TABLE IF NOT EXISTS paper_venue_activities (     -- non-fill (cash/div) rows (≈ live_activities)
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL, symbol TEXT, amount REAL, ts TEXT, raw TEXT
);

CREATE TABLE IF NOT EXISTS paper_venue_fill_cursor (    -- ingestion cursor (≈ live_fill_cursor)
    name TEXT PRIMARY KEY, cursor TEXT
);

CREATE TABLE IF NOT EXISTS paper_venue_activity_quarantine ( -- #250 dead-letter
    activity_id TEXT PRIMARY KEY, error TEXT NOT NULL, raw TEXT NOT NULL
);
```

Schema is additive. The single global `broker_order_id` uniqueness (partial index above) makes
the backfill attribution lookup `SELECT strategy FROM paper_venue_orders WHERE broker_order_id=?`
deterministic (Codex M10).

### Generalized ledger (the `LedgerTables` seam)

`algua/execution/live_ledger.py`'s ingestion + belief functions are generalized to operate on a
typed table set, so the paper ledger reuses the #250-hardened machinery verbatim:

```python
@dataclass(frozen=True)
class LedgerTables:
    fills: str; activities: str; cursor: str; orders: str; quarantine: str

# A CLOSED allowlist: the ONLY two instances, both defined here. To make this enforced rather than
# merely asserted (Codex C7), the ingest/belief functions accept a `LedgerKind` enum and map it to
# the constant internally — caller-supplied LedgerTables strings never reach SQL interpolation.
class LedgerKind(Enum): LIVE = "live"; PAPER = "paper"

_TABLES = {
    LedgerKind.LIVE:  LedgerTables("live_fills","live_activities","live_fill_cursor",
                                   "live_orders","live_activity_quarantine"),
    LedgerKind.PAPER: LedgerTables("paper_venue_fills","paper_venue_activities",
                                   "paper_venue_fill_cursor","paper_venue_orders",
                                   "paper_venue_activity_quarantine"),
}
```

`ingest_activities`, `_ingest_one_activity`, `_quarantine_activity`, `fill_cursor`,
`believed_positions` take a `kind: LedgerKind` and resolve `_TABLES[kind]` internally. All
existing live call sites pass `LedgerKind.LIVE` explicitly (no defaulted param, no compat shim).
Because table names come only from the private `_TABLES` map keyed by a closed enum, the f-string
interpolation cannot be an injection vector — the allowlist is *enforced*, not asserted.

`paper_believed_positions(conn, strategy)` = `believed_positions(conn, strategy, LedgerKind.PAPER)`.

### Crash-safe order recording

**A new `run_tick` hook is required.** `run_tick` submits *then* calls `on_submitted`
(`live_loop.py:215`) — a post-accept hook, so "record before submit" is **not** achievable
through it (Codex r2 HIGH; live has the same post-submit window today). Add a pre-submit hook
`before_submit(intent, client_order_id) -> None` fired immediately **before**
`broker.submit_sized`. The wall-clock lane records intent there; the existing `on_submitted`
backfills the broker id after accept:

- `before_submit`: `record_paper_venue_order` (intent into `paper_venue_orders`, keyed by
  `client_order_id`, status `submitted`) — written before the broker call.
- broker `submit_sized` returns the `broker_order_id`.
- `on_submitted`: `backfill_paper_venue_broker_order_id(client_order_id, broker_order_id)` —
  attaches the broker id, back-attributing any fill already ingested under it while `strategy`
  was NULL.

A crash between venue-accept and record can no longer orphan a fill: the intent row exists before
the order is ever sent. (Live can adopt `before_submit` later; closing live's window is out of
scope for #249.)

**Readers that move to the venue ledger** (the wall-clock lane's source of truth becomes
`paper_venue_orders`/`paper_venue_fills`, not `paper_orders`):

- `_strategy_held_symbols` (flatten symbol scope) → `paper_venue_orders` ∪ universe.
- The forward gate's `_classify_activities` (`forward_promotion.py:153`) → match FILLs against
  `paper_venue_orders` by `(strategy_id, broker_order_id)`. **Load-bearing**: this is the gate's
  `no_unattributable_fills` attribution; it must read the table the wall-clock lane now writes.
- `paper show` for a wall-clock/forward strategy → positions from `paper_believed_positions`,
  order count/recent from `paper_venue_orders`. The SimBroker `paper run` replay path keeps its
  own `paper_orders`/`paper_fills` view unchanged (Codex M11 — sim and venue stay distinct; a
  strategy that only sim-replayed still shows its sim view, a wall-clock strategy shows its venue
  view).

`paper_orders`/`paper_fills` and `persist_run` are otherwise untouched (SimBroker replay only).

### Strategy-scoped offset flatten (S4)

The breach handler in `trade-tick` and the `flatten` command replace
`broker.close_positions(symbols)` (account-netted) with the live offset pattern, so a flatten
never liquidates a co-tenant/sibling (defense even though evidence is single-tenant):

```python
ingest_activities(conn, paginated_paper_activities(...), LedgerKind.PAPER)   # refresh belief
for sym, qty in paper_believed_positions(conn, name).items():
    if abs(qty) <= _RECONCILE_TOL:        # submit_offset would return "noop"; skip (Codex r2 MEDIUM)
        continue
    coid = client_order_id(name, decision_ts, f"offset-{sym}")
    record_paper_venue_order(conn, name, sym, "sell" if qty > 0 else "buy", ..., coid)
    oid = broker.submit_offset(sym, qty, coid)
    backfill_paper_venue_broker_order_id(conn, coid, oid)   # oid is a real id, never "noop"
```

Filtering sub-tolerance qty before submit guarantees `submit_offset` never returns the sentinel
`"noop"`, so a fake `"noop"` broker id can't be backfilled and collide on the partial-unique
`broker_order_id` index.

Reports `liquidation_submitted: True` (not "flat"): the belief reaches flat only after the offset
fills are ingested on a later tick. The `clear_derived_positions` #163 band-aid is removed — the
venue ledger is the belief, driven flat by real offset fills, not a DELETE.

## Slices (crash-safe recording S2 BEFORE the reconcile S3 — Codex: don't wire the reconcile before venue fills are reliably attributable)

- **S1 — ledger foundation.** Schema (5 tables, v30); `LedgerKind`/`LedgerTables` seam +
  generalized `ingest_activities`/`believed_positions`/`fill_cursor` (with the explicit
  `cursor_value`) and all live call sites threaded to `LedgerKind.LIVE`; `paper_believed_positions`;
  a paginated fail-closed paper ingest helper (broker-time high-water cursor). No behavior change
  yet. Unit-tested (dedup, backfill, quarantine, pagination fail-closed, belief, cursor overlap).
- **S2 — crash-safe venue order recording.** `paper_venue_orders`; `record_paper_venue_order` +
  `backfill_paper_venue_broker_order_id`; add the `before_submit` hook to `run_tick` and record
  intent there (backfill in `on_submitted`); move `_strategy_held_symbols` + the forward gate's
  `_classify_activities` onto `paper_venue_orders`. Ordered first so the reconcile can rely on
  reliably-attributable venue fills.
- **S3 — single-tenant reconcile.** Add the `venue_belief` hook; in `run_tick`, reconcile the
  belief vs whole-account `broker.get_positions()` with `1e-6` tolerance (replacing the
  exact-equality `derived_positions` branch); keep `TickResult.reconcile_ok`; remove
  `clear_derived_positions` (retain sim `derive_positions`). End-to-end test: fill at venue → next
  tick reconciles clean (the phantom-flatten regression); orphan/manual holding trips fail-closed;
  a fractional fill within tolerance does not trip; a held-but-dropped symbol is still reconciled.
- **S4 — offset flatten + fail-closed evidence.** Strategy-scoped `submit_offset` flatten in the
  breach handler + `flatten` (sub-tolerance qty skipped, never backfills `"noop"`); "liquidation
  submitted" semantics; ingestion-failure aborts the tick with no snapshot (no `reconcile_ok=True`).
  Test: breach liquidates only the strategy's own symbols; a failed ingest fabricates no passing
  reconcile tick.

Each slice keeps the gate green
(`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`). All four
ship in **one PR** (maintainer scope decision), reviewed as one diff at GATE 2.

## Testing strategy

- **Ledger (S1):** ingest dedup by `activity_id`; COALESCE backfill of a fill ingested before its
  order mapping; quarantine advances the cursor past a malformed activity; id-less fails closed;
  `account_activities_window` pagination fail-closed on a partial page; `paper_believed_positions`
  sums signed nonzero only.
- **Crash-safe recording (S2):** an order recorded before submit is attributable even if backfill
  never runs; backfill back-attributes a NULL-strategy fill; the forward gate's
  `no_unattributable_fills` now sees wall-clock fills via `paper_venue_orders`.
- **Reconcile (S3):** phantom-flatten regression (fill → clean next tick); orphan/manual holding
  trips fail-closed; a fractional-share residual within `1e-6` does not trip; a held-but-dropped
  symbol (out of universe) is still reconciled (whole-account `get_positions`, not universe snap).
- **Flatten / fail-closed (S4):** breach liquidates only the breached strategy's symbols; an
  ingestion failure aborts the tick without stamping `reconcile_ok=True`.

Tests must not monkeypatch `run_tick`/`trade-tick` wholesale (the gap #249 calls out): the
reconcile path is exercised with a fake paper broker whose `account_activities_window` /
`get_positions` are scripted.

## Out of scope (deferred)

- A grace-windowed paper account reconcile (`live_reconcile_state` analog) — paper keeps a strict
  per-tick check; the live grace window is a separate nicety.
- Multi-tenant paper trading via an attributed-net reconcile — explicitly rejected above; the gate
  forbids co-tenant evidence, so it would be machinery without a promotable use.
- Average-cost P&L / NAV off `paper_venue_fills`; corporate-action handling of
  `paper_venue_activities` (recorded for audit; nothing consumes them yet).
- Unblocking PR #288 (`run-all`): that branch rebases onto main+#249 after this merges.
```
