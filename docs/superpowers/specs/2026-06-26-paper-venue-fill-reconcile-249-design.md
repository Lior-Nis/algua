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

Give the lane a real, independently-maintained position belief sourced from the Alpaca paper
account's own fill stream — i.e. **port the live lane's attributed-fill reconcile architecture
to the paper venue**. The live lane already has every primitive we need; the work is to
generalize them to a second account-scoped ledger and wire them into the paper lane.

## The seven invariants (from the GATE-1 design review on #249)

1. **Attributed-net reconcile.** The paper account is **shared/multi-strategy**
   (`_alpaca_broker_from_settings`, one paper account for all strategies). A naive per-strategy
   `paper_venue_fills` belief compared against the whole-account broker book would trip on a
   sibling's legitimate holding. It needs the attributed-net model the live lane uses
   (`attributed_live_net`): the per-strategy expectation is *broker-net minus other paper-lane
   strategies' attributed net*.
2. **Dedicated ledger.** New `paper_venue_fills` (+ cursor/activities/quarantine), mirroring
   `live_fills`. Do **not** reuse `paper_fills` — SimBroker `paper run` owns it (`persist_run`
   does a leading `DELETE`, no dedup key). Attribute via `paper_orders.broker_order_id`.
3. **Ingestion via the #250-hardened path.** Reuse `ingest_activities` (generalized to be
   account-scoped via a **typed `LedgerTables` allowlist**, NOT raw table-name strings), so
   quarantine / cursor crash-safety / COALESCE late-attribution come for free. The non-fill
   activities target must be parameterized (a paper `paper_venue_activities` table), else paper
   cash/dividend rows would contaminate the live `live_activities` ledger.
4. **Paginated, fail-closed ingestion.** Use the exhaustively-paginated `account_activities_window`
   (`alpaca_broker.py:354`) semantics, never the single-page `account_activities`. Partial
   ingestion must fail closed, never look clean.
5. **Crash-safe order recording.** Paper currently records `paper_orders` *after* the broker
   accepts (`on_submitted` hook). Adopt the live record-intent-before-submit + backfill pattern
   (`record_live_order` + `backfill_broker_order_id`), so a crash between venue-accept and record
   cannot leave an unattributable fill.
6. **Strategy-scoped offset flatten.** Breach / `paper flatten` must liquidate via
   `submit_offset(sym, qty)` from the believed strategy qty (not account-netted `close_positions`,
   which liquidates siblings on a shared account). Record offset orders before submit so their
   fills drive the belief flat; report "liquidation submitted", not "flat" (belief reaches flat
   only after the offset fills are ingested on a later tick).
7. **Forward-gate fail-closed.** A tick whose ingestion failed/was missing must not stamp
   `reconcile_ok = True`; distinguish missing-evidence from success.

## Architecture

### Data model (SCHEMA_VERSION 29 → 30)

Four new tables in `algua/registry/db.py`, mirroring the live ledger exactly:

```sql
CREATE TABLE IF NOT EXISTS paper_venue_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id     TEXT NOT NULL UNIQUE,
    broker_order_id TEXT,
    strategy        TEXT,                       -- nullable: orphan / pre-backfill
    symbol          TEXT NOT NULL,
    qty             REAL NOT NULL CHECK(qty != 0),
    price           REAL NOT NULL CHECK(price > 0),
    fill_ts         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_paper_venue_fills_strategy_symbol
    ON paper_venue_fills(strategy, symbol);

CREATE TABLE IF NOT EXISTS paper_venue_activities (        -- non-fill (cash/div) rows
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL, symbol TEXT, amount REAL, ts TEXT, raw TEXT
);

CREATE TABLE IF NOT EXISTS paper_venue_fill_cursor (       -- ingestion cursor
    name TEXT PRIMARY KEY, cursor TEXT
);

CREATE TABLE IF NOT EXISTS paper_venue_activity_quarantine ( -- #250 dead-letter
    activity_id TEXT PRIMARY KEY, error TEXT NOT NULL, raw TEXT NOT NULL
);
```

Schema is additive (no migration of existing rows). `paper_orders` already has
`broker_order_id` (the attribution key) and `strategy_id` — no change needed there for
attribution, though S3 changes *when* a `paper_orders` row is written (see below).

### Generalized ledger (the `LedgerTables` seam)

`algua/execution/live_ledger.py` is generalized so its ingestion + belief functions operate on
a **typed table set** rather than hardcoded `live_*` names:

```python
@dataclass(frozen=True)
class LedgerTables:
    fills: str
    activities: str
    cursor: str
    orders: str          # attribution source: SELECT strategy FROM <orders> WHERE broker_order_id=?
    quarantine: str

# The ONLY two instances — a closed allowlist of vetted constants. Table names are interpolated
# into SQL, so they must never originate from caller/runtime input; both constants are defined here.
LIVE_LEDGER  = LedgerTables("live_fills", "live_activities", "live_fill_cursor",
                            "live_orders", "live_activity_quarantine")
PAPER_LEDGER = LedgerTables("paper_venue_fills", "paper_venue_activities",
                            "paper_venue_fill_cursor", "paper_orders",
                            "paper_venue_activity_quarantine")
```

`ingest_activities`, `_ingest_one_activity`, `_quarantine_activity`, `fill_cursor`,
`believed_positions` take a `tables: LedgerTables` parameter. Per the no-dual-paths rule, all
existing live call sites are updated to pass `LIVE_LEDGER` explicitly (no defaulted param,
no compat shim). The interpolation is `f"... {tables.fills} ..."` over the closed constant set,
so it cannot be an injection vector.

`paper_believed_positions(conn, strategy)` = `believed_positions(conn, strategy, PAPER_LEDGER)`
(Σ this strategy's own `paper_venue_fills`, nonzero only).

### Attributed-net reconcile (the load-bearing piece)

A new `attributed_paper_net(conn)` in `live_reconcile.py`, the paper analog of
`attributed_live_net`, scoped to the **paper lane** = strategies whose stage is `paper` OR
`forward_tested` (both trade the paper venue; `flatten` already treats both as live-on-paper):

```sql
SELECT f.symbol, SUM(f.qty) AS q FROM paper_venue_fills f
  JOIN strategies s ON s.name = f.strategy AND s.stage IN ('paper','forward_tested')
 GROUP BY f.symbol;   -- nonzero only
```

**Wiring into `trade-tick` (no `run_tick` change).** `run_tick` reconciles
`hooks.derived_positions == positions_before` (whole broker book). We pass
`derived_positions = attributed_paper_net(conn)` — the sum over *all* paper-lane strategies'
attributed beliefs. Algebra:

```
Σ(all paper-lane attributed) == broker_net
  ⟺  this_belief + Σ(others) == broker_net
  ⟺  this_belief == broker_net − Σ(others)        # the attributed-net check, invariant #1
```

An orphan/unattributed broker holding O makes `broker_net = this + others + O ≠ Σ(attributed)`
→ reconcile trips → **fail closed** (matching the live resume reconcile). This reuses
`run_tick`'s existing equality check verbatim; only the *value* fed to the hook changes.

### Ingestion in the lane (paginated, fail-closed)

`trade-tick` ingests the paper venue's fills into `paper_venue_fills` **before** building the
reconcile belief, using the exhaustively-paginated `account_activities_window` (bounded by the
last cursor → the broker clock), and fails the tick closed if ingestion raises (no snapshot, no
`reconcile_ok = True` stamp — invariant #7). A single malformed activity is quarantined (#250),
not fatal; a transport/pagination failure rolls back and aborts.

### Crash-safe order recording (S3)

The `on_submitted` hook switches from `record_submitted_order` (post-accept) to the live
pattern: `record_paper_order` (intent, keyed by `client_order_id`, status `submitted`) is
written *before* the broker call, and `backfill_paper_broker_order_id` attaches the
`broker_order_id` once the broker accepts — back-attributing any fill already ingested under
that broker id while `strategy` was NULL. This requires `paper_orders` to carry a
`client_order_id` column (currently it stores only `broker_order_id`); added in S3 as an
additive nullable+unique-indexed column.

> **Open design point for review:** `paper_orders` today is `INSERT OR IGNORE` keyed by
> `(strategy, broker_order_id)` and is also the source of `_strategy_held_symbols` and the
> `paper show` order count. S3 must preserve those readers while moving the write earlier and
> keying on `client_order_id`. The live lane keeps `live_orders` (intent/attribution) separate
> from the operability views; the paper lane overloads `paper_orders` for both. The plan will
> decide whether S3 adds `client_order_id` to `paper_orders` in place or introduces the
> intent-before-submit as a column-level change — flagged here so GATE-1 weighs it.

### Strategy-scoped offset flatten (S4)

The breach handler in `trade-tick` and the `flatten` command replace
`broker.close_positions(symbols)` (account-netted) with the live offset pattern:

```python
ingest_activities(conn, paginated_paper_activities(...), PAPER_LEDGER)   # refresh belief
for sym, qty in paper_believed_positions(conn, name).items():
    coid = client_order_id(name, decision_ts, f"offset-{sym}")
    record_paper_order(conn, name, sym, "sell" if qty > 0 else "buy", ..., coid)
    oid = broker.submit_offset(sym, qty, coid)
    backfill_paper_broker_order_id(conn, coid, oid)
```

Result reports `liquidation_submitted: True` (not "flat"): the belief reaches flat only after
the offset fills are ingested on a later tick. The `clear_derived_positions` #163 band-aid
(which zeroed the *sim* `paper_fills` belief on flatten) is removed — the venue ledger is now
the belief, and it is driven flat by real offset fills, not a DELETE.

## Slices

- **S1 — ledger foundation.** Schema (4 tables, v30); `LedgerTables` seam + generalized
  `ingest_activities`/`believed_positions`/`fill_cursor` with all live call sites threaded to
  `LIVE_LEDGER`; `paper_believed_positions`; a paginated fail-closed paper ingest helper. No
  behavior change to the reconcile yet. Fully unit-tested (ingest dedup, backfill, quarantine,
  pagination fail-closed, attribution).
- **S2 — attributed-net reconcile.** `attributed_paper_net`; wire paper ingest + the
  attributed-net belief into `trade-tick`'s `derived_positions`. End-to-end test: fill at venue →
  next tick reconciles clean (no phantom flatten); sibling holding does not trip; orphan trips.
- **S3 — crash-safe order recording.** `client_order_id` on `paper_orders`; `record_paper_order`
  + `backfill_paper_broker_order_id`; switch the `on_submitted` hook to record-before-submit.
  Test the crash window (accept-without-record) is closed and early-fill back-attribution works.
- **S4 — offset flatten + fail-closed evidence.** Strategy-scoped `submit_offset` flatten in the
  breach handler + `flatten` command; "liquidation submitted" semantics; remove
  `clear_derived_positions`; ingestion-failure → no `reconcile_ok=True` stamp. Test breach
  liquidates only the strategy's own symbols (sibling untouched), and a failed ingest does not
  fabricate a passing reconcile tick.

Each slice keeps the quality gate green
(`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`).
All four ship in **one PR** (per the maintainer's scope decision), reviewed as one diff at GATE 2.

## Testing strategy

- **Ledger unit tests (S1):** ingest dedup by `activity_id`; COALESCE backfill of a fill ingested
  before its order mapping; quarantine of a malformed activity advances the cursor past it;
  id-less activity fails closed; `account_activities_window` pagination fail-closed on a partial
  page; `paper_believed_positions` sums signed nonzero only.
- **Reconcile tests (S2):** the phantom-flatten regression (fill → clean next tick); sibling
  paper-lane holding explains broker qty (no trip); orphan/non-paper holding trips fail-closed;
  `attributed_paper_net` excludes non-paper-stage strategies.
- **Crash-safe tests (S3):** order recorded before submit is attributable even if backfill never
  runs; backfill back-attributes a NULL-strategy fill.
- **Flatten tests (S4):** breach liquidates only the breached strategy's symbols, leaving a
  sibling's position open; `flatten` reports `liquidation_submitted`; an ingestion failure aborts
  the tick without stamping `reconcile_ok=True`.

Tests must not monkeypatch `run_tick` wholesale (the gap #249 calls out): the reconcile path is
exercised with a fake broker whose `account_activities_window`/`get_positions` are scripted.

## Out of scope (deferred)

- A grace-windowed paper account reconcile (`live_reconcile_state` analog). Paper keeps
  `run_tick`'s strict per-tick equality check fed by attributed-net; the live grace window is a
  separate nicety, not required to close #249.
- Average-cost P&L / NAV off `paper_venue_fills` (the live `strategy_nav` analog). The forward
  gate's equity series is unchanged by this work.
- Corporate-action handling of paper `paper_venue_activities` rows (they are recorded for audit
  and future use; nothing consumes them yet).
- Unblocking PR #288 (autonomous paper operator) `run-all`: that branch rebases onto main+#249
  after this merges; the rebase itself is not part of this PR.
```
