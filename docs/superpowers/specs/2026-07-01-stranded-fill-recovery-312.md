# Stranded-fill auto-recovery (#312)

## Problem
The #249 crash-safe recording pattern leaves an order row with `broker_order_id = NULL` if the
process dies **after** the broker accepts an order but **before** the local `broker_order_id`
backfill commits.

- **Paper lane:** `before_submit` inserts the `paper_venue_orders` intent row (NULL broker id)
  *before* the POST; `on_submitted` backfills the id *after* accept. Crash in between → sticky NULL.
- **Live lane:** `on_submitted` calls `record_live_order` (INSERT, NULL broker id) then
  `backfill_broker_order_id` (sets the id) — two separate commits. Crash between them → sticky NULL.

A subsequent fill ingested under that broker order id cannot be attributed (its `strategy` stays
NULL), so it shows up as an unexplained residual at the account reconcile (paper `attributed_paper_net`
/ live `attributed_live_net` both EXCLUDE strategy-NULL fills) and, after the grace window, halts the
loop / fails the forward gate's `no_unattributable_fills`. Fails closed today, but strands the fill
as a manual-recovery item with no auto-recovery.

## Scope (row-exists window only)
Recovery backfills a broker id onto an EXISTING local NULL-`broker_order_id` row. It covers:
- **Paper:** crash between `before_submit`'s intent insert (before POST) and `on_submitted`'s
  backfill.
- **Live:** crash between `record_live_order`'s commit and `backfill_broker_order_id`'s commit (both
  inside `on_submitted`, two separate commits) — the identical NULL-row window.

**Explicitly OUT of scope — the live "no-row" window:** a crash after Alpaca accepts but before
`record_live_order` commits leaves NO local row, so there is nothing to backfill onto. Closing it
requires recording the live intent BEFORE the POST (like paper), which reintroduces the #311
phantom-row problem for the live lane and thus its entire on-noop retraction apparatus — a materially
larger change. Deferred as a follow-up. Reconcile fail-closed remains the backstop meanwhile.

## Design (KISS, reconcile-layer, no schema change)

A recovery pass, run as a periodic reconcile step (and on live resume), that for each local
NULL-`broker_order_id` row asks the broker for the order carrying that exact `client_order_id` and,
on a symbol-verified match, backfills `broker_order_id` (which also back-attributes any fill already
ingested under that broker id — existing backfill semantics).

### New primitive — `_AlpacaBroker.get_order_by_client_order_id(coid) -> dict | None`
`GET /v2/orders:by_client_order_id?client_order_id=<coid>` (url-encoded). Returns the single order
dict (any status — open/filled/canceled) or `None` on 404 (the broker has no order with that coid).
Precise per-coid lookup — NO recency window (fixes the `limit=500` miss). A 404 naturally means "this
coid never reached the venue" (a pure #365 phantom / genuine noop), so the row is preserved.

### New function — `live_ledger.recover_stranded_broker_order_ids(conn, broker, *, kind) -> StrandedRecovery`
1. `stranded = {coid: symbol}` from `{kind}.orders WHERE broker_order_id IS NULL`.
2. **If empty, return without touching the broker** (cheap DB-first guard: no broker round-trip when
   nothing is stranded; also keeps unrelated test fakes from needing the method). Stranded rows are
   rare, so the per-coid loop is a handful of GETs at most.
3. For each `(coid, symbol)`: `order = broker.get_order_by_client_order_id(coid)`.
   - `None` (404) → skip (not at venue → preserve, per #365).
   - **Reject a malformed/inconsistent broker payload** (safety boundary): require the returned
     `client_order_id == coid` (exact), `symbol == local symbol`, and a non-empty string `id`. Any
     failure → **skip WITHOUT backfilling** and flag as mismatched (a coid collision / contamination
     must never mis-attribute; the unresolved NULL row still fails closed at reconcile). **Symbol is
     the reliable guard — side is NOT validated** because `submit_sized` derives the POSTed side from
     the delta sign while `before_submit` records `intent.side`, so the recorded side can
     legitimately differ from the broker order's side.
   - else → `_backfill_order(conn, kind, coid, order["id"])`; record as recovered iff it returns
     True (a concurrent backfill to a different id returns False → already resolved, skip).
4. Return `StrandedRecovery(recovered=[...], mismatched=[...])` (CLI audits both; `live_ledger`
   stays free of the audit dependency).

### DRY refactor + conditional backfill (TOCTOU-safe)
Extract the identical body of `backfill_paper_venue_broker_order_id` (PAPER) and
`backfill_broker_order_id` (LIVE) into a private `_backfill_order(conn, kind, coid, boid)` keyed by
`_TABLES[kind]`; both public functions delegate; recovery reuses it. Harden the UPDATE to be
CONDITIONAL — `UPDATE {orders} SET broker_order_id=? WHERE client_order_id=? AND broker_order_id IS
NULL` — so a row already backfilled by a concurrent tick/replay is NEVER blind-overwritten. Returns
`bool` (did we own the mapping). Fill back-attribution runs ONLY when the conditional UPDATE set the
row (`rowcount == 1`) OR a re-read confirms the existing `broker_order_id` already equals THIS
`boid` (idempotent replay); if the row already carries a DIFFERENT id, return False and attribute NO
fills (leave the strategy-NULL fills to fail closed). Fill back-attribution stays `WHERE strategy IS
NULL` (idempotent). Same end state for the on_submitted callers (fresh NULL row → sets it once);
strictly safer under concurrency.

### Wiring (call sites — after fill ingest, before reconcile)
- Paper `trade-tick` and `run-all`: after `_ingest_paper_venue`, before `paper_reconcile` — so a
  recovered fill is attributed before the reconcile-drift check runs (else it halts within grace).
  Fail-closed on a broker error (mirrors venue ingest). Audit `stranded_order_recovered` when >0.
- Live `run-all`: after `ingest_activities`, before `live_reconcile` (BrokerError propagates via
  `json_errors`, consistent with the adjacent ingest).
- Live `resume` (`_live_strategy_flat`): after `ingest_activities`, before the flatness computation
  — so a stranded fill no longer blocks resume ("on resume" per the issue).

## Safety argument
- **No double-submit:** recovery is a pure broker READ + local UPDATE; it never POSTs an order.
- **No mis-attribution:** the join is `client_order_id` (UNIQUE locally AND Alpaca-account-unique)
  AND a broker-symbol equality check; `_backfill_order` sets `strategy` from the LOCAL row's own
  column, never by parsing the coid. A coid collision surfaces as a symbol mismatch → skip, not a
  wrong attribution.
- **Respects #311 / #365:** recovery only ADDS a broker id, never deletes a row, so the #311 rule
  ("preserve a pre-existing NULL row — it may be a crashed real order") is intact. A pure #365
  phantom (a genuine noop whose coid was never POSTed) 404s at `by_client_order_id` → preserved
  (its retraction is #365's separate scope). A NULL row whose coid IS present at the broker is
  exactly the crash-stranded real order this fix resolves.
- **Idempotent / TOCTOU-safe:** the conditional `WHERE broker_order_id IS NULL` UPDATE never
  overwrites a concurrently-backfilled row, and fill back-attribution is `WHERE strategy IS NULL`, so
  concurrent ticks / replays converge with no lost update or double attribution.
- **Canceled/rejected broker order:** its coid still resolves at `by_client_order_id`; backfilling
  its id is correct — there is no fill, so the fill back-attribution UPDATE touches zero rows.

## Non-goals / deferred
- The live "no-row" crash window (see Scope) — deferred follow-up.
- The #365 phantom RETRACTION (404-at-venue ⇒ delete the stale NULL row) stays out of scope; this
  fix only backfills the present-at-venue case.
- No schema change (the `broker_order_id` column is already nullable).
