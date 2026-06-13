# Breach → Flatten → Resume: end-to-end test + paper-flatten scope fix (issue #163)

**Status:** design (GATE-1)
**Date:** 2026-06-13
**Issue:** #163 — "Breach→flatten→resume path is untested end-to-end (and flatten still
doesn't reconcile the per-strategy ledger)" (P2, area:execution, bug, severity:medium)

## Problem

The breach-handling chain — risk breach raises → CLI trips the kill-switch → flatten →
later resume — is the least-verified money path in the system. The issue names three asks:

1. Ledger reconciliation after a flatten, on both resume paths.
2. Flatten over the union of broker-held symbols and the strategy universe, not the
   universe alone (held-but-dropped symbols are otherwise left open).
3. An end-to-end test: trip a breach mid-tick → assert the kill-switch trips, flatten
   closes **all** positions (including a held-but-dropped symbol), and a subsequent resume
   reconciles with zero drift.

## What the investigation found (this reshapes the scope)

- **The live path is already correct.** `live_cmd.py:152-178` (the `run-all` breach
  handler) already does: scoped-cancel → `ingest_activities` → iterate
  `believed_positions` (the *ledger-held* symbols, which include held-but-dropped) →
  `submit_offset` + `record_live_order`/`backfill_broker_order_id` for each. That closes
  both Gap 1 and Gap 2 for the live (real-money) lane. It is, however, **untested
  end-to-end**: `test_run_all_breach_liquidates_per_strategy` *mocks* `believed_positions`
  to a constant, returns `[]` from `account_activities`, and never runs `resume` or asserts
  zero drift.

- **The paper wall-clock venue maintains no real fill ledger.** `paper_fills` is written
  **only** by `persist_run`, the `run_paper` *simulation* — its own docstring defers the
  "real wall-clock paper adapter." The wall-clock `trade-tick` records submitted orders
  (`record_submitted_order` → `paper_orders`) but **no fills**. So `derive_positions` for a
  wall-clock-/forward-tested strategy does not reflect the real paper account. But it is NOT
  inert: `trade-tick` feeds `derive_positions` into its reconcile hook, so **stale sim
  `paper_fills` left after a flatten are a real wrong-belief hazard** — the next tick's
  reconcile compares the stale belief against the now-flat broker and re-trips
  (`RiskBreach("reconcile")`), re-trapping the operator. So the paper lane needs BOTH a
  scope fix (the universe-only close at `paper_cmd.py:337` trade-tick handler and `:476`
  flatten command leaves a held-but-dropped position open) AND a belief reset on flatten.
  What stays deferred is the *real* per-strategy wall-clock paper fill ledger (the sim
  ledger and the real paper account are different worlds — see Out of scope).

- **Per-strategy symbol attribution exists for paper via `paper_orders`.** The strategy's
  own order history (`SELECT DISTINCT symbol FROM paper_orders WHERE strategy=?`) is the
  per-strategy set of symbols it could be holding — including a held-but-dropped symbol it
  traded before its universe changed. This is the paper analog of the live lane's
  `believed_positions` (its own `live_fills`), and it lets the paper flatten scope to the
  strategy's own symbols rather than the account-wide broker snapshot (which would close a
  sibling strategy's positions on a shared paper account).

- **The operator-trap.** After a live breach the offset *orders* are recorded but their
  *fills* land async; `believed_positions` stays non-flat until the next `ingest_activities`.
  The kill-switch is tripped (can't tick), and `resume`'s live flat-gate refuses while belief
  is non-flat — so without an ingest the operator is deadlocked. Today the only escape is the
  test manually `DELETE`-ing `live_fills`.

## Design

### 1. Paper `flatten` + `trade-tick` breach handler — per-strategy close + belief reset

Two changes, applied at both `paper_cmd.py` sites (the `flatten` command and the
`trade-tick` breach handler) via one shared helper so they cannot drift:

**(a) Close the strategy's OWN symbols (Gap 1), not the account.** Replace
`broker.close_positions(strategy.universe)` with a close over
`universe ∪ {DISTINCT symbol FROM paper_orders WHERE strategy=?}`:

```python
def _strategy_held_symbols(conn, strategy: str, universe: list[str]) -> list[str]:
    """Universe ∪ every symbol THIS strategy has submitted a paper order for — so a
    held-but-dropped symbol (traded before its universe changed) is still exited, WITHOUT
    closing a sibling strategy's positions on a shared paper account. Closing a flat name is
    a 404 no-op, so over-including never-filled names is harmless."""
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM paper_orders WHERE strategy = ?", (strategy,)
    ).fetchall()
    return sorted(set(universe) | {r["symbol"] for r in rows})

# at both call sites, after cancel_open_orders():
broker.close_positions(_strategy_held_symbols(conn, name, strategy.universe))
```

This is per-strategy (the paper analog of the live lane's `believed_positions`) and
strictly safer than the account-wide broker snapshot: a symbol the strategy never traded
(a sibling's, or a manual position) is left untouched. **Residual limitation (honest):** on
a *shared* paper account where two strategies' universes overlap, a symbol BOTH have traded
is in both their `paper_orders`, so flattening one can still close the other's position in
that shared symbol. `paper_orders` proves the strategy *traded* a symbol, not that it *owns*
the current account position — truly per-strategy paper flatten needs real wall-clock paper
position attribution (deferred). Net: this catches the held-but-dropped safety gap and
removes the never-traded-sibling over-close; the shared-symbol residual is documented, not
claimed solved.

**(b) Reset the stale belief on flatten (Gap 2, paper lane).** After the broker close
succeeds, clear the strategy's derived paper positions so `derive_positions` returns flat
and a subsequent `trade-tick` reconcile does not re-trip on stale sim positions:

```python
# order_state.py — extracted from persist_run's existing first statement
def clear_derived_positions(conn, strategy: str) -> None:
    conn.execute(
        "DELETE FROM paper_fills WHERE order_id IN "
        "(SELECT id FROM paper_orders WHERE strategy = ?)", (strategy,))
    conn.commit()
```

`paper_orders` rows are kept (the audit trail / the symbol source in (a)); only the derived
fills are dropped. Ordering is fail-safe: trip → cancel → close → (on close success) clear
belief. A close failure leaves both the broker and the belief untouched and the strategy
halted. The closed symbol set is emitted in the JSON payload and audit row so the exit is
visible.

### 2. `resume` (Stage.LIVE) + `resume-all` — ingest + reconcile against BROKER truth

Before clearing the kill-switch, ingest pending broker activities so the async offset fills
land in `live_fills`, **then verify the strategy is flat at the broker, not just in the
ledger.** The ledger alone is insufficient — a missed/late activity or a manual position can
leave `believed_positions == {}` while the account is still exposed (GATE-1 finding). Broker
truth is the real "zero drift" guarantee.

- **Read-only live client.** Add a validated read-only live client exposing
  `account_activities(after)` and `get_positions()`, reusing the same Alpaca-live host
  validation as the trading broker (do NOT hand-roll a bare `requests` call that could skip
  the host allowlist). No `LiveAuthorization` / go-live signature — both endpoints are GETs.
  Reuse `fill_cursor` + `ingest_activities` so dedupe/cursor semantics match the trading path.
- **The reconcile (single `resume`, Stage.LIVE):** the broker is account-wide, so per-symbol
  "broker holds ~0" is WRONG when a sibling legitimately holds the same symbol (GATE-1 R2).
  Use an account-wide reconcile, the same truth `run-all` uses:
  1. ingest pending activities (catch up offset fills);
  2. read broker net positions (`get_positions`);
  3. compute the **unexplained residual** = broker net − Σ `believed_positions` over **all**
     live strategies (per symbol), within tolerance;
  4. **flat iff** this strategy's own `believed_positions(name)` is empty **AND** there is no
     unexplained residual in any of the strategy's symbols
     (`universe ∪ its live_orders.symbol ∪ its live_fills.symbol`) — i.e. the broker's
     holding of those symbols is fully accounted for by other strategies' books;
  5. if not flat → **refuse**, emitting the residual (this strategy's `believed_positions`
     plus any unexplained broker qty in its symbols) with guidance ("offset fills pending or
     liquidation incomplete — re-flatten or retry after fills land");
  6. if flat → proceed to the existing peak-rebase + kill-switch reset.
  Paper/forward_tested resume is unchanged (no live ledger, no broker reconcile).
- **`resume-all`:** keep its existing coarse contract — it clears the **global** halt and
  rebases peaks but leaves every per-strategy kill-switch tripped, so a not-flat strategy
  still cannot trade until it individually passes single-`resume`'s broker-truth reconcile.
  The only change: **ingest once (account-wide) BEFORE computing `not_flat`**, so the
  `not_flat` warning reflects post-ingest own-ledger belief (offset fills that have landed no
  longer show as held). It does NOT gate the global-halt clear on whole-account flatness:
  that wall lives in the per-strategy kill-switch + single-`resume` reconcile, not the coarse
  global switch, and an unowned/manual account residual is not attributable to any strategy
  (operator concern, out of scope). (When no live strategies / no live creds, unchanged.)
- **Missing creds / broker read failure:** resume of a live strategy refuses with a clear,
  JSON-wrapped message (cannot confirm flat without broker access). `resume` / `resume-all`
  decorators widen to catch `BrokerError`/`ValueError` so an ingest/read failure surfaces as
  a clean envelope distinct from "still not flat", never an uncaught traceback.

Ordering keeps the existing fail-safe discipline: the un-halt write stays last, so any
earlier failure (ingest error, broker-read error, still-not-flat) leaves the strategy safely
halted and resume is retryable.

### 3. End-to-end test (the core deliverable)

A real integration test of the live breach→flatten→resume chain (not mock-`believed_positions`):

- **Seed `live_fills`** with real fills via the ingest path / direct rows: `AAA +5` (in the
  strategy universe) **and** `ZZZ +3` (held-but-dropped — not in the universe), each mapped
  to a `live_orders` row so attribution works.
- **Fake live broker**: `run_tick` raises a gross/drawdown `RiskBreach`; `submit_offset`
  records `(symbol, qty)` and returns an order id; `account_activities(after=…)` returns the
  offset **FILL** activities *after* the offsets are submitted (so a later ingest drops belief
  to flat).
- **Assert** (happy path):
  - kill-switch tripped;
  - offsets submitted for **both** `AAA` and `ZZZ` (held-but-dropped included), sized to the
    believed qty and recorded in `live_orders`;
  - `paper resume` → resume-reconcile ingests the offset fills → `believed_positions` empty
    AND broker `get_positions` flat → resume **succeeds** (zero drift);
  - re-asserts the ledger is flat after resume.
- **Negative / reconcile tests (GATE-1):**
  - *partial offset fill* — broker returns a fill for only part of the offset → belief
    non-flat → `resume` refuses, payload carries the residual qty.
  - *unexplained broker residual while ledger flat* — `account_activities` returns the full
    offset fills (belief → flat) but `get_positions` still reports a nonzero qty in a
    strategy symbol that NO live ledger explains → `resume` refuses (broker-truth reconcile
    catches what the ledger missed).
  - *sibling holds the same symbol (no false block)* — strategy A flat after ingest, but
    sibling B legitimately holds `AAA` (in B's `believed_positions`); the broker's `AAA` is
    fully explained → `resume A` **succeeds** (account-wide reconcile, not per-symbol broker
    zero).
  - *`resume-all` ingests before warning* — two live strategies, one whose offset fills have
    landed (now flat after ingest) and one still holding → global halt clears, peaks rebase,
    and `not_flat` lists only the still-holding strategy (the ingested-flat one drops out).
    Per-strategy kill-switches stay tripped.
- **Paper-path test**: with a fake paper broker, a strategy whose `paper_orders` include a
  symbol no longer in its universe → `flatten` calls `close_positions` with
  `universe ∪ paper_orders-symbols` (the dropped symbol included) and a *sibling* strategy's
  symbol is NOT closed; after flatten, `derive_positions` for the strategy is flat.

### Error handling

- Flatten stays fail-safe: trip first, then cancel + close; a close failure leaves the
  strategy halted (`flatten_error` surfaced) and the belief un-cleared. Belief is cleared
  only after a successful close.
- Resume stays fail-safe: ingest + broker-truth reconcile before the reset; the kill-switch
  reset is the final write; any earlier failure (broker read error, residual) leaves the
  strategy halted and resume retryable.
- Resume of a live strategy refuses on missing creds, a broker read error, or a residual
  (ledger non-flat OR broker still holds a strategy symbol) — the residual is emitted.

### Components touched

- `algua/cli/paper_cmd.py` — `_strategy_held_symbols` helper; `flatten` cmd + `trade-tick`
  breach handler use it + `clear_derived_positions` on close-success; `resume` (Stage.LIVE
  branch) + `resume-all` gain the ingest + broker-truth reconcile; decorators widened to
  `BrokerError`.
- `algua/execution/order_state.py` — `clear_derived_positions(conn, strategy)` (extracted
  from `persist_run`'s existing DELETE).
- `algua/execution/alpaca_broker.py` (or a `_common` factory) — a validated read-only live
  client exposing `account_activities` + `get_positions` with no `LiveAuthorization`,
  reusing the live host validation.
- `tests/test_cli_live.py` (or a new `tests/test_breach_flatten_resume.py`) — the E2E test +
  negative tests.
- `tests/test_cli_paper.py` — the paper-flatten per-strategy scope + belief-reset tests.

### Out of scope (explicit deferrals → follow-up)

- The real wall-clock paper fill ledger / true per-strategy paper *position* attribution
  (the "real wall-clock paper adapter" `persist_run` defers). The sim `paper_fills` and the
  real Alpaca-paper account remain different worlds; this change only keeps the *derived
  belief* consistent (flat after flatten) and scopes the close by the strategy's own
  `paper_orders`.
- Activity-feed pagination / out-of-order-id robustness in `account_activities` +
  `ingest_activities`. This is a **pre-existing** property shared with the live trading
  path, not introduced here; with the broker-truth reconcile (§2), a missed offset fill now
  fails **closed** (broker shows residual → refuse) rather than silently resuming exposed.
  Tracked as a separate hardening follow-up.
- A dedicated `live resume` alias — today `paper resume` owns kill-switch reset for all
  stages. Cosmetic; documented, not changed here.
- Any change to the already-correct live offset/ingest breach logic itself (we test it, not
  rewrite it).

## Quality gate (every commit)

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
