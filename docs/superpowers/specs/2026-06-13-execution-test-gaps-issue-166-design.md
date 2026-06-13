# Execution test gaps + provenance identity primitive (issue #166)

**Date:** 2026-06-13
**Issue:** #166 (P2, rank 7 of repo review 2026-06-10) — `area:execution`, `severity:medium`
**Process:** light (worktree → TDD → quality gate → one multi-model diff review → merge).
Stop-and-flag if any test surfaces a real production bug.

## Problem

`algua/execution/` is the most under-tested package relative to criticality (~1,154 source
lines vs ~316 test lines), and `algua/provenance/lockfile.py` — the dependency-identity
primitive the backtest stamp and the live gate both hang off — has zero dedicated tests.

This change is **additive test coverage only**. No production source changes unless a test
surfaces a genuine bug, in which case work pauses for an explicit human decision.

## Conventions

Follow the existing per-file test convention: each test file defines its own
`_conn(tmp_path)` + `migrate` helper and local fakes. No shared `conftest.py` is
introduced (it would churn ~6 files for marginal gain and widen the conflict surface
against concurrent sessions on `main`). Extend existing files where the subject already
lives; add new files only for genuinely new subjects.

## The six gaps

1. **Fill-before-order-mapping backfill** — `algua/execution/live_ledger.py:85-89`
   (the back-attribution branch of `backfill_broker_order_id`).
   *Where:* extend `tests/test_live_ledger_orders.py`.
   *Test:* `ingest_activities` records a FILL whose `order_id` has no matching
   `live_orders` row yet → the fill lands with `strategy IS NULL`. Then `record_live_order`
   + `backfill_broker_order_id(client_order_id, broker_order_id)` → assert the fill's
   `strategy` attaches, and `live_reconcile.reconcile(broker_net=…)` returns `clean=True`
   (zero drift) for the now-attributed net.

2. **Broker submit timeout-then-retry idempotency** — `algua/execution/alpaca_broker.py`.
   *Where:* extend `tests/test_alpaca_broker.py`.
   *Test:* a fake `requests.post` raises `RequestException` once then returns 201
   (sleep monkeypatched out). Assert: `submit_sized` returns the single order id, and
   **both** POST attempts carry the identical `client_order_id`.
   *Scope note:* this proves only the client-side guarantee (a retried POST re-sends the
   same deterministic id, so Alpaca can dedup). Actual no-double-fill is Alpaca's
   server-side dedup, out of unit-test scope — the test comment says so.

3. **Live-order idempotency** — `algua/execution/order_state.record_submitted_order`.
   *Where:* extend `tests/test_live_loop.py`.
   *Test (the "live replay equivalent"):* a loop-level replay — run the same `run_tick`
   twice against a fake broker that dedups on `client_order_id` (returns the SAME
   `broker_order_id`), with `on_submitted` wired to `record_submitted_order`. Assert
   exactly one `paper_orders` row persists. The existing unit test
   (`test_order_state.py:109-119`) covers the bare INSERT-OR-IGNORE; this covers the live
   loop wiring.

4. **Sizing-equity derealization downstream** — `algua/execution/live_sizing.py`.
   *Where:* extend `tests/test_live_loop.py`.
   *Test:* allocation=10k, fills drive NAV→8k. Route `build_live_sizing_snapshot` through
   `run_tick`'s `live_snapshot` hook; assert the submitted order's notional is sized off
   the derealized **8k**, not the 10k allocation — i.e. the derealized equity actually
   flows downstream into order sizing, not just into the snapshot.

5. **`provenance/lockfile.py`** — currently zero dedicated tests.
   *Where:* new `tests/test_lockfile.py`.
   *Tests:* (a) round-trip determinism — same bytes → same sha256, matching `hashlib`;
   (b) version pinning — changed content → changed hash; (c) absent lockfile →
   `None` (fail-closed). Exercised via `_file_hash` on a temp path and `dependency_hash`
   with `_ROOT` monkeypatched at a temp directory.
   *Interpretation note:* `lockfile.py` is a pure byte-hash with no parsing, so there is
   no "corrupt → parse failure" path. The real fail-closed is *absent → None*; any content
   change simply shifts the hash, which is the correct identity behavior. Tested that way.

6. **Three-way warmup parity** — one unified regression.
   *Where:* new `tests/test_warmup_parity.py`.
   *Test:* for the same strategy + data and `warmup_bars = N`, backtest, paper, and live
   all hold the first N sessions flat and first decide at session index N. The existing
   pairwise tests (`tests/test_decision_parity.py`,
   `tests/test_live_loop.py:115-148`) remain; this adds the single three-way guard.

## Out of scope

- Any production source change (deferred to a flagged human decision if a bug surfaces).
- A shared `conftest.py` / fixture refactor.
- Coverage of execution paths beyond the six listed gaps.

## Verification

Quality gate at every commit:
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
