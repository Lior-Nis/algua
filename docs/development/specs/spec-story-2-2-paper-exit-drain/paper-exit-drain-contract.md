# Story 2.2 paper exit drain contract

Normative companion to [SPEC.md](SPEC.md). Line references are to `e875b6d` (main after PR #686).
"Paper lane" means the stages `paper` and `forward_tested`, which `paper run-all` ticks and the paper
account reconcile counts (`algua/execution/paper_reconcile.py:36-58`). The "live lane" is `live`.
The corrections from the
[2026-10-05 readiness report](../../implementation-readiness-report-2026-10-05-story-2-2.md)
(M1–M3, m1–m11) are applied; the decision log's 2026-10-05 entry maps each one to its section.

## 1. Entry-point inventory

Every path that can move a strategy's stage, judged by whether it takes the strategy off the paper
lane. Verified by reading every caller of `transition_strategy`, `apply_transition` and
`_apply_transition_locked`, every `UPDATE strategies SET stage` in `algua/`, and by running the 32
test files that drive transitions under a probe that recorded every allocation-shedding call
(2026-10-05: 797 passed; 53 allocation-shedding calls, 33 of them from a paper-lane source, none
with a guard).

| # | Path | Code | Edge | Reaches `_assert_flat_for_bench` | Guard today | Guard after 2.2 |
|---|---|---|---|---|---|---|
| 1 | `registry transition NAME --to dormant\|retired\|candidate` (source `paper`) | `cli/registry_cmd.py:181-261`, call `:249` -> `registry/transitions.py:36`, revoke flag `:63`, store call `:97` -> `registry/store/crud.py:279-312`, check `:303` (defined `:319-353`) | `paper -> dormant\|retired\|candidate` | Yes (paper ledger, `crud.py:340`) | No: `_live_exit_guard` returns `None` for a non-live source (`registry_cmd.py:61-63`) | Paper guard (§3) |
| 2 | `registry transition NAME --to retired` (source `forward_tested`) | as row 1 | `forward_tested -> retired` | Yes (paper ledger) | No | Paper guard |
| 3 | `registry transition NAME --to live --actor human --signature F` | `registry_cmd.py:225-250`; live gate `transitions.py:65-74` (`_validate_live_gate`, `:116-157`); store `crud.py:261-267` lets this one edge carry both the authorization and the revoke | `forward_tested -> live` | Yes (paper ledger: source is not `live`, `crud.py:340`) | No | Paper guard, after the live gate (§2.2) |
| 4 | `registry transition NAME --to live` without `--signature` | `registry_cmd.py:203-224` | none (issues a challenge, returns before `transition_strategy`) | No | n/a | n/a |
| 5 | `transition_strategy` from `evaluation/backtest_run.py:105` | target `backtested`; `contracts/lifecycle.py:30-31` gives `paper`/`forward_tested` no such edge, so `validate_transition` (`transitions.py:52`) refuses first | none from the paper lane | No | n/a | Default selector applies to any future allocation-shedding caller (§2.3) |
| 6 | `paper promote NAME` (pass) | `registry/store/forward_gate.py:161-167`, `_apply_transition_locked(..., FORWARD_TESTED)`, no revoke | `paper -> forward_tested` (stays on the lane, `transitions.py:28-29`) | No | none | none (AC6) |
| 7 | `registry transition --to forward_tested` (human raw) / `--to paper` from `forward_tested` | `transitions.py:85-93`; not in `_REVOKE_ON_EXIT` | within the lane | No | none | none (AC6) |
| 8 | `paper intake`, merge-back intake | `registry/store/deployment.py:208` | `candidate -> paper` (entry) | No | n/a | n/a |
| 9 | `research promote` | `registry/store/gate.py:454` | `backtested -> candidate` | No | n/a | n/a |
| 10 | merge-back `_advance_idea_to_backtested` | `registry/mergeback_intake.py:233-250` (raw CAS) | `idea -> backtested` | No | n/a | n/a |
| 11 | schema migration | `registry/db/core.py:75` | `shortlisted -> candidate` | No | n/a | n/a |
| 12 | `paper flatten`, `paper kill`, `paper halt-all`, run-all/trade-tick breach | `cli/paper_cmd.py:1276-1320` and the breach paths | none: kill switch or global halt only; the strategy stays on the lane and its fills keep counting | No | n/a | n/a |
| 13 | `allocations.deallocate` | `registry/allocations.py:127-133` | none (no production caller; an allocation revoked without a stage change leaves the strategy on the lane) | No | n/a | n/a |
| 14 | `dormant -> retired` | registry transition, not in `_REVOKE_ON_EXIT` | strategy already left the lane when benched (and, after 2.2, drained then); production holds no dormant strategy (registry copy, 2026-10-05) | No | n/a | n/a |
| 15 | Live-source exits `live -> paper\|dormant\|retired` | row 1's call path with the live guard | live lane | Yes (live ledger) | Live guard | Live guard, unchanged (§2.4) |
| 16 | `web/backend` monitor, research/leap/forage drivers | read-only / forbidden from these transitions | none | No | n/a | n/a |

Direct `repo.apply_transition(..., revoke_allocation=True)` calls exist only in `transitions.py:97`
in production; the store-internal `_apply_transition_locked` callers (rows 6, 8, 9) never revoke. A
new structural test pins both facts (§7, T16).

Re-runs: re-running a refused exit drains again (cancel of a gone order is a 404/422 no-op,
`alpaca_broker.py:419-428`; ingest de-duplicates by activity id, `live_ledger.py:484-538`) and
settles again every order a drain has asked the venue to cancel for the strategy (§3.3 step 6).
Re-running a completed exit fails `validate_transition` before any drain.

## 2. Edge set, selection and call order

### 2.1 Which edges are guarded

- The guarded edges are exactly `_REVOKE_ON_EXIT` (`transitions.py:30-33`). The five with a
  paper-lane source get the paper guard; the three with source `live` get the live guard. No other
  edge gets a guard, so `paper -> forward_tested` and `forward_tested -> paper` keep their slice
  without a drain (AC6).
- `_REVOKE_ON_EXIT` equals the set of allowed edges whose source is in a lane's operating stages and
  whose target is outside that lane (paper lane `{paper, forward_tested}`, live lane `{live}`),
  computed from `ALLOWED_TRANSITIONS`. Verified against the code on 2026-10-05; T16 pins it so a new
  lifecycle edge that leaves a lane cannot be added without its drain.

### 2.2 Go-live (`forward_tested -> live`)

The go-live edge leaves the paper book, so it gets the **paper** guard: the strategy's resting
orders are paper orders, and the store already checks its paper positions on this edge. The live-side
walls are unchanged and run first: actor, frozen-deployment refusal, forward certificate and
signature verification (`transitions.py:65-74`), all before the operator lock and before any venue
call, so an unauthorized or uncertified attempt cancels nothing. The challenge is consumed under the
write lock (`store/base.py:48-78`) after the drain's re-check, so a refused drain leaves the challenge
unconsumed and the same signature can complete later while it is unexpired (a refusal for a held
`operator.lock` is covered in §2.5). With production wiring a missing paper credential is refused
first by the certificate verifier's own message (`transitions.py:256-267`); the drain's unavailable
branch is reached on this edge only with an injected verifier. The live ledger is not drained on go-live: a `forward_tested` strategy has no live
orders, and any earlier live tenure ended through the live drain.

### 2.3 `transition_strategy`

```python
ExitGuardSelector = Callable[[StrategyRepository, str, Stage, Stage], ExitLaneGuard]

def transition_strategy(
    repo, name, to, actor, reason=None, approval_verifier=None,
    forward_certificate_verifier=None, exit_guard_selector: ExitGuardSelector | None = None,
) -> StrategyRecord: ...
```

- `ExitGuardSelector` is declared beside `ApprovalVerifier` and `ForwardCertificateVerifier`. The
  `exit_guard` parameter is removed (no alias, no dual path).
- Order, unchanged up to the lock: read `rec`; the `candidate -> paper` refusal; `validate_transition`;
  the dormant-reason check; the live, shortlist or forward-gate validation. Then:

  ```python
  with operator_transition_lock(rec.stage, target):
      guard = ((exit_guard_selector or _default_exit_guard_selector())(repo, name, rec.stage, target)
               if revoke_allocation else None)
      return repo.apply_transition(..., revoke_allocation=revoke_allocation, exit_guard=guard)
  ```

- The selector receives `rec.stage`, the stage the store's compare-and-swap checks
  (`store/base.py:121-130`). The guard's lane therefore always matches the source the exit commits
  from; selecting in the CLI from a separate read, as today, could pair one lane's guard with another
  lane's exit if the stage moved between the two reads.
- `_default_exit_guard_selector()` lazily imports and returns `algua.execution.lane_exit.
  select_exit_guard` (the same lazy, module-attribute seam as `_default_forward_certificate_verifier`,
  `transitions.py:233-277`). Every caller of `transition_strategy` gets both lanes' drains without
  passing anything; no production module passes `exit_guard_selector` (T16).
- `registry_cmd.transition` calls `transition_strategy(repo, name, target, Actor(actor), reason,
  approval_verifier=verifier)`. `_LIVE_EXIT_TARGETS`, `_live_exit_guard` and their imports are deleted.

### 2.4 `select_exit_guard(repo, name, source, target) -> ExitLaneGuard`

In `algua/execution/lane_exit.py`.

- `conn = getattr(repo, "connection", None)`; `None` raises `TransitionError("the exit drain needs a
  sqlite-backed repository")`.
- `source is Stage.LIVE`: the body of today's `_live_exit_guard` from `verify_live_authorization`
  onward (`registry_cmd.py:64-93`), moved verbatim with one exception: it calls
  `live_gate.verify_live_authorization(conn, repo, name, live_gate.ALLOWED_SIGNERS_PATH)`, reading
  the allowed-signers path through the module at call time instead of a from-import, so a test
  re-targets one attribute (`algua.registry.live_gate.ALLOWED_SIGNERS_PATH`; no test patches
  `algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH` before a live exit today). The branch returns the
  authorized broker, else the account-credential drain broker with the
  `live_exit_drain_account_creds` audit, else writes the `live_exit_drain_unavailable` audit and
  raises its `TransitionError`. Messages and audit actions are byte-identical.
- **Live timing change.** Today `_live_exit_guard` runs in the CLI before `validate_transition`,
  before the dormant-reason check and before the lock. So `live -> dormant` without a reason, or
  `live -> retired` while `operator.lock` is held, builds a broker and can write a
  `live_exit_drain_account_creds` or `live_exit_drain_unavailable` row before it is refused. After 2.2
  the selector runs inside the lock, after every validation (§2.3), so those refusals build no
  broker and write no `live_exit_drain_*` row. The messages are unchanged, but the audit side effects
  of a refused live exit are not byte-identical to today. T15 pins both cases.
- `source in {Stage.PAPER, Stage.FORWARD_TESTED}`: `broker = paper_exit_drain.build_paper_drain_broker()`,
  called through the module attribute (the test seam). On `None`, audit `paper_exit_drain_unavailable`
  and raise the unavailable `TransitionError` (§4). Otherwise return
  `paper_exit_drain.PaperExitGuard(conn, broker, name)`.
- Any other source raises `ValueError(f"no exit drain for {source.value} -> {target.value}")`
  (unreachable from `transition_strategy`).
- `build_paper_drain_broker() -> AlpacaPaperBroker | None`, in `algua/execution/paper_exit_drain.py`
  (§6), is `maybe_broker(BrokerKind.ALPACA_PAPER)` (`broker_factory.py:152-181`). A malformed setting
  (pydantic `ValidationError`, a host-pinning `BrokerError`) propagates unchanged and unaudited, as
  on every paper command.

There is no paper break-glass. Paper has one credential set, so there is no account-level fallback;
`paper flatten` needs the same credentials. The remedy is to configure them.

### 2.5 Operator lock

`algua/operator/deployment_lock.py`: `deployment_retirement_lock` is renamed
`operator_transition_lock(from_stage, target)`. It takes `operator.lock` (non-blocking,
`operator/schedule.py:105-125`) when `target is RETIRED`, or when `from_stage` is in the paper lane
and `target` is not. That adds `paper -> dormant` and `forward_tested -> live` to today's set
(`paper -> candidate`, every `-> retired`). Messages:

- edges locked today: `"operator.lock is held; deployment retirement cannot interleave with a paper
  tick"` (unchanged bytes; `live -> retired` is a live exit, AC6);
- the two new edges: `"operator.lock is held; a paper-lane exit cannot interleave with a paper
  tick"`.

The lock is acquired before the selector runs, so a held lock refuses before any broker is built,
any audit row is written or any venue call is made. The paper timer fires every 20 minutes and treats
a held lock as a benign no-op (`cli/operator_cmd.py:279-313`). Merge-back holds the same lock for its
whole saga (`operator_cmd.py:248`), so an exit is refused while merge-back holds it (up to
`TimeoutStartSec=3600`, `deploy/systemd/algua-mergeback-drain.service:21`) and is retried, as
retirement is today. The flock is non-blocking: no exit waits for the lock, and a nested acquire in
one process is refused, not deadlocked.

- **Scope.** Exclusion holds only when the command runs from the operator's checkout (production: the
  main checkout, the `algua-paper.service` WorkingDirectory). The lock path is derived from the git
  dir of the checkout whose code runs (`deployment_lock.py:15-23`), so a command run from a worktree
  resolves a different `operator.lock` and excludes nothing.
- **Go-live and the challenge TTL.** A go-live completion refused because `operator.lock` is held
  (`TransitionError`, `wrong_stage`, §4) leaves its challenge unconsumed. The challenge expires 10
  minutes after issue (`registry/challenges.py:32`), and a merge-back can hold the lock longer. There
  is no lock wait and no TTL extension: if the challenge expires before the lock is free, the human
  issues a new challenge (`registry transition NAME --to live --actor human` without `--signature`),
  signs it and completes. Check that the paper and merge-back services are idle before requesting the
  challenge.

## 3. The paper exit guard

### 3.1 Broker capability

`algua/execution/paper_exit_drain.py` (new, §6) declares, beside `PaperExitGuard`:

```python
@runtime_checkable
class PaperExitDrainBroker(ScopedCancelBroker, ActivityWindowBroker, OrderLookupBroker, Protocol):
    def clock(self) -> str: ...
```

It composes the contract protocols from `algua.contracts.types`, which is not edited: that module is
at its ratchet pin (468/468, `tests/test_module_size_ratchet.py:68`). The protocol lives with the
guard, not in `lane_exit.py`: `lane_exit.py` imports `paper_exit_drain.py` for the selector, so a
protocol declared in `lane_exit.py` would make the two modules import each other.

`AlpacaPaperBroker` satisfies it (`list_open_orders` `:394-400`, `cancel_order` `:419-428`,
`account_activities_window` `:450-478`, `get_order_by_client_order_id` `:402-417`, `clock`
`:439-446`). `cancel_open_orders` (`:233-246`) is outside the protocol.

### 3.2 Venue sync

`ingest_paper_venue` and `recover_stranded` (with `_PAPER_CURSOR_FAR_PAST`) move unchanged from
`algua/cli/paper_venue.py:43-60,69-78` to a new `algua/execution/venue_sync.py`; `paper_cmd.py`
and `cli/paper_venue.py` (`live_strategy_flat`) import them from there. No re-export. `paper_scoped_cancel`
stays in `cli/paper_venue.py`. The paper operator keeps `ingest_paper_venue`, which advances the
shared cursor to its `until`.

**The drain never moves the shared paper cursor.** `venue_sync.py` gains the drain's own ingest:

```python
def ingest_paper_venue_keep_cursor(
    conn: sqlite3.Connection, broker: ActivityWindowBroker, until: str
) -> None:
    after = fill_cursor(conn, LedgerKind.PAPER) or _PAPER_CURSOR_FAR_PAST
    acts = broker.account_activities_window(after, until)
    ingest_activities(conn, acts, LedgerKind.PAPER, cursor_value=after)
```

- It fetches `(cursor, until]` exactly like `ingest_paper_venue`, then stores `after`, the value
  every reader already derives from the row. The only reader is `fill_cursor(...) or
  _PAPER_CURSOR_FAR_PAST` in these two functions. An existing cursor keeps its bytes. An absent or
  empty row becomes the far-past sentinel, which reads identically.
- Re-fetching is idempotent, because ingest de-duplicates by activity id (`live_ledger.py:484-538`).
  A fill the venue publishes after a drain sync is therefore still inside the next sync's window,
  whether that sync is the operator's or a retried exit's.
- `cursor_value=None` is forbidden here. On that path `ingest_activities` stores the highest activity
  id (the live ledger's cursor form), which would corrupt the paper time cursor.

The guard's *sync* is the paper operator's ingest pairing (`paper_cmd.py:980-981`) with the drain's
ingest in place of the operator's:

1. `until, source = tick_clock(broker.clock)`; `source != "broker"` is a drain failure at step
   `clock`. The local-clock fallback is refused: a local `until` behind the venue's clock would end
   the fetch window before fills the venue has already published, and the ordering in §3.3 step 5
   would no longer be anchored to the venue;
2. `ingest_paper_venue_keep_cursor(conn, broker, until)` then `recover_stranded(conn, broker,
   LedgerKind.PAPER)` (step `ingest`).

### 3.3 `PaperExitGuard.cancel_and_ingest()` (before the lock)

`PaperExitGuard(conn, broker, name, *, sleep=time.sleep)`; `sleep` is the test seam for step 4's
re-list interval. `state` starts `new`; `unsettled` and `unpublished` start empty.

1. Sync.
2. If the strategy holds a material paper position (`believed_positions(conn, name, PAPER)` minus
   every `paper_dust` residual: the expression at `crud.py:341-343`), set `state = skipped` and
   return. Nothing is cancelled and nothing is settled: the exit will be refused on positions, and
   resting liquidation offsets must survive a premature exit.
3. `owned = owned_open_order_ids(conn, broker, name, kind=PAPER)` (step `open_orders`).
4. If `owned` is non-empty:
   1. For each `oid` in `owned`, in order: append the audit row `paper_exit_drain_cancel_requested`
      (§4; its reason is exactly `oid`), then `cancel_order(oid)` (step `cancel`). The row is
      committed before the DELETE is sent, so neither a crash nor a failure after the DELETE can lose
      the record that the drain asked the venue to cancel that order.
   2. `unsettled = owned_open_order_ids(...)` (step `recheck`). While `set(unsettled) & set(owned)`
      is non-empty and fewer than 3 further re-lists have run: `sleep(1.0)`, then `unsettled =
      owned_open_order_ids(...)` (step `recheck`). Then `unsettled` is frozen. Alpaca's `status=open`
      includes `pending_cancel`, so an accepted cancel can still be listed for a moment; the bounded
      re-list keeps that from refusing the exit spuriously. A `422` (not cancelable) is a no-op at
      the broker, so such an order stays in `unsettled`.
5. Sync again. Its `until` is read after the last observation of the strategy's open orders (step 3
   when nothing was owned, step 4.2's last re-list otherwise).
6. **Settle** (step `settle`). `requested` is the set of `reason` values of every
   `paper_exit_drain_cancel_requested` row for this strategy (`audit.log.read(conn, strategy=name,
   actor="system", action="paper_exit_drain_cancel_requested")`). It covers this drain and every
   earlier drain of the strategy. That makes it durable: an order whose fill was unpublished when an
   earlier attempt was refused is no longer open, so a retry's `owned` would not see it again. For
   each `oid` in `sorted(requested - set(unsettled))`:
   1. `coid` = `SELECT client_order_id FROM paper_venue_orders WHERE broker_order_id = ? AND
      strategy = ?` (`oid`, `name`). If there is no row, raise `ValueError(f"no paper order row maps
      broker order {oid!r} to {name!r}")`. Step 5's stranded-order recovery has already backfilled
      any crash-stranded row, so a row that is still unmapped is an anomaly for the operator.
   2. `order = broker.get_order_by_client_order_id(coid)`. `None` (a 404) raises
      `ValueError(f"the paper venue has no order with client_order_id {coid!r}")`. A timeout, a
      transport error or a non-404 failure is the broker's own `BrokerError`, raised after its three
      attempts (`alpaca_broker.py:115-148`, `:402-417`).
   3. `str(order.get("id")) != oid` raises `ValueError(f"the paper venue's order for client_order_id
      {coid!r} has id {order.get('id')!r}, expected {oid!r}")`.
   4. `venue = float(order["filled_qty"])`. A missing, non-numeric, negative or non-finite value raises
      `ValueError(f"order {oid!r}: bad filled_qty {order.get('filled_qty')!r}")`.
   5. `held = abs(SELECT COALESCE(SUM(qty), 0.0) FROM paper_venue_fills WHERE broker_order_id = ?)`.
      All fills of one order have the same sign, so this is the quantity the ledger holds for that
      order, whichever strategy the fills are attributed to.
   6. If `venue - held > DUST_SHARES` (`execution/dust.py:13`, 1e-6 shares, the Story 1.4
      float-noise rule; it is above Alpaca's 1e-9-share quantum, so parsing the venue's 9-decimal
      strings never trips it), add `oid` to `unpublished`. If `held - venue > DUST_SHARES`, raise
      `ValueError(f"order {oid!r}: the paper ledger holds {held} shares but the venue reports
      {venue} filled")`. Otherwise the order is settled.

   No `oid` in this loop was open at the last observation. An earlier drain's order that is still open
   is in this drain's `owned`, and so in `unsettled` while it stays open. Every order checked is
   therefore closed at the venue, and its `filled_qty` is final. With no requested rows, the step
   makes no venue call.
7. `state = drained`.

Every step failure is handled by §4. On return the connection has no open transaction.

### 3.4 `PaperExitGuard.owned_open_order_ids()` (under the lock)

- `state == new`: `RuntimeError("paper exit re-check ran before its drain")`.
- `state == skipped`: `TransitionError(f"{name} is not flat (open paper positions when the exit
  drain ran); flatten before this transition")`. Normally unreachable: the store's positions check
  runs first and refuses with the symbols. It fires only if a concurrent ingest flattened the ledger
  in between.
- `unpublished` non-empty: `TransitionError(f"{name} is not flat (the paper venue reports fills on
  order(s) {sorted(unpublished)} that the paper ledger does not hold yet); retry this transition once
  the venue publishes them")`, with no broker call. The store's positions check ran first and found
  the ledger flat, because the fill is not in it yet. A retry's sync ingests the fill once it is
  published (§3.2), and the retry then commits or is refused on positions.
- `unsettled` non-empty: return `sorted(unsettled)` without a broker call; the store refuses with
  its open-order message.
- Otherwise return `sorted(owned_open_order_ids(...))`. A `sqlite3.Error` propagates; any other
  exception becomes the under-lock `BrokerError` (§4), unaudited.
- It never writes to the database and never commits.

### 3.5 Invariants

- An exit commits only if all three hold:
  - after a sync whose `until` was read after the final pre-lock observation showed none of the
    strategy's orders open, the ledger holds no material paper position;
  - for every order any drain has asked the venue to cancel for the strategy, the ledger holds the
    venue's filled quantity within `DUST_SHARES`;
  - a re-list under the write lock shows none of the strategy's orders open.
- The drain never moves the shared paper fill cursor (§3.2); only the paper operator's ingest
  advances it.
- Only orders whose `client_order_id` maps to the strategy in `paper_venue_orders`
  (`live_ledger.py:605-620`) are cancelled. `cancel_open_orders` is never called on this path.
- A full page of 500 open orders fails closed (`alpaca_broker.py:398-399`).
- Any owned open order blocks, whatever its size. A residual is flat only by the Story 1.4 rule.
- The guard trips no kill switch: a refused exit leaves the strategy on the lane, and the next
  paper tick re-plans it.

## 4. Outcomes, audit and envelope

No new error codes; `docs/contracts/cli-error-envelope.md` is unchanged. Success JSON is unchanged:
`{"ok": true, "name", "stage"}`. `wrong_stage` and `broker_error` stay non-retryable in the envelope,
as today, even where the message says to retry.

| Condition | Where | Audit action (actor `system`, strategy = name) | Raised | `code` |
|---|---|---|---|---|
| `operator.lock` held | `operator_transition_lock` | none | `TransitionError` (§2.5 messages) | `wrong_stage` |
| Paper credentials not configured | selector | `paper_exit_drain_unavailable`, reason `"Alpaca paper credentials not configured; cannot drain resting paper orders"` | `TransitionError(f"cannot exit paper-lane strategy {name!r}: Alpaca paper credentials are not configured, so its resting paper orders cannot be drained; set ALGUA_ALPACA_API_KEY and ALGUA_ALPACA_API_SECRET, then retry")` | `wrong_stage` |
| Step `clock`, `ingest`, `open_orders`, `cancel`, `recheck` or `settle` raises any `Exception` other than `sqlite3.Error`. This includes a full open-order page, a malformed venue payload, and the §3.3 step 6 settle failures: a by-coid 404, a lookup timeout or 5xx, a bad `filled_qty`, an id mismatch, an unmapped order and a ledger holding more than the venue | `cancel_and_ingest` | `paper_exit_drain_failed`, reason `f"{step}: {type(exc).__name__}: {exc}"` | `BrokerError(f"cannot exit paper-lane strategy {name!r}: its exit drain failed at {step} ({type(exc).__name__}: {exc}); its stage and allocation are unchanged; retry when the paper venue is reachable") from exc` | `broker_error` |
| `sqlite3.Error` during the drain | `cancel_and_ingest` | none (the audit write would meet the same fault) | unchanged | `db_unavailable` (retryable) for `OperationalError` |
| A cancel is about to be sent | `cancel_and_ingest` step 4.1, one row per order, committed before its DELETE | `paper_exit_drain_cancel_requested`, reason exactly the broker order id (machine-read by the settle step; no other text) | none | n/a |
| Material paper position | store, under the lock (`crud.py:344-347`) | none | existing `f"{name} is not flat (open paper positions {sorted(held)}); flatten before this transition"` | `wrong_stage` |
| Venue reports a fill the ledger does not hold yet, on an order a drain asked to cancel | guard (§3.4) | none | `TransitionError` (§3.4 `unpublished` message) | `wrong_stage` |
| Own order open after the cancel and its re-lists, or in the under-lock re-list | store (`crud.py:348-353`) | none | existing `f"{name} is not flat ({len(open_ids)} open paper order(s) {open_ids}); flatten before this transition"` | `wrong_stage` |
| Drain skipped, ledger flat under the lock | guard (§3.4) | none | `TransitionError` (§3.4) | `wrong_stage` |
| Under-lock re-list fails | guard (§3.4) | none: the transaction rolls back, and `audit.log.append` would commit it half-way (AC3 as amended) | `BrokerError(f"cannot exit paper-lane strategy {name!r}: re-listing its open paper orders under the registry lock failed ({type(exc).__name__}: {exc}); the transition was rolled back; retry") from exc` | `broker_error` |
| Re-check before drain (caller bug) | guard | none | `RuntimeError` | `internal` |

Every refusal leaves stage, allocation, deployment, gate tokens and go-live challenge unchanged.
Orders whose cancel was requested before a refusal stay cancelled at the venue, and their
`paper_exit_drain_cancel_requested` rows record them. A row records that a cancel was requested,
not that it succeeded: `cancel_order` is silent on 404/422.

## 5. Interactions

- **Paper operator.** `operator.lock` (§2.5) serializes every paper-lane exit with timer-driven
  `paper run-all` and merge-back, when the exit runs from the operator's checkout. A paper command
  run by hand outside the wrapper can still submit for the strategy after the exit commits; that
  stays operator discipline, as `run-all`'s own docstring already states (`paper_cmd.py:917-918`).
  The drain's ingest re-stores the cursor it read and never advances it (§3.2). If a hand-run paper
  ingest advances the cursor in between, the drain's re-store moves it back. That only re-fetches an
  overlap, which ingest de-duplicates.
- **Dust.** A dust residual is flat for both the skip decision and the store check (one rule,
  `execution/dust.py:18-29`). A resting order is never dust.
- **Story 1.4 legacy exit.** The flatten-wait-retire procedure keeps working. Production (read-only
  copy of the registry, 2026-10-05): 17 tenants retired; `liquidity_stable_quality_momentum` is still
  at `paper`, kill-switched by `flatten` at 2026-10-03T13:51Z, believing 3.552855 UNH (material),
  with a resting UNH sell offset (the order row does not record its quantity; Story 1.4 caps it at
  the ledger net, 3.55273207). Retiring it before that offset fills is refused on positions and
  leaves the offset resting (§3.3 step 2). If the offset fills in full, the remaining 0.000123 share
  residual is dust, and the retirement drains nothing and commits. No drain has asked to cancel any
  of its orders, so the settle step makes no venue call. Retiring through this guard needs the paper
  credentials the box already has, must run from the main checkout (§2.5), and is refused while a
  merge-back attempt holds `operator.lock`.
- **Frozen tenants.** A frozen deployment's orders go through the same `paper_venue_orders` ledger,
  so its exits drain identically. Go-live stays refused for frozen deployments
  (`frozen_live_unsupported`) before any drain.
- **Rollback runbook.** `deploy/systemd/README.md`'s "retire every frozen deployment" step now
  drains; it needs the paper venue reachable and the paper credentials configured.

## 6. Placement, size and protection

- `algua/execution/paper_exit_drain.py` (new) holds `PaperExitDrainBroker` (§3.1), `PaperExitGuard`
  and `build_paper_drain_broker`. The paper guard was first placed in `lane_exit.py`. The readiness
  prototype of that file reached 211 lines without docstrings. The settle step, the bounded re-list
  and docstrings at the repository's density bring it to about 310, past the 300-line ratchet floor
  for an unpinned module. So the paper guard has its own module, and no pin is added.
  `paper_exit_drain.py` may import `algua.contracts`, `algua.execution.{alpaca_broker,
  broker_factory, live_ledger, tick_clock, dust, venue_sync, errors}` and `algua.audit.log`. It does
  not import `algua.registry`.
- `algua/execution/lane_exit.py` holds `LiveExitGuard` (unchanged), `build_live_broker`,
  `build_live_drain_broker` and `select_exit_guard`. It may import `algua.contracts`,
  `algua.execution.{alpaca_broker, broker_factory, live_ledger, paper_exit_drain, errors}`,
  `algua.audit.log` and `algua.registry.live_gate`.
- Neither module may import `algua.execution.flatten`, `order_state`, `live_reconcile`, or anything
  under `algua.live` or `algua.cli`. The existing contract "registry stays off the live lane" enforces
  this transitively, because import-linter follows the lazy import in `transitions.py`. This was
  verified with grimp on 2026-10-05 and mutation-checked by the readiness review: an `algua.live`
  import in `venue_sync` broke that contract through the lazy chain. No contract changes and no
  exemption.
- `algua/execution/venue_sync.py` holds the two moved functions and `ingest_paper_venue_keep_cursor`
  (§3.2). It must not import `algua.live` (it does not take `live_strategy_flat`, which needs
  `live_loop._RECONCILE_TOL`).
- `algua/contracts/types.py` is not edited. It is at its ratchet pin (468/468), and the protocol
  lives in `paper_exit_drain.py` (§3.1).
- `transitions.py` (277 lines) must stay at or below 290 lines; it is unpinned and the ratchet
  forbids a new module at 300 or more.
- `registry_cmd.py` loses the selection (about 60 lines). Lower its pin in
  `tests/test_module_size_ratchet.py:59` to the new size in the same commit.
- `paper_exit_drain.py`, `lane_exit.py`, `venue_sync.py` and `deployment_lock.py` each stay under
  300 lines with no pin; if one would reach 300, the implementer carves further instead of adding a
  pin. `store/crud.py`, `dust.py`, `alpaca_broker.py`, `live_ledger.py` and `audit/log.py` are not
  edited, except the docstring of `_assert_flat_for_bench` (`crud.py:329-334`), which says the drain
  is wired for the live lane only.
- CODEOWNERS and `INTEGRITY_CRITICAL_MODULES` (`tests/test_repo_hygiene.py:237`) gain four modules,
  and the store trusts all four:
  - `algua/execution/paper_exit_drain.py`;
  - `algua/execution/lane_exit.py`, which is not protected today;
  - `algua/execution/venue_sync.py`, whose code was protected in `cli/paper_venue.py`;
  - `algua/operator/deployment_lock.py`.

  `cli/paper_venue.py` stays protected.
- **Merge order: Story 2.2 merges before Story 2.1** (coordinator decision, 2026-10-05; readiness
  report, "Story 2.1 Coordination"). Story 2.1 (v49, forward-only) then rebases onto this story:
  - it deletes T14's human-raw `paper -> forward_tested` case, because 2.1 refuses that edge before
    the selector;
  - it reuses T16's single-caller `.apply_transition(` assertion instead of adding its own;
  - it keeps `exit_guard=guard` in the `apply_transition` call.

  The textual conflicts it resolves:
  - `transitions.py`: the `apply_transition` call, the deleted branches directly above the lock
    block, and the `transition_strategy` signature;
  - `registry_cmd.py`: the imports and the issuance edit;
  - adjacent CODEOWNERS and `INTEGRITY_CRITICAL_MODULES` insertions;
  - `tests/test_book_exit_revoke.py:300-335`.

  If the order inverts anyway, this story's side is: drop inventory row 7's raw
  `paper -> forward_tested` and T14's human-raw case, and keep a single `.apply_transition(`
  single-caller test (2.1 adds the same assertion).

## 7. Tests

New tests, with fake venues (no network; the suite-wide guard in §8 refuses any Alpaca host).
`FakePaperVenue` implements §3.1 and records calls; its `cancel_open_orders` raises `AssertionError`.
It answers `get_order_by_client_order_id` with the order's `id`, `client_order_id`, `symbol` and
`filled_qty`, and it can withhold a fill activity from the feed until the test releases it.

- T1 Each of the five paper-source edges through the default selector: without a resting order the
  exit commits, revokes the allocation and makes no by-coid call. With an own resting order and no
  material position:
  - the order is cancelled;
  - its `paper_exit_drain_cancel_requested` row exists when its DELETE reaches the fake;
  - the settle step reads it by client_order_id with `filled_qty` `"0"`;
  - the exit commits.
- T2 A sibling's resting order on the same account is neither cancelled nor blocking.
- T3 Fill between cancel and ingest, published at once: the fake fills the order on cancel and
  publishes the activity. The second sync attributes it, the settle step finds the ledger holding the
  venue's `filled_qty`, and the exit is refused on positions with everything unchanged.
- T3b Late publication: the fake fills the order at cancel (DELETE 422; by-coid `filled_qty` `"2"`)
  and withholds the fill activity. The exit is refused with the §3.4 `unpublished` message. Stage,
  allocation, deployment, tokens and challenge are unchanged, and no kill switch is set.
- T3c Immediate retry, still unpublished: the order is no longer open, so `owned` is empty. The exit is
  refused again with the same message, because the settle set comes from the durable audit rows.
- T3d The fake releases the activity, with a `transaction_time` earlier than the first attempt's
  `until`. A retry's first sync ingests it:
  - (i) a fill of 0.0005 shares at $100 (above `DUST_SHARES`, below the $1 venue minimum) leaves the
    ledger flat by the Story 1.4 rule; settle passes and the exit commits;
  - (ii) a fill of 2 shares makes the drain skip, and the exit is refused on positions.
- T3e Cursor: across T1, T3–T3d, a skipped drain and a failure at each step, the
  `paper_venue_fill_cursor` row is byte-identical before and after `cancel_and_ingest`, while the
  fills it fetched are in the ledger. With no cursor row, the row reads `_PAPER_CURSOR_FAR_PAST`
  afterwards. The operator's `ingest_paper_venue` still advances the cursor to its `until`.
- T4 Non-cancelable order (the cancel is a no-op; the order stays open through every re-list): refused
  under the lock with the open-order message, after 1 + 3 re-lists and three injected 1-second sleeps.
- T4b An order open at the recheck but gone from the under-lock re-list is still refused.
- T4c `pending_cancel`: the order is still listed at the first recheck and gone at the second. The exit
  commits after one injected 1-second sleep.
- T5 Material position plus a resting offset: refused on positions; the offset is not cancelled, and
  no cancel-requested row is written. T5b Dust-only residual plus a resting order: cancelled, exit
  commits.
- T6 Credentials absent, each paper-source edge and through the CLI envelope (`wrong_stage`, exit 1):
  `paper_exit_drain_unavailable` row, no broker call, nothing changed.
- T7 Failure at each step: `paper_exit_drain_failed` row naming the step, `BrokerError`, CLI
  `broker_error`, nothing changed. The steps:
  - `clock`, via a raising or a naive clock;
  - `ingest`;
  - `open_orders`, including a 500-row page;
  - `cancel`, with a non-404/422 status;
  - `recheck`;
  - `settle`: a by-coid 404, a by-coid 5xx after retries, a payload without `filled_qty`, a
    non-numeric or negative `filled_qty`, an `id` mismatch, a cancel-requested id that no paper
    order row maps, and a ledger holding more than the venue reports.
- T8 A `sqlite3.OperationalError` during ingest propagates unchanged, unaudited.
- T9 Under-lock re-list failure: `BrokerError`, transaction rolled back, audit row count unchanged.
- T10 Re-check before drain raises `RuntimeError`. T11 skipped drain then a flattened ledger: the
  under-lock method refuses.
- T12 `cancel_and_ingest` leaves `conn.in_transaction` false; `owned_open_order_ids` leaves
  `conn.total_changes` unchanged.
- T13 Call order: the second sync's clock read follows the last open-order observation; the settle
  lookups follow the second sync's activity read; both ingests use broker-clock `until` values.
- T14 Non-revoking edges (`paper -> forward_tested` human raw, `forward_tested -> paper`) never call
  the selector (spy).
- T15 Live behavior: today's `test_registry_live_exit_guard.py` cases are ported to
  `select_exit_guard` and assert the same guard choice, the same audit actions and byte-identical
  messages. A CLI live exit still drains. Two timing pins (§2.4): `live -> dormant` without a reason,
  and `live -> retired` while `operator.lock` is held, each refuse with today's message, build no
  broker and write no `live_exit_drain_*` row.
- T16 Structural (`tests/test_lane_parity.py`):
  - `_REVOKE_ON_EXIT` equals the lane-exit set derived from `ALLOWED_TRANSITIONS`;
  - `select_exit_guard` returns `PaperExitGuard` for every paper-source edge and `LiveExitGuard` for
    every live-source edge;
  - in `algua/`, only `transitions.py` calls `.apply_transition(`, and no module passes
    `exit_guard_selector=`.
- T17 Operator lock: `paper -> dormant` and `forward_tested -> live` refuse while the lock is held,
  with the new message and no selector call; the existing edges' message is byte-identical.
- T18 Go-live:
  - a failing signature or certificate makes no venue call;
  - flat with a resting paper order, the order is cancelled and go-live completes;
  - with a non-cancelable order it is refused and the challenge stays unconsumed; it then completes
    with the same signature once the order is gone;
  - refused for a held `operator.lock`, the challenge stays unconsumed.
- T19 Skip-rule equivalence: over a matrix of ledgers (none, dust, material, mixed, no recorded
  price), the guard skips if and only if `_assert_flat_for_bench` refuses on positions.
- T20 Suite-wide Alpaca guard (§8), in `tests/test_no_real_alpaca_http.py`, with the fixture requested
  by name to read its record. Each of the following raises `AssertionError`, is recorded and sends
  nothing:
  - `requests.get("https://paper-api.alpaca.markets/v2/clock")`;
  - `requests.delete("https://api.alpaca.markets/v2/orders/x")`;
  - `requests.Session().request("GET", "https://data.alpaca.markets/v2/stocks/bars")`.

  The host predicate rejects `alpaca.markets` and every `*.alpaca.markets`, and it accepts
  `alpaca.markets.example.test` and `notalpaca.markets`. The test clears the record before teardown.

Mutation checks (break, see a named test fail, restore byte for byte):

| Mutation | Test that fails |
|---|---|
| Drop the skip rule | T5 |
| Drop `unsettled` from the under-lock answer | T4b |
| Return no guard, or fall open on missing credentials | T6 |
| Swallow a step failure | T7 |
| Accept the local clock | T7 |
| Drop the second sync | T3 |
| Drop the lock extension | T17 |
| Select the live guard for `forward_tested -> live` | T16 |
| Drop the state check | T10, T11 |
| Drop the settle step | T3b |
| Settle only this drain's `owned` instead of the durable audit set | T3c |
| Use the advancing `ingest_paper_venue` in the drain sync | T3e, T3d(i) |
| Write the cancel-requested row after the DELETE | T1 |
| Drop the bounded re-list | T4c |
| Remove the autouse Alpaca guard | T20 |

## 8. Known churn

- Tests passing `exit_guard=` to `transition_strategy` (`test_book_exit_revoke.py`, 3) switch to
  `exit_guard_selector=`. `test_registry_live_exit_guard.py` moves to `select_exit_guard` and
  monkeypatches `algua.execution.lane_exit` instead of `registry_cmd`.
- Tests that drive an allocation-shedding edge without a guard now meet the default drain. The probe
  found 32 paper-source cases (in `test_book_exit_revoke`, `test_cli_live`, `test_cli_paper`,
  `test_cli_registry`, `test_deployments`, `test_e2e_lifecycle`, `test_forward_certificate`,
  `test_paper_dust`, `test_paper_intake`, `test_registry_approvals`, `test_registry_store`,
  `test_shortlist_gate`, `test_transitions`) and 16 live-source cases (in `test_book_exit_revoke`,
  `test_cli_live`, `test_paper_dust`, `test_registry_store`, `test_transitions`). They opt in to an
  explicit, non-autouse fixture, `empty_exit_venues`, in a `tests/_exit_drain.py` helper; the name is
  the implementer's choice. The fixture does three things, so the real guards still run:
  - it patches both lanes' broker builders to empty fakes:
    `algua.execution.paper_exit_drain.build_paper_drain_broker`, and
    `algua.execution.lane_exit.build_live_broker` and `build_live_drain_broker`;
  - it patches `algua.registry.live_gate.verify_live_authorization` to raise
    `LiveAuthorizationError` unless the test supplies an authorization. Otherwise 9 of the 16
    live-source cases raise `StrategyNotFound` from the identity recompute
    (`registry/live_gate.py:178`) before any builder runs;
  - it redirects `algua.operator.deployment_lock._operator_lock_path` to `tmp_path`, as
    `test_deployments.py:563` does. More edges now take the checkout's real `operator.lock`, so
    without the redirect a merge-back drainer holding it would fail these tests spuriously.

  No autouse fixture disables the drain.
- **No real Alpaca HTTP in any test.** `tests/conftest.py` gains an autouse fixture,
  `_no_real_alpaca_http`, that wraps `requests.Session.request`.
  - **Why this seam.** Every Alpaca HTTP call in `algua/` comes from `_AlpacaBroker._request`
    (`alpaca_broker.py:115-148`, behind `_get`/`_post`/`_delete`) or from the data provider
    (`data/providers/alpaca.py:153`). Both call the module functions `requests.get/post/delete`.
    Those delegate to `requests.api.request`, which calls `Session.request` on a fresh session. So
    `Session.request` is the one function every real request passes through, whatever the verb or
    the lane.
  - **Existing fakes are unaffected.** No existing test fakes `Session.request`. The HTTP fakes
    replace the `requests` reference in `alpaca_broker` (`tests/test_alpaca_broker.py`) or
    `requests.get` itself (`tests/test_data_providers.py`), so a faked call never reaches the guard.
  - **The check.** The wrapper takes `urlsplit(url).hostname`, lower-cased. If it equals
    `alpaca.markets` or ends in `.alpaca.markets` (paper, live and data hosts alike), the wrapper
    records the method and URL and raises `AssertionError(f"real Alpaca HTTP in a test: {method}
    {url}")` without sending. Any other host goes to the original method unchanged.
  - **Teardown.** A raise alone can be swallowed: the drain turns a step exception into an audited
    `BrokerError`, and the CLI's catch-all turns any exception into an envelope, so a test that only
    asserts a refusal would still pass. At teardown the fixture therefore calls `pytest.fail`, listing
    every recorded request. There is no opt-out marker.
  - **Venues in tests.** The fixture does not disable the drain. A test that needs a venue uses the
    explicit `empty_exit_venues` fixture, a `FakePaperVenue` (§7), or a fake of the HTTP layer as
    `test_alpaca_broker.py` does. A test that forgets fails instead of reaching the network.
  - **Baseline.** The full suite at `15e9f7e` was run with this exact wrapper in recording mode, in
    a scratch copy on 2026-10-05: 6105 passed, 5 skipped, zero Alpaca requests recorded. No existing
    test reaches an Alpaca host, even with the raise swallowed, so adopting the guard and its
    teardown failure changes no existing test.
- `tests/test_paper_venue_reconcile.py` imports `ingest_paper_venue` from `algua.execution.venue_sync`.
- The operator-lock tests (`test_cli_registry.py:47-64`, `test_deployments.py:550-571`) are unchanged:
  the lock refuses before the selector runs.
- Docs:
  - `docs/architecture.md`: the lane-parity wall gains one sentence on exit drains.
  - `CLAUDE.md`: the `registry transition` entry notes that a paper-lane exit drains the strategy's
    resting paper orders and needs the paper credentials and the paper venue. It must run from the
    operator's checkout to exclude the paper timer. A refusal right after cancellation, or for a fill
    the venue has not published yet, clears on retry.
  - `deploy/systemd/README.md:126`: the "retire every frozen deployment" step now drains and needs the
    paper venue and credentials.
  - The `algua/cli/paper_venue.py` module docstring: it no longer holds the venue ingest or stranded
    recovery.
  - `docs/contracts/cli-error-envelope.md` is unchanged (§4).

## 9. Residuals and follow-ups

- The live guard keeps two gaps this guard closes (AC6 keeps it unchanged; file one issue): it cancels
  without first checking positions, so an early exit destroys resting liquidation offsets; and its
  final ingest precedes its last open-order observation, so an order the cancel could not stop and
  that fills between that ingest and the under-lock re-list escapes both checks.
- **Feed lag, narrowed.** A fill the venue has executed but not yet published when the drain's final
  sync reads the feed escapes the drain only if its order closed before the drain began and no drain
  ever asked to cancel it. The settle set (§3.3 step 6) covers every order a drain has asked to
  cancel. An order that closed before the drain began was not resting at the exit, so it is outside
  #685's orphan class. The paper account reconcile still sees the drift.
- **Run-all's time cursor (out of scope; the coordinator files an issue).** `paper run-all` ingests
  with `ingest_paper_venue`, whose cursor is each cycle's broker-clock `until`. Alpaca can publish an
  activity late with a `transaction_time` at or before that `until`. Such an activity falls outside
  every later window and is never ingested. This story stops the drain adding cursor advances; it
  does not change the operator's cursor.
- **A fill that never reaches the ledger.** An example is a quarantined activity
  (`live_ledger.py:588-602`). If it belongs to an order a drain asked to cancel, every later exit of
  that strategy is refused with the `unpublished` message until the operator triages it. This fails
  closed and matches the account reconcile, which reports the same drift.
- **Settle cost.** Each later exit of a strategy re-reads every order a drain ever asked to cancel
  for it, at one by-coid GET each.
- A paper command run by hand outside the operator wrapper can submit for an exiting strategy (§5).
- An under-lock re-list failure is reported only by the envelope (§4; AC3 as amended).
- A command run from a worktree does not share the operator's `operator.lock` (§2.5).

## 10. Readiness: invocations to execute against a fake

The readiness review executes the guard against a fake Alpaca paper endpoint and checks each call's
shape. The 2026-10-05 review did this (probes P1–P9):

- `GET /v2/clock`;
- `GET /v2/account/activities?after=…&until=…&direction=asc&page_size=100`, paginated by
  `page_token`; after this story, `after` is the stored cursor on every drain sync;
- `GET /v2/orders?status=open&limit=500`;
- `DELETE /v2/orders/{id}`, with 200/204/404/422 as no-ops;
- `GET /v2/orders:by_client_order_id?client_order_id=…` for each stranded row and, after this
  story, for each order in the settle set.

It also runs `uv run lint-imports` with the planned import edges in place. The delta check after
these corrections covers §3, §6 and §8.
