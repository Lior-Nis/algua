# Story 2.2 paper exit drain contract

Normative companion to [SPEC.md](SPEC.md). Line references are to `e875b6d` (main after PR #686).
"Paper lane" means the stages `paper` and `forward_tested`, which `paper run-all` ticks and the paper
account reconcile counts (`algua/execution/paper_reconcile.py:36-58`). The "live lane" is `live`.

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
`alpaca_broker.py:419-428`; ingest de-duplicates by activity id, `live_ledger.py:484-538`). Re-running
a completed exit fails `validate_transition` before any drain.

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
unconsumed and the same signature can complete later while it is unexpired. With production wiring a
missing paper credential is refused first by the certificate verifier's own message
(`transitions.py:256-267`); the drain's unavailable branch is reached on this edge only with an
injected verifier. The live ledger is not drained on go-live: a `forward_tested` strategy has no live
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
  onward (`registry_cmd.py:64-93`), moved verbatim: the authorized broker, else the account-credential
  drain broker with the `live_exit_drain_account_creds` audit, else the `live_exit_drain_unavailable`
  audit and its `TransitionError`. Messages and audit actions are byte-identical.
- `source in {Stage.PAPER, Stage.FORWARD_TESTED}`: `broker = build_paper_drain_broker()`. On `None`,
  audit `paper_exit_drain_unavailable` and raise the unavailable `TransitionError` (§4). Otherwise
  return `PaperExitGuard(conn, broker, name)`.
- Any other source raises `ValueError(f"no exit drain for {source.value} -> {target.value}")`
  (unreachable from `transition_strategy`).
- `build_paper_drain_broker() -> AlpacaPaperBroker | None` is `maybe_broker(BrokerKind.ALPACA_PAPER)`
  (`broker_factory.py:152-181`), the test seam. A malformed setting (pydantic `ValidationError`, a
  host-pinning `BrokerError`) propagates unchanged and unaudited, as on every paper command.

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
a held lock as a benign no-op (`cli/operator_cmd.py:279-313`); merge-back holds the same lock for its
whole saga (`operator_cmd.py:248`), so exits wait for it as retirement does today.

## 3. The paper exit guard

### 3.1 Broker capability

`algua/contracts/types.py` gains, beside `ExitDrainBroker`:

```python
@runtime_checkable
class PaperExitDrainBroker(ScopedCancelBroker, ActivityWindowBroker, OrderLookupBroker, Protocol):
    def clock(self) -> str: ...
```

`AlpacaPaperBroker` satisfies it (`list_open_orders` `:394-400`, `cancel_order` `:419-428`,
`account_activities_window` `:450-478`, `get_order_by_client_order_id` `:402-417`, `clock`
`:439-446`). `cancel_open_orders` (`:233-246`) is outside the protocol.

### 3.2 Venue sync

`ingest_paper_venue` and `recover_stranded` (with `_PAPER_CURSOR_FAR_PAST`) move unchanged from
`algua/cli/paper_venue.py:43-60,69-78` to a new `algua/execution/venue_sync.py`; `paper_cmd.py`
and `cli/paper_venue.py` (`live_strategy_flat`) import them from there. No re-export. `paper_scoped_cancel`
stays in `cli/paper_venue.py`. The guard's *sync* is the paper operator's ingest pairing
(`paper_cmd.py:980-981`):

1. `until, source = tick_clock(broker.clock)`; `source != "broker"` is a drain failure at step
   `clock` (the local-clock fallback is refused: a skewed `until` stored as the shared cursor can
   skip fills for good);
2. `ingest_paper_venue(conn, broker, until)` then `recover_stranded(conn, broker, LedgerKind.PAPER)`
   (step `ingest`).

### 3.3 `PaperExitGuard.cancel_and_ingest()` (before the lock)

`state` starts `new`; `unsettled` starts empty.

1. Sync.
2. If the strategy holds a material paper position (`believed_positions(conn, name, PAPER)` minus
   every `paper_dust` residual: the expression at `crud.py:341-343`), set `state = skipped` and
   return. Nothing is cancelled: the exit will be refused on positions, and resting liquidation
   offsets must survive a premature exit.
3. `owned = owned_open_order_ids(conn, broker, name, kind=PAPER)` (step `open_orders`).
4. If `owned` is non-empty: `cancel_order(oid)` for each (step `cancel`); then, whether the loop
   finished or not, one `paper_exit_drain_cancelled` audit row lists the ids actually cancelled (if
   any); then `unsettled = owned_open_order_ids(...)` (step `recheck`). A `422` (not cancelable) is
   a no-op at the broker, so such an order stays in `unsettled`.
5. Sync again. Its `until` is read after the last observation of the strategy's open orders (step 3
   when nothing was owned, step 4's recheck otherwise).
6. `state = drained`.

Every step failure is handled by §4. On return the connection has no open transaction.

### 3.4 `PaperExitGuard.owned_open_order_ids()` (under the lock)

- `state == new`: `RuntimeError("paper exit re-check ran before its drain")`.
- `state == skipped`: `TransitionError(f"{name} is not flat (open paper positions when the exit
  drain ran); flatten before this transition")`. Normally unreachable: the store's positions check
  runs first and refuses with the symbols. It fires only if a concurrent ingest flattened the ledger
  in between.
- `unsettled` non-empty: return `sorted(unsettled)` without a broker call; the store refuses with
  its open-order message.
- Otherwise return `sorted(owned_open_order_ids(...))`. A `sqlite3.Error` propagates; any other
  exception becomes the under-lock `BrokerError` (§4), unaudited.
- It never writes to the database and never commits.

### 3.5 Invariants

- An exit commits only if, after a sync whose `until` was read after the final pre-lock observation
  showed none of the strategy's orders open, the ledger holds no material paper position, and a
  re-list under the write lock shows none open either.
- Only orders whose `client_order_id` maps to the strategy in `paper_venue_orders`
  (`live_ledger.py:605-620`) are cancelled. `cancel_open_orders` is never called on this path.
- A full page of 500 open orders fails closed (`alpaca_broker.py:398-399`).
- Any owned open order blocks, whatever its size. A residual is flat only by the Story 1.4 rule.
- The guard trips no kill switch: a refused exit leaves the strategy on the lane, and the next
  paper tick re-plans it.

## 4. Outcomes, audit and envelope

No new error codes; `docs/contracts/cli-error-envelope.md` is unchanged. Success JSON is unchanged:
`{"ok": true, "name", "stage"}`.

| Condition | Where | Audit action (actor `system`, strategy = name) | Raised | `code` |
|---|---|---|---|---|
| `operator.lock` held | `operator_transition_lock` | none | `TransitionError` (§2.5 messages) | `wrong_stage` |
| Paper credentials not configured | selector | `paper_exit_drain_unavailable`, reason `"Alpaca paper credentials not configured; cannot drain resting paper orders"` | `TransitionError(f"cannot exit paper-lane strategy {name!r}: Alpaca paper credentials are not configured, so its resting paper orders cannot be drained; set ALGUA_ALPACA_API_KEY and ALGUA_ALPACA_API_SECRET, then retry")` | `wrong_stage` |
| Step `clock`, `ingest`, `open_orders`, `cancel` or `recheck` raises any `Exception` other than `sqlite3.Error` (including a full open-order page and a malformed venue payload) | `cancel_and_ingest` | `paper_exit_drain_failed`, reason `f"{step}: {type(exc).__name__}: {exc}"` | `BrokerError(f"cannot exit paper-lane strategy {name!r}: its exit drain failed at {step} ({type(exc).__name__}: {exc}); its stage and allocation are unchanged; retry when the paper venue is reachable") from exc` | `broker_error` |
| `sqlite3.Error` during the drain | `cancel_and_ingest` | none (the audit write would meet the same fault) | unchanged | `db_unavailable` (retryable) for `OperationalError` |
| One or more orders cancelled | `cancel_and_ingest` step 4 | `paper_exit_drain_cancelled`, reason `f"cancelled {len(ids)} resting paper order(s): {ids}"` | none | n/a |
| Material paper position | store, under the lock (`crud.py:344-347`) | none | existing `f"{name} is not flat (open paper positions {sorted(held)}); flatten before this transition"` | `wrong_stage` |
| Own order open after the cancel, or in the under-lock re-list | store (`crud.py:348-353`) | none | existing `f"{name} is not flat ({len(open_ids)} open paper order(s) {open_ids}); flatten before this transition"` | `wrong_stage` |
| Drain skipped, ledger flat under the lock | guard (§3.4) | none | `TransitionError` (§3.4) | `wrong_stage` |
| Under-lock re-list fails | guard (§3.4) | none: the transaction rolls back, and `audit.log.append` would commit it half-way | `BrokerError(f"cannot exit paper-lane strategy {name!r}: re-listing its open paper orders under the registry lock failed ({type(exc).__name__}: {exc}); the transition was rolled back; retry") from exc` | `broker_error` |
| Re-check before drain (caller bug) | guard | none | `RuntimeError` | `internal` |

Every refusal leaves stage, allocation, deployment, gate tokens and go-live challenge unchanged.
Orders already cancelled before a refusal stay cancelled; the cancelled audit row records them.

## 5. Interactions

- **Paper operator.** `operator.lock` (§2.5) serializes every paper-lane exit with timer-driven
  `paper run-all` and merge-back. A paper command run by hand outside the wrapper can still submit for
  the strategy after the exit commits; that stays operator discipline, as `run-all`'s own docstring
  already states (`paper_cmd.py:917-918`). The drain's ingest writes the shared paper cursor; a
  concurrent writer can move it back, which only re-fetches an overlap that ingest de-duplicates.
- **Dust.** A dust residual is flat for both the skip decision and the store check (one rule,
  `execution/dust.py:18-29`). A resting order is never dust.
- **Story 1.4 legacy exit.** The flatten-wait-retire procedure keeps working. Production (read-only
  copy of the registry, 2026-10-05): 17 tenants retired; `liquidity_stable_quality_momentum` is still
  at `paper`, kill-switched by `flatten` at 2026-10-03T13:51Z, believing 3.552855 UNH (material),
  with a resting UNH sell offset (the order row does not record its quantity; Story 1.4 caps it at
  the ledger net, 3.55273207). Retiring it before that offset fills is refused on positions and
  leaves the offset resting (§3.3 step 2). If the offset fills in full, the remaining 0.000123 share
  residual is dust, and the retirement drains nothing and commits. Retiring through this guard
  needs the paper credentials the box already has.
- **Frozen tenants.** A frozen deployment's orders go through the same `paper_venue_orders` ledger,
  so its exits drain identically. Go-live stays refused for frozen deployments
  (`frozen_live_unsupported`) before any drain.
- **Rollback runbook.** `deploy/systemd/README.md`'s "retire every frozen deployment" step now
  drains; it needs the paper venue reachable.

## 6. Placement, size and protection

- `algua/execution/lane_exit.py` holds `LiveExitGuard` (unchanged), `PaperExitGuard`,
  `build_paper_drain_broker` and `select_exit_guard`. It may import `algua.contracts`,
  `algua.execution.{alpaca_broker, broker_factory, live_ledger, tick_clock, dust, venue_sync, errors}`,
  `algua.audit.log` and `algua.registry.live_gate`; it must not import `algua.execution.flatten`,
  `order_state`, `live_reconcile` or anything under `algua.live` or `algua.cli`. The existing
  contract "registry stays off the live lane" enforces this transitively, because import-linter
  follows the lazy import in `transitions.py` (verified with grimp on 2026-10-05). No contract
  changes and no exemption.
- `algua/execution/venue_sync.py` holds the two moved functions. It must not import
  `algua.live` (it does not take `live_strategy_flat`, which needs `live_loop._RECONCILE_TOL`).
- `transitions.py` (277 lines) must stay at or below 290 lines; it is unpinned and the ratchet
  forbids a new module at 300 or more. Story 2.1 also edits `_validate_live_gate`; the later of the
  two to merge rebases and keeps the file under 300 or carves.
- `registry_cmd.py` loses the selection (about 60 lines). Lower its pin in
  `tests/test_module_size_ratchet.py:59` to the new size in the same commit.
- `lane_exit.py`, `venue_sync.py` and `deployment_lock.py` stay under 300 lines. `store/crud.py`,
  `dust.py` and `alpaca_broker.py` are not edited, except the docstring of `_assert_flat_for_bench`
  (`crud.py:329-334`), which says the drain is wired for the live lane only.
- CODEOWNERS and `INTEGRITY_CRITICAL_MODULES` (`tests/test_repo_hygiene.py:237`) gain
  `algua/execution/lane_exit.py`, `algua/execution/venue_sync.py` (its code was protected in
  `cli/paper_venue.py`) and `algua/operator/deployment_lock.py`. The store trusts all three.

## 7. Tests

New tests, with fake venues (no network). `FakePaperVenue` implements §3.1 and records calls;
its `cancel_open_orders` raises `AssertionError`.

- T1 Each of the five paper-source edges through the default selector: without a resting order the
  exit commits and revokes the allocation; with an own resting order and no material position, the
  order is cancelled, `paper_exit_drain_cancelled` lists it, and the exit commits.
- T2 A sibling's resting order on the same account is neither cancelled nor blocking.
- T3 Fill between cancel and ingest: the fake fills the order on cancel and publishes the activity;
  the second sync attributes it and the exit is refused on positions with everything unchanged.
- T4 Non-cancelable order (cancel is a no-op): refused under the lock with the open-order message.
- T4b An order open at the recheck but gone from the under-lock re-list is still refused.
- T5 Material position plus a resting offset: refused on positions; the offset is not cancelled and
  no cancelled row is written. T5b dust-only residual plus a resting order: cancelled, exit commits.
- T6 Credentials absent, each paper-source edge and through the CLI envelope (`wrong_stage`, exit 1):
  `paper_exit_drain_unavailable` row, no broker call, nothing changed.
- T7 Failure at each step (`clock` via a raising or naive clock, `ingest`, `open_orders` including a
  500-row page, `cancel` with a non-404/422 status, `recheck`): `paper_exit_drain_failed` row naming
  the step, `BrokerError`, CLI `broker_error`, nothing changed.
- T8 A `sqlite3.OperationalError` during ingest propagates unchanged, unaudited.
- T9 Under-lock re-list failure: `BrokerError`, transaction rolled back, audit row count unchanged.
- T10 Re-check before drain raises `RuntimeError`. T11 skipped drain then a flattened ledger: the
  under-lock method refuses.
- T12 `cancel_and_ingest` leaves `conn.in_transaction` false; `owned_open_order_ids` leaves
  `conn.total_changes` unchanged.
- T13 Call order: the second sync's clock read follows the last open-order observation; both
  ingests use broker-clock `until` values.
- T14 Non-revoking edges (`paper -> forward_tested` human raw, `forward_tested -> paper`) never call
  the selector (spy).
- T15 Live unchanged: today's `test_registry_live_exit_guard.py` cases ported to `select_exit_guard`,
  asserting the same guard choice, the same audit actions and byte-identical messages; a CLI live
  exit still drains.
- T16 Structural (`tests/test_lane_parity.py`): `_REVOKE_ON_EXIT` equals the lane-exit set derived
  from `ALLOWED_TRANSITIONS`; `select_exit_guard` returns `PaperExitGuard` for every paper-source
  edge and `LiveExitGuard` for every live-source edge; in `algua/` only `transitions.py` calls
  `.apply_transition(`, and no module passes `exit_guard_selector=`.
- T17 Operator lock: `paper -> dormant` and `forward_tested -> live` refuse while the lock is held,
  with the new message and no selector call; the existing edges' message is byte-identical.
- T18 Go-live: a failing signature or certificate makes no venue call; flat with a resting paper
  order, the order is cancelled and go-live completes; with a non-cancelable order it is refused and
  the challenge stays unconsumed, then completes with the same signature once the order is gone.
- T19 Skip-rule equivalence: over a matrix of ledgers (none, dust, material, mixed, no recorded
  price), the guard skips if and only if `_assert_flat_for_bench` refuses on positions.

Mutation checks (break, see a named test fail, restore byte for byte): drop the skip rule (T5);
drop `unsettled` from the under-lock answer (T4b); return no guard or fall open on missing
credentials (T6); swallow a step failure (T7); accept the local clock (T7); drop the second sync
(T3); drop the lock extension (T17); select the live guard for `forward_tested -> live` (T16); drop
the state check (T10, T11).

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
  explicit, non-autouse fixture (for example `empty_exit_venues` in a `tests/_exit_drain.py`
  helper) that patches both lanes' broker builders to empty fakes, so the real guards still run. No
  autouse fixture disables the drain.
- `tests/test_paper_venue_reconcile.py` imports `ingest_paper_venue` from `algua.execution.venue_sync`.
- The operator-lock tests (`test_cli_registry.py:47-64`, `test_deployments.py:550-571`) are unchanged:
  the lock refuses before the selector runs.
- Docs: `docs/architecture.md`'s lane-parity wall gains one sentence on exit drains; `CLAUDE.md`'s
  `registry transition` entry notes that a paper-lane exit drains the strategy's resting paper orders
  and needs the paper credentials.

## 9. Residuals and follow-ups

- The live guard keeps two gaps this guard closes (AC6 keeps it unchanged; file one issue): it cancels
  without first checking positions, so an early exit destroys resting liquidation offsets; and its
  final ingest precedes its last open-order observation, so an order the cancel could not stop and
  that fills between that ingest and the under-lock re-list escapes both checks.
- A fill the venue has executed but not yet published in the activity feed when the final sync reads
  it is skipped by the cursor, as for every paper ingest.
- A paper command run by hand outside the operator wrapper can submit for an exiting strategy (§5).
- An under-lock re-list failure is reported only by the envelope (§4).

## 10. Readiness: invocations to execute against a fake

The readiness review executes the guard against a fake Alpaca paper endpoint and checks each call's
shape: `GET /v2/clock`; `GET /v2/account/activities?after=…&until=…&direction=asc&page_size=100`
(paginated by `page_token`); `GET /v2/orders?status=open&limit=500`; `DELETE /v2/orders/{id}` with
200/204/404/422 as no-ops; `GET /v2/orders:by_client_order_id?client_order_id=…` for each stranded
row. It also runs `uv run lint-imports` with the planned import edges in place.
