---
stepsCompleted:
  - step-01-document-discovery
  - step-02-prd-analysis
  - step-03-epic-coverage-validation
  - step-04-ux-alignment
  - step-05-epic-quality-review
  - step-06-final-assessment
filesIncluded:
  - docs/development/stories/2-2-drain-resting-paper-orders-on-lane-exit.md
  - docs/development/specs/spec-story-2-2-paper-exit-drain/ (SPEC, paper-exit-drain-contract, decision log)
  - docs/development/specs/spec-story-2-1-unrelaxed-live-qualification/ (overlap check only)
  - docs/agent/story-delivery.md §2, issues #685 and #690, AGENTS.md, CLAUDE.md
assessmentScope: Story 2.2
baseline: e875b6d (main after PR #686); contracts at 047d6d0
---

# Implementation Readiness Assessment Report

**Date:** 2026-10-05 · **Project:** Algua, Story 2.2 · **Assessor:** independent reviewer (Claude),
BMAD method, run non-interactively with default answers.

**Method.** Every contract claim was checked against the code at `047d6d0`, which matches `e875b6d`
for every file cited. The contract's planned edges were built as a scratch prototype in a separate
copy of the repo (`git archive`, with its own `.git` and therefore its own `operator.lock`). The
prototype covers `venue_sync.py`, `PaperExitGuard`, `select_exit_guard`, `operator_transition_lock`,
the selector in `transition_strategy` and the trimmed `registry_cmd.py`. Against it, the review ran
`lint-imports`, the size ratchet, the existing tests that drive transitions, and the probes below.
The probes used a fake Alpaca HTTP layer: `requests.get/delete/post` were replaced with recorded v2
response shapes, and only `paper-api.alpaca.markets` was accepted. The production registry was read
only through `file:…/data/algua.db?mode=ro`, and the rehearsal ran on a `backup()` copy in scratch.
Nothing in the worktree changed except this report.

> **Disclosure.** Running the existing suite against the prototype made about a dozen real HTTPS
> requests to Alpaca: to `paper-api.alpaca.markets` from `tests/test_paper_intake.py`, and to
> `api.alpaca.markets` from `tests/test_cli_live.py`. Those files set dummy credentials (`k`/`s`,
> `lk`/`ls`; test_paper_intake.py:68-69, test_cli_live.py:72-73). The new default selector built
> real brokers from them, and every request was rejected with 401. No real credentials were present:
> the environment exports none, and `tests/conftest.py` disables `.env`. After this, a guard
> blocking Alpaca hosts was added to the scratch conftest. This is the hazard behind **M3**.

**Probes** (`scratchpad/readiness-2-2/repo/probe.py`; output in `probe-out.txt`):

| Probe | What it exercised | Result |
|---|---|---|
| P1 shapes | A `paper -> retired` exit with an own resting order, a sibling's order and a stranded row, through the default selector and a real `AlpacaPaperBroker` over the fake | The calls match §10 exactly: `GET /v2/clock`, the activities window, a by-coid lookup, the open-order list, `DELETE` of the own order, a recheck, then the second clock/window/lookup and the under-lock list. The sibling's order stays `accepted`; the exit commits and revokes the allocation |
| P2 commit trace | sqlite trace callback on a `paper -> dormant` exit | Cursor, audit and cursor commits all happen before `BEGIN IMMEDIATE`. After it come only the guard's re-list, the stage UPDATE and one `COMMIT` |
| P3 race | The order fills at cancel (422) and the fill is published at once | Refused on positions; stage unchanged |
| P3b lagged race | Same, but the fill activity is published only after the second sync | **Exit commits.** The venue reports the order `filled` with `filled_qty=2`, while the ledger belief is `{}` (**M2**) |
| P4 go-live | Order not cancelable (422), injected verifiers, pending challenge | Refused with the open-order message. `consumed_at` stays NULL, the stage stays `forward_tested` and the allocation is intact. A retry with the same authorization after the order is gone goes live and consumes the challenge |
| P5 dust / material | Dust residual plus a resting order; material UNH plus a resting offset | Dust: the order is cancelled and the exit commits. Material: refused on positions, the offset stays `accepted`, and no cancelled-orders row is written |
| P6 failures | Clock 503, a tz-naive clock, 501 open orders, cancel returning 500 | Each one: `BrokerError`, stage unchanged, and a `paper_exit_drain_failed` row naming the step (`clock`, `clock`, `open_orders`, `cancel`) |
| P7 credentials | Paper credentials unset | `TransitionError` with the §4 message, a `paper_exit_drain_unavailable` row, and no venue call |
| P8 nested lock | A paper exit while the same process holds `operator.lock` | Refused at once with the new §2.5 message, after zero venue calls. The flock is non-blocking, so there is no deadlock |
| P9 production copy | Retiring `liquidity_stable_quality_momentum`, (a) offset still resting, (b) offset filled | (a) Refused on positions; the offset is untouched; 9 stranded by-coid lookups. (b) Retires; 18 by-coid lookups; no new audit rows |

## Requirements Traceability

| AC | Contract | Code seams (verified) | Status |
|---|---|---|---|
| 1 drain before the lock | §2.3, §3.2-3.3 | `apply_transition` runs `exit_guard.cancel_and_ingest()` before `BEGIN IMMEDIATE` (store/crud.py:292-302). `owned_open_order_ids` scopes by coid (execution/live_ledger.py:605-620). The paper ingest pairing is at cli/paper_cmd.py:980-981, with bodies at cli/paper_venue.py:46-60 and :69-78 | covered (P1); **M2** |
| 2 under-lock re-check | §3.4 | `_assert_flat_for_bench` checks positions, then `exit_guard.owned_open_order_ids()` (crud.py:340-353). Challenge consumption and the revoke follow in the same transaction (store/base.py:59-70, :116-120) | covered (P2, P4) |
| 3 fail closed, audited | §2.4, §4 | `maybe_broker(ALPACA_PAPER)` returns None when credentials are missing (execution/broker_factory.py:139, :152-163). `tick_clock` falls back to local time (execution/tick_clock.py:18-21). `audit.append` commits (audit/log.py:7-15) | covered (P6, P7); **m2** |
| 4 sibling safety | §3.5 | `cancel_order` by id (execution/alpaca_broker.py:419-428). `cancel_open_orders` (:233-246) is never reached; the fake asserts on `DELETE /v2/orders` | covered (P1) |
| 5 race coverage | §3.3 steps 4-5, T3/T4/T4b | Holds only when the venue publishes the fill instantly | **M2** |
| 6 live unchanged | §2.4 live branch, T15 | The `_live_exit_guard` body (cli/registry_cmd.py:64-93) moves verbatim. Its messages and audit actions are unchanged; its timing moves (inside the lock, after validation) | covered; **m6**, **m7** |
| 7 structure | §2.1, §2.3, §6, T16 | `_REVOKE_ON_EXIT` (registry/transitions.py:30-33) equals the derived lane-exit set from `ALLOWED_TRANSITIONS` (contracts/lifecycle.py:26-41), checked by hand. `.apply_transition(` is called only at transitions.py:97 (grep). `registry_cmd.py` goes 444 → 378 in the prototype; `transitions.py` goes 277 → 283 | covered; **M1** |
| 8 green, mutation-checked | §7 mutation list | The ratchet fails on the §3.1 placement; `lint-imports` gives 29 kept | **M1**, **M3** |

No clause adds a lifecycle edge, gate, schema change, CLI flag, error code or authority. Every
refusal leaves the stage, allocation, deployment, tokens and challenge unchanged (P4, P6, P7). The
live wall's checks still run first.

## Code Feasibility Verification

**(1) Entry-point inventory.** The inventory is complete. Searches over `algua/` and `web/backend`
found these stage-changing paths:

- **`transition_strategy` callers:** cli/registry_cmd.py:249 and evaluation/backtest_run.py:105. The
  second targets `backtested` only, and the lifecycle has no paper-lane edge to `backtested`
  (contracts/lifecycle.py:30-31).
- **`.apply_transition(` callers:** transitions.py:97 only.
- **`_apply_transition_locked` callers:** store/crud.py:304 and :314, store/deployment.py:208
  (`candidate -> paper`), store/gate.py:454 (`backtested -> candidate`) and store/forward_gate.py:163
  (`paper -> forward_tested`). None of the store-internal callers revokes.
- **Raw `UPDATE strategies SET stage`:** store/base.py:125 (the compare-and-swap),
  registry/mergeback_intake.py:238 (`idea -> backtested`) and registry/db/core.py:75 (the
  `shortlisted` migration).
- **Paths that never change stage:**
  - paper run-all, kill, flatten and halt-all, plus every breach path;
  - the merge-back driver, which reaches `transition_strategy` only through `run_backtest_task`
    toward `backtested`;
  - paper intake;
  - the operator wrapper, which runs drivers as subprocesses (cli/operator_cmd.py:70-75, :167);
  - `live_cmd`;
  - `allocations.deallocate`, which has no production caller.
- **Runbook invocations:** deploy/systemd/README.md:126 only, the rollback "retire every frozen
  deployment" step (`registry transition`, so it reaches the selector too).

So every paper-lane exit reaches the guard through the one default selector.

**(2) Guard algorithm.**

| Concern | Verdict | Evidence |
|---|---|---|
| `operator.lock` on every paper exit | Holds | The new rule (§2.5) locks all five paper-source edges. Today's rule covers only `paper -> candidate` and `-> retired` (operator/deployment_lock.py:29). The lock is taken before the selector, so a held lock costs zero venue calls (P8) |
| Self-deadlock in one process | None | `operator_run_lock` is a non-blocking flock (operator/schedule.py:104-123), so a nested acquire is refused, not blocked (P8). No in-process holder calls `transition_strategy`: the paper job and merge-back run their children as subprocesses, and the only lock-held child that transitions (merge-back → backtest_run) targets `backtested`, which needs no lock. There is no separate live lock (cli/live_cmd.py has none). `live -> retired` takes the same lock as today, once |
| Concurrent paper run-all | Excluded only from the operator's checkout | The installed unit runs from `/home/liornisimov/Projects/algua` (`systemctl --user cat algua-paper.service`). The lock path is per git dir (deployment_lock.py:15-23). A run from a worktree gets no exclusion (**m3**). A hand-run `paper run-all` is a declared non-goal |
| Broker clock outage | Fails closed before the lock | BrokerError, ValueError and TypeError become `"local"` (tick_clock.py:18-21), and the contract refuses that. Probed with a 503 and a naive clock (P6) |
| 500-order cap | Fails closed | `len(rows) >= 500` raises (alpaca_broker.py:398-399). Before the lock this is audited as step `open_orders` (P6); under the lock it becomes the unaudited `BrokerError` |
| Stranded-order recovery | Sound | Ownership is keyed by coid, so stranded (NULL broker id) orders are still listed and cancelled. Each sync runs ingest then `recover_stranded`, which back-attributes fills (live_ledger.py:302-342, :374-413). Production has 9 stranded rows, so each sync makes 9 by-coid GETs (P9), and a transient failure on any of them refuses the exit (retry) |
| Dust | Sound | One rule: `paper_dust` (execution/dust.py:18-28) is used for both the skip and the store check (crud.py:341-343). A resting order blocks at any size (P5) |
| Partial fills | Sound | `partially_filled` counts as open. Cancelling it removes the remainder; the second sync ingests the partial fill and the positions check refuses |
| Cancel accepted but still `pending_cancel` | Fails closed, spuriously | Alpaca's `status=open` includes non-terminal states. A recheck right after `DELETE` can still list the order, so the exit is refused with "flatten before this transition" and succeeds on retry (**m1**) |
| Fill published late | **Gap** | P3b; see **M2** |
| Go-live challenge | Holds | The live gate runs before the lock (transitions.py:65-74). The challenge is consumed inside the exit transaction, after the re-check (crud.py:302-307 → base.py:59-70). A refused drain leaves it unconsumed and the same signature can complete later (P4). The challenge TTL is 10 minutes (registry/challenges.py:32) (**m3**) |
| Credentials absent | Fails closed, audited | P7. On go-live with production wiring, the certificate verifier refuses first (transitions.py:256-267) |

**(3) Commit points on the guard path.** No audit row and no commit happens under the registry write
lock (P2). The full list:

- **Before the lock, under `operator.lock`:**
  - the selector's `paper_exit_drain_unavailable` audit, or for the live branch
    `live_exit_drain_unavailable` / `live_exit_drain_account_creds` (audit/log.py:15);
  - in each sync, the `ingest_activities` commit, or a rollback on error (live_ledger.py:535, :537);
  - per recovered row, a `_backfill_order` commit (live_ledger.py:333, :341);
  - the `stranded_order_recovered` and `stranded_recovery_mismatch` audits;
  - the `paper_exit_drain_cancelled` audit;
  - the `paper_exit_drain_failed` audit.
- **Under the lock:** only the store's own `COMMIT` (crud.py:308), or the rollback (:310). The guard's
  under-lock method only SELECTs and makes HTTP calls.

**(4) Alpaca calls.** All five shapes in §10 were executed against recorded responses (P1):

- The field names the code reads match the v2 shapes: `timestamp` (clock), `id`, `activity_type`,
  `side`, `qty`, `price`, `symbol`, `order_id` and `transaction_time` (activities), and `id`,
  `client_order_id` and `symbol` (orders and by-coid).
- The nanosecond, offset-bearing `until` is percent-encoded. That form is already accepted in
  production: the stored cursor is `2026-10-03T13:51:36.072468822+00:00`.
- `DELETE` treats 200, 204, 404 and 422 as no-ops, and raises on 500 (P6).
- No POST occurs, and the account-wide cancel is never reached.

**(5) Placement.**

- **Import contracts:** `lint-imports` gives 29 kept with the planned edges. The new edges are
  execution → `registry.live_gate`, execution → `audit`, and registry → `execution.lane_exit`
  (lazy). The contract's claim that import-linter follows the lazy import was mutation-checked:
  adding `algua.live` to `venue_sync` breaks "registry stays off the live lane" through
  `transitions → lane_exit (l.281) → venue_sync`.
- **Size:** **`algua/contracts/types.py` is at its pin, 468/468.** Adding `PaperExitDrainBroker`
  makes it 475 and `test_no_pinned_module_grew_past_its_budget` fails (**M1**). The prototype's
  `lane_exit.py` reaches 211 lines without docstrings (**m9**).
- **Protection:** the CODEOWNERS and `INTEGRITY_CRITICAL_MODULES` additions are consistent with
  tests/test_repo_hygiene.py:318-341. `cli/paper_venue.py` stays protected.

**(6) Test churn.** The prototype was run against the 25 transition-heavy test files with no fixture
in place: **48 failed, 569 passed**. Every failure is in a file §8 names, plus the expected
`exit_guard=` TypeErrors (3) and the `test_registry_live_exit_guard.py` port (4). The failures fall
into three groups:

- missing paper credentials, as planned;
- **`StrategyNotFound`** (9): live-source exits with synthetic names, raised by
  `verify_live_authorization`'s identity recompute (registry/live_gate.py:178) before any broker
  builder is reached;
- real-network refusals (**M3**).

The planned fixture, "patches both lanes' broker builders", therefore does not fix the nine live
cases (**m7**). The operator-lock tests (test_cli_registry.py:47-64, test_deployments.py:550-573)
are unaffected.

The full suite at baseline (no prototype), with the M3 Alpaca-host guard autoused in the scratch
conftest: **6105 passed, 5 skipped** (10 min 33 s). No existing test needs a real Alpaca host, so the
M3 fix is safe to adopt.

**(7) Today's production state** (read-only, user_version 48).

- **Registry:** one strategy at `paper`, id 20 `liquidity_stable_quality_momentum`. It is
  kill-switched (`flatten`, 2026-10-03T13:51:35Z) and has no active deployment.
- **Positions:** it believes UNH 3.552855252, plus AMZN, BAC and CSCO at about 1e-16 (dust).
- **Resting offset:** coid `…-20261003T135136Z-UNH`, broker id `b52ab62c…`.
- **Cursor:** stuck at 2026-10-03T13:51:36Z, because no run-all has ingested since every tenant was
  tripped.
- **Stranded rows:** 9 with a NULL broker id, including this tenant's 10-01 UNH sell (id 5366),
  which has never passed through recovery. No UNH fill carries a NULL strategy, so recovering 5366
  cannot change the belief.
- **Timers:** `algua-paper.timer` is inactive; `algua-mergeback-drain.timer` is active.

If 2.2 is deployed before the retirement:

- **Before the offset fills:** the guard skips, the store refuses on positions, and the offset is
  left resting (P9a).
- **After it fills:** the residual 0.000123182 × the latest UNH fill price (~$360) ≈ $0.04, below
  the $1 minimum, so it is dust. The retirement retires and writes no new audit rows (P9b).

The guard also does the ingest the current procedure lacks. With the current code, a retirement
after the open is refused on positions until something ingests (a second `paper flatten`), because
the stuck cursor never sees the fill. The guard needs the paper credentials that `.env` in the main
checkout already provides. Two operational conditions apply: run it from the main checkout so it
shares the timer's lock, and expect a refusal while a merge-back attempt holds `operator.lock` (up to
3600 s). Production is not harmed either way. Deploying 2.2 first makes this exit easier.

## Findings

### BLOCKER

None. Every real path can succeed, and every gap below has a text-only fix.

### MAJOR

**M1 — The §3.1 placement fails the size ratchet.** `algua/contracts/types.py` is 468 lines and
pinned at 468 (tests/test_module_size_ratchet.py:68). The protocol adds at least 4 lines; the
prototype reached 475 and the ratchet failed. The contract says no choice is left to the
implementer, so followed literally it cannot meet AC8 without raising a pin.

**Fix — replace §3.1's first sentence and code block with:**

> `algua/execution/lane_exit.py` declares, beside `PaperExitGuard`:
>
> ```python
> @runtime_checkable
> class PaperExitDrainBroker(ScopedCancelBroker, ActivityWindowBroker, OrderLookupBroker, Protocol):
>     def clock(self) -> str: ...
> ```
>
> It composes the contract protocols from `algua.contracts.types`, which is not edited: it is at its
> ratchet pin (468/468).

In §6, add `algua/contracts/types.py` to the "not edited" list.

**M2 — The race guarantee holds only if the venue publishes fills instantly.** The §3.5 invariant,
the decision log ("cannot fill unseen") and AC5 all assume that a fill which raced the cancel is in
the activity feed when the second sync reads it. §9 itself admits feed lag, and P3b shows the result:

1. The order fills at cancel and the DELETE gets 422.
2. The recheck shows it not open, so `unsettled` is empty.
3. The second sync misses the unpublished fill.
4. The positions check sees flat, the under-lock re-list sees nothing open, and **the exit commits**
   while the venue reports `filled_qty=2`.

The second sync has also moved the shared cursor to the broker's current time, past the fill's
`transaction_time`, so no later ingest will ever fetch it. The account then holds an unexplained
position: the #685 deferral, and then the global halt. Resting market orders mostly exist outside
market hours, which narrows the window to exits near the open. It is still the exact race AC5
exists to cover, and the drain adds two cursor advances at the moment of greatest exposure.

**Fix — append to §3.2:**

> Neither drain sync moves the shared cursor. `venue_sync` gains `ingest_paper_venue_window(conn,
> broker, until)`, which fetches `(cursor, until]` exactly like `ingest_paper_venue` and passes
> `cursor_value=<the cursor it read, or _PAPER_CURSOR_FAR_PAST>` to `ingest_activities`. The guard
> uses it for both syncs; the paper operator keeps `ingest_paper_venue`. Re-fetching is idempotent,
> because ingest de-duplicates by activity id (live_ledger.py:484-538). The next sync, whether the
> operator's or a retried exit's, therefore fetches an activity the venue published late instead
> of skipping it.

**Insert §3.3 step 5a:**

> **Settle (step `settle`).** For each id in `owned` that is not in `unsettled`:
>
> 1. Read the order by the client_order_id that `paper_venue_orders` records for it, using
>    `get_order_by_client_order_id`.
> 2. If `float(order["filled_qty"])` exceeds `abs(SUM(paper_venue_fills.qty))` for that
>    `broker_order_id` by more than `DUST_SHARES`, add the id to `unpublished`.
> 3. A payload that is missing or malformed is a step failure (§4).

**Add to §3.4, after the `skipped` branch:**

> - `unpublished` non-empty: `TransitionError(f"{name}: the venue reports fills on order(s)
>   {sorted(unpublished)} that its activity feed has not published yet; retry shortly")`, with no
>   broker call.

Add T3b (the fill is executed at cancel and published after the second sync: refused; after
publication a retry is refused on positions) and its mutation check (drop the settle step → T3b
fails). In §9, narrow the feed-lag residual to orders that completed before the drain began.

If the owner prefers to keep today's scope, the alternative is wording. Replace "cannot fill unseen"
and the §3.5 invariant with "…unseen, provided the venue has published the fill when the second sync
reads the feed", mark AC5 as covering instant publication only, and record the owner's acceptance in
the story.

**M3 — Default-on drains turn existing tests into real Alpaca requests.** Several test files set
dummy credentials suite-wide in their own autouse fixtures (test_cli_live.py:68-73 for live,
test_paper_intake.py:68-69 for paper; 17 test modules reference an `ALGUA_ALPACA_*` variable). Under the default selector, a
transition in those files constructs a real `AlpacaLiveDrainBroker` (pointed at `api.alpaca.markets`)
or `AlpacaPaperBroker`, and makes HTTPS requests. This was observed in this review (see the
disclosure).

The opt-in fixture plan (§8) fails open in exactly this configuration. A test that forgets to opt
in can still pass while making a request, for example by asserting `exit_code != 0` on a refused
exit. This matters most for Stories 2.4 and 2.6, which add callers.

**Fix — append to §8:**

> `tests/conftest.py` gains an autouse fixture that wraps `requests.get`, `requests.post` and
> `requests.delete`. Any URL whose host ends in `alpaca.markets` raises `AssertionError("real Alpaca
> HTTP in a test")`; other hosts pass through unchanged. The fixture does not disable the drain: a
> test that has not opted in to the exit-drain fixture fails closed instead of reaching the network.
> Tests that fake the HTTP layer patch these names after the autouse fixture runs and are
> unaffected.

The full-suite baseline with this guard passes (6105 passed, 5 skipped; see (6)).

### MINOR

- **m1 `pending_cancel` gives spurious refusals.** A cancel that Alpaca has accepted but not yet
  completed is still listed as open, so the exit is refused with "flatten before this transition".
  Add to §3.3 step 4: "while any cancelled id is still open, re-list up to 3 times at 1-second
  intervals before freezing `unsettled`". Add to the CLAUDE.md `registry transition` note: "a
  refusal right after cancellation clears on retry".
- **m2 AC3 says every drain failure is audited; §4 leaves the under-lock re-list failure unaudited.**
  Either amend AC3 ("…audited, except a re-list failure under the write lock, which rolls back and
  is reported by the envelope"), or add to §2.3: "`transition_strategy` catches the guard's
  under-lock `BrokerError` after `apply_transition` has rolled back
  (`conn.in_transaction is False`), appends `paper_exit_drain_failed` with step `relist`, and
  re-raises". The second option keeps the store untouched and fits the 290-line budget.
- **m3 Lock scope and the challenge TTL.** Add to §2.5:
  - "Exclusion holds only when the command runs from the operator's checkout (production: the main
    checkout, the `algua-paper.service` WorkingDirectory); a worktree resolves a different
    `operator.lock`."
  - "A go-live refused for a held lock leaves its 10-minute challenge unconsumed but may outlive it;
    check that the paper and merge-back services are idle before requesting the challenge."
- **m4 Wording in §2.5.** The text says "so exits wait for it as retirement does today", but the lock
  is non-blocking. Replace with "so an exit is refused while merge-back holds it (up to
  `TimeoutStartSec=3600`) and is retried, as retirement is today".
- **m5 The cancelled-orders row overstates.** `cancel_order` returns silently on 404/422, so
  "cancelled {ids}" lists orders that were not cancelled (the T4 case). Change the reason to
  `f"requested cancel of {len(ids)} resting paper order(s): {ids}; still open after recheck:
  {unsettled}"`, written after the recheck.
- **m6 Live audit timing changes.** Today `_live_exit_guard` runs before validation and before the
  lock, so `live -> dormant` without a reason, or `live -> retired` with the lock held, writes
  `live_exit_drain_*` audit rows and builds a broker before being refused. After 2.2 it does
  neither. Messages are unchanged, but the audit side effects are not byte-identical. State this in
  §2.4 and pin it in T15.
- **m7 The test fixture must also cover the live authorization read.** Add to §8: "`empty_exit_venues`
  also patches `algua.registry.live_gate.verify_live_authorization` to raise `LiveAuthorizationError`
  unless the test supplies an authorization; 9 of the 16 live-source cases otherwise raise
  `StrategyNotFound` from the identity recompute before any broker builder runs." Read
  `ALLOWED_SIGNERS_PATH` through `live_gate.` at call time, so that tests patching
  `algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH` before a live exit (none today) can re-target one
  attribute.
- **m8 Operator lock in tests.** More edges now take the real git-dir lock. Without a redirect,
  running the suite from the main checkout while the merge-back drainer holds `operator.lock` fails
  these tests spuriously. `empty_exit_venues` should redirect
  `algua.operator.deployment_lock._operator_lock_path` to `tmp_path`, as
  test_deployments.py:563 does.
- **m9 `lane_exit.py` budget.** The prototype is 211 lines without docstrings, and M2 adds about 20.
  Add to §6: "if `lane_exit.py` would reach 300 lines, `PaperExitGuard` and its protocol move to
  `algua/execution/paper_exit_guard.py` (CODEOWNERS + integrity-critical); no pin is added".
- **m10 Docs.** Add to §8:
  - `deploy/systemd/README.md:126`: frozen retirement now drains and needs the paper venue.
  - The `cli/paper_venue.py` module docstring, which no longer holds ingest or recovery.
  - `docs/contracts/cli-error-envelope.md` is unchanged, as stated.
- **m11 Story 2.1 coordination**, which works in either merge order. Add to §6: "If Story 2.1 merges
  first, drop inventory row 7's raw `paper -> forward_tested` and T14's human-raw case (2.1 refuses
  the edge before the selector), and keep a single `.apply_transition(` single-caller test (2.1
  adds the same assertion)."

### NOTE

- **N1** The under-lock re-list holds the SQLite write lock across one HTTPS call. The worst case is
  3 attempts with 30 s timeouts plus 1.5 s of backoff, so other writers can see `db_unavailable`
  after the 5 s `busy_timeout`. This is the same as the live guard, and AC2 requires the call.
- **N2** `broker_error` and `wrong_stage` are non-retryable while the messages say "retry". This is
  the existing convention, unchanged.
- **N3** A forward-tested strategy must already be flat in paper and unswitched to go live
  (crud.py:340-347, registry/live_certificate.py:138-149). This predates the story; the handoff
  belongs to Story 2.4.

## Story 2.1 Coordination

**No semantic conflict.** Story 2.1 changes what `_validate_live_gate` checks: one
`verify_live_qualification` call, before the lock and before the approval verifier. It also deletes
the raw `backtested -> candidate` and `paper -> forward_tested` edges ahead of `validate_transition`.
Story 2.2 changes what happens after validation: the lock rule, guard selection and the
`apply_transition` kwarg. On go-live the order stays: live gate (2.1) → lock → paper drain (2.2) →
under-lock re-check → challenge consumption.

**Textual conflicts the second story to merge must resolve:**

- **transitions.py:**
  - the `repo.apply_transition(...)` call (2.1 drops `consume_gate_id`/`consume_forward_gate_id`;
    2.2 replaces `exit_guard=`);
  - the deleted branches at :75-93, which sit directly above 2.2's lock block at :94-96;
  - the `transition_strategy` signature.
- **registry_cmd.py:** 2.2 deletes the imports at :16-21; 2.1 rewrites :22 and :216-217.
- **CODEOWNERS and `INTEGRITY_CRITICAL_MODULES`:** adjacent insertions.
- **tests/test_book_exit_revoke.py:300-335:** 2.1's `qualified_live_world` against 2.2's drain
  fixture.

**Sizes in either order:** `transitions.py` is about 245-250 lines; `registry_cmd.py` is about 378.
2.1's issuance edit is line-neutral, so a lowered 2.2 pin holds.

**Merge order: Story 2.2 first.** It reduces risk, has no schema change, unblocks Stories 2.3 and
2.10, and simplifies the last legacy exit. Story 2.1 (v49, forward-only) then rebases:

- it deletes 2.2's raw-forward case from T14;
- it reuses 2.2's T16 single-caller assertion;
- it keeps `exit_guard=guard` in the `apply_transition` call.

If 2.1 merges first, m11 covers 2.2's side.

## Risk to Existing Behaviour

- **Store:** untouched apart from one docstring. `apply_transition`, `_assert_flat_for_bench`, the
  dust rule and both "not flat" messages are reused as they are.
- **Paper run-all and flatten:** they import the moved functions from `venue_sync`. Behaviour is
  unchanged.
- **Live exits:** the same guard, messages and audit actions. Selection now happens after
  validation and inside the lock (m6).
- **New refusals:** `paper -> dormant` and go-live are now refused while `operator.lock` is held.
- **Production:** limited to the one remaining paper tenant (see (7)), and only made easier.

## Summary and Verdict

**Overall readiness: READY WITH CONDITIONS.** The entry-point inventory is complete. The guard
algorithm is sound against the concurrent operator (with the checkout caveat), clock outages, the
500-order cap, stranded orders, dust, partial fills, missing credentials and the go-live challenge.
No audit or commit happens under the write lock. The Alpaca calls match the broker code, and
production is unaffected.

Three MAJOR corrections must be applied first:

- **M1:** move `PaperExitDrainBroker` out of the pinned `contracts/types.py`;
- **M2:** close the published-late-fill race with a non-advancing drain cursor and a `filled_qty`
  settle check, or narrow the claim with owner acceptance;
- **M3:** block real Alpaca hosts suite-wide, so the opt-in drain fixture fails closed.

After that, a delta check of §3, §6 and §8 is enough; no further full review is needed. Fold in
m1-m11 as convenient. Readiness never authorizes trading, capital use or a production run.
