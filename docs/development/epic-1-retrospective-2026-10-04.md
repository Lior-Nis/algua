# Epic 1 retrospective: preserve qualified strategy behavior while development continues

Date: 2026-10-04. Epic window: 2026-09-24 (canonical vision, PR #665) to 2026-10-03 (Story 1.4,
PR #686). Participants: Lior (owner), Claude (implementation, coordination, fallback review), Codex
(implementation of Stories 1.1-1.2, review until its quota ran out on 2026-09-29), and BMAD reviewer
subagents (Blind Hunter, Edge Case Hunter, Acceptance Auditor).

Method: the `bmad-retrospective` structure, run non-interactively. Sources are the story records,
normative specs and readiness reports in `docs/development/`, merged PRs #665-#686, issues #624,
#643, #662, #677, #682 and #685, a read-only query of the production registry on 2026-10-04, and
facts recorded by the delivery session. This is the first retrospective of the Phase 1 cycle, so
there are no earlier commitments to check.

## 1. Outcome against the epic goal

Goal (`epics.md`): "a strategy can accumulate trustworthy paper evidence against a recoverable
immutable deployment while unrelated repository development proceeds."

Verdict: the capability is delivered and deployed, but it has not run in production yet.

- **Delivered.** A new paper strategy is prepared as a content-addressed source bundle plus a locked
  environment (1.3b) and admitted only through frozen intake. Each planner phase runs in a fresh
  subprocess from that content while the supervisor keeps every authority (1.3c). Ticks count as
  evidence only when linked to a recorded, successful attempt, and `paper promote` qualifies the
  strategy from its recorded descriptor after fresh verification (1.3d). One test replays a restart
  byte-exact after the checkout module is edited or deleted. An opt-in test ran prepare, verify and
  dispatch against a real uv environment.
- **Not yet run in production.** On 2026-10-04 the production registry (schema v48) held 0
  `strategy_deployments`, 0 `deployment_artifacts` and 0 `frozen_invocations` rows, and 0
  candidates; 3 strategies are at `backtested`. No strategy has earned a single session of frozen
  evidence.
- **One step remains.** `liquidity_stable_quality_momentum` is the last legacy tenant still at
  `paper`. Its capped UNH sale (3.55273207 shares) fills at the 2026-10-05 open. After the fill:
  `paper flatten` to ingest it (running it before the fill would cancel the sale), retire the
  strategy, confirm the paper account is flat, and restart `algua-paper.timer`. That completes
  Story 1.4 AC7. The other 17 legacy tenants were retired on 2026-10-03.

The outcome is proven only when the first candidate is admitted frozen and earns linked evidence
in production (action A3). That depends on the research funnel, not on Epic 1 code.

## 2. Delivery record

| Story | Ready -> merged | PRs (contract, code) | First readiness verdict | Review | Recorded findings |
|---|---|---|---|---|---|
| 1.1 planner extraction | 09-24 -> 09-24 | #666, #667 | READY (1.1 only) | 1 round | 1 patch, 2 deferred |
| 1.2 deployment epochs | 09-24 -> 09-24 | #668, #669 | none recorded | 1 round | 11 patches |
| 1.3 parent outcome | 09-24 -> decomposed 09-25 | #671, #672 | NOT READY (too large) | n/a | split into 1.3a-d |
| 1.3a two-phase boundary | 09-25 -> 09-26 | #673, #674 | NEEDS WORK, rerun READY | 1 round | 11 patches |
| 1.3b recoverable artifacts | 09-27 -> 09-30 | #675, #676 | READY | ~31 rounds (4.1) | 166 patches, 32 superseded |
| 1.3c frozen paper execution | 09-30 -> 09-30 | #678, #680 | NOT READY (1 blocker, 7 major) | 1 round + verify | 14 patches, 4 deferred |
| 1.3d evidence, qualification | 09-30 -> 10-01 | #681, #683 | READY WITH CONDITIONS | 1 round | 7 patches, 3 dismissed, 2 deferred |
| 1.4 legacy cohort exit | 10-01 -> 10-03 | #686 | NOT READY, owner rescoped | 1 combined round | 5 patches, 2 deferred |

- **Implementers.** Codex implemented 1.1 and 1.2, per their Dev Agent Records. 1.3a's two commits
  have no Claude trailer. Claude implemented from 1.3b onward, after `docs/agent/story-delivery.md`
  (2026-09-28) assigned implementation to Claude and review to Codex. In 1.3b, 116 of 152 commits
  carry a Claude trailer.
- **Suite and boundaries.** The root suite grew from 4,016 tests passing at 1.1 to 6,105 passing
  (5 skipped) at 1.4. The import contracts kept went from 28 to 29.
- **Schema and deploys.** v48 (`frozen_invocations`) was first run against a copy of the production
  registry (1,025 legacy ticks) during the 1.3d review and applied on 2026-10-01. 1.3c was deployed
  on 2026-09-30 and 1.4 on 2026-10-03. Outside the stories, #679 (incident fix) and #684 (urllib3
  advisories) also merged.

## 3. What worked

1. **Readiness review caught blockers before any code.** Story 1.3 was too large for one cycle. In
   1.3c, the strategy-view mechanism failed for every strategy (B1). In 1.4, the first design
   wedged 6 production tenants on float dust and could not compose its transaction (B1/B2). Each
   was fixed in contract text before implementation began.
2. **Contracts came first.** From 1.3a on, each story merged its normative SPEC and field-level
   companion as a separate docs PR before implementation. Reviewers judged code against a fixed
   contract instead of re-arguing scope.
3. **Decomposition paid off.** After 1.3b, each child story needed one review round. 1.3c went from
   readiness to merge and deploy on 2026-09-30. 1.3d went from contract to merge within a day.
4. **One bounded review round converged.** 1.3c's single round covered three file groups, an
   acceptance audit and a real-environment test. It produced 14 fixes, which one verification pass
   confirmed with reproductions, reverts and a 4,000-scenario differential fuzz.
5. **Real-environment testing found defects unit tests missed.** The opt-in test exposed two 1.3b
   defects: environment digests were not reproducible (uv writes the staging path into
   `bin/activate.csh`), so every admission would have published a new ~1 GB environment; and
   orphaned `__pycache__` directories on the main checkout blocked every frozen intake as
   `frozen_source_drift`.
6. **Read-only rehearsals on production data.** v48 was migrated on a copy of production before
   merge. The 1.4 dust rule was applied read-only to the production ledger during review.
7. **Reviews found authority holes before production** (pattern in 4.3). None shipped.
8. **The walls held.** No story widened agent authority, go-live stayed human-signed, and frozen
   deployments are refused at go-live until Epic 2.
9. **Owner decisions were fast when framed as a clear choice.** "Cut it" (1.3b analyzer, 09-30) and
   "kill all strats" (1.4, 10-01) each removed a large body of work the same day.
10. **Mutation checks kept "green" meaningful:** 12 of 12 mutants caught in 1.4; gates by exit code.

## 4. What did not work

### 4.1 Story 1.3b's review loop did not converge

Between 09-27 and 09-30 the story record shows about 31 review rounds: an initial 18-patch round,
rounds 2-15 on the strategy-closure refresh seam, two publication rounds, eleven commit-pinned
rounds, two rescue rounds and the final BMAD review of the rescope.

A test-only guard ("no direct `bounded_walk` call") grew from a regex into an AST guard, then into
a ~3,400-line Python flow analyzer. Each round found new static evasions. The rescope superseded
29 open analyzer findings, and removing the analyzer took the suite from 5,529 to 5,109 tests.
Meanwhile the normative `uv sync` argv carried `--no-env-file`, which uv 0.9.26 rejects, so
`deployment prepare` could never provision an environment. That defect sat as a deferred "contract
change".

Root causes:

- A static guard over arbitrary Python has no fixed point, and there was no stopping rule.
- The threat model was never stated. The realistic one, accidental direct use by three callers,
  was solved by construction in one change (a module-private `_bounded_walk`).
- A deferral label hid a functional blocker: a finding meaning "the real path cannot succeed" was
  treated as deferrable.
- The contract pinned the bad argv and nobody executed it. Contract-first delivery carries a wrong
  fact about an external tool forward until something actually runs it.

### 4.2 The shared paper account drove most operational problems

All four production problems in section 5 share one cause. Eighteen tenants traded on one Alpaca
paper account. Sizing, order ids, cancels and reconcile were designed per tenant but executed
against an account-wide venue. Those problems cost sessions and set most of Story 1.4's scope: the
dust rule, the scoped cancel and the ledger-net cap.

### 4.3 The top finding was usually a second door into a guarded state

In each story below, a reviewer found a path that reached guarded state without going through the
guarded path:

- **1.2:** the generic transition API bypassed atomic deployment intake.
- **1.3c:** the raw `registry transition --to forward_tested` and the go-live path both bypassed the
  frozen promotion refusal. Separately, the supervisor trusted the result kind a child process
  chose, so drawdown, reconcile, realized-gross and mark walls could be skipped.
- **1.3d:** `run_forward_gate` trusted any (deployment, hashes) pair a direct caller passed in. The
  fix is an opaque `PromotionIdentity` that only the promotion chokepoint can mint.

#682 (pre-existing, still open) has the same shape. Root cause: the contracts specify the intended
path to a guarded state change but do not list every other path that reaches it.

### 4.4 Readiness rejected first designs, and one was thrown away

Four of seven first-pass readiness verdicts were not ready (1.3 parent, 1.3a, 1.3c, 1.4). That is
the gate working. However, 1.4's first design, a requalification-preserving `migrate-legacy`
command, was discarded entirely. The review showed its cost: helpers that commit themselves, float
dust blocking 6 tenants, no source for open orders, and an unnamed operator lock. Once that cost
was visible, the owner chose to retire the whole cohort. Root cause: the cheap option was not put
beside the expensive one before the contract was written.

### 4.5 Recurring structural friction

- **Size ratchet.** `alpaca_broker.py` sits at its pin (547/547), `live_ledger.py` at its pin
  (620/620), and `paper_cmd.py` at 1,387/1,409. To land a production fix, #679 had to move helpers
  into `alpaca_rejections.py` and `sizing.py`. 1.3b carved out `module_source_scan.py`,
  `module_commit_check.py` and an artifact-ledger mixin. Each carve was correct, but each was
  unplanned work on protected modules, and Epic 2 changes exactly these modules.
- **Test fixtures in the source tree.** Tests write fixture strategy modules into the source tree
  (#643). With parallel subagents in one worktree, concurrent runs collided on fixed file names.
  The fix was deferred in 1.3c and again in 1.3d.
- **Unpinned uv in CI.** CI installs the latest uv. A branch's first CI run exposed a test that
  depended on the uv version (1.3b).
- **Status drift.** The "Current handoff" in `docs/development/README.md` still lists 1.3d as
  `ready-for-dev`, and `sprint-status.yaml` was last updated on 09-30. Story 1.4 still reads
  `review`, with "Merge and deploy" unchecked, after PR #686 merged.

### 4.6 Review independence dropped mid-epic

Codex was out of quota from 2026-09-29 until 2026-10-04. The reviews of 1.3b's rescope, 1.3c,
1.3d and 1.4 therefore used the BMAD fallback: Claude subagents reviewing Claude's code. It found
substantive defects in every one of those stories (the trusted child result kind, the forward-gate
identity, the global-halt path), so it worked, but it is one model family checking itself.

## 5. Production incidents and lessons

1. **Every paper cycle aborted, 2026-09-25 to 09-30 (#677).**
   - *What happened:* full exits were sent as notional orders sized at the prior close. Alpaca
     converts at the current price, so on a down move the order asked for more shares than the
     tenant held. Alpaca returned a 403, and the resulting `BrokerError` is book-critical, so it
     stopped every later tenant.
   - *Silent variant:* where sibling tenants held the same symbol, the oversell was accepted and
     long-only tenants went short (BAC, MRK, UNH). Account-level reconcile nets across tenants, so
     it could not see this.
   - *Fix:* PR #679 trades exact shares on full exits.
   - *Lessons:* nothing checked the per-tenant invariant that a long-only tenant is never short.
     The outage also lasted five sessions with no alert: no unit in `deploy/systemd/` runs
     `fleet health` or declares `OnFailure=`.
2. **Production ran unmerged code.**
   - *What happened:* an earlier agent left the box's main checkout, which is the systemd working
     directory, on `feat/story-1-3b-frozen-artifacts`. Merge-back failed closed, because it
     requires a clean `main`. `paper run-all` has no such check and kept running.
   - *Fix:* operational only. Deploy now means fast-forwarding that checkout while the services
     are idle.
   - *Lesson:* the operator checkout is production. Agents work in worktrees, and the runtime
     should refuse to run, or at least report, a checkout that is not on `main`.
3. **Session 2026-09-30 was lost (#662).**
   - *What happened:* the morning catch-up run for 09-29 and the evening run for 09-30 both
     decided on the 09-29 bar. They produced the same `client_order_id` for opposite sides (a UNH
     buy, then a sell). Duplicate-id recovery correctly refused to attribute the order to the
     second run, so the cycle aborted every 20 minutes.
   - *Status:* still open as #662 slice 1.
   - *Lesson:* recovering from an outage creates states the steady-state design never meets. Order
     idempotency keys must include the operator session.
4. **The legacy exit stalled on leftovers from #677 (2026-10-01).**
   - *What happened:* notional fills had split the account's UNH across two tenants: +3.552855 and
     -0.000123 shares, against an account holding 3.55273207. The sale was refused for quantity
     and the buy-back for falling under the $1 minimum. Six tenants also carried float residuals
     around 1e-16 shares, which the strict bench check refused.
   - *Fix:* Story 1.4 added the dust rule, the ledger-net cap and the scoped cancel.
   - *Near miss:* 1.4's review found that retiring with dust would have dropped the residual out of
     the account reconcile and engaged the global halt, which also stops `live run-all`. It was
     caught before merge.

## 6. Process changes adopted during the epic

- 09-25: a readiness review per story, rerun after any amendment (introduced by the 1.3
  decomposition).
- 09-25: contract first. The normative SPEC and field-level companion merge as a docs PR before
  implementation starts (1.3a onward).
- 09-28: `docs/agent/story-delivery.md` sets the model: Claude implements, Codex reviews, and work
  is mirrored to Todoist.
- 09-29: when Codex is unavailable, the BMAD fallback runs: a Blind Hunter and an Edge Case Hunter
  per bounded file group, plus an Acceptance Auditor over the whole story.
- 09-30: non-converging review loops are cut. Invariants are enforced by construction against a
  stated threat model (1.3b).
- 09-30, from 1.3c on: parallel Claude subagents work in one worktree, each owning its own files;
  only the coordinator commits, using scoped adds; one bounded review round, then fix agents, then
  one verification pass.
- Release discipline: every new guard is mutation-checked; the full gate is judged by exit code;
  merges happen only on green CI with `--match-head-commit`; deploys fast-forward the box's main
  checkout while the systemd services are idle.
- Reviews rehearse read-only against production data (1.3d, 1.4).

Only the readiness step (the README), the implement/review split and the fallback reviewers
(`story-delivery.md`) are written into the repository. The contract-first docs PR, the single-round
rule, a stopping rule, file ownership, mutation checks and the deploy procedure are not (action A5).

## 7. Carried-forward technical debt and open issues

Open issues:

- **#662:** the remaining shared-paper slices: `client_order_id` uniqueness (the cause of the 09-30
  loss), wash-trade rejection between tenants, and `venue_blocked` evidence and health.
- **#682:** a raw `registry transition --to forward_tested --actor human`, and the raw shortlist
  edge, are unauthenticated and skip the forward gate for non-frozen strategies. Frozen deployments
  are refused on that edge, and go-live still needs a signature and a fresh certificate.
- **#685:** paper-lane bench exits do not re-check for resting orders (the live lane does). Until
  fixed, the operator must flatten and wait for fills before retiring a strategy.
- **#643:** tests write strategy modules into `algua/strategies/`.
- **#624:** the Phase 1 parent issue, which holds Epic 2's scope.

Untracked, with no open issue:

- **#677 follow-ups.** #677 is closed, but partial rebalances are still notional and can still leave
  cross-tenant residuals, and there is still no per-tenant check for negative quantities in
  long-only tenants.
- **Legacy and working-tree tick paths.** These can be deleted now that the cohort is empty; the
  1.4 SPEC lists this as a non-goal.
- **Name-reuse laundering policy.** Deferred by the 1.4 owner decision.

Recorded deferrals:

- **1.1:** a NaN holding weight silently omits an intent, and mixed non-string labels can raise
  `TypeError` (`deferred-work.md`).
- **1.3b:** uv is not pinned in CI (`deferred-work.md`). Model-backed strategies are refused by
  frozen preparation, so no model lane can be frozen yet.
- **1.3c:** the bars digest is recomputed per phase, so both phases would time out near 80% of the
  256 MiB bound (today's bar sets are 10-25 MiB); breach messages over 8 KiB are truncated on the
  frozen side only; intake re-provisions environments for candidates it then refuses on capital
  (up to 124 s per frozen tenant per cycle when an environment hangs).
- **1.3 parent:** artifacts are kept with no garbage collection, and each environment is about
  1 GB.
- **1.4:** a stale last-fill price can misjudge a residual within about $1 of the venue minimum.

## 8. Readiness of Epic 1's outputs for Epic 2

**Epic 2 can rely on:**

- the two-phase versioned planner seam, `deployment prepare`/`verify`, and frozen intake as the
  only `candidate -> paper` path;
- subprocess execution in which the supervisor re-derives every strategy-free wall, and append-only
  attempt evidence, with frozen-epoch evidence limited to linked ticks;
- `paper promote` of frozen deployments through the unchanged forward gate, behind a
  `PromotionIdentity` chokepoint, with the human-actor challenge bound to the deployment epoch and
  manifest digest; deployment epochs with no back-crediting;
- the shared dust rule, the sibling-safe flatten, schema v48 in production, and the box checkout
  on `main` at `e875b6d`.

**Epic 2 must not assume:**

- **Frozen go-live works.** Go-live and the live lane refuse frozen deployments
  (`frozen_live_unsupported`). Deployment-bound live signing (FR8) is not built.
- **A deployment or candidate exists.** Production has no frozen deployment, no deployment record
  and no candidate. The paper book will be empty after A1, so no live candidate's evidence clock has
  started.
- **63 sessions is a pass.** 63 observations is the forward gate's floor, not a pass condition.
  Clearing the LCB bar at that floor needs an observed annual Sharpe of about 3.8
  (`algua/research/forward_gates.py:53-55`). The owner decided on 2026-10-04 that a relaxed gate or
  certificate can never authorize go-live.
- **Preparation has run in production.** A real `deployment prepare` has never run against the
  production registry, and environments have no garbage collection.
- **Capital policy is enforced.** The 10% pause and the instrument and capital policy are not
  implemented. Current defaults are `book_max_drawdown` 15% and `book_max_gross` 2.0 (#624).
- **Unattended operation and alerting exist.** No 72-hour unattended run has happened, and there is
  no alerting path (5.1).
- **The paper qualification lane is clean.** It still carries the shared-account defects (#662,
  #685).

**Findings that change Epic 2 planning.** This is a change of sequencing and acceptance, not of
direction; it is for whoever prepares the Epic 2 stories.

1. Live acceptance under FR1 depends on research output and forward-evidence time, not on
   engineering. Epic 2 stories should be built and verified against fixture or paper deployments,
   and should not wait for a live-eligible strategy.
2. Owner decisions on 2026-10-04 settle the policy behind FR11 and FR14:
   - the pause measures drawdown from a cash-flow-adjusted equity high-water mark;
   - it triggers at >= 10%;
   - it halts, cancels resting orders and keeps positions;
   - a signed resumption re-bases the peak;
   - live qualification must pass unrelaxed gates.

   Both stories can be prepared now.
3. Any outage catch-up collides on order ids (#662 slice 1). That fix should come before the FR13
   72-hour exercise.
4. The broker and ledger modules Epic 2 must change are at their size pins. Plan the carve as the
   first task of each such story.

## 9. Action items

| ID | Action | Owner | Due condition | Done when |
|---|---|---|---|---|
| A1 | Retire `liquidity_stable_quality_momentum`: after the UNH fill, `paper flatten` (ingest), retire, confirm the account is flat, start `algua-paper.timer` | Claude | After the 2026-10-05 open fill is visible at the broker | 0 strategies at `paper`; paper account flat; the next timer cycle exits 0 |
| A2 | Close the Epic 1 records: Story 1.4 to `done` with AC7 evidence, the README handoff, and `epic-1` plus `epic-1-retrospective` in `sprint-status.yaml` | Claude (the agent that owns those files) | Same change set as A1's evidence | All four files agree; the gate passes |
| A3 | Prove the outcome in production: admit the first candidate frozen, then verify a `strategy_deployments` row, `frozen_invocations` rows and linked ticks, and record environment disk use | Claude | First candidate from `research promote` | One frozen tenant with at least 1 linked tick, recorded in a short evidence note |
| A4 | Add outage alerting: a `fleet health` watchdog unit with `OnFailure=` notification | Claude builds; Lior picks the channel | Before a frozen tenant trades, and before the FR13 72-hour exercise | A forced failed cycle reaches Lior |
| A5 | Write the section 6 practices into `docs/agent/story-delivery.md`. Add a stopping rule: if a second round finds new findings in the same guard, stop and present an enforce-by-construction option. Add a triage rule: a deferral meaning "the real path cannot succeed" is a blocker | Claude | Before Story 2.1 implementation starts | The doc states each rule; the Story 2.1 record cites it |
| A6 | Add an entry-point inventory to every contract and readiness check: list each path that reaches the guarded state change, and execute each pinned external-tool invocation at readiness | Claude | Story 2.1 contract | Readiness report includes the inventory table |
| A7 | Make `paper run-all` and `live run-all` (or `fleet health`) refuse, or report, a checkout that is not a clean `main` | Claude (issue + story) | Issue filed before Epic 2 planning review; fix before the 72-hour exercise | Test proves a run on a non-main checkout is refused or reported |
| A8 | Fix #662 slice 1 (session-aware `client_order_id`) | Claude | Before the 72-hour exercise | A catch-up plus regular run on the same bar no longer aborts (test) |
| A9 | File an issue for the #677 follow-ups: a per-tenant negative-quantity check for long-only tenants, and notional partial rebalances | Claude | With A2 | Issue open, with the 09-29 BAC/MRK/UNH evidence |
| A10 | Fix #643 with a `tmp_path` strategy-family fixture | Claude | Before Epic 2 work runs parallel subagents in one worktree again | Full suite leaves `algua/strategies/` untouched (hygiene test) |
| A11 | Decide #682: authenticate raw forward and shortlist edges, or remove them for all actors | Lior | Before the Epic 2 live-wall story is ready | Decision recorded on #682 |
| A12 | Restore Codex as the independent reviewer for Epic 2 authority stories (FR8, FR11), keeping the one-round rule | Claude | Story 2.1 review | Story record names the Codex review round |
| A13 | Complete the account and deployment prerequisites (Todoist `6hcQ3CHr49rCRV3p`) | Lior | Before live activation (it does not block building) | Task closed with evidence |

## 10. Team agreements

- One bounded review round and one verification pass per story. A second round that keeps finding
  evasions in the same guard is a stop-and-decide point, not round N+1.
- A finding that means "the real path cannot succeed" is never deferred.
- Every contract lists every entry point to the state it guards.
- Agents never change the branch of the box's main checkout. Deploying means a fast-forward while
  the services are idle.
- Tests passing, code merged, deployment verified and evidence earned are reported separately.
