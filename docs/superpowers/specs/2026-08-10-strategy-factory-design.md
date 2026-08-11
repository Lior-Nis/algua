# The Strategy Factory — market-first selection at scale (design)

## Context

Algua's moat, per the operator, is scale: test as many **uncorrelated** alpha hypotheses as
possible and let harsh thresholds pass the few real ones. The current funnel was calibrated for
low throughput: the promotion gate rations binding FDR tests (16/365d, #529), funnel-wide breadth
deflation raises the bar with every measured sweep, and the agent novel-family seed inherits the
funnel-wide lifetime search total (#524). At factory throughput those controls strangle the very
volume that is the point. Meanwhile compute is explicitly NOT a constraint (the operator will
scale Codex subscriptions with ROI), and backtest research iterates over a historical panel —
throughput is not limited by the one-bar-per-day arrival of live data.

This design was brainstormed with the operator (2026-08-10); every decision below is theirs.

## The philosophy pivot: market-first selection

Backtest evidence is cheap and search can overfit it — that is why the DSR/FDR/breadth stack
exists. Forward evidence cannot be overfit by prior search: it did not exist when the search ran.
The factory therefore **moves selection from the backtest gate to the forward gate**:

- The `backtested -> candidate` gate becomes an **integrity floor** (correctness, not statistics).
- The paper book becomes **wide** (~50–100 concurrent virtual slices) and is the new rationed
  resource, FIFO as today.
- The **forward gate is unchanged** and becomes THE harsh threshold: ≥63 broker-clocked
  observations, realized Sharpe ≥ max(0.5×holdout, 0.3), vol/drawdown bounds, ≥90% session
  coverage. Note the self-punishing property: overfit-inflated holdout Sharpe RAISES the forward
  bar for that strategy.
- The **live wall is untouched**: human ssh signature + fresh forward certificate, as today.

The DSR/FDR/breadth/family machinery is **not deleted**: it still computes and is recorded in
`gate_evaluations` as advisory telemetry (and its holdout Sharpe still parameterizes the forward
bar); it stops vetoing on the agent path. Rejected explicitly: scratch pre-screening that feeds a
gate pretending breadth is small (the file-drawer hole — it silently falsifies the PASS stamp).
Selection now happens where multiplicity cannot reach, so honest accounting is preserved without
per-test deflation.

## Ground truth this design builds on (verified 2026-08-10)

- `.codex/scripts/run-research-loop.sh` is **explore-isolated**: each run copies the registry +
  snapshots into a throwaway worktree's `.funnel-scratch`; the Codex agent's `research promote`
  is a realistic PREVIEW (sees real breadth/families/burned holdouts) that cannot mutate the real
  funnel. The driver commits authored strategies to `research-run/<stamp>` and stops.
- `paper merge-back` (#485) is the **sole authoritative reconciler** — currently human-run only;
  no loop run has ever been merged. Four research-run branches are parked unmerged.
- The installed research unit is a systemd **user** unit (`~/.config/systemd/user/`), daily at
  06:00 UTC. `algua-paper.timer` is **not installed** — paper evidence does not accrue today.
- The authoritative ledger has exactly one binding FDR test (2026-07-27, pre-launcher); the daily
  loop burns FDR only in scratch copies.

## Target architecture

```
[every 2h] research run (explore-isolated, N hypotheses, anti-dup context)
    └─ preview soft-gate PASS?ES → driver runs AUTHORITATIVE `paper merge-back` (slice 3)
            └─ diff policy → preview-merge → full quality gate → real promote → FIFO paper intake
[daily]   paper run-all tick (timer, slice 3) → evidence accrual + EARLY-KILL slot recycling
[≥63 obs] forward gate (unchanged) → forward_tested
[human]   signed go-live (unchanged)
```

Feedback for improving algua: every run appends a structured outcome record to a durable digest;
the monitor PWA (Funnel/Activity/Ideas + fleet health) is the observation surface.

## Decisions (operator-approved)

1. **Selection philosophy: market-first** (over gate-recalibration and over a hybrid ladder).
2. **Binding integrity floor** at `backtested -> candidate` for agents: PIT universe; delisting
   handling; default-on costs; holdout ≥ 63 observations; signal-panel parity; holdout Sharpe
   > 0; clean integrity + account hygiene. Everything statistical (DSR confidence, FDR ledger,
   breadth deflation, regime robustness, family breadth terms) records as advisory. The FDR
   throttle stops binding on this path (ledger rows still written for audit).
3. **Paper slot economy: wide book + early-kill.** Target 50–100 equal virtual slices, FIFO
   intake (`promoted_queued` when full). Early-exit: at ≥30 observations, realized Sharpe below
   a hard floor or a drawdown breach → auto-retire with reason, slot recycles. Survivors face the
   unchanged 63-obs forward gate. Single-account attribution at this width is an explicit
   stress-test target.
4. **Cadence: every 2h** (12 runs/day), with cadence and future runner-count as configuration —
   scaling to more Codex subscriptions must be a config change. Overlap protection stays the
   existing non-blocking flock (an overlapping run skips).
5. **Exam-policy note:** an earlier decision (separate examiner rationing the 16/365d budget) was
   SUPERSEDED by the market-first pivot — with the statistical exam advisory, there is no binding
   budget to ration; paper capacity + FIFO is the rationing.
6. **Auto merge-back (the one new trust step).** When a run's preview promote PASSES, the trusted
   driver — not the sandboxed agent — invokes authoritative `paper merge-back` for that branch,
   inside #485's existing guardrails: repo-global `merge_back.lock`, allowlist/CODEOWNERS-denylist
   diff policy BEFORE any merge, preview-merge + full quality gate on the staged tree, commit only
   on green, revert on proven promote failure, per-attempt idempotency token. Mutual exclusion
   with paper ticks (#316 operator discipline) is encoded as schedule separation plus the lock.
   Ships as its own slice so the operator can veto before it lands.
7. **Diversity minimum now, diversity workstream later.** Near-term: the driver injects recent
   run-report hypothesis summaries into the prompt as "already tested — do not repeat", and the
   thesis rotates across alpha categories. The structured idea pool (+ `source-ideas` feeding,
   category quotas, richer dedup) is a separate workstream. LLM-inside-strategy lanes
   (needs_model) are out of scope here.
8. **Data roadmap acknowledged, out of scope:** 2y daily via yfinance now (agility); deeper
   history (polygon/databento/firstrate) when the flow is proven — more holdout observations =
   higher-powered forward decisions, same machinery.

## Slices (each its own spec'd PR; CODEOWNERS-touching slices are human-merged)

**Slice 1 — loop throughput + feedback (no CODEOWNERS):**
- Timer to `OnCalendar=0/2:00 UTC` (12/day), `Persistent=true`; cadence env-documented; repo
  template + installed user unit both updated.
- Launcher: `THESIS` rotation across a small alpha-category list (momentum, mean-reversion,
  seasonality, vol-structure, cross-sectional value/quality proxies, liquidity/microstructure —
  configurable file); anti-dup context block built from the last N run-reports' hypothesis
  sections (trusted driver read of `research-run/*` branches); `N_HYPOTHESES` stays agent-guided.
- Feedback digest: driver appends one JSON line per run to `data/research-runs.jsonl` (stamp,
  branch, hypotheses attempted, stages reached, preview gate verdicts + failed checks, wall time,
  codex exit, rate-limit signals) — the improve-algua backlog. Failure of the digest write never
  fails the run.
- Install `algua-paper.timer`/service as user units on this box (daily post-close tick) so paper
  evidence accrues the moment the first strategy arrives; schedule offset from research runs.

**Slice 2 — the soft gate (CODEOWNERS: gates/promotion; human-merged):**
- Agent-path `research promote`: binding set reduced to the integrity floor (decision 2);
  statistical stack computes + records as advisory (`*_binding=false` semantics);
  `gate_evaluations` rows are still written; LORD++ BINDING stream rows are NOT written while
  stats are advisory (`fdr_binding` NULL, skip reason `stats_advisory`); the ledger machinery is
  preserved for re-tightening. Human paths unchanged. Tests pin: floor
  failures still fail closed; statistical failures no longer veto; advisory fields still recorded;
  forward-bar parameterization (0.5×holdout) unchanged.
- Rollout invariant: slice 2 may merge before slice 3 only because authoritative
  merge-back/paper intake remains HUMAN-run — the soft gate must not be combined with automated
  intake until the wide book + early-kill land.

**Slice 3 — flow + paper at scale (CODEOWNERS adjacency; human-merged).** Fully designed
2026-08-11 after slice 2 merged; concrete plan below.

### Ground truth this slice builds on (verified 2026-08-11)

- `paper merge-back` requires exactly `--strategy` / `--universe` / `--start` / `--end` per
  invocation — no branch-enumeration path. Nothing today derives these from a branch's committed
  files; only the agent's free-form `run-report.md` prose names them.
- **Landmine:** the driver commits `run-report.md` at the **repo root**. The merge-back diff
  policy allowlists only `algua/strategies/<family>/**.py` and `kb/**` — a root-level file makes
  `evaluate_diff` reject the ENTIRE branch (`diff_policy_rejected`) before any promote is
  attempted. This must be fixed as part of this slice, not discovered after deploy.
- The push is a real `git push origin <merge_sha>:refs/heads/main` compare-and-swap (verified by
  reading `gitops.py`) — not a local-only or simulated operation.
- `attempt_token` is purely deterministic (`sha256(strategy, branch_tip, merge_sha, ...)`) — an
  automated caller needs no special minting; re-invoking an already-terminal cycle is a safe no-op
  (`already_done`/`promoted_allocated`/etc. short-circuit before any git/registry mutation).
- `algua/operator/jobs.py` **already anticipated this gap** in its own comment: *"a static
  ExecStart cannot resolve WHICH candidate branch/strategy to merge back — that selection
  mechanism must land first."* Only `algua/cli/paper_cmd.py`, `store.py`, `transitions.py` are
  CODEOWNERS-protected among the touched files — `jobs.py` and `diff_policy.py` are not.
- No book-capacity constant exists — `--max-concurrent` defaults independently per command
  (intake=5, merge-back=5, allocate=8). No early-kill mechanism exists anywhere in the codebase.
- `paper run-all` already loops every PAPER/FORWARD_TESTED strategy per tick with per-strategy
  fault isolation — a natural pre-tick hook point for early-kill, no new command needed.
- Merge-back and `paper run-all`/`trade-tick` share **no lock today** — `operator.lock` only
  wraps jobs invoked through `algua operator run --job <name>`, and only `"paper"` is registered.

### Design — revised after Gate-1 (2 CRITICAL + 3 HIGH folded; scope narrowed)

(Also folds a Gate-1 LOW: `algua/operator/jobs.py`'s existing comment claiming merge-back "drops
in here as a new manifest entry + unit pair" is stale and actively misleading given the fix below
— it gets corrected in this slice so a future implementer doesn't reintroduce the session-gating
bug it would cause.)

Two Gate-1 CRITICALs reshaped this design significantly:
- `algua operator run --job X` is **session-gated** (`_run_session` in `operator_cmd.py` calls
  `session_gate` unconditionally — a once-per-XNYS-session marker). Reusing it for merge-back
  (which must fire repeatedly through the day) would silently cap it at once per session. Fix:
  **no `OPERATOR_JOBS` entry for merge-back at all.** Instead a new, purely additive command
  `algua operator lock-run -- <command...>` reuses the existing `operator_run_lock` primitive
  (already a generic non-blocking flock context manager) with **no session marker, no
  completion recording** — it only acquires `operator.lock`, execs the trailing command
  transparently (stdout/stderr/exit code passthrough), and treats lock contention as a benign
  no-op (matching the existing "paper" job's contention handling). Zero changes to the proven,
  live-paper-trading session-gate code path — this is the load-bearing safety property of this
  fix, not just a convenience.
- The queue has two writers (research driver enqueues, drainer updates status) with no shared
  lock — an atomic-file-replace alone prevents torn writes, not lost updates. Fix: a dedicated
  `data/mergeback-queue.lock` (flock, non-blocking with a short bounded retry — queue mutations
  are sub-second) wraps every read-modify-write on both sides.

Also folded: 1 item drained per 30 min cycle (not 5 — a single quality gate already runs ~9 min;
5/cycle would self-overlap its own 30 min window); early-kill is **deferred to a separate slice
(3b)** — the book is nearly empty today, so there is nothing to calibrate an eviction threshold
against yet, and splitting shrinks this PR's CODEOWNERS footprint to book-capacity defaults only.

**1. Fix the diff-policy landmine.** The driver commits the run report to `kb/research-runs/
<stamp>.md` instead of root `run-report.md` (`kb/**` is already allowlisted — zero changes to
`diff_policy.py`). Update the digest's `report` field and every doc/skill referencing the old
path.

**2. Extend the trailer schema — per-hypothesis verdict, not just a title.** The prompt already
asked for a `verdict` field per hypothesis (`"discarded"|"candidate-preview-pass"|"error"`); the
parser silently discarded everything but `title`. Fixed: each `hypotheses[]` entry now parses
`{title, verdict, merge_back: {strategy, universe, start, end} | null}`. `merge_back` is only
read (and only meaningful) when `verdict == "candidate-preview-pass"`.

**Schema/compatibility, made explicit (Gate-1 finding):** the digest's top-level `hypotheses`
field changes from `list[str]` (title only) to `list[{title, verdict, merge_back}]` — an additive
widening of the SAME field, not a new field. The existing `preview_gate` digest field (an
aggregate `{passed, failed_checks}` for whichever promote run the agent called "final") is KEPT
unchanged alongside it — it remains the cheap "did anything in this run look promising" signal
for the monitor/digest reader; `verdict` is the new PER-hypothesis, machine-actionable source of
truth for enqueueing. The anti-dup reader (slice 1) is updated to accept BOTH shapes when reading
historical digest lines: a bare string (old rows) contributes a title with no verdict/merge_back;
an object (new rows) contributes `title` the same way — anti-dup only ever needed titles, so this
is a trivial `isinstance` branch, and no historical digest line needs rewriting. Driver-side
validation
(untrusted agent output about to become CLI argv reaching a real push — same rigor as slice 1's
prompt-injection bound):
  - `branch` is NEVER taken from the trailer — the driver already knows its own `${BRANCH}`.
  - `strategy` must match `^[A-Za-z_][A-Za-z0-9_]*$` AND equal the module name of one of the
    files this run's own commit actually added under `algua/strategies/**` (cross-checked via
    `git show --name-only`, already computed for `n_strategy_files`) — an agent cannot nominate
    an unrelated strategy for merge-back.
  - **Uniqueness**: if two hypotheses in the same run claim the same `strategy` name, only the
    first is kept as a candidate; the rest are dropped (logged) — the queue must never hold two
    live candidates for one strategy name.
  - `universe` matches `^[A-Za-z0-9_-]+$` (semantic validity is `promote_task`'s own concern; it
    already fails closed on an unknown universe).
  - `start`/`end` match `^\d{4}-\d{2}-\d{2}$`, `start <= end <= today (UTC)`.
  - Any violation drops that hypothesis's merge-back candidacy silently (logged), never aborts
    the run or the digest write.

**3. A durable merge-back queue, decoupled from the research run's own timeout, with a real
shared lock.** Validated candidates are appended to `data/mergeback-queue.json` (atomic
tmp+fsync+replace, mirroring the push-subscription pattern from the monitor's
`web/backend/push.py`): one object per `(strategy, branch)` key — `{strategy, universe, start,
end, branch, enqueued_at, attempts, status, last_attempt_at, last_result}`. **Every
read-modify-write on this file — by the research driver (enqueue) or the drainer (status
update) — happens under a dedicated `data/mergeback-queue.lock` (non-blocking flock, short
bounded retry; queue mutations are sub-second, never held across a merge-back attempt itself).**
The research driver only enqueues; it never blocks on or runs merge-back itself, so queue depth
can never extend a research cycle past its 45 m budget.

**4. A drainer script + its own systemd unit pair.** `.codex/scripts/drain-mergeback-queue.sh`
(trusted, unsandboxed, no LLM involved — pure plumbing): every 30 min, processes **exactly 1**
`pending`/retry-eligible queue item (a single quality gate already runs ~9 min; draining more
per cycle risks self-overlap within the 30 min window). Invokes `algua operator lock-run --
algua paper merge-back --branch <b> --strategy <s> --universe <u> --start <a> --end <e>` (see 5
— NOT `operator run`), parses the JSON result, and updates the queue entry under the queue lock:
  - `promoted_allocated` / `promoted_queued` / `already_done` → terminal, kept in the file as a
    terminal record for audit/monitor display.
  - `diff_policy_rejected` / `promote_failed` → terminal-failure, NOT retried (the branch content
    itself is wrong; retrying wastes gate cycles for a guaranteed-identical outcome).
  - `gate_failed` → retryable, capped at `MAX_MERGEBACK_ATTEMPTS = 3` (tightened from the original
    5 — build-time check: if the merge-back JSON exposes WHICH quality-gate phase failed, two
    identical-phase failures in a row terminate early rather than burning the full cap; otherwise
    the flat cap of 3 is the mitigation).
  - Any hard fail-closed exception from the CLI (moved remote, drift) → leave `pending`, log
    loudly, let the next drain cycle retry (transient-environment, not branch-content).
  - Lock contention on `operator.lock` itself (the daily paper tick is running) → leave `pending`
    untouched, retry next cycle — never counts as an `attempts` increment.
New `deploy/systemd/algua-mergeback-drain.service`/`.timer` (30 min cadence), installed via the
existing `install-user-units.sh`.

**5. `algua operator lock-run -- <command...>`** (new, additive-only command in
`algua/cli/operator_cmd.py`, NOT CODEOWNERS-protected): resolves the same `operator.lock` path
`_run_session` uses, acquires it via the existing `operator_run_lock` context manager (already a
generic reusable primitive — no new locking code), execs the trailing command with transparent
stdout/stderr/exit-code passthrough, and treats `OperatorLockHeld` as a benign no-op exit
(matching the existing "paper" job's contention handling) rather than an error. **Deliberately
bypasses `session_gate`/`SessionMarker` entirely** — merge-back has no notion of "once per
trading session." This makes merge-back and the daily paper tick **share `operator.lock`**,
turning today's "operator discipline" into real kernel-enforced mutual exclusion, without adding
any risk to the existing, live, session-gated paper-trading code path (zero lines of
`_run_session`/`session_gate` touched).

**6. Wide paper book.** A single `ALGUA_PAPER_BOOK_CAPACITY` setting (`algua/config/settings.py`,
not CODEOWNERS; default 64, inside the spec's 50–100 range) becomes the shared default for
`--max-concurrent` on `paper intake`, `paper merge-back`, and `paper allocate` (the three
independently-hardcoded 5/5/8 defaults in `paper_cmd.py` — CODEOWNERS, human-merged here). An
explicit `--max-concurrent` still overrides per-invocation.

**7. Early-kill is DEFERRED to slice 3b (not built here).** Gate-1's rollout-split
recommendation, adopted: the paper book is nearly empty today (one authoritative strategy), so
there is nothing real to calibrate an eviction threshold against, and deferring it narrows this
slice's CODEOWNERS footprint to the book-capacity defaults only. When 3b is designed, the target
transition is `Stage.DORMANT` (not `RETIRED`) — a deliberate departure from the spec's literal
"auto-retire" wording: `DORMANT` is reversible (CLAUDE.md: entered only from live/paper, recovers
via `dormant -> paper`) and the codebase already ships a purpose-built re-audition tool for
exactly this (`research dormant-sweep`); `PAPER -> DORMANT` is confirmed already a free edge for
AGENT/SYSTEM with allocation auto-revoked via `_REVOKE_ON_EXIT` (verified in `transitions.py` —
no `transitions.py` change needed for 3b). Sketch for 3b: hook into `paper run-all`'s per-strategy
loop, PAPER-stage only (never `FORWARD_TESTED` — that stage has no `-> DORMANT` edge), two
triggers (statistical futility at n≥30 obs below a Sharpe floor; a standing kill-switch trip past
a grace window, reusing the existing breach signal rather than recomputing drawdown, with an
explicit check that no human resume/reset happened since the trip) → flatten → transition with a
reason.

### Tests (blast radius, CODEOWNERS files touched: `paper_cmd.py` book-capacity defaults only)
Trailer per-hypothesis verdict + merge-back validation (strategy cross-check against the branch's
own commit, same-strategy-twice-in-one-run dedup, format rejects, silent-drop-not-abort); queue
read-modify-write under the queue lock (concurrent-writer safety — simulate driver+drainer racing
the same file); drainer: exactly 1 item/cycle, retry/cap/terminal-vs-retryable classification per
merge-back status, lock-contention-is-not-an-attempt; `algua operator lock-run`: acquires/releases
`operator.lock`, passes through exit code and stdout verbatim, contention is a benign no-op exit
(not an error), and — the load-bearing regression test — **the existing `"paper"` job's
session-gate behavior is byte-identical after this change** (zero lines of `_run_session`/
`session_gate` touched, verified by re-running its existing test suite unmodified); book-capacity
setting threads through all three commands + explicit `--max-concurrent` override still wins.
`kb/research-runs/` path migration: digest `report` field + any doc/skill referencing root
`run-report.md` updated; existing digest lines from before the migration are NOT rewritten
(historical record).

## Risks, named

- **Codex $20-plan rate limits at 12 runs/day** — expected; rate-limit failures are themselves
  digest feedback; runner-count scaling waits for more subscriptions.
- **Paper multiplexing at width 100 is untested** — deliberate stress target; the reconcile
  machinery fails closed.
- **Weak-gate flooding self-punishes at the forward gate** (0.5×holdout term) — accepted; the
  early-kill rule bounds the calendar cost of flooded slots.
- **First true alpha verdicts are calendar-bound** (~3 months of paper evidence); machinery
  feedback is immediate, alpha feedback is not — nothing can compress this.
- **Auto merge-back writes to main autonomously** — bounded by #485's diff policy + quality gate
  + revert; still the largest single expansion of agent authority in the project; operator veto
  point before slice 3.
