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
  statistical stack computes + records as advisory (`*_binding=false` semantics); FDR ledger rows
  still written; throttle non-binding on this path. Human paths unchanged. Tests pin: floor
  failures still fail closed; statistical failures no longer veto; advisory fields still recorded;
  forward-bar parameterization (0.5×holdout) unchanged.

**Slice 3 — flow + paper at scale (CODEOWNERS adjacency; human-merged):**
- Driver auto-invokes `paper merge-back` on preview PASS (decision 6).
- Paper book capacity config (~50–100 slices) + early-kill rule (decision 3) in the paper session
  engine; retire-with-reason transition; slot recycling; monitor already renders the fallout.

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
