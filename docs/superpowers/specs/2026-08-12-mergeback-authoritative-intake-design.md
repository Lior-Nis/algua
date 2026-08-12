# Merge-back authoritative intake — registration + evidence reproduction (design)

**Status:** approved (3-round adversarial design review, Codex panel; OpenCode unavailable —
degradation noted, fail-closed rules respected).
**Problem discovered:** 2026-08-12, first end-to-end exercise of the zero-human factory path.

## Problem

The factory research loop (#544–#547) explores on a per-run SCRATCH registry inside an isolated
worktree; the producer is forbidden from writing the authoritative DB. Survivors are enqueued to
`data/mergeback-queue.json` and the trusted drainer runs `paper merge-back` (the #485 saga) against
the authoritative repo. Three compounding gaps mean **no factory survivor can ever land**:

1. **No authoritative registry row.** The saga dispatch reads `stage_of(strategy)` from the
   authoritative registry and its fresh path requires stage `backtested`
   (`algua/operator/mergeback.py:344`); a factory survivor has no row at all →
   `{ok:false, code:"not_found"}` before the journal/diff-policy even run.
2. **No authoritative promote evidence.** Authoritative `research promote` hard-fails without
   recorded search breadth (`promotion.py:428` — `search_trials` from an authoritative
   `backtest sweep`), and family classification reads `backtest_returns` (`promotion.py:178`);
   both exist only in the scratch DB. The queue payload (strategy/universe/start/end/branch)
   carries no evidence.
3. **No data context at the promote seam.** `promote_task` defaults `demo=False, snapshot=None`
   and `select_provider` raises without one (`_common.py:266`) — the merge-back promote seam
   (`paper_cmd.py:639`) passes neither, so the slice-3 merge-back could never promote **even a
   registered strategy**. (It predates the factory and was never exercised end-to-end.)

## Design principles

- **Scratch evidence is NEVER imported.** A sandboxed agent could under-report breadth (the exact
  gaming axis breadth-deflation defends). The queue carries a validated **recipe** (data-context
  IDs + sweep grid); the **trusted drainer reproduces all evidence authoritatively** post-merge.
- Under the market-first soft gate (#546) breadth's Sharpe-bar effect is advisory (the forward
  gate is the harsh wall), so a declared-grid re-run is integrity-honest. Residual: the sandbox
  may have searched more than it declares — bounded under the soft gate; recorded as a policy
  remark, not defended here.
- The producer copies the authoritative data dir into scratch, so scratch snapshot IDs are valid
  authoritative snapshot IDs — transporting IDs (never bars) suffices.
- All new logic lives in unprotected modules; the CODEOWNERS-protected `paper_cmd.py` diff stays
  minimal wiring.

## A. Trailer / queue payload (producer + `mergeback_queue.py`)

Each `merge_back` candidate now carries an `eval_context` object, validated **fail-closed at
enqueue** (invalid → candidate rejected with a loud warning; never a malformed queue item):

- `snapshot: <id>` XOR `demo: true`; optional `fundamentals_snapshot`, `news_snapshot`,
  `delistings`.
- `sweep_grid: {param: [values...]}` — the exact grid the scratch preview swept. Static shape
  validation at enqueue (JSON-canonical: parsed values, sorted keys, no NaN; combos ≤ the engine
  cap `_MAX_COMBOS = 200`; `construction.<key>` names permitted). Key-vs-strategy validation
  happens drainer-side post-merge (the module isn't on main at enqueue time).
- `rank_by`: allowlisted against the sweep engine's `_RANK_KEYS` (`mean_sharpe|min_sharpe`) at
  enqueue.
- `windows`/`holdout_frac`/thresholds/relaxations are **NOT transported**: the recipe pins
  `promote_task` strict-agent defaults. The producer REJECTS (refuses to enqueue) a candidate
  whose scratch preview deviated from those defaults — the authoritative run must evaluate the
  same partition the preview claimed.

## B. Drainer → CLI transport

`drain-mergeback-queue.sh` passes the context through as argv (array-built, as today) to new
`paper merge-back` options: `--snapshot | --demo`, `--fundamentals-snapshot`, `--news-snapshot`,
`--delistings`, `--rank-by`, repeatable `--sweep-param KEY=v1,v2` (parsed with the same
`parse_grid` the sweep CLI uses).

## C. Saga chokepoint (in `run_merge_back`, `algua/operator/mergeback.py`)

Order, on BOTH the fresh path and the have_merge crash-resume path, after the gate-green merge is
committed (module durably on local main), immediately before promote — and NEVER on
`diff_policy_rejected` / `gate_failed` / `already_done` / revert-completion / intake-resume paths:

1. **`ensure_backtested()`** — new unprotected module `algua/registry/mergeback_intake.py`.
   ONE `BEGIN IMMEDIATE` transaction: create-if-absent + CAS transition to `backtested`, single
   commit. Actor `AGENT`; reason + tags bind provenance (`mergeback:intake`, branch, branch_tip,
   merge_sha, base_sha) so a promote-failed orphan row is classifiable by gc/clustering/
   dashboards. Returns `created | existed` (decided inside the same tx — race-free input to the
   skip predicate). Idempotent: row at `backtested` → no-op; at `idea` → transition only; any
   other stage → fail closed.
2. **`produce_evidence()`** — SKIP when the row pre-existed this attempt (`existed`) AND
   `SqliteStrategyRepository.total_search_combos(name) > 0` (the direct-authoritative-funnel
   no-op predicate — a #534-style strategy with fresh authoritative breadth must not be re-swept
   or double-counted). Else, against the transported pinned context:
   - authoritative `backtest sweep` (true measured breadth),
   - a full-period `backtest run` persisting `backtest_returns` (the classifier's
     return-correlation axis), with the same delistings semantics
     (`assume_terminal_last_close=False`) and the **agent cost floor asserted before persisting**
     (`assert_gated_costs`-equivalent — classifier returns must be the same cost-realistic stream
     promote evaluates),
   - grid keys validated against the now-on-main strategy module (shared
     `validate_sweep_grid(strategy, grid)` API extracted in `sweep.py` — validation today is
     split across `parse_grid`/`_combos`/`_override`; the shared API unifies it).
   **Idempotency:** prefer recording trial row + returns + a `mergeback_evidence` marker
   (unique on strategy_id + branch_tip) in ONE transaction at the end of the compute (if
   `record_search_trial` is a single-row-per-sweep insert this is exact: mid-crash → nothing
   recorded → clean re-run). If per-combo recording makes one-tx infeasible, fall back to
   marker-last + keyed dedup of trials on (strategy, branch_tip, grid_hash) — duplicate trials
   are NOT harmless (they permanently inflate funnel/window breadth and the agent-NOVEL lifetime
   seed for later siblings, distorting the founder tax) so some attempt-idempotency at the trial
   layer is REQUIRED either way. Journal mirrors `evidence_status` for observability only.
   A `produce_evidence` failure routes through the same revert machinery as a promote exception.
3. **`promote(attempt_token)`** — the seam now passes the transported context
   (snapshot/demo/fundamentals/news/delistings) into `promote_task`; everything else stays
   strict-agent defaults; no relaxation flags reachable (unchanged guarantee).

## D. `stage_of → str | None` + journal-proof corruption guards

`stage_of` catches `StrategyNotFound` → `None` (unregistered = canonical factory-fresh state).
A missing row is treated as fresh ONLY when the journal carries no promoted/allocated proof:
`rec.terminal ∈ {promoted_allocated, promoted_queued}`, `rec.intake_status == "allocated"`, or
`rec.promote_status == "passed"` with a missing row → new `JournalRegistryMismatchError`
(fail closed — journal replay must never manufacture success out of registry corruption).
Fresh guard becomes `stage not in (None, "backtested")` → error. The candidate/paper drift
cross-check is unchanged (`None` is not in that set).

## E. Timeouts

`TimeoutStartSec` 1200 → **3600** and `RESERVATION_STALE_SECONDS` 1800 → **4200**, raised
together (stale > timeout always; the flock + one-item-per-firing keep the 30-min timer safe — a
long-running attempt just makes the next firing a no-op selection). Rationale: the cycle now
stacks quality gate (~9 min observed) + ≤200-combo sweep + full-period backtest + promote
walk-forward/holdout.

## F. Protected-file diff (`paper_cmd.py`, CODEOWNERS — human merge)

Minimal wiring only: `stage_of` None-catch; wiring for `ensure_backtested` /
`produce_evidence` (bodies in unprotected modules); the new CLI options passed through; the
promote seam gains the context kwargs. Everything else lives in unprotected files
(`mergeback.py`, `mergeback_intake.py`, `sweep.py` validator, `mergeback_queue.py`, the two
`.codex/scripts`, `deploy/systemd`, `db.py` schema for the evidence marker if used).

## G. Tests

- Queue validation matrix (grid shape/caps/rank_by/context XOR — fail-closed).
- Intake helper: single-tx ensure; `created|existed`; idempotent re-run; fail-closed stages.
- Saga unit tests (fake callables): chokepoint ordering; skip predicate; ensure/produce never
  called on early-terminal paths; the None-vs-journal corruption matrix; produce_evidence
  failure → revert; crash-resume after evidence, before promote (journal has merge_sha, no gate
  row) re-drives promote without re-producing evidence.
- Evidence idempotency: crash mid-compute → nothing recorded; completed evidence → marker blocks
  re-record.
- CLI E2E: unregistered strategy + `--demo` context → full cycle registers, produces evidence,
  promotes. Direct-funnel no-op: pre-registered strategy with breadth → produce_evidence skipped.

## Accepted residuals (recorded)

- Promote-failed revert leaves a provenance-tagged `backtested` row whose module is off main —
  inert, classifiable, and reverting registry state is the dangerous direction (#485 philosophy).
- Declared-grid trust: the sandbox may search more than it declares; bounded under the soft gate
  (forward gate self-punishes), flagged as policy, not defended mechanically here.
