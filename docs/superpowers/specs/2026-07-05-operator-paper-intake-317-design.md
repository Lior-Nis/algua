# Operator paper intake — admit candidates into the shared paper book (#317)

**Status:** Draft (GATE-1 revision — closes the design TBDs in the #317 issue body)
**Date:** 2026-07-05
**Issue:** #317 (epic #318 — autonomous paper operator)
**Depends on:** #316 (`paper run-all`, MERGED), the shared `strategy_allocations` primitive
(`algua/registry/allocations.py`), the multi-tenant paper book (#313/#314/#316a).
**Epic design (source-of-truth):** `docs/superpowers/specs/2026-06-25-autonomous-paper-operator-design.md`
(landing under **PR #487**, docs-only — see §7 for why it is NOT a code blocker).

---

## 1. Problem

With multi-tenancy restored (#313/#314/#316), the autonomous operator can run a real concurrent
forward-test book. What is still missing is the **intake** step: nothing admits validated
`candidate` strategies into the paper book up to capacity. Per the epic §5 Group 1, this is the
`paper allocate` + auto-allocator pair; §6.1 step 5a is where it fires in the research cycle.

The issue body left four things "TBD in design"; this doc resolves them:
1. candidate **selection order** (FIFO by candidate time vs best holdout Sharpe);
2. the **headroom check** (Σ allocations < equity AND a max-concurrent cap);
3. **composition** with `paper allocate` (Slice-1) and `paper run-all` (#316);
4. **concurrency / idempotency** under the multi-tenant book — plus whether PR #487 must land first.

### 1.1 What actually exists today (verified against the branch)

- **`paper allocate` does NOT exist.** Only `live allocate` (`live_cmd.py:117`) is wired. Both lanes
  share ONE primitive — `algua/registry/allocations.py::allocate(conn, strategy_id, capital, actor,
  account_equity)` — which atomically enforces `Σ(active capital) ≤ account_equity` inside a
  `BEGIN IMMEDIATE`. The paper lane has no CLI wrapper over it yet. **#317 must build one.**
- **The paper tick already REQUIRES an allocation.** `_run_paper_strategy_tick` (`paper_cmd.py:377`)
  calls `active_allocation(conn, rec.id)` and `raise ValueError(f"{name} has no paper allocation")`
  when it is absent; sizing runs off that capital via `build_paper_sizing_snapshot(... allocation
  ...)`. `paper run-all` iterates **every** `stage IN ('paper','forward_tested')` strategy, so a
  paper-stage strategy without an allocation makes the whole cycle raise. **This makes "allocate
  before the strategy is at stage=paper" a load-bearing ordering invariant (§5).**
- **`candidate→paper` is a legal, non-token-gated transition** (`lifecycle.py:29`
  `Stage.CANDIDATE: {Stage.PAPER, ...}`). Only `backtested→candidate` (research promote) and
  `paper→forward_tested` (paper promote) mint gate tokens; the audition step `candidate→paper` is a
  plain `registry transition --actor agent`. So intake needs no token — the multiple-testing
  defenses were already paid at the `candidate` gate.

---

## 2. Goal

Deterministic paper intake (Model A — no LLM in the order path): while the book has headroom
(both **Σ paper allocations + slice ≤ paper-account equity** AND **active paper-lane count <
max-concurrent cap**), admit the next queued `candidate` by **allocating it a slice first, then
transitioning `candidate→paper`**. When the book is full, leave the candidate queued (no
transition, no allocation). Idempotent and convergent: re-running produces the same book.

### Non-goals (deferred, off #317's minimal path)
- The `is_due` / session scheduler and the systemd timers (epic Group 4) — separate slices.
- The `family-audit` quarantine guard (epic §6.5) — intake exposes a filter **seam** for it (§4.4)
  but does not build the guard.
- Lane-tagging the shared allocations table for a mixed live+paper book (§6.1) — YAGNI under the
  paper-only operator; filed as a follow-up.
- Per-strategy slice sizing / risk-parity weighting — fixed slice only (epic §7).

---

## 3. Decision 1 — selection order: **FIFO by candidate-entry time** (NOT best holdout Sharpe)

**Decision: admit in FIFO order of the `candidate`-stage arrival timestamp** — the
`stage_transitions` row that moved the strategy into its **current** `candidate` episode (the most
recent `*→candidate`, NOT the historical earliest ever, so a strategy that bounced
`candidate→backtested→candidate` queues by its latest entry) — tie-broken by ascending strategy id
for a total, stable order. **Explicitly NOT ranked by holdout Sharpe.**

**Why FIFO, not Sharpe:**
- The holdout Sharpe is a **single-burn, breadth-deflated, pass/fail gate statistic** (#192/#137/#211),
  not a fine-grained ranking signal. Re-using it to *order* intake re-introduces exactly the
  selection-on-a-noisy-estimate pressure the DSR/LORD++ machinery exists to defeat — a marginally
  higher in-sample-lucky Sharpe would perpetually jump the queue. Every candidate already **cleared**
  the quality bar; among survivors the estimate's ranking information is mostly noise.
- FIFO is **non-gameable and diversity-preserving**: it can't be exploited by tuning a strategy to a
  fractionally higher holdout Sharpe to leapfrog older validated candidates, and it keeps the
  forward-test book from concentrating on one lucky-Sharpe cluster.
- FIFO is **deterministic and auditable** — a hard requirement for Model A (the order path must be
  reproducible from the DB, no judgment call).
- Fairness: a candidate that keeps missing the cut because the book is full is admitted the moment a
  slot frees, in arrival order — no starvation.

**Rejected alternative (best-Sharpe-first):** only defensible if slots were so scarce that "admit
the single best" dominated fairness. They are not (cap 8, epic §7), and the argument above says the
ranking signal is untrustworthy post-gate. Recorded here so it is not re-litigated.

---

## 4. Decision 2 — the headroom check: two independent bounds, both must hold

A candidate is admitted iff **both** bounds have room for one slice:

1. **Capital bound (HARD, atomic):** `Σ(active paper allocations) + slice ≤ paper_account_equity`.
   This is enforced by the existing `allocations.allocate` inside its `BEGIN IMMEDIATE` (it re-reads
   `total_allocated` under the write lock and raises `AllocationError` if the sum would exceed
   equity). Intake does **not** re-implement the check — it calls the primitive and treats an
   `AllocationError` as "no capital headroom, stop."
2. **Count bound (soft cap, operator-serialized):** `active_paper_lane_count < MAX_CONCURRENT`
   (default **8**, epic §7). `active_paper_lane_count` = strategies at `stage ∈ {paper,
   forward_tested}` holding a non-revoked allocation (a `forward_tested` strategy still occupies a
   slot and trades in `run-all`, so it counts).

Both are checked **per candidate** in the admit loop; the first bound to bind stops intake for this
run. With the epic defaults (8 × $10k = $80k ≤ ~$100k equity) the **count cap binds first** — that
is intended (it leaves capital headroom as a buffer).

**Slice size:** fixed **$10k** default (epic §7), a `--slice` CLI option / operator config, never
code. **Overflow:** a candidate that fits neither bound stays `candidate` (queued) — no allocation
row, no transition.

### 4.1 Why "< equity" AND a count cap (not just Σ ≤ equity)
Σ ≤ equity alone would admit ~10 strategies at $10k against $100k — but every extra tenant dilutes
the shared `run-all` buying-power pool (#316) and multiplies reconcile/tick surface. The count cap
is the operator's concurrency governor independent of capital; capital is the hard financial
backstop. They are orthogonal and both wanted.

### 4.2 Where equity comes from
`paper_account_equity = broker.account().equity` (the same source `paper account` reads,
`paper_cmd.py:365`), via a new `_paper_account_equity()` helper mirroring live's
`_live_account_equity()`. Read once at the top of an intake run and passed into each `allocate`.

### 4.3 Interaction with the shared allocations table
`allocations.total_allocated` sums **all** non-revoked rows regardless of lane. Under the
autonomous **paper-only** operator there are no live allocations, so `Σ = paper Σ` and the equity
denominator is the paper account — correct. The latent mixed-lane hazard (a live allocation counted
against paper equity, or vice-versa) is called out in §6.1 as a filed follow-up, not built here.

### 4.4 Family-audit quarantine seam (not built here)
Epic §6.5 will let a `family-audit` guard flag a candidate for breadth-evasion; intake must not
admit a flagged candidate. That flag/table does not exist yet, so intake's candidate selection is
structured as `select_admissible_candidates()` with an explicit "unflagged" filter that today is a
pass-through — the single call-site where the §6.5 guard drops in later.

---

## 5. Decision 3 — composition with `paper allocate` (Slice-1) and `paper run-all` (#316)

### 5.1 `paper allocate` is built HERE (Slice-1 never landed)
The issue says intake "reuses Slice-1 `paper allocate`," but Slice-1 shipped only in PR #288, which
is being **closed** (see PR #487) — so `paper allocate` is absent from main. #317 therefore builds
it as the thin mirror of `live allocate`:

```
paper allocate <name> --capital $X
  → rec = repo.get(name); reject stage==DORMANT (mirror live_cmd.py:129)
  → allocations.allocate(conn, rec.id, capital, actor="agent",
                          account_equity=_paper_account_equity())
```

Intake and manual `paper allocate` therefore share **one** capital primitive; there is no second
Σ-check to drift. `paper allocate` is the human/manual single-strategy path; **intake is the
automated batch** that loops candidates and calls the same primitive plus the transition.

**Resize semantics (explicit):** because `allocations.allocate` revokes-and-reinserts, a `paper
allocate` on a strategy that already holds an active allocation (an existing `paper`/`forward_tested`
tenant) silently **resizes** it. That is the intended manual-adjust path, but it is a book-changing
action — so `paper allocate` states this in its help text and emits the prior→new capital in its JSON
result. Intake itself only ever allocates candidates being freshly admitted (never resizes a live
tenant), so this only matters on the manual command.

### 5.2 The admit sequence per candidate — **allocate FIRST, then transition** (load-bearing order)
```
for cand in select_admissible_candidates()  # FIFO, §3
    if count >= MAX_CONCURRENT: break        # count bound, §4
    try:
        allocations.allocate(conn, cand.id, slice, actor="agent", account_equity=equity)  # capital bound
    except AllocationError:
        break                                # no capital headroom → stop, leave queued
    transition(cand, CANDIDATE → PAPER, actor="agent", reason="operator paper intake")
    count += 1
```
**Allocate strictly precedes the stage change.** This mirrors the epic §6.2 "merge precedes promote"
discipline and is required because `paper run-all` raises on any `stage=paper` strategy that lacks an
allocation (§1.1). The two ordering outcomes under a crash:
- crash **after allocate, before transition** → a still-`candidate` strategy holds an allocation row.
  `run-all` never ticks candidates, so this is invisible/harmless; the next intake run re-selects it,
  `allocate` idempotently revokes+re-inserts, and completes the transition. **No `run-all` breakage.**
- the reverse order (transition then allocate) is **rejected**: a crash between would leave a
  `stage=paper` strategy with no allocation, and the very next `run-all` would raise `ValueError` and
  abort the entire multi-tenant cycle. So the order is not a preference — it is the safety invariant.

### 5.3 Composition with `paper run-all` (#316)
Intake and `run-all` are **decoupled through the registry+allocations state**, not called inline:
- Intake **produces** book membership (stage=paper + an allocation). `run-all` **consumes** it, reads
  `active_allocation` fresh each cycle, sizes off it (#314), and pools buys (#316).
- A newly admitted strategy starts **flat** (no fills), so its first `run-all`: NAV snapshot =
  allocation, reconcile unaffected (nothing to attribute), and it simply joins the shared BP pool.
- **Ordering in the operator:** intake must run **before** the `run-all` that first ticks the new
  tenant (epic §4/§6.1 places intake in the research/merge-back job; `run-all` in the post-close paper
  runtime — naturally intake-then-tick across jobs). Within one job, run intake, then run-all.

---

## 6. Decision 4 — concurrency & idempotency under the multi-tenant book

### 6.1 Concurrency
- **Capital bound is race-proof.** `allocations.allocate`'s `BEGIN IMMEDIATE` serializes the
  read-check-write, so even two racing intake processes can **never** over-commit capital — the
  second sees the first's row under the write lock and raises. Capital can't be breached.
- **Count bound is a soft cap under operator serialization.** The `count < MAX_CONCURRENT` read sits
  outside that txn, so two concurrent intake runs could each read count=7 and both admit → 9.
  Mitigation: intake runs inside the **same single-operator discipline / research-cycle file lock**
  that already serializes the paper job and `run-all` (epic §6.4; #316's operator-discipline note).
  There is exactly one operator, so concurrent intake is not a real path. Worst case under a
  hypothetical race: the count cap is transiently exceeded by a bounded amount, **still floored by
  the hard capital bound**, and self-corrects on the next run (no admission happens once Σ hits
  equity). The count cap is therefore explicitly a **governor, not a safety invariant**; capital is
  the invariant.
- **Mixed-lane allocation hazard (filed, not built):** `total_allocated` is lane-agnostic. If a live
  book ever coexists with the paper operator, the paper Σ-check would wrongly include live capital
  (and vice-versa). Under the paper-only autonomous operator this cannot occur. Fix (a `lane` column
  on `strategy_allocations` + lane-scoped `total_allocated`) is a **follow-up**, deferred — it is a
  schema bump and live is human-gated / out of the autonomous path.

### 6.2 Idempotency
Intake is **convergent, not additive** — unlike `run-all`'s trade path it needs no session-idempotency
guard:
- A strategy admitted once is now `stage=paper`, so it is no longer a `candidate` and cannot be
  re-selected. Re-running intake with no freed slot is a **no-op**.
- `allocate` is itself idempotent (revoke-active + re-insert), so a re-run mid-partial-intake simply
  re-converges the same allocation.
- Slots free only when a tenant leaves the book (`paper→dormant` / `→retired` revokes its allocation,
  or a human puts it `live`). The next intake run then admits the FIFO-next candidate.

### 6.3 Interaction with the multi-tenant book primitives
A fresh tenant is flat, so it perturbs nothing in reconcile (#313) or the NAV snapshot (#314); it
only widens the `run-all` buying-power pool (#316), which already caps aggregate buys. No new
coupling.

---

## 7. Does PR #487 need to land first?

**No — #487 is not a code blocker for #317.** PR #487 is **docs-only**: it salvages the epic design
doc + slice-1 plan onto main. #317's code depends on `allocations.py`, `paper_cmd.py`,
`lifecycle.py`, and the registry — **none** of which #487 touches. #317's own design doc (this file)
is self-contained and cross-links the epic.

Recommendation: **merge #487 first for doc-graph cleanliness** (so this spec's "source-of-truth"
link resolves on main), but it is a soft ordering preference, not a dependency. If #487 is still
open when #317 implements, this doc cites it as "pending PR #487" and nothing is blocked.

---

## 8. Surface & files to change

- **[NEW] `paper allocate <name> --capital $X`** — `algua/cli/paper_cmd.py`. Mirror of `live
  allocate`: dormant-reject, `allocations.allocate(actor="agent", account_equity=_paper_account_equity())`.
- **[NEW] `paper intake [--slice 10000] [--max-concurrent 8]`** — `algua/cli/paper_cmd.py`. The
  deterministic auto-allocator: FIFO candidate selection (§3), the two-bound headroom loop (§4),
  allocate-then-transition per admission (§5.2), emit JSON `{admitted:[...], queued:[...],
  equity, slots_used, slots_cap}`.
- **[NEW] `_paper_account_equity()`** — `algua/cli/paper_cmd.py`, mirrors `_live_account_equity()`.
- **[NEW] candidate FIFO selector** — a small query over `strategies` + `stage_transitions`
  ordering current-candidate strategies by their **current** candidate-episode entry ts (the most
  recent `*→candidate`, §3), with the §4.4 unflagged pass-through seam. Lives in `algua/registry/`
  (a read helper), not in the CLI.
- **Tests** — `tests/test_cli_paper.py` / a new `tests/test_paper_intake.py`: FIFO order; both
  headroom bounds (count-binds and capital-binds cases); allocate-before-transition crash-safety
  (transition never precedes allocate); idempotent re-run is a no-op; a full book queues the rest;
  `paper allocate` dormant-reject + Σ≤equity; and an intake→`run-all` composition test (admitted
  tenant ticks; a paper strategy is never left allocation-less).

**CODEOWNERS check:** `paper_cmd.py`, `allocations.py` (NOT listed), tests, and this doc are **not**
CODEOWNERS-protected. `lifecycle.py` and `registry/transitions.py`/`promotion.py` **are** — so
intake must drive the `candidate→paper` step through the **existing** `registry transition` code path
(no edit to the protected transition machinery), keeping the PR mergeable without human review. **No
schema bump** (reuses `strategy_allocations`; the mixed-lane `lane` column is the deferred follow-up).

---

## 9. Deferred follow-ups (filed, out of scope)
- **Lane-tagged allocations** — a `lane` column on `strategy_allocations` + lane-scoped
  `total_allocated`, so a mixed live+paper book can't cross-count Σ against the wrong equity (§6.1).
  Schema bump; only needed if the operator ever runs live and paper books concurrently.
- **`family-audit` quarantine guard** (epic §6.5) — the §4.4 seam's real filter: block intake of a
  flagged candidate, bench a flagged paper tenant `paper→dormant`.
- **`is_due` scheduler + systemd timers** (epic Group 4) — the always-on driver that periodically
  invokes intake + `run-all`.
- **Weighted / risk-parity slice sizing** — replace the fixed slice with per-strategy sizing.
