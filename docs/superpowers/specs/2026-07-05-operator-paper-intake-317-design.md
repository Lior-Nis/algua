# Operator paper intake — admit candidates into the shared paper book (#317)

**Status:** Design complete; **PARTIALLY IMPLEMENTED — Slice-1 only** (see the banner below). The
sections that follow describe the FULL intended design; §§3, 4.1–4.2, 5.2, 5.3, 5.5 are what this PR
actually ships, while §4.3, §5.1, and §5.4 (the exit-side revoke wiring, `paper allocate`,
`live allocate` stage-gating, and the shared `allocate_in_lane` primitive) are **DESIGNED BUT NOT
YET BUILT** and are tracked as follow-up **#497**. Do not read this doc as evidence the end-to-end
allocation invariant is enforced today — it is not (see the leak note below).

> ### ⚠️ Implementation status — what this PR actually ships vs. #497
>
> **SHIPPED in this PR (#317):**
> - The atomic `intake_candidate_to_paper` primitive — `candidate→paper` admission that, under ONE
>   `BEGIN IMMEDIATE`, re-checks the max-concurrent count cap, capital-checks + allocates an equal
>   cents-floored slice (Σ ≤ paper-account equity, via the shared commit-less `allocate_locked`), and
>   CASes `candidate→paper`, committing/rolling back together (§5.2). This closes finding #3.
> - The `paper intake` CLI over that primitive (FIFO order by `stage_transitions.id`; `skipped_stale`
>   on a raced-out selection).
> - The `paper run-all` `skipped_unallocated` defensive skip: a trading-stage strategy with no active
>   allocation is skipped, never ticked, so it can't `raise "no paper allocation"` and abort the
>   whole cycle (§5.5). This is a resilience guard — it is NOT the exit-side revoke.
> - Supporting `allocations` helpers: `active_paper_lane_count`, `allocate_locked`, `CountCapReached`.
>
> **NOT shipped — deferred to #497 (the exit-side half of the invariant):**
> - Book-exit / lane-crossing revoke wiring in `transitions.py` (`paper→dormant`, `paper→retired`,
>   `paper→candidate`, `forward_tested→retired`, `live→paper`, `live→retired`) — §5.4.
> - `paper allocate` CLI, `live allocate` stage-gating to `stage==LIVE` (with in-lock TOCTOU
>   re-assert), the shared `allocate_in_lane` primitive, deletion of the stage-blind `allocate()`,
>   the go-live paper-slice shed, and generalizing `_assert_flat_for_bench` to the source stage — §4.3/§5.1.
> - The Section-8 test coverage for all of the above (book-exit revoke, go-live shed, `live→paper`
>   mirror, stage-gate rejection). None of these tests exist in this PR.
>
> **KNOWN LIMITATION (capital leak) until #497 lands:** because no book-exit revoke is wired,
> benching / retiring / back-stepping a paper-lane strategy (`paper→dormant`, `paper→retired`,
> `paper→candidate`, `forward_tested→retired`) leaves its `strategy_allocations` row active
> (`revoked_ts IS NULL`). Its slice's headroom is **permanently consumed** against `total_allocated()`
> until #497, so intake will eventually starve. Operators should treat `total_allocated()` as
> potentially over-counting departed tenants and, until #497, manually revoke on book exit if needed.
>
> **The end-to-end invariant below (an active allocation ⟹ trading stage, closed at BOTH ends) is the
> DESIGN TARGET of #497 — it does NOT hold as shipped in #317.**

The unifying invariant is a **one-way** implication (NOT an iff — Codex round-4), intended to be
closed at BOTH ends:
**an active allocation ⟹ the strategy is at `stage ∈ {paper, forward_tested, live}`** — maintained at the
*entry* points (every allocator refuses to create a row outside those stages: `live allocate` → `LIVE`
only, `paper allocate` → `{paper, forward_tested}` only, `intake` → `candidate→paper` only, §5.1) and at
the *exit* points (an allocation is revoked atomically with the stage CAS on every transition that leaves
those three stages or crosses lanes, §5.4). Of these, ONLY the `intake` entry-point and its count/capital
checks ship in #317; the `paper allocate` / `live allocate` entry gates and ALL exit revokes are #497.
The **converse does not hold**: a strategy can be at a
trading stage *without* an allocation — a recovery/demotion re-entrant (`dormant→paper`, `live→paper`),
or a strategy freshly at `live` after go-live shed its paper slice — until it is (re-)allocated. So an
**active book tenant** is defined as `stage ∈ lane AND has an active allocation`; an unallocated
trading-stage strategy is a not-yet-(re)admitted non-tenant, skipped by `run-all` (§5.5). The intake
side (this PR): the count cap is enforced INSIDE the atomic `intake_candidate_to_paper` transaction, so it is
race-proof under the `BEGIN IMMEDIATE` write lock, not merely operator-serialized; the atomic
`intake_candidate_to_paper` primitive closes finding #3; `run-all` skips (not raises on) unallocated
re-entrants (§5.5). FIFO orders by `stage_transitions` row **id** (not wall-clock ts); the intake loop
handles a stale-selection CAS failure (surfaced as `skipped_stale`); `run-all` checks the allocation
**before** loading the strategy module and reports `skipped_unallocated` in **every** envelope; the
allocation paths use `BaseException` rollback discipline.
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
  paper-stage strategy without an allocation makes the whole cycle **raise** today. #317's atomic admit
  (§5.2) keeps admitted tenants always-allocated, and once §5.4 revokes on book-exit the recovery/
  demotion re-entrants would trip this raise into a book-wide abort — so #317 relaxes it to a **skip**
  (§5.5): allocation == active book membership, and an unallocated (always-flat) paper strategy is simply
  not ticked.
- **`candidate→paper` is a legal, non-token-gated transition** (`lifecycle.py:29`
  `Stage.CANDIDATE: {Stage.PAPER, ...}`). Only `backtested→candidate` (research promote) and
  `paper→forward_tested` (paper promote) mint gate tokens; the audition step `candidate→paper` is a
  plain `registry transition --actor agent`. So intake needs no token — the multiple-testing
  defenses were already paid at the `candidate` gate.

---

## 2. Goal

Deterministic paper intake (Model A — no LLM in the order path): while the book has headroom
(both **Σ paper allocations + slice ≤ paper-account equity** AND **active paper-lane count <
max-concurrent cap**), admit the next queued `candidate` via **ONE atomic registry operation** that
re-reads the count, capital-checks, allocates a slice, and CASes `candidate→paper` in a single write
transaction (§5.2) — so there is no observable allocated-but-still-candidate intermediate state and
**both** bounds are enforced under the same `BEGIN IMMEDIATE` write lock (neither can be raced past;
§4/§6.1). When the book is full,
leave the candidate queued (no transition, no allocation). Idempotent and convergent: re-running
produces the same book. In the FULL design an allocation is **revoked atomically on any book-exit or
lane crossing** (§5.4) so slots genuinely free and neither lane can ever inherit the other's slice —
but that exit-revoke is **deferred to #497 and does NOT ship in #317** (see the top-of-doc banner); as
shipped, slots do not auto-free on book exit. The `run-all` skip of unallocated re-entrants (§5.5) does
ship.

### Non-goals (deferred, off #317's minimal path)
- The `is_due` / session scheduler and the systemd timers (epic Group 4) — separate slices.
- The `family-audit` quarantine guard (epic §6.5) — intake exposes a filter **seam** for it (§4.4)
  but does not build the guard.
- A `lane` **column** on the shared allocations table (row-level lane attribution) — deferred as a
  schema bump. The FULL design avoids needing it by revoking the allocation on the crossing edge (§5.4)
  to close both lane-crossing *sizing* leaks (a paper slice sized by live on go-live, and a live slice
  sized by paper on demotion) **without** a column — but that revoke is itself **deferred to #497**, so
  in #317 those sizing leaks are NOT yet closed. The column is only needed for a genuinely *concurrent*
  live+paper book (both lanes allocating against a shared Σ at once), which the paper-only operator never
  runs (§9).
- Per-strategy slice sizing / risk-parity weighting — fixed slice only (epic §7).

---

## 3. Decision 1 — selection order: **FIFO by candidate-entry time** (NOT best holdout Sharpe)

**Decision: admit in FIFO order of the `stage_transitions` row that moved the strategy into its
**current** `candidate` episode** — the row with the **greatest `id`** whose `to_stage='candidate'`
(the most recent `*→candidate`, NOT the historical earliest ever, so a strategy that bounced
`candidate→backtested→candidate` queues by its latest entry). Candidates are then ordered by that
entry-row's **`id` ascending** (older episode → admitted first), tie-broken by ascending strategy id
for a total, stable order. **Explicitly NOT ranked by holdout Sharpe.**

**Order by `stage_transitions.id`, NOT `created_at` (Codex round-3).** `stage_transitions` has a
monotonic autoincrement `id` (`db.py`), whereas `created_at` is a wall-clock string that can collide
at sub-second resolution or move backwards under a clock adjustment. Using the row `id` gives a true,
gap-free DB insertion order for both selecting the current episode (max id to `candidate`) and
ordering the queue (min such id first) — no ambiguity, no clock dependence. `created_at` is retained
only as a human-readable audit field, never as the sort key.

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
2. **Count bound (HARD, atomic — Codex round-3 HIGH-2):** `active_paper_lane_count < MAX_CONCURRENT`
   (default **8**, epic §7). `active_paper_lane_count` = strategies at `stage ∈ {paper,
   forward_tested}` holding a non-revoked allocation (a `forward_tested` strategy still occupies a
   slot and trades in `run-all`, so it counts). This count is **re-read INSIDE
   `intake_candidate_to_paper`'s `BEGIN IMMEDIATE` transaction** (§5.2), alongside the capital check —
   so it is serialized by the same write lock and is **race-proof**: two concurrent intake processes
   cannot both read `count=7` and both admit to 9, because the second's `BEGIN IMMEDIATE` blocks until
   the first commits and then re-reads `count=8`. The admit loop *also* keeps an outer running count as
   a cheap early-`break` (to avoid launching doomed transactions once the book is known full), but the
   **authoritative** enforcement is the in-transaction re-read; the outer read is only an optimization,
   never the safety boundary.

Both are checked **per candidate** — the outer loop breaks on either bound; the in-transaction re-read
is authoritative for both. The first bound to bind stops intake for this run. With the epic defaults
(8 × $10k = $80k ≤ ~$100k equity) the **count cap binds first** — that is intended (it leaves capital
headroom as a buffer).

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

### 4.3 Interaction with the shared allocations table + the go-live sizing gate (finding #2)

> **DEFERRED to #497 — NOT shipped in #317.** Everything in this subsection (the `live allocate`
> stage gate, the `forward_tested→live` paper-slice shed, the exit revokes) is design only. In the
> shipped code `total_allocated()` still over-counts departed paper tenants; see the leak note at top.

`allocations.total_allocated` sums **all** non-revoked rows regardless of lane. Under the autonomous
**paper-only** operator there are no live allocations, so `Σ = paper Σ` and the equity denominator is
the paper account — correct.

The one lane-crossing hazard that is actually **reachable from #317's own state** is a promotion path,
not a concurrent-book one: a strategy admitted here holds a **$10k paper** allocation row; it can climb
`paper→forward_tested→live` (the `forward_tested→live` edge is the human go-live). Because
`active_allocation` is lane-agnostic, `live run-all` sizing (`live_cmd.py:169`) and its liveness guard
`_still_live_allocated` (`live_cmd.py:154`) would otherwise read that **paper-intake** row and silently
size the live strategy off the $10k paper slice. #317 must not defer past this point.

**Entry-point gate (Codex round-3 HIGH-1): `live allocate` is narrowed to `stage == LIVE` only.**
The exit-side revoke below is necessary but not sufficient on its own: today `live allocate`
(`live_cmd.py:117`) only rejects `DORMANT`, so it can still write an allocation row onto a
`CANDIDATE`, `PAPER`, or `FORWARD_TESTED` strategy — re-manufacturing exactly the lane-agnostic row
that `live run-all` would then size off, from the *live* side. #317 gates `live allocate` to
`stage == LIVE` **only** (rejecting every other stage), the exact mirror of the new `paper allocate`
`{paper, forward_tested}` gate — and, like it, the gate is re-asserted **inside the write lock** via the
shared `allocate_in_lane` primitive (`allowed_stages={LIVE}`, §5.1), never on a stale pre-transaction
read (Codex round-5 TOCTOU). This is coherent with the go-live flow: a strategy reaches
`live` **unallocated** (the paper slice is shed on `forward_tested→live`, below), and the human then
`live allocate`s it while it is already at `stage == LIVE` — so no legitimate case is lost. With both
allocator entry points stage-gated (and `intake` only ever creating a `candidate→paper` row), the
positive invariant **"an active allocation ⟹ `stage ∈ {paper, forward_tested, live}`"** holds by
construction, which is what makes the §5.4 revoke table provably exhaustive (§5.4). `live_cmd.py` is
**not** CODEOWNERS-protected, so this gate ships in-scope.

**Gate (lane-free, no schema bump): revoke the paper allocation on the `forward_tested→live` exit.**
The go-live transition revokes the outgoing paper-lane allocation in the SAME atomic write that records
`live_authorization` and CASes to `live` (§5.4). The strategy therefore enters `live` **unallocated**;
`_still_live_allocated` returns `False`, so `live run-all` **skips** it (never sizes it) until a human
explicitly `live allocate`s a fresh, live-sized allocation. A paper slice can thus never be consumed by
the live sizing path — the row is gone before live ever reads it. (This is strictly *more* correct than
the status quo even ignoring lanes: a human going live should size the live book deliberately, not
inherit a forward-test slice.) The residual mixed-lane accounting hazard — a live book and a paper book
allocating *concurrently* against each other's equity — is genuinely out of reach for the paper-only
operator and remains the deferred `lane`-column follow-up (§6.1/§9).

### 4.4 Family-audit quarantine seam (not built here)
Epic §6.5 will let a `family-audit` guard flag a candidate for breadth-evasion; intake must not
admit a flagged candidate. That flag/table does not exist yet, so intake's candidate selection is
structured as `select_admissible_candidates()` with an explicit "unflagged" filter that today is a
pass-through — the single call-site where the §6.5 guard drops in later.

---

## 5. Decision 3 — composition with `paper allocate` (Slice-1) and `paper run-all` (#316)

### 5.1 `paper allocate` is built HERE (Slice-1 never landed)

> **DEFERRED to #497 — NOT shipped in #317.** `paper allocate`, `live allocate` stage-gating, and the
> shared `allocate_in_lane` primitive (with in-lock stage re-read + count check) are design only. The
> shipped `allocations` module exposes the commit-less `allocate_locked` (used by `intake`) and still
> retains the stage-blind public `allocate()`; `allocate_in_lane` and the `paper allocate` CLI do not
> exist yet. The intake path (§5.2) is the ONLY allocator that ships in #317.

The issue says intake "reuses Slice-1 `paper allocate`," but Slice-1 shipped only in PR #288, which
is being **closed** (see PR #487) — so `paper allocate` is absent from main. #317 builds it as a
close analog of `live allocate`, but with a **tighter, lane-scoped stage gate** (a naive
"reject DORMANT" mirror is unsafe — see below):

```
paper allocate <name> --capital $X [--max-concurrent 8]
  → rec = repo.get(name)                            # friendly early error only — NOT authoritative
  → equity = _paper_account_equity()                # broker read BEFORE the txn (no I/O under lock)
  → allocations.allocate_in_lane(                    # ONE BEGIN IMMEDIATE: in-lock stage + count + Σ
        conn, rec.id, capital, actor="agent", account_equity=equity,
        allowed_stages={PAPER, FORWARD_TESTED},      # re-read & asserted UNDER the write lock
        max_concurrent=<cap>)                        # count-check only when currently unallocated
```

**Stage gate (lane safety, not just `reject DORMANT`).** Because allocations are lane-agnostic,
`paper allocate` MUST refuse every stage outside the paper book `{PAPER, FORWARD_TESTED}`, not merely
`DORMANT`:
- on a `LIVE` strategy it would create/overwrite exactly the row `live run-all` sizes off —
  reintroducing the cross-lane sizing leak §5.4 exists to prevent (use `live allocate` for live);
- on a `CANDIDATE` (or any pre-book stage) it would manufacture the allocated-but-not-a-book-member
  state §5.2 calls structurally unreachable — capital counted in Σ but invisible to
  `active_paper_lane_count`, and a `stage=paper` promotion later would inherit a stale slice. Candidates
  are admitted ONLY via the atomic `paper intake` (§5.2), never `paper allocate`.
This same `{PAPER, FORWARD_TESTED}` gate makes `paper allocate` the correct **re-allocation path** for a
recovered/demoted strategy that re-entered the book unallocated (`dormant→paper`, `live→paper`; §5.5) —
it is at `stage=paper`, so it is eligible. `paper allocate` enforces the hard capital bound (Σ≤equity,
via the shared primitive) AND the hard count bound — see below.

**`paper allocate` enforces the count cap for a COUNT-INCREASING allocation (Codex round-4 HIGH — closes
the recovery re-entry bypass).** An earlier revision made `paper allocate` cap-*exempt* and argued that
could not inflate the book, because the stage gate blocks admitting a `candidate`. That argument was
**wrong**: it ignored recovery *after a freed slot is reused*. Concrete bypass: a full book of 8 → tenant
A benches `paper→dormant` (revoke frees a slot) → `intake` admits candidate B into the freed slot (back
to 8) → A recovers `dormant→paper` (lands **unallocated**, §5.4) → a cap-exempt `paper allocate A` would
create a **9th** active paper-lane allocation, breaking the hard count bound. So `paper allocate` is
**NOT** cap-exempt. It takes the same `--max-concurrent` (default 8) and, **when the target strategy is
currently unallocated** (i.e. the allocation would *increase* `active_paper_lane_count`), it re-reads the
count **inside its `BEGIN IMMEDIATE`** and raises `CountCapReached` if `≥ max_concurrent` — atomically and
race-proof, exactly as `intake` does (§5.2 step 2). A **resize** of an already-allocated tenant is exempt
by construction — it does not change the count (the tenant is already counted), so it is always allowed.
With this, the count bound `active_paper_lane_count ≤ MAX_CONCURRENT` is a genuine **hard invariant across
BOTH allocators** (`intake` and `paper allocate`), not a per-command governor. (There is deliberately no
"human may knowingly exceed" escape hatch on this path: exceeding the operator's concurrency cap is done
by raising `--max-concurrent`, an explicit config change, not by a silent per-allocation override.)

**The stage gate + count check must be re-asserted INSIDE the write lock, not read pre-transaction
(Codex round-4/5 HIGH — TOCTOU).** A CLI-level `rec = repo.get(name)` read of the stage *before* the
allocation transaction is **not** authoritative: a concurrent transition can commit between the read and
the allocation write. Concrete bypass: `live allocate L` reads `stage=live`; a concurrent `live→paper`
demotion commits (revoking the live slice); then the stage-blind allocation inserts a fresh row onto the
now-`paper` L with **no paper count check** — a 9th paper tenant. The symmetric hole exists for `paper
allocate` (read `stage=paper`, concurrent `paper→dormant`, allocation lands on a `dormant` strategy). So
**both** `paper allocate` and `live allocate` route through ONE shared transactional primitive,
`allocations.allocate_in_lane(conn, strategy_id, capital, actor, account_equity, allowed_stages,
max_concurrent=None)`, which under a single top-level `BEGIN IMMEDIATE` (top-level-only guard +
`BaseException` rollback, mirroring `apply_transition`):
1. **re-reads the strategy's CURRENT stage from `strategies` under the write lock** and asserts it is in
   `allowed_stages` (membership, not an exact-value CAS — any concurrent transition that moved it *out*
   of the allowed set is now visible and rejected; a move to another *in-set* stage, e.g.
   `paper→forward_tested`, is still valid);
2. if `max_concurrent is not None` **and** the strategy is currently unallocated
   (`active_allocation(conn, id) is None` — a count-*increasing* allocation), re-reads
   `active_paper_lane_count` under the lock and raises `CountCapReached` if `≥ max_concurrent`;
3. calls the shared commit-less `allocate_locked(...)` (the Σ-check body);
4. commits (or rolls back on any exception).

`paper allocate` calls it with `allowed_stages={PAPER, FORWARD_TESTED}, max_concurrent=<cap>`; `live
allocate` with `allowed_stages={LIVE}, max_concurrent=None` (the live lane has no #317 count cap). The
pre-transaction `repo.get()` in either CLI is kept ONLY for a friendly early error message — it is never
the authority. The **stage-blind public `allocate()` is removed** (its sole production caller was `live
allocate`): after this change there is no way to create an active allocation without the in-lock stage
+ cap gate (closing the round-5 IMPORTANT that `allocate` remained an unsafe public primitive).
`allocate_locked` stays commit-less and is only ever called under an owning transaction (`intake`,
`allocate_in_lane`). One stage-gate + count-check + Σ-check shape, shared by both allocators and intake;
no drift.

**Broker equity is read BEFORE the transaction opens (Codex round-5 IMPORTANT).** `account_equity` is
obtained via `_paper_account_equity()` / `_live_account_equity()` (a broker HTTP call) and passed *by
value* into `allocate_in_lane`, so SQLite's write lock is never held across network I/O — no avoidable
lock contention or failure coupling.

Intake and manual `paper allocate` therefore share **one** capital primitive; there is no second
Σ-check to drift. `paper allocate` is the human/manual single-strategy path; **intake is the
automated batch** that admits candidates via the atomic §5.2 primitive.

**Shared Σ-check (supports §5.2's atomicity):** `allocations.allocate` today opens its own
`BEGIN IMMEDIATE ... commit` AND is stage-blind, so it can neither be composed inside intake's single
transaction nor safely gate a lane. #317 restructures it into two pieces, both in `allocations.py`
(**not** CODEOWNERS-protected):
- a commit-less **`allocate_locked(conn, strategy_id, capital, actor, account_equity)`** — the Σ
  read-check-revoke-insert body, no `BEGIN`/`commit` — callable inside any owning transaction;
- the transactional, lane-scoped **`allocate_in_lane(...)`** wrapper (§5.1 above): `BEGIN IMMEDIATE` →
  in-lock stage-membership assert → conditional count check → `allocate_locked` → commit, with the
  top-level-only + `BaseException` rollback discipline `apply_transition` uses (`store.py:346`) so a
  `KeyboardInterrupt`/`SystemExit`/`GeneratorExit` mid-body cannot leave the write lock held with an
  un-rolled-back transaction.

`intake_candidate_to_paper` (§5.2) calls the SAME `allocate_locked` under its own transaction alongside
the stage CAS + count check; `paper allocate` and `live allocate` call `allocate_in_lane`. The old
stage-blind `allocate()` is deleted (no production caller survives). One Σ-check body reused everywhere —
no drift.

**Resize semantics (explicit):** because `allocations.allocate` revokes-and-reinserts, a `paper
allocate` on a strategy that already holds an active allocation (an existing `paper`/`forward_tested`
tenant) silently **resizes** it. That is the intended manual-adjust path, but it is a book-changing
action — so `paper allocate` states this in its help text and emits the prior→new capital in its JSON
result. Intake itself only ever allocates candidates being freshly admitted (never resizes a live
tenant), so this only matters on the manual command.

### 5.2 The admit step per candidate — **ONE atomic registry primitive** (finding #3)
The allocate and the stage change are **not** two separate registry calls. They are a single atomic
registry operation, `StrategyRepository.intake_candidate_to_paper(rec, capital, actor, account_equity,
max_concurrent)`, that under ONE top-level `BEGIN IMMEDIATE` does, in order:
1. re-assert `rec.stage == CANDIDATE` (a stage CAS — closes the same `candidate→retired`-during-intake
   drift TOCTOU that #246 closed for research-promote);
2. **the count cap re-read (Codex round-3 HIGH-2):** re-read `active_paper_lane_count` under the write
   lock and raise `CountCapReached` if `≥ max_concurrent` — so the count bound is serialized by the same
   `BEGIN IMMEDIATE` as the capital bound and cannot be raced past (§4/§6.1);
3. the capital cap-check + allocation insert (`Σ(active) + slice ≤ equity`), via a commit-less
   `allocations.allocate_locked(...)` — the exact same Σ-check body `allocate` uses, extracted so the
   two paths can never drift (see §5.1);
4. the `candidate→paper` stage row + `stage_transitions` audit row (`_apply_transition_locked`);
5. `commit` — or, on any exception, `rollback` (`BaseException` discipline, §5.1).

```
for cand in select_admissible_candidates()   # FIFO snapshot, §3
    if count >= MAX_CONCURRENT: break         # cheap outer early-break; authoritative check is in-txn
    try:
        repo.intake_candidate_to_paper(cand, slice, actor="agent",
                                       account_equity=equity, max_concurrent=MAX_CONCURRENT)
    except CountCapReached:
        break                                 # count bound (now IN-txn/authoritative) → stop
    except AllocationError:
        break                                 # capital bound → stop, leave the rest queued
    except TransitionError:
        continue                              # STALE SELECTION → skip this one, keep going (§ below)
    admitted.append(cand.name); count += 1
```

**Stale-selection CAS failure (Codex round-3 IMPORTANT).** `select_admissible_candidates()` produces a
FIFO *snapshot*; the per-candidate `BEGIN IMMEDIATE` runs later. If a concurrent operator (or a human)
moves that candidate out of `CANDIDATE` between the snapshot and the txn, step 1's stage CAS fails and
`_apply_transition_locked` raises `TransitionError` ("stage is no longer 'candidate'"). Intake treats
this as **"already handled elsewhere"**: it `continue`s to the next snapshot entry (it does NOT break —
capital and count are unaffected — and does NOT re-select; a stale entry is simply passed over) **and
records the name in a `skipped_stale: [names…]` field on the intake JSON** (Codex round-4 IMPORTANT — a
stale candidate is neither `admitted` nor reliably still `queued`, so it needs its own explicit bucket
rather than vanishing from the output contract). This is distinct from `CountCapReached`/`AllocationError`
(book genuinely full → `break`). Under the single-operator discipline (§6.1) this path is not normally
reachable, but it is specified so a race is an *observably* skipped candidate, never a crash, a wrong
admission, or a silent disappearance.

**Atomicity scope excludes the broker equity read (Codex round-3 MINOR).** `account_equity` is read once
via `_paper_account_equity()` (a broker HTTP call, §4.2) at the *top* of the intake run and passed by
value into each `intake_candidate_to_paper`. The `BEGIN IMMEDIATE` transaction therefore covers only the
registry/allocation DB writes (stage CAS, count re-read, Σ-check, allocation insert, audit row) — **not**
the external account snapshot, which is a fixed input for the run (identical to `allocate`'s existing
`account_equity` contract). This is deliberate and safe: equity is a slowly-moving denominator, and the
hard financial backstop (Σ ≤ equity) is evaluated against whatever value was read, atomically, per
admit.

**Why atomic (and why this removes the partial-intake state entirely).** The allocation insert and the
`candidate→paper` CAS commit together or not at all, so the failure state finding #3 flagged — an
"allocated-but-still-`candidate`" row — is **structurally unreachable**: a crash mid-transaction rolls
the `BEGIN IMMEDIATE` back, leaving the strategy exactly `candidate` with no allocation. There is no
partial state to repair, audit, or special-case in `paper intake`'s JSON. Atomicity also guarantees a
freshly *admitted* tenant is `stage=paper` **with** its allocation committed in the same write, so it is
an active book tenant on its very first `run-all` (never in the §5.5 unallocated-and-skipped state — that
state is reserved for recovery/demotion re-entrants, not admissions). The reverse ordering
(transition-then-allocate as two writes) is not merely rejected but structurally impossible here: there
is no separate transition step to crash between.

**Concurrency.** `intake_candidate_to_paper` reuses the `apply_transition(revoke_allocation=True)`
top-level-only discipline (`store.py:335` — raises if called inside an open transaction; blanket
rollback on failure), so it can never nest inside or corrupt a surrounding transaction. The capital
bound is race-proof for the same reason `allocate` is: the `BEGIN IMMEDIATE` write lock serializes the
Σ read-check-write across concurrent callers.

### 5.3 Composition with `paper run-all` (#316)
Intake and `run-all` are **decoupled through the registry+allocations state**, not called inline:
- Intake **produces** book membership (stage=paper + an allocation). `run-all` **consumes** it, reads
  `active_allocation` fresh each cycle, sizes off it (#314), and pools buys (#316).
- A newly admitted strategy starts **flat** (no fills), so its first `run-all`: NAV snapshot =
  allocation, reconcile unaffected (nothing to attribute), and it simply joins the shared BP pool.
- **Ordering in the operator:** intake must run **before** the `run-all` that first ticks the new
  tenant (epic §4/§6.1 places intake in the research/merge-back job; `run-all` in the post-close paper
  runtime — naturally intake-then-tick across jobs). Within one job, run intake, then run-all.

### 5.4 The allocation invariant — revoke on any lane change or book exit (findings #1 + #2)

> **DEFERRED to #497 — NOT shipped in #317.** The revoke table below is design only; in the shipped
> code only the pre-existing `LIVE→DORMANT` edge revokes. Every other book-exit / lane-crossing edge
> still leaves the allocation row live — this is the shipped capital leak (see the banner at top). The
> "provably exhaustive" / "invariant holds by construction" language below describes the #497 END
> STATE, not #317.

Intake's convergence claim (§6.2) and the go-live gate (§4.3) both rest on one invariant that the
current transition machinery does **not** yet enforce:

> **An allocation is revoked, atomically with its stage CAS, on any transition that (a) EXITS the paper
> book `{paper, forward_tested}` to a non-trading stage, OR (b) CROSSES between the live lane and the
> paper book.** Equivalently: an allocation survives a transition only when the strategy stays in the
> *same* trading book (`paper↔forward_tested` in-book, or a strategy already `live` staying `live`).

Today only `LIVE→DORMANT` sets `revoke_allocation=True` (`transitions.py:66`). Every other exit/lane-
change currently leaves the allocation row **live**, which breaks the invariant three ways: the slot
never frees (finding #1), the paper slice leaks into live sizing on go-live (finding #2), and a *live*
slice leaks into paper sizing on a demotion (the mirror leak Codex flagged). All of these must be wired
in-scope, mirroring the existing `LIVE→DORMANT` pattern — **do not assume any of it already exists.**

**Edges that revoke** (every one is either a book exit or a lane crossing):

| edge | kind | flat-check ledger | why |
|---|---|---|---|
| `paper→dormant` | book exit | PAPER | benched; frees the slot |
| `paper→retired` | book exit | PAPER | tombstoned; frees the slot |
| `paper→candidate` | book exit | PAPER | back-step for re-audition; not in book |
| `forward_tested→retired` | book exit | PAPER | tombstoned (a `forward_tested` tenant still holds a slot, §4) |
| `forward_tested→live` | lane crossing (paper→live) | PAPER | **go-live gate, finding #2** — sheds the paper slice |
| `live→paper` | lane crossing (live→paper) | LIVE | **mirror gate** — sheds the *live* slice so paper sizing can't consume it |
| `live→dormant` | book exit (existing) | LIVE | already wired (#125/#247) |
| `live→retired` | book exit | LIVE | **added round-3 (HIGH-4)** — a `live` strategy retired directly must shed its live slice, else a stray `revoked_ts IS NULL` row survives on a terminal stage |

In-book edges (`paper↔forward_tested`) keep the allocation. Any transition **into** the paper book from
outside it (`dormant→paper` recovery, `live→paper` demotion) therefore lands the strategy **unallocated**
— its prior slice was revoked on the way out / on the lane crossing. Such a strategy is not yet an active
book tenant; `run-all` skips it until it is (re-)allocated (§5.5).

**Why the table is EXHAUSTIVE (Codex round-3 HIGH-4 — re-verified against `_LIVE_TRANSITIONS`).** The
completeness proof rests on the positive invariant of §4.3/§5.1: with all three allocator entry points
stage-gated (`live allocate`→`LIVE`, `paper allocate`→`{paper, forward_tested}`, `intake`→`candidate→
paper`), **an active allocation can only ever exist on `stage ∈ {paper, forward_tested, live}`.** So the
only transitions that can possibly carry a stray allocation are the *exits from those three stages*.
Enumerating every such edge in `lifecycle._LIVE_TRANSITIONS` (plus the derived `→RETIRED`):
- `paper →` `{forward_tested` (in-book, keep)`, candidate` (revoke)`, dormant` (revoke)`, retired` (revoke)`}` ✓
- `forward_tested →` `{live` (revoke, go-live)`, paper` (in-book, keep)`, retired` (revoke)`}` ✓
- `live →` `{paper` (revoke, mirror)`, dormant` (revoke)`, retired` (revoke — **added round-3**)`}` ✓

Every non-in-book exit is in the revoke table; there are no others. The three edges Codex asked about —
`candidate→backtested`, `candidate→retired`, `dormant→retired` — **provably cannot carry an allocation**,
because no allocator can create one on `candidate`/`backtested`/`dormant` (the entry-point gates forbid
it). They are therefore correctly absent from the table: revoking on them would be dead code, not a fix.
The invariant is closed at both ends (entry gates + exit revokes), so no path leaves a
`revoked_ts IS NULL` row invisible to `active_paper_lane_count` or strandable across a lane. (No
data-migration/backfill is needed: on a clean build no pre-existing stray rows exist; if a human had
manually created one on the old lane-agnostic `live allocate` before this PR, the next exit transition
from its stage now revokes it.)

**Atomicity + the flat re-check (uniform, including go-live).** Every revoke edge routes through the
existing atomic `apply_transition(revoke_allocation=True)` critical section (`store.py:326` — top-level
`BEGIN IMMEDIATE`, flat re-check, revoke, CAS, commit-or-rollback), so #247's TOCTOU fix carries over
unchanged. The one generalization: `_assert_flat_for_bench` currently hard-codes the **LIVE** ledger; it
must instead re-check the **source stage's lane ledger** — `LedgerKind.PAPER` for a paper-book source
(`paper`/`forward_tested`), `LedgerKind.LIVE` for a `live` source — because a tenant that exits/crosses
with open positions in its source lane would orphan them in that book's NAV/reconcile (the exact #247
hazard, now on whichever lane the strategy is leaving). **This flat re-check applies uniformly, INCLUDING
`forward_tested→live`**: a `forward_tested` tenant must be **paper-flat** before it can go live (its
source-lane ledger is PAPER), so revoking its paper allocation can never strand open simulated positions.
There is no contradiction with §9: paper-position safety on go-live is handled here in-scope by the same
uniform flat re-check; §9 no longer defers it.

**Go-live needs BOTH `store.py` `live_authorization` guards narrowed to the exact edge
`source==FORWARD_TESTED AND target==LIVE` (Codex round-3 HIGH-3).** There are two adjacent guards, and
round-3 tightens *both*:
1. **The revoke-incompatibility guard** (`store.py:316`) today raises `"live_authorization is
   incompatible with revoke_allocation"`, on the (now-outdated) assumption that the go-live write and an
   allocation revoke never co-occur. Finding #2 makes them co-occur *by design*. It is narrowed to permit
   `revoke_allocation` **only for `source==FORWARD_TESTED AND target==LIVE`** (not merely "target is
   LIVE") — every other `live_authorization`-bearing case still forbids revoke.
2. **The acceptance predicate itself** (`store.py:320`) today accepts a `live_authorization` for **any**
   `to is Stage.LIVE and actor is Actor.HUMAN`. Round-3 narrows the *acceptance* to require the source
   too: `rec.stage is Stage.FORWARD_TESTED AND to is Stage.LIVE AND actor is Actor.HUMAN` — so a
   `live_authorization` payload can be recorded ONLY on the one legitimate go-live edge, not on any
   hypothetical future `*→live` edge. `validate_transition` already guarantees `FORWARD_TESTED` is the
   only legal source into `LIVE` (`_LIVE_TRANSITIONS`), so this defense-in-depth tightening loses **zero**
   legitimate cases while removing the over-broad "target-only" acceptance surface.

Neither narrowing weakens any live-authorization security check — both make the accepted set *smaller*,
keyed on the exact edge. After go-live the strategy is `live` + unallocated; `_still_live_allocated`
(`live_cmd.py:154`) is `False`, so `live run-all` skips it until a human `live allocate`s a fresh,
live-sized allocation (§4.3).

**Note on #317's own reachability.** #317's automated loop only ever *admits* (candidate→paper, §5.2);
it never itself drives a book exit, go-live, or demotion (those are human/quarantine actions off the
autonomous path). But the wiring must land in this PR: §6.2's "slots free when a tenant leaves" and
§4.3's go-live gate are **false against the current code** without it, and the `live→paper` mirror leak
is a silent wrong-lane-sizing bug that this PR's own paper allocations make reachable the moment a
strategy is ever demoted. Tests cover each edge (§8).

### 5.5 Re-entry lands unallocated — `run-all` must SKIP, not raise (consequence of §5.4)
Once §5.4 revokes on book-exit/lane-crossing, a strategy re-entering the paper book from outside
(`dormant→paper` recovery, `live→paper` demotion) sits at `stage∈{paper}` with **no active allocation**.
`paper run-all` today *raises* `ValueError("… has no paper allocation")` on any such strategy
(`paper_cmd.py:377`), which would abort the **entire** multi-tenant cycle — a book-wide DoS every time a
strategy is recovered/demoted but not yet re-allocated. (This latent bug already exists on
`live→dormant→…→dormant→paper` today; §5.4 merely widens the surface, so #317 must resolve it.)

**Decision:** change `paper run-all` to **skip** (do not tick) a `stage∈{paper, forward_tested}`
strategy that has no active allocation, treating "no allocation" as "not currently a book tenant." This
is:
- **safe** — every §5.4 revoke is preceded by the source-lane flat re-check, so an unallocated paper
  strategy is always *flat*; skipping it attributes nothing and orphans nothing;
- **non-masking** — §5.2's atomic admit guarantees an *admitted* tenant is never allocation-less, so the
  only unallocated-in-book strategies are intentional recovery/demotion states awaiting re-admission
  (a manual `paper allocate`; intake itself only re-allocates `candidate`s, §3);
- **consistent** with §4's `active_paper_lane_count`, which already counts only *allocated* paper/
  forward_tested strategies. Among trading-stage strategies, holding an allocation is exactly what
  distinguishes an active book tenant from a not-yet-(re)admitted re-entrant (the "active book tenant"
  definition of §2), uniformly across intake, the count cap, and `run-all`.

**The skip is checked BEFORE the strategy module is loaded (Codex round-3 IMPORTANT).** Today the loop
calls `load_gated_strategy(conn, name, ...)` (`paper_cmd.py:717`) — which imports the strategy module
via `load_tradable_strategy` (`gating.py:27`) — *before* it would ever reach the allocation check. #317
moves the allocation check to the **top of the per-strategy loop body**: read `active_allocation(conn,
rec.id)` first and, if absent, record the skip and `continue` **without** calling `load_gated_strategy`.
So an unallocated (thus not-a-tenant) strategy whose module happens to be broken/uninstallable can never
abort the whole book with an import error — a non-tenant is never imported. (`active_allocation` needs
`rec.id`; `run-all` already has each `prec`/`rec` from `list_strategies`, so no extra DB round-trip.)

**Observability (not silent) — in EVERY envelope, computed up-front (Codex round-3 IMPORTANT).** So a
skipped in-book strategy is never invisible operator confusion, `paper run-all` surfaces the skips
explicitly in a `skipped_unallocated: [names…]` field. Crucially, this list is **computed once, up-front**
— immediately after the `paper` list is built (a single `active_allocation` check per listed strategy) —
and included in **every** run-all exit envelope, including the early returns that fire *before* the
strategy loop: `no paper-lane strategies`, `global halt engaged`, `reconcile halt`, and
`reconcile deferred`. (Previously the field would only have been populated in the normal
loop-completes path, so an operator whose cycle deferred on reconcile would not see which tenants are
awaiting re-allocation.) The strategies that actually tick are exactly `paper` minus
`skipped_unallocated`. Skip ≠ silence, in any exit path.

**Reconcile stays stage-scoped, and that is correct (Codex round-3 MINOR).** The account-level paper
reconcile attributes fills by *stage* (`attributed_paper_net` joins `stage IN ('paper',
'forward_tested')`, `paper_reconcile.py:35`), not by allocation. This is deliberately left unchanged: an
unallocated in-book strategy is **always flat** (every §5.4 revoke is preceded by the source-lane flat
re-check, and a freshly recovered/demoted strategy has no fills of its own yet), so it contributes
**zero** to `attributed_paper_net` — reconcile semantics are identical whether it is filtered by stage or
by allocation. The earlier "neither tick nor reconcile" phrasing is therefore narrowed to "not
**ticked**"; the account-wide reconcile legitimately still runs stage-scoped because a flat non-tenant is
accounting-invisible to it.

`paper_cmd.py` is **not** CODEOWNERS-protected, so this raise→skip change (and the JSON field) ships in
the same PR.

---

## 6. Decision 4 — concurrency & idempotency under the multi-tenant book

### 6.1 Concurrency
- **Capital bound is race-proof.** `intake_candidate_to_paper`'s `BEGIN IMMEDIATE` (which wraps the
  shared `allocate_locked` Σ-check, §5.1/§5.2) serializes the read-check-write, so even two racing
  intake processes can **never** over-commit capital — the second sees the first's row under the write
  lock and raises. Capital can't be breached.
- **Count bound is ALSO race-proof (Codex round-3 HIGH-2 — moved in-txn).** The authoritative
  `active_paper_lane_count < MAX_CONCURRENT` check is re-read **inside**
  `intake_candidate_to_paper`'s `BEGIN IMMEDIATE` (§5.2 step 2), so it is serialized by the same write
  lock as the capital check. Two concurrent intake processes can **no longer** both read count=7 and
  admit to 9: the second process's `BEGIN IMMEDIATE` blocks until the first commits, then re-reads
  count=8 and raises `CountCapReached`. The outer loop's `count < MAX_CONCURRENT` read (which does sit
  outside the txn) is retained ONLY as a cheap early-`break` to avoid launching doomed transactions; it
  is never the safety boundary. (Single-operator discipline — epic §6.4 / #316's file lock — still holds
  for the rest of the paper job, but the count cap no longer *depends* on it: it is now a hard bound like
  capital, not merely a governor.)
- **Lane crossing — the reachable case is closed in-scope; only the concurrent-book case is deferred.**
  `total_allocated` is lane-agnostic. The one lane-crossing hazard reachable from #317's state — a
  paper-allocated strategy promoted to `live` and sized off its paper slice — is **closed here** by
  revoking the paper allocation on the `forward_tested→live` exit (§4.3/§5.4), so it never survives into
  live sizing. What remains deferred is strictly the *concurrent* live+paper book (both lanes allocating
  against a shared, lane-agnostic Σ at the same time); that needs a `lane` column on
  `strategy_allocations` + lane-scoped `total_allocated` (a schema bump, §9) and cannot occur under the
  paper-only autonomous operator (live is human-gated and off the autonomous path).

### 6.2 Idempotency
Intake is **convergent, not additive** — unlike `run-all`'s trade path it needs no session-idempotency
guard:
- A strategy admitted once is now `stage=paper`, so it is no longer a `candidate` and cannot be
  re-selected. Re-running intake with no freed slot is a **no-op**.
- No partial-intake state exists to re-converge (§5.2): the admit is one atomic write, so a crash leaves
  the strategy exactly `candidate` with no allocation — the next run simply re-selects and re-admits it
  as if the crashed run never happened. (There is deliberately no "allocated-but-still-candidate" state
  for `paper intake`'s JSON to report; the JSON only ever reports `admitted` / `queued`, §8.)
- Slots free **because** a book-exit / lane-crossing revokes the tenant's allocation (§5.4):
  `paper→dormant`, `paper→retired`, `paper→candidate`, `forward_tested→retired`, `forward_tested→live`,
  and `live→paper` each revoke it. The next intake run then admits the FIFO-next candidate into the
  freed slot. (This is now true in code — §5.4 wires the revoke; before this PR, a benched tenant kept
  its allocation and the slot never freed.)

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

> **DEFERRED-vs-shipped map (see the top-of-doc banner).** Of the surface below, **#317 ships ONLY**:
> `intake_candidate_to_paper` (store.py), `paper intake` (paper_cmd.py), the `active_paper_lane_count`
> / `allocate_locked` / `CountCapReached` helpers (allocations.py), and the `paper run-all`
> `skipped_unallocated` skip (paper_cmd.py). **Everything else in this section is DEFERRED to #497**:
> `paper allocate`, `live allocate` stage-gating, `allocate_in_lane`, deletion of `allocate()`,
> `_paper_account_equity()`, ALL of `transitions.py` revoke wiring, the `_assert_flat_for_bench`
> generalization, the narrowed go-live `live_authorization` guards, and every test below that exercises
> those (book-exit revoke, `live→retired`, go-live shed, `live→paper` mirror, `live allocate` /
> `paper allocate` / `allocate_in_lane` stage gates, `allocate_locked` containment). The `[NEW]`/`[CHANGE]`
> tags below describe the FULL design, NOT this PR's diff.

CLI / non-protected:
- **[NEW] `paper allocate <name> --capital $X [--max-concurrent 8]`** — `algua/cli/paper_cmd.py`.
  Routes through the shared `allocations.allocate_in_lane(..., allowed_stages={PAPER, FORWARD_TESTED},
  max_concurrent=<cap>)` primitive: the stage gate is re-asserted **under the write lock** (not on the
  pre-txn `repo.get()`), and **when the target is currently unallocated** (a count-*increasing*
  allocation) the count is re-read under the lock and raises `CountCapReached` if `≥ max_concurrent`
  (Codex round-4/5 HIGH — closes the `dormant→paper` recovery re-entry bypass AND its TOCTOU variant); a
  **resize** of an already-allocated tenant is count-exempt (no count change). Equity read via
  `_paper_account_equity()` BEFORE the txn. Serves as the re-allocation path for §5.5 recovered/demoted
  in-book strategies, cap-bounded like intake (§5.1).
- **[CHANGE] `live allocate` gated to `stage == LIVE` only, in-lock** (§4.3, Codex round-3 HIGH-1 +
  round-5 TOCTOU) — `algua/cli/live_cmd.py` (NOT protected): replace the current `reject DORMANT`-only
  pre-txn check with a route through the same `allocations.allocate_in_lane(..., allowed_stages={LIVE},
  max_concurrent=None)` primitive, so `stage == LIVE` is re-asserted under the write lock (reject
  CANDIDATE/PAPER/FORWARD_TESTED/DORMANT/etc.). Equity read via `_live_account_equity()` before the txn.
  This closes the lane-agnostic sizing leak at the *live* entry point and, with `paper allocate` and
  `intake` similarly gated, maintains the "active allocation ⟹ trading stage" invariant at every
  allocator — what makes the §5.4 revoke table exhaustive.
- **[NEW] `paper intake [--slice 10000] [--max-concurrent 8]`** — `algua/cli/paper_cmd.py`. The
  deterministic auto-allocator: FIFO candidate selection (§3), the two-bound headroom loop (§4) with the
  `CountCapReached`/`AllocationError`→`break` and `TransitionError`(stale-selection)→`continue`
  handling (§5.2), and **one call to the atomic `repo.intake_candidate_to_paper(..., max_concurrent=…)`
  per admission** (§5.2). Emits JSON `{admitted:[...], queued:[...], skipped_stale:[...], equity,
  slots_used, slots_cap}` (`skipped_stale` = candidates a concurrent operator moved between selection and
  admit, §5.2) —
  there is no partial/allocated-but-candidate state to report (§5.2/§6.2).
- **[NEW] `_paper_account_equity()`** — `algua/cli/paper_cmd.py`, mirrors `_live_account_equity()`.
- **[NEW] candidate FIFO selector** — a small read query over `strategies` + `stage_transitions`
  selecting the **current** candidate episode as the row with the greatest `stage_transitions.id` whose
  `to_stage='candidate'`, and ordering the queue by that entry-row's **`id` ascending** (tie-break
  strategy id) — NOT by `created_at` (§3, Codex round-3 IMPORTANT), with the §4.4 unflagged pass-through
  seam. Lives in `algua/registry/` (a read helper), not in the CLI.
- **[REFACTOR] `algua/registry/allocations.py`** (NOT CODEOWNERS-protected) — restructure the stage-blind
  `allocate` into two pieces (§5.1): (1) a commit-less **`allocate_locked(...)`** (the Σ
  read-check-revoke-insert body, no `BEGIN`/`commit`), callable inside any owning transaction; (2) a
  transactional, lane-scoped **`allocate_in_lane(conn, strategy_id, capital, actor, account_equity,
  allowed_stages, max_concurrent=None)`** — `BEGIN IMMEDIATE` (top-level-only + **`BaseException`**
  rollback, mirroring `apply_transition`) → **in-lock re-read of the strategy's current stage** and
  membership assert against `allowed_stages` → conditional count check (`active_paper_lane_count <
  max_concurrent` only when currently unallocated) → `allocate_locked` → commit. The stage-blind public
  `allocate()` is **deleted** (its only production caller was `live allocate`, now on `allocate_in_lane`);
  tests that used it move to `allocate_in_lane`/`allocate_locked`. This gives `intake_candidate_to_paper`
  a Σ-check it can compose (§5.2) AND both CLI allocators a single TOCTOU-safe, cap-enforcing entry point
  (Codex round-4/5 HIGH).
- **[NEW] `active_paper_lane_count` read helper** — a small registry read (`strategies` JOIN
  `strategy_allocations` WHERE `stage IN ('paper','forward_tested') AND revoked_ts IS NULL`), NOT
  CODEOWNERS-protected, shared by `intake_candidate_to_paper` (§5.2) and `allocate_in_lane` (§5.1) so the
  count definition can't drift between the two cap enforcers.
- **[CHANGE] `paper run-all` skips unallocated in-book strategies** (§5.5) — `algua/cli/paper_cmd.py`
  (NOT protected): a `stage∈{paper,forward_tested}` strategy with no active allocation is **skipped**
  (not ticked) instead of raising `ValueError`. The allocation check is done at the **top of the
  per-strategy loop body, BEFORE `load_gated_strategy`** (so a broken non-tenant module can't abort the
  book), and `skipped_unallocated: [...]` is **computed up-front and emitted in every exit envelope**
  (including the pre-loop reconcile-halt / reconcile-defer / global-halt / no-strategies returns) — §5.5,
  Codex round-3 IMPORTANT. Removes the recovery/demotion book-wide DoS that §5.4's revoke would otherwise
  create, without making the skip silent.

CODEOWNERS-protected (this PR **touches** them — see the reversed check below):
- **[NEW] `StrategyRepository.intake_candidate_to_paper(rec, capital, actor, account_equity,
  max_concurrent)`** — `algua/registry/store.py` (**PROTECTED**). One top-level `BEGIN IMMEDIATE`:
  stage-CAS `== CANDIDATE` (drift guard, #246), **count re-read `active_paper_lane_count <
  max_concurrent` else raise `CountCapReached`** (§5.2 step 2, Codex round-3 HIGH-2), `allocate_locked`
  (capital bound), `_apply_transition_locked` (`candidate→paper` + audit row), commit-or-rollback (§5.2).
  Reuses the existing top-level-only + blanket-`BaseException`-rollback discipline at `store.py:335`.
- **[CHANGE] book-exit / lane-crossing allocation revoke** (§5.4, findings #1 + #2 + the `live→paper`
  mirror + the round-3 `live→retired` addition) — `algua/registry/transitions.py` (**PROTECTED**): set
  `revoke_allocation=True` for `paper→dormant`, `paper→retired`, `paper→candidate`,
  `forward_tested→retired`, `forward_tested→live` (go-live), `live→paper` (demotion), **and
  `live→retired`** — mirroring the existing `LIVE→DORMANT` branch at `transitions.py:66`. And
  `algua/registry/store.py` (**PROTECTED**): generalize `_assert_flat_for_bench` to re-check the
  **source stage's lane ledger** — `LedgerKind.PAPER` for a `paper`/`forward_tested` source (including
  go-live), `LedgerKind.LIVE` for a `live` source (`LIVE→DORMANT`, `live→paper`, `live→retired`). This
  makes the go-live paper-flat precondition in-scope, so nothing in §9 defers it.
- **[CHANGE] BOTH go-live `live_authorization` guards narrowed to the exact edge `source==FORWARD_TESTED
  AND target==LIVE`** (§4.3/§5.4, finding #2 + Codex round-3 HIGH-3) — `algua/registry/store.py`
  (**PROTECTED**): (1) narrow the `store.py:316` "`live_authorization` is incompatible with
  `revoke_allocation`" guard to permit revoke **only for `source==FORWARD_TESTED AND target==LIVE`** (NOT
  target-only); AND (2) narrow the `store.py:320` *acceptance* predicate from `to is Stage.LIVE and actor
  is Actor.HUMAN` to `rec.stage is Stage.FORWARD_TESTED and to is Stage.LIVE and actor is Actor.HUMAN`,
  so a `live_authorization` is accepted ONLY on the one legitimate go-live edge. `validate_transition`
  already guarantees `FORWARD_TESTED` is the sole legal source into `LIVE`, so both narrowings lose zero
  legitimate cases while shrinking the accepted surface — no live-authorization security check is
  weakened. No change to `live_cmd.py` sizing itself — the already-present `_still_live_allocated`
  (`live_cmd.py:154`) skips the now-unallocated live strategy for free.

Tests — `tests/test_cli_paper.py` / new `tests/test_paper_intake.py` (+ registry-level tests for the
protected changes):
- FIFO order **by `stage_transitions.id`** (current-episode = greatest id to `candidate`; queue ordered
  by min such id; tie-break strategy id) — incl. a `candidate→backtested→candidate` bounce re-queuing by
  its latest episode, and a same-`created_at` pair still ordered deterministically by id (§3, round-3).
- Both headroom bounds (count-binds and capital-binds cases); a full book queues the rest.
- **Count cap enforced IN-txn (round-3 HIGH-2):** `intake_candidate_to_paper` at `count==max_concurrent`
  raises `CountCapReached` and admits nothing (a registry-level test that the count is re-read under the
  transaction, not just the outer loop guard).
- **Atomic admit crash-safety (finding #3):** a forced failure between the allocation insert and the
  CAS rolls back BOTH — the strategy is left `candidate` with **no** allocation row (no
  allocated-but-candidate state); idempotent re-run then admits it cleanly.
- **Stale-selection CAS (round-3/4 IMPORTANT):** a candidate moved out of `CANDIDATE` between selection
  and the admit txn causes `intake_candidate_to_paper` to raise `TransitionError`, and the intake loop
  `continue`s past it (skips, does not break, does not crash), records it in `skipped_stale`, and still
  admits the next eligible one.
- **Book-exit revoke (finding #1):** `paper→dormant`, `paper→retired`, `paper→candidate`,
  `forward_tested→retired` each revoke the allocation and free a slot the next intake run fills; the
  paper-ledger flat re-check blocks benching a non-flat paper tenant.
- **`live→retired` revoke (round-3 HIGH-4):** retiring a `live` strategy revokes its live allocation
  (LIVE-ledger flat re-check applies), leaving no stray `revoked_ts IS NULL` row on the terminal stage.
- **Go-live gate (finding #2):** `forward_tested→live` on a non-paper-flat tenant is rejected; once
  flat, go-live revokes the paper allocation, the strategy is `live`+unallocated, `_still_live_allocated`
  is `False`, and `live run-all` does NOT size it off the former paper slice until a fresh `live allocate`.
- **`live_authorization` acceptance narrowed (round-3 HIGH-3):** a `live_authorization` payload is
  accepted ONLY for `source==FORWARD_TESTED AND target==LIVE`; the go-live path still succeeds, and the
  revoke-incompatibility guard permits revoke only on that same edge.
- **`live→paper` mirror gate:** a demoted live strategy has its *live* allocation revoked (LIVE-ledger
  flat re-check), so `paper run-all` does NOT size it off the former live slice — it is skipped until a
  `paper allocate`.
- **`live allocate` stage gate (round-3 HIGH-1):** `live allocate` on a `CANDIDATE`/`PAPER`/
  `FORWARD_TESTED`/`DORMANT` strategy is **rejected**; only `stage==LIVE` is accepted (the post-go-live
  re-allocation path).
- **`run-all` skip (§5.5):** an unallocated `stage=paper` strategy (recovered/demoted) is skipped, not
  raised on — the cycle completes, other tenants tick, the skip is decided **before** `load_gated_strategy`
  (a deliberately broken/unimportable non-tenant module does NOT abort the book), and the skipped name
  appears in `skipped_unallocated` — **including on a reconcile-deferred / halt early-return** envelope.
- **`paper allocate` stage gate + count cap (§5.1, round-4 HIGH):** allocating a `LIVE` or `CANDIDATE`
  (or DORMANT/other pre-book) strategy is **rejected**; allocating a `stage=paper` recovered/demoted
  strategy succeeds and re-admits it; Σ≤equity enforced. **Recovery re-entry bypass closed:** with a full
  book (8), a recovered `dormant→paper` strategy's `paper allocate` raises `CountCapReached` (does NOT
  create a 9th tenant); a **resize** of an already-allocated tenant is still allowed (count unchanged).
- **`allocate_in_lane` in-lock stage gate / TOCTOU (§5.1, round-5 HIGH):** a registry-level test that
  `allocate_in_lane` re-reads the stage under the `BEGIN IMMEDIATE` — a strategy whose stage is moved out
  of `allowed_stages` (e.g. `paper→dormant`, or `live→paper` for the live path) before the allocation
  write is **rejected** by the in-lock membership assert (the pre-txn read is not trusted); the deleted
  stage-blind `allocate()` has no remaining production caller.
- **`allocate_locked` containment (round-6 MINOR hardening):** `allocate_locked` is documented
  "owning-transaction only" (commit-less, stage-blind) and an AST/grep guard test asserts NO production
  module under `algua/` calls it except `allocate_in_lane` and `intake_candidate_to_paper` (the two safe,
  stage+cap-gated entry points) — so the stage-blind primitive can't be reintroduced as an unguarded
  allocation path.
- An intake→`run-all` composition test (an admitted tenant ticks on its first cycle; an unallocated
  in-book strategy is skipped, never crashes the cycle).

**CODEOWNERS check (REVERSED from revision 1).** This PR (#317) edits `algua/registry/store.py`
(the atomic `intake_candidate_to_paper` primitive) — **CODEOWNERS-protected** — so **the PR MUST stay
OPEN for human merge (auto-merge disabled)** regardless of the #497 split. The single-transaction
atomic admit cannot deliver atomicity without touching the protected transition/allocation machinery
in `store.py`. The deferred #497 work additionally edits `algua/registry/transitions.py` (the exit-revoke
flag wiring) and further `store.py` guards — also protected — so #497 will likewise be human-merged.
`algua/registry/transitions.py` is **NOT** touched by THIS PR (all revoke wiring is #497).
`paper_cmd.py` (the `paper allocate`/`paper intake` additions and the §5.5 raise→skip), `live_cmd.py`
(the round-3 `live allocate` stage gate), `allocations.py`, the FIFO read helper, tests, and this doc
remain non-protected. `algua/contracts/lifecycle.py` is **NOT** touched — every edge above
(`paper→{dormant,retired,candidate}`, `forward_tested→{retired,live}`, `live→{paper,retired}`) is already
legal in `_LIVE_TRANSITIONS`; only the revoke *flag*, the flat-ledger generalization, the two narrowed
go-live guards, and the atomic admit (with its in-txn count re-read) are new. **No schema bump** — the
go-live and `live→paper` gates are lane-free; the mixed-lane `lane` column stays the deferred §9
follow-up (only the *concurrent* live+paper book needs it).

---

## 9. Deferred follow-ups (filed, out of scope)
- **Exit-side allocation invariant — book-exit revoke + lane gating (#497).** The entire exit half of
  the design (§4.3, §5.1, §5.4): the `transitions.py` revoke wiring on every book-exit / lane-crossing
  edge, the `paper allocate` CLI, `live allocate` stage-gating (with in-lock TOCTOU re-assert), the
  shared `allocate_in_lane` primitive, deletion of the stage-blind `allocate()`, the go-live paper-slice
  shed + narrowed `live_authorization` guards, `_assert_flat_for_bench` generalization, and all the
  Section-8 tests for them. **Until #497 lands the shipped code has a known capital leak** (benching /
  retiring / back-stepping a paper tenant permanently consumes its slice's headroom). Tracked as #497.
- **Lane-tagged allocations** — a `lane` column on `strategy_allocations` + lane-scoped
  `total_allocated`, so a genuinely **concurrent** live+paper book can't cross-count Σ against the wrong
  *equity denominator* (§6.1). Schema bump; only needed if the operator ever runs live and paper books at
  the same time. This follow-up is **only** the concurrent-book Σ-accounting case. Both lane-crossing
  *sizing* leaks — a paper slice inherited by live sizing on go-live, AND a live slice inherited by paper
  sizing on demotion — are NOT deferred: they are closed in-scope by the §5.4 revoke-on-lane-crossing
  (which is lane-free and needs no column). Paper-position flattening on go-live is likewise NOT deferred
  — it is the same uniform source-lane flat re-check (§5.4).
- **`family-audit` quarantine guard** (epic §6.5) — the §4.4 seam's real filter: block intake of a
  flagged candidate, bench a flagged paper tenant `paper→dormant`.
- **`is_due` scheduler + systemd timers** (epic Group 4) — the always-on driver that periodically
  invokes intake + `run-all`.
- **Weighted / risk-parity slice sizing** — replace the fixed slice with per-strategy sizing.
