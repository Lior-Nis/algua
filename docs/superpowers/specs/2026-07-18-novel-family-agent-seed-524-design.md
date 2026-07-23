# NOVEL family: agent-create with a durable seeded lifetime-breadth prior (#524)

## R10 amendment (2026-07-20) — drop the human mint budget; keep the automatic rate cap as the SOLE bound

**Decision (user-approved, after a Codex adversarial consult).** The R9 implementation shipped TWO
count-bounds on agent-NOVEL family minting: (1) an **automatic per-window rate cap** (`AGENT_NOVEL_MINT_CAP`
≈ 8 mints / 90 days, canonical-UTC, fail-closed) and (2) a **human-replenished lifetime mint budget**
(`AGENT_NOVEL_MINT_LIFETIME_BUDGET` + an append-only `agent_mint_grants` ledger, topped up via a human-only
`registry grant-novel-mints` CLI). **R10 REMOVES bound (2) entirely and KEEPS bound (1) as the sole
automatic bound.**

**Rationale.** The project goal is zero-human autonomy; a human-replenished budget is a recurring
human-in-the-loop friction that defeats that goal, and its `grant-novel-mints` gate leaned on a declared
`--actor human` string (a forgeable surface; the governance-challenge signature machinery added to shore it
up was extra attack surface that existed ONLY for this one command). A Codex adversarial consult confirmed
that *some* automatic count-bound must remain: the deferred pass-time seed alone does NOT stop the
**repeated-founder attack**, because the family is minted (and seeded with the lifetime-breadth prior) only
*after* the founder's gate passes — the founder itself escapes the tax, so an attacker who keeps founding
fresh families each escapes their own arm-0 seed. The automatic per-window rate cap is the retained
count-bound that throttles that repetition without any human action. Removing the human budget therefore
(a) restores full autonomy, (b) deletes the forgeable-`--actor` grant gap, and (c) reopens **no** gaming
vector beyond the already-accepted rate-cap-throttled residual — the rate cap was always the burst control;
it is now simply the *only* control.

**What R10 removes:** the `registry grant-novel-mints` CLI + `grant_agent_novel_mints` repo method; the
`agent_mint_grants` ledger table (never created — dropped from the v37 schema); `AgentMintBudgetExhaustedError`;
the `AGENT_NOVEL_MINT_LIFETIME_BUDGET` constant + the under-lock budget check; and the non-strategy
`governance_challenges` table + `human_actor` governance-challenge helpers, which existed only to gate
`grant-novel-mints`. **What R10 keeps unchanged:** the deferred pass-time mint architecture, the
lifetime-breadth SEED (`seeded_prior_combos` / `founder_gate_id`), the automatic per-window rate cap
(`AgentMintCapError`), the still-NOVEL fingerprint CAS (`FamilyGraphDriftError`), the append-only triggers,
the on_peek release-on-drift, the type-safe `n_combos` CHECK, and the NOVEL+human fresh-0-prior-family path.

The R9 sections below describe the original two-bound design and are retained for history; where they
reference the human lifetime budget / `grant-novel-mints` / `agent_mint_grants`, read them as superseded by
this R10 amendment.

---

**Status:** design — **R9 (2026-07-18), revised after GATE-1 BLOCK.** R8 had a Codex APPROVE, but three
independent Codex read-only adversarial GATE-1 passes (condensed summary; summary + governance-residual
arguments; and the FULL verbatim body §1-9) all returned **BLOCK** with new *structural* objections —
not the previously-closed issues (naive locking, missing CAS, scalar math, slug DoS, type-unsafe SUM),
but what R8 still left documentation-only or partially-addressed. **R9 makes those invariants
MECHANICALLY ENFORCED** and closes the two governance/atomicity gaps with real controls, not prose. The
eight findings and their R9 resolutions (§0 details):
1. **Append-only invariant is now enforced by DB triggers**, not documented — `BEFORE UPDATE`/`BEFORE
   DELETE` triggers `RAISE(ABORT)` on the five classifier-read tables (the sole permitted mutations are
   the `family_members.removed_at` tombstone flip and the one-way legacy profile materialisation), plus a
   #277-style source-scan guard test that FAILS if a new classifier DB read is added without a
   fingerprint component (§2.3, §2.5).
2. **A genuine non-statistical lifetime mint budget is BUILT NOW** (not deferred): a durable,
   human-replenished agent-NOVEL mint quota (`AGENT_NOVEL_MINT_LIFETIME_BUDGET` + an append-only
   human-only `agent_mint_grants` ledger) enforced under the lock, so the cross-window channel is
   **lifetime-finite**, not merely rate-bounded — a governance quota, not a statistical bar, so it does
   not offend `#324` (§2.1, §5.1 step 5b, §6A).
3. **The profile axis is now transactionally checkable**: each member's `(code_hash, factors)` is
   **persisted in the DB at assignment** (`family_members.member_code_hash`/`member_factors_json`), the
   classifier reads the persisted (immutable) profile instead of live source, so the profile axis is
   fully covered by the under-lock `graph_fingerprint` CAS — the separate pre-lock-only `profile_digest`
   machinery is **eliminated** (§2.3, §7.1).
4. **The holdout-burn-on-drift-race channel is closed, not logged**: the reservation burn-commit at
   `on_peek` is preceded, **in the same transaction**, by the full pending-NOVEL snapshot revalidation
   (now entirely DB state per finding 3); on any drift it raises BEFORE finalising, so the reservation
   stays pending and the existing release path frees the window — **no holdout is burned for drift caught
   at/before the on_peek burn**. NARROWED RESIDUAL (documented + monitored, not fully closed): run_gate's
   own post-peek/pre-lock re-check runs AFTER on_peek has committed the burn, so drift first visible in that
   window burns the holdout then fails the gate; it gets release-on-failure + a WARNING audit record (§7.2).
5. **The cap/budget constants live in CODEOWNERS-protected `algua/registry/store.py`** as module
   constants — **no CLI flag, no env var** reachable by the autonomous loop; changing them requires a
   human PR to a protected file (§5.1 step 5b, §6A).
6. **A queryable founder linkage** — `families.founder_gate_id` (FK to the founding `gate_evaluations`
   row, stamped at mint) closes the audit blind spot left by `gate_evaluations.family_id` staying NULL
   for founders (§2.1, §5.1 step 5c, §6D).
7. **Overflow-guard ordering fixed**: the seed is a **WHERE-filtered SUM over well-typed in-range rows**
   (`typeof='integer' AND n_combos BETWEEN 1 AND MAX_N_COMBOS`), so one corrupt legacy row is excluded
   (contributes 0, never shrinks the seed below the legitimate rows) and can **never** overflow the SUM
   or permanently DoS all future mints — corruption is an audit signal, not a hard permanent block
   (§2.4, §5.1 step 5a).
8. **Fail-closed on non-canonical-UTC `created_at`** in the cap/budget query (each counted agent row's
   timestamp must parse as canonical UTC or the mint fails closed), plus explicit multi-parent /
   ancestor-dedupe test cases for the additive seed summation (§5.1 step 5b, §8-8, §8-18).

Supersedes the issue-body proposal and the R1..R8 drafts (all BLOCKED / approve-then-blocked). R0's
accepted-scalar mechanics are retained throughout.
**Issue:** #524 (`[ds]` — #222 family-governance follow-up)
**CODEOWNERS-protected files touched → PR stays OPEN for human merge:**
`algua/registry/store.py` (cap/budget constants + mint), `algua/registry/promotion.py`,
`algua/research/clustering.py` (import surface). Also `algua/registry/db.py` (schema bump + triggers)
and `algua/registry/repository.py` (Protocol).

---

## 0. GATE-1 history — the BLOCKs this design answers (R0..R8 → R9)

**R0 (issue body) — BLOCKED (5 HIGH):** "reuse the PARENTAGE edge" is impossible for NOVEL (no
incumbent to edge to); a 90-day `windowed_total` seed still forgets old tests; the windowed seed is a
no-op for the founder; no schema column exists for an injected scalar; agent minting is a new
authority and advisory `family-audit` is not an enforcement backstop.

**R1 (scalar-seed draft) — BLOCKED (3 HIGH + MEDIUM/LOW), all on LIFECYCLE/AUTHORITY TIMING** (the
scalar math was accepted). R1 minted the seeded family inside `_classify_and_assign_family`, which
`promotion_preflight` calls **before** measured-breadth validation and **before** the holdout burn:
1. **Minting was not economically gated** — a family was born in preflight even for a strategy with
   no measured sweep and before any single-use holdout burn, invalidating R1's "holdout economics
   rate-limit minting" claim.
2. **Stale shelf inventory** — a preflight-created family whose gate later FAILS persists with an
   *old* seed; because funnel-lifetime only grows, a stale seed is *smaller* than a fresh one, so a
   later sibling promoting against it pays a reduced tax — a reopened wait-out-the-window variant. R1's
   early-return idempotency made it worse (it prevented reseeding).
3. **Unpromoted strategies became clustering anchors** — R1 assigned membership in preflight, so a
   failed NOVEL attempt stayed an *active* family exemplar (`all_families_with_member_profiles` reads
   active members) and could steer future MERGE/PARENTAGE/NOVEL verdicts.
Plus MEDIUM (the human fresh-zero audit query was inaccurate post-migration) and LOW (no non-negative
CHECK on the new column).

**R2 (deferred-creation draft) — BLOCKED (1 HIGH + 2 MEDIUM + LOW), timing accepted.** R2's move
(defer creation to the atomic pass-moment) was accepted as closing R1-1/2/3, but four grounded holes
remained: (R3-HIGH) `run_gate`'s family CAS skips when `expected_family_id is None`, so a
pending-NOVEL request would not detect a concurrent None→assigned drift between preflight and commit;
(R3-MED-1) stamping the created `family_id`/seed onto the just-evaluated gate row contradicts the
evaluated `family_id=None`/`family_lifetime_effective=0` in `n_funnel`/`decision_json`; (R3-MED-2)
the "agent seed > 0" invariant was *assumed* from `search_trials` shape, but `n_combos` has no CHECK
and no writer validation; (R3-LOW) the mint must use raw locked INSERTs, not the public
`create_family`/`assign_strategy_to_family` helpers (which open their own transactions).

**R3 (mint-invariants draft) — BLOCKED (1 HIGH + 1 MEDIUM).** R3 closed all four R2 findings, but:
(R4-HIGH) a pre-existing *legacy negative* `search_trials.n_combos` row could shrink the seed `SUM`
so a too-small-but-positive seed passed the `seed > 0` guard and laundered the lifetime tax — the
guard must reject *corruption* (`MIN(n_combos) < 1`), not merely non-positivity, and must not rely on
future-only writer validation; (R4-MEDIUM) test-plan item 3 still described the R2 gate-row stamping
that R3 had removed. Both are fixed here: §5.1 step 5 now computes `(SUM, MIN, COUNT)` under the lock
and fail-closes on `MIN < 1` OR `SUM ≤ 0` (§6D), and test 3 now asserts NO stamping (§8).

**R4 (deferred-pass-time-mint draft) — BLOCKED (3 HIGH + 3 MEDIUM/LOW).** R4 closed all R3 findings
but three HIGHs survived the adversarial re-review, plus cheap MEDIUM/LOW notes:
1. **(R5-HIGH-1) Concurrent-mint re-classification gap.** The under-lock check was only "this strategy
   is *still unassigned*" — it did NOT re-validate the NOVEL verdict against the family graph *as it
   stands at commit*. Two related variants A and B both preflight NOVEL before either mints; A commits
   first and mints a seeded family; B then takes the lock, sees itself still unassigned, and mints a
   *second root family* — deliberate family-splitting via concurrency — even though B would now classify
   as MERGE/PARENTAGE against A's just-minted family. R4 even *blessed* "two concurrent NOVEL promotes →
   distinct families," which is only safe if re-validated as still-NOVEL after earlier commits.
2. **(R5-HIGH-2) Repeated-founder laundering.** The founder always pays `family_lifetime = 0`; the seed
   only taxes *future siblings within that one family*. An agent can therefore mint **N families with one
   founder each** instead of N siblings of one family, and each founder resets the family arm. Within a
   90-day window the funnel-wide `windowed_total` term already taxes the founders against each other, but
   founders spaced **>90 days apart** forget one another — the exact "windowed forgets" hole this feature
   exists to close, reintroduced at the founder granularity. R4 left the only mitigation ("per-window mint
   cap") as *optional defense-in-depth*; the review requires it be **load-bearing**.
3. **(R5-HIGH-3) Integrity guard is not type-safe.** SQLite dynamic typing: a legacy direct-insert row
   with a TEXT/BLOB/REAL `n_combos` (e.g. `'abc'`) can be silently coerced/omitted by `SUM` while
   `MIN(n_combos) < 1` does **not** reliably catch it — a positive-but-undersized seed still launders the
   tax. The guard must assert row TYPE, not just a numeric minimum.
MEDIUM/LOW: deterministic `{name}_family` slug is an unintended DoS surface (a reserved name aborts a
*passed* gate); the store-boundary actor guard must `Actor(actor)`-coerce BEFORE the identity check (a
string actor would otherwise mis-evaluate); and the human-NOVEL create-in-preflight asymmetry (failed
human attempts still anchor the classifier while agent attempts don't) must be documented or closed.

**R5 (three-HIGH-fix draft) — BLOCKED (1 CRITICAL + 3 HIGH + 3 MEDIUM/LOW).** R5 closed the three R4
HIGHs but the fixes introduced/left:
1. **(R6-CRITICAL) Fingerprint captured AFTER a stale classification.** R5 captured the graph
   fingerprint *after* the NOVEL verdict was computed. If a related family committed in the gap between
   the classifier's graph read and the fingerprint read, the pending row stored the *new* fingerprint
   against a verdict computed on the *old* graph — the commit CAS then matches and mints a second root.
   The CAS proved "unchanged since fingerprint read," not "unchanged since classification read."
2. **(R6-HIGH-1) The cap is a rate limit, not a durable founder correction.** 8 fresh-zero founders per
   window, forever; each singleton-family founder still pays `family_lifetime = 0`. Codex asked whether
   this is an accepted governance throttle or a statistical gap.
3. **(R6-HIGH-2) The cap trusted `family_events` (derived audit state), not canonical `families`.** A
   repair/migration/older-helper family row lacking a correctly-shaped event undercounts the cap.
4. **(R6-HIGH-3) The fingerprint did not cover the full classifier read-set.** An existing member's
   stored backtest-returns refresh can flip the return-correlation verdict with no family/event row
   added — fingerprint matches, stale NOVEL mints. R5 deferred exactly this axis.
5. **(R6-MEDIUM) `count+id_sum+event_count` is not a robust digest** under deletes/rewrites; the bounded
   slug probe is still a DoS if the bound is exhausted; and the `typeof` guard has no upper bound, so a
   single absurd-but-integer `n_combos` can overflow `SUM` into a permanent fail-closed DoS.
6. **(R6-LOW) `PendingNovelFamily.actor`/`verdict` unvalidated** at the store boundary.

**This design (R6) keeps R4/R5's deferred-pass-time-mint architecture** (create+seed+assign folded into
the `record_gate_with_fdr_and_maybe_promote` `BEGIN IMMEDIATE`, never in preflight; no gate-row stamping;
raw locked INSERTs; type-safe `seed > 0` fail-closed) **and closes the R5 BLOCK:**
(CRITICAL) the fingerprint is captured **before AND after** the whole classifier read and required equal,
so the stored fingerprint provably equals the graph the verdict was computed on (§7.1), and the
under-lock CAS then proves no drift since that snapshot (§5.1 step 1); (HIGH-1) the per-window cap is
retained as **load-bearing** and its residual is closed as an **explicit, #324-anti-scaling-grounded
governance decision** — a durable funnel-lifetime founder arm is *deliberately rejected* because it would
reintroduce the exact anti-scaling pathology #324's cohort restarts exist to prevent, and the agent
founder's fresh-zero is *symmetric* with the human fresh-zero founder (§6A); (HIGH-2) the cap counts
canonical `families` rows (`created_by_actor='agent'`), not events (§5.1 step 5b); (HIGH-3) the
fingerprint is **widened to the full DB classifier read-set** — families, active membership, parentage
edges, `family_events`, AND a `backtest_returns` monotone counter (the return-correlation axis) (§2.3);
(MEDIUM) the fingerprint is a monotone `(COUNT, MAX(id))`-per-table digest, the slug is a
uuid suffix (no probe, no bound), and `n_combos` gains a sane **upper-bound** CHECK
+ writer + mint-guard well-typed count (§2.4); (LOW) the store boundary validates
`pending.actor=='agent' AND pending.verdict=='novel'` (§5.1).

**R6 (full-hardening draft) — BLOCKED (3 HIGH + 2 MEDIUM + 1 LOW).** R6 closed the R5 BLOCK but:
1. **(R7-HIGH-1) The fingerprint missed the SOURCE-derived member profiles.** The classifier recomputes
   each member's `code_hash`/`factors` from module SOURCE (`all_families_with_member_profiles` calls
   `compute_artifact_hashes`/`factors_used_by` live — confirmed NOT DB-stored), so a member-source change
   flips the verdict with no DB-fingerprint change.
2. **(R7-HIGH-2) The cap "closes" claim was an overclaim.** It is a *rate* bound, not lifetime-finite;
   `CAP` fresh-zero founders/window is permitted forever. Codex accepted an explicit governance residual
   OR a non-statistical durable authority control.
3. **(R7-HIGH-3) The cap was enforced only under the lock — too late.** An at-cap agent burned a scarce
   holdout and left no audit row before the rollback; it must pre-check before the holdout peek.
4. **(R7-MEDIUM) §6A contradicted §5.1** (said the cap counts `family_events`, not canonical `families`),
   and the `family_members WHERE removed_at IS NULL` COUNT alone was not a robust digest (a `removed_at`
   UPDATE is not append-only; a same-count swap could hide).
5. **(R7-LOW) `uuid4().hex[:12]` is collision-RESISTANT, not "collision-proof"** as R6 claimed.

**This design (R7) keeps the R4/R5/R6 architecture and closes the R6 BLOCK:** (HIGH-1) the classification
snapshot becomes a `(graph_fingerprint, profile_digest)` pair — the DB fingerprint (authoritative,
under-lock) PLUS a hash of the source-derived member profiles, both captured before==after classification
(§7.1) and re-checked pre-lock (§7.2), backstopped by the additions-only immutable-module discipline
(§2.3); (HIGH-2) the cap is honestly stated as **rate-bounded, not lifetime-finite**, an explicitly
accepted governance residual with a deferred human-replenished-budget escalation (§6A); (HIGH-3) the cap
is **pre-checked in preflight before the holdout peek** and re-checked under the lock (§7.2, §5.1 step
5b); (MEDIUM) §6A is corrected to canonical `families`, and the `family_members` fingerprint covers
all-rows `(COUNT, MAX(id))` AND active-only COUNT so appends AND removals both register (§2.3); (LOW) the
uuid is the full 32-char hex, called **collision-resistant**, with an in-tx regenerate-retry so
correctness never depends on uniqueness (§5.1 step 5c). R0's accepted-scalar mechanics are retained.

**R7 (snapshot+cap-hardening draft) — BLOCKED (1 HIGH + 3 MEDIUM + 3 LOW).** The reviewer **accepted the
#324-grounded founder-tax governance decision as sound** ("global lifetime statistical terms are
anti-scaling; family-scoped lifetime terms are the allowed durable scope"), leaving one real defect and
polish:
1. **(R8-HIGH) Holdout-burn on stale snapshot.** R7 pre-checked only the *cap* before the holdout peek;
   the *snapshot* (fingerprint/profile) was re-checked only in `run_gate` AFTER the peek, so drift
   landing between preflight and `run_gate` still burned a scarce holdout before failing closed.
2. **(R8-MEDIUM) profile_digest / DB-fingerprint rest on non-schema-enforced invariants** (immutable
   member source; append-only tables) — acceptable but the out-of-scope trust boundary must be stated.
3. **(R8-MEDIUM) The cap monitoring/escalation was deferred** without even a defined audit query/owner.
4. **(R8-LOW) `created_at` cap comparison** assumes canonical UTC; **overflow claim** was absolute not
   bounded; **migrated DBs** lack the `seeded_prior_combos >= 0` CHECK.

**This design (R8) keeps the R4..R7 architecture and closes the R7 BLOCK:** (HIGH) the pending-NOVEL
snapshot revalidation (still-unassigned + `graph_fingerprint` + `profile_digest`) is moved into
`promotion_preflight` **before the holdout peek**, next to the cap pre-check, so drift fails closed with
NO holdout burned; the narrow during-`walk_forward` residual is explicitly documented and audited (§7.2).
(MEDIUM) an explicit repair-tool/tree-writer threat-model scope (§2.3) and a concrete cap monitoring
query + owner shipped now (§6A). (LOW) UTC-canonical `created_at` (§5.1 step 5b), bounded (not absolute)
overflow wording with a deferred row-count assertion (§2.4), and an application-level
`seeded_prior_combos >= 0` guard on migrated DBs (§2.1). R0's accepted-scalar mechanics are retained.

**R8 (approve-then-re-audited) — BLOCKED again (multiple HIGH, three independent Codex passes).** R8 got
one APPROVE, but three *independent* adversarial GATE-1 passes (condensed summary; summary + R8's own
append-only/governance-residual arguments; and the FULL verbatim body §1-9 — pure text review, no repo
exploration) each returned **BLOCK** with new *structural* objections. They are not the previously-closed
issues; they are the gaps R8 still left as documentation-only or partially-addressed:
1. **(R9-HIGH-1) The append-only invariant was only DOCUMENTED.** `graph_fingerprint`'s correctness rests
   on the five classifier-read tables being append-only (+ the `removed_at` tombstone) — but nothing
   *mechanically* enforced it, and nothing failed when a new classifier DB read was added without a
   matching fingerprint component. It must be enforced (DB triggers + a guard test), not trusted.
2. **(R9-HIGH-2) The founder budget was rate-bounded, not lifetime-finite, AND deferred.** R8 deferred the
   human-replenished budget to "a future escalation," resting on the `#324`/symmetry argument. Codex's
   independent judgment: full automation changes the throughput profile *qualitatively* vs. the human
   path, so the rate cap alone does not close it; a genuine non-statistical lifetime budget must be built
   NOW, or the residual needs explicit written human product/security sign-off (unavailable in-band).
3. **(R9-HIGH-3) `profile_digest` was a pre-lock-only trust boundary.** The source-derived member profiles
   were re-checked only pre-lock and otherwise trusted the additions-only/GENERATED_BY discipline as an
   out-of-scope boundary — a window the under-lock CAS could not close.
4. **(R9-HIGH-4) The holdout-burn-via-repeated-drift-race was only LOGGED.** R8's during-`walk_forward`
   residual burned a scarce holdout on drift and merely logged it; it needs an actual mitigation
   (reservation release-on-drift or bounded retry-before-burn).
5. **(R9-MEDIUM-1) The cap/window constants' HOME was unstated.** Nothing said the agent/runtime cannot
   self-modify `AGENT_NOVEL_MINT_CAP`/`_WINDOW_DAYS`; they must be a CODEOWNERS-protected constant, not a
   CLI flag or env var reachable by the loop.
6. **(R9-MEDIUM-2) Founders were an audit blind spot.** With `gate_evaluations.family_id` NULL for
   founders and no `founder_gate_id`/`created_family_id` column anywhere, there was no canonical way to
   query which gate/promotion founded a given family.
7. **(R9-MEDIUM-3) Overflow-guard ordering could permanently DoS mints.** R8 computed the integrity SUM
   and the validation in one query wrapped in try/except, so one corrupt overlarge legacy row failed ALL
   future agent NOVEL mints closed forever. Validate/filter before summing (or checked/chunked sum).
8. **(R9-LOW) `created_at` non-canonical-UTC handling + additive-seed dedupe tests.** The cap compare
   assumed canonical UTC with no fail-closed on a non-canonical value; and the additive seed summation
   had no explicit multi-parent/ancestor-dedupe test cases.

**This design (R9) keeps the R4..R8 deferred-pass-time-mint architecture and closes the R8 re-audit
BLOCK by replacing documentation with mechanism:** (H1) DB `BEFORE UPDATE`/`BEFORE DELETE` triggers make
the five classifier-read tables append-only in the engine + a source-scan guard test binds every
classifier read to a fingerprint component (§2.3, §2.5); (H2) a **human-replenished lifetime mint budget**
is built and enforced under the lock, making the founder channel lifetime-finite as a non-statistical
quota (§2.1, §5.1 step 5b, §6A); (H3) member profiles are **DB-persisted at assignment** and read from
the DB, so the profile axis is inside the under-lock `graph_fingerprint` CAS and `profile_digest` is
removed (§2.3, §7.1); (H4) the burn-commit at `on_peek` re-validates the (now fully-DB) snapshot and
releases the reservation for drift caught at/before the burn — **no burn for at-burn drift; the
post-peek run_gate re-check is a narrowed, monitored residual (release-on-failure + WARNING audit), not
fully closed** (§7.2); (M1) the
constants live in CODEOWNERS-protected `store.py`, no flag/env (§5.1 step 5b, §6A); (M2)
`families.founder_gate_id` gives a canonical founder→gate query (§2.1, §6D); (M3) the seed is a
WHERE-filtered SUM over well-typed in-range rows — a corrupt row is excluded, never a permanent DoS
(§2.4, §5.1 step 5a); (L) fail-closed on non-canonical-UTC `created_at` + explicit multi-parent/dedupe
tests (§5.1 step 5b, §8). R0's accepted-scalar mechanics are retained.

---

## 1. What we are building (one paragraph)

On a NOVEL verdict, `promotion_preflight` no longer fail-closes the agent and no longer creates a
family. It classifies the strategy as NOVEL, resolves breadth with **no family arm** (family_lifetime
= 0, identical to today's human-fresh-family founder), and carries a small **pending-NOVEL-family
spec** (auto-slug + clustering metadata) on the breadth context. If — and only if — the strategy then
**passes the gate**, the atomic promote transaction (`record_gate_with_fdr_and_maybe_promote`, which
already holds `BEGIN IMMEDIATE` and does the BACKTESTED→CANDIDATE stage CAS) additionally, in the same
all-or-nothing commit: computes the seed = **true funnel-wide LIFETIME total** (a WHERE-filtered,
overflow-safe SUM) of all `search_trials` under the write lock, INSERTs a `families` row with
`seeded_prior_combos = seed` and `founder_gate_id` = the founding gate row, writes the `family_created`
event, and assigns the strategy as the founding member (its profile DB-persisted). Future siblings
assigned into this family inherit the durable prior, preserving the multiple-testing tax after the
founding evidence rolls out of the 90-day window. Agent founding is bounded by a per-window rate cap AND
a **human-replenished lifetime mint budget** (both fail-closed, CODEOWNERS-constant), and the classifier
snapshot the mint re-checks is now **fully DB state** guarded by DB-engine append-only triggers — so drift
never burns a holdout (the burn-commit atomically re-checks and releases on drift) and never mints a
laundering split. The human `--new-family --actor human` path is unchanged (fresh-zero budget).
`family-audit` stays advisory and is **not** claimed as an enforcement control.

---

## 2. Schema — the injected-scalar prior (Findings R0-4, R1-LOW)

### 2.1 Diff (`algua/registry/db.py`), `SCHEMA_VERSION` 36 → 37

Confirmed the bump is free: `db.py` is at 36 and no other open PR touches it (CLAUDE.md "one bump in
flight"). The v37 bump carries FOUR coordinated changes: (1) `families` gains `seeded_prior_combos` and
`founder_gate_id`; (2) `family_members` gains the persisted member profile `member_code_hash` /
`member_factors_json` (Finding R9-HIGH-3); (3) append-only `BEFORE UPDATE`/`BEFORE DELETE` triggers on
the five classifier-read tables (Finding R9-HIGH-1); (4) a new append-only `agent_mint_grants` ledger for
the human-replenished lifetime budget (Finding R9-HIGH-2). Fresh-DB DDL in `_SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS families (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL UNIQUE,
    created_at          TEXT NOT NULL,
    created_by_actor    TEXT NOT NULL,
    created_by_strategy TEXT,
    seeded_prior_combos INTEGER NOT NULL DEFAULT 0 CHECK (seeded_prior_combos >= 0),  -- v37 (#524)
    founder_gate_id     INTEGER REFERENCES gate_evaluations(id)                       -- v37 (#524, R9-M2)
);
-- v37 (#524, R9-HIGH-3): persisted member profile, materialised at ASSIGNMENT from the (code_hash,
-- factors) the member was classified under. IMMUTABLE once non-NULL (append-only trigger below), so
-- the classifier's member-profile input is DB state, transactionally covered by graph_fingerprint.
CREATE TABLE IF NOT EXISTS family_members (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id       INTEGER NOT NULL REFERENCES families(id),
    strategy_name   TEXT NOT NULL,
    joined_at       TEXT NOT NULL,
    joined_by_actor TEXT NOT NULL,
    removed_at      TEXT,
    member_code_hash    TEXT,   -- v37 (#524): NULL only on un-materialised legacy rows
    member_factors_json TEXT    -- v37 (#524): sorted JSON array of factor names
);

-- v37 (#524, R9-HIGH-2): human-replenished lifetime agent-NOVEL mint budget. APPEND-ONLY ledger of
-- grants; the lifetime allowance = AGENT_NOVEL_MINT_LIFETIME_BUDGET (a CODEOWNERS-protected store.py
-- constant) + COALESCE(SUM(grant_count),0). Only a HUMAN actor may append a grant (a governance quota,
-- NOT a statistical term — so it does not offend #324). Consumed = lifetime COUNT of agent families.
CREATE TABLE IF NOT EXISTS agent_mint_grants (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    granted_at       TEXT NOT NULL,
    granted_by_actor TEXT NOT NULL CHECK (granted_by_actor = 'human'),
    grant_count      INTEGER NOT NULL CHECK (grant_count >= 1),
    reason           TEXT
);

-- v37 (#524, R9-HIGH-1): make the classifier read-set append-only IN THE ENGINE, not just by
-- discipline. families / family_parents / family_events / backtest_returns are pure INSERT-append
-- (confirmed: no UPDATE/DELETE writer in store.py) → forbid both. family_members permits exactly two
-- one-way UPDATEs: the removed_at tombstone flip (NULL→ts) and the one-time legacy profile
-- materialisation (member_code_hash/member_factors_json NULL→value); everything else ABORTs.
CREATE TRIGGER IF NOT EXISTS trg_families_append_only_upd BEFORE UPDATE ON families
  BEGIN SELECT RAISE(ABORT, 'families is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_families_append_only_del BEFORE DELETE ON families
  BEGIN SELECT RAISE(ABORT, 'families is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_parents_append_only_upd BEFORE UPDATE ON family_parents
  BEGIN SELECT RAISE(ABORT, 'family_parents is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_parents_append_only_del BEFORE DELETE ON family_parents
  BEGIN SELECT RAISE(ABORT, 'family_parents is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_events_append_only_upd BEFORE UPDATE ON family_events
  BEGIN SELECT RAISE(ABORT, 'family_events is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_events_append_only_del BEFORE DELETE ON family_events
  BEGIN SELECT RAISE(ABORT, 'family_events is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_backtest_returns_append_only_upd BEFORE UPDATE ON backtest_returns
  BEGIN SELECT RAISE(ABORT, 'backtest_returns is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_backtest_returns_append_only_del BEFORE DELETE ON backtest_returns
  BEGIN SELECT RAISE(ABORT, 'backtest_returns is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_members_no_delete BEFORE DELETE ON family_members
  BEGIN SELECT RAISE(ABORT, 'family_members is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_family_members_append_only_upd BEFORE UPDATE ON family_members
  WHEN NOT (
      -- (a) removed_at tombstone flip, all else unchanged
      (OLD.removed_at IS NULL AND NEW.removed_at IS NOT NULL
         AND NEW.member_code_hash IS OLD.member_code_hash
         AND NEW.member_factors_json IS OLD.member_factors_json
         AND NEW.family_id=OLD.family_id AND NEW.strategy_name=OLD.strategy_name
         AND NEW.joined_at=OLD.joined_at AND NEW.joined_by_actor=OLD.joined_by_actor)
      OR
      -- (b) one-time legacy profile materialisation: NULL→value, nothing else changes
      (OLD.member_code_hash IS NULL AND NEW.member_code_hash IS NOT NULL
         AND NEW.removed_at IS OLD.removed_at AND NEW.family_id=OLD.family_id
         AND NEW.strategy_name=OLD.strategy_name AND NEW.joined_at=OLD.joined_at
         AND NEW.joined_by_actor=OLD.joined_by_actor)
  )
  BEGIN SELECT RAISE(ABORT, 'family_members: only removed_at or one-time profile materialise (#524)'); END;
```

Existing DBs (`migrate()`), ordered so the one-time legacy materialisation is NOT blocked by its own
trigger:
```python
# v37 (#524): the established _add_missing_columns ALTER pattern. Legacy families predate agent-minting
# (human-created, real accumulated lifetime), so seeded_prior_combos DEFAULT 0 / founder_gate_id NULL is
# exactly correct and changes NONE of their family_lifetime values. ALTER cannot add a CHECK; the >=0
# invariant on legacy rows holds by construction (all backfilled to 0) and the ONLY nonzero writer is the
# §5.1 mint (seed = a WHERE-filtered COALESCE(SUM,...) — always >= 0, asserted in app code).
_add_missing_columns(conn, "families", {
    "seeded_prior_combos": "INTEGER NOT NULL DEFAULT 0",
    "founder_gate_id": "INTEGER",
})
_add_missing_columns(conn, "family_members", {
    "member_code_hash": "TEXT",
    "member_factors_json": "TEXT",
})
# Order matters: the CREATE TABLE IF NOT EXISTS bootstrap already made agent_mint_grants; the append-only
# TRIGGERS are created LAST (below), AFTER _materialise_legacy_member_profiles() (a store-layer step that
# CAN load modules) has run its NULL→value UPDATEs, so the one-time legacy backfill is not blocked. Fresh
# DBs have no legacy members, so ordering is moot for them.
```

**Backfill (`seeded_prior_combos`/`founder_gate_id`):** the column defaults backfill every pre-#524 row
to `0`/NULL — a no-op on `family_lifetime` and on founder-audit (legacy families were not agent-founded).
No data rewrite. Idempotent, cross-process safe.

**Legacy member-profile materialisation (Finding R9-HIGH-3).** New assignments persist the profile at
INSERT; pre-#524 member rows have `member_code_hash`/`member_factors_json` NULL. A store-layer one-time
`_materialise_legacy_member_profiles()` (invoked from the store's post-`migrate()` bootstrap, where module
loads are legal) computes each legacy active member's `(code_hash, sorted factors)` via
`compute_artifact_hashes`/`factors_used_by` and writes them with a trigger-permitted NULL→value UPDATE.
It runs BEFORE the append-only triggers are installed (migration order above), so the backfill is not
self-blocked; thereafter the columns are immutable and the classifier reads DB state exclusively — no
dual live-vs-persisted path in steady state (families are nascent, so the legacy set is small).

**Application-level guards on migrated DBs.** SQLite ALTER cannot add `CHECK (seeded_prior_combos >= 0)`
to an upgraded table, so EVERY writer of `seeded_prior_combos` (the §5.1 mint is the only one) asserts
`seed > 0` in application code before the INSERT, and the `lifetime_combos_for_families` reader treats any
negative as a corruption error rather than silently subtracting — the non-negativity invariant holds on
legacy DBs by enforcement, not merely by the fresh-DB CHECK. A `data verify`-style integrity scan over
`families.seeded_prior_combos` is deferred defense-in-depth for a hypothetical out-of-band negative write.

### 2.2 Summation semantics (`algua/registry/store.py`)

`lifetime_combos_for_families` (the sole 3-way-max family arm) gains the per-family seed sum over the
**same ancestor-closure family set** it already unions member strategies over:

```python
def lifetime_combos_for_families(self, family_ids: Iterable[int]) -> int:
    """Lifetime combos across the UNION of families + transitive ancestors: the deduped SUM of
    member strategies' real search_trials PLUS the SUM of each family's injected
    seeded_prior_combos (#524). The seed is an ADDITIVE prior — real member trials accumulate on top,
    so family_lifetime is monotone non-decreasing in both and can never fall below the seed."""
    closure: set[int] = set()
    all_strategies: set[str] = set()
    for fid in family_ids:
        closure.add(fid)
        closure.update(self.family_ancestry(fid))
        all_strategies.update(self._family_member_strategies(fid))
    real = 0
    if all_strategies:
        ph = ",".join("?" * len(all_strategies))
        real = int(self._conn.execute(
            f"SELECT COALESCE(SUM(st.n_combos),0) FROM search_trials st"
            f" WHERE st.strategy_name IN ({ph})", list(all_strategies)).fetchone()[0])
    seed = 0
    if closure:
        fp = ",".join("?" * len(closure))
        seed = int(self._conn.execute(
            f"SELECT COALESCE(SUM(seeded_prior_combos),0) FROM families"
            f" WHERE id IN ({fp})", list(closure)).fetchone()[0])
    return real + seed
```

- Seed summed over the **ancestor closure**, deduped by `set` (a family reachable via several inputs
  counts once) — mirrors `_family_member_strategies` walking `[fid] + family_ancestry(fid)`.
- **Deliberate conservative over-count:** the seed (funnel-lifetime-total captured at pass) *includes*
  the founding strategy's own trials, and the founder is *also* a real member, so its combos count
  twice. This only ever RAISES the bar (monotone-up), never lowers it (§5 proves it). We accept it
  rather than add a subtractive special-case that could underflow the guarantee.
- `family_lifetime_combos(family_id)` (the entry `run_gate` uses) delegates to
  `lifetime_combos_for_families([family_id])`, so it picks up the seed with no further change.

### 2.3 Repository API surface (`algua/registry/repository.py`)

- `SearchBreadthLedger` gains the **true lifetime total** (Finding R0-2):
  ```python
  def funnel_lifetime_search_combos(self) -> int:
      """Sum of n_combos across ALL strategies' search_trials for ALL TIME (no window) — the
      funnel-wide LIFETIME search effort. Distinct from windowed_search_combos (rolling 90d) and
      total_search_combos (per-strategy). Uses the SAME WHERE-filtered, overflow-safe summation as the
      §5.1 mint seed (`typeof(n_combos)='integer' AND n_combos BETWEEN 1 AND MAX_N_COMBOS`), so the seed
      and this accessor agree exactly and a corrupt legacy row is uniformly excluded rather than
      overflowing or coercing. Always >= 0; 0 iff no well-typed in-range rows exist."""
      ...
  ```
- `FamilyRepository` gains a pure-SQL **family-graph fingerprint** (Findings R5-HIGH-1, R6-HIGH-3,
  R6-MEDIUM) — a cheap, monotone digest of **the FULL set of DB inputs the NOVEL classification reads**,
  so a concurrent mutation between preflight and commit is detectable without importing the clustering
  code into the store:
  ```python
  def family_graph_fingerprint(self) -> tuple[int, ...]:
      """A monotone digest over EVERY DB table the classifier reads:
        - families               (COUNT, MAX(id))  -- a new root/child family mint
        - family_members ALL rows (COUNT, MAX(id)) AND active-only (COUNT WHERE removed_at IS NULL)
                                                    -- assignments (append) AND removals (removed_at
                                                       UPDATE): all-rows COUNT/MAX(id) catches every
                                                       INSERT; active-only COUNT catches every removal
                                                       (R6-MEDIUM: `removed_at` is an UPDATE not append)
        - family_parents         (COUNT, MAX(id))  -- parentage/ancestry edges (ancestry closure)
        - family_events          (COUNT, MAX(id))  -- create + assign audit
        - backtest_returns       (COUNT, MAX(id))  -- the return-correlation axis (R6-HIGH-3):
                                                       a member's returns refresh INSERTs a new row
      ANY DB mutation that could re-decide a NOVEL verdict — a family mint, a member assignment OR
      removal, a parentage edge, or a member-returns refresh — bumps at least one component. Cheap (a
      handful of COUNT/MAX scans), boundary-clean (pure SQL, no `algua.research`)."""
      ...
  ```
  - **Robustness — now MECHANICALLY enforced (Finding R9-HIGH-1, was R6-MEDIUM doc-only).** The tuple's
    monotone-digest logic relies on the five tables being append-only (+ the `family_members.removed_at`
    tombstone and the one-time legacy profile materialisation). R9 makes that **an engine invariant, not a
    convention**: the `BEFORE UPDATE`/`BEFORE DELETE` triggers of §2.1 `RAISE(ABORT)` on any other
    mutation, so a hard-DELETE or in-place id/row rewrite is impossible through *any* connection to the DB
    (store API, a repair tool, or a raw `sqlite3` shell) — not merely absent from `store.py`. Given that,
    the digest is exact: **all-rows** `(COUNT, MAX(id))` strictly increases on any INSERT (a new
    assignment always INSERTs a new `family_members` row); the **active-only COUNT** strictly decreases on
    a removal `removed_at` UPDATE; a remove-then-readd bumps all-rows COUNT/MAX(id) — no same-count swap
    can hide, and no DELETE/rewrite can silently move an id. A canonical SHA over the ordered id set is now
    a pure micro-optimisation, not a correctness dependency (the triggers, not the digest shape, guarantee
    append-only), so it is dropped from the deferred list.
  - **Source-derived member profiles are now DB state, inside this fingerprint (Finding R9-HIGH-3 — the
    former `profile_digest` is REMOVED).** Previously the classifier recomputed each member's
    `code_hash`/`factors` from module SOURCE live (`all_families_with_member_profiles`), so the axis was
    non-DB and could only be guarded pre-lock. R9 **persists** the member profile at assignment
    (`family_members.member_code_hash` / `member_factors_json`, §2.1) and `all_families_with_member_profiles`
    reads the persisted (immutable) columns instead of recomputing. Because the profile now lives in the
    `family_members` row and that row is immutable once written (append-only trigger), every new member
    (with its frozen profile) bumps the all-rows `(COUNT, MAX(id))` component, and no existing member's
    profile can change under a live source edit — so **the member-profile axis is fully covered by this
    single under-lock fingerprint**, with no companion digest and no pre-lock-only window. The classifier
    verdict is thus a pure function of DB state (persisted member profiles + returns + graph) plus the
    candidate's OWN live profile, and the candidate's own profile is guarded by the existing #339
    `FunnelSnapshot`/identity hashing (it is being classified, not yet a member).
  - **Residual threat-model scope (much narrower than R8).** The ONLY axis now outside the under-lock CAS
    is the candidate's own live profile (identity-hash-guarded by #339) and the theoretical divergence
    between a member's *persisted* profile and its *current* module source — but the classifier no longer
    reads current source for members, so that divergence cannot flip a verdict; it is a pure
    audit/integrity concern (the additions-only/GENERATED_BY discipline still holds member modules
    immutable). An in-place rewrite of `families`/`family_members`/`family_parents`/`family_events`/
    `backtest_returns` is now **rejected by the DB engine itself** (the triggers), so it is no longer even
    a documented trust boundary — it is a hard error. `#524`'s classifier-drift surface is therefore
    entirely transactional.
- A `PendingNovelFamily` NamedTuple (created in `repository.py`, plain scalars only — no
  `algua.research` import, so the `registry`→`research` boundary stays clean). **No `profile_digest`
  field — the profile axis is now inside `graph_fingerprint`:**
  ```python
  class PendingNovelFamily(NamedTuple):
      slug_base: str          # readable stem; the final UNIQUE name is a uuid-suffixed mint name (§5.1 step 5c)
      actor: str              # MUST be 'agent' — validated at the store boundary (§5.1)
      verdict: str            # MUST be 'novel' — validated at the store boundary (§5.1)
      similarity_score: float
      clustering_version: str
      clustering_config_json: str
      axis_json: str
      graph_fingerprint: tuple[int, ...]   # FULL DB read-set incl. persisted member profiles; captured
                                            # before==after classification (§7.1); re-verified under the
                                            # lock (§5.1 step 1) and at the atomic burn (§7.2)
      founder_code_hash: str               # the founder's OWN classified code_hash — persisted onto the
      founder_factors_json: str            # founding family_members row at mint (§5.1 step 5c), so the
                                            # store never loads modules under the write lock (Finding R9-HIGH-3)
  ```
- `record_gate_with_fdr_and_maybe_promote` (on `GateLedger`) gains one optional param
  `pending_novel_family: PendingNovelFamily | None = None`. When it is set **and** `final_passed` is
  True, the method — inside its existing single `BEGIN IMMEDIATE` — additionally creates the seeded
  family and assigns the strategy (§5). Default `None` keeps every existing caller unchanged.

No standalone `create_seeded_family_and_assign` is introduced (R1's separate top-level method is
obsolete): folding the create+seed+assign into the promote tx is both simpler and the fix for R1's
timing HIGHs.

### 2.4 `n_combos` bounds + filter-before-sum ordering (Findings R6-MEDIUM, R9-MEDIUM-3)

`n_combos` is a per-`search_trials`-row search-breadth count. It is bounded on **both** ends: fresh-DB
`search_trials` DDL `CHECK (typeof(n_combos)='integer' AND n_combos >= 1 AND n_combos <= MAX_N_COMBOS)`;
`record_search_trial` validates `type(n_combos) is int and 1 <= n_combos <= MAX_N_COMBOS` (note
`type() is int`, NOT `isinstance` — `bool` is an `int` subclass, so `isinstance(True, int)` is True and a
stray `True` would store as `1` and pass a `typeof='integer'` check; R8-MEDIUM). `MAX_N_COMBOS =
1_000_000_000` (a per-sweep combo count above any legitimate grid), defined beside the schema constants.

**Overflow-guard ORDERING — the seed is a FILTERED SUM, never a permanent DoS (Finding R9-MEDIUM-3).** R8
computed the SUM and the well-typed validation in one query wrapped in try/except and **failed ALL future
agent NOVEL mints closed** whenever any legacy row was corrupt — so a single overlarge or mistyped legacy
row was a permanent, un-clearable DoS on the whole feature. R9 fixes the ordering: the seed SUM **filters
to well-typed in-range rows in the WHERE clause, so summation only ever runs over safe summands**:
```sql
SELECT COALESCE(SUM(n_combos), 0) FROM search_trials
 WHERE typeof(n_combos)='integer' AND n_combos BETWEEN 1 AND :max_n_combos;   -- overflow-safe: each summand ≤ 1e9
```
A corrupt row (`'abc'`, `2.5`, a BLOB, `≤ 0`, or an over-`1e9` integer) simply fails the predicate and
contributes **0** — it can never overflow the SUM (the overlarge value is never added) and can never
shrink the seed below the sum of the legitimate rows (a corrupt row has no legitimate contribution to
remove; it can only omit *its own* would-be positive term, which merely makes the already-conservative
seed marginally less conservative — never a laundering lever, since it cannot subtract from another
strategy's trials). Corruption is therefore an **observability signal, not a hard block**: the mint still
requires `seed > 0` (an all-corrupt/empty funnel legitimately fails closed — there is no real breadth to
seed), and a separate `(COUNT(*) vs well-typed COUNT)` diagnostic is surfaced for audit (§5.1 step 5a),
but one corrupt legacy row can no longer wedge every future mint. With each summand ≤ `1e9`, overflow
would need `> 2^63/1e9 ≈ 9.2e9` well-typed rows — astronomically beyond any real table (a hard row-count
assertion / chunked checked sum remains the deferred belt-and-suspenders, Finding R7-LOW-6 honesty).

### 2.5 Fingerprint↔classifier-read binding guard test (Finding R9-HIGH-1)

`graph_fingerprint`'s soundness depends on it covering **every** DB table the classification path reads;
a future edit that adds a new classifier DB read without a matching fingerprint component would silently
reopen a drift channel. To make that a build-time failure, a **source-scan guard test** (modeled on the
#277 canonical AST data-wall scanner) statically extracts (i) the set of table names read by the
classification read-path methods (`strategy_family`, `all_families_with_member_profiles`,
`family_ancestry`, and the helpers `_classify_and_assign_family` calls) and (ii) the set of table names
`family_graph_fingerprint`'s SQL touches, and **asserts (i) ⊆ (ii)** — so adding a classifier read of a
new table fails the test until that table is added to the fingerprint. A companion test asserts each of
the five append-only tables has both its `BEFORE UPDATE` and `BEFORE DELETE` trigger present in the live
schema (an accidental trigger deletion fails the suite).

---

## 3. Seed with the TRUE lifetime total, not the rolling window (Finding R0-2)

The seed is `funnel_lifetime_search_combos()` = `SELECT COALESCE(SUM(n_combos),0) FROM search_trials WHERE
typeof(n_combos)='integer' AND n_combos BETWEEN 1 AND MAX_N_COMBOS` (the WHERE-filtered, overflow-safe
lifetime total, §2.4/§5.1 5a), captured **under the promote tx's write lock at commit time**. The issue's own
rationale for a family-lifetime term is that `windowed_total` *forgets* (finite lookback); seeding
from `windowed_total` would bake the forgetting into the "durable" prior (R0-2). A funnel burst 120
days before the promotion has left `windowed_total` but is still in the lifetime SUM, so the seed
captures it — the wait-out-the-window reset is closed. Capturing at **pass/commit** (not preflight)
also means the seed reflects the newest funnel state, closing R1-2's stale-shelf variant.

---

## 4. What the seed buys — the honest claim (Finding R0-3)

Because the agent family is created **only after** the gate passes, it does **not exist** during gate
evaluation. So for the FOUNDING promotion the family arm is `0` and
`n_funnel = max(own, windowed_total, 0) = max(own, windowed_total)` — **exactly today's bar**, and
identical to what a human-created fresh family founder pays. The seed therefore has **zero effect on
the founding pass** (a true no-op, cleaner than R1's "no-op-or-strengthening"): the founding strategy
is *empirically* novel (it survived the return-correlation escalation, §6B), so charging it only the
funnel-wide windowed tax is correct.

The seed's **entire, load-bearing purpose is future siblings.** Once the founding-era tests roll past
the 90-day window, a later sibling's own `windowed_total` has forgotten them; the durable
`family_lifetime = seed + accumulated members` preserves the correction indefinitely, so a sibling can
never promote against a window-forgotten reset. **Stated plainly and confirmed as the intended,
narrower claim:** the seed protects future sibling promotions; for the founder it is a no-op and never
lowers the bar below today's (§5 monotonicity).

---

## 5. Atomicity, timing, and the drift question (Findings R1-1/2/3, R0-4)

### 5.1 Everything happens in ONE existing transaction, at pass-time

`record_gate_with_fdr_and_maybe_promote` already runs top-level under a single `BEGIN IMMEDIATE`
(write lock from the stream-state SELECT through the stage CAS; `store.py:1398-1403`). The new steps
fold in, in order, so the whole promote is one all-or-nothing commit:

1. **Pending-NOVEL unassigned CAS + graph-fingerprint CAS (Findings R3-HIGH, R5-HIGH-1).** If
   `pending_novel_family` is set, under the write lock:
   (a) re-read `strategy_family(name)` and require it is **still `None`** — else the strategy is no
   longer an unassigned NOVEL founder (a concurrent assignment landed); and
   (b) re-read `family_graph_fingerprint()` and require it **equals the classification-stable
   fingerprint** on the pending spec (`pending_novel_family.graph_fingerprint` — captured
   before==after the classifier read, §7.1, so it provably equals the graph the NOVEL verdict was
   computed on, closing R6-CRITICAL). A mismatch means a family was minted, a member (re)assigned, a
   parentage edge added, or a member's returns refreshed since this strategy was classified NOVEL (all
   covered by the widened fingerprint, §2.3), so the NOVEL verdict is **stale** and might now be
   MERGE/PARENTAGE against the changed graph. Blindly minting a second root here is exactly the
   concurrency-driven family-split the review flagged.
   On EITHER (a) or (b) failing, **rollback and raise `FamilyGraphDriftError`** (fail-closed); the CLI
   handles it like the #339 `FunnelDriftError` — **re-run `promotion_preflight` from scratch**, which
   re-classifies against the now-current graph (yielding MERGE/PARENTAGE if a sibling family now
   exists, or a fresh NOVEL with an updated fingerprint if still genuinely novel). Because the member
   profiles are now DB-persisted and covered by `graph_fingerprint` (§2.3, Finding R9-HIGH-3), this
   single under-lock fingerprint CAS is the authoritative TOCTOU-safe check for the **entire** classifier
   read-set — the DB graph AND the member-profile axis; there is no longer a separate source-profile
   compare or a pre-lock-only window. §7.2 keeps a cheap pre-lock mirror in `run_gate` (still-unassigned +
   `graph_fingerprint`) purely as a fast, lock-free early reject; the same fingerprint is re-checked
   authoritatively here under the lock (and once more atomically at the burn instant, §7.2, Finding
   R9-HIGH-4). The candidate's OWN profile is guarded by the existing #339 `FunnelSnapshot`/identity hash.

   Why a fingerprint CAS and not re-running classification under the lock: the classifier lives in
   `algua.research.clustering`; calling it from inside the store's `BEGIN IMMEDIATE` would breach the
   `registry`→`research` import boundary (import-linter) and run heavy correlation math under the write
   lock. The fingerprint is a pure-SQL, boundary-clean proxy: it cannot miss a family mint or member
   assignment (both bump a monotone component, §2.3), and its false-positive mode (a fingerprint change
   that would NOT actually flip the verdict) only costs a cheap preflight replay, never a wrong mint.
2. #339 `FunnelSnapshot` CAS (re-read + verify every mutable funnel field). For a pending-NOVEL
   founder the snapshot carries `family_id = None`, `family_lifetime_effective = 0`; the CAS re-reads
   `strategy_family(name)` → still `None` (guaranteed by step 1) → passes (`0 == 0`).
3. FDR stream read + `final_passed` compute.
4. INSERT the gate row with the **evaluated** breadth, `family_id = None`,
   `family_lifetime_effective = 0` — honest and IMMUTABLE: the family did not exist at evaluation, and
   these columns are the evaluation inputs (they must agree with `n_funnel`/`decision_json`). They are
   **never overwritten** (Finding R3-MEDIUM-1: no post-evaluation stamping). The created family's
   linkage is recorded solely in `family_events` (`family_created` + `strategy_assigned`, both keyed
   by the new `family_id` and the founding `strategy_name` with the same `created_at`), which is the
   authoritative mint audit trail.
5. **If `final_passed` AND `pending_novel_family` is set:** stage CAS BACKTESTED→CANDIDATE, then, under
   the held lock, mint the seeded family through these fail-closed gates, in order:

   **(5a) Filter-before-sum seed + positivity guard (Findings R3-MEDIUM-2, R4-HIGH, R5-HIGH-3,
   R6-MEDIUM overflow, R9-MEDIUM-3 ordering).** The seed SUM **filters to well-typed in-range rows in the
   WHERE clause**, so no corrupt/overlarge row is ever a summand (overflow-safe by construction, §2.4); a
   separate diagnostic COUNT surfaces corruption for audit without wedging the mint:
   ```sql
   -- seed: only ever sums safe summands (each ≤ 1e9) → cannot overflow, cannot go negative
   SELECT COALESCE(SUM(n_combos), 0)
     FROM search_trials
    WHERE typeof(n_combos)='integer' AND n_combos BETWEEN 1 AND 1000000000;   -- → seed
   -- audit diagnostic: how many rows are NOT well-typed (observability, NOT a hard block)
   SELECT COUNT(*),
          COUNT(CASE WHEN typeof(n_combos)='integer' AND n_combos BETWEEN 1 AND 1000000000 THEN 1 END)
     FROM search_trials;                                                       -- → (n_rows, n_well_typed)
   ```
   Require **`seed > 0`**; else **rollback and raise** (a genuinely empty/all-corrupt funnel has no real
   breadth to seed → fail closed). A corrupt row (`'abc'`, `2.5`, BLOB, `≤ 0`, or over-`1e9` integer) is
   *excluded* by the WHERE, contributing 0 — it can neither overflow the SUM nor shrink the seed below the
   sum of the legitimate rows (it has no legitimate term to remove; it can only omit *its own* would-be
   positive contribution, which merely relaxes the already-conservative over-count — never a laundering
   lever, §2.4). This is the R9 ordering fix: **one corrupt legacy row can no longer permanently DoS every
   future agent NOVEL mint** (R8 summed-then-validated in one try/except and failed all mints closed
   forever on any corruption). `n_rows != n_well_typed` is logged as a corruption-observability signal
   (surfaced in the `family-audit` block, §6A) so the operator can clean legacy data, but it does NOT
   block the mint — the security invariant "agent-created ⟹ `seed` is a strictly-positive sum of
   well-typed, in-range funnel trials" holds from the filtered SUM alone, with no reliance on future
   writer validation and no permanent-DoS surface. `record_search_trial`'s `type() is int` bound + the
   fresh-DB DDL CHECK stop NEW corruption at the source (§2.4).

   **(5b) TWO fail-closed authority bounds — a per-window RATE cap AND a lifetime BUDGET (Findings
   R5-HIGH-2, R6-HIGH-2, R9-HIGH-2, R9-MEDIUM-1, R9-LOW UTC).** Both count the **canonical `families`
   table**, not the derived event stream (R6-HIGH-2 — a repair/migration/older helper could create a
   family row without a correctly-shaped `family_created` event and undercount an event-based cap). Both
   fail closed on exceed. Both constants live in **CODEOWNERS-protected `algua/registry/store.py` as
   module constants — NOT a CLI flag, NOT an env var** (Finding R9-MEDIUM-1): the autonomous loop has no
   surface to read or raise them, and changing either requires a human PR to a protected file (the same
   human-gate as the `paper promote`/`research promote` relaxation flags).

   *Rate cap (burst control):*
   ```sql
   SELECT COUNT(*) FROM families
   WHERE created_by_actor='agent' AND created_at >= :cutoff;   -- cutoff = now − AGENT_NOVEL_MINT_WINDOW_DAYS
   ```
   If `>= AGENT_NOVEL_MINT_CAP` → **rollback and raise `AgentMintCapError`**.

   *Lifetime budget (durable finite authority — Finding R9-HIGH-2, the R8 deferral BUILT NOW):*
   ```sql
   SELECT COUNT(*) FROM families WHERE created_by_actor='agent';                 -- lifetime consumed
   SELECT COALESCE(SUM(grant_count), 0) FROM agent_mint_grants;                  -- human-granted top-ups
   ```
   The lifetime allowance = `AGENT_NOVEL_MINT_LIFETIME_BUDGET` (the CODEOWNERS-protected store.py epoch
   constant) `+ SUM(grant_count)`. If `consumed >= allowance` → **rollback and raise
   `AgentMintBudgetExhaustedError`** (fail-closed). This is a **non-statistical governance quota**: unlike
   a durable funnel-lifetime *statistical* founder arm (rejected as anti-scaling per `#324`, §6A), a mint
   budget is a finite *authorization* count, so it does not lower anyone's statistical bar and does not
   offend `#324`. It makes the cross-window founder channel **lifetime-finite** — once the epoch budget is
   spent the agent CANNOT mint another root family until a **human replenishes** by appending an
   `agent_mint_grants` row (a human-actor-only, CHECK-`granted_by_actor='human'` operation; the only way
   to add a grant is the human-only `registry grant-novel-mints --actor human --count N` CLI, which mirrors
   the relaxation-flag authorization pattern). This is R9's answer to the R6-HIGH-1/R7-HIGH-2 "rate-bounded
   not lifetime-finite" objection with a genuine control, not the `#324`/symmetry argument alone.

   *UTC fail-closed (Finding R9-LOW / R7-LOW-5).* `families.created_at` is written by `_now()` =
   `datetime.now(UTC).isoformat()` — canonical UTC ISO-8601 — so the lexicographic `>= :cutoff` compare in
   the rate cap is a correct time comparison. To make that robust rather than assumed, the rate-cap read
   **parses each counted agent row's `created_at` and fail-closes (`AgentMintCapError`) if any does not
   parse as a canonical UTC datetime** (offset-aware, `+00:00`), so a stray local/naive/malformed
   timestamp can never silently mis-bucket a row across the cutoff. A boundary test asserts the `== cutoff`
   row is included, a one-second-earlier row excluded, and a non-canonical timestamp fails closed.

   These under-lock checks are the race-safe authority; the cheap **preflight pre-check (§7.2) already
   refused BOTH conditions BEFORE the holdout peek** (Finding R6-HIGH-3), and the atomic re-check at the
   burn instant (§7.2, Finding R9-HIGH-4) means a holdout is never even burned on a cap/budget breach.
   When either fires, further NOVEL founding requires human action — the rate cap by waiting out the
   window, the budget by a human grant.

   **(5c) Collision-RESISTANT mint (Findings R3-LOW, R4-MEDIUM slug-DoS, R6-MEDIUM/LOW probe-DoS).** The
   mint name is `f"{pending_novel_family.slug_base}__{uuid4().hex}"` — the FULL 32-char uuid on the
   readable stem, NOT a bounded `base_2/base_3` probe (R6-MEDIUM: an exhausted probe bound is still a DoS
   on a *passed* gate). A random uuid suffix cannot be pre-reserved by an adversary, so a passed gate is
   not aborted by pre-reserved deterministic stems; the readable founder back-reference is preserved
   regardless in `created_by_strategy` and the `family_created` event. It is **collision-RESISTANT, not
   collision-proof** (R6-LOW honesty): to make correctness independent of uuid uniqueness, on the
   astronomically-unlikely residual `families.name` UNIQUE violation the mint **regenerates the uuid and
   retries** (bounded small N, still inside the one `BEGIN IMMEDIATE`); only after N failed regenerations
   does it abort the tx fail-closed. Then, using **direct raw INSERTs under the held lock — NOT the public
   `create_family` / `assign_strategy_to_family` helpers, which open their own `with self._conn:`
   transactions and would break the single `BEGIN IMMEDIATE`** (Finding R3-LOW): INSERT
   `families(name=<uuid-suffixed>, seeded_prior_combos=seed, created_by_actor='agent',
   created_by_strategy=name, founder_gate_id=<the gate row id from step 4>)` — the founding
   `gate_evaluations` row was just INSERTed in step 4 of this same tx, so its `lastrowid` is available to
   stamp as `founder_gate_id` (Finding R9-MEDIUM-2: this is the canonical, queryable founder→gate link
   that closes the audit blind spot left by the gate row's own `family_id` staying NULL — §6D); INSERT the
   `family_created` event **populating `strategy_name` with the founder** (audit clarity, R4-LOW-2); INSERT
   the founding `family_members` row **with `member_code_hash`/`member_factors_json` set to the founder's
   classified profile** (from the pending spec / recomputed for the founder — Finding R9-HIGH-3, so the
   founder is DB-materialised from birth like every future member); INSERT the `strategy_assigned` event.

**Store-boundary guard + actor coercion + pending-object validation (R4-LOW-1, R5-MEDIUM
actor-ordering, R6-LOW).** At method entry, `record_gate_with_fdr_and_maybe_promote` **first coerces
`actor = Actor(actor)`** (callers may pass a raw string), and only THEN, when `pending_novel_family is
not None`, fail-closes unless ALL hold: `actor is Actor.AGENT` **and**
`pending_novel_family.actor == Actor.AGENT.value` (`== 'agent'`) **and**
`pending_novel_family.verdict == 'novel'` (the string literal — the store must NOT import
`algua.research.SimVerdict`, so the expected verdict is a plain-string constant, keeping the
`registry`→`research` boundary clean; the value is asserted equal to `SimVerdict.NOVEL.value` by a test
that lives on the `research` side). Coercing before the identity check is
load-bearing: an `is Actor.AGENT` identity test against an un-coerced string would mis-evaluate (a
legitimate `'agent'` string would wrongly fail closed, and — worse if the assert were ever loosened to
`==` — a crafted string could slip through). Validating the pending object's OWN `actor`/`verdict`
fields (R6-LOW) stops an internal caller that constructs an inconsistent spec (e.g.
`PendingNovelFamily(actor='human', verdict='merge', …)`) from writing mismatched audit/classification
metadata. The mint is an agent-only capability and this transaction method is the safety boundary, not
just the (trusted) caller path; any inconsistency is a caller bug and fail-closes.

A process crash before commit rolls back the gate row, the stage CAS, AND the family create — no
half-promoted strategy, no orphan family, no orphan membership. **Finding R1-1** (economic gating):
the family is minted only after a pass, which required a measured sweep and a burned holdout — minting
is now genuinely rate-limited by the loop's scarcest operation. **Finding R1-2** (stale seed): the
seed is captured at commit, never in preflight, so it is always the current funnel lifetime — there is
no shelf inventory and no stale-seed sibling attack. **Finding R1-3** (clustering anchors): a failed
attempt commits no family and no membership, so an unpromoted strategy is never an active exemplar.

### 5.2 The seed cannot drift and need not equal the promotion's `windowed_total`

The seed is a distinct quantity from the founder's `windowed_total` (lifetime ≥ window) — by design.
Once written it is **immutable**. It is born inside the same commit that promotes, so no window exists
between its capture and its first use. Future re-reads (`family_lifetime_combos` in a later sibling's
`run_gate`, and that sibling's own #339 CAS) see `seed + member_trials`; the seed component never
moves, and the member component is already CAS-guarded by `search_trials_fingerprint`. So **no new
TOCTOU and no new drift check are introduced.**

### 5.3 Concurrency and idempotency

- **Two concurrent NOVEL promotes (different strategies):** each `record_gate_with_fdr_and_maybe_
  promote` serializes on the write lock; each captures its seed under the lock (monotone in
  serialization order, append-only table). **The outcome is now verdict-dependent, not "always distinct
  families" (corrected per R5-HIGH-1):** whoever commits first mints its family and bumps the graph
  fingerprint; the second, on taking the lock, sees a fingerprint mismatch (§5.1 step 1b) and
  fail-closes with `FamilyGraphDriftError` → its preflight re-runs and re-classifies against the
  now-existing first family. If the two strategies are genuinely independent, the replay re-confirms
  NOVEL (against a graph that now includes the first family) and mints a distinct second family; if they
  are related variants, the replay yields MERGE/PARENTAGE and the second joins/inherits instead of
  minting a laundering split. Distinct auto-slugs are still collision-resistant (§5.1 step 5c); the
  fingerprint CAS — not slug uniqueness — is what prevents the concurrent split.
- **Same-strategy re-promote:** idempotent by the **stage guard**, not a bespoke early-return. After a
  pass the strategy is CANDIDATE; `promotion_preflight` rejects any source stage ≠ BACKTESTED
  (`promotion.py:319`), so it cannot double-promote and thus cannot double-mint. A *failed* attempt
  leaves the strategy BACKTESTED and mints nothing, so a retry is clean.

---

## 6. Abuse analysis for the new agent minting authority (Finding R0-5, R1-MEDIUM)

**A. Runaway minting AND repeated-founder laundering — enforced controls (Finding R5-HIGH-2).**
Two distinct threats share this analysis:

*Spam / governance DoS.* Minting happens only on a **gate PASS**, which requires (i) a measured sweep
(`total_search_combos(name) > 0`) and (ii) a burned single-use holdout interval (#205, burn-once). Each
family costs one genuinely-passing, properly-powered thesis; idempotency is enforced by the stage guard
(§5.3); every mint writes a `family_created` row (`actor='agent'`, `created_by_strategy`) so history is
fully queryable.

*Repeated-founder breadth laundering (the R5-HIGH-2 hole).* The founder always pays `family_lifetime =
0` (§4), and the seed taxes only *future siblings within that one family*. So an agent could try to mint
**N one-founder families** rather than N siblings of a single family, resetting the family arm each time.
Two layers bound this:
- **Within a rolling 90-day window it is already taxed** by the funnel-wide `windowed_total` term:
  `n_funnel = max(own, windowed_total, family_lifetime)` and `windowed_total` sums *all* strategies'
  sweeps in the window, so every founder's bar already includes every other recent founder's search
  effort. The channel only opens for founders spaced **>90 days apart**, and each such founder still
  costs a measured sweep + a burned holdout + a genuinely-passing thesis + the wall-clock wait.
- **Across windows the RATE is bounded by the load-bearing per-window mint cap** (§5.1 step 5b): at most
  `AGENT_NOVEL_MINT_CAP` agent family mints per rolling `AGENT_NOVEL_MINT_WINDOW_DAYS`, enforced by
  counting **canonical `families` rows** (`created_by_actor='agent'`, NOT the derived `family_events`
  stream — Finding R6-HIGH-2), fail-closed on exceed. This rate-bounds the cross-window reset channel that
  `windowed_total` cannot see.
- **Over a LIFETIME the total is bounded by the human-replenished mint budget** (§5.1 step 5b, Finding
  R9-HIGH-2 — the R8 deferral BUILT NOW): the agent may mint at most `AGENT_NOVEL_MINT_LIFETIME_BUDGET +
  SUM(agent_mint_grants.grant_count)` root families over all time, fail-closed on exceed. This is the
  genuine non-statistical control the R8 re-audit demanded: it makes cross-window founding **lifetime-
  finite**, not merely rate-bounded — the rate cap throttles bursts, the budget caps the total, and once
  the budget is spent a **human must replenish** (append an `agent_mint_grants` row via the human-only
  `registry grant-novel-mints` CLI) before the agent can found again. Because a budget is a finite
  *authorization count*, not a statistical bar, it does not lower anyone's promotion threshold and so does
  not reintroduce the `#324` anti-scaling pathology (contrast a durable funnel-lifetime *statistical*
  founder arm, still rejected below).

Default sizing and HOME (Finding R9-MEDIUM-1): `AGENT_NOVEL_MINT_WINDOW_DAYS = 90` (matches the funnel
window the durable seed exists to outlast), `AGENT_NOVEL_MINT_CAP = 8` (a legitimate genuinely-novel-
thesis cadence, a hard ceiling on burst churn), and `AGENT_NOVEL_MINT_LIFETIME_BUDGET = 32` (a generous
multi-window epoch allowance before a human sign-off is required). All three are **module constants in
CODEOWNERS-protected `algua/registry/store.py`** — there is **no CLI flag and no environment variable**
that the autonomous loop can read or set to change them; raising any of them requires a human PR to a
protected file (the same human-gate as the platform's other relaxation flags), and topping up the
lifetime budget requires a human-actor `agent_mint_grants` append. The agent/runtime therefore cannot
self-widen any bound. The *mechanism* (fail-closed + CODEOWNERS-gated + human-replenished) is the
load-bearing part; the numeric defaults are tunable by a human.

**Concrete monitoring control shipped WITH this PR (Finding R7-MEDIUM, extended for R9).** THIS PR ships
a real, callable surface — an `agent_novel_mints` block on the existing `family-audit` advisory JSON
output (which already scans the family DAG read-only), reporting: `mints_in_window` + `window_cap` +
`window_days` (rate-cap headroom), `lifetime_consumed` + `lifetime_allowance` (`= budget + SUM(grants)`)
+ `lifetime_remaining` (budget headroom — the operator's signal that a human grant will soon be needed),
and `search_trials_corruption_count` (`n_rows − n_well_typed`, the §5.1 step-5a observability signal so
legacy data corruption is visible without blocking mints). All are machine-readable from the CLI the
moment the feature lands. Owner: the human operator running the autonomous loop (same actor who holds the
CODEOWNERS-protected constants and the human-only grant CLI). Sustained rate-cap saturation OR a small
`lifetime_remaining` is the signal to investigate the founder stream and, if the theses are genuine, to
append an `agent_mint_grants` grant; the ops-dashboard row + alert threshold on top is the small deferred
wiring.

*Why the rate cap is the RIGHT answer, and a durable funnel-lifetime founder tax is deliberately
rejected (Finding R6-HIGH-1 — the honest resolution).* R5's cap was challenged as "only a rate limit,
not a statistical reset fix": each singleton-family founder still pays `family_lifetime = 0`. The
tempting "fix" — give every agent-NOVEL founder a durable arm equal to the funnel-LIFETIME total (so
repeated founders across windows all pay a growing, never-forgetting bar) — is **deliberately NOT
adopted, because it re-introduces the exact anti-scaling pathology `#324` exists to prevent.** The
platform's settled design is: **global multiple-testing terms are WINDOWED (bounded), and only
FAMILY-SCOPED terms are lifetime (also bounded, to a member set).** `#324` replaced the lifetime LORD++
stream with count-partitioned cohort restarts precisely because a single lifetime stream let a dry spell
of failed attempts drive everyone's bar toward zero ("testing more garbage lowered everyone's bar"); the
breadth-deflation 3-way max mirrors this by using `windowed_total` (not funnel-lifetime) for its global
arm. A funnel-lifetime founder arm would make the founding bar grow without bound in total funnel
throughput — anti-scaling, and inconsistent with `#324`. Two further points make the fresh-zero founder
correct rather than a gap: (i) **it is symmetric with the human fresh-zero founder** — an agent founding
a NOVEL family pays *exactly* what a human `--new-family` founder pays today (`max(own, windowed_total)`),
so this feature grants the agent no cheaper founding bar than the already-accepted human path; (ii) the
founder is *empirically* novel (survived the return-correlation escalation, §6B), so charging it the
funnel-WINDOWED tax is the honest multiple-testing price for the tests actually competing in its window.
The repeated-founder channel is therefore **rate-bounded** by a **layered, platform-consistent** control
set — not a single durable term: within-window `windowed_total` taxes co-window founders; the
**load-bearing per-window cap** bounds the cross-window *rate* to `CAP` mints/window (fail-closed,
human-only to raise); family-scoped seeds durably tax *siblings*; and the `families` audit trail monitors
mint history.

**The former residual is now CLOSED by a built control (Finding R9-HIGH-2, was the R6-HIGH-1/R7-HIGH-2
accepted residual).** R8 accepted a *rate-bounded-but-not-lifetime-finite* residual, arguing the
`#324`/human-symmetry case made it acceptable and deferring the lifetime budget. The R8 re-audit rejected
that: **full automation changes the throughput profile qualitatively** vs. the human path, so the rate
cap + symmetry argument alone do not close the channel. R9 therefore **builds the non-statistical lifetime
mint budget now** (§5.1 step 5b, §2.1): the agent's total lifetime root-family mints are hard-capped at
`AGENT_NOVEL_MINT_LIFETIME_BUDGET + SUM(grants)`, fail-closed, replenished only by a human-actor grant.
This makes the cross-window founder channel **lifetime-finite**, not merely rate-bounded — the exact
control the reviewer asked for, and one that does NOT offend `#324` (a finite authorization count is not a
statistical bar). The layered set is now: within-window `windowed_total` taxes co-window founders; the
**per-window rate cap** throttles bursts; the **lifetime budget** caps the total and forces periodic human
sign-off; family-scoped seeds durably tax *siblings*; and the `families`/`agent_mint_grants` audit trail
(surfaced in `family-audit`, §6A monitoring) makes the whole thing observable. This is a closed control,
not an accepted residual. (A durable funnel-lifetime *statistical* founder arm remains deliberately
rejected as anti-scaling per `#324`, above — the budget is the correct non-statistical alternative.)

**B. Relabel laundering.** NOVEL is reached only after `_rank(escalate=True)` (the return-correlation
re-rank, `promotion.py:196-198`) fails to find any family ≥ PARENTAGE_THRESHOLD. A disguised clone
trades like its origin and is rescued into MERGE/PARENTAGE by that escalation — it never reaches the
NOVEL mint path. The empirical classifier is the enforced anti-laundering control; a NOVEL verdict is
positive evidence of independence.

**C. `family-audit` is advisory only — NOT cited as a backstop.** Per `CLAUDE.md:104` it is
read-only (no graph mutation / ledger writes / transitions). The enforced controls are (A)
pass-gated + idempotent minting, (B) the empirical classifier, and the seed (no breadth reset).
`family-audit` remains a monitoring aid; it will now also observe seeded families' breadth (a
harmless, consistent consequence of §2.2, since it reads the same `lifetime_combos_for_families`).

**D. Human fresh-zero override — the honest audit invariant (R1-MEDIUM / R3-MEDIUM-2).** The human
`--new-family --actor human` path keeps `seeded_prior_combos = 0`. The R1 claim that
`seeded_prior_combos = 0 AND created_by_actor = 'human'` "returns exactly" the deliberate grants was
**false** — it also matches legacy human families backfilled to 0. We drop that claim. The
security-relevant invariant is instead: **every agent-created family has `seeded_prior_combos > 0`**.
This is not *assumed* from `search_trials` shape (R3-MEDIUM-2 correctly noted `n_combos` has no CHECK
and `record_search_trial` does not validate it today; R4-HIGH noted a *legacy negative* row could
shrink the `SUM` so a too-small seed still passes `seed > 0` and launders the tax) — it is
**ENFORCED** by the §5.1 step-5 mint guard, which computes `(SUM, MIN, COUNT)` under the lock and
**fail-closes unless `n_rows == n_well_typed` (every row a genuine `typeof='integer' AND >=1`) AND
`seed > 0`** (R5-HIGH-3 type-safe form — see §5.1 step 5a; strictly stronger than R4's `MIN < 1`, which
SQLite dynamic typing could evade with a TEXT/BLOB/REAL row), so the seed is provably the sum of a
fully well-typed, strictly-positive row set — no reliance on future writes. As complementary hardening
in the same PR: `record_search_trial` validates `type(n_combos) is int and 1 <= n_combos <= MAX_N_COMBOS` (`type() is
int` excludes `bool`, an `int` subclass — R8-MEDIUM) and the
fresh-DB `search_trials` DDL gains `CHECK (typeof(n_combos)='integer' AND n_combos >= 1)` (these stop
NEW corruption; the mint-time integrity guard is the load-bearing control that also covers pre-existing
legacy rows). Thus no agent driving the platform **through the CLI** (the CLAUDE.md contract — "drive
the system through `uv run algua …`; never reach into modules to bypass the CLI") can ever obtain a
zero-prior (reset) or corruption-undersized family: the only agent-reachable path to a family is the
gated `research promote` NOVEL mint, which always seeds `>0`. The lower-level
`SqliteStrategyRepository.create_family` / `assign_strategy_to_family` helpers still accept an
arbitrary `actor` string with no caller-identity enforcement, so a script that *bypasses the CLI* to
call them directly could pre-seat a strategy into a zero-prior family — but that is a pre-existing
property of those helpers (the same bypass has existed for MERGE/PARENTAGE assignment since #222, and
is not a #524 regression), squarely outside the CLI trust boundary this design defends. Hardening the
store helpers to enforce actor identity themselves is a separate, cross-cutting follow-up.
Zero-seed families are exactly the human-authorized ones (legacy
*or* deliberate fresh grant) — both legitimately carry no prior because a human vouched. The
legacy-vs-deliberate distinction is not security-relevant (both are human-authorized); if a future
audit needs it, the `family_created` event's `created_at` (post-migration) and `created_by_strategy`
(non-NULL only for a strategy-triggered grant) already separate them without new schema.

**E. Human-NOVEL create-in-preflight asymmetry — documented, accepted (Finding R5-MEDIUM).** The AGENT
path defers creation to pass-time (§5.1), so a *failed* agent NOVEL attempt mints nothing and never
becomes a clustering anchor (§5.1, Finding R1-3). The HUMAN `--new-family` path is unchanged and still
creates the root family **in preflight** (§7.1). This is asymmetric: a human NOVEL attempt that later
fails its gate leaves a family + member row on the shelf, and — because
`all_families_with_member_profiles` reads active members — that failed human founder can act as a
classification anchor for *future* agent verdicts (nudging a later strategy toward MERGE/PARENTAGE with
it). We **accept and document** this rather than change human behavior in this PR: (i) it is exactly
today's pre-#524 human behavior (no regression); (ii) it is human-authorized — a human operator vouched
for creating that family and bears the shelf-inventory/anchor cost; (iii) the anchor only ever
*tightens* future classification (pulls toward an existing family / higher breadth), never mints or
resets breadth, so it cannot be a laundering lever. Symmetrizing the human path to pass-time creation
(preserving fresh-zero authority) is a deferred option (§9), not required to close #524.

**F. Founder→gate audit link — no silent blind spot (Finding R9-MEDIUM-2).** The founding gate row is
honestly and immutably stored with `family_id = NULL` (the family did not exist at evaluation, §5.1 step
4), so `gate_evaluations` alone cannot answer "which gate founded family X?" or "did gate Y found a
family?". R9 closes that with the queryable `families.founder_gate_id` FK, stamped at mint from the
step-4 gate row (§5.1 step 5c). The two canonical audit queries are now:
```sql
-- which gate/promotion founded this family?
SELECT founder_gate_id FROM families WHERE id = :family_id;
-- did this gate found a family, and which one?
SELECT id FROM families WHERE founder_gate_id = :gate_evaluation_id;
```
So the founder linkage is fully reconstructable from canonical tables — not only from the `family_events`
audit stream — closing the blind spot the NULL `gate_evaluations.family_id` would otherwise leave. Legacy
and human-created families carry `founder_gate_id = NULL` (they were not agent-founded via a gate), which
is itself the honest, queryable distinction.

---

## 7. Wiring (`algua/registry/promotion.py`)

### 7.1 `_classify_and_assign_family` — NOVEL agent arm no longer creates

The helper returns a small result carrying the resolved `family_id` **and** an optional
`PendingNovelFamily`:

```python
# NOVEL verdict
if actor is Actor.AGENT:
    if not has_any_family:
        raise ValueError(... empty registry: a human must create the first family ...)  # unchanged
    # #524: do NOT create here. Defer to the atomic pass-moment. Return no family_id + a pending spec.
    # slug_base is a readable stem; the collision-RESISTANT UNIQUE name is uuid-suffixed at mint (§5.1 5c).
    # The CLASSIFICATION SNAPSHOT is a SINGLE graph_fingerprint (R9-HIGH-3: member profiles are now
    # DB-persisted, so the whole classifier read-set — graph + member profiles + returns — is one DB
    # digest; there is NO separate source-profile digest). It MUST equal the state the NOVEL verdict was
    # computed on (Finding R6-CRITICAL): the helper captured fp_before at the TOP (before/at the classifier
    # read) and HERE re-reads fp_after and requires fp_before == fp_after. A mismatch means the DB graph
    # (incl. a member's persisted profile via a new assignment) mutated DURING classification, so the
    # verdict is already stale — raise FamilyGraphDriftError (CLI re-runs preflight). Because members'
    # profiles are read from the immutable persisted columns (all_families_with_member_profiles, §2.3), a
    # live member-source edit CANNOT flip the verdict at all — the classifier no longer reads live source
    # for members — so no source-drift axis remains.
    fp_after = repo.family_graph_fingerprint()
    if fp_after != fp_before:
        raise FamilyGraphDriftError(name, axis="graph_fingerprint")  # changed mid-classification; re-run
    pending = PendingNovelFamily(
        slug_base=f"{name}_family", actor=actor.value, verdict=best_verdict.value,
        similarity_score=best_score, clustering_version=cv,
        clustering_config_json=clustering_config_json, axis_json=axis_json,
        graph_fingerprint=fp_after,
        # the founder's OWN classified profile — persisted onto the founding family_members row at mint
        # (§5.1 5c) so the store never loads modules under the write lock:
        founder_code_hash=strategy_code_hash,
        founder_factors_json=json.dumps(sorted(strategy_factors)))
    return ClassifyResult(family_id=None, pending_novel_family=pending)
else:
    # Human: fresh-zero root family created in preflight (unchanged pre-existing behavior). The human
    # create path also persists the founding member's profile columns (§2.1) at assignment.
    if new_family_slug is None:
        raise ValueError(... provide --new-family <slug> ...)
    new_fam_id = repo.create_family(new_family_slug, actor=actor.value, created_by_strategy=name)
    _do_assign(new_fam_id, matched_family_id=None)
    return ClassifyResult(family_id=new_fam_id, pending_novel_family=None)
```

- MERGE and PARENTAGE arms return `ClassifyResult(family_id=..., pending_novel_family=None)` unchanged,
  and their assignment INSERT now also persists the joining member's `member_code_hash`/`member_factors_json`
  (§2.1) so every new member is DB-materialised.
- **Human NOVEL stays create-in-preflight** (pre-existing, human-authorized behavior — a human accepts
  the shelf-inventory/anchor cost, exactly as today). Only the AGENT path is deferred. This divergence
  is intentional and documented.

### 7.2 `promotion_preflight` and `run_gate`

- `promotion_preflight` puts `ClassifyResult.pending_novel_family` on the `BreadthContext` (new field,
  default `None`) and sets `ctx.family_id = ctx.expected_family_id = ClassifyResult.family_id` (None
  for a pending agent NOVEL). Breadth resolution is unchanged: `effective_funnel_breadth(own,
  windowed_total)` (family arm defaults 0).
- **Early pre-peek pending-NOVEL revalidation — bounds AND snapshot, BEFORE the holdout is touched
  (Findings R6-HIGH-3, R7-HIGH-1).** `_classify_and_assign_family` runs in `promotion_preflight`
  *before* the walk-forward/holdout peek (`promotion.py:286` "runs BEFORE walk_forward, so every hard
  refusal happens before the holdout"). Immediately after it returns a `pending_novel_family`, and as
  the **last preflight step before returning to the caller**, preflight re-validates the pending-NOVEL,
  in one place:
  1. **Rate cap AND lifetime budget:** the §5.1 step-5b queries — `COUNT(*) … created_at >= cutoff` vs
     `AGENT_NOVEL_MINT_CAP` (→ `AgentMintCapError`) and lifetime `consumed >= budget + SUM(grants)` (→
     `AgentMintBudgetExhaustedError`), with the same UTC fail-closed parse.
  2. **Still-unassigned + snapshot:** require `repo.strategy_family(name) is None` AND
     `repo.family_graph_fingerprint() == pending.graph_fingerprint` (one DB digest now covers the whole
     classifier read-set incl. persisted member profiles — no separate `profile_digest`); any mismatch →
     raise `FamilyGraphDriftError(axis=…)`.
  All are **clean pre-peek refusals that burn NO holdout and mint NO consumable gate row**. The §5.1
  step-1/step-5b under-lock checks remain the authoritative race-safe re-checks.
- **The holdout-burn-on-drift window is NARROWED at the burn instant (Finding R9-HIGH-4, shrinks — does NOT
  fully close — the R8 during-`walk_forward` residual).** R8 left a residual: drift landing *during* `walk_forward`
  (after the pre-peek check, while the holdout metric is computed) still burned the holdout and was only
  logged. R9 closes it by exploiting the existing reserve→run→finalize holdout architecture (#161): the
  burn commits at the `on_peek` callback (the instant before the holdout metric is read), and the CLI
  **wraps `on_peek` so that, in the SAME transaction that finalises the reservation, it first re-runs the
  pending-NOVEL revalidation (still-unassigned + `graph_fingerprint` + cap + budget — now ALL pure DB
  reads thanks to R9-HIGH-3)**; on any drift it raises `FamilyGraphDriftError`/`AgentMintCapError`/
  `AgentMintBudgetExhaustedError` **before** `finalize_holdout_reservation` runs, so the reservation stays
  **pending** and the existing `except`-clause `release_holdout_reservation` frees the window — **no
  holdout burned** for drift caught at/before the burn. RESIDUAL, honestly scoped (not "no residual race"):
  run_gate runs its OWN pre-lock pending-NOVEL re-check AFTER `walk_forward` returns — i.e. AFTER `on_peek`
  already committed the burn — so drift that first becomes visible in that post-peek/pre-lock window burns
  the holdout and then fails the gate. That path is given the SAME release-on-failure discipline as
  `walk_forward` (a post-burn no-op `release_holdout_reservation`) PLUS an explicit WARNING audit record
  (`holdout_burned_post_peek_gate_failed`), so the narrowed race is observable and monitored, never
  silently claimed as closed — consistent with how R7/R8 residuals were required to be explicitly scoped.
  Fully closing it would require folding `finalize_holdout_reservation` into the same atomic tx as the mint
  (`record_gate_with_fdr_and_maybe_promote`); deferred as out of scope for this round. This is the reviewer's
  asked-for "reservation semantics releasable on drift," not logging-only. (The subsequent under-lock §5.1
  step-1/5b checks in the promote tx remain as the final authority for anything landing between the burn and the
  promote commit; those, if they fire, roll back the whole promote with no family and no consumable gate
  row — and, since the reservation already committed at `on_peek`, that is the one narrow case a holdout
  is spent, now reduced to "drift in the microseconds between the atomic burn and the promote-tx lock,"
  which the guard logs by strategy name + drift axis for observability.)
- **`FamilyGraphDriftError` carries the DRIFT AXIS (R8-LOW)** — an enum `{still_assigned,
  graph_fingerprint}` (the `profile_digest` axis is gone — it folded into `graph_fingerprint`);
  `AgentMintCapError` and `AgentMintBudgetExhaustedError` are the two bound axes — so a recurring
  drift/abort pattern is diagnosable (a concurrent assignment vs. a DB-graph mutation vs. a bound hit).
- `run_gate` passes `breadth.pending_novel_family` through to
  `record_gate_with_fdr_and_maybe_promote(...)`. The founder's `family_lifetime_effective` is `0` and
  `family_id` is `None` at evaluation (family not yet born).
- **CAS fix for the pending case (Findings R3-HIGH, R5-HIGH-1).** Today `run_gate`'s family CAS is
  `if breadth.expected_family_id is not None and family_id != breadth.expected_family_id: raise`
  (`promotion.py:424`) — it *skips* the check when `expected_family_id is None`, which R2 made a
  meaningful value. Change it so a pending-NOVEL request enforces the invariants pre-lock: when
  `breadth.pending_novel_family is not None`, require (a) `repo.strategy_family(name) is None`
  (still-unassigned) AND (b) `repo.family_graph_fingerprint() ==
  breadth.pending_novel_family.graph_fingerprint` (the full classifier read-set unchanged since
  classification — DB graph AND persisted member profiles, R9-HIGH-3); on any failure, raise so the CLI
  re-runs `promotion_preflight`. Otherwise keep the existing `expected_family_id` CAS for the
  assigned/None-non-pending cases. This is a fast, lock-free reject; the authoritative re-checks are §5.1
  step 1 (a+b) under the write lock and the atomic `on_peek` re-check above — all pure DB, all
  serializable, with no non-DB axis remaining.

### 7.3 `CLAUDE.md`

Update the `research promote` NOVEL bullet: "NOVEL → the agent auto-creates a **seeded** root family
**at the pass moment** (funnel-lifetime prior, atomic with the CANDIDATE transition; no breadth
reset, nothing minted on a failed gate), subject to a **per-window agent mint cap** AND a **human-
replenished lifetime mint budget** (both fail-closed on exceed; CODEOWNERS-constant, human-only to raise;
top up the lifetime budget via `registry grant-novel-mints --actor human --count N`) and a **still-NOVEL
graph re-check** (a concurrent family mint/member change since classification fails closed and
re-classifies). `--new-family --actor human` remains a human privilege granting a **fresh-zero** budget."
Also add a one-line entry for the new human-only `registry grant-novel-mints` command to the command
surface.

---

## 8. Test plan (Finding R0-6 / R1-LOW, plus the R1-timing regressions)

Files: `tests/test_family_registry.py`, `tests/test_registry_db.py`,
`tests/test_cli_research_promote_pit.py`, `tests/test_cli_research.py`, `tests/test_concurrency.py`.

1. **Schema migration/backfill (R9: columns + triggers + grants table).** Fresh DB: `seeded_prior_combos`
   `NOT NULL DEFAULT 0` CHECK present, `founder_gate_id` present, `family_members.member_code_hash`/
   `member_factors_json` present, `agent_mint_grants` table with the `granted_by_actor='human'` +
   `grant_count>=1` CHECKs, all ten append-only triggers present, `PRAGMA user_version = 37`. Legacy DB
   (families/members rows, no new columns): `migrate()` adds them, every legacy family reads
   `seeded_prior_combos=0`/`founder_gate_id=NULL` with unchanged `family_lifetime_combos`, legacy member
   profiles are materialised (non-NULL after the store bootstrap) BEFORE the triggers install; second
   `migrate()` is a no-op.
2. **True-lifetime seed vs rolling window (adversarial old trials).** Record `search_trials` at
   `created_at = now − 120d` (outside 90d) + some in-window. Agent promotes a passing NOVEL strategy.
   Assert the created family's `seeded_prior_combos == funnel_lifetime_search_combos()` == the FULL
   sum (in+out of window) and `> windowed_search_combos(90)`.
3. **Founder pass is a true no-op (R0-3, R1-2).** The founding promotion's `n_funnel ==
   max(own, windowed)` with the family arm 0 at evaluation; the family is created only after
   `final_passed`, and the gate row's `family_id`/`family_lifetime_effective` stay `NULL`/`0` (NOT
   stamped — see test 11); the linkage is in `family_events` only.
4. **Nothing minted on a FAILED gate (R1-1/R1-3).** An agent NOVEL strategy that FAILS the gate leaves
   `families` empty, no `family_members` row, and does NOT appear in
   `all_families_with_member_profiles` (never an anchor). Strategy stays BACKTESTED.
5. **Future-sibling protection.** After the founder passes and is seeded, advance the clock past 90d so
   the founding sweep leaves the window; assign a sibling; assert the sibling's family arm still
   reflects the seed (durable) while its `windowed_total` dropped — no reset.
6. **Monotonicity property.** For arbitrary funnel states,
   `effective_funnel_breadth(own, windowed, seed + members) >= effective_funnel_breadth(own, windowed,
   members) >= effective_funnel_breadth(own, windowed, 0)`. The agent-seeded family's future-sibling
   `n_funnel` is always ≥ the human-fresh-zero baseline and ≥ today's windowed-only bar.
7. **Concurrent NOVEL passes — INDEPENDENT vs RELATED (R8-LOW, verdict-dependent per §5.3).** Two
   workers promote concurrently on the `test_concurrency.py` `busy_timeout` harness. *Independent case:*
   two genuinely-unrelated NOVEL strategies → whoever commits second sees the fingerprint change, replays
   preflight, RE-confirms NOVEL against the now-larger graph, and mints a distinct second family; seeds
   monotone in serialization order, no lost update, no deadlock, no double-mint. *Related case:* two
   sibling variants → the second's replay re-classifies MERGE/PARENTAGE and JOINS the first family (mints
   NO second root) — the concurrency-split defense. The old "both always mint distinct families" behavior
   is explicitly NOT asserted.
8. **Idempotent re-promote via stage guard.** After a pass the strategy is CANDIDATE; a second
   `research promote` is rejected (source stage ≠ BACKTESTED) → no second family. Slug UNIQUE collision
   (pre-existing same-name family) → the atomic tx aborts fail-closed, strategy stays BACKTESTED.
9. **Agent-seed positivity + FILTER-BEFORE-SUM, no permanent DoS (R4-HIGH / R5-HIGH-3 / R9-MEDIUM-3).**
   Every agent-created family has `seeded_prior_combos > 0`; the human `--new-family` path yields `0`. The
   §5.1 seed = the WHERE-filtered SUM over well-typed in-range rows. **Corruption cases (parametrized,
   direct-INSERT bypassing the writer):** (a) numeric `n_combos ≤ 0`; (b) TEXT `'abc'`; (c) REAL `2.5`;
   (d) BLOB; (e) over-bound integer `9223372036854775807`. In each case, **with legitimate rows also
   present, the mint STILL SUCCEEDS** — the corrupt row is excluded from the SUM (contributes 0), the seed
   equals the sum of the legitimate rows, and one corrupt legacy row does NOT DoS the mint (the R9 fix; a
   dedicated assertion covers the over-bound-integer case that R8 would have overflowed + failed-closed
   forever). **With ONLY corrupt/no legitimate rows**, `seed == 0` → the mint fails closed (no family,
   strategy stays BACKTESTED). Assert `search_trials_corruption_count = n_rows − n_well_typed` is surfaced
   (family-audit) but never blocks. Also: `record_search_trial` rejects a non-`int` (incl. `bool`), `< 1`,
   or `> MAX_N_COMBOS`; the fresh-DB `search_trials` DDL carries `CHECK (typeof(n_combos)='integer' AND
   n_combos >= 1 AND n_combos <= 1000000000)`.
10. **Pending-NOVEL drift aborts — assignment AND graph-fingerprint (R3-HIGH, R5-HIGH-1).**
    *Assignment variant:* assign the strategy to an existing family AFTER a NOVEL preflight but BEFORE
    `run_gate` → the pre-lock `run_gate` check raises (still-unassigned violated); a concurrency sub-case
    lands the assignment only after the pre-lock check but before the write lock → the §5.1 step-1 (a)
    under-lock re-check rolls back fail-closed, minting nothing.
    *Fingerprint variant (R5-HIGH-1):* between this strategy's NOVEL preflight and its commit, a
    concurrent promote mints a NEW family (bumping `family_graph_fingerprint`); assert the §5.1 step-1
    (b) under-lock fingerprint CAS raises `FamilyGraphDriftError`, nothing is minted, and a preflight
    replay against the now-larger graph re-classifies (a related-variant replay yields MERGE/PARENTAGE
    and joins the first family instead of minting a second root — the concurrency-split defense).
    Sub-cases, one per widened fingerprint component (R6-HIGH-3): a new member assignment, a member
    **removal** (`removed_at` UPDATE — drops the active-only COUNT, R7-MEDIUM), a new parentage edge, and
    **a member backtest-returns refresh** (`persist_backtest_returns` on an existing member — the
    return-correlation axis) each trip the same CAS.
    *Persisted-profile is DB state, live source cannot flip a member (R9-HIGH-3):* materialise a member,
    then EDIT its module source so its live `code_hash`/`factors` differ from the persisted columns;
    assert `all_families_with_member_profiles` still returns the PERSISTED (frozen) profile, the
    classifier verdict is unchanged, and `family_graph_fingerprint()` is unchanged (a live source edit
    bumps NO DB component) — i.e. the former source-drift channel no longer exists.
    *Capture-timing race (R6-CRITICAL):* simulate a family mint / member (re)assignment landing DURING
    classification — between the helper's top-of-function `fp_before` read and its post-verdict `fp_after`
    read (patch the graph read to mutate mid-call); assert `_classify_and_assign_family` raises
    `FamilyGraphDriftError` rather than storing a snapshot that disagrees with the state the verdict was
    computed on. A control case (no mid-classification mutation) stores a stable snapshot and proceeds.
11. **Gate-row columns are the EVALUATED breadth, never stamped (R3-MED-1).** A passing founder's gate
    row has `family_id = NULL`, `family_lifetime_effective = 0`, consistent with its `n_funnel` /
    `decision_json`; the created family appears only in `family_events` (`family_created` +
    `strategy_assigned`).
12. **Single-transaction atomicity (R3-LOW).** The mint uses raw locked INSERTs; a forced failure after
    the family INSERT but before commit rolls back the gate row, stage CAS, family, AND membership
    together (assert none persisted). The public `create_family`/`assign_strategy_to_family` helpers are
    NOT called inside the promote tx.
13. **#339 CAS integration.** A `search_trials` insert between preflight and the promote commit still
    trips `FunnelDriftError` for the founder's member/window component (the seed is born in-tx, not a
    drift source) — existing CAS behavior preserved.
14. **`family-audit` unaffected (advisory).** Still read-only; its breadth numbers now include seeds
    (consistent); no transitions/writes.
15. **Load-bearing per-window agent mint cap, canonical-count (R5-HIGH-2, R6-HIGH-2).** Seed
    `AGENT_NOVEL_MINT_CAP` agent-created `families` rows dated within the window; a further agent NOVEL
    founder that otherwise PASSES its gate is fail-closed with `AgentMintCapError` — no family minted,
    strategy stays BACKTESTED. Boundary cases: `CAP-1` recent mints → the next mint succeeds; a family
    row dated *before* the window cutoff does NOT count (rolls off); a human `--new-family` family is NOT
    counted (only `created_by_actor='agent'`). **R6-HIGH-2 case:** an agent `families` row WITHOUT a
    matching `family_created` event still counts toward the cap (the cap reads canonical `families`, not
    the event stream). The cap is enforced UNDER the write lock (a concurrent-mint variant on the
    `test_concurrency.py` harness cannot exceed `CAP` even with two workers racing).
    **Preflight pre-check, NO holdout burn — cap AND snapshot (R7-HIGH-3, R8-HIGH):** (a) with the window
    already at `CAP`, an agent NOVEL promote is refused in `promotion_preflight` BEFORE the holdout peek;
    (b) with a family minted / a member-returns refresh between classification and the end of preflight,
    the pre-peek snapshot revalidation raises `FamilyGraphDriftError` BEFORE the peek. In BOTH cases assert
    the holdout interval is NOT burned/reserved and NO `gate_evaluations` row is written.
    **Atomic burn-time release-on-drift — NO holdout burn even DURING walk_forward (R9-HIGH-4):** inject
    drift (a concurrent family mint / member (re)assignment, OR push the rate cap / lifetime budget to its
    limit) so it lands AFTER the pre-peek check but is present when `on_peek` fires; assert the wrapped
    `on_peek` raises BEFORE `finalize_holdout_reservation`, the reservation stays pending →
    `release_holdout_reservation` frees it, the holdout is NOT burned, nothing is minted, no
    `gate_evaluations` row is written, and the abort logs the strategy name + drift axis. A control (no
    drift) burns and mints normally. Asserts the R8 "unavoidable during-peek residual" is CLOSED.
16b. **Load-bearing LIFETIME mint budget + human replenishment (R9-HIGH-2).** Seed `lifetime_consumed ==
    AGENT_NOVEL_MINT_LIFETIME_BUDGET` agent families (rate cap NOT hit, e.g. spread across windows); a
    further agent NOVEL founder that otherwise PASSES is fail-closed with `AgentMintBudgetExhaustedError`
    (no family, stays BACKTESTED) at BOTH the preflight pre-check (no holdout burned) and the under-lock
    re-check. After a human appends an `agent_mint_grants` row, the next mint succeeds (allowance = budget
    + SUM(grants)). Assert an agent / non-human actor CANNOT append a grant (the `granted_by_actor='human'`
    CHECK + the human-only CLI guard reject it), so the agent cannot self-replenish. Budget enforced UNDER
    the lock on the concurrency harness (two workers cannot exceed the allowance).
16. **Collision-RESISTANT uuid mint slug + retry (R4-MEDIUM / R6-MEDIUM slug-DoS / R6-LOW).** Pre-create
    families named `{name}_family`, `{name}_family_2`, … (reserve the deterministic stems); an agent NOVEL
    strategy `{name}` that PASSES still mints — its family name is `{name}_family__<32-hex-uuid>`, the gate
    is NOT aborted regardless of pre-reserved stems, and `created_by_strategy`/`family_created.strategy_name`
    record the readable founder. Retry sub-case: force the first `uuid4()` to collide with an existing
    name (patched) and assert the mint regenerates and succeeds within the bounded retry, committing one
    family.
17. **Store-boundary actor coercion + pending-object validation (R5-MEDIUM, R6-LOW).**
    `record_gate_with_fdr_and_maybe_promote` accepts `actor='agent'` (string) equivalently to
    `Actor.AGENT` for a pending mint (coerced at entry). Fail-closed cases: a non-agent method actor
    (`'human'`/`Actor.HUMAN`) WITH a `pending_novel_family`; and a well-actored call carrying an
    inconsistent pending object (`pending.actor='human'` or `pending.verdict='merge'`). All asserted for
    both the string and the enum actor form.
18. **Append-only triggers ENFORCE immutability (R9-HIGH-1).** For each of `families`, `family_parents`,
    `family_events`, `backtest_returns`: a raw `UPDATE`/`DELETE` (bypassing the store API) `RAISE`s / is
    rejected. For `family_members`: a `DELETE` is rejected; an UPDATE of any non-`removed_at` column is
    rejected; the `removed_at` NULL→ts tombstone flip is PERMITTED; the one-time profile materialisation
    (`member_code_hash` NULL→value) is PERMITTED but a second UPDATE of an already-set profile is rejected.
    Assert the store's own removal/materialisation paths still succeed through the triggers.
19. **Fingerprint↔classifier-read binding guard test (R9-HIGH-1).** The source-scan guard asserts every
    table read by the classification read-path is a `family_graph_fingerprint` component; a deliberately
    introduced extra classifier read of an un-fingerprinted table makes the guard FAIL (simulated by a
    fixture); and a companion test asserts all ten append-only triggers exist in the live schema.
20. **Founder→gate audit link (R9-MEDIUM-2).** After an agent NOVEL pass, `families.founder_gate_id`
    equals the founding `gate_evaluations.id`; `SELECT id FROM families WHERE founder_gate_id=:gid` returns
    that family; a human-`--new-family` and a legacy family both carry `founder_gate_id = NULL`.
21. **Constants are CODEOWNERS-protected, not agent-reachable (R9-MEDIUM-1).** Assert
    `AGENT_NOVEL_MINT_CAP` / `_WINDOW_DAYS` / `_LIFETIME_BUDGET` are module constants in
    `algua/registry/store.py` and that NO CLI flag and NO environment variable feeds them (a source-scan
    asserts they are never read from `os.environ` nor exposed as a `research`/`registry` CLI option),
    mirroring the #277-style scanner approach.
22. **UTC fail-closed on the cap/budget query (R9-LOW).** A boundary test asserts the `created_at ==
    cutoff` agent row is counted and a one-second-earlier row is not; a directly-inserted agent family with
    a NAIVE/local/malformed `created_at` makes the cap read fail closed (`AgentMintCapError`) rather than
    silently mis-bucketing it across the cutoff.
23. **Additive-seed multi-parent / ancestor dedupe (R9-LOW).** Build a diamond ancestry (family D reachable
    from the query set via two distinct parent paths) and a multi-parent child; assert
    `lifetime_combos_for_families` counts each family's `seeded_prior_combos` EXACTLY ONCE (the `set`
    closure dedup), and that a family reachable through several inputs does not double-count its seed.

## 9. Non-goals / deferred

- The per-window rate cap AND the lifetime mint budget (§5.1 step 5b / §6A) are BOTH in scope and
  load-bearing (R5-HIGH-2, R9-HIGH-2); what is deferred is only *auto-tuning* them — the defaults
  `AGENT_NOVEL_MINT_CAP` / `_WINDOW_DAYS` / `_LIFETIME_BUDGET` are human-only CODEOWNERS-protected
  governance knobs, not adaptive.
- A **durable funnel-lifetime founder tax** (a *statistical* bar) is deliberately NOT built (§6A): it would
  reintroduce the `#324` anti-scaling pathology (global lifetime statistical terms are rejected
  platform-wide; only family-scoped terms are lifetime). The cross-window repeated-founder channel is
  instead closed by the **non-statistical human-replenished lifetime BUDGET** (built, §5.1 step 5b) plus
  the rate cap + within-window `windowed_total` + audit monitoring — an explicit, documented governance
  decision (§6A), not an oversight or an accepted residual.
- **Auto-replenishing** the lifetime budget (e.g. a slow drip) is NOT built — replenishment is a
  deliberate human-actor `agent_mint_grants` grant, so a human always signs off before automated founding
  resumes past the epoch budget.
- **Symmetrizing the human NOVEL path** to pass-time creation (removing the §6E shelf-inventory/anchor
  asymmetry while preserving human fresh-zero authority) is deferred; human NOVEL stays
  create-in-preflight (pre-existing, human-authorized behavior).
- A **canonical SHA over the ordered id set** in place of the `(COUNT, MAX(id))`-per-table
  `graph_fingerprint` tuple is a pure micro-optimisation, not a correctness need — the append-only DB
  triggers (§2.1), not the digest shape, now guarantee no hard-DELETE/id-rewrite can occur — so it is
  dropped rather than deferred.
- Eliminating even the tiny **legacy member-profile materialisation** dual path (by requiring all members
  DB-materialised at deploy) is unnecessary in practice (families are nascent) and left as a one-shot
  bootstrap step, not ongoing dual-path code.
- No change to the LORD++ FDR ledger (family-agnostic, cohort-partitioned) or clustering thresholds.
- No retroactive re-seeding of existing families (correct at 0 by construction).

## 10. Quality gates

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
(`algua/contracts`/`algua/features` untouched; `PendingNovelFamily` is plain scalars in `repository.py`
so the `registry`→`research` import boundary is unchanged).
