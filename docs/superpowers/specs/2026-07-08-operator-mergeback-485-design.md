# Autonomous research-cycle merge-back driver (#485)

Paper-operator Slice 3. Turns a research cycle's candidate branch into an on-main, allocated paper
strategy with **no human merging the branch**.

> **Implementation status (2026-07-08, post-GATE-2).** The correctness core of this design is now
> shipped in code and tested: the branch-tip-SHA identity + durable per-strategy journal
> (`algua/operator/journal.py`), the CODEOWNERS-aware diff policy with mode/rename/canonicalization
> hardening (`algua/operator/diff_policy.py`), the `GitOps` seam with the freshly-fetched
> `origin/main` content check and the remote compare-and-swap push (`algua/operator/gitops.py`), the
> journal-driven `run_merge_back` state machine with stage-drift fail-closed and token-bound promote
> attribution (`algua/operator/mergeback.py`), and the `attempt_token` binding on `gate_evaluations`
> (schema v35→v36 + partial unique index + `passing_gate_by_token`). This resolves findings
> **#1, #2, #3, #4, #5** and the substantive **C1, C2, C4, C5, R1, R2, R4, R5**.
>
> **Deferred to a documented follow-up (NOT shipped in this PR)**, so this doc does not over-claim:
> the *finer-grained locking* of C3 — the shipped slice takes a **single repo-global saga lock**
> (`merge_back.lock`) that serializes ALL merge-backs, rather than the per-strategy lock + a separate
> repo-global git-critical-section lock; and the **remote push-boundary preflight** of C6/R3/R6 (the
> driver *verifying*, at startup, that branch protection has force-push/deletion blocked +
> `required_linear_history` off and that a server-side path-restriction push ruleset is present).
> Those remote controls remain **required operating preconditions configured out-of-band** (see
> *"Security boundary for autonomous push"* and *"Scope / non-goals"*); the driver's local DIFF
> POLICY + full quality gate + remote-CAS push are the shipped preventive boundary. Task 7's
> preflight-verification wiring is the deferred item.

> **Gate-1 revision (Codex gpt-5.5 BLOCK — 1 CRITICAL + 5 HIGH), second pass.** Findings are labelled
> **C1..C6** to disambiguate from the original review's findings #1–#10. Summary of where each is
> addressed:
> - **C1 (CRITICAL)** — promote attribution bound to an explicit **attempt token** stamped on the
>   `gate_evaluations` row, not `id > captured_max AND final_passed` → *"PROMOTE"* + *"Attempt
>   identity"* sections, new **Task 0**.
> - **C2 (HIGH)** — recovery re-derives "already committed" from a **durable, re-derivable token**
>   persisted in the journal *before* the promote call, never from a fresh `MAX(id)` snapshot →
>   *"PROMOTE"*, *"Crash-idempotency"*.
> - **C3 (HIGH)** — a **repo-global git lock** serializes the MERGE/PUSH/REVERT critical section
>   across strategies (shared-checkout mutation) → *"Concurrency / serialization"*.
> - **C4 (HIGH)** — **positive content verification**: before PROMOTE and before INTAKE, the
>   allowlisted paths at `main` HEAD must byte-match the blobs `branch_tip` introduced (tree/blob
>   hash equality), not merely "no named revert" → *"Effective-presence content check"*.
> - **C5 (HIGH)** — `evaluate_diff` rejects **non-regular git modes** (symlinks `120000`, gitlinks
>   `160000`) and **canonicalizes paths** (Unicode NFC + case-fold + traversal reject) before
>   allow/deny matching → *"DIFF POLICY"*.
> - **C6 (HIGH)** — the security boundary that **replaces** the bypassed local push hook (remote
>   branch protection, scoped bot credential, per-push audit log) is documented as a required,
>   guarded tradeoff → new *"Security boundary for autonomous push (C6)"* section.
>
> **Gate-1 revision (Codex gpt-5.5 BLOCK — 2 CRITICAL + 4 HIGH), third pass.** The second pass's fix
> for C4 verified content against the **local** checkout's `HEAD`, which is not authoritative once the
> driver shares `main` with other clones/CI/humans. This pass makes **`origin/main` (freshly fetched)
> the single source of truth** for every presence/serialization decision. Findings are labelled
> **R1..R6**:
> - **R1 (CRITICAL)** — every effective-presence / merge-verification / stage-drift decision asserts
>   against a **freshly-fetched `origin/main` tip**, never the local checkout's `HEAD` → *"Remote is
>   authoritative"*, *"Effective-presence content check"*, *"PROMOTE"*, *"INTAKE"*.
> - **R2 (CRITICAL)** — the push is a real **remote compare-and-swap**: `git push
>   origin <merge_sha>:refs/heads/main` guarded by a fresh fetch asserting `origin/main == base_sha`
>   immediately before, and a post-push fetch asserting the pushed SHA is what is live → *"PUSH"*,
>   *"Remote is authoritative"*.
> - **R3 (HIGH)** — a **server-side path-based push policy** (pre-receive hook / GitHub push ruleset
>   restricting the paths the bot identity may touch) rejects CODEOWNERS-protected writes at the
>   remote itself, not just in the driver's local diff policy → *"Security boundary for autonomous
>   push (C6/R3)"*.
> - **R4 (HIGH)** — a **DB-level partial unique index on non-null `(strategy, attempt_token)`** plus
>   folding `actor` + a **relaxation-flags fingerprint** into both the token derivation and the
>   read-back match condition → *"PROMOTE"*, *"Attempt identity"*, **Task 0**.
> - **R5 (HIGH)** — the diff-entry model carries **`old_path`/`new_path`** for renames/copies and
>   evaluates **both** paths against allow/deny, rejecting any rename/copy touching a denylisted path
>   on either side → *"DIFF POLICY"*.
> - **R6 (HIGH)** — the branch-protection contradiction (`--no-ff` merge commits vs a "linear-history"
>   requirement) is resolved: **linear-history is dropped**; the relied-on GitHub protection flags are
>   named explicitly → *"Security boundary for autonomous push (C6/R3)"*.

The operation is a **saga across two non-transactional systems** (git and the registry SQLite DB).
`stage` alone is NOT the durable operation state — it cannot prove which branch was merged, which
commit was tested, which gate row was consumed, or whether a paper allocation corresponds to the
intended strategy code. Recovery is therefore driven by a **durable merge-back journal** owned by
the driver, keyed on the immutable **branch-tip SHA**, and every cross-system step is either
individually idempotent, verified against authoritative state, or fail-closed.

## Remote is authoritative — resolves findings R1, R2

The shared truth is **`origin/main`**, not the driver's local checkout. The driver is one of several
clones/runners/CI jobs/humans that can advance or revert `origin/main`; its local `main`/`HEAD` is a
possibly-stale cache. So the design makes one rule global: **every presence, merge-verification,
stage-drift, and content decision is taken against a freshly-fetched `origin/main` tip
(`git fetch origin main` immediately before the check, then read `refs/remotes/origin/main`), never
against the local checkout's `HEAD`.** Concretely:

- The merge-verification (`merge_sha` 2nd-parent match + ancestry) checks `merge_sha` is an ancestor
  of the **fetched `origin/main`**, not local `HEAD`.
- The effective-presence blob-equality check (C4) compares the captured blobs to
  **`origin/main:<path>`** (`GitOps.blob_at("origin/main", path)` after fetch), not `HEAD:<path>`.
- The push is a **remote compare-and-swap** (see *"PUSH"*): it asserts the fetched `origin/main`
  still equals the recorded `base_sha` immediately before pushing, pushes the explicit merge SHA to
  `refs/heads/main`, and re-fetches to assert the pushed SHA is now live — so a purely-local flock,
  which cannot see other clones, is backed by an actual remote serialization primitive.

The local flock (per-strategy + repo-global git lock, C3) still serializes **this** process's own
shared-checkout mutation; the remote CAS is what serializes **across** processes/clones. Both are
required: the flock keeps the local working tree coherent, the CAS keeps `origin/main` coherent.

## Ledger (durable recovery journal) — resolves finding #8

`algua/operator/journal.py` maintains a **per-strategy** append-only JSONL journal at
`merge_back.<strategy>.journal` beside the registry db (atomic tmp+rename+`fsync`, written only while
**that strategy's** per-strategy lock is held). **The journal is per-strategy, NOT one shared file**:
because each attempt writes journal records outside the repo-global git lock (e.g. recording
`promote_status` after the DB read-back, and `intake_status` after intake — both DB-only steps), a
single shared `merge_back.journal` would be written concurrently by different-strategy attempts that
each hold only their own per-strategy lock, racing the tmp+rename and losing recovery state. Keying
the file by strategy makes the existing per-strategy lock the **sole writer** of that file, so every
append is serialized without needing a separate journal-global lock. (The `<strategy>` component is
filesystem-sanitized; strategy names are already validated registry identifiers.) It is the driver's
**own** recovery state — NOT registry-domain state — so it needs **no db.py schema bump** for the
journal itself and stays out of the CODEOWNERS-protected `store.py`. `algua.operator` still imports
nothing from `algua`.

One record per merge-back **attempt**, identified by `(strategy, branch_tip)`:

| field | meaning |
|---|---|
| `strategy` | strategy name |
| `branch` | branch ref name (informational; NOT the identity) |
| `branch_tip` | resolved SHA of the branch at attempt start — the **immutable attempt identity** |
| `base_sha` | main HEAD the merge was computed against |
| `diff_policy` | `pending` \| `passed` \| `rejected` |
| `gate_status` | `pending` \| `green` \| `failed` |
| `merge_sha` | the merge commit SHA once committed (else null) |
| `push_status` | `pending` \| `pushed` |
| `attempt_token` | opaque per-attempt idempotency key stamped on this attempt's gate row (C1/C2) — `sha256("mergeback:" + strategy + ":" + branch_tip + ":" + merge_sha)`, **re-derivable** from the durable fields above, written **before** the promote call |
| `promote_status` | `pending` \| `passed` \| `failed` |
| `promote_gate_id` | the `gate_evaluations.id` minted by `research promote` (proves WHICH gate run — the row whose `attempt_token` matches) |
| `intake_status` | `pending` \| `allocated` \| `queued` \| `failed` |
| `revert_sha` | set iff the merge was reverted |
| `terminal` | null while in-flight, else one of the terminal outcomes below |

The latest record for `(strategy, branch_tip)` is authoritative. Recovery reads it and resumes from
the last durably-recorded step — it never re-derives progress from `stage` + git ancestry.

**Terminal outcomes** (all emitted as the command's JSON `outcome`): `already_done`,
`diff_policy_rejected`, `gate_failed`, `promote_failed_reverted`, `promote_discarded_reverted`,
`promoted_allocated`, `promoted_queued`.

## Attempt identity is the branch-tip SHA, not the branch name — resolves findings #2, #3

`is_merged` / `git merge-base --is-ancestor branch HEAD` is **abandoned**: a *reverted* merge leaves
`branch_tip` an ancestor of HEAD forever, so ancestry cannot distinguish "present on main" from
"merged then reverted." Instead:

- **"Did THIS driver merge this branch_tip?"** is answered by the journal (`merge_sha` recorded),
  then *verified against git*: `merge_sha` must exist, its **second parent must equal `branch_tip`**,
  and (after `git fetch origin main`) `merge_sha` must be an ancestor of the **freshly-fetched
  `origin/main` tip** (R1) — NOT the local checkout's `HEAD` — **and not reverted** (no `revert_sha`
  recorded, and no revert commit naming it on `origin/main`'s first-parent history). This is exact
  code identity on the shared remote, not local-ancestry heuristics.
- A **new branch tip** (corrected code) is a **new attempt** → new journal record. The driver never
  re-merges a `branch_tip` that already has a terminal record; re-invocation on a terminal attempt
  emits the recorded terminal outcome (idempotent) rather than blindly re-merging.

### Effective-presence content check — resolves finding C4

The "not reverted" test above is **absence-of-named-revert** — a heuristic that a
non-standard revert (a squashed revert, a manual counter-commit, a force-rewritten history that
dropped the merge's tree effect, or a later commit that re-modified the allowlisted paths) can
defeat. Before **PROMOTE** and again before **INTAKE**, the driver therefore adds a **positive
content assertion**, not just an absence check — **and it asserts against the freshly-fetched
`origin/main` tip, not the local checkout (R1)**:

- For each allowlisted path `branch_tip` introduced/modified, capture the **blob object id** as it
  exists in the merge commit's tree (`git rev-parse <merge_sha>:<path>` → blob sha, via
  `GitOps.tree_blobs(merge_sha, paths)`).
- `git fetch origin main`, then assert every such blob is **byte-identical at the fetched
  `origin/main` tip** (`GitOps.blob_at("origin/main", path)` equals the captured blob sha) — i.e. the
  code this attempt shipped is *actually the tree content live on the shared remote right now*, not
  merely "a merge commit sits somewhere in local history." Checking the local `HEAD` here would be
  fooled by another clone reverting or advancing `origin/main` while this checkout stays stale (R1).
- Any mismatch or missing path → the merge is treated as **not effectively present** → the driver
  **fails closed** (`error: merge_content_absent`, no revert, alert). It never promotes or intakes a
  strategy whose remote code no longer matches what it tested. This defends the
  `DB-promoted/allocated ⇒ intended code is live on origin/main` invariant against every revert
  shape, not just the one that names `merge_sha`.

The set of paths to check is the allowlisted change set already computed by the DIFF POLICY step
(persisted implicitly via `branch_tip` + `base_sha` — recomputable on recovery as
`changed_paths(merge_base(base_sha, branch_tip), branch_tip)` filtered to the allowlist), so the
check is fully re-derivable on a recovery pass.

## Flow (per-strategy lock for the whole saga; repo-global git lock for the git critical section)

1. **Recover + preconditions.** Acquire the per-strategy lock, then the repo-global git lock (fixed
   order — see *"Concurrency"* / C3). Resolve `branch_tip`. Load the journal record for
   `(strategy, branch_tip)`. If a `--no-commit` preview is dangling from a crash (never committed),
   `git merge --abort` (safe — done under the global git lock so a concurrent attempt's staged tree
   is never clobbered). `git fetch origin main` (so `origin/main` is fresh for every downstream check,
   R1), then require: on `main`, working tree clean, local `main` **not diverged** from the fetched
   `origin/main` (either equal, or behind by a fast-forward — the driver `git reset --hard
   origin/main` to adopt remote truth before recomputing; a local commit `origin/main` does not
   contain means a prior crash pushed nothing and left junk → fail closed for human triage). Else fail
   closed.

2. **Resume dispatch — journal-first, then stage as a cross-check.** The journal record decides:
   - `terminal != null` → emit the recorded terminal outcome. Done.
   - `intake_status == allocated` → emit `promoted_allocated`. Done.
   - `promote_status == passed` (and `merge_sha` git-verified per above) → go to **INTAKE**.
   - `merge_sha` recorded + git-verified, `promote_status == pending` → go to **PROMOTE**.
   - `diff_policy == passed`, no `merge_sha` → go to **MERGE**.
   - no record → fresh attempt: go to **DIFF POLICY**.

   Stage is used only as a **fail-closed cross-check**, never as the resume authority: if the journal
   says we have not promoted but `stage == candidate` (or `paper`), the strategy was advanced by
   **someone/something other than this attempt** → **fail closed** (`error: stage_drift`), do NOT
   intake. This directly closes finding #1 (a `candidate` produced off-branch, with no journal proof
   this driver merged its code to main, can never reach INTAKE).

3. **DIFF POLICY (pre-merge, hard gate) — resolves finding #3, hardened for C5, R5.** Compute the
   set of change entries the branch introduces vs `merge-base(main, branch_tip)` from `git diff --raw`
   (`GitOps.changed_entries`). Each entry is a **`DiffEntry(mode, change_type, old_path, new_path)`**
   tuple — modes and both paths come from the raw diff, NOT just a single name, because a
   filesystem-level trick or a **rename/copy** can smuggle a denied change through an
   allowlisted-looking destination name. The autonomous merge is a **security boundary the quality
   gate is not** (a branch can weaken the very tests/config the gate runs), so the diff is gated
   **independently of whether the branch's own gate passes**:
   - **Rename/copy dual-path guard (R5, evaluated on BOTH sides):** `git diff --raw` reports renames
     (`R100`) and copies (`C…`) with **two** path fields — the source (`old_path`) and the
     destination (`new_path`). The policy evaluates the **full allow/deny check against BOTH paths**:
     a rename/copy is permitted only if *both* its source and destination pass the allowlist and
     neither hits the denylist. This rejects `git mv algua/registry/store.py
     algua/strategies/foo/store.py` (denylisted source → dodged by a modification-only view of the
     destination) **and** `cp store.py → allowlisted dest` (copy-out of protected code). A pure
     addition/modification/deletion has `old_path == new_path` (or a null source for an add) and
     collapses to the single-path check below. `git diff --raw` MUST be invoked with rename/copy
     detection enabled (`-M -C`) so these entries are surfaced as `R`/`C` rather than
     add+delete pairs; a `T` (type-change) entry is treated as its post-image mode and still passes
     through the object-mode guard.
   - **Object-mode guard (C5, checked next):** every changed entry must be a **regular blob**
     (`100644` or `100755`). Any non-regular git mode — **symlink `120000`, gitlink/submodule
     `160000`**, or an unknown mode — is denied outright, even on an allowlisted path (a symlink
     under `algua/strategies/` could point a "strategy file" at `../registry/store.py`, and a
     gitlink could pull in unreviewed history). Mode `100755` (exec bit) on a `.py` strategy file is
     also denied — strategy artifacts are non-executable data.
   - **Path canonicalization (C5), applied before allow/deny matching to BOTH `old_path` and
     `new_path`:** each path is
     Unicode-**NFC-normalized**, rejected if it contains `..`, is absolute, has a `.git/` segment,
     or does not round-trip through normalization; matching is **case-folded** (so `ALGUA/Strategies`
     or an NFD-encoded homoglyph cannot present a denied path as an allowlisted one on a
     case-insensitive/Unicode-normalizing filesystem). A path that fails to canonicalize → denied.
   - **Allowlist (reject-by-default):** the ONLY additions/modifications permitted are strategy
     artifacts — `algua/strategies/<family>/**.py` (additions-only per the GENERATED_BY discipline)
     and `kb/**` report/doc artifacts. Anything not on the allowlist is denied.
     Additionally, deletions of allowlisted paths are denied (a strategy branch adds; it never
     removes on-main strategy code).
   - **Denylist (CODEOWNERS-aware, hard fail):** the denylist is **derived at runtime from the
     `CODEOWNERS` file** (so it cannot drift) — every protected path (`store.py`, `lifecycle.py`,
     `engine.py`, `gates.py`, `clustering.py`, `approvers/`, `live_gate.py`, `transitions.py`,
     `promotion.py`, `forward_gates.py`, `forward_promotion.py`, `human_actor.py`, the CLI
     enforcement call sites, `.github/`, `CODEOWNERS` itself) — **plus** `tests/**`, build/lint/type
     config (`pyproject.toml`, `ruff.toml`, `mypy.ini`, `.importlinter`/`setup.cfg`), and
     `algua/operator/**` (the driver may not merge changes to itself). Denylist matching runs on the
     **canonicalized** path too, so a normalization trick cannot dodge it.
   - Any denied entry (bad mode, non-canonical path on either side, rename/copy whose source or
     destination is non-allowlisted or denylisted, non-allowlisted, denylisted, or disallowed
     deletion) → record `diff_policy=rejected`, emit `diff_policy_rejected`, **no merge is
     attempted.** `CODEOWNERS` failing to parse → fail closed.

4. **MERGE (only if gate green).** `git merge --no-ff --no-commit <branch_tip>`, run the **full
   quality gate** (`pytest -q && ruff check . && mypy algua && lint-imports`) on the staged tree.
   red → `git merge --abort` (main untouched, nothing to revert), record `gate_status=failed`, emit
   `gate_failed`. green → `git commit --no-edit`; record `merge_sha` + `gate_status=green` in the
   journal (fsync) **before** the next step.

5. **PUSH — remote compare-and-swap, resolves findings #10, R2.** The push is a real remote CAS, not
   a blind `HEAD:main`:
   - **Pre-push CAS guard.** `git fetch origin main`; assert the fetched `origin/main` tip **still
     equals the recorded `base_sha`** (the main HEAD this merge was computed against). If it has
     moved (another clone/CI/human advanced or reverted `origin/main`), the local merge is stale →
     `git merge --abort` if still un-pushed, **fail closed** (`error: remote_moved`), and let a fresh
     attempt recompute the merge against the new base. This is the local half of the CAS.
   - **Explicit-SHA push (the atomic swap).** `git push origin <merge_sha>:refs/heads/main` — pushing
     the *explicit merge SHA* to the ref, not the ambiguous `HEAD`. Combined with the remote's
     non-force-push / non-fast-forward-only protection (see *"Security boundary"* R6), the remote
     **rejects the push if `origin/main` is no longer the ancestor the driver fetched** — i.e. the
     remote itself enforces the compare-and-swap: two clones racing the same base can't both win,
     the loser's non-fast-forward push is refused and it fails closed. This closes the gap a purely
     local flock (C3) can't cover across clones/runners/CI.
   - **Post-push re-verify.** `git fetch origin main` again and assert the fetched `origin/main` tip
     **now equals `merge_sha`** — the pushed SHA is what is actually live on the shared remote. A
     mismatch (someone reverted/rewrote between the accepted push and this fetch) → **fail closed**
     before any DB mutation (`error: remote_moved`), never promote against a remote that no longer
     carries our code. Only on success record `push_status=pushed`.

   This keeps the invariant *DB-promoted ⇒ code is on `origin/main`*: a crash after promote can never
   leave the registry ahead of the shared remote, and a concurrent remote mutation is detected
   instead of silently promoted over. A crash between step 4 and here leaves `push_status=pending`;
   recovery (step 1's fast-forward check + this pre-push CAS) re-drives the push idempotently — if
   `origin/main` already carries `merge_sha` (a prior push that crashed before journaling) the
   post-push re-verify passes immediately and it records `pushed` without re-pushing. The paper runner
   refuses to run a strategy whose `merge_sha` is not on `origin/main` (a cheap guard, stated as a
   downstream contract). This driver runs as a subprocess, not via the dev agent's Bash tool, so the
   local "no `git push` + `main`" hook does not apply to it — **the coverage that hook provided is
   deliberately replaced by the remote-side boundary in *"Security boundary for autonomous push
   (C6/R3)"* below**, and every push emits an audit-log record.

6. **PROMOTE (strict agent, single metered burn) — resolves findings #4, #9, and C1/C2, R1/R4.**
   **Effective-presence content check first** (C4/R1): `git fetch origin main`, then re-assert the
   allowlisted blobs are byte-live on the **fetched `origin/main` tip** (see *"Effective-presence
   content check"*); mismatch → fail closed before promoting.

   Then **derive and durably record `attempt_token`** into the journal (fsync) **before** the promote
   call. The token folds the **strict-agent gate context** into its derivation (R4):
   `attempt_token = sha256("mergeback:v2:" + strategy + ":" + branch_tip + ":" + merge_sha + ":" +
   relaxation_fingerprint)`, where `relaxation_fingerprint = sha256` of the canonical
   `(actor=Actor.AGENT, declared_combos=None, allow_holdout_reuse=False, allow_non_pit=False,
   assume_terminal_last_close=False, new_family=False)` tuple — the exact strict inputs the driver
   hard-wires. It is a deterministic function of durable fields, so every recovery pass re-derives the
   *same* token — there is no fresh `MAX(id)` snapshot to drift (C2). Because the fingerprint is part
   of the token, a row minted under *any other* input set (a relaxed human run) cannot collide with
   this attempt's token even by coincidence.

   Call the injected promote seam with **hard-wired strict inputs**: `actor=Actor.AGENT`,
   `declared_combos=None`, `allow_holdout_reuse=False`, `allow_non_pit=False`, no
   `--assume-terminal-last-close`, no `--new-family`, **plus `attempt_token=<token>`**. The seam
   stamps `attempt_token` on the `gate_evaluations` row it mints (additive nullable column, guarded by
   a **partial unique index on non-null `(strategy, attempt_token)`** — see Task 0, R4). The unique
   index makes a second insert of the same `(strategy, attempt_token)` a hard DB error, so even a
   buggy double-drive cannot mint two rows this attempt would both claim. The seam's **type signature
   exposes no relaxation parameter**, so a human-only relaxation is impossible-by-construction (a
   NOVEL family verdict already fails closed for an agent).

   **Success is read from authoritative registry state, never from "did the call raise"** (finding
   #4). `promote_task` commits the gate row + `candidate` stage inside the repository transaction,
   then keeps building its payload / syncing side effects — a post-commit raise is
   indistinguishable from a pre-commit failure at the call boundary. So after the call returns **or
   raises**, the driver re-reads the registry keyed on **`attempt_token`**:

   | registry observation | interpretation | action |
   |---|---|---|
   | a `final_passed` gate row **stamped with our `attempt_token`** exists **and** stage `candidate`/`paper` | promotion **committed** | record `promote_status=passed`, `promote_gate_id` (that row's id); **do NOT revert**; go to INTAKE |
   | stage `backtested` **and** no gate row bearing our `attempt_token` | promotion did **not** commit | record `promote_status=failed`; **revert** (step 7) |
   | anything else (ambiguous) | cannot prove state | **fail closed**, no revert, alert — reverting a possibly-committed promotion is the dangerous direction |

   The passing gate row is matched to *this* attempt by **exact `attempt_token` equality AND
   strict-agent context equality** (C1/R4): `passing_gate_by_token` returns the row only if its
   `attempt_token` matches **and** its stored `actor`/relaxation columns equal the strict-agent set
   (`actor=AGENT`, no relaxations) — not by "id > captured pre-promote max." Since the token already
   folds the relaxation fingerprint (R4), this is belt-and-suspenders: even a hash collision would
   also have to reproduce the exact strict context to be attributed. The token is caller-supplied,
   opaque, per-attempt unique (enforced by the partial unique index, R4), so a **concurrent external
   `research promote` of the same strategy** — which supplies a *different* token, or none (NULL) —
   can **never** be mistaken for this attempt's own gate row, even if it mints a passing row with a
   higher id in the same window. Idempotency on recovery follows for free: the read is by the same
   re-derived token, so re-running after a crash finds the same (or no) row.

7. **REVERT = git-only rollback, NOT system rollback — resolves finding #7.** On a proven
   non-committed promote failure only: `git revert -m 1 <merge_sha> --no-edit`, push the revert, and
   record `revert_sha`. This reverts **only main's code**. The research-integrity ledgers are
   append-only **by design** and are NOT rolled back: a holdout burned on peek, a gate row, a family
   assignment, and any negative-result record all **survive**. This is an explicit **terminal
   discard** path, not a clean rollback:
   - promote failed *before* burning the holdout → `terminal=promote_failed_reverted`.
   - promote *burned the holdout* then failed → `terminal=promote_discarded_reverted`; the strategy
     identity is spent. Corrected code must return as a **new strategy/branch** (new `branch_tip` →
     new attempt); the same burned identity is intentionally not re-promotable.

8. **INTAKE — target-verified, resolves findings #6, R1.** **Effective-presence content check again**
   (C4/R1) before allocating: even though PROMOTE passed it, an intervening revert/rewrite between
   PROMOTE and INTAKE (or across a recovery gap) could have removed the code — `git fetch origin
   main`, re-assert the allowlisted blobs are byte-live on the **fetched `origin/main` tip**, fail
   closed on mismatch (`merge_content_absent`), so capital is never allocated to a strategy whose code
   is no longer on the shared remote. Then run the shared
   FIFO intake (`_run_intake`, reusing #317's atomic `intake_candidate_to_paper`). Intake fairness is
   FIFO across ALL candidates
   **by design** — it may admit an older queued candidate ahead of ours. So the driver does **not**
   infer its outcome from "intake admitted something"; it **re-reads THIS strategy's state** and
   emits a target-specific outcome:
   - this strategy now `paper` with an active allocation → record `intake_status=allocated`,
     `terminal=promoted_allocated`.
   - this strategy still `candidate` (queued behind capacity/equity) → record
     `intake_status=queued`, `terminal=promoted_queued` — a **distinct, honest** non-error terminal;
     a later capacity-freeing invocation re-runs intake and can allocate it.

## Concurrency / serialization — resolves finding #5, hardened for C3

The driver takes a **per-strategy** exclusive non-blocking lock `merge_back:<strategy>.lock` (not one
global driver lock) beside the registry db, held across the whole saga. Distinct from the deferred
paper-tick lock (#493).

### Repo-global git lock for the shared-checkout critical section (C3)

The per-strategy lock is **insufficient for the git steps**: two merge-back attempts on *different*
strategies both mutate the **one shared working tree** — `git merge --no-commit`, the quality-gate
run against the staged tree, `git commit`, `git merge --abort`, `git revert`, `git push`. Interleaved,
strategy A's `merge --no-commit` + gate could run against a tree that strategy B's concurrent merge
has already staged, or B's `merge --abort` could blow away A's in-flight merge. So, **in addition** to
the per-strategy lock, the driver holds a **repo-global** exclusive non-blocking lock
`merge_back.git.lock` across the entire **MERGE → gate → commit → PUSH** critical section (and across
any REVERT), serializing all shared-checkout git mutation to one attempt at a time.

- **Lock ordering is fixed** to prevent deadlock: always acquire **per-strategy first, then
  repo-global**. An attempt holds exactly one of each; a consistent order makes deadlock impossible.
- The global lock is held only for the git critical section, **not** for the DB-only steps (promote
  read-back, intake) — those are already guarded by the per-strategy lock + stage-CAS — to keep the
  serialization window as small as correctness allows.
- The quality gate (`pytest`, minutes) runs **inside** the global lock because it executes against
  the shared staged tree; merge-backs of different strategies therefore gate **serially**. Accepted:
  the paper operator is low-throughput and correctness-dominated. *(Considered alternative: run each
  attempt's merge+gate in an isolated `git worktree` so gates parallelize. Rejected for this slice —
  it complicates path handling and gate invocation for a throughput win we don't need; noted as the
  scaling follow-up if merge-back volume ever grows.)*

A file lock cannot force *other* commands (`research promote`, `paper intake`, `registry transition`
run by a human/other process) to serialize behind it, so the serialization primitive is
**belt-and-suspenders**:

1. **Per-strategy lock** — mutual exclusion between concurrent merge-back attempts on the same
   strategy.
2. **Repo-global git lock** — mutual exclusion of shared-checkout git mutation across ALL strategies
   (C3), held only for the MERGE/PUSH/REVERT critical section.
3. **Expected-stage CAS at every mutation the driver performs.** At attempt start the driver captures
   `expected_stage` (`backtested`) and the current candidate/stage transition id. Every registry
   mutation it drives (promote, intake) is conditioned on a stage-CAS **inside `BEGIN IMMEDIATE`**
   (the established #246/#317 pattern); if the observed stage has drifted (an external command moved
   it), the operation **fails closed and the driver aborts** rather than proceeding on drifted state.
   Combined with the journal-vs-stage cross-check in step 2, an external concurrent stage mutation
   during an active merge-back is converted into a **safe abort**, never a corrupting double-action.

External stage mutation of a strategy that is mid-merge-back is thus an operator error the design
**detects and fails closed on**, rather than one that can leave git and the registry disagreeing. (A
future hardening — having `research promote`/`paper intake`/`registry transition` acquire the same
per-strategy lease — is noted as follow-up; it touches CODEOWNERS CLI call sites and is out of scope
here.)

## Security boundary for autonomous push (C6/R3)

The local `.claude/hooks/block-push-to-main.sh` hook (which denies any Bash command containing both
`git push` and `main`) is a **developer-workstation guardrail**; it does not — and is not meant to —
govern an autonomous subprocess that pushes to `main` as its whole purpose. Removing its coverage for
this driver is a **deliberate, guarded tradeoff**, not an unstated hole. The coverage is replaced by a
boundary at the layer that actually matters — the **remote** — plus a scoped credential and an audit
trail. These are **required operating preconditions** for enabling `paper merge-back`; the driver's
`doctor`-style preflight refuses to run if they are unverifiable:

1. **Remote branch protection on `main` (required, preventive) — resolves R6.** `origin/main` must
   have branch protection **compatible with the `--no-ff` merge commit the driver pushes**. The exact
   GitHub ruleset flags relied on are named so this is verifiable, not hand-waved:
   - **`allow_force_pushes: false`** (block force-push) and **`allow_deletions: false`** (block branch
     deletion) — the preventive backstop that a compromised/buggy driver cannot rewrite or destroy
     shared history. This is also the primitive the R2 remote CAS leans on: a non-fast-forward push
     that would rewrite the tip is refused by the remote.
   - **`required_linear_history` is explicitly NOT enabled.** This resolves the R6 contradiction: a
     linear-history requirement rejects merge commits, but the driver's whole verification model
     depends on a `--no-ff` merge commit whose **second parent equals `branch_tip`**. The two are
     mutually exclusive, so the design commits to **allowing merge commits** and relies on
     force-push/deletion blocks (not linear history) for history-integrity. (If a future policy ever
     *requires* linear history on `main`, the merge strategy must change to a fast-forward/rebase of
     `branch_tip` onto `base_sha` and the 2nd-parent identity check replaced by a
     range/patch-id check — called out as the migration, not silently assumed away.)
   - **A server-side path-restriction control (R3, see precondition 2)** — the actual protected-path
     enforcement.
   - **Required *pre-push* status checks are deliberately NOT used as the boundary here** — a
     required-check gate that rejects a commit until its CI has run would **deadlock an autonomous
     direct pusher** (the checks can only run *after* the commit exists on a branch), and working
     around it (push-to-side-branch → open PR → bot self-merges) would require giving the bot
     merge-approval rights on a protected branch, which contradicts precondition 4 (least-privilege,
     non-approver bot). Instead, the **authoritative preventive check is the driver's own DIFF POLICY
     + full local quality gate run before the merge commit** (byte-identical to what CI runs); the
     push carries a commit that has *already* passed that gate.

2. **Server-side path-based push policy (required, preventive) — resolves R3.** The driver's local
   DIFF POLICY (step 3) is the *first* line, but a local check on the pushing process cannot be the
   *only* thing standing between an autonomous bot and a CODEOWNERS-protected write — a bug or
   compromise in the driver would bypass it. So the **remote itself** rejects protected-path writes by
   this bot identity, independent of the driver:
   - **GitHub push ruleset with `restrict paths` (file-path condition), scoped to the bot actor.**
     A repository **push ruleset** (org/repo `rulesets` API, `type: file_path_restriction`) denies
     any pushed commit that adds/modifies a path outside the merge-back allowlist
     (`algua/strategies/**`, `kb/**`) — equivalently, that touches any CODEOWNERS-protected path,
     config, `tests/**`, or `algua/operator/**`. The bot's pushes are evaluated against this rule; a
     protected-path change is **rejected at the remote**, so even a driver that skipped its own DIFF
     POLICY cannot land a protected write.
   - **Or, where push rulesets are unavailable, a `pre-receive` hook** on the server enforcing the
     same path allow/deny list server-side. Either mechanism satisfies R3; the preflight verifies one
     is present (queries the ruleset API / confirms the hook is installed) and **refuses to run** if
     neither can be confirmed.
   - This is the enforcement that being "not a CODEOWNERS approver" does **not** provide: CODEOWNERS
     review only gates the *PR-merge* path, which a direct pusher bypasses entirely. The path
     restriction is a control that binds the *push* itself.

3. **Remote CI on `main` as a detective control (required, post-push).** Branch CI still runs on
   every pushed commit, but as a **monitoring/alerting** control, not a push gate: a red result on
   `main` **pages the operator and can trigger an automated revert** of the offending `merge_sha`.
   This catches an environment-specific divergence between the local gate and CI without creating the
   deadlock of a pre-push required check. It is the independent *second* line behind the local gate.
4. **Dedicated, least-privilege bot credential (required).** The push uses a **dedicated bot
   identity** whose token is scoped to **contents:write on this one repository only** — not a human
   PAT, not an org-wide token. It can push branches/commits to this repo and nothing else; it is
   **not** a CODEOWNERS approver and **cannot** satisfy required PR review, so it can never
   self-merge a change to a protected path via the PR path — and the precondition-2 server-side path
   restriction stops it landing one via the *direct-push* path too. The credential is provisioned
   out-of-band (operator secret store) and injected as an env var to the driver subprocess; it never
   lands in the repo or the journal.
5. **Audit logging of every autonomous push (required).** Immediately before and after each push the
   driver emits a structured audit record (the #346 structured-logging seam) — `event=autonomous_push`,
   `strategy`, `branch`, `branch_tip`, `base_sha`, `merge_sha`, `refspec=<merge_sha>:refs/heads/main`,
   `actor=merge_back_driver`, `result` — so every autonomous mutation of `main` is attributable after
   the fact. Reverts (`event=autonomous_revert`, with `revert_sha`) are logged the same way. This is
   the accountability record the removed local hook never produced.

Net: the local hook blocked an *interactive* footgun; the local gate + DIFF POLICY (preventive) +
branch protection with a server-side path restriction (preventive backstop, R3/R6) + remote CAS push
(R2) + post-push CI paging/auto-revert (detective) + scoped non-approver bot credential + per-push
audit log form a **stronger, auditable, deadlock-free** boundary appropriate to an autonomous direct
pusher. The tradeoff is stated explicitly so it is a reviewed decision, not a silent regression.

## Crash-idempotency argument

- **No double-merge:** a `branch_tip` with a recorded `merge_sha` (git-verified: 2nd-parent match +
  ancestor of the **freshly-fetched `origin/main`** + not reverted) is never re-merged; the journal,
  not `git merge`'s "already up to date", is the authority.
- **No double-burn:** inherited from `reserve_holdout`'s single-use guard; the driver additionally
  never re-runs promote once `promote_status` is durably `passed` in the journal. Attribution is by
  the re-derivable `attempt_token` (C1/C2), so a recovery pass that re-reads the registry finds
  *this attempt's* row (or none) exactly — never a concurrent same-strategy promote's row, and never
  a drifting `MAX(id)` baseline.
- **No mis-attributed promote:** the passing gate row is matched by exact `attempt_token` equality,
  a caller-unique opaque key stamped on the row (C1). A concurrent external promote of the same
  strategy carries a different token / NULL and can never be mistaken for ours.
- **No double-allocation:** intake is one `BEGIN IMMEDIATE`; once `intake_status=allocated` the
  strategy is `paper` and re-running is convergent (no longer a candidate).
- **No registry-ahead-of-remote:** the push (step 5) is a remote CAS (pre-fetch `origin/main ==
  base_sha`, explicit-SHA push, post-fetch `origin/main == merge_sha`, R2) that **precedes** promote
  (step 6); a pending push is re-driven on recovery before any DB mutation, and a concurrent remote
  mutation is detected and fails closed instead of being promoted over.
- **No allocate-code-not-on-main:** INTAKE is reachable only via a `merge_sha` verified against the
  **freshly-fetched `origin/main`** (R1), a journal-proven `promote_status=passed`, **and** the
  effective-presence content check (C4/R1 — the allowlisted blobs byte-match the fetched `origin/main`
  tip) re-run at intake time; a `candidate`/`paper` stage without all three fails closed. No revert
  shape (named, squashed, or rewritten) and no concurrent remote advance can leave capital allocated
  to code that is no longer live on the shared remote.

## Design decisions

- **`origin/main` is the single source of truth** (R1/R2): every presence/merge-verification/
  content/stage-drift decision is taken against a freshly-fetched `origin/main`, and the push is a
  remote compare-and-swap — the local checkout is a cache, never the authority.
- **Pure orchestration seam.** `algua/operator/mergeback.py` holds a pure `run_merge_back(...)` taking
  injected callables (`stage_of`, `run_gate`, `promote(..., attempt_token)`, `intake`,
  `read_gate_by_token`, `changed_entries`) + `GitOps` (now including `changed_entries` returning
  `DiffEntry(mode, change_type, old_path, new_path)` for R5, `fetch_remote(ref)`,
  `remote_tip(ref)->sha`, `tree_blobs`, and `blob_at(ref, path)` for the C4/R1 content check, plus the
  R2 CAS `push_cas(merge_sha, expected_base)`) and `Journal` protocols, plus an injected `audit_log`
  sink (C6) and the repo-global git-lock handle (C3), so every branch is unit-testable with a
  `FakeGit`, a fake journal, and stubs — no subprocess, no DB, no real merge.
- **Journal-driven recovery, stage as cross-check only.** The saga's durable state is the journal
  keyed on `branch_tip`; `stage` is used solely to fail closed on drift, never as the resume
  authority. Promote-outcome attribution is keyed on the re-derivable `attempt_token` (C1/C2), whose
  derivation and read-back both fold the strict-agent relaxation fingerprint (R4) and which is backed
  by a partial unique index — the only registry-side identity strong enough to survive concurrent
  same-strategy promotes.
- **Gate before commit; content-verify (against `origin/main`) before promote AND intake.** The
  quality gate runs on a `--no-commit` preview; a red gate aborts with nothing on main. A committed
  merge is undone only by an explicit git revert on a proven non-committed promote failure. A positive
  blob-equality check against the fetched `origin/main` (C4/R1) — not an absence-of-revert heuristic,
  not the local checkout — guards both PROMOTE and INTAKE.
- **Shared-checkout git mutation is globally serialized locally AND remotely** (C3/R2): per-strategy
  lock for the saga, a repo-global git lock for the MERGE/PUSH/REVERT critical section (fixed acquire
  order), plus a remote compare-and-swap push that serializes across clones the local flock cannot see.
- **Autonomous push is guarded at the remote, not the workstation** (C6/R3/R6): the bypassed local
  hook is replaced by branch protection (force-push/deletion blocked, linear-history NOT required so
  the `--no-ff` merge is legal) + a **server-side path-restriction push policy** + a scoped
  non-approver bot credential + per-push audit logging.
- New `algua.operator` package imports nothing from `algua` → no import-linter change.

## Scope / non-goals

- CLI command `paper merge-back` lives in the CODEOWNERS-protected `paper_cmd.py`, and the
  `attempt_token` stamping (C1) touches CODEOWNERS-protected `store.py`/`promotion.py`/`db.py` → **PR
  stays open for a human to merge** this merge-to-main keystone (appropriate: a human authorizes
  shipping the autonomous-merge capability once; the capability then merges strategy branches
  autonomously).
- **One additive schema bump (`SCHEMA_VERSION` 35 → 36):** a nullable `attempt_token TEXT` column on
  `gate_evaluations` **plus a partial unique index `CREATE UNIQUE INDEX ... ON gate_evaluations
  (strategy, attempt_token) WHERE attempt_token IS NOT NULL`** (R4), both added via the established
  `_add_missing_columns` / index-create migration pattern (C1). It is backward-compatible (NULL for
  every existing/non-driver row, and the partial index ignores NULL rows so pre-existing data is
  untouched) and is the **only** bump this design introduces — it must be the sole `SCHEMA_VERSION`
  bump in flight when it lands (coordinate with any other open schema PR, e.g. #324's v32→33, so two
  bumps don't collide on the same version number).
- **Operating precondition, not code (R3/R6):** the remote **branch-protection ruleset** (force-push
  + deletion blocked, `required_linear_history` off) and the **server-side path-restriction push
  policy / pre-receive hook** are configured on the GitHub repo out-of-band; the driver only *verifies*
  them at preflight and refuses to run if unverifiable. They are documented here as required
  deploy-time setup, not shipped in this PR.
- No new registry/store query beyond a read-only "passing gate row for `(strategy, attempt_token)`"
  helper and the target-allocation read (both read-only).
- Cross-command per-strategy leasing (promote/intake/transition taking the merge-back lease) is
  deferred follow-up.
- Isolated-worktree merge execution (to parallelize gates across strategies instead of the global
  git lock) is a deferred scaling follow-up (C3).
- The journal itself is operator-local recovery state, not a registry table — it needs no schema
  bump; only the cross-system `attempt_token` binding does.

## Implementation task list

Per-task check is the FAST loop (`ruff check . && mypy algua && lint-imports && pytest -q <this
task's tests>`); the FULL gate runs only at integration and finish. `algua.operator` imports nothing
from `algua`, so none of the new modules touch import-linter contracts.

0. **`attempt_token` binding on the promote seam (C1/C2/R4) — CODEOWNERS + schema bump.** Additive
   nullable column `attempt_token TEXT` on `gate_evaluations` **plus a partial unique index on
   `(strategy, attempt_token) WHERE attempt_token IS NOT NULL`** (R4), via `_add_missing_columns` +
   index-create (`SCHEMA_VERSION` 35 → 36; must be the sole in-flight bump — coordinate version number
   with #324). Thread an optional `attempt_token: str | None = None` param through the promote seam
   (`promotion.py` → the `store.py` gate-INSERT at the `VALUES(...)` site), stamped on the row
   (NULL for all non-driver callers — backward-compatible). Add the read helper
   `passing_gate_by_token(conn, strategy, attempt_token) -> int | None` that returns the `final_passed`
   row id whose `attempt_token` matches **AND whose stored `actor`/relaxation columns equal the
   strict-agent set** (R4 belt-and-suspenders), else None. Tests: a driver promote stamps the token
   and is found by it; a concurrent non-driver promote of the same strategy (NULL token, higher id) is
   **not** returned by `passing_gate_by_token`; **a second insert of the same `(strategy,
   attempt_token)` raises the unique-index violation (R4)**; **a row bearing our token but a relaxed
   `actor`/flag set is NOT returned (R4)**; migration adds the column NULL + creates the partial index
   on an existing db.
   *Test files:* `tests/registry/test_gate_attempt_token.py`, extend `tests/registry/test_db_migrations.py`.
   **CODEOWNERS-protected (`store.py`/`promotion.py`/`db.py`) → PR stays OPEN for human merge.**

1. **`algua/operator/__init__.py` + `journal.py` — durable per-strategy merge-back journal
   (finding #8, C2).** `MergeBackRecord` dataclass (fields above, **incl. `attempt_token`**);
   `Journal` protocol + a **per-strategy** JSONL file impl (`merge_back.<strategy>.journal`,
   sanitized) with atomic tmp+rename+`fsync` append and "latest record for `(strategy, branch_tip)`"
   read. Pure/self-contained. Tests: append+read-back, latest-wins on duplicate key, crash-truncated
   last line is ignored (read stops at last well-formed record), `attempt_token` round-trips and is
   re-derivable from `(strategy, branch_tip, merge_sha)`, **two different strategies write disjoint
   files (no cross-strategy interference)**. *Test file:* `tests/operator/test_journal.py`.

2. **`algua/operator/diff_policy.py` — CODEOWNERS-aware allow/deny gate + mode/path/rename hardening
   (finding #3, C5, R5).** Pure `evaluate_diff(changed_entries, codeowners_text) -> DiffPolicyResult`
   where `changed_entries` is **`DiffEntry(mode, change_type, old_path, new_path)`** tuples (modes +
   both paths from `git diff --raw -M -C`): **rename/copy dual-path guard (R5 — evaluate the full
   allow/deny check against BOTH `old_path` and `new_path`; reject any `R`/`C` whose source or
   destination is denylisted or non-allowlisted)**; **object-mode guard** (reject any non-`100644`/
   `100755` mode — symlink `120000`, gitlink `160000`, unknown; reject `100755` on `.py` strategy
   files); **path canonicalization applied to BOTH paths** (Unicode-NFC, case-fold matching, reject
   `..`/absolute/`.git`/non-round-tripping); then reject-by-default allowlist
   (`algua/strategies/<family>/**.py`, `kb/**`), deny deletions of allowlisted paths, denylist derived
   by parsing `CODEOWNERS` ∪ static extras (`tests/**`, config, `algua/operator/**`) matched on the
   canonical path; fail closed on unparseable CODEOWNERS. Tests: strategy-only diff passes;
   `store.py`/`tests/`/`pyproject.toml` reject; CODEOWNERS-derived path rejects; **a rename of a
   denylisted source into an allowlisted destination rejects (R5); a copy of protected code into an
   allowlisted destination rejects (R5)**; **a symlink or gitlink entry under `algua/strategies/`
   rejects**; **a case-folded/NFD variant of a denied path rejects**; malformed CODEOWNERS fails
   closed. *Test file:* `tests/operator/test_diff_policy.py`.

3. **`algua/operator/gitops.py` — `GitOps` protocol + subprocess impl (incl. C4/R1 content check + R2
   CAS push).** Methods: `current_branch`, `working_tree_clean`, `resolve(ref)->sha`,
   `merge_base(a,b)`, `changed_entries(base, tip)` (→ `DiffEntry(mode, change_type, old_path,
   new_path)` via `git diff --raw -M -C`, R5), `merge_no_commit(tip)`, `merge_abort`, `commit_no_edit`,
   `commit_second_parent(sha)`, **`fetch_remote(ref)` and `remote_tip(ref)->sha`** (R1 — refresh and
   read `refs/remotes/origin/main`), `is_ancestor(sha, ref)`, `revert_merge(sha)`, **`push_cas(merge_sha,
   expected_base) -> bool`** (R2 — fetch, assert `origin/main == expected_base`, `git push origin
   <merge_sha>:refs/heads/main`, re-fetch and assert `origin/main == merge_sha`; returns False / raises
   on any CAS failure), `remote_has(sha)`, and for C4/R1 `tree_blobs(sha, paths) -> {path: blob_sha}` +
   `blob_at(ref, path) -> blob_sha|None` (used with `ref="origin/main"`). Thin subprocess wrapper;
   branch-tip-SHA identity + 2nd-parent verification + `origin/main`-fetched blob-equality content
   check + remote CAS live here. Tested via `FakeGit` in the orchestrator tests plus a real-git smoke
   test on a temp repo with a bare-repo "origin" (**incl. a symlink/gitlink diff entry, a rename
   old→new diff entry, a revert-then-content-mismatch case, and a stale-base CAS-reject case**).
   *Test file:* `tests/operator/test_gitops_smoke.py`.

4. **`algua/operator/mergeback.py` — pure `run_merge_back(...)` orchestrator (findings #1,#2,#4,#5,#6,#7,#9,#10 + C1–C6 + R1,R2,R4,R5).**
   The state machine over the injected seams + `GitOps`/`Journal` + `audit_log` + repo-global
   git-lock handle. Encodes: journal-first resume with stage cross-check fail-closed; branch-tip
   identity + revert-aware merge verification **against the freshly-fetched `origin/main` (R1)**;
   **C4/R1 effective-presence blob-equality check against the fetched `origin/main` tip before PROMOTE
   and before INTAKE**; **remote-CAS push (`push_cas`: pre-fetch `origin/main == base_sha`,
   explicit-SHA push, post-fetch `origin/main == merge_sha`, fail closed as `remote_moved`, R2)**;
   strict-agent promote inputs with `attempt_token` whose derivation folds the relaxation fingerprint
   (R4); **promote-outcome read from registry by `attempt_token` + strict-context equality (C1/R4),
   not by captured-max id, not from raise**; token derived+journalled before the promote call (C2);
   git-only revert with explicit discard terminals; push-before-promote ordering; **repo-global git
   lock around MERGE/PUSH/REVERT with fixed per-strategy→global acquire order (C3)**; **audit-log emit
   around each push/revert (C6)**; target-verified intake outcome; expected-stage CAS wiring.
   Exhaustively unit-tested with `FakeGit`/fake journal/stubs — one test per resume branch and per
   crash point (after diff-reject, after gate-fail, after merge pre-push, after push pre-promote,
   after promote-commit pre-intake, after intake-allocated, after intake-queued, promote post-commit
   raise, stage-drift), **plus: concurrent-same-strategy foreign passing row is NOT attributed
   (C1/R4); content-mismatch against `origin/main` at PROMOTE and at INTAKE fails closed (C4/R1);
   pre-push CAS stale-base and post-push remote-moved both fail closed (R2); global-git-lock
   contention serializes (C3)**. *Test file:* `tests/operator/test_mergeback.py`.

5. **Read-only registry helpers.** The token-keyed `passing_gate_by_token(conn, strategy,
   attempt_token) -> int | None` (from Task 0) and a target-allocation read (reuse
   `active_allocation`/`active_paper_lane_count` where possible) exposed via the repository Protocol —
   read-only. Tests cover "row bearing our token ⇒ ours" and "foreign/NULL-token row for same
   strategy ⇒ not ours." *Test file:* `tests/operator/test_promote_state_read.py`.

6. **`_run_intake` extraction + target-verified emit (finding #6).** Lift the FIFO intake
   orchestration out of `paper intake` into a shared helper both commands call (no dual path), then
   have the driver re-read THIS strategy's allocation to emit `promoted_allocated` vs
   `promoted_queued`. *Test file:* `tests/cli/test_paper_intake.py` (extend) + covered in task 4.

7. **`paper merge-back` CLI wiring in `paper_cmd.py` (CODEOWNERS — PR stays OPEN for human merge).**
   Acquire the per-strategy flock **and** open the repo-global `merge_back.git.lock` handle in the
   fixed order (C3), wire the concrete `GitOps`/`Journal`/registry seams + the structured `audit_log`
   sink (C6) into `run_merge_back`, hard-wire strict-agent promote inputs + `attempt_token`, run the
   **C6/R3 push-boundary preflight** (refuse to run unless: the scoped bot credential is present;
   remote branch protection is verifiable with **force-push + deletion blocked and
   `required_linear_history` OFF**, R6; **and a server-side path-restriction push ruleset / pre-receive
   hook is confirmed present**, R3 — query the GitHub rulesets API and fail closed if the path rule
   cannot be confirmed), and emit the terminal outcome JSON. Thin glue; logic is in task 4. *Test
   file:* `tests/cli/test_paper_mergeback.py` (seam-stubbed, incl. preflight-refuse when the path-rule
   / branch-protection probe is unverifiable, and audit-emit assertions).

**Integration:** run the FULL gate (`uv run pytest -q && uv run ruff check . && uv run mypy algua &&
uv run lint-imports`) once all tasks land, before opening the PR. Because `paper_cmd.py` **and the
Task 0 CODEOWNERS-protected registry files** (`store.py`/`promotion.py`/`db.py`) are touched, the PR
is left **OPEN for human merge** (auto-merge disabled).
