# Stage 6c — Registry Extractions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish §6 item 4. Move the book-intake loop out of the CLI to join the pure helpers it already imports, and split two oversized registry gate modules along the seams their own structure already implies.

**Architecture:** Three moves, no new packages. `paper_cmd._run_intake` (+ its two private helpers) joins the existing `algua/registry/intake.py`. `promotion.py`'s 208-line family-classification body becomes `registry/family_assignment.py`. `forward_promotion.py` splits into `registry/forward_evidence.py` (tick admissibility + evidence assembly) and `registry/live_certificate.py` (`verify_forward_certificate`), preserving the preflight → guard → run_gate → reason symmetry with `promotion.py`.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §6 item 4, bullets 2, 6 and 7.

**Ground truth:** a research pass against `main`@`44e3be0`. Every size and dependency below was measured, not estimated.

### Decisions (recorded for a reviewer to check, not re-derive)

1. **`registry/intake.py` ALREADY EXISTS (64 lines) and is the right destination.** It holds the pure
   half — `Candidate`, `slice_capital`, `order_candidates` — and `paper_cmd.py:104` already imports
   from it. This is precisely what the spec means by *"joins its own pure helpers"*: the orchestration
   loop rejoins the module whose pure functions it drives. **Do not create a new module**, and do not
   confuse it with the unrelated `registry/mergeback_intake.py`.
2. **`_run_intake`'s dependency closure is clean — the Stage 6a cascade does NOT recur.** Measured: it
   uses `Candidate`, `order_candidates`, `slice_capital`, `SqliteStrategyRepository`,
   `intake_candidate_to_paper`, `active_paper_lane_count`, `allocate_in_lane`, `audit_append` — all
   registry/audit/allocation. **Zero `cli/_common` dependencies.** Two private helpers travel with it
   (`_candidate_entry_id` ~10 lines at `paper_cmd.py:146`, `_unallocated_book_tenants` ~17 lines at
   `:156`), and **neither is referenced anywhere outside `paper_cmd.py`** — verified across `algua/`
   and `tests/`.
3. **The forward split follows the module's existing internal seam, not an arbitrary line count.**
   `forward_promotion.py` (665 lines) already separates into: helpers + `assemble_forward_evidence`
   (`:88`–`:409`, the evidence half) and `verify_forward_certificate` (`:411`–`:529`, the certificate
   half), with the gate scaffolding (`guard_forward_relaxations`, `forward_promotion_preflight`,
   `run_forward_gate`, `_forward_gate_reason`) after it. The spec asks to preserve the
   preflight → guard → run_gate → reason symmetry with `promotion.py`, so **that scaffolding stays in
   `forward_promotion.py`** — only the two bodies and their private helpers leave.
4. **`SessionCalendar` (`forward_promotion.py:59`) stays put.** Stage 5c deliberately left it alone:
   it is a 4-method Protocol with **zero** overlap with `contracts.SessionSpanCalendar`. Whichever new
   module needs it imports it; do not merge, move, or "harmonize" it.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. Baseline is **25 contracts, 0 broken**.
- **The full suite takes ~7–10 minutes on this machine.** Pass `timeout: 600000` explicitly to the Bash tool. If the harness backgrounds the run anyway, **do NOT end your turn waiting** — read the output file the harness names, or re-run in the foreground. Seven agents on this program have stalled exactly this way. To check whether a real pytest is live, match the **executable** (`readlink /proc/PID/exe`), not `pgrep -f pytest` — the cmdline match is polluted by leaked test processes whose data-dir path contains "pytest" (issue #588).
- **A "pure move" NEVER licenses duplication to satisfy it.** SIX instances across stages 6a/6b, every one undone. If a moved body needs a shared helper, **the helper moves to its owning layer**. If you catch yourself writing "local equivalent of X" / "private copy of X" / "byte-identical duplicate of X" / inlining a one-liner to dodge an import — stop.
- **Every scripted edit must assert it changed something.** An unguarded `.replace()` that silently misses is indistinguishable from one that succeeded; in Stage 6a that shipped a duplicated helper past a commit.
- **Probe, don't reason, about layering.** Stage 6b placed a module exactly where the spec said and a throwaway import-linter contract proved it created a transitive inversion no one saw by reading imports. Task 4 does this for 6c.
- **Regenerate every count untruncated at the moment you assert it** — no `| head`/`| tail` on an enumeration you are counting.
- Moved bodies keep their names, signatures, `# noqa` codes, docstrings and comments **verbatim**. These modules are the paper→live wall: the comments encode the holdout burn/release saga, the single-use gate-token semantics, the `attempt_token` attribution, and the family-mint CAS. Losing one loses the reason a wall exists.
- **CODEOWNERS-protected files this stage touches: `algua/cli/paper_cmd.py`, `algua/registry/promotion.py`, `algua/registry/forward_promotion.py`.** All three are on the paper→live wall — this is the most safety-adjacent stage of the program. Note it prominently in the PR.
- `git add` scoped to named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known hazard: some test writes demo strategy files into `algua/strategies/momentum/`. Check `git status --porcelain` before staging and delete any stray untracked file there.

---

### Task 1: Move the intake loop into `algua/registry/intake.py`

**Files:**
- Modify: `algua/registry/intake.py`, `algua/cli/paper_cmd.py`, `tests/test_cli_merge_back.py`

**Interfaces:**
- Produces: `run_intake(conn, *, equity, max_concurrent, actor) -> dict` — today's exact signature at `paper_cmd.py:417`, minus the leading underscore (it stops being module-private).

- [ ] **Step 1: Read the destination first**

Read `algua/registry/intake.py` in full (64 lines). Its module docstring states the invariant that makes this safe: *the admission DECISION is not made here* — it is made transactionally by `StrategyRepository.intake_candidate_to_paper`, which re-checks the count cap and the Σ≤equity bound under the write lock. The module computes only the fixed slice and stable order. **Your moved loop must not weaken that**: it still offers candidates one at a time to the atomic primitive and never does its own bounds-check.

- [ ] **Step 2: Move all three bodies verbatim**

`_run_intake` (~99 lines, `paper_cmd.py:417`) → `run_intake`.
`_candidate_entry_id` (~10 lines, `:146`) and `_unallocated_book_tenants` (~17 lines, `:156`) travel with it, keeping their underscore (still private, now to `intake.py`).

Re-verify each line number against the live file before editing.

- [ ] **Step 3: Re-point the two internal callers**

`paper_cmd.py:556` and `:782`. Import `run_intake` from `algua.registry.intake`.

- [ ] **Step 4: Re-point the test pin — and PROVE it still fires**

`tests/test_cli_merge_back.py:157` does `monkeypatch.setattr(paper_cmd, "_run_intake", _fake_intake)`. After the move, `paper_cmd` no longer owns that attribute, so patching it would silently stop covering the merge-back path.

Re-point it at the binding `paper_cmd` actually reads, then verify in **both** directions with `PYTHONDONTWRITEBYTECODE=1` and `__pycache__` cleared:
- make `_fake_intake` raise → the merge-back tests using it must **FAIL**
- restore → they must **PASS**

Report the observed counts. **This is the failure mode that silently disarmed a go-live guard in Stage 5b**: a patch that resolves, binds a real attribute, and covers nothing. Do not report this step as "confirmed" without numbers.

- [ ] **Step 5: Full gate, then commit.**

---

### Task 2: Extract family classification to `algua/registry/family_assignment.py`

**Files:**
- Create: `algua/registry/family_assignment.py`
- Modify: `algua/registry/promotion.py` (CODEOWNERS-protected)

**Interfaces:**
- Produces: `classify_and_assign_family(...)` and `get_all_family_members_for_clustering(...)` — today's exact signatures, underscore dropped since they cross a module boundary.

- [ ] **Step 1: Read both bodies and the caller**

`_get_all_family_members_for_clustering` (`promotion.py:103`, ~11 lines) and `_classify_and_assign_family` (`:114`, ~208 lines). Read `promotion_preflight` (`:323`) to see how it calls them.

This body implements #222/#524 family governance: the MERGE / PARENTAGE / NOVEL verdicts, the deferred pass-time mint, the `AGENT_NOVEL_MINT_CAP` rate bound, and the family-graph fingerprint CAS re-check under the write lock. **Every comment is load-bearing** — the deferred-mint design exists because a repeated-founder attack defeats the naive version, and that reasoning lives only in these comments.

- [ ] **Step 2: Move both verbatim**

Keep the private `_`-prefixed helpers they call private, moving any that are used ONLY by these two. **Regenerate that list untruncated** rather than trusting this plan — if a helper has another caller in `promotion.py`, it stays and is imported.

- [ ] **Step 3: Check for a circular import**

`family_assignment.py` will import from `registry/store` and `registry/db`; `promotion.py` imports `family_assignment`. Verify no cycle: `uv run python -c "import algua.registry.promotion"` must succeed. If it does not, report rather than papering over it with a function-level import.

- [ ] **Step 4: Full gate, then commit.**

---

### Task 3: Split `forward_promotion.py` along its existing seam

**Files:**
- Create: `algua/registry/forward_evidence.py`, `algua/registry/live_certificate.py`
- Modify: `algua/registry/forward_promotion.py` (CODEOWNERS-protected)

**Interfaces:**
- `forward_evidence.py` produces: `AssembledEvidence`, `assemble_forward_evidence(...)`, `qualified_holdout_sharpe(...)` and the private helpers used only by them.
- `live_certificate.py` produces: `verify_forward_certificate(...)`.
- `forward_promotion.py` KEEPS: `SessionCalendar`, `guard_forward_relaxations`, `forward_promotion_preflight`, `ForwardPromotionOutcome`, `run_forward_gate`, `_forward_gate_reason`.

- [ ] **Step 1: Confirm the seam before cutting**

Measured layout on `main`@`44e3be0`:
```
:59  SessionCalendar (Protocol)         STAYS
:74  AssembledEvidence                  -> forward_evidence
:88  _parse_dt   :105 _identity_matches
:116 _inadmissible_reason  :140 _classify_activities   -> forward_evidence (verify each caller)
:165 qualified_holdout_sharpe           -> forward_evidence
:185 assemble_forward_evidence (~225)   -> forward_evidence
:411 verify_forward_certificate (~119)  -> live_certificate
:531 guard_forward_relaxations          STAYS
:557 forward_promotion_preflight        STAYS
:577 ForwardPromotionOutcome            STAYS
:583 run_forward_gate                   STAYS
:656 _forward_gate_reason               STAYS
```
**Re-verify every line number and every private helper's callers untruncated.**

**One cross-half helper is already known and decided, so you do not rediscover it mid-move:**
`_parse_dt` (`:88`) is used by **both** halves — measured 8 references: 7 in the evidence half
(`:88`, `:126`, `:131`, `:228`, `:255`, `:355`, `:364`) and 1 in the certificate half (`:469`).
It **goes to `forward_evidence.py`** (which owns 7 of the 8 uses) and `live_certificate.py`
**imports it**. It does NOT get copied, and it does NOT get inlined at the single certificate site —
that is duplication with the name filed off, and it is the exact shape that produced six separate
incidents in stages 6a/6b.

If you find any OTHER helper spanning both halves, apply the same rule: home it where the majority
of uses are, import it in the other, and say so in your report.

- [ ] **Step 2: Move the evidence half, then the certificate half.** Verbatim bodies; keep `# noqa` codes.

- [ ] **Step 3: Verify the structural symmetry the spec asks for**

After the split, `forward_promotion.py` should read as preflight → guard → run_gate → reason, mirroring `promotion.py`. Compare the two files' `grep -n "^def \|^class "` output and say in your report whether they now match. If they do not, say so plainly rather than claiming success.

- [ ] **Step 4: Full gate, then commit.**

---

### Task 4: Close-out verification

**Files:** none expected (verification only; fix anything found).

- [ ] **Step 1: The bodies left the CLI and the god-modules shrank**

Report before/after line counts for `paper_cmd.py` (**1551** on `main`@`44e3be0`), `promotion.py` (**773**), `forward_promotion.py` (**665**).

- [ ] **Step 2: Probe for layering inversions — do not reason about this**

This is the step that caught a real transitive inversion in Stage 6b. Temporarily add each contract, run `lint-imports`, record the result, and remove it:
- `forbidden: algua.registry -> algua.cli` — expected to FAIL on exactly ONE edge, `human_actor.py:210` (known debt, issue #592). **If it fails on anything else, this stage introduced it.**
- `forbidden: algua.registry -> algua.live` and `forbidden: algua.registry -> algua.backtest` — if either PASSES, add it permanently; a guarantee is worth only the contract enforcing it.

Report each probe's verdict and any violating chain verbatim.

- [ ] **Step 3: No duplication survived**

```bash
grep -rn "local equivalent\|private copy\|byte-identical duplicate\|reproduced here" --include='*.py' algua/
```
Plus: confirm each moved name has exactly ONE definition tree-wide.

- [ ] **Step 4: Full gate + CLI smoke** (`timeout: 600000`): `uv run pytest -q`, ruff, mypy, lint-imports, then `uv run algua doctor` and `uv run algua fleet status`.

- [ ] **Step 5: Commit any fixes.** If nothing needed fixing, make no commit — expected, and consistent with every prior close-out in this program.
