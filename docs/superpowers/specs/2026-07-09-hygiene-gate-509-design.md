# Structural CI Hygiene Gate — placement / provenance / no-junk (#509)

**Date:** 2026-07-09
**Branch:** `hygiene-gate-509`. **Status:** Design (scoped to what this PR actually ships).
**Severity:** medium — sustainability/hygiene for unattended operation, not a correctness/live-money
defect. Pairs with #510 (retention/GC of what already exists).

> **Scope note.** This document describes ONLY what this PR ships. A tighter boundary
> (top-level-directory enforcement, per-segment junk matching, `CONFIG.name == stem` name agreement,
> a generated-report stamp rule, and an `algua strategy verify` + `doctor` registry-consistency
> probe) was scoped out and is tracked in **#511**. The rationale for each deferred piece — including
> the DB-free CI split forced by GATE-1 — lives in §7 (Deferred). CLAUDE.md's command surface is NOT
> touched here (no new command ships); the `algua strategy verify` line lands with #511.

## 1. Problem

The autonomous operator authors FILES (strategy modules, kb notes, reports, specs) with no
enforcement of where they go, how they are named, or whether they are junk. The sqlite registry
(strategies/gates/allocations/ledgers) is the clean, governed system of record, but the *file*
surfaces have no hygiene gate, so the tree silts up under multi-PR-per-day churn. Evidence already
on main: `scratchpad-spec-332.md` at the repo root; untracked `kb/experience/`, `kb/strategies/`,
`kb/.sync.lock` polluting `git status`; `.gitignore` covers only a handful of dirs; the
review-gated workflow manually excludes `kb/` junk on every commit.

## 2. Shape (mirrors the #277 AST wall-scanner)

One canonical structural test, `tests/test_repo_hygiene.py`, with every policy as an **editable
module-level constant** so tightening/loosening is a one-line, code-reviewed change. It is the hard
CI enforcement layer; the existing advisory `algua doctor` `generated_provenance` probe
(`algua/cli/app.py::_generated_provenance_detail`) stays as-is. The test imports **no** `algua.*`
runtime module that opens the DB — it uses `subprocess`/`git ls-files`, `pathlib`, `ast`, and
`fnmatch` only.

**Source of truth = `git ls-files` (tracked paths only).** CI cares only about what is committed;
enumerating the git index (not an `rglob` of the working tree, contrast the #277 data-wall scanner)
avoids false positives from local untracked scratch files a developer legitimately keeps in the
tree. All three tests share the one `_tracked_files()` helper.

## 3. The three shipped rules

### Rule 1 — repo-root file whitelist (`test_repo_root_is_whitelisted`)
Every tracked path with **no `/`** (i.e. a repo-root file) must have its **basename** in
`ROOT_WHITELIST`; anything else fails closed (catches `scratchpad-spec-332.md`). A new legitimately-
root file is added by a one-line, reviewed edit to `ROOT_WHITELIST` — a whitelist miss is treated as
"needs review", not "definitely junk" (the same intended brittleness #277 uses for import
boundaries).

This governs repo-root **files only**. Governing new top-level *directories* (junk one dir deep) is
explicitly deferred to #511 — see §7.

### Rule 2 — known junk never tracked (`test_no_tracked_junk`)
No tracked path's **basename** may match any pattern in `JUNK_PATTERNS`
(`*.sync.lock`, `scratchpad-*`, `.DS_Store`, `Thumbs.db`, `*.swp`, `*.swo`, `*~`), matched with
`fnmatch`. Matching against every path *segment* (so `foo/bar.sync.lock` also fails) is deferred to
#511.

### Rule 3 — strategy placement + provenance (`test_strategies_placement_and_provenance`)
Scoped to tracked paths under `algua/strategies/`. A *strategy module* is a tracked
`algua/strategies/<family>/<name>.py` where `<name>` is not `__init__` and not `_`-prefixed. For
each:
- **placement:** a non-infra module directly at the strategies top level is an unplaced strategy and
  fails (only `INFRA_TOP_LEVEL = {__init__.py, base.py, loader.py}` are allowed there).
- **provenance:** the module assigns a module-level `GENERATED_BY` (AST scan, plain or annotated —
  the exact `_has_generated_by` predicate mirrored from `app.py`). This is a **presence-only** check;
  the stamp's *value* is not inspected.

A `CONFIG.name == filename-stem` name-agreement check is **not** part of this PR (deferred to #511)
— it is DB-adjacent (the honest form pairs it with a registry comparison) and belongs with the
`strategy verify` work.

## 4. Backfill `GENERATED_BY` into committed strategies (unblocks the hard check)

None of the 5 currently-committed strategy modules carried `GENERATED_BY` —
`cross_sectional_momentum.py`, `fundamentals_earnings_tilt.py`, `model_linear_scores.py`,
`model_scaled_linear.py`, `news_coverage_tilt.py`. They are hand-authored seed examples, so this PR
adds `GENERATED_BY = "human"` to each (the value states the honest provenance; the check requires
only that the marker *exists*). This makes Rule 3 green on main from the first commit — no
grandfather list, additions-only discipline stays intact.

## 5. `ROOT_WHITELIST` (shipped set)

```python
ROOT_WHITELIST = frozenset({
    "AGENTS.md", "CLAUDE.md", "CODEOWNERS", "docker-compose.yml", "Dockerfile",
    ".dockerignore", ".env.example", ".gitignore", ".gitleaks.toml",
    ".pip-audit-ignore.txt", "pyproject.toml", "README.md", "uv.lock",
})
```

These are exactly the 13 files tracked at the repo root today. Adding a root file is a deliberate act
— extend the set on purpose, or (preferably) place the file in a subdirectory.

## 6. `.gitignore` additions (stop churn at the source) + the tracked-only caveat

Append the runtime churn the operator writes so it never reaches `git status` / an accidental
`git add`:

```gitignore
# operator runtime churn (issue #509)
scratchpad-*
kb/**/.sync.lock
kb/experience/
```

**Caveat:** the structural test runs on a **clean checkout** and can only fail **tracked** junk.
Untracked local pollution (a stray `kb/.sync.lock`, an un-added `scratchpad-foo.md`) is *not*
something CI observes; that is precisely what these `.gitignore` lines prevent from ever being
staged. The two mechanisms are complementary: `.gitignore` keeps junk untracked; the CI test fails
junk that slipped past into tracking.

## 7. Deferred to #511 (with rationale)

The initial design proposed a wider boundary. To keep this PR small and each tightening independently
reviewable, the following move to **#511**:

1. **Top-level directory enforcement (Rule A dir half).** An `ALLOWED_ROOT_DIRS` allowlist so a
   tracked path under a *new, un-allowlisted* top-level dir (`tmp/`, `experiments/`, `scratch/`,
   `out/…`) fails closed — turning a root-file whitelist into a genuine repo-hygiene boundary. This
   is what makes `.superpowers/sdd/task-11-report.md` (a tracked-but-gitignored stray) a violation;
   the fix (`git rm --cached` that file) rides with this deferred rule, not this PR. This PR does not
   red on that file because the shipped root rule governs only files with no `/`.
2. **Per-segment junk matching (Rule D).** Match `JUNK_PATTERNS` against every path segment, not just
   the basename, so `foo/bar.sync.lock` and `scratchpad-x/keep.md` fail.
3. **Name agreement (Rule B3).** `CONFIG.name` (AST string literal) == filename stem, DB-free.
4. **Generated-report stamp (Rule C).** A `> Generated <stamp> from MLflow …` header requirement on
   `kb/strategies/*/reports/*/report.md`, with an explicit baseline-allow escape hatch. Authored
   `docs/superpowers/{specs,plans}/*.md` are excluded by design (human-authored, not machine-
   stamped).
5. **DB↔file consistency via `algua strategy verify` + `doctor`.** CI has no runtime registry
   (`data/algua.db` is gitignored, and its open path migrates/creates), so the cross-store invariant
   cannot live in a static test. #511 adds an operator-run `algua strategy verify` (JSON on stdout)
   enforcing the **asymmetric** invariant — row→file REQUIRED (a governed non-retired row must resolve
   to a matching module: `registry.name == CONFIG.name == filename-stem` AND on-disk family dir ==
   `family_package_dir(registry.family)`), file→row ADVISORY (a bundled/unregistered module is a
   normal pre-registration state; the 5 bundled examples are unregistered on a fresh DB, which
   `doctor` must still pass). It is wired into `doctor` as a `strategy_registry_consistency` probe
   reported under a `registry_integrity` key. CLAUDE.md's command surface gains the `algua strategy
   verify` line when #511 lands.

## 8. Non-goals / deferred
- No retention/GC of already-present files — that is #510.
- No change to the advisory `doctor` provenance probe.
- The tighter boundary and the `strategy verify`/`doctor` probe — **#511** (§7).

## 9. Decomposition / build order
See the companion plan `docs/superpowers/plans/2026-07-09-hygiene-gate-509.md`.
