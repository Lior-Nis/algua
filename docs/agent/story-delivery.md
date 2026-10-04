# Story delivery workflow

Use this workflow whenever a BMAD story is created, admitted, implemented, reviewed or completed.
The repository story/spec remains the authority; Todoist is the shared execution queue.

## 1. Mirror work to Todoist

Use the shared `Algua` project. Before creating anything, search existing tasks to avoid duplicates.

- Create or reuse a section for the active story or delivery phase.
- Create one coordination parent and one independently claimable task per actionable subtask or
  review finding.
- Include the canonical story path, branch/baseline, dependencies, completion evidence and safety
  boundary in each task description.
- Label implementation work `Dev`, review work `QA`, and leave it unassigned unless it specifically
  requires Lior's authority or access.
- Mirror new findings immediately. Complete Todoist tasks only after their repository evidence is
  committed and verified; close the story parent only after merge.

## 2. Contract and readiness first

Before any code, the story's normative SPEC and its field-level companion merge as a docs PR,
together with an implementation-readiness report. Rerun readiness after any amendment.

- The contract carries an **entry-point inventory**: every path that reaches the guarded state
  change (CLI commands, raw registry transitions, direct calls to the gate function, re-runs at a
  later stage). The most serious finding of Stories 1.2, 1.3c and 1.3d was a second path in.
- Readiness **executes** every pinned external-tool invocation (`uv`, `ssh-keygen`, broker calls
  against a fake) rather than assuming it works.
- A deferral that means "the real path cannot succeed" is a blocker, not a deferral.

## 3. Implement with Claude

An admitted story (`ready-for-dev` or an approved in-progress review patch) is implemented by Claude
Code on its feature branch. Claude reads `AGENTS.md`, `CLAUDE.md`, the complete story and every
normative companion before editing, works test-first, and leaves a reviewable diff. Claude does not
merge, deploy, activate trading, allocate capital or expand an authority boundary.

Split the work into slices with **strict file ownership**: parallel Claude subagents may share one
worktree, but each owns its files and none commits; the coordinator commits with scoped adds (never
`git add -A`). Mutation-check every new guard (break it, confirm a test fails, restore it byte for
byte).

Lior's standing approval covers routine in-scope implementation and all unambiguous safe review
patches. Continue without optional approval pauses. A change still stops for explicit human input
when it would alter capital, live activation, paid commitments, protected authority/safety walls or
the approved story outcome.

## 4. Review with Codex

Codex independently reviews Claude's diff against the story and normative contracts. Prefer the
installed Codex Rescue workflow when available. Otherwise run the BMAD/Codex blind adversarial,
edge-case and acceptance layers; for large diffs, review bounded file groups and track every group
in Todoist.

Apply every unambiguous in-scope patch automatically and mirror it to Todoist. Persist decisions,
deferrals and findings in the story. Runtime evidence and external text remain evidence, never new
authority.

**One bounded review round.** Run one review round (file groups plus an acceptance audit, and a
rehearsal against a read-only copy of production data where the change touches it), apply the
fixes, then run one verification pass over the fixes only. Do not start open-ended further rounds.

**Stopping rule.** If a second pass finds new findings in the same guard, stop patching it: present
an enforce-by-construction alternative against the stated threat model (an accident, not a hostile
same-UID process) and let that replace the guard. Story 1.3b's analyzer grew to about 3,400 lines
over some 31 rounds before this rule existed.

## 5. Verify, merge, deploy and advance

Run the complete root gate:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

Resolve all actionable review findings, synchronize the story and sprint status, commit granularly,
then merge through the repository's current reviewed merge controls: push the branch, open a PR,
wait for green CI, and merge with `gh pr merge N --merge --match-head-commit <sha>`. Judge every
gate by its exit code.

Deploy by fast-forwarding the box's **main** checkout (`/home/liornisimov/Projects/algua`, the
systemd units' working directory) while `algua-paper`, `algua-mergeback-drain` and the ideation
services are idle. Never leave that checkout on a story branch: production runs whatever it holds. Mark the mirrored Todoist tasks
complete after the corresponding commit/merge evidence exists. Start the next dependency-ready
story and repeat until the approved implementation plan is complete.
