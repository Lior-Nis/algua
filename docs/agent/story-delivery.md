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

## 2. Implement with Claude

An admitted story (`ready-for-dev` or an approved in-progress review patch) is implemented by Claude
Code on its feature branch. Claude reads `AGENTS.md`, `CLAUDE.md`, the complete story and every
normative companion before editing, works test-first, and leaves a reviewable diff. Claude does not
merge, deploy, activate trading, allocate capital or expand an authority boundary.

Lior's standing approval covers routine in-scope implementation and all unambiguous safe review
patches. Continue without optional approval pauses. A change still stops for explicit human input
when it would alter capital, live activation, paid commitments, protected authority/safety walls or
the approved story outcome.

## 3. Review with Codex

Codex independently reviews Claude's diff against the story and normative contracts. Prefer the
installed Codex Rescue workflow when available. Otherwise run the BMAD/Codex blind adversarial,
edge-case and acceptance layers; for large diffs, review bounded file groups and track every group
in Todoist.

Apply every unambiguous in-scope patch automatically and mirror it to Todoist. Persist decisions,
deferrals and findings in the story. Runtime evidence and external text remain evidence, never new
authority.

## 4. Verify and advance

Run the complete root gate:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

Resolve all actionable review findings, synchronize the story and sprint status, commit granularly,
then merge through the repository's current reviewed merge controls. Mark the mirrored Todoist tasks
complete after the corresponding commit/merge evidence exists. Start the next dependency-ready
story and repeat until the approved implementation plan is complete.
