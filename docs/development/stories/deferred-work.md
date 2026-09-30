# Deferred work

## Deferred from: code review of 1-1-extract-in-process-decision-planner (2026-09-24)

- `algua/live/planner.py::build_intents`: a NaN current holding weight makes the delta comparison
  false, silently omitting that symbol's intent. This behavior existed in `paper_loop.build_intents`
  before extraction. Review risk-input validation separately; do not silently change fail/flatten
  policy while preserving behavior in Story 1.1. No production incident was established.
- `algua/live/planner.py::build_intents`: shared validation permits zero-weight out-of-universe
  labels; mixing non-string labels with string holdings can then raise `TypeError` while sorting.
  Pre-existing before extraction. Review the shared symbol-label/error contract separately rather
  than adding inconsistent planner-only validation. No new exposure bypass was established.

## Resolved: deferred from code review of 1-3b-materialize-and-verify-recoverable-planner-artifacts (2026-09-28)

- `algua/registry/planner_environment.py::SYNC_FLAGS`: the keyed `uv sync` argv carried
  `--no-env-file`, which uv 0.9.26 accepts only for `uv run`, so real locked provisioning could
  never succeed (Todoist 6hfM2j4PvvqFgHMp). Resolved 2026-09-30 by dropping the flag (`uv sync`
  never reads `.env` files). The argv now equals the companion's argv block exactly, and a real
  offline dry run of it against the committed lock is part of the root gate.

## Superseded: deferred from code review of 1-3b-materialize-and-verify-recoverable-planner-artifacts (2026-09-28, `47062b9`)

- `tests/primitives/test_scoped_walk.py:517` (suppressing context managers, Todoist
  6hfPJj8H8RXq3vmG) and `tests/primitives/test_scoped_walk.py:486` (class-body `global`, Todoist
  6hfPJj9J9jQVvpFG), line numbers as of `22aac24`: superseded 2026-09-30, not implemented. The raw
  walk is now module-private (`_bounded_walk`) and the flow analyzer they targeted is removed; see
  the Story 1.3b rescope decision.

## Deferred from: code review of 1-3b-materialize-and-verify-recoverable-planner-artifacts (2026-09-30)

- `.github/workflows/ci.yml:13`: nothing pins the uv version (no `required-version`; CI installs the
  latest uv). The argv tests exercise whichever uv production would run, and the environment key
  records the uv version, so drift separates environments rather than corrupting one. Pinning the
  toolchain is a repository-wide decision outside this story.
