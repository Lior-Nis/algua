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

## Deferred from: code review of 1-3b-materialize-and-verify-recoverable-planner-artifacts (2026-09-28)

- `algua/registry/planner_environment.py::SYNC_FLAGS`: the normative `uv sync` argv includes
  `--no-env-file`, which uv 0.9.26 accepts only for `uv run`; `uv sync` exits with `unexpected
  argument '--no-env-file'`, so real locked provisioning cannot succeed. Pre-existing and already
  tracked as an open Story 1.3b finding. Correcting it changes the keyed sync argv, the
  environment-key golden vector and the normative companion together, so it needs its own
  reviewed contract change; this review round leaves `SYNC_FLAGS`, the golden vectors and the
  normative sync argv unchanged.
