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
