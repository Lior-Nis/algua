"""``record_tracking`` -- the best-effort tracker-logging side effect shared by every task body
that produces a payload worth tracking.

Lives in the tracking layer rather than in ``cli/backtest_cmd`` because non-CLI lanes
(``algua.evaluation``) run the same task bodies and need the same four-state semantics.
Duplicating it per lane is what this placement prevents.

Warns on stderr with ``print`` rather than ``typer.echo``: this layer must not import the CLI
framework, and for a plain string the two are equivalent.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from algua.tracking.base import TRACKING_SKIPPED


def record_tracking(payload: dict, call: Callable[[], str]) -> None:
    """Best-effort tracker logging. The result already exists by the time this runs, so a
    tracker failure (flaky URI, serialization bug) must NOT discard a completed evaluation (#341).
    On success record `mlflow_run_id`; on failure record a non-fatal `mlflow_tracking_error` (with
    the exception type for triage) and warn to stderr — never raise. Keys are added ONLY when
    tracking was requested, so the JSON distinguishes FOUR states: not-requested (no keys),
    succeeded (`mlflow_run_id` set, no error), failed (`mlflow_run_id` null + error), and skipped
    (`mlflow_run_id` null + `mlflow_tracking_skipped`, when a no-op backend was selected — distinct
    from failed so a null run id never has to be read as "the error key went missing")."""
    try:
        run_id = call()
    except Exception as exc:  # noqa: BLE001 - tracking is a best-effort side effect
        detail = f"{type(exc).__name__}: {exc}"
        payload["mlflow_run_id"] = None
        payload["mlflow_tracking_error"] = detail
        print(f"warning: mlflow tracking failed (result preserved): {detail}", file=sys.stderr)
        return
    if run_id == TRACKING_SKIPPED:
        # A no-op backend was selected. Report that honestly rather than as a null run id, which
        # would be indistinguishable from a failure whose error key went missing.
        payload["mlflow_run_id"] = None
        payload["mlflow_tracking_skipped"] = "tracking backend logs nothing"
        return
    payload["mlflow_run_id"] = run_id
