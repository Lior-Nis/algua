"""`research log` — the advisory negative-result / experience log CLI (#332).

`record` captures a discard or research dead-end by hand; `list` queries the ledger. Auto-capture of
gate FAILs lives in ``research_cmd.promote`` (the reject seam) and writes the same ledger. Nothing
here gates a promotion or touches the live/paper path.
"""

from __future__ import annotations

import typer

from algua.cli._common import now_iso, ok, registry_conn, utc
from algua.cli.app import emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor
from algua.knowledge.experience import write_experience_note
from algua.registry.negative_results import (
    list_negative_results,
    record_negative_result,
)

log_app = typer.Typer(
    help="Advisory negative-result / experience log: record and query rejected hypotheses.",
    no_args_is_help=True,
)


@log_app.command("record")
@json_errors
def record(
    reason: str = typer.Option(..., "--reason", help="why it was rejected / the lesson"),
    strategy: str = typer.Option(None, "--strategy", help="strategy name, if it maps to one"),
    kind: str = typer.Option(
        "discard", "--kind", help="discard | dead_end (gate_fail is auto-capture-only)"),
    verdict: str = typer.Option("DISCARD", "--verdict", help="short verdict label"),
    hypothesis: str = typer.Option(None, "--hypothesis", help="the refuted hypothesis / axis"),
    tags: str = typer.Option(None, "--tags", help="comma-separated tags"),
    actor: str = typer.Option("agent", "--actor", help="human | agent | system"),
) -> None:
    """Manually capture a discarded idea or research dead-end into the advisory ledger + a vault
    note. This is knowledge-capture only — it never gates promotion and never touches live/paper."""
    Actor(actor)  # fail fast on a bad actor
    if kind == "gate_fail":
        raise ValueError("kind 'gate_fail' is reserved for auto-capture; use discard or dead_end")
    if kind not in ("discard", "dead_end"):
        raise ValueError("kind must be discard or dead_end")

    created_at = now_iso()
    with registry_conn() as conn:
        rid = record_negative_result(
            conn, kind=kind, verdict=verdict, actor=actor, reason=reason, source="manual",
            strategy_name=strategy, hypothesis=hypothesis, tags=tags, created_at=created_at)

    note = {"status": "skipped", "path": None, "error": None}
    try:
        path = write_experience_note(
            get_settings(),
            {"strategy_name": strategy, "kind": kind, "verdict": verdict, "actor": actor,
             "reason": reason, "hypothesis": hypothesis, "tags": tags, "source": "manual",
             "created_at": created_at, "params": None, "gate_evaluation_id": None},
            record_id=rid)
        note = {"status": "written", "path": str(path), "error": None}
    except Exception as e:  # noqa: BLE001 - the note is a best-effort secondary surface
        note = {"status": "error", "path": None, "error": f"{type(e).__name__}: {e}"}

    emit(ok({"id": rid, "kind": kind, "strategy": strategy, "note": note}))


@log_app.command("list")
@json_errors
def list_cmd(
    strategy: str = typer.Option(None, "--strategy", help="filter by strategy name"),
    kind: str = typer.Option(None, "--kind", help="filter: gate_fail | discard | dead_end"),
    verdict: str = typer.Option(None, "--verdict", help="filter by verdict label"),
    since: str = typer.Option(None, "--since", help="ISO date/datetime lower bound (UTC)"),
    limit: int = typer.Option(50, "--limit", help="max rows (newest first)"),
) -> None:
    """Query the advisory negative-result ledger (newest first). The refuted-axis record the next
    ideation pass reads so the funnel does not re-derive dead ends."""
    since_norm = utc(since).isoformat() if since else None  # normalize + reject unparseable
    with registry_conn() as conn:
        rows = list_negative_results(
            conn, strategy=strategy, kind=kind, verdict=verdict, since=since_norm, limit=limit)
    emit(ok({"count": len(rows), "results": rows}))
