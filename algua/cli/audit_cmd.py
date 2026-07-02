from __future__ import annotations

from datetime import UTC, datetime

import typer

from algua.audit import log as audit_log
from algua.cli._common import registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors

audit_app = typer.Typer(help="Operational audit trail (read-only)", no_args_is_help=True)
app.add_typer(audit_app, name="audit")


def _norm_ts(value: str) -> str:
    """Normalize a user-supplied timestamp to a canonical UTC ISO-8601 string.

    The audit_log stores ``ts`` as ``datetime.now(UTC).isoformat()`` (always
    ``+00:00``), so filter bounds must compare against that exact format for the
    text comparison to be chronologically correct. A naive input is assumed UTC;
    an offset-aware input is CONVERTED to the true UTC instant. Unparseable input
    raises ``ValueError`` (rendered as the JSON error envelope by @json_errors).
    """
    dt = datetime.fromisoformat(value)
    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return dt.isoformat()


@audit_app.command("log")
@json_errors
def log(
    strategy: str = typer.Option(None, "--strategy", help="filter by strategy name"),
    actor: str = typer.Option(None, "--actor", help="filter by actor (agent|human|system)"),
    action: str = typer.Option(None, "--action", help="filter by action"),
    since: str = typer.Option(
        None, "--since", help="inclusive lower bound on ts (ISO-8601, assumed UTC if naive)"),
    until: str = typer.Option(
        None, "--until", help="exclusive upper bound on ts (ISO-8601, assumed UTC if naive)"),
    limit: int = typer.Option(100, "--limit", help="maximum rows to return (>=1)"),
    offset: int = typer.Option(0, "--offset", help="rows to skip (pagination)"),
    all_: bool = typer.Option(
        False, "--all", help="return every matching row (ignores --limit)"),
) -> None:
    """Query the operational audit trail, most-recent-first. Emits a bare JSON array.

    The audit_log is the append-only record of who did what (kill-switch trips,
    flattens, halts, venue-ingest failures). This is the read seam over that trail
    for incident review; it never writes.
    """
    if all_ and limit != 100:
        raise ValueError("pass either --all or --limit, not both")
    effective_limit = None if all_ else limit
    with registry_conn() as conn:
        rows = audit_log.read(
            conn,
            strategy=strategy,
            actor=actor,
            action=action,
            since=_norm_ts(since) if since is not None else None,
            until=_norm_ts(until) if until is not None else None,
            limit=effective_limit,
            offset=offset,
        )
    emit([dict(r) for r in rows])
