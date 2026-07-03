"""CLI seam for the SR 11-7 model-governance / validation-record inventory (issue #393).

Governance records live as governance-owned frontmatter on the strategy's kb-vault doc (see
``algua.knowledge.governance``). This module is the registry-binding seam: EVERY record/list/overdue
operation resolves the strategy against the registry first, so a governance record can never be
attached to — or reported for — a phantom strategy. ``governance record`` additionally verifies the
linked ``--gate-eval-id`` is a real ``gate_evaluations`` row belonging to THIS strategy.

``governance overdue`` exits non-zero when any registered strategy is overdue (or undocumented), so
a monitor / CI job can enforce the review cadence. Overdue is fail-closed: a missing or malformed
next-review date counts as overdue, never silently ok (see ``governance.is_overdue``).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.knowledge import governance
from algua.knowledge.sync import strategy_doc_path
from algua.registry.store import SqliteStrategyRepository

governance_app = typer.Typer(
    help="SR 11-7 model-governance / validation-record inventory", no_args_is_help=True
)
app.add_typer(governance_app, name="governance")


def _today() -> date:
    """The monitor's 'now' — UTC calendar date (never the process-local date)."""
    return datetime.now(UTC).date()


def _parse_date(label: str, value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


@governance_app.command("record")
@json_errors
def record(
    name: str = typer.Argument(..., help="registered strategy name"),
    owner: str = typer.Option(..., "--owner", help="accountable owner (SR 11-7 model owner)"),
    next_review: str = typer.Option(
        ..., "--next-review", help="scheduled next validation/review date (ISO YYYY-MM-DD)"),
    assumption: list[str] = typer.Option(
        None, "--assumption", help="intended-use assumption (repeatable)"),
    limitation: list[str] = typer.Option(
        None, "--limitation", help="known limitation / failure mode (repeatable)"),
    validation_summary: str = typer.Option(
        None, "--validation-summary", help="conceptual-soundness / validation summary"),
    last_validated: str = typer.Option(
        None, "--last-validated", help="date last validated (ISO YYYY-MM-DD)"),
    gate_eval_id: int = typer.Option(
        None, "--gate-eval-id", help="linked gate_evaluations.id (must belong to this strategy)"),
) -> None:
    """Record/refresh a strategy's governance dossier onto its kb-vault doc.

    Binds to a REAL strategy: the strategy must exist in the registry (else `not_found`) and have a
    kb doc (else `file_not_found`). A linked --gate-eval-id must be a real gate_evaluations row for
    THIS strategy, so governance for strategy A can never cite strategy B's (or a phantom) row.
    """
    next_review_d = _parse_date("--next-review", next_review)
    last_validated_d = _parse_date("--last-validated", last_validated) if last_validated else None

    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)  # raises StrategyNotFound if phantom
        if gate_eval_id is not None:
            row = conn.execute(
                "SELECT strategy_id, passed FROM gate_evaluations WHERE id=?", (gate_eval_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"no gate_evaluations row with id={gate_eval_id}")
            if row["strategy_id"] != rec.id:
                raise ValueError(
                    f"gate_evaluations id={gate_eval_id} belongs to "
                    f"strategy_id={row['strategy_id']}, not {name!r} (id={rec.id})"
                )
            if not row["passed"]:
                # The governance record cites the *passing* validation evidence; a failed gate row
                # is not evidence of conceptual soundness.
                raise ValueError(
                    f"gate_evaluations id={gate_eval_id} did not pass; cite a passing evaluation"
                )

    settings = get_settings()
    doc_path = strategy_doc_path(settings, name)
    written = governance.record_governance(
        doc_path, name,
        owner=owner,
        assumptions=assumption or [],
        limitations=limitation or [],
        validation_summary=validation_summary,
        next_review=next_review_d,
        last_validated=last_validated_d,
        gate_eval_id=gate_eval_id,
    )
    emit(ok(governance.record_to_json(written, today=_today())))


def _gate_eval_valid(conn, strategy_id: int, gate_eval_id: int) -> bool:
    """True iff `gate_eval_id` is a PASSING gate_evaluations row for THIS strategy. A stale,
    nonexistent, cross-strategy, or failed cited id is NOT valid evidence."""
    row = conn.execute(
        "SELECT strategy_id, passed FROM gate_evaluations WHERE id=?", (gate_eval_id,)
    ).fetchone()
    return row is not None and row["strategy_id"] == strategy_id and bool(row["passed"])


def _read_all(only_overdue: bool) -> tuple[list[dict], int]:
    """Read every REGISTERED strategy's governance record. Returns (rows, n_overdue).

    Scanning is registry-driven (not vault-driven) so a stale doc without a registry row is never
    reported, and — critically — a registered strategy with NO doc/record still shows up as overdue.

    A doc's cited ``gate_eval_id`` is RE-VALIDATED against the DB on every read (a hand-edited doc
    could cite a phantom/cross-strategy/failed row): an invalid citation is surfaced as
    ``gate_eval_valid: false`` AND forces the record fail-closed overdue, so unverifiable validation
    evidence never reads as healthy.
    """
    today = _today()
    settings = get_settings()
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        recs = sorted(repo.list_strategies(), key=lambda r: r.name)
        rows: list[dict] = []
        n_overdue = 0
        for rec_row in recs:
            name = rec_row.name
            rec = governance.read_governance(strategy_doc_path(settings, name), name)
            payload = governance.record_to_json(rec, today=today)
            # A cited gate id must be a passing, same-strategy row. A present-but-malformed id
            # (rec.gate_eval_id_malformed) is unverifiable and also invalid — never 'no citation'.
            if rec.gate_eval_id_malformed:
                gate_valid = False
            elif rec.gate_eval_id is not None:
                gate_valid = _gate_eval_valid(conn, rec_row.id, rec.gate_eval_id)
            else:
                gate_valid = True
            payload["gate_eval_valid"] = gate_valid
            if not gate_valid:
                payload["overdue"] = True  # unverifiable evidence => fail-closed
            if payload["overdue"]:
                n_overdue += 1
            if only_overdue and not payload["overdue"]:
                continue
            rows.append(payload)
    return rows, n_overdue


@governance_app.command("list")
@json_errors
def list_() -> None:
    """List every registered strategy's governance record (with a derived `overdue` verdict).

    Emits a bare JSON array. A registered strategy with no governance record appears with
    `present: false` and `overdue: true` — the undocumented-model view SR 11-7 requires.
    """
    rows, _ = _read_all(only_overdue=False)
    emit(rows)


@governance_app.command("overdue")
@json_errors
def overdue() -> None:
    """Flag strategies overdue for revalidation (or undocumented). EXITS NON-ZERO if any — the
    hook a monitor / CI cadence job enforces. Fail-closed: a missing/malformed next-review date is
    overdue, not silently healthy."""
    rows, n_overdue = _read_all(only_overdue=True)
    emit(ok({"overdue_count": n_overdue, "overdue": rows}))
    if n_overdue:
        raise typer.Exit(code=1)
