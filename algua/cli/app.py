from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

import typer

from algua import __version__
from algua.calendar.market_calendar import MarketCalendar
from algua.config.settings import get_settings

app = typer.Typer(
    help="Algua — agent-first algotrading platform", no_args_is_help=True
)


def emit(data: Any) -> None:
    """Print a value as indented JSON — the shared machine + human surface.

    CLI JSON-envelope convention: success payloads that are objects carry ``"ok": true`` (see
    ``cli._common.ok``); failures carry ``{"ok": false, "error": ..., "code": ...}`` where ``code``
    is a stable machine-readable identifier (see ``cli.errors`` and ``cli.main``, and
    ``docs/contracts/cli-error-envelope.md``). Commands that return a collection (``registry list``,
    ``data inspect``) emit a bare JSON array instead — the one documented exception.
    """
    typer.echo(json.dumps(data, indent=2, default=str))


@app.command()
def version() -> None:
    """Print the package version as JSON."""
    emit({"ok": True, "name": "algua", "version": __version__})


def _check(name: str, fn: Callable[[], str]) -> dict[str, Any]:
    """Run one readiness probe: ``fn`` returns a detail string on success or raises on failure.
    Either way it becomes a uniform ``{check, ok, detail}`` row (no per-check try/except)."""
    try:
        return {"check": name, "ok": True, "detail": fn()}
    except Exception as exc:  # noqa: BLE001 - any failure is reported as a check result
        return {"check": name, "ok": False, "detail": str(exc)}


def _registry_db_detail() -> str:
    from algua.cli._common import registry_conn

    with registry_conn():
        pass
    return str(get_settings().db_path)


def _calendar_detail() -> str:
    settings = get_settings()
    MarketCalendar(settings.exchange)
    return settings.exchange


def _knowledge_base_detail() -> str:
    """Knowledge-base drift probe: registry stage is read at the seam and passed in, so the
    knowledge layer stays registry-free. Drift (a missing doc or a stale synced stage) raises,
    so ``_check`` renders it as a failed check with the drift detail."""
    from algua.cli._common import registry_conn
    from algua.knowledge.sync import kb_check
    from algua.registry.store import SqliteStrategyRepository

    with registry_conn() as conn:
        stages = {
            rec.name: rec.stage.value for rec in SqliteStrategyRepository(conn).list_strategies()
        }
    healthy, detail = kb_check(get_settings(), stages)
    if not healthy:
        raise RuntimeError(detail)
    return detail


@app.command()
def doctor() -> None:
    """Check environment readiness. Exits non-zero if any check fails."""
    checks: list[dict[str, Any]] = [
        {"check": "python", "ok": sys.version_info >= (3, 12),
         "detail": sys.version.split()[0]},
        _check("registry_db", _registry_db_detail),
        _check("calendar", _calendar_detail),
        _check("knowledge_base", _knowledge_base_detail),
    ]
    all_ok = all(c["ok"] for c in checks)
    emit({"ok": all_ok, "checks": checks})
    raise typer.Exit(code=0 if all_ok else 1)
