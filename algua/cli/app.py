from __future__ import annotations

import json
import sys
from typing import Any

import typer

from algua import __version__
from algua.calendar.market_calendar import MarketCalendar
from algua.config.settings import get_settings
from algua.registry.db import connect, migrate

app = typer.Typer(
    help="Algua — agent-first algotrading platform", no_args_is_help=True
)


def emit(data: Any) -> None:
    """Print a value as indented JSON — the shared machine + human surface."""
    typer.echo(json.dumps(data, indent=2, default=str))


@app.command()
def version() -> None:
    """Print the package version as JSON."""
    emit({"name": "algua", "version": __version__})


@app.command()
def doctor() -> None:
    """Check environment readiness. Exits non-zero if any check fails."""
    settings = get_settings()
    checks: list[dict[str, Any]] = [
        {"check": "python", "ok": sys.version_info >= (3, 12),
         "detail": sys.version.split()[0]},
    ]
    try:
        conn = connect(settings.db_path)
        migrate(conn)
        conn.close()
        checks.append({"check": "registry_db", "ok": True, "detail": str(settings.db_path)})
    except Exception as exc:  # noqa: BLE001 - report any failure as a check result
        checks.append({"check": "registry_db", "ok": False, "detail": str(exc)})
    try:
        MarketCalendar(settings.exchange)
        checks.append({"check": "calendar", "ok": True, "detail": settings.exchange})
    except Exception as exc:  # noqa: BLE001
        checks.append({"check": "calendar", "ok": False, "detail": str(exc)})

    try:
        from algua.knowledge.sync import kb_check

        kb_ok, kb_detail = kb_check(settings)
        checks.append({"check": "knowledge_base", "ok": kb_ok, "detail": kb_detail})
    except Exception as exc:  # noqa: BLE001
        checks.append({"check": "knowledge_base", "ok": False, "detail": str(exc)})

    all_ok = all(c["ok"] for c in checks)
    emit({"ok": all_ok, "checks": checks})
    raise typer.Exit(code=0 if all_ok else 1)
