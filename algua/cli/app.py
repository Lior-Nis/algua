from __future__ import annotations

import ast
import json
import sys
from collections.abc import Callable
from pathlib import Path
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


def _check(name: str, fn: Callable[[], str], *, required: bool = True) -> dict[str, Any]:
    """Run one readiness probe: ``fn`` returns a detail string on success or raises on failure.
    Either way it becomes a uniform ``{check, ok, required, detail}`` row (no per-check try/except).

    ``required`` rows gate ``doctor``'s exit code; advisory rows (``required=False``) are reported
    but never flip the exit code, so a green top-level ``ok`` can coexist with a failing advisory
    row (e.g. missing paper creds) — an agent reads the row to know the paper loop can't run yet."""
    try:
        detail, ok = fn(), True
    except Exception as exc:  # noqa: BLE001 - any failure is reported as a check result
        detail, ok = str(exc), False
    return {"check": name, "ok": ok, "required": required, "detail": detail}


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


def _paper_credentials_detail() -> str:
    """Advisory: are Alpaca paper-trading credentials configured? Without them the paper loop can't
    reach the broker, so pre-flight should surface it even though research/backtest run fine."""
    settings = get_settings()
    if not settings.alpaca_api_key or not settings.alpaca_api_secret:
        raise RuntimeError("alpaca_api_key/alpaca_api_secret not set (paper loop can't run)")
    return settings.alpaca_paper_url


def _bars_snapshot_detail() -> str:
    """Advisory: does at least one BARS snapshot exist? Cheap manifest count only — deep payload
    verification belongs to ``data verify``. A green pre-flight with zero bars means backtests and
    the paper loop have nothing to run on."""
    from algua.data.models import Dataset
    from algua.data.store import DataStore

    records = DataStore(get_settings().data_dir).manifest.list_records(Dataset.BARS)
    if not records:
        raise RuntimeError("no BARS snapshots ingested (run `algua data ingest-bars`)")
    return f"{len(records)} bars snapshot(s)"


def _global_halt_detail() -> str:
    """Advisory safety-state probe: is the account-wide halt engaged? A green pre-flight while a
    global halt is live would hide that the trading loops are frozen — so an engaged halt raises."""
    from algua.cli._common import registry_conn
    from algua.risk import global_halt

    with registry_conn() as conn:
        info = global_halt.get(conn)
    if info is not None:
        raise RuntimeError(
            f"global halt ENGAGED: {info['reason']} "
            f"(by {info['actor']} at {info['created_at']})"
        )
    return "no global halt engaged"


def _kill_switches_detail() -> str:
    """Advisory safety-state probe: are any per-strategy kill switches tripped? A tripped switch
    blocks that strategy from trading; surface it rather than reporting all-green."""
    from algua.cli._common import registry_conn
    from algua.risk import kill_switch

    with registry_conn() as conn:
        tripped = kill_switch.list_tripped(conn)
    if tripped:
        raise RuntimeError(f"{len(tripped)} kill switch(es) tripped: {', '.join(tripped)}")
    return "no kill switches tripped"


def _live_authorizations_detail() -> str:
    """Advisory safety-state probe: is every LIVE strategy still human-authorized at the current
    trust anchor? Re-verifies each live strategy's signature; a revoked/unverifiable authorization
    (or one whose artifact hash can't be recomputed) is reported per-strategy so one failure doesn't
    mask others. Any failure raises with the aggregated per-strategy detail."""
    from algua.cli._common import registry_conn
    from algua.contracts.lifecycle import Stage
    from algua.registry.live_gate import ALLOWED_SIGNERS_PATH, verify_live_authorization
    from algua.registry.store import SqliteStrategyRepository

    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        live = repo.list_strategies(Stage.LIVE)
        if not live:
            return "no live strategies"
        failures: list[str] = []
        for rec in live:
            try:
                verify_live_authorization(conn, repo, rec.name, ALLOWED_SIGNERS_PATH)
            except Exception as exc:  # noqa: BLE001 - per-strategy so one failure doesn't mask others
                failures.append(f"{rec.name}: {exc}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return f"{len(live)} live strategy(ies) authorized"


def _generated_provenance_detail() -> str:
    """Advisory: how many bundled strategy modules carry the ``GENERATED_BY`` provenance marker the
    author-a-strategy contract mandates? Static AST scan of the strategy family dirs — imports
    nothing (the dir is derived from this file's path, no package code runs). Reports coverage;
    never fails the pre-flight."""
    root = Path(__file__).parents[1] / "strategies"
    modules = [
        py
        for fam in sorted(root.iterdir())
        if fam.is_dir() and not fam.name.startswith("_") and (fam / "__init__.py").exists()
        for py in sorted(fam.glob("*.py"))
        if py.name != "__init__.py" and not py.name.startswith("_")
    ]
    marked = sum(1 for py in modules if _has_generated_by(py))
    total = len(modules)
    if marked < total:
        raise RuntimeError(f"{marked}/{total} strategy modules carry GENERATED_BY")
    return f"{marked}/{total} strategy modules carry GENERATED_BY"


def _has_generated_by(path: Path) -> bool:
    """True iff the module assigns a module-level ``GENERATED_BY`` name, plain or annotated
    (``GENERATED_BY = ...`` or ``GENERATED_BY: str = ...``). AST-only — no import."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "GENERATED_BY" for t in node.targets
        ):
            return True
        if (
            isinstance(node, ast.AnnAssign)
            and node.value is not None  # a bare `GENERATED_BY: str` annotation assigns no marker
            and isinstance(node.target, ast.Name)
            and node.target.id == "GENERATED_BY"
        ):
            return True
    return False


@app.command()
def doctor() -> None:
    """Check environment readiness. Exits non-zero if any REQUIRED check fails; advisory rows
    (``required=false`` — trading-segment / provenance readiness) are reported but never gate exit.
    """
    checks: list[dict[str, Any]] = [
        {"check": "python", "ok": sys.version_info >= (3, 12),
         "required": True, "detail": sys.version.split()[0]},
        _check("registry_db", _registry_db_detail),
        _check("calendar", _calendar_detail),
        _check("knowledge_base", _knowledge_base_detail),
        _check("paper_credentials", _paper_credentials_detail, required=False),
        _check("bars_snapshot", _bars_snapshot_detail, required=False),
        _check("generated_provenance", _generated_provenance_detail, required=False),
        _check("global_halt", _global_halt_detail, required=False),
        _check("kill_switches", _kill_switches_detail, required=False),
        _check("live_authorizations", _live_authorizations_detail, required=False),
    ]
    all_ok = all(c["ok"] for c in checks if c["required"])
    emit({"ok": all_ok, "checks": checks})
    raise typer.Exit(code=0 if all_ok else 1)
