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
    return {"check": name, "ok": ok, "required": required, "detail": detail, "skipped": False}


def _skip_row(name: str, detail: str) -> dict[str, Any]:
    """A lane-gated probe whose lane flag is absent: emitted at the *call site* so the probe body —
    and any side effect it carries (a strategy-module import, a network call) — never runs. The
    ``skipped: true`` boolean is the machine-legible discriminator, so a consumer branching on it
    can never mistake an un-probed dependency for a passing one (never string-match ``detail``)."""
    return {"check": name, "ok": True, "required": False, "detail": detail, "skipped": True}


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
    """Required safety-state probe (gates in EVERY mode): is the account-wide halt engaged? A green
    pre-flight while a global halt is live would hide that no trading tick can start — every tick
    raises on its next call — so an engaged halt raises here and flips ``doctor`` red."""
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
    """Live-lane authorization probe — runs ONLY under ``doctor --live`` (its body, not merely its
    ``required`` flag, is gated at the call site). Re-verifies every ``Stage.LIVE`` strategy's
    go-live signature against the current trust anchor. STRICT rule: ``ok = (no LIVE strategies) OR
    (EVERY LIVE strategy re-verifies)`` — any revoked/unverifiable strategy raises with the
    aggregated per-strategy detail so one failure never masks another. The detail always surfaces
    ``live_strategies=N`` (an empty book reads as ``live_strategies=0``, ready — not a bare green).

    This body is gated behind ``--live`` because the verification path
    ``verify_live_authorization -> compute_artifact_hashes -> load_strategy`` **imports strategy
    modules** (arbitrary import-time code), a side effect the default ``doctor`` must never trigger.
    ``doctor`` is deliberately stricter than ``live run-all`` (which skips a revoked strategy and
    trades the rest): the pre-flight's job is to make the operator notice the bad one before a
    cycle starts. Authorization ONLY — not allocation/kill-switch/candidate presence (#400)."""
    from algua.cli._common import registry_conn
    from algua.contracts.lifecycle import Stage
    from algua.registry.live_gate import ALLOWED_SIGNERS_PATH, verify_live_authorization
    from algua.registry.store import SqliteStrategyRepository

    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        live = repo.list_strategies(Stage.LIVE)
        n = len(live)
        failures: list[str] = []
        for rec in live:
            try:
                verify_live_authorization(conn, repo, rec.name, ALLOWED_SIGNERS_PATH)
            except Exception as exc:  # noqa: BLE001 - per-strategy so one failure doesn't mask others
                failures.append(f"{rec.name}: {exc}")
    if failures:
        raise RuntimeError(
            f"live_strategies={n}; {len(failures)} unauthorized/unverifiable: "
            + "; ".join(failures)
        )
    if n == 0:
        return "live_strategies=0 (no live strategies to authorize)"
    return f"live_strategies={n}; all authorized"


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
def doctor(
    paper: bool = typer.Option(
        False,
        "--paper",
        help="Gate exit on the paper lane's dependencies: promotes paper_credentials to required.",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help=(
            "Gate exit on the live lane's AUTHORIZATION ONLY: run and require the "
            "live_authorizations probe so every existing LIVE strategy must re-verify its go-live "
            "signature. This is the only mode that imports strategy modules. Does NOT assert any "
            "strategy will actually trade — allocation / kill-switch / candidate presence are "
            "fleet-status concerns (see `fleet health` / #400)."
        ),
    ),
) -> None:
    """Check trading-readiness: is it safe and possible to start a trading cycle right now?

    Exits non-zero if any REQUIRED check fails; advisory rows (``required=false``) are reported but
    never gate exit. ``global_halt`` gates in EVERY mode — an engaged account-wide halt means no
    trading tick can start. ``--paper`` promotes the paper-credential probe to required; ``--live``
    runs and requires ``live_authorizations`` (whose body — which imports strategy modules — is
    skipped without ``--live``). Every row carries a ``skipped`` boolean so a consumer never treats
    an un-probed dependency as a passing one. (Research/backtest commands ignore this exit code.)
    """
    checks: list[dict[str, Any]] = [
        {"check": "python", "ok": sys.version_info >= (3, 12),
         "required": True, "detail": sys.version.split()[0], "skipped": False},
        _check("registry_db", _registry_db_detail),
        _check("calendar", _calendar_detail),
        _check("knowledge_base", _knowledge_base_detail),
        _check("paper_credentials", _paper_credentials_detail, required=paper),
        _check("bars_snapshot", _bars_snapshot_detail, required=False),
        _check("generated_provenance", _generated_provenance_detail, required=False),
        _check("global_halt", _global_halt_detail, required=True),
        _check("kill_switches", _kill_switches_detail, required=False),
        _check("live_authorizations", _live_authorizations_detail, required=True)
        if live
        else _skip_row(
            "live_authorizations", "skipped: pass --live to probe live authorizations"
        ),
    ]
    all_ok = all(c["ok"] for c in checks if c["required"])
    emit({"ok": all_ok, "checks": checks})
    raise typer.Exit(code=0 if all_ok else 1)
