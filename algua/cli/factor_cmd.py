# algua/cli/factor_cmd.py
from __future__ import annotations

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.data.capabilities import supported_capabilities
from algua.features.catalogue import (
    FactorKind,
    FactorSpec,
    filter_factors,
    get_factor,
)
from algua.registry.lineage import dependents_of, factors_used_by
from algua.registry.store import SqliteStrategyRepository

factor_app = typer.Typer(
    help="Factor catalogue: discover and trace composable factors", no_args_is_help=True
)
app.add_typer(factor_app, name="factor")


def _spec_json(spec: FactorSpec, *, full: bool) -> dict[str, object]:
    supported = supported_capabilities()
    data: dict[str, object] = {
        "name": spec.name,
        "summary": spec.summary,
        "kind": spec.kind.value,
        "tags": list(spec.tags),
        "data_needs": [c.value for c in spec.data_needs],
        "import_path": spec.import_path,
        "signature": spec.signature,
        "platform_supported": all(c in supported for c in spec.data_needs),
    }
    if full:
        data["module"] = spec.module
        data["doc"] = spec.doc
    return data


def _coerce_kind(raw: str | None) -> FactorKind | None:
    if raw is None:
        return None
    try:
        return FactorKind(raw)
    except ValueError as exc:
        allowed = ", ".join(k.value for k in FactorKind)
        raise ValueError(f"unknown kind {raw!r}; allowed: {allowed}") from exc


@factor_app.command("list")
@json_errors()
def list_factors(
    tag: str = typer.Option(None, "--tag", help="filter by tag"),
    kind: str = typer.Option(None, "--kind", help="filter by FactorKind"),
) -> None:
    """List catalogued factors as a JSON array (filters AND-combined)."""
    specs = filter_factors(tag=tag, kind=_coerce_kind(kind))
    emit([_spec_json(s, full=False) for s in specs])


@factor_app.command("show")
@json_errors()
def show_factor(name: str = typer.Argument(..., help="factor name")) -> None:
    """Show one factor's full spec as JSON."""
    emit(ok(_spec_json(get_factor(name), full=True)))


@factor_app.command("dependents")
@json_errors()
def factor_dependents(
    name: str = typer.Argument(..., help="factor name"),
    allow_partial: bool = typer.Option(
        False, "--allow-partial", help="exit 0 even if some strategies are unloadable"
    ),
) -> None:
    """Registered strategies whose closure reaches this factor (blast radius)."""
    with registry_conn() as conn:
        result = dependents_of(SqliteStrategyRepository(conn), name)
    emit(ok({"factor": result.factor, "dependents": result.dependents,
             "unloadable": result.unloadable}))
    if result.unloadable and not allow_partial:
        raise typer.Exit(code=1)


@factor_app.command("uses")
@json_errors()
def factor_uses(strategy: str = typer.Argument(..., help="strategy name")) -> None:
    """Catalogued factors a strategy composes."""
    specs = factors_used_by(strategy)
    emit(ok({"strategy": strategy, "factors": [s.name for s in specs]}))
