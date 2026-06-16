# algua/cli/factor_cmd.py
from __future__ import annotations

import typer

from algua.backtest.engine import BacktestError
from algua.backtest.factor_eval import evaluate_factor
from algua.cli._common import ok, registry_conn, resolve_universe_inputs, select_provider, utc
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


def _coerce_scalar(raw: str) -> object:
    """Coerce a CLI value string to int, then float, else leave as str."""
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _parse_kv(items: list[str], flag: str) -> dict[str, object]:
    """Parse repeatable `KEY=value` flags into a single dict (values coerced int->float->str).
    Each key takes ONE value (a factor eval is a single point, not a grid)."""
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"malformed {flag} {item!r}: expected KEY=value")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key or raw.strip() == "":
            raise ValueError(f"malformed {flag} {item!r}: empty key or value")
        if key in out:
            raise ValueError(f"duplicate {flag} key {key!r}")
        out[key] = _coerce_scalar(raw.strip())
    return out


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


@factor_app.command("eval")
@json_errors(ValueError, LookupError, BacktestError)
def eval_factor(
    name: str = typer.Argument(..., help="standalone factor name"),
    symbols: str = typer.Option(..., "--symbols", help="comma-separated evaluation universe"),
    construction: str = typer.Option(..., "--construction", help="construction policy id"),
    construction_param: list[str] = typer.Option(
        [], "--construction-param", help="KEY=value for the construction policy (repeatable)"),
    param: list[str] = typer.Option(
        [], "--param", help="KEY=value factor param, e.g. lookback=60 (repeatable)"),
    horizon: int = typer.Option(1, "--horizon", help="forward-return horizon in bars"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="evaluate an ingested bars snapshot id"),
    universe: str = typer.Option(
        None, "--universe", help="point-in-time universe name (PIT membership for the backtest)"),
) -> None:
    """Evaluate ONE standalone factor on its own: a PIT backtest (via a 1-factor adapter) plus a
    construction-free rank IC/IR block. Ephemeral — writes nothing to the registry; the IC t-stat
    is NOT multiple-testing corrected (#140 slice E)."""
    spec = get_factor(name)
    provider = select_provider(demo, snapshot)
    start_dt, end_dt = utc(start), utc(end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise ValueError("--symbols must list at least one symbol")
    result = evaluate_factor(
        spec, provider, start_dt, end_dt,
        symbols=syms,
        params=_parse_kv(param, "--param"),
        construction=construction,
        construction_params=_parse_kv(construction_param, "--construction-param"),
        horizon=horizon,
        universe_by_date=universe_by_date,
        universe_name=universe,
        universe_snapshots=universe_prov,
    )
    emit(ok(result.to_dict()))
