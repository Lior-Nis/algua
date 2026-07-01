# algua/cli/factor_cmd.py
from __future__ import annotations

import hashlib
import inspect
import json as _json_mod
from datetime import UTC, datetime

import typer

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
    load_factor_callable,
)
from algua.registry.lineage import dependents_of, factors_used_by
from algua.registry.store import SqliteStrategyRepository
from algua.research.factor_fdr import correct_factor_ic, trial_ir_variance
from algua.research.gates import FUNNEL_WINDOW_DAYS, effective_funnel_breadth

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


def _factor_hashes(
    spec: FactorSpec,
    params: dict,
    horizon: int,
    construction: str,
    construction_params: dict,
    start_iso: str,
    end_iso: str,
) -> tuple[str, str]:
    """Return (code_hash, hypothesis_hash) for a factor eval invocation.

    code_hash: SHA-256 of the factor function's source (changes when code changes).
    hypothesis_hash: SHA-256 of the canonical identity tuple — so the same
      factor+params+window is a re-run (dedup), different params are a new hypothesis.
    """
    fn = load_factor_callable(spec)
    code_hash = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    identity = _json_mod.dumps(
        {
            "import_path": spec.import_path,
            "code_hash": code_hash,
            "params": sorted(params.items()),
            "horizon": horizon,
            "construction": construction,
            "construction_params": sorted(construction_params.items()),
            "start": start_iso,
            "end": end_iso,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    hypothesis_hash = hashlib.sha256(identity.encode()).hexdigest()
    return code_hash, hypothesis_hash


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
@json_errors
def list_factors(
    tag: str = typer.Option(None, "--tag", help="filter by tag"),
    kind: str = typer.Option(None, "--kind", help="filter by FactorKind"),
) -> None:
    """List catalogued factors as a JSON array (filters AND-combined)."""
    specs = filter_factors(tag=tag, kind=_coerce_kind(kind))
    emit([_spec_json(s, full=False) for s in specs])


@factor_app.command("show")
@json_errors
def show_factor(name: str = typer.Argument(..., help="factor name")) -> None:
    """Show one factor's full spec as JSON."""
    emit(ok(_spec_json(get_factor(name), full=True)))


@factor_app.command("dependents")
@json_errors
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
@json_errors
def factor_uses(strategy: str = typer.Argument(..., help="strategy name")) -> None:
    """Catalogued factors a strategy composes."""
    specs = factors_used_by(strategy)
    emit(ok({"strategy": strategy, "factors": [s.name for s in specs]}))


@factor_app.command("eval")
@json_errors
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
    actor: str = typer.Option("agent", "--actor", help="attribution actor for the eval ledger"),
) -> None:
    """Evaluate ONE standalone factor: PIT backtest + FDR-corrected rank IC/IR.

    Each evaluation is recorded in the factor_evaluations ledger. The IC t-stat is
    multiple-testing corrected: breadth haircut sqrt(2*ln N) + DSR-confidence layer
    (#219 slice E of #140). `significant` in the `fdr` block is the honest verdict.
    """
    spec = get_factor(name)
    provider = select_provider(demo, snapshot)
    start_dt, end_dt = utc(start), utc(end)
    universe_by_date, universe_prov = resolve_universe_inputs(universe, start_dt, end_dt)
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise ValueError("--symbols must list at least one symbol")

    params = _parse_kv(param, "--param")
    construction_params = _parse_kv(construction_param, "--construction-param")

    result = evaluate_factor(
        spec, provider, start_dt, end_dt,
        symbols=syms,
        params=params,
        construction=construction,
        construction_params=construction_params,
        horizon=horizon,
        universe_by_date=universe_by_date,
        universe_name=universe,
        universe_snapshots=universe_prov,
    )

    # --- FDR accounting (#219) ---
    code_hash, hypothesis_hash = _factor_hashes(
        spec, params, horizon, construction, construction_params, start, end,
    )
    ic = result.ic
    created_at = datetime.now(UTC).isoformat()

    # Data provenance for the ledger row.
    data_source = "demo" if demo else ("snapshot" if snapshot else "provider")
    snapshot_id_val: str | None = snapshot  # None unless --snapshot given

    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)

        # Blast radius: how many registered strategies compose this factor?
        dep_result = dependents_of(repo, name)
        n_dependents = len(dep_result.dependents)

        row_id = repo.record_factor_evaluation(
            factor_name=name,
            import_path=spec.import_path,
            code_hash=code_hash,
            hypothesis_hash=hypothesis_hash,
            period_start=start,
            period_end=end,
            horizon=horizon,
            params_json=_json_mod.dumps(params, separators=(",", ":")),
            construction=construction,
            construction_params_json=_json_mod.dumps(construction_params, separators=(",", ":")),
            n_obs=ic.get("n_obs"),
            mean_ic=ic.get("mean_ic"),
            ic_ir=ic.get("ir"),
            t_stat=ic.get("t_stat"),
            ic_skew=ic.get("ic_skew"),
            ic_kurtosis=ic.get("ic_kurtosis"),
            n_dependents=n_dependents,
            data_source=data_source,
            snapshot_id=snapshot_id_val,
            actor=actor,
            created_at=created_at,
        )

        # Breadth (self included — inserted above before the query).
        own, windowed = repo.factor_hypothesis_breadth(name, FUNNEL_WINDOW_DAYS)
        n_hypotheses = effective_funnel_breadth(own, windowed)
        irs = repo.windowed_factor_irs(FUNNEL_WINDOW_DAYS)
        tr_var = trial_ir_variance(irs)

        fdr = correct_factor_ic(
            t_stat=ic.get("t_stat"),
            ir=ic.get("ir"),
            n_obs=ic.get("n_obs") or 0,
            ic_skew=ic.get("ic_skew"),
            ic_kurtosis=ic.get("ic_kurtosis"),
            n_hypotheses=n_hypotheses,
            trial_ir_var=tr_var,
        )

        repo.finalize_factor_evaluation(
            row_id,
            n_hypotheses=n_hypotheses,
            dsr_confidence=fdr.get("dsr_confidence"),
            significant=bool(fdr.get("significant")),
        )

    result_dict = result.to_dict()
    result_dict["ic"]["fdr_corrected"] = True   # overrides the raw False
    result_dict["fdr"] = fdr
    result_dict["n_dependents"] = n_dependents
    emit(ok(result_dict))
