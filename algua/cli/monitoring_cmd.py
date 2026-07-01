# algua/cli/monitoring_cmd.py
from __future__ import annotations

import typer

from algua.backtest.factor_eval import forward_returns as _forward_returns
from algua.backtest.factor_eval import score_panel as _score_panel
from algua.cli._common import ok, resolve_eval_inputs, resolve_universe_inputs, utc
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.monitoring.drift import drift_report

monitoring_app = typer.Typer(
    help="Advisory leading-indicator drift monitoring (gates nothing)", no_args_is_help=True
)
app.add_typer(monitoring_app, name="monitoring")


@monitoring_app.command("drift")
@json_errors
def drift(
    name: str = typer.Argument(..., help="registered strategy name"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="use an ingested bars snapshot id"),
    universe: str = typer.Option(
        None, "--universe", help="point-in-time universe name (as-of member masking)"),
    symbols: str = typer.Option(
        None, "--symbols", help="comma-separated override of the strategy's declared universe"),
    reference_end: str = typer.Option(
        None, "--reference-end",
        help="pin the reference/recent split date (bars <= this are the frozen reference)"),
    reference_frac: float = typer.Option(
        0.5, "--reference-frac", help="positional split when --reference-end is not pinned"),
    horizon: int = typer.Option(1, "--horizon", help="forward-return horizon in bars (tier B)"),
    psi_bins: int = typer.Option(10, "--psi-bins", help="reference quantile bins for PSI"),
    min_obs: int = typer.Option(
        20, "--min-obs", help="minimum usable IC bars per window before tier B is assessed"),
) -> None:
    """ADVISORY: compare a strategy's signal over a frozen REFERENCE era vs a RECENT era.

    Emits a leading-indicator drift verdict (TIER A: label-free signal-distribution / turnover /
    coverage drift) plus a coincident corroborating read (TIER B: rank-IC / hit-rate decay). This
    gates NOTHING, persists NOTHING, and never touches the registry, promotion/forward gates, or
    the live/paper order path — sustained drift is a re-audition prompt, not an enforcement.

    Exit code is 0 even when the verdict is `alarm` (drift is a finding, not a CLI error).
    """
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(name, demo, snapshot, start, end)
    if strategy.config.needs_fundamentals or strategy.config.needs_news:
        raise ValueError(
            "drift monitoring supports OHLCV signal strategies only; the sidecar-fed "
            "needs_fundamentals/needs_news lane is a follow-up"
        )
    universe_by_date, _prov = resolve_universe_inputs(universe, start_dt, end_dt)

    if symbols is not None:
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        if not syms:
            raise ValueError("--symbols, if given, must list at least one symbol")
    else:
        syms = sorted(set(strategy.config.universe))
    if not syms:
        raise ValueError("strategy declares an empty universe; pass --symbols")

    bars = provider.get_bars(syms, start_dt, end_dt, "1d")
    panel = _score_panel(strategy, bars, universe_by_date=universe_by_date)
    adj = (
        bars.reset_index()
        .pivot(index="timestamp", columns="symbol", values="adj_close")
        .sort_index()
    )
    lag = strategy.execution.decision_lag_bars
    fwd = _forward_returns(adj, lag=lag, horizon=horizon)

    split_ts = utc(reference_end) if reference_end else None
    report = drift_report(
        panel, fwd, split=split_ts, reference_frac=reference_frac,
        psi_bins=psi_bins, min_obs=min_obs, forward_embargo=lag + horizon,
    )
    payload = report.to_dict()
    payload["strategy"] = name
    payload["horizon"] = horizon
    emit(ok(payload))
