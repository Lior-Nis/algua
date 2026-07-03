# algua/cli/monitoring_cmd.py
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd
import typer

from algua.backtest.factor_eval import forward_returns as _forward_returns
from algua.backtest.factor_eval import score_panel as _score_panel
from algua.calendar.market_calendar import MarketCalendar
from algua.cli._common import ok, registry_conn, resolve_eval_inputs, resolve_universe_inputs, utc
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.monitoring.decay import CertifiedBaseline, decay_report
from algua.monitoring.drift import drift_report
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.forward_promotion import _inadmissible_reason, _parse_dt
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.research.forward_gates import CERTIFICATE_FRESH_SESSIONS, MIN_FORWARD_OBSERVATIONS

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
    if horizon < 1:
        raise ValueError("--horizon must be >= 1 (a forward-return label needs a future bar)")
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


def _live_return_series(
    conn: sqlite3.Connection,
    strategy_id: int,
    identity: ArtifactIdentity,
    calendar: MarketCalendar,
    now_utc: datetime,
    *,
    after_dt: datetime,
) -> tuple[pd.Series, float, int]:
    """Build the daily realized-return series from admissible ``lane='live'`` ticks.

    Reuses the forward gate's own admissibility filter (``_inadmissible_reason``: broker clock,
    identity match, non-null account, well-formed non-future ``tick_ts``, fresh ``decision_ts``)
    so a malformed/future/identity-drifted tick can never leak into the realized read. Ticks at or
    before ``after_ts`` (the certificate instant) are additionally dropped so a certificate
    refreshed mid-live never mixes pre-refresh returns into the comparison.

    Returns ``(daily_returns, session_coverage, n_inadmissible)``. Coverage is decided sessions
    over the trading sessions from the FIRST session strictly after the certificate (``after_dt``)
    through the last observed session — so a long post-certification gap with no live ticks fails
    the coverage floor instead of masking as ``1.0``. 0.0 when there are no observations.

    ``after_dt`` MUST be an aware datetime (the parsed certificate instant); the caller fails closed
    to ``unknown`` when the certificate timestamp does not parse, so this helper never runs with an
    unbounded window against a real baseline.
    """
    rows = conn.execute(
        "SELECT id, tick_ts, decision_ts, equity, clock_source, code_hash, config_hash,"
        " dependency_hash, account_id FROM tick_snapshots"
        " WHERE lane='live' AND strategy_id=? ORDER BY id",
        (strategy_id,),
    ).fetchall()

    admissible: list[sqlite3.Row] = []
    n_inadmissible = 0
    for row in rows:
        if _inadmissible_reason(row, identity, calendar, now_utc) is not None:
            n_inadmissible += 1
            continue
        tick_dt = _parse_dt(row["tick_ts"])
        # Window start: strictly AFTER the certification instant (admissibility already proved
        # tick_ts parses). A tick at or before the certificate is pre-certification evidence.
        if tick_dt is not None and tick_dt <= after_dt:
            continue
        admissible.append(row)

    # Last admissible tick per decision session wins (id order -> later assignment is max id).
    by_session = {}
    for row in admissible:
        decision_dt = _parse_dt(row["decision_ts"])
        assert decision_dt is not None  # admissibility proved decision_ts parses
        by_session[calendar.session_on_or_before(decision_dt.date())] = row
    sessions = sorted(by_session)
    equities = [float(by_session[s]["equity"]) for s in sessions]
    returns = pd.Series(equities, dtype=float).pct_change().dropna()
    # Coverage numerator is RETURN observations (one fewer than equity sessions); the denominator
    # is the trading sessions from the FIRST session strictly after the certificate through the
    # last observation, minus one (the boundary session has no prior equity to return against).
    # Anchoring at the certificate — not at sessions[0] — means a long post-cert silence before a
    # dense recent burst reads as sparse coverage (fails the floor), never a false 1.0.
    n_returns = int(len(returns))
    if n_returns > 0:
        cert_session = calendar.session_on_or_before(after_dt.date())
        expected_sessions = len(calendar.sessions_in_range(cert_session, sessions[-1])) - 1
        coverage = n_returns / expected_sessions if expected_sessions > 0 else 0.0
    else:
        coverage = 0.0
    return returns, coverage, n_inadmissible


@monitoring_app.command("decay")
@json_errors
def decay(
    name: str = typer.Argument(..., help="registered strategy name"),
    min_observations: int = typer.Option(
        MIN_FORWARD_OBSERVATIONS, "--min-observations",
        help="minimum admissible live daily-return observations before a verdict is rendered"),
    recert_stale_sessions: int = typer.Option(
        CERTIFICATE_FRESH_SESSIONS, "--recert-stale-sessions",
        help="certificate age (sessions) beyond which recert_needed is flagged"),
) -> None:
    """ADVISORY: compare a LIVE strategy's realized return distribution against its certified
    forward-test baseline, and flag decay / recertification-needed.

    Reads the newest forward-test certificate for the current identity (a non-passing newest row
    invalidates any prior pass -> verdict `unknown`), builds a realized daily-return series from
    the strategy's admissible ``lane='live'`` ticks recorded SINCE the certificate, and compares
    the realized Sharpe against the same degraded bar the promotion gate uses
    (max(0.5*holdout_sharpe, floor)). Fails closed to `unknown`/`insufficient_data` on any
    missing/degenerate input — it never emits a false `ok`.

    This gates NOTHING, persists NOTHING, and never touches the registry, promotion/forward gates,
    or the live/paper order path. `decay_warn` is a re-audition prompt, not an enforcement. Exit
    code is 0 even when the verdict is `decay_warn` (decay is a finding, not a CLI error).
    """
    if min_observations < 1:
        raise ValueError("--min-observations must be >= 1")
    if recert_stale_sessions < 0:
        raise ValueError("--recert-stale-sessions must be >= 0")

    identity = compute_artifact_hashes(name)
    calendar = MarketCalendar()
    now_utc = datetime.now(UTC)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(name)  # unknown name -> LookupError -> {ok:false}
        row = repo.latest_forward_gate_row(
            rec.id, identity.code_hash, identity.config_hash, identity.dependency_hash)
        baseline: CertifiedBaseline | None = None
        cert_dt: datetime | None = None
        # A passing certificate whose `created_at` does not parse to an aware instant is UNUSABLE:
        # without a parseable boundary the live window is unbounded, so pre-certification ticks
        # could produce a false `ok`. Fail closed to `unknown` (baseline stays None) in that case.
        if row is not None and row["passed"]:
            cert_dt = _parse_dt(row["created_at"])
            if cert_dt is not None:
                baseline = CertifiedBaseline(
                    holdout_sharpe=row["holdout_sharpe"],
                    certified_realized_sharpe=row["realized_sharpe"],
                    created_at=row["created_at"],
                    age_sessions=calendar.sessions_between(cert_dt.date(), now_utc.date()),
                )
        if baseline is not None and cert_dt is not None:
            returns, coverage, n_inadmissible = _live_return_series(
                conn, rec.id, identity, calendar, now_utc, after_dt=cert_dt)
        else:
            # No usable baseline -> the verdict is `unknown` regardless of live ticks; skip the
            # realized read entirely.
            returns, coverage, n_inadmissible = pd.Series(dtype=float), 0.0, 0

    report = decay_report(
        returns, coverage, n_inadmissible, baseline,
        min_observations=min_observations, recert_stale_sessions=recert_stale_sessions,
    )
    payload = report.to_dict()
    payload["strategy"] = name
    payload["stage"] = rec.stage.value
    emit(ok(payload))
