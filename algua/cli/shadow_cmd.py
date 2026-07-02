from __future__ import annotations

import typer

from algua.audit.log import append as audit_append
from algua.cli._common import ok, registry_conn, utc
from algua.cli._common import select_provider as _select_provider
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.store import SqliteStrategyRepository
from algua.shadow.evaluate import SHADOW_TIMEFRAME, ShadowResult, shadow_replay
from algua.shadow.store import latest_shadow_evaluation, record_shadow_evaluation
from algua.strategies.loader import load_strategy

shadow_app = typer.Typer(
    help="Shadow-mode / champion-challenger evaluation — ADVISORY only (never trades, never "
         "promotes)",
    no_args_is_help=True,
)
app.add_typer(shadow_app, name="shadow")


def _evaluate(
    conn, *, challenger: str, champion: str | None, snapshot: str,
    start: str, end: str, cash: float,
) -> tuple[ShadowResult, dict]:
    """Load, replay, and record ONE challenger in shadow. Returns (result, recorded-row-summary).

    A PLAIN load (not the gated loader): shadow is not a gated action, so any registered strategy at
    ANY stage — including a fresh backtested ML candidate — may be shadow-evaluated. The registry
    `get` rejects an unknown name (LookupError) before any replay. Uses ONLY the SimBroker via
    shadow_replay: no real order, no allocation, no holdout, no token, no transition."""
    SqliteStrategyRepository(conn).get(challenger)  # unknown name -> LookupError -> {ok: false}
    strategy = load_strategy(challenger)
    provider = _select_provider(False, snapshot)
    result = shadow_replay(
        strategy, provider, utc(start), utc(end), timeframe=SHADOW_TIMEFRAME, cash=cash,
    )
    identity = compute_artifact_hashes(challenger)
    record_shadow_evaluation(
        conn, challenger=challenger, champion=champion, snapshot_id=snapshot,
        timeframe=SHADOW_TIMEFRAME, start=start, end=end, cash=cash,
        universe=list(strategy.universe), result=result,
        code_hash=identity.code_hash, config_hash=identity.config_hash,
    )
    audit_append(conn, actor="agent", action="shadow_eval",
                 reason=f"champion={champion or '-'} sharpe={result.sharpe:.4f}",
                 strategy=challenger)
    return result, _summary(result)


def _summary(r: ShadowResult) -> dict:
    return {
        "strategy": r.strategy, "final_equity": r.final_equity, "total_return": r.total_return,
        "ann_return": r.ann_return, "ann_volatility": r.ann_volatility, "sharpe": r.sharpe,
        "max_drawdown": r.max_drawdown, "n_bars": r.n_bars, "final_positions": r.final_positions,
    }


@shadow_app.command("run")
@json_errors
def run(
    challenger: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    champion: str = typer.Option(None, "--champion",
                                 help="advisory label for the live incumbent being shadowed"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    cash: float = typer.Option(100_000.0, "--cash", help="starting SIM notional (not a live "
                                                         "allocation)"),
) -> None:
    """Replay a challenger in SHADOW on a bars snapshot and record its hypothetical, order-free
    performance. Never submits an order, never allocates capital, never promotes."""
    if cash <= 0:
        raise ValueError("--cash must be > 0")
    with registry_conn() as conn:
        _result, summary = _evaluate(
            conn, challenger=challenger, champion=champion, snapshot=snapshot,
            start=start, end=end, cash=cash,
        )
    emit(ok({"challenger": challenger, "champion": champion, "snapshot": snapshot,
             "start": start, "end": end, **summary}))


@shadow_app.command("compare")
@json_errors
def compare(
    champion: str = typer.Option(..., "--champion", help="the live incumbent strategy"),
    challenger: str = typer.Option(..., "--challenger", help="the candidate strategy"),
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    cash: float = typer.Option(100_000.0, "--cash", help="starting SIM notional for BOTH sides"),
) -> None:
    """Head-to-head champion-vs-challenger in SHADOW. Replays BOTH strategies over the IDENTICAL
    {snapshot, timeframe=1d, start, end, cash} so the comparison is fair, records both rows, and
    emits the metric deltas plus a PURELY ADVISORY `challenger_leads` flag. Read-only re: lifecycle:
    no transition, no gate, no token — a decision aid, NOT a promotion gate."""
    if cash <= 0:
        raise ValueError("--cash must be > 0")
    if champion == challenger:
        raise ValueError("--champion and --challenger must be different strategies")
    with registry_conn() as conn:
        champ_result, champ_summary = _evaluate(
            conn, challenger=champion, champion=None, snapshot=snapshot,
            start=start, end=end, cash=cash,
        )
        chal_result, chal_summary = _evaluate(
            conn, challenger=challenger, champion=champion, snapshot=snapshot,
            start=start, end=end, cash=cash,
        )
    sharpe_delta = chal_result.sharpe - champ_result.sharpe
    emit(ok({
        "champion": champ_summary,
        "challenger": chal_summary,
        "surface": {"snapshot": snapshot, "timeframe": SHADOW_TIMEFRAME,
                    "start": start, "end": end, "cash": cash},
        "delta": {
            "sharpe": sharpe_delta,
            "ann_return": chal_result.ann_return - champ_result.ann_return,
            "max_drawdown": chal_result.max_drawdown - champ_result.max_drawdown,
            "total_return": chal_result.total_return - champ_result.total_return,
        },
        # ADVISORY only: a strictly-higher shadow Sharpe. NOT a gate, NOT a promotion signal.
        "challenger_leads": sharpe_delta > 0.0,
        "note": "advisory only — shadow evaluation never promotes or trades",
    }))


@shadow_app.command("show")
@json_errors
def show(name: str) -> None:
    """Show the most recent shadow evaluation where `name` was the evaluated (challenger) side.
    Read-only. Rejects an unknown strategy name."""
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)  # unknown -> LookupError -> {ok: false}
        latest = latest_shadow_evaluation(conn, name)
    emit(ok({"strategy": name, "stage": rec.stage.value, "latest_shadow_evaluation": latest}))
