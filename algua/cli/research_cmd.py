from __future__ import annotations

import typer

from algua.cli._common import ok, project
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.registry.promote_run import promote_task

research_app = typer.Typer(help="Research workflow: gates and promotion", no_args_is_help=True)
app.add_typer(research_app, name="research")

# --summary keep-list (#349): the decision essence of a promote — the pass/fail verdict, the
# per-check breakdown (`checks` carries each gate's name/value/threshold/pass), the breadth and
# the binding flags, the holdout/stability scalars, and provenance. Keep-list (not drop-list) so
# the ~25 deep dsr_*/fdr_* internals, per-regime sharpes, and shadow-audit fields are
# excluded-by-default from the operator-facing summary (context-rot defense).
_PROMOTE_SUMMARY_KEYS = (
    "promoted", "strategy", "passed", "checks", "n_combos", "n_funnel", "breadth_provenance",
    "base_min_holdout_sharpe", "effective_min_holdout_sharpe", "pit_ok", "pit_override",
    "dsr_binding", "dsr_bootstrap_binding", "fdr_binding", "regime_robustness_binding",
    "ir_binding", "returns_available", "holdout", "stability", "config_hash", "snapshot_id",
    "universe_name", "universe_snapshots", "fundamentals_snapshot", "news_snapshot",
    "holdout_reuse",
)


@research_app.command("promote")
# sqlite3.OperationalError keeps lock-contention ("database is locked") from reserve_holdout's
# BEGIN IMMEDIATE inside the JSON envelope, not a leaked traceback (CLI JSON-output contract).
@json_errors
def promote(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    fundamentals_snapshot: str = typer.Option(
        None, "--fundamentals-snapshot",
        help="ingested fundamentals snapshot id (required for a needs_fundamentals strategy)"),
    news_snapshot: str = typer.Option(
        None, "--news-snapshot",
        help="ingested news snapshot id (required for a needs_news strategy)"),
    universe: str = typer.Option(
        None, "--universe",
        help="point-in-time universe name (opt into survivorship-bias-free membership)"),
    windows: int = typer.Option(4, "--windows", help="walk-forward windows"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
    min_holdout_sharpe: float = typer.Option(0.5, "--min-holdout-sharpe"),
    min_holdout_return: float = typer.Option(0.0, "--min-holdout-return"),
    min_pct_positive: float = typer.Option(0.6, "--min-pct-positive"),
    min_window_sharpe: float = typer.Option(0.0, "--min-window-sharpe"),
    n_combos: int = typer.Option(
        None, "--n-combos",
        help="OPERATOR DECLARATION of search breadth, used ONLY when no measured sweep trials "
             "exist; the measured sum from `backtest sweep` is preferred and always wins",
    ),
    allow_holdout_reuse: bool = typer.Option(
        False, "--allow-holdout-reuse",
        help="OVERRIDE the single-use holdout guard: re-evaluate a holdout window already burned "
             "by a prior promote. Records the reuse (reused=1) and marks it in the audit trail. "
             "Statistically costly — only with fresh justification.",
    ),
    allow_non_pit: bool = typer.Option(
        False, "--allow-non-pit",
        help="HUMAN-ONLY override: promote a non-PIT (survivorship-biased) backtest. Audited. "
             "Agents may not pass this.",
    ),
    delistings: str = typer.Option(
        None, "--delistings",
        help="delistings snapshot handle (survivorship-free: realize held delisted names)"),
    assume_terminal_last_close: bool = typer.Option(
        False, "--assume-terminal-last-close",
        help="HUMAN-ONLY: realize a held-into-gap name at its last close when no delisting record "
             "exists. An agent must supply explicit delisting records; no-record-gap fails closed.",
    ),
    actor: str = typer.Option("agent", "--actor", help="human | agent | system"),
    actor_signature: str = typer.Option(
        None, "--actor-signature",
        help="path to the SSH signature over the printed human-actor challenge (#329). Required to "
             "authenticate --actor human: a bare --actor human unlocks NO human-only path — run "
             "once without this to print a challenge, sign it with your enrolled algua-human-actor "
             "key (ssh-keygen -Y sign -n algua-human-actor), then re-run with --actor-signature."),
    new_family: str = typer.Option(
        None, "--new-family",
        help="HUMAN-ONLY: name a new family with FRESH (zero-prior) breadth on a NOVEL/PARENTAGE "
             "verdict. Required for a human NOVEL; ignored once the strategy is assigned. An agent "
             "NOVEL now AUTO-creates a family (auto-named) SEEDED with the funnel-wide breadth "
             "prior (no human signature) — this flag is ignored for agents.",
    ),
    summary: bool = typer.Option(
        False, "--summary",
        help="emit only decision-relevant scalars (drops deep dsr_*/fdr_*/regime diagnostics; "
             "context-rot defense)"),
) -> None:
    """Gate backtested->candidate on the INTEGRITY FLOOR; promote only on pass (factory soft gate).

    BINDING (the floor): PIT universe (`--universe`; non-PIT fails closed unless a human passes
    --allow-non-pit), holdout >= 63 observations (underpowered holdouts fail closed), and raw
    holdout Sharpe > 0. Everything statistical — the breadth-DEFLATED holdout-Sharpe bar, window
    stability, DSR/bootstrap evidence, regime robustness, idiosyncratic alpha — is computed and
    recorded as ADVISORY (`"advisory": true` on the check; never vetoes). The FORWARD gate is the
    harsh threshold (its bar is 0.5x the holdout Sharpe recorded here — overfit self-punishes).
    Breadth is still MEASURED as the sum of recorded `search_trials` (from `backtest sweep`); an
    agent must have measured breadth (no measured trials => refused). Declaring breadth via
    --n-combos is HUMAN-ONLY and recorded with provenance="declared" (auditably less trustworthy).
    On pass for an agent this mints the single-use gate token the BACKTESTED->CANDIDATE
    transition consumes. See docs/superpowers/specs/2026-08-10-strategy-factory-design.md.
    """
    payload = promote_task(
        name, start=start, end=end, demo=demo, snapshot=snapshot,
        fundamentals_snapshot=fundamentals_snapshot, news_snapshot=news_snapshot,
        universe=universe, windows=windows, holdout_frac=holdout_frac,
        min_holdout_sharpe=min_holdout_sharpe, min_holdout_return=min_holdout_return,
        min_pct_positive=min_pct_positive, min_window_sharpe=min_window_sharpe,
        n_combos=n_combos, allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit,
        delistings=delistings, assume_terminal_last_close=assume_terminal_last_close,
        actor=actor, actor_signature=actor_signature, new_family=new_family,
    )
    out = ok(payload)
    emit(project(out, _PROMOTE_SUMMARY_KEYS) if summary else out)
