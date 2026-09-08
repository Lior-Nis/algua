# algua/cli/idea_ops_cmd.py
"""Driver-facing idea-pool commands (spec 2026-09-08 §7). Agents never run these against
authority; the research/leap/forage drivers and the merge-back drainer do."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from algua.cli._common import ok
from algua.cli.app import emit
from algua.cli.errors import json_errors
from algua.cli.idea_json import idea_json
from algua.config.settings import get_settings
from algua.contracts.idea import AttemptOutcome
from algua.registry.db import connect, migrate, registry_conn
from algua.registry.idea_attempts import IdeaAttemptsRepository
from algua.registry.idea_import import (
    import_critic_rejections,
    import_ideas,
    reclassify,
    refuted_with_reasons,
)
from algua.registry.idea_scorecard import scorecard as _scorecard
from algua.registry.ideas import IdeaRepository
from algua.registry.store import SqliteStrategyRepository

idea_ops_app = typer.Typer(no_args_is_help=True)


@idea_ops_app.command("claim")
@json_errors
def claim(run: str = typer.Option(..., "--run", help="run stamp that owns the claims"),
          limit: int = typer.Option(..., "--limit", min=1),
          category: str = typer.Option(None, "--category", help="restrict to this category"),
          ) -> None:
    """Reap expired claims, then claim up to --limit open ideas for --run (driver only)."""
    s = get_settings()
    with registry_conn() as conn:
        att = IdeaAttemptsRepository(conn)
        claimed = att.claim(run_stamp=run, limit=limit, ttl_minutes=s.idea_claim_ttl_minutes,
                            category=category)
        repo = IdeaRepository(conn)
        emit(ok({"run": run, "claimed": [idea_json(i, repo) for i in claimed]}))


@idea_ops_app.command("record-outcome")
@json_errors
def record_outcome(
    idea_id: int = typer.Argument(..., metavar="ID"),
    token: str = typer.Option(..., "--token"),
    outcome: AttemptOutcome = typer.Option(..., "--outcome"),
    reason: str = typer.Option(..., "--reason"),
    evidence_ref: str = typer.Option(None, "--evidence-ref"),
    strategy_name: str = typer.Option(None, "--strategy-name"),
) -> None:
    """Write a claimed idea's attempt outcome once (token-fenced); refuting outcomes refute."""
    with registry_conn() as conn:
        idea = IdeaAttemptsRepository(conn).record_outcome(
            idea_id, token=token, outcome=outcome, reason=reason, evidence_ref=evidence_ref,
            strategy_name=strategy_name)
        emit(ok(idea_json(idea, IdeaRepository(conn))))


@idea_ops_app.command("link")
@json_errors
def link(idea_id: int = typer.Argument(..., metavar="ID"),
         strategy: str = typer.Option(..., "--strategy"),
         token: str = typer.Option(..., "--token")) -> None:
    """Merge-back succeeded: link the idea to its registered strategy (drainer only)."""
    with registry_conn() as conn:
        strat = SqliteStrategyRepository(conn).get(strategy)
        idea = IdeaAttemptsRepository(conn).link(
            idea_id, token=token, strategy_id=strat.id, strategy_name=strategy)
        emit(ok(idea_json(idea, IdeaRepository(conn))))


@idea_ops_app.command("depth")
@json_errors
def depth() -> None:
    """Pool depth vs the refill trigger / ceiling (counts derived from settings)."""
    s = get_settings()
    with registry_conn() as conn:
        emit(ok(IdeaAttemptsRepository(conn).depth(
            runs_per_day=s.research_runs_per_day,
            hypotheses_per_run=s.research_hypotheses_per_run,
            floor_days=s.idea_pool_floor_days, ceiling_days=s.idea_pool_ceiling_days)))


@idea_ops_app.command("refuted")
@json_errors
def refuted(limit: int = typer.Option(50, "--limit", min=1)) -> None:
    """Refuted ideas with their latest attempt reason (bare JSON array)."""
    with registry_conn() as conn:
        emit(refuted_with_reasons(conn, limit=limit))


@idea_ops_app.command("import")
@json_errors
def import_(
    from_db: Path = typer.Option(..., "--from", help="scratch registry DB the leap agent wrote"),
    run: str = typer.Option(..., "--run"),
    max_new: int = typer.Option(..., "--max", min=1),
    seeded_max_id: int = typer.Option(
        None, "--seeded-max-id",
        help="max ideas.id at seed time (default: read from authority)"),
    critic_file: Path = typer.Option(None, "--critic-file", help="leap-critic.jsonl"),
) -> None:
    """Move new scratch ideas into authority under a fresh dedup + eligibility check (driver)."""
    s = get_settings()
    ceiling = s.research_runs_per_day * s.research_hypotheses_per_run * s.idea_pool_ceiling_days
    scratch = connect(from_db)
    migrate(scratch)
    try:
        with registry_conn() as auth:
            if seeded_max_id is None:
                seeded_max_id = auth.execute(
                    "SELECT COALESCE(MAX(id),0) FROM ideas").fetchone()[0]
            result = import_ideas(auth, scratch, run_stamp=run, max_new=max_new,
                                  ceiling=ceiling, seeded_max_id=seeded_max_id)
            critic_rows = 0
            if critic_file is not None and critic_file.exists():
                rows = [json.loads(line) for line in critic_file.read_text().splitlines()
                        if line.strip()]
                critic_rows = import_critic_rejections(auth, rows, run_stamp=run,
                                                       max_rows=3 * max_new)
            emit(ok({**result, "critic_rows": critic_rows, "ceiling": ceiling}))
    finally:
        scratch.close()


@idea_ops_app.command("reclassify")
@json_errors
def reclassify_() -> None:
    """Re-open parked ideas whose market/horizon/data became supported."""
    with registry_conn() as conn:
        emit(ok(reclassify(conn)))


@idea_ops_app.command("scorecard")
@json_errors
def scorecard(days: int = typer.Option(90, "--days", min=1)) -> None:
    """Attempt outcomes and downstream stage by venue / category / obscurity / inspiration."""
    with registry_conn() as conn:
        emit(ok(_scorecard(conn, days=days)))
