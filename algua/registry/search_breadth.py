from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algua.backtest.sweep import SweepResult
    from algua.registry.store import SqliteStrategyRepository


def record_search_breadth(
    repo: SqliteStrategyRepository, name: str, result: SweepResult,
) -> dict[str, int]:
    """Record this sweep's measured breadth into the registry, keyed by strategy NAME.

    Recorded UNCONDITIONALLY — even for a not-yet-registered strategy. Exploration precedes
    registration: keying by name (not the registry id) means a pre-registration sweep still
    counts toward the promotion breadth, so an agent can't sweep broadly first and then promote a
    freshly-registered strategy under a smaller declared --n-combos. Returns the recorded count
    plus the new cumulative family total for the emitted JSON.

    The transaction is CALLER-OWNED: the CLI wraps this in ``with registry_conn() as conn:`` and
    passes ``SqliteStrategyRepository(conn)``.
    """
    repo.record_search_trial(
        name, result.n_combos, json.dumps(result.grid, sort_keys=True),
        trial_sharpe_count=result.trial_sharpe_count,
        trial_sharpe_mean=result.trial_sharpe_mean,
        trial_sharpe_var_ann=result.trial_sharpe_var_ann,
    )
    return {"n_combos": result.n_combos, "cumulative": repo.total_search_combos(name)}
