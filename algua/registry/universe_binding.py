"""Bind a paper deployment to the universe its gate evidence was produced on (#559).

A strategy module's ``CONFIG.universe`` is agent-authored and can silently diverge from the
universe the promote gate actually validated (``research promote --universe``): the gate row
records the gated-universe NAME (``gate_evaluations.universe_name``, v39), and the operational
tick must trade THAT universe, never the module's template. ``resolve_operational_universe`` is
the single chokepoint the paper tick uses to resolve the symbols a strategy is allowed to see:

- newest PASSING gate row carries a ``universe_name`` -> resolve the CURRENT membership from the
  universe-snapshot timeline (as-of today, greatest ``effective_date`` <= today — the same as-of
  rule the backtest engine's ``members_as_of`` applies) -> ``source="gate"``;
- newest PASSING gate row predates v39 (``universe_name`` NULL) -> fall back to the module's
  ``CONFIG.universe`` with ``source="config_legacy"`` (the caller MUST log a loud warning);
- no passing gate row at all -> ``LookupError`` (fail closed: an unpromoted strategy has no
  business ticking).
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from algua.data.store import DataStore

#: ``source`` values returned by :func:`resolve_operational_universe`.
SOURCE_GATE = "gate"
SOURCE_CONFIG_LEGACY = "config_legacy"


def resolve_operational_universe(
    conn: sqlite3.Connection,
    data_dir: Path,
    strategy_name: str,
    config_universe: list[str],
) -> tuple[list[str], str]:
    """Resolve the symbols ``strategy_name`` may operationally trade, bound to its gate evidence.

    Returns ``(symbols, source)`` with ``source`` in {``"gate"``, ``"config_legacy"``}. Raises
    ``LookupError`` when the strategy has no passing gate row (fail closed) or when the gated
    universe has no membership effective on or before today (an empty operational universe is a
    data error, not a tradable state).
    """
    row = conn.execute(
        "SELECT g.universe_name FROM gate_evaluations g"
        " JOIN strategies s ON s.id = g.strategy_id"
        " WHERE s.name = ? AND g.passed = 1"
        " ORDER BY g.id DESC LIMIT 1",
        (strategy_name,),
    ).fetchone()
    if row is None:
        raise LookupError(
            f"strategy {strategy_name!r} has no passing gate_evaluations row; an unpromoted "
            "strategy has no business ticking (#559)"
        )
    universe_name = row["universe_name"]
    if universe_name is None:
        # Legacy pre-v39 gate row: the gated-universe identity was never recorded, so the module
        # CONFIG is the only universe we have. The caller logs a loud warning; a fresh passing
        # `research promote` run stamps universe_name and upgrades this strategy to gate-bound.
        return list(config_universe), SOURCE_CONFIG_LEGACY
    timeline = DataStore(data_dir).read_universe(universe_name)
    today = datetime.now(UTC).date()
    eligible = [snap for snap in timeline if snap.effective_date <= today]
    if not eligible:
        raise LookupError(
            f"gated universe {universe_name!r} for strategy {strategy_name!r} has no membership "
            f"snapshot effective on or before {today.isoformat()}; cannot resolve an operational "
            "universe (fail closed)"
        )
    # read_universe returns the timeline sorted ascending by effective_date; the as-of-today
    # membership is the last eligible snapshot (greatest effective_date <= today).
    return sorted(eligible[-1].symbols), SOURCE_GATE
