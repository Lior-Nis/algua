from __future__ import annotations

import sqlite3

from algua.contracts.lifecycle import Stage
from algua.registry.repository import StrategyRecord
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch
from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import load_tradable_strategy


def load_gated_strategy(
    conn: sqlite3.Connection, name: str, command: str,
) -> tuple[LoadedStrategy, StrategyRecord]:
    """Load a strategy and clear the two gates every paper trading command shares: it must be at
    the PAPER or FORWARD_TESTED stage and its kill-switch (and the global halt) must not be
    engaged. ``command`` is a caller-supplied label that only colours the stage-error text.

    Returns ``(strategy, rec)`` so callers can read the registry record (e.g. ``rec.id``) without
    a second DB round-trip. A forward_tested strategy keeps accumulating evidence ticks while
    awaiting the go-live signature, so it is treated the same as paper for trading purposes.

    Lives in ``registry`` (not ``cli``) so any non-CLI consumer shares the SAME gate — paper/live
    gating can no longer drift via a copy in a command module.
    """
    strategy = load_tradable_strategy(name)
    rec = SqliteStrategyRepository(conn).get(name)
    if rec.stage not in (Stage.PAPER, Stage.FORWARD_TESTED):
        raise ValueError(
            f"{name} is at stage '{rec.stage.value}'; "
            f"{command} requires 'paper' or 'forward_tested'"
        )
    if global_halt.is_engaged(conn):
        # Distinguishable type (subclass of ValueError): a book-wide halt must abort a multi-tenant
        # run-all cycle whole, never be demoted to a single tenant's isolatable setup fault (#374).
        raise global_halt.GlobalHaltActive(
            "global halt active; clear with 'algua paper resume-all'")
    if kill_switch.is_tripped(conn, name):
        raise ValueError(f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'")
    return strategy, rec
