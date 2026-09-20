from __future__ import annotations

import sqlite3

from algua.contracts.lifecycle import Stage
from algua.registry.repository import StrategyRecord
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch
from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import load_tradable_strategy


class StageNotTradable(ValueError):
    """The strategy is not at a stage this lane may trade."""


class KillSwitchTripped(ValueError):
    """This strategy's kill-switch is engaged."""


class NoAllocation(ValueError):
    """The strategy has no capital allocation on this lane."""


# WHY THESE EXIST. `StrategySetupError.code` is derived from the raising exception's CLASS NAME,
# deliberately, so no raw message (which can carry credentials or paths) ever reaches the JSON
# envelope or the audit trail. That redaction is right, but it means a bare `ValueError` audits as
# the useless string "ValueError" -- and for 24 hours on the live box, four halted strategies wrote
# exactly that, with no way to tell a tripped kill-switch from a wrong stage from a missing
# allocation without reproducing the tick.
#
# Naming the failure IS the fix: the existing mechanism then yields a stable, meaningful,
# leak-free code. These stay ValueError subclasses so every existing handler keeps working, the
# same way global_halt.GlobalHaltActive does.


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
        raise StageNotTradable(
            f"{name} is at stage '{rec.stage.value}'; "
            f"{command} requires 'paper' or 'forward_tested'"
        )
    if global_halt.is_engaged(conn):
        # Distinguishable type (subclass of ValueError): a book-wide halt must abort a multi-tenant
        # run-all cycle whole, never be demoted to a single tenant's isolatable setup fault (#374).
        raise global_halt.GlobalHaltActive(
            "global halt active; clear with 'algua paper resume-all'")
    if kill_switch.is_tripped(conn, name):
        raise KillSwitchTripped(
            f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'")
    return strategy, rec
