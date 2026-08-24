"""sqlite-backed registry store, carved by Protocol (spec §8). Each Protocol's implementation
lives in its own module (crud.py, approvals.py, search_breadth.py, holdout.py, gate.py,
forward_gate.py, family.py, backtest_returns.py, runs.py); a shared base.py holds the one helper
genuinely called from more than one domain (_apply_transition_locked); SqliteStrategyRepository
composes them via mixins and keeps only the truly Protocol-agnostic members (__init__, connection)
on itself directly."""
from __future__ import annotations

import sqlite3

# StrategyExists/StrategyNotFound/StrategyRecord are declared in repository.py, not here —
# re-exported (as `... as ...` per this repo's existing re-export convention, e.g.
# algua/data/store/__init__.py's `normalize_symbols as normalize_symbols`) so the current
# `from algua.registry.store import StrategyExists, ...` call sites keep working unmodified.
from algua.registry.repository import (
    StrategyExists as StrategyExists,
)
from algua.registry.repository import (
    StrategyNotFound as StrategyNotFound,
)
from algua.registry.repository import (
    StrategyRecord as StrategyRecord,
)
from algua.registry.store.approvals import ApprovalLedgerMixin
from algua.registry.store.backtest_returns import BacktestReturnsLedgerMixin
from algua.registry.store.crud import CrudMixin
from algua.registry.store.family import (
    AGENT_NOVEL_MINT_CAP as AGENT_NOVEL_MINT_CAP,
)
from algua.registry.store.family import FamilyGraphMixin
from algua.registry.store.forward_gate import ForwardGateMixin
from algua.registry.store.gate import GateLedgerMixin
from algua.registry.store.holdout import HoldoutLedgerMixin
from algua.registry.store.runs import RunLedgerMixin
from algua.registry.store.search_breadth import SearchBreadthLedgerMixin

__all__ = [
    "AGENT_NOVEL_MINT_CAP",
    "SqliteStrategyRepository",
    "StrategyExists",
    "StrategyNotFound",
    "StrategyRecord",
]


class SqliteStrategyRepository(
    CrudMixin,
    ApprovalLedgerMixin,
    SearchBreadthLedgerMixin,
    HoldoutLedgerMixin,
    GateLedgerMixin,
    ForwardGateMixin,
    FamilyGraphMixin,
    BacktestReturnsLedgerMixin,
    RunLedgerMixin,
):
    """sqlite-backed ``StrategyRepository``: the only module that embeds registry SQL."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Read-only handle to the underlying sqlite connection, for protected verifiers (the
        live wall's forward-certificate check) that read operational tables alongside the
        repository. Deliberately NOT part of the ``StrategyRepository`` Protocol — the seam
        stays I/O-agnostic; non-sqlite repos must inject their own verifier."""
        return self._conn
