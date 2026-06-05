from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

from algua.contracts.lifecycle import Actor, Stage


class ArtifactIdentity(NamedTuple):
    """The full identity a human approval binds to and the live gate recomputes.

    A ``NamedTuple`` so callers can either unpack ``(code_hash, config_hash, dependency_hash)``
    or read fields by name. ``dependency_hash`` is ``None`` only when the lockfile is absent;
    such an identity can never match a stored approval (see ``has_valid_approval``)."""

    code_hash: str
    config_hash: str
    dependency_hash: str | None


class StrategyExists(ValueError):
    pass


class StrategyNotFound(LookupError):
    pass


@dataclass
class StrategyRecord:
    id: int
    name: str
    stage: Stage
    created_at: str
    updated_at: str


class StrategyRepository(Protocol):
    """Persistence seam for the registry.

    The lifecycle policy (transitions, approvals) depends on this Protocol, never on a concrete
    database driver. The sqlite implementation lives in ``algua.registry.store`` and is the only
    place that knows SQL; swapping the backing store means writing another implementation, not
    touching policy code.
    """

    def add(self, name: str) -> StrategyRecord:
        """Insert a new strategy at stage ``idea`` with its initial transition row.

        Raises ``StrategyExists`` if the name is already registered.
        """
        ...

    def get(self, name: str) -> StrategyRecord:
        """Return the strategy by name, or raise ``StrategyNotFound``."""
        ...

    def list_strategies(self, stage: Stage | None = None) -> list[StrategyRecord]:
        """List strategies, optionally filtered to a single stage, ordered by insertion."""
        ...

    def list_transitions(self, name: str) -> list[dict]:
        """Return the strategy's ordered stage-transition history."""
        ...

    def apply_transition(
        self,
        rec: StrategyRecord,
        to: Stage,
        actor: Actor,
        reason: str | None = None,
        code_hash: str | None = None,
        config_hash: str | None = None,
    ) -> StrategyRecord:
        """Atomically advance ``rec`` to ``to``, append a transition row, return the new state."""
        ...

    def record_approval(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        approved_by: str,
    ) -> int:
        """Persist a human approval pinning ``code_hash``/``config_hash``/``dependency_hash``;
        return its row id."""
        ...

    def has_valid_approval(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
    ) -> bool:
        """True iff a non-revoked approval pins exactly this strategy + code + config +
        dependency set. A ``None`` ``dependency_hash`` (no lockfile) never matches, and a stored
        row with a NULL ``dependency_hash`` never matches a concrete hash — both fail closed."""
        ...
