"""Raw lifecycle edges that only a promote command may take (#682).

``backtested -> candidate`` and ``paper -> forward_tested`` are written by ``research promote`` and
``paper promote`` (each in its own gated transaction, never via ``registry transition``). The raw
``registry transition`` command refuses them for EVERY actor: a bare ``--actor human`` string used
to skip both gates.
"""
from __future__ import annotations

from algua.contracts.lifecycle import Stage, TransitionError

RAW_REFUSED_EDGES: dict[tuple[Stage, Stage], str] = {
    (Stage.BACKTESTED, Stage.CANDIDATE): "research promote",
    (Stage.PAPER, Stage.FORWARD_TESTED): "paper promote",
}


class RawEdgeRefused(TransitionError):
    """Stable code ``raw_edge_refused``."""


def refuse_raw_edge(source: Stage, target: Stage) -> None:
    command = RAW_REFUSED_EDGES.get((source, target))
    if command is not None:
        raise RawEdgeRefused(
            f"{source.value} -> {target.value} is not available through `registry transition`; "
            f"`algua {command}` is the only path")
