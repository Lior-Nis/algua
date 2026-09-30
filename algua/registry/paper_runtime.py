"""Pre-effect deployment/configuration setup for paper tenants.

The book preflight that routes working-tree and frozen tenants (``prepare_paper_book``) lives in
``frozen_runtime`` beside ``resolve_paper_tenant``, which composes the functions here.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from algua.contracts.lifecycle import Stage
from algua.registry import deployment_runtime as deploy
from algua.registry.allocations import active_allocation
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.registry.universe_binding import SOURCE_CONFIG_LEGACY, resolve_operational_universe

RuntimeTuple = tuple[Any, Any, ArtifactIdentity]


def paper_gate_universe(
    conn: sqlite3.Connection,
    name: str,
    config_universe: list[str],
    *,
    data_dir: Path,
    research_gate_id: int | None,
    logger: Any = None,
) -> list[str]:
    """The gate-bound operational universe a paper tenant ticks on (#559), shared by the
    working-tree and frozen (Story 1.3c) tenant paths; a pre-v39 gate falls back loudly."""
    universe, source = resolve_operational_universe(
        conn, data_dir, name, config_universe, research_gate_id=research_gate_id)
    if source == SOURCE_CONFIG_LEGACY and logger is not None:
        logger.warning("universe_binding_config_legacy", extra={"fields": {
            "strategy": name, "lane": "paper",
            "note": "gate has no universe_name; ticking on CONFIG.universe"}})
    return universe


def prepare_paper_runtime(
    conn: sqlite3.Connection,
    name: str,
    strategy: Any,
    rec: Any,
    *,
    data_dir: Path,
    identity_loader: Callable[[str], ArtifactIdentity],
    logger: Any = None,
) -> RuntimeTuple:
    """Verify deployment identity/config/universe before any provider or venue effect."""
    deployment, identity = deploy.resolve_tick(conn, rec.id, name, identity_loader)
    universe = paper_gate_universe(
        conn, name, strategy.universe, data_dir=data_dir,
        research_gate_id=(deployment.research_gate_id if deployment is not None else None),
        logger=logger)
    if universe != strategy.universe:
        strategy = replace(
            strategy, config=strategy.config.model_copy(update={"universe": universe}))
    return strategy, deployment, identity


def still_paper_allocated(conn: sqlite3.Connection, name: str) -> bool:
    """Re-read stage/allocation at submit time so a lane exit stops further submissions."""
    rec = SqliteStrategyRepository(conn).get(name)
    return (rec.stage in (Stage.PAPER, Stage.FORWARD_TESTED)
            and active_allocation(conn, rec.id) is not None)
