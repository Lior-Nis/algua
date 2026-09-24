"""Pre-effect deployment/configuration setup for paper tenants."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from algua.contracts.lifecycle import Stage
from algua.registry import deployment_runtime as deploy
from algua.registry.allocations import active_allocation
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.registry.universe_binding import SOURCE_CONFIG_LEGACY, resolve_operational_universe
from algua.risk.global_halt import GlobalHaltActive

RuntimeTuple = tuple[Any, Any, ArtifactIdentity]


@dataclass(frozen=True)
class PaperBookSetup:
    runtimes: dict[str, RuntimeTuple]
    tickable: list[Any]
    failures: list[tuple[str, Exception]]


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
    universe, source = resolve_operational_universe(
        conn, data_dir, name, strategy.universe,
        research_gate_id=(deployment.research_gate_id if deployment is not None else None))
    if source == SOURCE_CONFIG_LEGACY and logger is not None:
        logger.warning("universe_binding_config_legacy", extra={"fields": {
            "strategy": name, "lane": "paper",
            "note": "gate has no universe_name; ticking on CONFIG.universe"}})
    if universe != strategy.universe:
        strategy = replace(
            strategy, config=strategy.config.model_copy(update={"universe": universe}))
    return strategy, deployment, identity


def prepare_paper_book(
    conn: sqlite3.Connection,
    tenants: Iterable[Any],
    *,
    data_dir: Path,
    loader: Callable[[sqlite3.Connection, str, str], tuple[Any, Any]],
    identity_loader: Callable[[str], ArtifactIdentity],
    logger: Any = None,
) -> PaperBookSetup:
    """Preflight all tenants; isolate local setup errors and propagate systemic state."""
    runtimes: dict[str, RuntimeTuple] = {}
    tickable: list[Any] = []
    failures: list[tuple[str, Exception]] = []
    for tenant in tenants:
        try:
            strategy, rec = loader(conn, tenant.name, "paper run-all")
            runtimes[tenant.name] = prepare_paper_runtime(
                conn, tenant.name, strategy, rec, data_dir=data_dir,
                identity_loader=identity_loader, logger=logger)
        except (sqlite3.Error, GlobalHaltActive):
            raise
        except Exception as exc:  # noqa: BLE001 - per-tenant pre-effect isolation boundary
            failures.append((tenant.name, exc))
        else:
            tickable.append(tenant)
    return PaperBookSetup(runtimes=runtimes, tickable=tickable, failures=failures)


def still_paper_allocated(conn: sqlite3.Connection, name: str) -> bool:
    """Re-read stage/allocation at submit time so a lane exit stops further submissions."""
    rec = SqliteStrategyRepository(conn).get(name)
    return (rec.stage in (Stage.PAPER, Stage.FORWARD_TESTED)
            and active_allocation(conn, rec.id) is not None)
