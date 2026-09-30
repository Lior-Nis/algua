"""Paper tenant resolution for the supervisor (Story 1.3c §2).

``resolve_paper_tenant`` is the one routing point on the paper tick path (``trade-tick`` directly,
``run-all`` through the ``prepare_paper_book`` preflight). A ``frozen`` tenant resolves from its
RECORDED descriptor and offline-verified content only: the checkout strategy module is never
imported, the identity is never recomputed from the checkout, and the working-tree verifier never
sees the row. ``working_tree`` and legacy tenants take exactly today's ``load_gated_strategy`` +
``prepare_paper_runtime`` path. ``paper run`` (checkout replay) refuses a frozen deployment.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from algua.registry import paper_runtime
from algua.registry.artifact_contract import BundleDescriptor
from algua.registry.artifact_recording import parse_frozen_deployment_manifest
from algua.registry.artifact_store import ArtifactStoreError, verify_bundle
from algua.registry.deployment import DeploymentError
from algua.registry.environment_contract import EnvironmentDescriptor
from algua.registry.environment_store import EnvironmentStoreError, verify_published_environment
from algua.registry.frozen_manifest_contract import FrozenManifest
from algua.registry.frozen_tenant_errors import (
    FrozenContentUnavailable,
    FrozenLiveUnsupported,
    FrozenTenantUnsupported,
)
from algua.registry.frozen_view import (
    FrozenStrategyView,
    decode_tenant_config,
    overlay_gate_universe,
)
from algua.registry.gating import load_gated_strategy, require_paper_gates
from algua.registry.repository import ArtifactIdentity, StrategyRecord
from algua.registry.store import DeploymentRecord, SqliteStrategyRepository
from algua.risk.global_halt import GlobalHaltActive
from algua.strategies.base import LoadedStrategy, StrategyConfig

Loader = Callable[[sqlite3.Connection, str, str], tuple[Any, Any]]
_Content = BundleDescriptor | EnvironmentDescriptor
# Exactly what the Story 1.3b offline verifier treats as missing or corrupt content.
_CONTENT_FAILURES = (ArtifactStoreError, EnvironmentStoreError, OSError, ValueError)


class FrozenContentVerifier:
    """Offline Story 1.3b verification of frozen content beneath ONE trusted store root.

    Each verdict (the verified root, or the failure) is cached per descriptor for the life of the
    object, so a supervisor process verifies each distinct bundle and environment once and a
    shared environment's failure fails every tenant on it individually. There is no lease: Phase 1
    never removes published content, and a future garbage collector must add one. The store root
    is bound here, not per call, so a cached verdict can never be reused under another root.
    """

    def __init__(self, store_root: Path) -> None:
        self._store_root = Path(store_root).resolve()
        self._verdicts: dict[_Content, tuple[Path | None, BaseException | None]] = {}

    @property
    def store_root(self) -> Path:
        return self._store_root

    def bundle(self, descriptor: BundleDescriptor, *, deployment_id: int | None = None) -> Path:
        return self._verified(descriptor, deployment_id)

    def environment(
        self, descriptor: EnvironmentDescriptor, *, deployment_id: int | None = None,
    ) -> Path:
        return self._verified(descriptor, deployment_id)

    def _verified(self, descriptor: _Content, deployment_id: int | None) -> Path:
        if descriptor not in self._verdicts:
            try:
                if isinstance(descriptor, BundleDescriptor):
                    root = verify_bundle(self._store_root, descriptor)
                else:
                    root = verify_published_environment(self._store_root, descriptor)
            except _CONTENT_FAILURES as exc:
                self._verdicts[descriptor] = (None, exc)
            else:
                self._verdicts[descriptor] = (root, None)
        verified, failure = self._verdicts[descriptor]
        if verified is None:
            raise FrozenContentUnavailable(deployment_id=deployment_id) from failure
        return verified


@dataclass(frozen=True)
class WorkingTreeTenant:
    """A ``working_tree`` or legacy tenant: exactly today's gated load + runtime preparation."""

    strategy: LoadedStrategy  # gate universe overlaid by prepare_paper_runtime
    rec: StrategyRecord
    deployment: DeploymentRecord | None
    identity: ArtifactIdentity

    @property
    def name(self) -> str:
        return self.rec.name

    @property
    def runtime(self) -> paper_runtime.RuntimeTuple:
        return self.strategy, self.deployment, self.identity


@dataclass(frozen=True)
class FrozenTenant:
    """A ``frozen`` tenant: descriptor, verified content locators and the supervisor view."""

    rec: StrategyRecord
    deployment: DeploymentRecord
    manifest: FrozenManifest
    bundle_root: Path
    environment_root: Path
    view: FrozenStrategyView

    @property
    def name(self) -> str:
        return self.rec.name

    @property
    def strategy_id(self) -> int:
        return self.rec.id

    @property
    def deployment_id(self) -> int:
        return self.deployment.id

    @property
    def artifact_id(self) -> int:
        return self.deployment.artifact_id

    @property
    def interpreter(self) -> Path:
        return self.environment_root / "bin" / "python"

    @property
    def identity(self) -> ArtifactIdentity:
        """The descriptor's identity: what the tick is stamped with and the tick-snapshot guard
        checks against the artifact row. Never recomputed from the checkout."""
        return ArtifactIdentity(
            self.manifest.code_hash, self.manifest.config_hash, self.manifest.dependency_hash)

    @property
    def runtime(self) -> paper_runtime.RuntimeTuple:
        return self.view, self.deployment, self.identity


PaperTenant = WorkingTreeTenant | FrozenTenant


def _active_frozen_deployment(conn: sqlite3.Connection, name: str) -> int | None:
    """The active deployment's id when it is ``frozen``. Routes without raising, so a non-frozen
    tenant fails in exactly today's order."""
    row = conn.execute(
        "SELECT d.id FROM strategies s"
        " JOIN strategy_deployments d ON d.strategy_id=s.id AND d.retired_at IS NULL"
        " JOIN deployment_artifacts a ON a.id=d.artifact_id"
        " WHERE s.name=? AND a.source_kind='frozen'",
        (name,),
    ).fetchone()
    return None if row is None else int(row[0])


def refuse_checkout_replay(conn: sqlite3.Connection, name: str) -> None:
    """``paper run`` replays the CHECKOUT module, which never stands in for a frozen deployment:
    refuse it with ``frozen_live_unsupported`` before anything is loaded or replayed (§2)."""
    deployment_id = _active_frozen_deployment(conn, name)
    if deployment_id is not None:
        raise FrozenLiveUnsupported(deployment_id)


def resolve_paper_tenant(
    conn: sqlite3.Connection,
    name: str,
    *,
    command: str,
    verifier: FrozenContentVerifier,
    data_dir: Path,
    identity_loader: Callable[[str], ArtifactIdentity],
    logger: Any = None,
    loader: Loader | None = None,
) -> WorkingTreeTenant | FrozenTenant:
    """Resolve one paper tenant before any provider or venue effect.

    ``loader`` and ``identity_loader`` serve only the unchanged working-tree/legacy path; the
    frozen path uses the recorded descriptor, ``verifier`` and the paper gates alone."""
    if _active_frozen_deployment(conn, name) is None:
        strategy, rec = (loader or load_gated_strategy)(conn, name, command)
        strategy, deployment, identity = paper_runtime.prepare_paper_runtime(
            conn, name, strategy, rec, data_dir=data_dir, identity_loader=identity_loader,
            logger=logger)
        return WorkingTreeTenant(strategy, rec, deployment, identity)
    return _resolve_frozen(
        conn, name, command=command, verifier=verifier, data_dir=data_dir, logger=logger)


def recorded_descriptor(
    deployment: DeploymentRecord, name: str,
) -> tuple[FrozenManifest, StrategyConfig]:
    """A frozen deployment's RECORDED descriptor (its ``frozen_wire`` stamp is the supported
    protocol) and its config, strictly decoded for ``name``: ``frozen_content_unavailable`` for a
    corrupt descriptor, ``frozen_content_unsupported`` for a config this supervisor cannot run.
    Shared by tenant resolution and forward promotion; touches no content and no checkout."""
    try:
        manifest = parse_frozen_deployment_manifest(deployment.manifest())
    except DeploymentError as exc:
        raise FrozenContentUnavailable(
            "recorded descriptor", deployment_id=deployment.id) from exc
    try:
        return manifest, decode_tenant_config(manifest, name)
    except FrozenTenantUnsupported as exc:
        exc.deployment_id = deployment.id
        raise


def _resolve_frozen(
    conn: sqlite3.Connection, name: str, *, command: str, verifier: FrozenContentVerifier,
    data_dir: Path, logger: Any,
) -> FrozenTenant:
    rec = require_paper_gates(conn, name, command)
    deployment = SqliteStrategyRepository(conn).active_deployment(rec.id)
    if deployment is None or deployment.source_kind != "frozen":
        # The routing probe and this read are separate statements: an epoch change in between
        # isolates this tenant for the cycle rather than resolving a half-read deployment.
        raise DeploymentError(f"{name} active deployment changed during tenant resolution")
    manifest, config = recorded_descriptor(deployment, name)
    universe = paper_runtime.paper_gate_universe(
        conn, name, list(config.universe), data_dir=data_dir,
        research_gate_id=deployment.research_gate_id, logger=logger)
    view = overlay_gate_universe(config, universe)
    bundle_root = verifier.bundle(manifest.bundle, deployment_id=deployment.id)
    environment_root = verifier.environment(manifest.environment, deployment_id=deployment.id)
    return FrozenTenant(rec, deployment, manifest, bundle_root, environment_root, view)


@dataclass(frozen=True)
class PaperBookSetup:
    tenants: dict[str, PaperTenant]
    tickable: list[Any]
    failures: list[tuple[str, Exception]]

    @property
    def frozen_views(self) -> dict[str, FrozenStrategyView]:
        """Each frozen tenant's supervisor view: what cycle planning reads for it."""
        return {name: t.view for name, t in self.tenants.items() if isinstance(t, FrozenTenant)}


def prepare_paper_book(
    conn: sqlite3.Connection,
    records: Iterable[Any],
    *,
    verifier: FrozenContentVerifier,
    data_dir: Path,
    identity_loader: Callable[[str], ArtifactIdentity],
    logger: Any = None,
    loader: Loader | None = None,
) -> PaperBookSetup:
    """Preflight every tenant before planning, refresh, reconciliation or any venue effect: resolve
    it (a frozen tenant's content is verified offline here, once per digest for the process),
    isolate a tenant-local failure and propagate systemic state."""
    tenants: dict[str, PaperTenant] = {}
    tickable: list[Any] = []
    failures: list[tuple[str, Exception]] = []
    for record in records:
        try:
            tenants[record.name] = resolve_paper_tenant(
                conn, record.name, command="paper run-all", verifier=verifier,
                data_dir=data_dir, identity_loader=identity_loader, logger=logger, loader=loader)
        except (sqlite3.Error, GlobalHaltActive):
            raise
        except Exception as exc:  # noqa: BLE001 - per-tenant pre-effect isolation boundary
            failures.append((record.name, exc))
        else:
            tickable.append(record)
    return PaperBookSetup(tenants=tenants, tickable=tickable, failures=failures)


def recheck_paper_gates(
    conn: sqlite3.Connection, tenant: PaperTenant, command: str, *, loader: Loader | None = None,
) -> None:
    """Re-clear the paper gates right before ``tenant`` ticks (a sibling may have tripped them
    since preflight): a working-tree tenant re-loads exactly as before, a frozen tenant clears the
    same gates without touching the checkout."""
    if isinstance(tenant, FrozenTenant):
        require_paper_gates(conn, tenant.name, command)
    else:
        (loader or load_gated_strategy)(conn, tenant.name, command)
