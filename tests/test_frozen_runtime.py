"""Story 1.3c §2: registry-side tenant resolution for the paper supervisor.

A ``frozen`` tenant resolves from its RECORDED descriptor and verified content only: no checkout
strategy load, no identity recomputation, no working-tree verification. ``working_tree`` and legacy
tenants keep today's ``load_gated_strategy`` + ``prepare_paper_runtime`` path unchanged. The live
lane refuses a frozen row with ``frozen_live_unsupported``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from algua.contracts.canonical import canonical_json
from algua.contracts.lifecycle import Actor
from algua.contracts.types import ExecutionContract
from algua.data.store import DataStore
from algua.execution.tick_snapshots import record_tick_snapshot
from algua.registry import frozen_runtime, paper_runtime
from algua.registry.artifact_contract import BundleDescriptor
from algua.registry.artifact_recording import frozen_deployment_manifest
from algua.registry.artifact_store import publish_bundle
from algua.registry.db import connect, migrate
from algua.registry.deployment import DeploymentError
from algua.registry.deployment_runtime import resolve_tick
from algua.registry.environment_contract import EnvironmentDescriptor, EnvironmentKey
from algua.registry.environment_store import publish_environment
from algua.registry.frozen_manifest_contract import FrozenManifest
from algua.registry.frozen_runtime import (
    FrozenContentVerifier,
    FrozenTenant,
    WorkingTreeTenant,
    resolve_paper_tenant,
)
from algua.registry.frozen_source import FrozenFile
from algua.registry.frozen_tenant_errors import (
    FrozenContentUnavailable,
    FrozenLiveUnsupported,
    FrozenTenantUnsupported,
)
from algua.registry.planner_environment import current_interpreter_identity, inventory_environment
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch
from algua.strategies.base import StrategyConfig, config_hash
from algua.strategies.loader import _index, _loaded_for_test, load_tradable_strategy
from tests._deployment_helpers import force_legacy_strategy, frozen_manifest
from tests._frozen_evidence_helpers import record_final_invocation
from tests._venv_fixture import SITE_PACKAGES, uv_like_venv

DEP = "d" * 64
GATE_UNIVERSE = ["MSFT", "GOOGL"]


# --- fixtures ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Content:
    store: Path
    bundles: dict[str, BundleDescriptor]
    environment: EnvironmentDescriptor


def _publish(root: Path, names: tuple[str, ...]) -> Content:
    """Really publish one bundle per tenant and ONE shared environment (Story 1.3b stores)."""
    store = root / "store"
    bundles: dict[str, BundleDescriptor] = {}
    for name in names:
        files = tuple(sorted((
            FrozenFile("_algua/protocol.json", "100644", b'{"descriptor_version":1}'),
            FrozenFile("algua/__init__.py", "100644", f"TENANT = {name!r}\n".encode()),
        ), key=lambda item: item.path.encode()))
        bundle = BundleDescriptor.from_files(tuple(item.contract_entry for item in files))
        publish_bundle(store, files, bundle)
        bundles[name] = bundle
    built = uv_like_venv(root / "build")
    (built / SITE_PACKAGES / "six.py").write_text("VERSION = '1.17.0'\n")
    identity = current_interpreter_identity()
    key = EnvironmentKey(
        build_inputs_digest="1" * 64, dependency_hash=DEP, interpreter=identity,
        uv_version="uv 0.9.26", create_argv=("uv", "venv"), sync_argv=("uv", "sync"),
    )
    environment = EnvironmentDescriptor(key, inventory_environment(built).digest, identity)
    publish_environment(store, built, environment)
    return Content(store, bundles, environment)


def _config(name: str, **overrides) -> StrategyConfig:
    fields = {
        "name": name, "universe": ["AAPL", "MSFT", "NVDA"],
        "execution": ExecutionContract(rebalance_frequency="1d", warmup_bars=3),
        "params": {"lookback": 20}, "construction": "top_k_equal_weight",
        "construction_params": {"top_k": 2}, "feature_lookback": 20,
    }
    fields.update(overrides)
    return StrategyConfig(**fields)


def _recorded(config: StrategyConfig) -> dict:
    return json.loads(canonical_json(config.model_dump(mode="json")))


def _frozen(content: Content, bundle_of: str, config: StrategyConfig, *,
            universe_name: str | None = "liquid-us", recorded: dict | None = None,
            digest: str | None = None) -> FrozenManifest:
    return FrozenManifest(
        source_ref="a" * 40, code_hash="b" * 32,
        config_hash=digest if digest is not None else config_hash(_loaded_for_test(config)),
        dependency_hash=DEP,
        resolved_config=recorded if recorded is not None else _recorded(config),
        universe_name=universe_name, bundle=content.bundles[bundle_of],
        environment=content.environment,
    )


def _registry(tmp_path: Path) -> SqliteStrategyRepository:
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def _gate(repo: SqliteStrategyRepository, strategy_id: int, *, code_hash: str, config_hash: str,
          dependency_hash: str, universe_name: str | None) -> int:
    return repo.record_gate_evaluation(
        strategy_id, passed=True, n_funnel=1, own_lifetime_combos=1,
        windowed_total_combos=1, funnel_window_days=90, breadth_provenance="measured",
        pit_ok=True, pit_override=False, holdout_n_bars=63, min_holdout_observations=63,
        code_hash=code_hash, config_hash=config_hash, dependency_hash=dependency_hash,
        data_source="test", snapshot_id="snap", period_start="2024-01-01",
        period_end="2024-12-31", holdout_frac=0.2, actor="human", decision_json="{}",
        universe_name=universe_name,
    )


def _admit(repo: SqliteStrategyRepository, name: str, manifest: FrozenManifest) -> int:
    """Register ``name`` as a qualified candidate and admit it through the real frozen intake."""
    rec = repo.add(name)
    gate_id = _gate(
        repo, rec.id, code_hash=manifest.code_hash, config_hash=manifest.config_hash,
        dependency_hash=manifest.dependency_hash, universe_name=manifest.universe_name)
    conn = repo.connection
    created_at = conn.execute(
        "SELECT created_at FROM gate_evaluations WHERE id=?", (gate_id,)).fetchone()[0]
    conn.execute("UPDATE strategies SET stage='candidate' WHERE id=?", (rec.id,))
    conn.execute(
        "INSERT INTO stage_transitions(strategy_id, from_stage, to_stage, actor, reason,"
        " code_hash, config_hash, dependency_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (rec.id, "backtested", "candidate", "human", "fixture", manifest.code_hash,
         manifest.config_hash, manifest.dependency_hash, created_at),
    )
    conn.commit()
    repo.intake_candidate_to_paper(
        repo.get(name), capital=1000.0, actor=Actor.AGENT, account_equity=100_000.0,
        max_concurrent=10, deployment_manifest=frozen_deployment_manifest(manifest),
        research_gate_id=gate_id,
    )
    return rec.id


def _data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    DataStore(data).ingest_universe(
        universe="liquid-us", symbols=GATE_UNIVERSE, effective_date="2020-01-01",
        as_of="2020-01-01T00:00:00Z", source="test")
    return data


def _no_loader(*_args):
    raise AssertionError("a frozen tenant must never load the checkout strategy")


def _no_identity(_name):
    raise AssertionError("a frozen tenant must never recompute identity from the checkout")


def _resolve(repo, name, verifier, data_dir, **overrides):
    kwargs = {"command": "trade-tick", "verifier": verifier, "data_dir": data_dir,
              "identity_loader": _no_identity, "loader": _no_loader}
    kwargs.update(overrides)
    return resolve_paper_tenant(repo.connection, name, **kwargs)


@pytest.fixture
def world(tmp_path):
    """Two frozen tenants on distinct bundles sharing ONE published environment."""
    content = _publish(tmp_path, ("s", "t"))
    repo = _registry(tmp_path)
    manifests = {name: _frozen(content, name, _config(name)) for name in ("s", "t")}
    ids = {name: _admit(repo, name, manifests[name]) for name in ("s", "t")}
    return content, repo, manifests, ids, _data_dir(tmp_path)


class _Counting:
    """Wrap a 1.3b verifier in the frozen_runtime namespace and count the real calls."""

    def __init__(self, monkeypatch, name: str) -> None:
        self.calls: list[object] = []
        real = getattr(frozen_runtime, name)

        def counted(store_root, descriptor):
            self.calls.append(descriptor)
            return real(store_root, descriptor)

        monkeypatch.setattr(frozen_runtime, name, counted)


# --- frozen resolution -------------------------------------------------------------------------


def test_frozen_tenant_resolves_from_descriptor_and_verified_content(world):
    content, repo, manifests, ids, data_dir = world
    tenant = _resolve(repo, "s", FrozenContentVerifier(content.store), data_dir)

    assert isinstance(tenant, FrozenTenant)
    deployment = repo.active_deployment(ids["s"])
    assert deployment is not None
    assert tenant.name == "s"
    assert tenant.strategy_id == ids["s"] == tenant.rec.id
    assert tenant.deployment == deployment
    assert tenant.deployment_id == deployment.id
    assert tenant.artifact_id == deployment.artifact_id
    assert tenant.manifest == manifests["s"]
    root = content.store.resolve()
    assert tenant.bundle_root == root / manifests["s"].bundle.locator
    assert tenant.environment_root == root / manifests["s"].environment.locator
    assert tenant.interpreter == tenant.environment_root / "bin" / "python"
    assert tenant.interpreter.exists()
    # Identity is the descriptor's, never recomputed from the checkout.
    assert tenant.identity == ArtifactIdentity(
        manifests["s"].code_hash, manifests["s"].config_hash, manifests["s"].dependency_hash)
    # The view carries the GATE universe; the descriptor keeps its template universe.
    assert tenant.view.universe == ("GOOGL", "MSFT")
    assert tenant.view.config.universe == ["GOOGL", "MSFT"]
    assert tenant.view.name == "s"
    assert tenant.view.execution.warmup_bars == 3
    assert tenant.manifest.resolved_config["universe"] == ["AAPL", "MSFT", "NVDA"]
    assert tenant.runtime == (tenant.view, deployment, tenant.identity)


def test_frozen_tenant_never_imports_its_checkout_strategy_module(tmp_path):
    """A real repo strategy admitted frozen resolves with its module absent from the process."""
    name = "momentum_regime_stop"
    loaded = load_tradable_strategy(name)
    content = _publish(tmp_path, (name,))
    repo = _registry(tmp_path)
    manifest = _frozen(content, name, loaded.config, digest=config_hash(loaded))
    _admit(repo, name, manifest)
    dotted = _index()[name]
    sys.modules.pop(dotted, None)

    tenant = _resolve(repo, name, FrozenContentVerifier(content.store), _data_dir(tmp_path))

    assert isinstance(tenant, FrozenTenant)
    assert dotted not in sys.modules
    assert tenant.view.config.overlays == loaded.config.overlays
    assert tenant.view.universe == ("GOOGL", "MSFT")


def test_frozen_tick_is_stamped_with_the_descriptor_identity(world):
    content, repo, _manifests, ids, data_dir = world
    tenant = _resolve(repo, "s", FrozenContentVerifier(content.store), data_dir)
    # A frozen tick carries its final invocation link (Story 1.3d, the v48 tick trigger).
    link = record_final_invocation(
        repo.connection, deployment_id=tenant.deployment_id, snapshot_id="snap-1")

    def stamp(identity: ArtifactIdentity) -> None:
        record_tick_snapshot(
            repo.connection, "s", tick_ts="2026-09-30T20:00:00+00:00", decision_ts=None,
            equity=1000.0, peak_equity=1000.0, positions={}, n_submitted=0, reconcile_ok=True,
            lane="paper", strategy_id=tenant.strategy_id, code_hash=identity.code_hash,
            config_hash=identity.config_hash, dependency_hash=identity.dependency_hash,
            account_id="paper", cash=1000.0, clock_source="broker", snapshot_id="snap-1",
            deployment_id=tenant.deployment_id, frozen_invocation_id=link)

    stamp(tenant.identity)
    row = repo.connection.execute(
        "SELECT code_hash, config_hash, dependency_hash, deployment_id FROM tick_snapshots"
        " WHERE strategy_id=?", (ids["s"],)).fetchone()
    assert tuple(row) == (*tenant.identity, tenant.deployment_id)
    with pytest.raises(DeploymentError, match="artifact identity"):
        stamp(tenant.identity._replace(code_hash="e" * 32))


def test_frozen_resolution_keeps_the_paper_gates(world):
    content, repo, _manifests, _ids, data_dir = world
    verifier = FrozenContentVerifier(content.store)
    kill_switch.trip(repo.connection, "s", reason="test", actor="human")
    with pytest.raises(ValueError, match="kill-switch tripped for s"):
        _resolve(repo, "s", verifier, data_dir)
    global_halt.engage(repo.connection, reason="test", actor="human")
    with pytest.raises(global_halt.GlobalHaltActive):
        _resolve(repo, "t", verifier, data_dir)
    global_halt.clear(repo.connection)
    repo.connection.execute("UPDATE strategies SET stage='retired' WHERE name='t'")
    repo.connection.commit()
    with pytest.raises(ValueError, match="requires 'paper' or 'forward_tested'"):
        _resolve(repo, "t", verifier, data_dir)


def test_frozen_config_for_another_strategy_is_unsupported(tmp_path):
    content = _publish(tmp_path, ("s",))
    repo = _registry(tmp_path)
    sid = _admit(repo, "s", _frozen(content, "s", _config("impostor")))

    with pytest.raises(FrozenTenantUnsupported) as info:
        _resolve(repo, "s", FrozenContentVerifier(content.store), _data_dir(tmp_path))

    deployment = repo.active_deployment(sid)
    assert deployment is not None
    assert info.value.code == "frozen_content_unsupported"
    assert info.value.deployment_id == deployment.id


def test_frozen_undecodable_config_is_unsupported_before_content_is_touched(tmp_path,
                                                                            monkeypatch):
    content = _publish(tmp_path, ("s",))
    repo = _registry(tmp_path)
    config = _config("s", execution=ExecutionContract(rebalance_frequency="1d", fees=0))
    recorded = _recorded(config)
    recorded["execution"]["fees"] = 0  # a JSON number that does not dump back exactly
    _admit(repo, "s", _frozen(content, "s", config, recorded=recorded))
    bundles = _Counting(monkeypatch, "verify_bundle")

    with pytest.raises(FrozenTenantUnsupported):
        _resolve(repo, "s", FrozenContentVerifier(content.store), _data_dir(tmp_path))
    assert bundles.calls == []


# --- the verification cache --------------------------------------------------------------------


def test_shared_environment_is_verified_once_per_verifier(world, monkeypatch):
    content, repo, _manifests, _ids, data_dir = world
    environments = _Counting(monkeypatch, "verify_published_environment")
    bundles = _Counting(monkeypatch, "verify_bundle")
    verifier = FrozenContentVerifier(content.store)

    first = _resolve(repo, "s", verifier, data_dir)
    second = _resolve(repo, "t", verifier, data_dir)
    again = _resolve(repo, "s", verifier, data_dir)

    assert len(environments.calls) == 1
    assert len(bundles.calls) == 2
    assert first.environment_root == second.environment_root == again.environment_root
    assert first.bundle_root != second.bundle_root
    # The cache lives for the life of the verifier only: a new process-cycle re-verifies.
    _resolve(repo, "s", FrozenContentVerifier(content.store), data_dir)
    assert len(environments.calls) == 2


def test_missing_content_is_unavailable(world, tmp_path):
    _content, repo, _manifests, ids, data_dir = world
    empty = tmp_path / "empty-store"
    empty.mkdir()

    with pytest.raises(FrozenContentUnavailable) as info:
        _resolve(repo, "s", FrozenContentVerifier(empty), data_dir)

    deployment = repo.active_deployment(ids["s"])
    assert deployment is not None
    assert info.value.code == "frozen_content_unavailable"
    assert info.value.deployment_id == deployment.id
    assert isinstance(info.value.__cause__, ValueError)


def _unseal_and_write(path: Path, data: bytes) -> None:
    parent = path.parent
    parent.chmod(0o755)
    path.chmod(0o644)
    path.write_bytes(data)
    path.chmod(0o444)
    parent.chmod(0o555)


def test_corrupt_bundle_is_unavailable(world):
    content, repo, manifests, _ids, data_dir = world
    target = content.store / manifests["s"].bundle.locator / "algua" / "__init__.py"
    _unseal_and_write(target, b"TENANT = 'tampered'\n")

    with pytest.raises(FrozenContentUnavailable):
        _resolve(repo, "s", FrozenContentVerifier(content.store), data_dir)


def test_shared_environment_failure_fails_each_tenant_individually(world, monkeypatch):
    content, repo, manifests, ids, data_dir = world
    site = content.store / manifests["s"].environment.locator / SITE_PACKAGES
    site.chmod(0o755)  # permission drift: the published environment is no longer sealed
    environments = _Counting(monkeypatch, "verify_published_environment")
    verifier = FrozenContentVerifier(content.store)

    failures = {}
    for name in ("s", "t"):
        with pytest.raises(FrozenContentUnavailable) as info:
            _resolve(repo, name, verifier, data_dir)
        failures[name] = info.value.deployment_id

    expected = {}
    for name in ("s", "t"):
        deployment = repo.active_deployment(ids[name])
        assert deployment is not None
        expected[name] = deployment.id
    assert failures == expected
    assert len(environments.calls) == 1  # the failed verdict is cached too
    site.chmod(0o555)


def test_corrupt_descriptor_is_unavailable(world):
    content, repo, _manifests, ids, data_dir = world
    conn = repo.connection
    deployment = repo.active_deployment(ids["s"])
    assert deployment is not None
    trigger = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='trg_deployment_artifacts_no_update'"
    ).fetchone()[0]
    conn.execute("DROP TRIGGER trg_deployment_artifacts_no_update")
    conn.execute("UPDATE deployment_artifacts SET environment_digest=? WHERE id=?",
                 ("0" * 64, deployment.artifact_id))
    conn.execute(trigger)
    conn.commit()

    with pytest.raises(FrozenContentUnavailable) as info:
        _resolve(repo, "s", FrozenContentVerifier(content.store), data_dir)
    assert info.value.deployment_id == deployment.id


# --- working-tree and legacy tenants keep today's path -----------------------------------------


class _ForbiddenVerifier(FrozenContentVerifier):
    def __init__(self) -> None:
        super().__init__(Path("/nonexistent"))

    def bundle(self, *_args, **_kwargs):
        raise AssertionError("working-tree tenants never touch frozen content")

    def environment(self, *_args, **_kwargs):
        raise AssertionError("working-tree tenants never touch frozen content")


def _working_tree_row(repo, strategy_id: int, gate_id: int, identity: ArtifactIdentity) -> None:
    conn = repo.connection
    conn.execute(
        "INSERT INTO deployment_artifacts(manifest_digest, manifest_json, code_hash,"
        " config_hash, dependency_hash, resolved_config_json, universe_name,"
        " environment_digest, python_implementation, python_version, abi_tag, platform_tag,"
        " planner_protocol_version, source_kind, source_ref, asset_digests_json, created_at)"
        " VALUES ('wt-fixture','{}',?,?,?,'{}','liquid-us','e','CPython','3.12','abi',"
        " 'platform',1,'working_tree','ref','[]','test-fixture')", tuple(identity))
    artifact_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO strategy_deployments(strategy_id, artifact_id, research_gate_id,"
        " activated_at) VALUES (?,?,?,'2020-01-01T00:00:00+00:00')",
        (strategy_id, artifact_id, gate_id))
    conn.execute("UPDATE strategies SET stage='paper' WHERE id=?", (strategy_id,))
    conn.commit()


def _todays_path(repo, name, loader, identity_loader, data_dir):
    strategy, rec = loader(repo.connection, name, "trade-tick")
    return paper_runtime.prepare_paper_runtime(
        repo.connection, name, strategy, rec, data_dir=data_dir,
        identity_loader=identity_loader)


def test_working_tree_tenant_is_todays_path(tmp_path, monkeypatch):
    repo = _registry(tmp_path)
    data_dir = _data_dir(tmp_path)
    identity = ArtifactIdentity("c" * 32, "f" * 32, DEP)
    sid = repo.add("w").id
    gate_id = _gate(repo, sid, code_hash=identity.code_hash, config_hash=identity.config_hash,
                    dependency_hash=DEP, universe_name="liquid-us")
    _working_tree_row(repo, sid, gate_id, identity)
    verified: list[object] = []
    monkeypatch.setattr("algua.registry.deployment.verify_working_tree_manifest",
                        lambda manifest, repo_root: verified.append(manifest))
    loaded = _loaded_for_test(_config("w"))
    loads: list[tuple[str, str]] = []

    def loader(conn, name, command):
        loads.append((name, command))
        return loaded, SqliteStrategyRepository(conn).get(name)

    tenant = _resolve(repo, "w", _ForbiddenVerifier(), data_dir, loader=loader,
                      identity_loader=lambda name: identity)

    assert isinstance(tenant, WorkingTreeTenant)
    assert loads == [("w", "trade-tick")]
    assert len(verified) == 1  # the working-tree descriptor is still verified every tick
    expected = _todays_path(repo, "w", loader, lambda name: identity, data_dir)
    assert tenant.runtime == expected
    assert tenant.strategy.universe == ["GOOGL", "MSFT"]
    assert tenant.deployment is not None and tenant.deployment.source_kind == "working_tree"
    assert tenant.identity == identity
    assert tenant.rec.id == sid and tenant.name == "w"


def test_legacy_tenant_is_todays_path(tmp_path):
    repo = _registry(tmp_path)
    data_dir = _data_dir(tmp_path)
    identity = ArtifactIdentity("c" * 32, "f" * 32, DEP)
    sid = repo.add("legacy").id
    _gate(repo, sid, code_hash=identity.code_hash, config_hash=identity.config_hash,
          dependency_hash=DEP, universe_name=None)
    force_legacy_strategy(repo.connection, sid)
    loaded = _loaded_for_test(_config("legacy"))

    def loader(conn, name, command):
        return loaded, SqliteStrategyRepository(conn).get(name)

    tenant = _resolve(repo, "legacy", _ForbiddenVerifier(), data_dir, loader=loader,
                      identity_loader=lambda name: identity)

    assert isinstance(tenant, WorkingTreeTenant)
    assert tenant.deployment is None
    assert tenant.runtime == _todays_path(repo, "legacy", loader, lambda name: identity,
                                          data_dir)
    assert tenant.strategy.universe == ["AAPL", "MSFT", "NVDA"]  # pre-v39 gate: CONFIG universe


def test_non_frozen_routing_keeps_the_loader_first_error_order(tmp_path):
    """The routing probe never raises: an unknown strategy still fails in the loader exactly as
    ``load_gated_strategy`` fails today, not with a registry lookup error first."""
    repo = _registry(tmp_path)

    class ModuleMissing(LookupError):
        pass

    def loader(conn, name, command):
        raise ModuleMissing(name)

    with pytest.raises(ModuleMissing):
        _resolve(repo, "ghost", _ForbiddenVerifier(), tmp_path, loader=loader)


# --- routing: the working-tree verifier never sees a frozen row; the live lane refuses it ------


def _synthetic_frozen(repo: SqliteStrategyRepository) -> int:
    """A frozen deployment with synthetic content (routing never touches content)."""
    config = _config("s")
    return _admit(repo, "s", frozen_manifest(
        code_hash="b" * 32, config_hash=config_hash(_loaded_for_test(config)),
        dependency_hash=DEP, resolved_config=_recorded(config), universe_name="liquid-us"))


def test_require_tick_deployment_refuses_frozen_without_working_tree_verification(
    tmp_path, monkeypatch,
):
    repo = _registry(tmp_path)
    sid = _synthetic_frozen(repo)
    verified: list[object] = []
    monkeypatch.setattr("algua.registry.deployment.verify_working_tree_manifest",
                        lambda manifest, repo_root: verified.append(manifest))

    with pytest.raises(FrozenLiveUnsupported) as info:
        repo.require_tick_deployment(sid)

    deployment = repo.active_deployment(sid)
    assert deployment is not None
    assert info.value.code == "frozen_live_unsupported"
    assert info.value.deployment_id == deployment.id
    assert verified == []


def test_live_resolve_tick_refuses_frozen_before_any_checkout_hashing(tmp_path, monkeypatch):
    repo = _registry(tmp_path)
    sid = _synthetic_frozen(repo)
    monkeypatch.setattr("algua.registry.deployment.verify_working_tree_manifest",
                        lambda manifest, repo_root: pytest.fail("working-tree verification"))

    with pytest.raises(FrozenLiveUnsupported) as info:
        resolve_tick(repo.connection, sid, "s", _no_identity)

    assert info.value.code == "frozen_live_unsupported"
    # Still a DeploymentError: the live lane's existing per-tenant isolation is unchanged.
    assert isinstance(info.value, DeploymentError)


def test_require_tick_deployment_still_verifies_working_tree_rows(tmp_path, monkeypatch):
    repo = _registry(tmp_path)
    identity = ArtifactIdentity("c" * 32, "f" * 32, DEP)
    sid = repo.add("w").id
    gate_id = _gate(repo, sid, code_hash=identity.code_hash, config_hash=identity.config_hash,
                    dependency_hash=DEP, universe_name="liquid-us")
    _working_tree_row(repo, sid, gate_id, identity)
    verified: list[object] = []
    monkeypatch.setattr("algua.registry.deployment.verify_working_tree_manifest",
                        lambda manifest, repo_root: verified.append(manifest))

    deployment = repo.require_tick_deployment(sid)

    assert deployment is not None
    assert verified == [deployment.manifest()]


def test_error_codes_are_stable_and_bound_to_the_deployment():
    assert FrozenContentUnavailable(deployment_id=3).code == "frozen_content_unavailable"
    assert FrozenLiveUnsupported(deployment_id=3).code == "frozen_live_unsupported"
    assert FrozenTenantUnsupported("bad", deployment_id=3).code == "frozen_content_unsupported"
    for error in (FrozenContentUnavailable(deployment_id=3), FrozenLiveUnsupported(3)):
        assert error.deployment_id == 3
        assert isinstance(error, DeploymentError)
