from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from algua.registry.artifact_contract import ArtifactFile, BundleDescriptor
from algua.registry.artifact_preparation import prepare_frozen_artifact
from algua.registry.artifact_recording import frozen_deployment_manifest
from algua.registry.db import connect, migrate
from algua.registry.deployment import DeploymentError
from algua.registry.environment_contract import (
    EnvironmentDescriptor,
    EnvironmentKey,
    InterpreterIdentity,
)
from algua.registry.frozen_manifest_contract import FrozenManifest
from algua.registry.frozen_source import FrozenFile
from algua.registry.repository import ArtifactIdentity
from algua.registry.store import SqliteStrategyRepository


def _frozen_manifest(*, universe_name: str | None = "liquid-us") -> FrozenManifest:
    interpreter = InterpreterIdentity(
        implementation="CPython", version="3.12.11", cache_tag="cpython-312",
        soabi="cpython-312-x86_64-linux-gnu", platform_tag="linux-x86_64",
        os_name="linux", machine="x86_64",
    )
    key = EnvironmentKey(
        build_inputs_digest="1" * 64, dependency_hash="d" * 64,
        interpreter=interpreter, uv_version="uv 0.9.26",
        create_argv=("uv", "venv"), sync_argv=("uv", "sync"),
    )
    environment = EnvironmentDescriptor(key, "2" * 64, interpreter)
    bundle = BundleDescriptor.from_files((
        ArtifactFile("algua/__init__.py", "100644", 0,
                     "e3b0c44298fc1c149afbf4c8996fb924"
                     "27ae41e4649b934ca495991b7852b855"),
    ))
    return FrozenManifest(
        source_ref="a" * 40, code_hash="b" * 32, config_hash="c" * 32,
        dependency_hash="d" * 64, resolved_config={"name": "s"},
        universe_name=universe_name, bundle=bundle, environment=environment,
    )


def _candidate(repo: SqliteStrategyRepository, manifest: FrozenManifest) -> tuple[int, int]:
    rec = repo.add("s")
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1,
        windowed_total_combos=1, funnel_window_days=90, breadth_provenance="measured",
        pit_ok=True, pit_override=False, holdout_n_bars=63,
        min_holdout_observations=63, code_hash=manifest.code_hash,
        config_hash=manifest.config_hash, dependency_hash=manifest.dependency_hash,
        data_source="test", snapshot_id="snap", period_start="2024-01-01",
        period_end="2024-12-31", holdout_frac=0.2, actor="human",
        decision_json="{}", universe_name=manifest.universe_name,
    )
    created_at = repo.connection.execute(
        "SELECT created_at FROM gate_evaluations WHERE id=?", (gate_id,),
    ).fetchone()["created_at"]
    repo.connection.execute("UPDATE strategies SET stage='candidate' WHERE id=?", (rec.id,))
    repo.connection.execute(
        "INSERT INTO stage_transitions(strategy_id, from_stage, to_stage, actor, reason,"
        " code_hash, config_hash, dependency_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (rec.id, "backtested", "candidate", "human", "fixture", manifest.code_hash,
         manifest.config_hash, manifest.dependency_hash, created_at),
    )
    repo.connection.commit()
    return rec.id, gate_id


def test_frozen_descriptor_round_trips_through_existing_ledger(tmp_path) -> None:
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    frozen = _frozen_manifest()
    strategy_id, gate_id = _candidate(repo, frozen)
    deployment = frozen_deployment_manifest(frozen)

    first = repo.record_frozen_artifact("s", deployment, research_gate_id=gate_id)
    second = repo.record_frozen_artifact("s", deployment, research_gate_id=gate_id)
    stored = repo.deployment_artifact_by_digest(frozen.digest)

    assert first == second == stored.id
    assert stored.manifest() == deployment
    assert stored.frozen_manifest() == frozen
    assert conn.execute("SELECT COUNT(*) FROM strategy_deployments").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM strategy_allocations").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM stage_transitions WHERE strategy_id=?", (strategy_id,),
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT consumed FROM gate_evaluations WHERE id=?", (gate_id,),
    ).fetchone()[0] == 0


def test_frozen_descriptor_fetch_rejects_missing_and_denormalized_corruption(tmp_path) -> None:
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    frozen = _frozen_manifest()
    _strategy_id, gate_id = _candidate(repo, frozen)
    deployment = frozen_deployment_manifest(frozen)
    artifact_id = repo.record_frozen_artifact("s", deployment, research_gate_id=gate_id)

    with pytest.raises(LookupError, match="artifact"):
        repo.deployment_artifact_by_digest("0" * 64)
    conn.execute("DROP TRIGGER trg_deployment_artifacts_no_update")
    conn.execute(
        "UPDATE deployment_artifacts SET environment_digest=? WHERE id=?",
        ("0" * 64, artifact_id),
    )
    conn.commit()
    with pytest.raises(DeploymentError, match="descriptor"):
        repo.deployment_artifact_by_digest(frozen.digest).frozen_manifest()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("stage", "candidate"),
        ("entry", "candidate episode"),
        ("gate", "newest"),
        ("universe", "universe"),
        ("anchored", "anchored"),
    ],
)
def test_record_revalidates_candidate_gate_and_identity(tmp_path, mutation, match) -> None:
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    frozen = _frozen_manifest()
    strategy_id, gate_id = _candidate(repo, frozen)
    deployment = frozen_deployment_manifest(frozen)
    if mutation == "stage":
        conn.execute("UPDATE strategies SET stage='paper' WHERE id=?", (strategy_id,))
    elif mutation == "entry":
        conn.execute(
            "UPDATE stage_transitions SET code_hash='drift' WHERE strategy_id=?"
            " AND to_stage='candidate'", (strategy_id,),
        )
    elif mutation == "gate":
        conn.execute(
            "UPDATE gate_evaluations SET code_hash='drift' WHERE id=?", (gate_id,),
        )
    elif mutation == "universe":
        deployment = frozen_deployment_manifest(replace(frozen, universe_name="other"))
    else:
        with conn:
            artifact_id = repo.resolve_deployment_artifact_locked(deployment)
            conn.execute(
                "INSERT INTO strategy_deployments(strategy_id, artifact_id, research_gate_id,"
                " activated_at) VALUES (?,?,?,?)",
                (strategy_id, artifact_id, gate_id, "2026-09-27T00:00:00+00:00"),
            )
    conn.commit()

    with pytest.raises(DeploymentError, match=match):
        repo.record_frozen_artifact("s", deployment, research_gate_id=gate_id)
    if mutation != "anchored":
        assert conn.execute("SELECT COUNT(*) FROM deployment_artifacts").fetchone()[0] == 0


def test_record_rolls_back_partial_insert(tmp_path, monkeypatch) -> None:
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    frozen = _frozen_manifest()
    _strategy_id, gate_id = _candidate(repo, frozen)
    deployment = frozen_deployment_manifest(frozen)
    original = repo.resolve_deployment_artifact_locked

    def insert_then_fail(manifest):
        original(manifest)
        raise RuntimeError("injected database boundary failure")

    monkeypatch.setattr(repo, "resolve_deployment_artifact_locked", insert_then_fail)
    with pytest.raises(RuntimeError, match="injected"):
        repo.record_frozen_artifact("s", deployment, research_gate_id=gate_id)
    assert conn.execute("SELECT COUNT(*) FROM deployment_artifacts").fetchone()[0] == 0


def test_final_revalidation_is_immediately_before_begin_immediate(tmp_path) -> None:
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    frozen = _frozen_manifest()
    _strategy_id, gate_id = _candidate(repo, frozen)
    events: list[str] = []
    conn.set_trace_callback(lambda sql: events.append(sql) if sql == "BEGIN IMMEDIATE" else None)

    def revalidate() -> None:
        assert not conn.in_transaction
        events.append("revalidate")

    repo.record_frozen_artifact(
        "s", frozen_deployment_manifest(frozen), research_gate_id=gate_id,
        pre_begin_check=revalidate,
    )
    assert events[:2] == ["revalidate", "BEGIN IMMEDIATE"]


def test_preparation_runs_slow_work_outside_transaction_and_records_only_descriptor(
    tmp_path, monkeypatch,
) -> None:
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    expected = _frozen_manifest()
    strategy_id, gate_id = _candidate(repo, expected)
    calls: list[str] = []

    def outside(name, value):
        def invoke(*_args, **_kwargs):
            assert not conn.in_transaction
            calls.append(name)
            return value
        return invoke

    source = (FrozenFile("algua/__init__.py", "100644", b""),)
    inputs = (
        FrozenFile(".python-version", "100644", b"3.12\n"),
        FrozenFile("pyproject.toml", "100644", b"[project]\nname='algua'\n"),
        FrozenFile("uv.lock", "100644", b"version = 1\npackage = []\n"),
    )
    identity = ArtifactIdentity("b" * 32, "c" * 32, "d" * 64)
    loaded = SimpleNamespace(
        model_handle=None,
        config=SimpleNamespace(model_dump=lambda **_kwargs: {"name": "s"}),
    )
    inventory = SimpleNamespace(digest="2" * 64)
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.load_strategy_config",
        outside("declared-config", SimpleNamespace(needs_model=False, model_ref=None)),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.compute_artifact_hashes",
        outside("identity", identity),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.load_tradable_strategy",
        outside("strategy", loaded),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.assert_clean_head",
        outside("head", "a" * 40),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.export_source", outside("source", source),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.export_build_inputs", outside("inputs", inputs),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.installer_version",
        outside("uv", "uv 0.9.26"),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.build_environment_key",
        outside("key", expected.environment.key),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.publish_bundle",
        outside("bundle", Path("published-bundle")),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.provision_environment",
        outside("provision", inventory),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.publish_environment",
        outside("environment", Path("published-environment")),
    )

    result = prepare_frozen_artifact(
        repo, "s", repo_root=tmp_path, store_root=tmp_path / "store",
    )

    assert result.artifact_id > 0
    assert result.research_gate_id == gate_id
    assert result.manifest.digest == repo.deployment_artifact_by_digest(
        result.manifest.digest).frozen_manifest().digest
    assert calls.count("head") == 2
    assert calls.count("identity") == 2
    assert conn.execute("SELECT COUNT(*) FROM strategy_deployments").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM strategy_allocations").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM stage_transitions WHERE strategy_id=?", (strategy_id,),
    ).fetchone()[0] == 2


def test_preparation_rejects_model_config_before_artifact_identity_or_model_bytes(
    tmp_path, monkeypatch,
) -> None:
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.assert_clean_head", lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.load_strategy_config",
        lambda _name: SimpleNamespace(needs_model=True, model_ref=object()),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.compute_artifact_hashes",
        lambda _name: (_ for _ in ()).throw(AssertionError("identity/model bytes touched")),
    )

    from algua.registry.artifact_errors import FrozenAssetsUnsupported

    with pytest.raises(FrozenAssetsUnsupported):
        prepare_frozen_artifact(
            repo, "s", repo_root=tmp_path, store_root=tmp_path / "store",
        )
    assert conn.execute("SELECT COUNT(*) FROM deployment_artifacts").fetchone()[0] == 0
    assert not (tmp_path / "store").exists()


class _Published(Exception):
    """Raised by the first publishing step, proving preparation got past the config checks."""


def _nfd_preparation(monkeypatch, tmp_path, params: dict):
    """A candidate whose loaded CONFIG carries ``params``; every step from ``export_source`` on
    (nothing of which may run for a refused config) raises ``_Published``."""
    from algua.contracts.types import ExecutionContract
    from algua.strategies.base import StrategyConfig

    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    _candidate(repo, _frozen_manifest())
    config = StrategyConfig(
        name="s", universe=["AAPL"], execution=ExecutionContract(rebalance_frequency="1d"),
        params=params, construction="top_n")
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.assert_clean_head", lambda _root: "a" * 40)
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.load_strategy_config", lambda _name: config)
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.compute_artifact_hashes",
        lambda _name: ArtifactIdentity("b" * 32, "c" * 32, "d" * 64))
    monkeypatch.setattr(
        "algua.registry.artifact_preparation.load_tradable_strategy",
        lambda _name: SimpleNamespace(model_handle=None, config=config))

    def published(*_args, **_kwargs):
        raise _Published()

    for step in ("export_source", "export_build_inputs", "publish_bundle",
                 "provision_environment", "publish_environment"):
        monkeypatch.setattr(f"algua.registry.artifact_preparation.{step}", published)
    return conn


@pytest.mark.parametrize("params", [
    {"label": "café"},        # decomposed (NFD) text value
    {"café": 1},              # decomposed (NFD) key
    {"nested": ["x", {"k": "Å"}]},
])
def test_preparation_refuses_decomposed_unicode_before_publishing(
    tmp_path, monkeypatch, params,
) -> None:
    """Canonical JSON NFC-normalises the recorded config but ``config_hash`` hashes the raw text,
    so a decomposed string records a config that can never re-hash to the descriptor's identity:
    admitted, then refused at every tick. It is refused up front as invalid source instead."""
    from algua.registry.artifact_errors import FrozenSourceInvalid

    conn = _nfd_preparation(monkeypatch, tmp_path, params)

    with pytest.raises(FrozenSourceInvalid):
        prepare_frozen_artifact(
            SqliteStrategyRepository(conn), "s", repo_root=tmp_path,
            store_root=tmp_path / "store")
    assert conn.execute("SELECT COUNT(*) FROM deployment_artifacts").fetchone()[0] == 0
    assert not (tmp_path / "store").exists()


def test_preparation_passes_composed_non_ascii_text_to_publishing(tmp_path, monkeypatch) -> None:
    """The refusal is exactly "NFC would change it": already-composed text proceeds."""
    conn = _nfd_preparation(monkeypatch, tmp_path, {"label": "café", "k": "Å"})

    with pytest.raises(_Published):
        prepare_frozen_artifact(
            SqliteStrategyRepository(conn), "s", repo_root=tmp_path,
            store_root=tmp_path / "store")
