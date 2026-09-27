from __future__ import annotations

from algua.registry.artifact_recording import frozen_deployment_manifest
from algua.registry.artifact_verification import verify_frozen_artifact
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from tests.test_frozen_artifact_ledger import _candidate, _frozen_manifest


def test_verify_uses_only_ledger_and_trusted_store(tmp_path, monkeypatch) -> None:
    conn = connect(tmp_path / "registry.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    frozen = _frozen_manifest()
    _strategy_id, gate_id = _candidate(repo, frozen)
    artifact_id = repo.record_frozen_artifact(
        "s", frozen_deployment_manifest(frozen), research_gate_id=gate_id,
    )
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "algua.registry.artifact_verification.verify_bundle",
        lambda root, descriptor: calls.append(("bundle", (root, descriptor))),
    )
    monkeypatch.setattr(
        "algua.registry.artifact_verification.verify_published_environment",
        lambda root, descriptor: calls.append(("environment", (root, descriptor))),
    )

    result = verify_frozen_artifact(repo, frozen.digest, store_root=tmp_path / "store")

    assert result.strategy == "s"
    assert result.artifact_id == artifact_id
    assert result.manifest == frozen
    assert [name for name, _args in calls] == ["bundle", "environment"]
    assert conn.execute("SELECT COUNT(*) FROM deployment_artifacts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM strategy_deployments").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM strategy_allocations").fetchone()[0] == 0
    assert conn.execute(
        "SELECT consumed FROM gate_evaluations WHERE id=?", (gate_id,),
    ).fetchone()[0] == 0
