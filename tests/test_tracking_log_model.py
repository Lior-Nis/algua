"""log_model (#376): the model registry is authoritative (registration propagates failures), and a
best-effort MLflow mirror failure does not lose the durable registration."""
from __future__ import annotations

import pytest

from algua.models import ModelRegistryError, get_version
from algua.tracking.mlflow_tracker import log_model


def test_log_model_registers_and_returns_version(tmp_path):
    root = tmp_path / "models"
    v = log_model(
        "m", b"artifact",
        tracking_uri=str(tmp_path / "mlruns"),
        training_snapshot_id="snap", training_as_of="2020-01-01", code_hash="c",
        hyperparameters={"a": 1}, seed=2, eval_report={"ic": 0.01}, root=root,
    )
    assert v.version == 1
    # The registry is authoritative: the version is durably present regardless of MLflow.
    got = get_version("m", 1, root=root)
    assert got.digest == v.digest
    assert got.provenance_digest == v.provenance_digest


def test_log_model_survives_mlflow_mirror_failure(tmp_path, monkeypatch):
    """If the MLflow mirror raises AFTER a successful registration, log_model swallows it and still
    returns the registered version (registered-but-not-mirrored is allowed; the reverse is not)."""
    root = tmp_path / "models"
    import algua.tracking.mlflow_tracker as tracker

    def _boom(*a, **k):
        raise RuntimeError("mlflow down")

    # _run is the first mlflow touch inside the mirror block.
    monkeypatch.setattr(tracker, "_run", _boom)
    v = log_model(
        "m", b"artifact",
        tracking_uri=str(tmp_path / "mlruns"),
        training_snapshot_id="snap", training_as_of="2020-01-01", code_hash="c", root=root,
    )
    assert v.version == 1
    assert get_version("m", 1, root=root).digest == v.digest


def test_log_model_empty_artifact_propagates(tmp_path):
    with pytest.raises(ModelRegistryError):
        log_model(
            "m", b"",
            tracking_uri=str(tmp_path / "mlruns"),
            training_snapshot_id=None, training_as_of="2020-01-01", code_hash=None,
            root=tmp_path / "models",
        )
