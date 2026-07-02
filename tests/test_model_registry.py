"""Model-artifact registry (#376): immutability, torn-write recovery, tamper/history-rewrite
detection, and fail-closed reads."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from algua.contracts.model_types import compute_provenance_digest
from algua.models import (
    ModelRegistryError,
    get_version,
    list_versions,
    load_artifact_bytes,
    register,
)


def _register(root: Path, name: str = "m", data: bytes = b"artifact-v1", **over):
    kw = dict(
        training_snapshot_id="snap-2019",
        training_as_of="2020-01-01",
        code_hash="deadbeef",
        hyperparameters={"a": 1},
        seed=3,
        eval_report={"ic": 0.02},
    )
    kw.update(over)
    return register(name, data, root=root, **kw)


def test_register_assigns_monotonic_versions_and_roundtrips(tmp_path):
    v1 = _register(tmp_path, data=b"one")
    v2 = _register(tmp_path, data=b"two")
    assert (v1.version, v2.version) == (1, 2)
    assert load_artifact_bytes("m", 1, root=tmp_path) == b"one"
    assert load_artifact_bytes("m", 2, root=tmp_path) == b"two"
    assert [v.version for v in list_versions("m", root=tmp_path)] == [1, 2]


def test_get_version_carries_full_provenance(tmp_path):
    v = _register(tmp_path, data=b"xyz")
    got = get_version("m", 1, root=tmp_path)
    assert got.training_snapshot_id == "snap-2019"
    assert got.training_as_of == "2020-01-01"
    assert got.hyperparameters == {"a": 1}
    assert got.seed == 3
    assert got.eval_report == {"ic": 0.02}
    assert got.provenance_digest == v.provenance_digest


def test_empty_artifact_fails_closed(tmp_path):
    with pytest.raises(ModelRegistryError):
        _register(tmp_path, data=b"")


def test_missing_version_fails_closed(tmp_path):
    _register(tmp_path, data=b"one")
    with pytest.raises(ModelRegistryError):
        get_version("m", 99, root=tmp_path)


def test_tampered_artifact_bytes_detected(tmp_path):
    _register(tmp_path, data=b"original")
    artifact = tmp_path / "m" / "v1" / "artifact.bin"
    artifact.write_bytes(b"tampered!")  # same version, different bytes
    with pytest.raises(ModelRegistryError, match="digest"):
        get_version("m", 1, root=tmp_path)
    with pytest.raises(ModelRegistryError):
        load_artifact_bytes("m", 1, root=tmp_path)


def test_rewritten_manifest_metadata_detected(tmp_path):
    """Editing a provenance field in the manifest (keeping the artifact) is caught because the
    recomputed provenance_digest no longer matches the stored one."""
    _register(tmp_path, data=b"original")
    manifest = tmp_path / "m" / "manifest.jsonl"
    row = json.loads(manifest.read_text().splitlines()[0])
    row["training_as_of"] = "2099-01-01"  # rewrite history, keep the same artifact + digest
    manifest.write_text(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(ModelRegistryError, match="provenance"):
        get_version("m", 1, root=tmp_path)


def test_duplicate_version_rows_fail_closed(tmp_path):
    _register(tmp_path, data=b"original")
    manifest = tmp_path / "m" / "manifest.jsonl"
    line = manifest.read_text().splitlines()[0]
    manifest.write_text(line + "\n" + line + "\n")  # duplicate v1 row
    with pytest.raises(ModelRegistryError, match="duplicate"):
        get_version("m", 1, root=tmp_path)


def test_missing_artifact_but_row_present_fails_closed(tmp_path):
    _register(tmp_path, data=b"original")
    (tmp_path / "m" / "v1" / "artifact.bin").unlink()
    with pytest.raises(ModelRegistryError, match="missing"):
        get_version("m", 1, root=tmp_path)


def test_torn_write_dir_without_row_is_not_reused_and_not_served(tmp_path):
    """A crash between the artifact rename and the manifest append leaves a dangling vN/ with no
    row. It must (a) never be reused as a version number, and (b) fail closed on read."""
    _register(tmp_path, data=b"one")  # v1 (valid)
    # Simulate a torn v2: dir exists, no manifest row.
    torn = tmp_path / "m" / "v2"
    torn.mkdir()
    (torn / "artifact.bin").write_bytes(b"torn")
    # A read of the dangling version fails closed (no manifest row).
    with pytest.raises(ModelRegistryError):
        get_version("m", 2, root=tmp_path)
    # The next registration must NOT reuse v2 (its number is reserved by the dir listing).
    v_next = _register(tmp_path, data=b"three")
    assert v_next.version == 3
    assert not (tmp_path / "m" / "v2" / "artifact.bin").read_bytes() == b"three"
    assert list_versions("m", root=tmp_path)[-1].version == 3  # v2 (torn) excluded from valid list


def test_provenance_digest_is_deterministic():
    a = compute_provenance_digest(
        digest="d", training_snapshot_id="s", training_as_of="2020-01-01",
        code_hash="c", hyperparameters={"x": 1}, seed=1, eval_report={"m": 2},
    )
    b = compute_provenance_digest(
        digest="d", training_snapshot_id="s", training_as_of="2020-01-01",
        code_hash="c", hyperparameters={"x": 1}, seed=1, eval_report={"m": 2},
    )
    assert a == b
    c = compute_provenance_digest(
        digest="d", training_snapshot_id="s", training_as_of="2020-01-01",
        code_hash="c", hyperparameters={"x": 2}, seed=1, eval_report={"m": 2},
    )
    assert a != c


def test_corrupt_manifest_line_fails_closed(tmp_path):
    _register(tmp_path, data=b"one")
    manifest = tmp_path / "m" / "manifest.jsonl"
    manifest.write_text(manifest.read_text() + "{not json\n")
    with pytest.raises(ModelRegistryError, match="corrupt manifest"):
        list_versions("m", root=tmp_path)


def test_list_versions_fails_closed_on_tampered_artifact(tmp_path):
    _register(tmp_path, data=b"one")
    _register(tmp_path, data=b"two")
    (tmp_path / "m" / "v2" / "artifact.bin").write_bytes(b"tampered")
    with pytest.raises(ModelRegistryError):
        list_versions("m", root=tmp_path)


def test_invalid_name_fails_closed(tmp_path):
    with pytest.raises(ModelRegistryError):
        register("../escape", b"x", training_snapshot_id=None, training_as_of="2020-01-01",
                 code_hash=None, root=tmp_path)
