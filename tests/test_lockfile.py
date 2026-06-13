"""#166 gap 5: the dependency-identity primitive (algua/provenance/lockfile.py) — the SINGLE
source of truth both the backtest reproducibility stamp and the live-approval gate hang off — had
zero dedicated tests. A uv.lock bump that can change fill/numerical semantics MUST shift this hash
for both at once; an absent lockfile MUST fail closed (no deterministic identity to pin).

Note: dependency_hash() is a pure byte-hash with no parsing, so there is no "corrupt -> parse
failure" path. The meaningful fail-closed is absent -> None; ANY content change shifts the hash,
which is exactly the identity behavior the gates rely on."""
from __future__ import annotations

import hashlib

from algua.provenance import lockfile


def test_file_hash_roundtrips_to_sha256(tmp_path):
    # Round-trip: the hash is the plain sha256 of the file bytes, stable across calls.
    path = tmp_path / "uv.lock"
    payload = b"version = 1\n[[package]]\nname = 'pandas'\nversion = '2.2.0'\n"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert lockfile._file_hash(path) == expected
    assert lockfile._file_hash(path) == expected  # deterministic on re-read


def test_file_hash_pins_version_content_change_shifts_hash(tmp_path):
    # Version pinning: a single-character version bump must change the hash (the gates would see a
    # different dependency identity), while identical bytes hash identically.
    base = b"name = 'pandas'\nversion = '2.2.0'\n"
    a = tmp_path / "a.lock"
    a.write_bytes(base)
    same = tmp_path / "same.lock"
    same.write_bytes(base)
    bumped = tmp_path / "bumped.lock"
    bumped.write_bytes(base.replace(b"2.2.0", b"2.2.1"))

    assert lockfile._file_hash(a) == lockfile._file_hash(same)   # same bytes -> same identity
    assert lockfile._file_hash(a) != lockfile._file_hash(bumped)  # a bump shifts the identity


def test_file_hash_absent_file_fails_closed(tmp_path):
    # Absent lockfile -> None: there is no deterministic identity to pin, so it must fail closed
    # rather than fabricate one.
    assert lockfile._file_hash(tmp_path / "does-not-exist.lock") is None


def test_dependency_hash_reads_repo_lockfile(monkeypatch, tmp_path):
    # dependency_hash() resolves uv.lock under the repo root (lockfile._ROOT). Point _ROOT at a
    # temp dir to assert it hashes THAT uv.lock — same source of truth for stamp and gate.
    payload = b"version = 1\n"
    (tmp_path / "uv.lock").write_bytes(payload)
    monkeypatch.setattr(lockfile, "_ROOT", tmp_path)
    assert lockfile.dependency_hash() == hashlib.sha256(payload).hexdigest()


def test_dependency_hash_absent_lockfile_is_none(monkeypatch, tmp_path):
    # No uv.lock under the root -> None (fail closed), mirroring _file_hash's contract.
    monkeypatch.setattr(lockfile, "_ROOT", tmp_path)
    assert lockfile.dependency_hash() is None


def test_repo_dependency_hash_is_present_and_stable():
    # In this repo uv.lock exists, so the real call must yield a stable, non-None 64-char sha256
    # hex digest — the value the backtest stamp and live gate actually pin.
    h = lockfile.dependency_hash()
    assert h is not None and len(h) == 64 and h == lockfile.dependency_hash()
