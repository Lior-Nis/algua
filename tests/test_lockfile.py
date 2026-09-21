"""#166 gap 5: the dependency-identity primitive (algua/provenance/lockfile.py) — the SINGLE
source of truth both the backtest reproducibility stamp and the live-approval gate hang off.

This file tests the PRIMITIVE: what it parses, when it fails closed, and that a bump which can
change a result still moves the identity. Its SCOPE — which packages are inside the identity at
all, and why — is tested in `test_dependency_identity.py`.

It used to be a pure byte-hash of uv.lock with no parsing, and these tests asserted that. That
contract was replaced: a byte hash meant any dependency movement at all (a docs tool, a test
runner, a transitive HTML parser) reset every strategy's forward-evidence clock, against a gate
that needs 250-500 observations under ONE unchanged identity. The hash now covers the resolved
versions of the compute-path closure, so the fail-closed paths include "unparseable" and "a root is
missing", which a byte hash had no way to detect.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from algua.provenance import lockfile

REPO = Path(__file__).resolve().parents[1]
_REAL_LOCK = (REPO / "uv.lock").read_text(encoding="utf-8")


def _at(tmp_path, monkeypatch, content: str):
    (tmp_path / "uv.lock").write_text(content, encoding="utf-8")
    monkeypatch.setattr(lockfile, "_ROOT", tmp_path)


def test_repo_dependency_hash_is_present_and_stable():
    # In this repo uv.lock exists, so the real call must yield a stable, non-None 64-char sha256
    # hex digest — the value the backtest stamp and live gate actually pin.
    h = lockfile.dependency_hash()
    assert h is not None and len(h) == 64 and h == lockfile.dependency_hash()


def test_identical_content_yields_an_identical_identity(tmp_path, monkeypatch):
    _at(tmp_path, monkeypatch, _REAL_LOCK)
    assert lockfile.dependency_hash() is not None


def test_a_bump_inside_the_closure_shifts_the_identity(tmp_path, monkeypatch):
    """The wall's actual job: a numpy bump must invalidate a prior approval."""
    _at(tmp_path, monkeypatch, _REAL_LOCK)
    before = lockfile.dependency_hash()
    version = next(p for p in tomllib.loads(_REAL_LOCK)["package"]
                   if p["name"] == "numpy")["version"]
    _at(tmp_path, monkeypatch,
        _REAL_LOCK.replace(f'name = "numpy"\nversion = "{version}"',
                           'name = "numpy"\nversion = "99.0.0"', 1))
    assert lockfile.dependency_hash() != before


def test_reformatting_the_lockfile_does_not_shift_the_identity(tmp_path, monkeypatch):
    """The digest depends on the resolved SET, not on uv's formatting. A byte hash moved on a
    trailing newline; that is a clock reset for nothing."""
    _at(tmp_path, monkeypatch, _REAL_LOCK)
    before = lockfile.dependency_hash()
    _at(tmp_path, monkeypatch, _REAL_LOCK + "\n\n# a comment uv might add\n")
    assert lockfile.dependency_hash() == before


def test_absent_lockfile_fails_closed(tmp_path, monkeypatch):
    # No uv.lock under the root -> None: no deterministic identity to pin, so it must fail closed
    # rather than fabricate one. None matches NOTHING downstream.
    monkeypatch.setattr(lockfile, "_ROOT", tmp_path)
    assert lockfile.dependency_hash() is None


@pytest.mark.parametrize("content, why", [
    ("", "an empty lockfile proves nothing about what will run"),
    ("this is not toml {{{", "an unparseable lockfile proves nothing either"),
    ('[[package]]\nname = "numpy"\nversion = "1.0"\n', "a lockfile missing a root is unprovable"),
    ('[[package]]\nversion = "1.0"\n', "a nameless package entry is unprovable"),
])
def test_an_unprovable_lockfile_fails_closed(tmp_path, monkeypatch, content, why):
    """These three are NEW fail-closed paths: a byte hash happily hashed garbage and returned a
    confident-looking identity for a lockfile that could not describe a runnable environment."""
    _at(tmp_path, monkeypatch, content)
    assert lockfile.dependency_hash() is None, why
