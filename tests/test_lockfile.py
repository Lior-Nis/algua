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
    first = lockfile.dependency_hash()
    _at(tmp_path, monkeypatch, _REAL_LOCK)
    assert first is not None and lockfile.dependency_hash() == first


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


def _bump(lock: str, name: str, to: str = "99.0.0") -> str:
    """Rewrite one package's version in the lockfile text."""
    version = next(p["version"] for p in tomllib.loads(lock)["package"] if p["name"] == name)
    return lock.replace(f'name = "{name}"\nversion = "{version}"',
                        f'name = "{name}"\nversion = "{to}"', 1)


def test_a_bump_OUTSIDE_the_closure_does_not_shift_the_identity(tmp_path, monkeypatch):
    """THE point of the change, and the property nothing else tested.

    `ruff` is a linter. It cannot alter a number or a fill, so bumping it must not reset every
    strategy's forward-evidence clock -- which is exactly what the old byte hash did.
    """
    _at(tmp_path, monkeypatch, _REAL_LOCK)
    before = lockfile.dependency_hash()
    _at(tmp_path, monkeypatch, _bump(_REAL_LOCK, "ruff"))
    assert lockfile.dependency_hash() == before


def test_a_TRANSITIVE_bump_inside_the_closure_does_shift_it(tmp_path, monkeypatch):
    """numba is reached only through vectorbt, and a numba bump absolutely can change a result.
    Pinning only the direct roots would be a wall with a hole in it."""
    _at(tmp_path, monkeypatch, _REAL_LOCK)
    before = lockfile.dependency_hash()
    _at(tmp_path, monkeypatch, _bump(_REAL_LOCK, "numba"))
    assert lockfile.dependency_hash() != before


@pytest.mark.parametrize("root", ["exchange-calendars", "yfinance", "pyarrow", "pydantic-settings"])
def test_a_bump_to_a_late_added_root_shifts_the_identity(tmp_path, monkeypatch, root):
    """The four the first attempt at this scoping missed. Sessions, bar values, snapshot decoding
    and settings parsing all change decisions, and none of them is numpy."""
    _at(tmp_path, monkeypatch, _REAL_LOCK)
    before = lockfile.dependency_hash()
    _at(tmp_path, monkeypatch, _bump(_REAL_LOCK, root))
    assert lockfile.dependency_hash() != before


def test_every_resolution_variant_is_hashed_not_just_one_per_name(tmp_path, monkeypatch):
    """uv resolves a package more than once when markers differ -- this lock carries
    typing-extensions at two versions. Collapsing by name would record whichever entry came last
    and could describe a different environment than the one that actually runs.

    Bumping the FIRST of the two variants must move the hash; under a name-collapsing
    implementation it would not.
    """
    parsed = tomllib.loads(_REAL_LOCK)
    versions = [p["version"] for p in parsed["package"] if p["name"] == "typing-extensions"]
    assert len(versions) > 1, "this test needs a genuinely duplicated package to be meaningful"
    _at(tmp_path, monkeypatch, _REAL_LOCK)
    before = lockfile.dependency_hash()
    _at(tmp_path, monkeypatch,
        _REAL_LOCK.replace(f'name = "typing-extensions"\nversion = "{versions[0]}"',
                           'name = "typing-extensions"\nversion = "99.0.0"', 1))
    assert lockfile.dependency_hash() != before


def test_a_dependency_edge_pointing_outside_the_lock_fails_closed(tmp_path, monkeypatch):
    """A graph that names a package the lock does not contain does not describe a resolvable
    environment. Hashing a partial closure would assert an identity we cannot prove."""
    broken = _REAL_LOCK.replace('name = "numpy"\n', 'name = "numpy"\n', 1)
    parsed = tomllib.loads(broken)
    scipy = next(p for p in parsed["package"] if p["name"] == "scipy")
    assert scipy.get("dependencies"), "this test needs scipy to have a dependency edge"
    _at(tmp_path, monkeypatch,
        broken.replace('[[package]]\nname = "numpy"', '[[package]]\nname = "numpy-gone"', 1))
    assert lockfile.dependency_hash() is None


def test_a_versionless_package_inside_the_closure_fails_closed(tmp_path, monkeypatch):
    """`name==` is not an identity. A confident-looking digest over an unversioned closure member
    is worse than no digest."""
    version = next(p["version"] for p in tomllib.loads(_REAL_LOCK)["package"]
                   if p["name"] == "numpy")
    _at(tmp_path, monkeypatch,
        _REAL_LOCK.replace(f'name = "numpy"\nversion = "{version}"', 'name = "numpy"', 1))
    assert lockfile.dependency_hash() is None


@pytest.mark.parametrize("content", [
    'package = "not a list"',
    '[[package]]\nname = 42\nversion = "1.0"',
    '[[package]]\nname = "numpy"\nversion = "1.0"\ndependencies = "not a list"',
])
def test_structurally_malformed_but_valid_toml_fails_closed(tmp_path, monkeypatch, content):
    """Valid TOML can still be the wrong SHAPE. Structure we cannot read is structure we cannot
    attest to -- it must not raise out of a provenance primitive either."""
    _at(tmp_path, monkeypatch, content)
    assert lockfile.dependency_hash() is None
