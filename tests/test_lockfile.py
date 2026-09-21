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


# --- mutating the REAL lock -------------------------------------------------------------------
#
# Every mutation below starts from the real lockfile and changes ONE thing. A hand-built fixture is
# how the first version of these tests passed for the wrong reason: a stub containing only numpy
# fails at the root-presence check, long before it reaches the behaviour the test is named after --
# which concealed a real fail-open bug in the production code.


def _mutate(fn):
    """The real lock as a dict, with `fn` applied."""
    lock = tomllib.loads(_REAL_LOCK)
    fn(lock)
    return lock


def _pkg(lock: dict, name: str) -> dict:
    return next(p for p in lock["package"] if p["name"] == name)


def _payload_hash(lock: dict) -> str | None:
    payload = lockfile._payload(lock)
    return None if payload is None else payload


def test_a_bump_OUTSIDE_the_closure_does_not_shift_the_identity():
    """THE point of the change, and the property nothing else tested.

    `ruff` is a linter. It cannot alter a number or a fill, so bumping it must not reset every
    strategy's forward-evidence clock -- which is exactly what the old byte hash did.
    """
    before = _payload_hash(tomllib.loads(_REAL_LOCK))
    after = _payload_hash(_mutate(lambda lk: _pkg(lk, "ruff").__setitem__("version", "99.0.0")))
    assert before is not None and after == before


def test_a_TRANSITIVE_bump_inside_the_closure_does_shift_it():
    """numba is reached only through vectorbt, and a numba bump absolutely can change a result."""
    before = _payload_hash(tomllib.loads(_REAL_LOCK))
    after = _payload_hash(_mutate(lambda lk: _pkg(lk, "numba").__setitem__("version", "99.0.0")))
    assert before is not None and after is not None, "a bump must not FREEZE the fleet"
    assert after != before


@pytest.mark.parametrize("root", ["exchange-calendars", "yfinance", "pyarrow", "pydantic-settings"])
def test_a_bump_to_a_late_added_root_shifts_the_identity(root):
    """The four the first attempt at this scoping missed. Sessions, bar values, snapshot decoding
    and settings parsing all change decisions, and none of them is numpy."""
    before = _payload_hash(tomllib.loads(_REAL_LOCK))
    after = _payload_hash(_mutate(lambda lk: _pkg(lk, root).__setitem__("version", "99.0.0")))
    assert before is not None and after is not None
    assert after != before


def test_swapping_marker_sets_between_two_variants_shifts_the_identity():
    """The hole in hashing `name==version` alone.

    uv resolves typing-extensions twice, selected by `resolution-markers`. Both versions appear in
    the lock either way, so a name==version digest is IDENTICAL after swapping the marker sets --
    while a different version now installs on a given interpreter. Hashing only name and version
    looks conservative and is not.
    """
    def swap(lock):
        a, b = [p for p in lock["package"] if p["name"] == "typing-extensions"]
        a["resolution-markers"], b["resolution-markers"] = (
            b["resolution-markers"], a["resolution-markers"])

    before = _payload_hash(tomllib.loads(_REAL_LOCK))
    after = _payload_hash(_mutate(swap))
    assert before is not None and after is not None
    assert after != before


def test_changing_a_package_source_shifts_the_identity():
    """Same version from a different index is not the same artifact."""
    def repoint(lock):
        _pkg(lock, "numpy")["source"] = {"registry": "https://evil.example/simple"}

    before = _payload_hash(tomllib.loads(_REAL_LOCK))
    assert _payload_hash(_mutate(repoint)) not in (None, before)


def test_a_lock_format_change_shifts_the_identity():
    """A new lock `version` reinterprets every field beneath it."""
    before = _payload_hash(tomllib.loads(_REAL_LOCK))
    assert _payload_hash(_mutate(lambda lk: lk.__setitem__("version", 99))) != before


# --- fail-closed paths, each reached for the RIGHT reason --------------------------------------

def test_a_dangling_TRANSITIVE_edge_fails_closed():
    """Renaming a ROOT would return None at the root-presence check without ever reaching this
    branch -- which is how the first version of this test passed for the wrong reason. numba is
    inside the closure but is NOT a root, so only the dangling-edge path can trip."""
    assert "numba" not in lockfile.COMPUTE_ROOTS
    mutated = _mutate(lambda lk: _pkg(lk, "numba").__setitem__("name", "numba-gone"))
    assert _payload_hash(mutated) is None


def test_a_reachable_dependency_edge_without_a_name_fails_closed():
    """An edge we cannot resolve leaves the graph incomplete, and an incomplete graph silently
    shrinks the closure. Applied to the REAL lock, so every root is still present and only this
    condition can cause the None."""
    def bad_edge(lock):
        _pkg(lock, "scipy").setdefault("dependencies", []).append({"version": "999"})

    assert _payload_hash(_mutate(bad_edge)) is None


def test_a_non_list_dependencies_field_fails_closed():
    def not_a_list(lock):
        _pkg(lock, "scipy")["dependencies"] = "not a list"

    assert _payload_hash(_mutate(not_a_list)) is None


def test_a_versionless_package_inside_the_closure_fails_closed():
    """`name==` is not an identity. A confident-looking digest over an unversioned closure member
    is worse than no digest."""
    assert _payload_hash(_mutate(lambda lk: _pkg(lk, "numpy").pop("version"))) is None


def test_a_versionless_package_OUTSIDE_the_closure_is_tolerated():
    """The project's own entry carries no version in some layouts. Failing closed on a package the
    compute path cannot reach would freeze the fleet for nothing."""
    assert _payload_hash(_mutate(lambda lk: _pkg(lk, "ruff").pop("version"))) is not None


@pytest.mark.parametrize("lock", [
    {"package": "not a list"},
    {"package": [{"name": 42, "version": "1.0"}]},
    {"package": [{"version": "1.0"}]},
    "not a mapping at all",
])
def test_structurally_malformed_locks_fail_closed(lock):
    """Valid TOML can still be the wrong SHAPE. A provenance primitive must answer, never raise."""
    assert lockfile._payload(lock) is None
