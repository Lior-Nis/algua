"""`dependency_hash` must cover everything that can change a number or a fill -- and nothing else.

`dependency_hash` is one third of the artifact identity, and a paper tick whose identity does not
match the strategy's CURRENT identity is dropped from forward evidence as `identity_drift`. So the
scope of this hash decides how long an evidence clock can run.

It used to be a byte hash of the whole lockfile, which meant a docs tool, a test runner or a
transitive HTML parser reset every strategy's clock. Measured over 180 days: 15 lockfile commits,
15 resets, against a gate that needs 250-500 observations under one unchanged identity.

The risk of scoping it is the opposite failure -- a package that CAN change results drifting
outside the identity. The first test below is the defence: it re-derives the compute path's
third-party imports from source rather than trusting the curated set.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

from algua.provenance.lockfile import COMPUTE_ROOTS, _closure

REPO = Path(__file__).resolve().parents[1]

#: Packages whose code runs between a bar arriving and an order reaching the venue.
_COMPUTE_PACKAGES = ("backtest", "features", "portfolio", "contracts", "strategies",
                     "execution", "live", "risk", "primitives")

# Import names that differ from their distribution name on PyPI.
_IMPORT_TO_DIST = {"yaml": "pyyaml", "dateutil": "python-dateutil"}


def _third_party_imports(package: str) -> set[str]:
    stdlib = set(sys.stdlib_module_names)
    found: set[str] = set()
    for path in (REPO / "algua" / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top not in stdlib and top != "algua":
                    found.add(_IMPORT_TO_DIST.get(top, top))
    return found


def test_every_compute_path_import_is_inside_the_dependency_identity():
    """THE defence against scoping this hash too narrowly.

    Derived from source, so adding a new numerical dependency to the decision or execution path
    without widening COMPUTE_ROOTS fails here. Widen the roots; do not relax this test.
    """
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    packages = {p["name"]: p for p in lock["package"] if "name" in p}
    covered = set(_closure(packages))

    uncovered: dict[str, set[str]] = {}
    for package in _COMPUTE_PACKAGES:
        missing = {d for d in _third_party_imports(package) if d not in covered}
        if missing:
            uncovered[package] = missing
    assert not uncovered, (
        f"compute-path imports outside the dependency identity: {uncovered}. A bump to one of "
        f"these could change a result or a fill without invalidating a prior approval. Add it to "
        f"COMPUTE_ROOTS in algua/provenance/lockfile.py."
    )


def test_the_numerical_stack_is_covered_transitively():
    """`vectorbt` brings numba; a numba bump absolutely can change a result. Pinning only the
    direct roots would be a wall with a hole in it."""
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    covered = set(_closure({p["name"]: p for p in lock["package"] if "name" in p}))
    assert {"numba", "llvmlite"} <= covered

def test_roots_are_all_present_in_the_lock():
    """A root that is not actually a dependency would silently make the hash fail closed forever."""
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    names = {p["name"] for p in lock["package"] if "name" in p}
    assert COMPUTE_ROOTS <= names
