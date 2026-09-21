"""`dependency_hash` must cover everything that can change a number or a fill -- and nothing else.

`dependency_hash` is one third of the artifact identity, and a paper tick whose identity does not
match the strategy's CURRENT identity is dropped from forward evidence as `identity_drift`. So the
scope of this hash decides how long an evidence clock can run.

It used to be a byte hash of the whole lockfile, which meant a docs tool, a test runner or a
trailing newline reset every strategy's clock. Replaying 180 days of lockfile revisions (15
adjacent pairs): the byte hash moved on all 15, this closure on 6 -- against a gate that needs
250-500 observations under one unchanged identity.

The risk of scoping it is the opposite failure -- a package that CAN change results drifting outside
the identity. `test_every_runtime_import_is_inside_the_dependency_identity` is the defence, and it
is deliberately built to survive the three ways the first version of it was blind:

  1. it FOLLOWS first-party imports transitively (scanning `live` alone never sees what
     `algua.calendar` imports, which is how `exchange-calendars` was missed);
  2. it resolves a package directory's `__init__.py`, not just `module.py` (which is how
     `algua.data.store` -> `files.py` -> `pyarrow` was missed);
  3. it seeds whole packages that are resolved BY NAME at runtime -- `get_provider(name)` means no
     static walk can ever reach `providers/yfinance.py`, which is how `yfinance` was missed.

It still cannot see a third-party module imported through `importlib` on a computed string. That is
a known residual, recorded here rather than papered over.
"""

from __future__ import annotations

import ast
import json
import sys
import tomllib
from pathlib import Path

from algua.provenance.lockfile import COMPUTE_ROOTS, _payload

REPO = Path(__file__).resolve().parents[1]

#: Where the decision-and-execution path starts. Everything reachable from these is runtime.
_ENTRY_POINTS = (
    "algua.strategies.loader",      # loads the signal + construction the decision runs
    "algua.backtest",               # the engine, metrics and sweep (BLAS pinning lives here)
    "algua.live.live_loop",         # the live tick
    "algua.live.paper_loop",        # the paper tick
    "algua.data.serve",             # the bars a decision reads
    "algua.data.providers",         # resolved BY NAME -> seed the whole package
    "algua.calendar.factory",       # which sessions exist at all
    "algua.execution.broker_factory",  # the broker REGISTRATION seam, not one broker
    "algua.strategies",             # strategy modules are imported by COMPUTED name
    "algua.config.settings",        # parses the risk and execution settings
)

# Import name -> distribution name, where they differ.
_IMPORT_TO_DIST = {
    "yaml": "pyyaml", "dateutil": "python-dateutil",
    "exchange_calendars": "exchange-calendars", "pydantic_settings": "pydantic-settings",
    "dateparser": "dateparser",
}


def _module_path(mod: str) -> Path | None:
    base = REPO / Path(mod.replace(".", "/"))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _submodules(mod: str) -> list[str]:
    """Every module under a package directory, RECURSIVELY. A package resolved by NAME at runtime
    can reach any of them, so a static walk must treat the whole subtree as reachable -- `glob`
    stopped at the top level and would miss a provider or strategy in a nested package."""
    directory = REPO / Path(mod.replace(".", "/"))
    if not directory.is_dir():
        return []
    out = []
    for path in directory.rglob("*.py"):
        rel = path.relative_to(REPO).with_suffix("")
        parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
        out.append(".".join(parts))
    return out


def _runtime_third_party() -> dict[str, str]:
    """{distribution: an importing module}, over the whole first-party runtime closure."""
    stdlib = set(sys.stdlib_module_names)
    seen: set[str] = set()
    third: dict[str, str] = {}
    stack = list(_ENTRY_POINTS)
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        path = _module_path(mod)
        if path is None:
            continue
        stack.extend(_submodules(mod))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                if name.startswith("algua"):
                    stack.append(name)
                    continue
                top = name.split(".")[0]
                if top not in stdlib:
                    third.setdefault(_IMPORT_TO_DIST.get(top, top), mod)
    return third


def _covered() -> set[str]:
    """Distribution names inside the dependency identity, read back out of the real payload."""
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    payload = _payload(lock)
    assert payload is not None, "the repo lockfile must produce a provable closure"
    return set(json.loads(payload)["packages"])


def test_every_entry_point_still_exists():
    """The entry-point list is hand-maintained, so it can drift the same way the old package list
    did. A renamed or deleted module must FAIL here rather than silently shrink what the guard
    covers -- a guard that quietly stops looking is worse than no guard."""
    missing = [mod for mod in _ENTRY_POINTS if _module_path(mod) is None]
    assert not missing, f"entry points no longer resolve: {missing}"


def test_every_runtime_import_is_inside_the_dependency_identity():
    """THE defence against scoping this hash too narrowly.

    Derived from source, so adding a dependency to the decision or execution path without widening
    COMPUTE_ROOTS fails here. Widen the roots; do not relax this test.
    """
    covered = _covered()
    uncovered = {dist: mod for dist, mod in _runtime_third_party().items() if dist not in covered}
    assert not uncovered, (
        f"runtime imports outside the dependency identity: {uncovered}. A bump to one of these "
        f"could change a result or a fill without invalidating a prior approval. Add it to "
        f"COMPUTE_ROOTS in algua/provenance/lockfile.py."
    )


def test_the_roots_that_were_missed_the_first_time_are_covered():
    """Regression for the four the first attempt omitted: sessions, bar values, snapshot decoding
    and settings parsing all change decisions, and none of them is numpy."""
    covered = _covered()
    assert {"exchange-calendars", "yfinance", "pyarrow", "pydantic-settings"} <= covered


def test_the_numerical_stack_is_covered_transitively():
    """`vectorbt` brings numba; a numba bump absolutely can change a result. Pinning only the direct
    roots would be a wall with a hole in it."""
    assert {"numba", "llvmlite"} <= _covered()


def test_the_identity_is_narrower_than_the_whole_lockfile():
    """The point of the change. If this ever equals the lockfile again, the clock-reset problem is
    back and the scoping has been silently undone."""
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    assert len(_covered()) < len({p["name"] for p in lock["package"] if "name" in p})


def test_roots_are_all_present_in_the_lock():
    """A root that is not actually a dependency would make the hash fail closed forever."""
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    assert COMPUTE_ROOTS <= {p["name"] for p in lock["package"] if "name" in p}
