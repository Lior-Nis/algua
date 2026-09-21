from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

# Repo root: algua/provenance/lockfile.py -> parents[2] is the project root that holds
# uv.lock. Kept here (not in backtest) so the registry can pin the same locked-dependency set
# the backtest stamps record, without the registry importing the backtest engine.
_ROOT = Path(__file__).resolve().parents[2]

#: Third-party packages the DECISION AND EXECUTION path imports directly. The dependency identity
#: is the transitive closure of these, not the whole lockfile.
#:
#: WHY THIS IS SCOPED AT ALL. `dependency_hash` is one third of the artifact identity, and a tick
#: whose identity does not match the strategy's CURRENT identity is excluded from forward evidence
#: as `identity_drift`. Hashing the whole lockfile therefore meant that ANY dependency movement --
#: a docs tool, a test runner, a transitive HTML parser -- reset every strategy's evidence clock to
#: zero. Measured over 180 days: 15 lockfile commits, 15 resets. Under this closure: 4. The other
#: 11 were packages no strategy can reach, and one of them (a security bump of an HTTP and an HTML
#: library) is what exposed this.
#:
#: That mattered because the forward gate needs roughly 250-500 observations under ONE unchanged
#: identity -- one to two years of daily ticks -- while the lockfile moved every ~11 days. The wall
#: was unpassable for a reason unrelated to whether a strategy was any good.
#:
#: WHAT IT STILL CATCHES. Everything that can change a number or a fill: the numerical stack
#: (numpy/pandas/scipy/vectorbt and their closure, including numba), validation (pydantic), BLAS
#: thread pinning (threadpoolctl -- it changes float reduction order, so it changes results), and
#: the HTTP client the broker adapter posts orders through (requests).
#:
#: KEEPING IT HONEST. `tests/test_dependency_identity.py` re-derives the third-party imports of the
#: compute-path packages by AST and fails if any is not covered here, so a new numerical dependency
#: cannot silently fall outside the identity. Widen this set rather than editing that test.
COMPUTE_ROOTS = frozenset({
    "numpy", "pandas", "scipy", "vectorbt", "pydantic", "requests", "threadpoolctl",
})


def dependency_hash() -> str | None:
    """Hash the locked versions of the compute-path dependency closure.

    This is the SINGLE source of truth for the dependency identity: the backtest reproducibility
    stamps and the live-approval gate both call it, so a lockfile bump that can change fill or
    numerical semantics shifts the hash for both at once. A bump that provably cannot -- a package
    outside the closure of `COMPUTE_ROOTS` -- no longer does.

    Fails closed to ``None`` (no deterministic identity, which matches nothing) when the lockfile is
    absent, unparseable, or does not contain every root: in each case we cannot PROVE what the
    compute path will run, and an identity that cannot be proved must not be asserted.
    """
    path = _ROOT / "uv.lock"
    if not path.is_file():
        return None
    try:
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError):
        return None
    packages = {p["name"]: p for p in lock.get("package", []) if "name" in p}
    if not COMPUTE_ROOTS <= packages.keys():
        return None  # a root vanished from the lock: cannot prove the closure
    closure = _closure(packages)
    # name==version lines, sorted, so the digest depends on the resolved SET and nothing else --
    # not on uv's field order, formatting, or where a package sits in the file.
    payload = "\n".join(f"{name}=={packages[name].get('version', '')}" for name in closure)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _closure(packages: dict[str, dict]) -> list[str]:
    """Every package reachable from COMPUTE_ROOTS through `dependencies`, sorted.

    Transitive on purpose: `vectorbt` brings numba, and a numba bump absolutely can change a
    result, so pinning only the direct roots would be a wall with a hole in it.
    """
    seen: set[str] = set()
    stack = list(COMPUTE_ROOTS)
    while stack:
        name = stack.pop()
        if name in seen or name not in packages:
            continue
        seen.add(name)
        stack.extend(d["name"] for d in packages[name].get("dependencies", []) if "name" in d)
    return sorted(seen)
