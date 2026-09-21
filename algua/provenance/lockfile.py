from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

# Repo root: algua/provenance/lockfile.py -> parents[2] is the project root that holds
# uv.lock. Kept here (not in backtest) so the registry can pin the same locked-dependency set
# the backtest stamps record, without the registry importing the backtest engine.
_ROOT = Path(__file__).resolve().parents[2]

#: Distribution names the DECISION AND EXECUTION path imports. The dependency identity is the
#: transitive closure of these, not the whole lockfile.
#:
#: WHY THIS IS SCOPED AT ALL. `dependency_hash` is one third of the artifact identity, and a tick
#: whose identity does not match the strategy's CURRENT identity is excluded from forward evidence
#: as `identity_drift`. Hashing the whole lockfile meant ANY dependency movement -- a docs tool, a
#: test runner, a trailing newline -- reset every strategy's evidence clock. Replaying the last 180
#: days of uv.lock revisions (16 revisions, 15 adjacent pairs): the byte hash moved on ALL 15.
#: This closure moves on 6 -- 5 if you exclude the bootstrap pair where the closure first becomes
#: attestable. The gate needs 250-500 observations under ONE unchanged identity, so the wall was
#: unpassable for reasons unrelated to any strategy's quality.
#:
#: WHAT IS IN, AND WHY EACH. Everything that can change a number or a fill:
#:   numpy/pandas/scipy/vectorbt  the numerical stack (vectorbt brings numba, hence TRANSITIVE)
#:   threadpoolctl               pins BLAS threading -> float reduction order -> results
#:   pydantic / pydantic-settings validate and parse the risk and execution settings
#:   requests                    the HTTP client the broker adapter posts orders through
#:   exchange-calendars          selects SESSIONS; a holiday-table change moves every decision
#:   pyarrow                     decodes the bar snapshots the decision reads
#:   yfinance                    the default paper provider; it produces the bar values themselves
#:
#: The last four were missed on the first attempt at this scoping, which is also why the earlier
#: claim that #657 (anyio/soupsieve) "cannot touch a strategy's numbers" was WRONG: yfinance pulls
#: beautifulsoup4, which pulls soupsieve, so that bump was reachable from ingestion after all.
#:
#: KEEPING IT HONEST. `tests/test_dependency_identity.py` re-derives this set by walking the
#: first-party import graph from the runtime entry points and collecting its third-party imports,
#: then fails if any is uncovered. Widen this set rather than editing that test.
COMPUTE_ROOTS = frozenset({
    "numpy", "pandas", "scipy", "vectorbt", "threadpoolctl",
    "pydantic", "pydantic-settings", "requests",
    "exchange-calendars", "pyarrow", "yfinance",
})


def dependency_hash() -> str | None:
    """Hash the locked identity of the compute-path dependency closure.

    The SINGLE source of truth for the dependency identity: the backtest reproducibility stamps and
    the live-approval gate both call it, so a lockfile change that can alter fill or numerical
    semantics shifts the hash for both at once. A change that provably cannot -- a package outside
    the closure of `COMPUTE_ROOTS` -- no longer does.

    THE WHOLE RECORD IS HASHED, not `name==version`. uv resolves a package more than once when
    environment markers differ (this lock carries typing-extensions at two versions, selected by
    `resolution-markers`). Hashing only name and version records BOTH variants and so looks
    conservative, but it is not: swapping the markers between them changes which version installs on
    a given interpreter while leaving the digest identical. The marker set, the source, the artifact
    hashes and the dependency edges are all part of what will actually run, so all of them are in
    the digest -- along with the lock's own `version`/`revision`/`requires-python`, since a lock
    format change reinterprets everything above.

    Fails closed to ``None`` -- which matches nothing downstream -- when the lockfile is absent,
    unparseable, structurally malformed, missing a root, missing a version, or references a
    dependency that is not in the lock. In each case we cannot PROVE what the compute path will run,
    and an identity that cannot be proved must not be asserted.
    """
    path = _ROOT / "uv.lock"
    if not path.is_file():
        return None
    try:
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
        payload = _payload(lock)
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError,
            TypeError, AttributeError, KeyError, ValueError):
        # Valid TOML can still be the wrong SHAPE. Structure we cannot read is structure we cannot
        # attest to, and this is a provenance primitive: it must answer, never raise.
        return None
    if payload is None:
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _payload(lock: object) -> str | None:
    """Canonical JSON of the closure, or None if the lock cannot be proved."""
    if not isinstance(lock, dict):
        return None
    variants = _variants(lock)
    if variants is None or not COMPUTE_ROOTS <= variants.keys():
        return None
    reachable = _reachable(variants)
    if reachable is None:
        return None
    records = {
        name: sorted(json.dumps(pkg, sort_keys=True) for pkg in variants[name])
        for name in sorted(reachable)
    }
    return json.dumps(
        {
            # A lock-format change reinterprets every field below it.
            "lock_version": lock.get("version"),
            "revision": lock.get("revision"),
            "requires_python": lock.get("requires-python"),
            "resolution_markers": lock.get("resolution-markers"),
            "packages": records,
        },
        sort_keys=True,
    )


def _variants(lock: dict) -> dict[str, list[dict]] | None:
    """name -> every locked entry for it, or None if any entry is structurally unusable.

    Validated here rather than at use: a package list we cannot read is a lock we cannot attest to,
    and skipping the unreadable parts is how a partial closure gets a confident digest.
    """
    packages = lock.get("package", [])
    if not isinstance(packages, list):
        return None
    out: dict[str, list[dict]] = {}
    for pkg in packages:
        if not isinstance(pkg, dict):
            return None
        name = pkg.get("name")
        if not isinstance(name, str) or not name:
            return None
        deps = pkg.get("dependencies", [])
        if not isinstance(deps, list):
            return None
        for dep in deps:
            # A dependency edge we cannot resolve to a name leaves the graph incomplete, and an
            # incomplete graph silently shrinks the closure. Refuse instead of skipping it.
            if not isinstance(dep, dict) or not isinstance(dep.get("name"), str):
                return None
        out.setdefault(name, []).append(pkg)
    return out


def _reachable(variants: dict[str, list[dict]]) -> set[str] | None:
    """Every package reachable from COMPUTE_ROOTS, or None if the graph does not resolve.

    Transitive on purpose: vectorbt brings numba, and a numba bump absolutely can change a result,
    so pinning only the direct roots would be a wall with a hole in it. Edges are followed across
    ALL variants of a name, because a marker-selected variant may pull something the others do not.
    """
    seen: set[str] = set()
    stack = list(COMPUTE_ROOTS)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        if name not in variants:
            # An edge pointing outside the lock means it does not describe a resolvable
            # environment; refuse rather than hash a partial closure.
            return None
        seen.add(name)
        for pkg in variants[name]:
            if not isinstance(pkg.get("version"), str) or not pkg["version"]:
                return None  # inside the closure and unversioned: unprovable
            stack.extend(dep["name"] for dep in pkg.get("dependencies", []))
    return seen
