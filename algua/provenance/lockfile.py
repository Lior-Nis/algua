from __future__ import annotations

import hashlib
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
#: days of uv.lock revisions: 13 of 15 consecutive pairs moved the byte hash, against a gate that
#: needs 250-500 observations under ONE unchanged identity. Under this closure, 5. The wall was
#: unpassable for reasons unrelated to any strategy's quality.
#: (Both counts are transitions between consecutive revisions, not distinct digests -- an earlier
#: draft of this comment reported states as resets and understated the remaining churn.)
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
    """Hash the locked versions of the compute-path dependency closure.

    The SINGLE source of truth for the dependency identity: the backtest reproducibility stamps and
    the live-approval gate both call it, so a lockfile bump that can change fill or numerical
    semantics shifts the hash for both at once. A bump that provably cannot -- a package outside the
    closure of `COMPUTE_ROOTS` -- no longer does.

    EVERY RESOLUTION VARIANT IS HASHED, not one per name. uv resolves a package more than once when
    environment markers differ (this lock carries typing-extensions at two versions), so collapsing
    by name would record whichever entry happened to be last and could describe a different
    environment than the one that runs. Hashing all variants is conservative in the safe direction:
    it can only ever notice MORE change, never less, and needs no marker evaluation.

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
        variants = _variants(lock)
        if variants is None or not COMPUTE_ROOTS <= variants.keys():
            return None
        payload = _closure_payload(variants)
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError,
            TypeError, AttributeError, KeyError):
        # Valid TOML can still be the wrong SHAPE (a string where a table is expected, a package
        # list that is a table). Structure we cannot read is structure we cannot attest to.
        return None
    if payload is None:
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _variants(lock: dict) -> dict[str, list[dict]] | None:
    """name -> every locked entry for it. None if any entry is unusable (no name, no version)."""
    out: dict[str, list[dict]] = {}
    for pkg in lock.get("package", []):
        name, version = pkg.get("name"), pkg.get("version")
        if not isinstance(name, str) or not name:
            return None
        if not isinstance(version, str) or not version:
            # A sourceless/versionless entry (the project package itself carries no version in some
            # layouts) cannot be attested. Only fail if it is one we would hash -- checked by the
            # caller via the closure -- so record it and let _closure_payload decide.
            out.setdefault(name, []).append({"name": name, "version": None,
                                             "dependencies": pkg.get("dependencies", [])})
            continue
        out.setdefault(name, []).append(pkg)
    return out


def _closure_payload(variants: dict[str, list[dict]]) -> str | None:
    """Sorted ``name==version`` for every variant reachable from COMPUTE_ROOTS.

    Transitive on purpose: vectorbt brings numba, and a numba bump absolutely can change a result,
    so pinning only the direct roots would be a wall with a hole in it.
    """
    seen: set[str] = set()
    stack = list(COMPUTE_ROOTS)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        if name not in variants:
            # A dependency edge pointing outside the lock means the graph does not describe a
            # resolvable environment; refuse rather than hash a partial closure.
            return None
        seen.add(name)
        for pkg in variants[name]:
            stack.extend(d["name"] for d in pkg.get("dependencies", []) if "name" in d)
    lines = []
    for name in sorted(seen):
        for pkg in variants[name]:
            if pkg.get("version") is None:
                return None  # inside the closure and unversioned: unprovable
            lines.append(f"{name}=={pkg['version']}")
    return "\n".join(sorted(lines))
