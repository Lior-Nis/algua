# Anti-Gaming Cross-Family Detector (#228) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only `algua research family-audit` command that detects deliberate-split gaming across families (one thesis spread over separate families to dodge breadth budget) and recommends a human-governed consolidation — mutating nothing.

**Architecture:** A pure detection core (`algua/research/family_audit.py`) runs a 5-step pipeline — `flag_edges` → (CLI fetches pairwise breadth) → `build_components` (evasion-based skip + connected components) → (CLI fetches component breadth) → `rank_clusters`. Similarity math is the single protected source in `clustering.py` (new `pairwise_axes`, with `family_similarity` refactored to call it). The CLI reads everything under one consistent snapshot and emits JSON.

**Tech Stack:** Python, typer (CLI), sqlite3 (registry store), pandas (return series), pytest.

## Global Constraints

- Quality gate (ALL must pass before any commit is considered done): `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- `algua/research/clustering.py` and `algua/registry/store.py` are CODEOWNERS-protected — changes are additive / behavior-preserving only; human review on PR.
- The audit command is READ-ONLY: no holdout peek/burn, no ledger/`family_events`/`gate_evaluations` writes, no transitions, no return-series recompute. Reads only.
- No schema change. No new protected gate constants (audit thresholds live in the unprotected `family_audit.py`, importing `clustering.py`'s floors).
- Every data command emits JSON on stdout (platform contract).
- Detection runs over ACTIVE members; breadth is LIFETIME (includes removed members) — a deliberate asymmetry, reported explicitly.
- Spec: `docs/superpowers/specs/2026-06-19-anti-gaming-cross-family-detector-issue-228-design.md`.

---

### Task 1: `pairwise_axes` primitive + `family_similarity` refactor (protected `clustering.py`)

**Files:**
- Modify: `algua/research/clustering.py`
- Test: `tests/research/test_clustering.py` (add cases; existing reference tests MUST stay green)

**Interfaces:**
- Consumes: existing `_return_correlation_axis`, `WEIGHT_*`, `MERGE_THRESHOLD`, `PARENTAGE_THRESHOLD`, `SimVerdict`.
- Produces: `pairwise_axes(code_a: str, factors_a: set[str], returns_a, code_b: str, factors_b: set[str], returns_b) -> tuple[float, dict]` returning `(blended, axes)` where `axes = {"code": float, "factor": float, "return": float | None}` (`"return"` is `None` when the axis is not evaluable). `family_similarity` keeps its existing signature and behavior.

- [ ] **Step 1: Write failing tests for `pairwise_axes` and family_similarity equivalence**

Add to `tests/research/test_clustering.py`:

```python
import math
import pandas as pd
from algua.research import clustering as C
from algua.research.clustering import pairwise_axes, family_similarity, SimVerdict


def _series(vals, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def test_pairwise_axes_code_exact_match():
    blended, axes = pairwise_axes("h1", set(), None, "h1", set(), None)
    assert axes["code"] == 1.0
    assert axes["factor"] == 0.0
    assert axes["return"] is None
    assert blended == C.WEIGHT_CODE_ANCESTRY  # 0.50


def test_pairwise_axes_factor_jaccard():
    blended, axes = pairwise_axes("", {"a", "b"}, None, "", {"b", "c"}, None)
    assert axes["code"] == 0.0
    assert axes["factor"] == 1 / 3
    assert axes["return"] is None


def test_pairwise_axes_return_axis_present_and_clamped():
    s = _series([0.01, -0.02, 0.03, 0.0, 0.01] * 20)  # 100 points
    blended, axes = pairwise_axes("", set(), s, "", set(), s)
    assert axes["return"] == 1.0  # perfect self-correlation
    assert math.isclose(blended, C.WEIGHT_RETURN_CORRELATION)  # 0.20


def test_pairwise_axes_return_axis_none_when_thin_overlap():
    s1 = _series([0.01] * 10)
    s2 = _series([0.01] * 10, start="2024-01-01")  # disjoint dates
    _blended, axes = pairwise_axes("", set(), s1, "", set(), s2)
    assert axes["return"] is None


def test_pairwise_axes_symmetric():
    s1 = _series([0.01, -0.01, 0.02, 0.0] * 20)
    s2 = _series([0.0, 0.01, -0.01, 0.02] * 20)
    a = pairwise_axes("h1", {"x"}, s1, "h2", {"y"}, s2)
    b = pairwise_axes("h2", {"y"}, s2, "h1", {"x"}, s1)
    assert a == b


def test_family_similarity_still_routes_and_matches_reference():
    # code-exact member → MERGE at 0.50 floor (code weight), no returns
    members = [{"code_hash": "h1", "factors": set(), "name": "m1"}]
    verdict, score = family_similarity("h1", set(), members, returns_lookup=None)
    assert verdict == SimVerdict.PARENTAGE  # 0.50 == PARENTAGE floor, < MERGE 0.85
    assert math.isclose(score, 0.50)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/research/test_clustering.py -k "pairwise_axes or routes" -v`
Expected: FAIL — `pairwise_axes` not defined (ImportError).

- [ ] **Step 3: Add `pairwise_axes` and refactor `family_similarity` to use it**

In `algua/research/clustering.py`, add after `_return_correlation_axis`:

```python
def pairwise_axes(
    code_a: str,
    factors_a: set[str],
    returns_a: object | None,
    code_b: str,
    factors_b: set[str],
    returns_b: object | None,
) -> tuple[float, dict]:
    """Per-axis similarity between two strategies + the weighted blend.

    Single source of the axis math (family_similarity routes through this). The "return"
    axis is None when not evaluable (< MIN_OVERLAP shared dates or a series missing);
    blended uses 0.0 for an unevaluable return axis. Mirrors family_similarity's prior
    inner loop exactly: empty hash -> 0.0, empty factor sets -> 0.0, negative corr -> 0.0.
    """
    if not code_a or not code_b:
        code_score = 0.0
    else:
        code_score = 1.0 if code_a == code_b else 0.0

    union = factors_a | factors_b
    factor_score = (len(factors_a & factors_b) / len(union)) if union else 0.0

    return_axis = _return_correlation_axis(returns_a, returns_b)
    return_score = return_axis if return_axis is not None else 0.0

    blended = (
        WEIGHT_CODE_ANCESTRY * code_score
        + WEIGHT_FACTOR_LINEAGE * factor_score
        + WEIGHT_RETURN_CORRELATION * return_score
    )
    return blended, {"code": code_score, "factor": factor_score, "return": return_axis}
```

Then replace the body of `family_similarity`'s `for member in family_members:` loop so it calls `pairwise_axes` (behavior-preserving):

```python
def family_similarity(
    strategy_code_hash: str,
    strategy_factors: set[str],
    family_members: list[dict],
    *,
    returns_lookup: dict | None = None,
) -> tuple[SimVerdict, float]:
    if not family_members:
        return (SimVerdict.NOVEL, 0.0)

    strategy_returns = returns_lookup.get("__strategy__") if returns_lookup is not None else None
    best_score = 0.0
    for member in family_members:
        member_returns = None
        if returns_lookup is not None:
            member_name = member.get("name")
            member_returns = returns_lookup.get(member_name) if member_name else None
        score, _axes = pairwise_axes(
            strategy_code_hash, strategy_factors, strategy_returns,
            member["code_hash"], member["factors"], member_returns,
        )
        if not math.isfinite(score):
            return (SimVerdict.NOVEL, 0.0)
        if score > best_score:
            best_score = score

    if not math.isfinite(best_score):
        return (SimVerdict.NOVEL, 0.0)
    if best_score >= MERGE_THRESHOLD:
        return (SimVerdict.MERGE, best_score)
    if best_score >= PARENTAGE_THRESHOLD:
        return (SimVerdict.PARENTAGE, best_score)
    return (SimVerdict.NOVEL, best_score)
```

- [ ] **Step 4: Run the full clustering test file (new + existing reference tests)**

Run: `uv run pytest tests/research/test_clustering.py -v`
Expected: PASS — new `pairwise_axes` tests AND all pre-existing `family_similarity` reference tests (byte-identical behavior).

- [ ] **Step 5: Commit**

```bash
git add algua/research/clustering.py tests/research/test_clustering.py
git commit -m "feat(228): pairwise_axes primitive; family_similarity routes through it (behavior-preserving)"
```

---

### Task 2: Read-only store accessors (protected `store.py` + `repository.py`)

**Files:**
- Modify: `algua/registry/store.py` (near `family_lifetime_combos`, ~line 1370)
- Modify: `algua/registry/repository.py` (Protocol, near line 574)
- Test: `tests/test_family_registry.py` (add cases)

**Interfaces:**
- Consumes: existing `_family_member_strategies`, `search_trials.n_combos`, `families` table.
- Produces: `lifetime_combos_for_families(family_ids: Iterable[int]) -> int` (union of ancestor-expanded members, deduped, summed once); `family_names() -> dict[int, str]`. `family_lifetime_combos(fid)` delegates to the union method.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_family_registry.py` (follow the file's existing fixture pattern for a repo + seeded families/search_trials):

```python
def test_lifetime_combos_for_families_dedups_shared_strategy(family_repo):
    # family A and B both (somehow) reference strategy "s_shared" via membership;
    # union breadth must count its trials exactly once.
    repo = family_repo
    fa = repo.create_family("fam_a", actor="human")
    fb = repo.create_family("fam_b", actor="human")
    _seed_member_with_trials(repo, fa, "s_a", n_combos=100)
    _seed_member_with_trials(repo, fb, "s_b", n_combos=200)
    # s_shared assigned to A then reassigned to B → appears in both (append-only)
    _seed_member_with_trials(repo, fa, "s_shared", n_combos=50)
    repo.assign_strategy_to_family(
        "s_shared", fb, actor="human", verdict="merge", similarity_score=0.9,
        clustering_version="v", clustering_config_json="{}", axis_json="{}")
    union = repo.lifetime_combos_for_families([fa, fb])
    assert union == 100 + 200 + 50  # s_shared counted once


def test_lifetime_combos_for_families_singleton_equals_family_lifetime_combos(family_repo):
    repo = family_repo
    fa = repo.create_family("fam_solo", actor="human")
    _seed_member_with_trials(repo, fa, "s_a", n_combos=77)
    assert repo.lifetime_combos_for_families([fa]) == repo.family_lifetime_combos(fa)


def test_family_names_returns_id_to_name(family_repo):
    repo = family_repo
    fa = repo.create_family("alpha", actor="human")
    names = repo.family_names()
    assert names[fa] == "alpha"
```

If a `_seed_member_with_trials` helper does not already exist in the test file, add it near the top:

```python
def _seed_member_with_trials(repo, family_id, strategy_name, *, n_combos):
    repo.assign_strategy_to_family(
        strategy_name, family_id, actor="human", verdict="merge", similarity_score=0.9,
        clustering_version="v", clustering_config_json="{}", axis_json="{}")
    repo._conn.execute(
        "INSERT INTO search_trials (strategy_name, n_combos, created_at) VALUES (?, ?, ?)",
        (strategy_name, n_combos, "2023-01-01T00:00:00+00:00"),
    )
    repo._conn.commit()
```

(Confirm the `search_trials` column names against the schema in `algua/registry/db.py` before running; adjust the INSERT if the real columns differ.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_family_registry.py -k "lifetime_combos_for_families or family_names" -v`
Expected: FAIL — `lifetime_combos_for_families` / `family_names` not defined.

- [ ] **Step 3: Implement the accessors**

In `algua/registry/store.py`, replace `family_lifetime_combos` and add the union + names methods:

```python
def lifetime_combos_for_families(self, family_ids: Iterable[int]) -> int:
    """Lifetime search combos across the UNION of the given families + all their
    transitive ancestors. A strategy reachable via several of the families is counted
    exactly once (the union of member-strategy sets is deduped before the sum)."""
    all_strategies: set[str] = set()
    for fid in family_ids:
        all_strategies.update(self._family_member_strategies(fid))
    if not all_strategies:
        return 0
    placeholders = ",".join("?" * len(all_strategies))
    row = self._conn.execute(
        f"SELECT COALESCE(SUM(st.n_combos), 0) FROM search_trials st"
        f" WHERE st.strategy_name IN ({placeholders})",
        list(all_strategies),
    ).fetchone()
    return int(row[0])

def family_lifetime_combos(self, family_id: int) -> int:
    """Lifetime search combos across this family + all transitive ancestors."""
    return self.lifetime_combos_for_families([family_id])

def family_names(self) -> dict[int, str]:
    """All family ids → names (read-only)."""
    rows = self._conn.execute("SELECT id, name FROM families").fetchall()
    return {int(r[0]): r[1] for r in rows}
```

Ensure `from collections.abc import Iterable` is imported at the top of `store.py` (add if missing).

In `algua/registry/repository.py`, add to the Protocol near `family_lifetime_combos`:

```python
def lifetime_combos_for_families(self, family_ids: Iterable[int]) -> int:
    """Lifetime combos across the union of families + transitive ancestors (deduped)."""
    ...

def family_names(self) -> dict[int, str]:
    """All family ids → names."""
    ...
```

(Add `from collections.abc import Iterable` to `repository.py` if missing.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_family_registry.py -k "lifetime_combos or family_names" -v`
Expected: PASS (including the pre-existing `family_lifetime_combos` tests, now via delegation).

- [ ] **Step 5: Commit**

```bash
git add algua/registry/store.py algua/registry/repository.py tests/test_family_registry.py
git commit -m "feat(228): read-only lifetime_combos_for_families (union dedup) + family_names accessors"
```

---

### Task 3: Pure detector — `flag_edges` (`family_audit.py`)

**Files:**
- Create: `algua/research/family_audit.py`
- Test: `tests/research/test_family_audit.py`

**Interfaces:**
- Consumes: `clustering.pairwise_axes`, `clustering.MERGE_THRESHOLD`, `clustering.PARENTAGE_THRESHOLD`, `clustering._RETURN_CORRELATION_MIN_OVERLAP`.
- Produces:
  - `AUDIT_FLAG_THRESHOLD = PARENTAGE_THRESHOLD` (0.50); `RETURN_INDEPENDENT_THRESHOLD = MERGE_THRESHOLD` (0.85); `_MATERIAL_OVERLAP_FRACTION = 0.5`.
  - `@dataclass(frozen=True) class Edge` with fields: `family_a:int, family_b:int, audit_score:float, blended:float, axes:dict, status:str, tier:str, return_overlap_days:int, provenance_comparable:bool, representative_pair:tuple[str,str], flagged:bool`.
  - `provenance_comparable(returns_a, returns_b) -> tuple[bool, int]` → `(comparable, overlap_days)`.
  - `flag_edges(profiles: dict[int, list[dict]], returns: dict[str, object]) -> list[Edge]`.

- [ ] **Step 1: Write failing tests**

Create `tests/research/test_family_audit.py`:

```python
import pandas as pd
from algua.research import family_audit as FA
from algua.research.family_audit import flag_edges


def _series(vals, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def _prof(name, code="", factors=None):
    return {"code_hash": code, "factors": factors or set(), "name": name}


def test_return_authoritative_flag_independent_of_blend():
    # near-duplicate by RETURNS only (no code/factor overlap) → flagged via the 0.85 return path,
    # even though the blend would be only ~0.18.
    rng = _series([0.01, -0.02, 0.015, 0.0, 0.005, -0.01] * 40)  # 240 pts, comparable provenance
    profiles = {1: [_prof("a", code="ha")], 2: [_prof("b", code="hb")]}
    returns = {"a": rng, "b": rng}  # perfect correlation
    edges = flag_edges(profiles, returns)
    e = next(x for x in edges if {x.family_a, x.family_b} == {1, 2})
    assert e.flagged is True
    assert e.status == "flagged"
    assert e.axes["return"] == 1.0
    assert e.blended < 0.5  # blend alone would NOT flag


def test_no_flag_when_dissimilar():
    profiles = {1: [_prof("a", code="ha", factors={"x"})], 2: [_prof("b", code="hb", factors={"y"})]}
    edges = flag_edges(profiles, returns={})
    assert all(not e.flagged for e in edges) or edges == []


def test_code_factor_flag_when_returns_missing():
    # exact code match, no returns → flagged_code_factor (blend hits 0.50 floor on code alone)
    profiles = {1: [_prof("a", code="same")], 2: [_prof("b", code="same")]}
    edges = flag_edges(profiles, returns={})
    e = next(x for x in edges if {x.family_a, x.family_b} == {1, 2})
    assert e.flagged is True
    assert e.status == "flagged_code_factor"


def test_inconclusive_when_returns_high_but_provenance_not_comparable():
    # high return corr but only a thin tail overlaps (not material) → inconclusive, not flagged
    a = _series([0.01, -0.01, 0.02] * 40)                  # Jan–...
    b = _series([0.01, -0.01, 0.02] * 40, start="2023-12-01")  # mostly disjoint window
    profiles = {1: [_prof("a", code="ha")], 2: [_prof("b", code="hb")]}
    edges = flag_edges(profiles, {"a": a, "b": b})
    matches = [x for x in edges if {x.family_a, x.family_b} == {1, 2}]
    # either dropped (truly dissimilar overlap) or surfaced as inconclusive — never silently "flagged"
    assert all(x.status != "flagged" for x in matches)


def test_max_linkage_picks_hottest_pair():
    rng = _series([0.01, -0.02, 0.015, 0.0] * 60)  # 240 pts
    # family 1 has a diverse member + a hidden clone of family 2's member
    profiles = {
        1: [_prof("div", code="hd", factors={"z"}), _prof("clone", code="hc")],
        2: [_prof("target", code="ht")],
    }
    returns = {"div": _series([0.0, 0.0, 0.0, 0.0] * 60), "clone": rng, "target": rng}
    edges = flag_edges(profiles, returns)
    e = next(x for x in edges if {x.family_a, x.family_b} == {1, 2})
    assert e.flagged is True
    assert e.representative_pair in (("clone", "target"), ("target", "clone"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/research/test_family_audit.py -v`
Expected: FAIL — module `algua.research.family_audit` does not exist.

- [ ] **Step 3: Implement `flag_edges`**

Create `algua/research/family_audit.py`:

```python
"""Pure cross-family gaming detector (#228) — no I/O, no DB.

Detects deliberate-split gaming: separate families that empirically behave as one thesis.
Advisory only; recommends a human-governed consolidation. See the design spec
docs/superpowers/specs/2026-06-19-anti-gaming-cross-family-detector-issue-228-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from algua.research.clustering import (
    MERGE_THRESHOLD,
    PARENTAGE_THRESHOLD,
    _RETURN_CORRELATION_MIN_OVERLAP,
    pairwise_axes,
)

AUDIT_FLAG_THRESHOLD = PARENTAGE_THRESHOLD          # 0.50 — multi-axis blend floor
RETURN_INDEPENDENT_THRESHOLD = MERGE_THRESHOLD      # 0.85 — return-only flag floor (beta-safe)
_MATERIAL_OVERLAP_FRACTION = 0.5                    # shared window must cover this fraction of each


@dataclass(frozen=True)
class Edge:
    family_a: int
    family_b: int
    audit_score: float
    blended: float
    axes: dict
    status: str          # "flagged" | "flagged_code_factor" | "inconclusive"
    tier: str            # "merge" | "parentage" | ""
    return_overlap_days: int
    provenance_comparable: bool
    representative_pair: tuple[str, str]
    flagged: bool


def provenance_comparable(returns_a: object | None, returns_b: object | None) -> tuple[bool, int]:
    """Conservative comparability from what stored data offers: enough shared dates AND a
    shared window covering a material fraction of BOTH series. Returns (comparable, overlap_days)."""
    if returns_a is None or returns_b is None:
        return (False, 0)
    shared = returns_a.index.intersection(returns_b.index)  # type: ignore[attr-defined]
    days = len(shared)
    if days < _RETURN_CORRELATION_MIN_OVERLAP:
        return (False, days)
    material = (days >= _MATERIAL_OVERLAP_FRACTION * len(returns_a)  # type: ignore[arg-type]
               and days >= _MATERIAL_OVERLAP_FRACTION * len(returns_b))  # type: ignore[arg-type]
    return (bool(material), days)


def _tier(score: float) -> str:
    if score >= MERGE_THRESHOLD:
        return "merge"
    if score >= AUDIT_FLAG_THRESHOLD:
        return "parentage"
    return ""


def _best_pair(members_a: list[dict], members_b: list[dict], returns: dict) -> Edge | None:
    """Max-linkage: the cross-family strategy pair with the strongest flagging signal."""
    best: Edge | None = None
    for ma in members_a:
        ra = returns.get(ma.get("name"))
        for mb in members_b:
            rb = returns.get(mb.get("name"))
            blended, axes = pairwise_axes(
                ma["code_hash"], ma["factors"], ra, mb["code_hash"], mb["factors"], rb)
            ret = axes["return"]
            comparable, overlap = provenance_comparable(ra, rb)
            return_flag = ret is not None and comparable and ret >= RETURN_INDEPENDENT_THRESHOLD
            blend_flag = blended >= AUDIT_FLAG_THRESHOLD
            flagged = bool(return_flag or blend_flag)
            audit_score = max(blended, ret if (ret is not None and comparable) else 0.0)

            if flagged:
                status = "flagged" if (return_flag or (ret is not None and comparable and ret > 0.0)) \
                    else "flagged_code_factor"
            elif ret is not None and ret >= RETURN_INDEPENDENT_THRESHOLD and not comparable:
                status = "inconclusive"
            else:
                status = ""  # nothing worth surfacing

            if not status:
                continue
            cand = Edge(
                family_a=0, family_b=0, audit_score=audit_score, blended=blended, axes=axes,
                status=status, tier=_tier(audit_score) if flagged else "",
                return_overlap_days=overlap, provenance_comparable=comparable,
                representative_pair=(ma["name"], mb["name"]), flagged=flagged)
            # rank candidates: flagged beats inconclusive, then by audit_score
            key = (1 if cand.flagged else 0, cand.audit_score)
            if best is None or key > (1 if best.flagged else 0, best.audit_score):
                best = cand
    return best


def flag_edges(profiles: dict[int, list[dict]], returns: dict[str, object]) -> list[Edge]:
    """One Edge per family pair that is flagged or inconclusive (dissimilar pairs dropped)."""
    edges: list[Edge] = []
    for fa, fb in combinations(sorted(profiles), 2):
        best = _best_pair(profiles[fa], profiles[fb], returns)
        if best is None:
            continue
        edges.append(Edge(
            family_a=fa, family_b=fb, audit_score=best.audit_score, blended=best.blended,
            axes=best.axes, status=best.status, tier=best.tier,
            return_overlap_days=best.return_overlap_days,
            provenance_comparable=best.provenance_comparable,
            representative_pair=best.representative_pair, flagged=best.flagged))
    return edges
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/research/test_family_audit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/research/family_audit.py tests/research/test_family_audit.py
git commit -m "feat(228): pure flag_edges — return-authoritative max-linkage + status taxonomy"
```

---

### Task 4: Pure detector — `build_components` + `rank_clusters` (`family_audit.py`)

**Files:**
- Modify: `algua/research/family_audit.py`
- Test: `tests/research/test_family_audit.py` (add cases)

**Interfaces:**
- Consumes: `Edge` (Task 3).
- Produces:
  - `build_components(candidate_edges: list[Edge], *, pair_breadth: dict[frozenset[int], int], individual_breadth: dict[int, int]) -> tuple[list[frozenset[int]], list[Edge]]` → `(components, kept_flagged_edges)` after the evasion-based skip.
  - `rank_clusters(components, kept_edges, candidate_edges, *, component_breadth: dict[frozenset[int], int], individual_breadth: dict[int, int], family_names: dict[int, str], active_counts: dict[int, int]) -> list[dict]` → ranked cluster dicts (the `clusters` array of the output JSON).

- [ ] **Step 1: Write failing tests**

Add to `tests/research/test_family_audit.py`:

```python
from algua.research.family_audit import build_components, rank_clusters, Edge


def _edge(a, b, score=0.9, flagged=True, status="flagged"):
    return Edge(a, b, audit_score=score, blended=score, axes={"code": 0, "factor": 0, "return": score},
                status=status, tier="merge", return_overlap_days=200, provenance_comparable=True,
                representative_pair=(f"s{a}", f"s{b}"), flagged=flagged)


def test_zero_evasion_pair_is_skipped():
    edges = [_edge(1, 2)]
    pair_breadth = {frozenset({1, 2}): 500}      # == max individual → no evasion
    individual = {1: 500, 2: 300}
    components, kept = build_components(edges, pair_breadth=pair_breadth, individual_breadth=individual)
    assert kept == []
    assert components == []


def test_sibling_split_stays_in_scope():
    edges = [_edge(1, 2)]
    pair_breadth = {frozenset({1, 2}): 800}      # > max(500, 300) → real evasion
    individual = {1: 500, 2: 300}
    components, kept = build_components(edges, pair_breadth=pair_breadth, individual_breadth=individual)
    assert len(kept) == 1
    assert components == [frozenset({1, 2})]


def test_connected_components_transitive():
    edges = [_edge(1, 2), _edge(2, 3)]
    pair_breadth = {frozenset({1, 2}): 999, frozenset({2, 3}): 999}
    individual = {1: 10, 2: 10, 3: 10}
    components, _kept = build_components(edges, pair_breadth=pair_breadth, individual_breadth=individual)
    assert components == [frozenset({1, 2, 3})]


def test_rank_clusters_orders_by_breadth_delta_and_builds_remediation():
    components = [frozenset({1, 2}), frozenset({3, 4})]
    kept = [_edge(1, 2), _edge(3, 4)]
    component_breadth = {frozenset({1, 2}): 800, frozenset({3, 4}): 1000}
    individual = {1: 500, 2: 300, 3: 950, 4: 100}
    names = {1: "a", 2: "b", 3: "c", 4: "d"}
    active = {1: 2, 2: 1, 3: 3, 4: 1}
    clusters = rank_clusters(components, kept, kept, component_breadth=component_breadth,
                             individual_breadth=individual, family_names=names, active_counts=active)
    # cluster {1,2}: delta 800-500=300 ; cluster {3,4}: delta 1000-950=50 → {1,2} ranks first
    assert clusters[0]["family_breadth_delta"] == 300
    assert clusters[1]["family_breadth_delta"] == 50
    assert clusters[0]["consolidation_target_family_id"] == 1  # highest individual breadth
    assert "recommended_remediation" in clusters[0]
    assert len(clusters[0]["flagged_edges"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/research/test_family_audit.py -k "evasion or sibling or components or rank_clusters" -v`
Expected: FAIL — `build_components` / `rank_clusters` not defined.

- [ ] **Step 3: Implement**

Append to `algua/research/family_audit.py`:

```python
def _connected_components(edges: list[Edge]) -> list[frozenset[int]]:
    """Union-find over flagged edges; deterministic (sorted) output."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for e in edges:
        union(e.family_a, e.family_b)
    groups: dict[int, set[int]] = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return sorted((frozenset(g) for g in groups.values()), key=lambda s: min(s))


def build_components(
    candidate_edges: list[Edge],
    *,
    pair_breadth: dict[frozenset[int], int],
    individual_breadth: dict[int, int],
) -> tuple[list[frozenset[int]], list[Edge]]:
    """Apply the evasion-based skip to flagged edges, then group survivors into components.

    Skip a pair iff unifying it adds no breadth to the larger family
    (breadth_of({A,B}) == max(breadth(A), breadth(B))) — the only zero-evasion case.
    """
    kept: list[Edge] = []
    for e in candidate_edges:
        if not e.flagged:
            continue
        union_breadth = pair_breadth[frozenset({e.family_a, e.family_b})]
        if union_breadth == max(individual_breadth[e.family_a], individual_breadth[e.family_b]):
            continue
        kept.append(e)
    return _connected_components(kept), kept


def _edge_json(e: Edge) -> dict:
    return {
        "family_a": e.family_a, "family_b": e.family_b,
        "audit_score": round(e.audit_score, 4), "tier": e.tier, "status": e.status,
        "axis_breakdown": e.axes, "return_overlap_days": e.return_overlap_days,
        "provenance_comparable": e.provenance_comparable,
        "representative_pair": {"strategy_a": e.representative_pair[0],
                               "strategy_b": e.representative_pair[1]},
    }


def rank_clusters(
    components: list[frozenset[int]],
    kept_edges: list[Edge],
    candidate_edges: list[Edge],
    *,
    component_breadth: dict[frozenset[int], int],
    individual_breadth: dict[int, int],
    family_names: dict[int, str],
    active_counts: dict[int, int],
) -> list[dict]:
    clusters: list[dict] = []
    for comp in components:
        unified = component_breadth[comp]
        max_indiv = max(individual_breadth[f] for f in comp)
        delta = unified - max_indiv
        # consolidation target: highest individual breadth, tie → lowest id
        target = sorted(comp, key=lambda f: (-individual_breadth[f], f))[0]
        flagged = [_edge_json(e) for e in kept_edges
                   if {e.family_a, e.family_b} <= set(comp)]
        inconclusive = [_edge_json(e) for e in candidate_edges
                        if e.status == "inconclusive" and {e.family_a, e.family_b} <= set(comp)]
        ids = sorted(comp)
        clusters.append({
            "families": [{"id": f, "name": family_names.get(f, str(f)),
                          "lifetime_combos": individual_breadth[f],
                          "active_member_count": active_counts.get(f, 0)} for f in ids],
            "unified_breadth": unified,
            "max_individual_breadth": max_indiv,
            "family_breadth_delta": delta,
            "flagged_edges": flagged,
            "inconclusive_edges": inconclusive,
            "consolidation_target_family_id": target,
            "recommended_remediation": (
                f"human review: consolidate families {ids} into family {target} by reassigning "
                f"members (assign_strategy_to_family, --actor human) so future promotions face the "
                f"pooled lifetime breadth. NOTE: add_parent_edge is directional and does not "
                f"symmetrically unify breadth."),
        })
    clusters.sort(key=lambda c: (-c["family_breadth_delta"], c["families"][0]["id"]))
    return clusters
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/research/test_family_audit.py -v`
Expected: PASS (all Task 3 + Task 4 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/research/family_audit.py tests/research/test_family_audit.py
git commit -m "feat(228): evasion-based skip + connected components + family_breadth_delta ranking"
```

---

### Task 5: CLI command `research family-audit` + docs

**Files:**
- Modify: `algua/cli/research_cmd.py`
- Modify: `CLAUDE.md` (command surface)
- Test: `tests/cli/test_research_family_audit.py` (new; follow existing CLI test patterns, e.g. `tests/cli/`)

**Interfaces:**
- Consumes: `family_audit.flag_edges/build_components/rank_clusters`; store accessors `all_families_with_member_profiles`, `family_ancestry`, `family_lifetime_combos`, `lifetime_combos_for_families`, `family_names`, `load_backtest_returns`; CLI helpers `registry_conn`, `emit`, `ok`, `json_errors`.
- Produces: `algua research family-audit` command emitting the documented JSON.

- [ ] **Step 1: Write a failing CLI smoke test**

Create `tests/cli/test_research_family_audit.py`:

```python
import json
from typer.testing import CliRunner
from algua.cli.app import app

runner = CliRunner()


def test_family_audit_empty_returns_clean_json(tmp_registry):  # tmp_registry: existing fixture for an isolated registry DB
    result = runner.invoke(app, ["research", "family-audit"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    data = payload["data"] if "data" in payload else payload
    assert data["clusters"] == []
    assert "n_families_scanned" in data
    assert "wall_time_seconds" in data
```

(Match the envelope shape `emit(ok(...))` produces — inspect another `research` command test for the exact key, e.g. `ok`/`data`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_research_family_audit.py -v`
Expected: FAIL — no `family-audit` command registered.

- [ ] **Step 3: Implement the command**

Add to `algua/cli/research_cmd.py` (new imports at top: `import time`; `from itertools import combinations`; `from algua.research import family_audit`):

```python
@research_app.command("family-audit")
@json_errors(ValueError, LookupError, sqlite3.OperationalError)
def family_audit_cmd() -> None:
    """ADVISORY cross-family gaming detector. Scans the family DAG for separate families that
    empirically behave as one thesis (deliberate-split breadth evasion that #222's assignment-time
    clustering can't see), ranks them by family-term breadth dodged, and recommends a human-governed
    consolidation. READ-ONLY: no holdout, no ledger writes, no transitions, no graph mutation."""
    started = time.monotonic()
    with registry_conn() as conn:
        # One connection = one consistent read snapshot for the whole scan (no writes).
        repo = SqliteStrategyRepository(conn)
        profile_list = repo.all_families_with_member_profiles()
        profiles = {fid: members for fid, members in profile_list}
        family_names = repo.family_names()

        # Batch-load each distinct strategy's returns ONCE (O(M), not O(M^2)).
        all_names = {m["name"] for members in profiles.values() for m in members}
        returns = {name: repo.load_backtest_returns(name) for name in all_names}
        returns = {k: v for k, v in returns.items() if v is not None}

        # Pipeline step 1: similarity-only candidate edges.
        candidate_edges = family_audit.flag_edges(profiles, returns)

        # Step 2 (CLI I/O): pairwise union breadth for candidate pairs only + per-family breadth.
        individual_breadth = {fid: repo.family_lifetime_combos(fid) for fid in profiles}
        pair_breadth = {
            frozenset({e.family_a, e.family_b}):
                repo.lifetime_combos_for_families([e.family_a, e.family_b])
            for e in candidate_edges
        }

        # Step 3: evasion skip + components.
        components, kept = family_audit.build_components(
            candidate_edges, pair_breadth=pair_breadth, individual_breadth=individual_breadth)

        # Step 4 (CLI I/O): unified breadth per component.
        component_breadth = {
            comp: repo.lifetime_combos_for_families(list(comp)) for comp in components}

        # Step 5: rank + assemble.
        active_counts = {fid: len(members) for fid, members in profiles.items()}
        clusters = family_audit.rank_clusters(
            components, kept, candidate_edges, component_breadth=component_breadth,
            individual_breadth=individual_breadth, family_names=family_names,
            active_counts=active_counts)

    n_pairs = len(list(combinations(profiles, 2)))
    emit(ok({
        "note": ("ADVISORY cross-family gaming screen; NOT a gate. Flags separate families that "
                 "behave as one thesis (return-correlation authoritative). Acting on a cluster is a "
                 "human judgement: consolidate via member reassignment (--actor human). Mutates "
                 "nothing."),
        "clusters": clusters,
        "n_families_scanned": len(profiles),
        "n_pairs_evaluated": n_pairs,
        "n_pairs_flagged_or_inconclusive": len(candidate_edges),
        "n_pairs_skipped_zero_evasion": len([e for e in candidate_edges if e.flagged]) - len(kept),
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "config": {
            "audit_flag_threshold": family_audit.AUDIT_FLAG_THRESHOLD,
            "return_independent_threshold": family_audit.RETURN_INDEPENDENT_THRESHOLD,
            "return_correlation_min_overlap": family_audit._RETURN_CORRELATION_MIN_OVERLAP,
        },
    }))
```

- [ ] **Step 4: Run the CLI test + confirm read-only**

Run: `uv run pytest tests/cli/test_research_family_audit.py -v`
Expected: PASS.

- [ ] **Step 5: Add the command-surface doc line**

In `CLAUDE.md`, under "## Command surface", after the `research dormant-sweep` entry, add:

```markdown
- `uv run algua research family-audit` — ADVISORY cross-family gaming detector. Scans the family DAG
  for separate families that empirically behave as one thesis (deliberate-split breadth evasion #222's
  assignment-time clustering can't see), using return-correlation as the authoritative axis; ranks
  flagged clusters by family-term breadth dodged and recommends a human-governed consolidation
  (member reassignment). READ-ONLY: no holdout, no ledger writes, no transitions, no graph mutation —
  a prioritization signal, not a gate.
```

- [ ] **Step 6: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add algua/cli/research_cmd.py CLAUDE.md tests/cli/test_research_family_audit.py
git commit -m "feat(228): research family-audit CLI — advisory cross-family gaming detector"
```

---

## Self-Review

**Spec coverage:**
- §1 pipeline (flag_edges → CLI pair_breadth → build_components → CLI component_breadth → rank_clusters) → Tasks 3,4,5 ✓
- §2 audit_score / return-authoritative / pairwise_axes refactor → Tasks 1,3 ✓
- §3 per-edge report + remediation → Task 4 (`flagged_edges`, `recommended_remediation`, `consolidation_target_family_id`) ✓
- §4 evasion-based skip → Task 4 `build_components` ✓
- §5 family_breadth_delta + `lifetime_combos_for_families` → Tasks 2,4 ✓
- §6 coverage status taxonomy + provenance + active/lifetime asymmetry → Tasks 3 (`status`, `provenance_comparable`), 4/5 (`active_member_count`) ✓
- §7 output JSON shape → Task 5 ✓
- §8 read-only + single snapshot + telemetry → Task 5 ✓

**Placeholder scan:** no TBD/TODO; every code step shows full code. One field-verification note (search_trials column names in Task 2 Step 1) is explicit, not a placeholder.

**Type consistency:** `Edge` fields are identical across Tasks 3–4; `flag_edges`/`build_components`/`rank_clusters` signatures match between their definitions (Tasks 3,4) and the CLI caller (Task 5); `lifetime_combos_for_families`/`family_names` match between store (Task 2) and CLI (Task 5).

## Execution Handoff

Two execution options — see below.
