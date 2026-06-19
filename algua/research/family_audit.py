"""Pure cross-family gaming detector (#228) — no I/O, no DB.

Detects deliberate-split gaming: separate families that empirically behave as one thesis.
Advisory only; recommends a human-governed consolidation. See the design spec
docs/superpowers/specs/2026-06-19-anti-gaming-cross-family-detector-issue-228-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from algua.research.clustering import (
    _RETURN_CORRELATION_MIN_OVERLAP,
    MERGE_THRESHOLD,
    PARENTAGE_THRESHOLD,
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


def provenance_comparable(
    returns_a: object | None, returns_b: object | None
) -> tuple[bool, int]:
    """Conservative comparability from what stored data offers: enough shared dates AND a
    shared window covering a material fraction of BOTH series.
    Returns (comparable, overlap_days)."""
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
                status = "flagged" if return_flag else "flagged_code_factor"
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
