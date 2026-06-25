import pandas as pd

from algua.research.family_audit import Edge, build_components, flag_edges, rank_clusters


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
    profiles = {
        1: [_prof("a", code="ha", factors={"x"})],
        2: [_prof("b", code="hb", factors={"y"})],
    }
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
    # either dropped (truly dissimilar overlap) or surfaced as inconclusive
    # — never silently "flagged"
    assert all(x.status != "flagged" for x in matches)


def test_code_match_with_low_return_is_flagged_code_factor():
    # identical code hash (blend flags via code=1.0) but only weak return corr (< 0.85)
    # over a comparable window → status must be "flagged_code_factor", NOT "flagged".
    a = _series([0.01, -0.02, 0.015, 0.0, 0.005, -0.01] * 40)   # 240 pts
    b = _series([-0.01, 0.02, -0.015, 0.0, -0.005, 0.01] * 40)  # anti-ish/low corr, same dates
    profiles = {1: [_prof("a", code="same")], 2: [_prof("b", code="same")]}
    edges = flag_edges(profiles, {"a": a, "b": b})
    e = next(x for x in edges if {x.family_a, x.family_b} == {1, 2})
    assert e.flagged is True
    assert e.status == "flagged_code_factor"
    assert e.axes["return"] is not None and e.axes["return"] < 0.85


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


# --- Task 4 tests: build_components and rank_clusters ---

def _edge(a, b, score=0.9, flagged=True, status="flagged"):
    return Edge(
        a, b,
        audit_score=score, blended=score,
        axes={"code": 0, "factor": 0, "return": score},
        status=status, tier="merge", return_overlap_days=200,
        provenance_comparable=True,
        representative_pair=(f"s{a}", f"s{b}"), flagged=flagged,
    )


def test_zero_evasion_pair_is_skipped():
    edges = [_edge(1, 2)]
    pair_breadth = {frozenset({1, 2}): 500}  # == max individual → no evasion
    individual = {1: 500, 2: 300}
    components, kept = build_components(
        edges, pair_breadth=pair_breadth, individual_breadth=individual
    )
    assert kept == []
    assert components == []


def test_sibling_split_stays_in_scope():
    edges = [_edge(1, 2)]
    pair_breadth = {frozenset({1, 2}): 800}  # > max(500, 300) → real evasion
    individual = {1: 500, 2: 300}
    components, kept = build_components(
        edges, pair_breadth=pair_breadth, individual_breadth=individual
    )
    assert len(kept) == 1
    assert components == [frozenset({1, 2})]


def test_connected_components_transitive():
    edges = [_edge(1, 2), _edge(2, 3)]
    pair_breadth = {frozenset({1, 2}): 999, frozenset({2, 3}): 999}
    individual = {1: 10, 2: 10, 3: 10}
    components, _kept = build_components(
        edges, pair_breadth=pair_breadth, individual_breadth=individual
    )
    assert components == [frozenset({1, 2, 3})]


def test_rank_clusters_orders_by_breadth_delta_and_builds_remediation():
    components = [frozenset({1, 2}), frozenset({3, 4})]
    kept = [_edge(1, 2), _edge(3, 4)]
    component_breadth = {frozenset({1, 2}): 800, frozenset({3, 4}): 1000}
    individual = {1: 500, 2: 300, 3: 950, 4: 100}
    names = {1: "a", 2: "b", 3: "c", 4: "d"}
    active = {1: 2, 2: 1, 3: 3, 4: 1}
    clusters = rank_clusters(
        components, kept, kept,
        component_breadth=component_breadth,
        individual_breadth=individual, family_names=names, active_counts=active,
    )
    # cluster {1,2}: delta 800-500=300 ; cluster {3,4}: delta 1000-950=50 → {1,2} ranks first
    assert clusters[0]["family_breadth_delta"] == 300
    assert clusters[1]["family_breadth_delta"] == 50
    assert clusters[0]["consolidation_target_family_id"] == 1  # highest individual breadth
    assert "recommended_remediation" in clusters[0]
    assert len(clusters[0]["flagged_edges"]) == 1
