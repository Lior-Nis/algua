import pandas as pd

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
