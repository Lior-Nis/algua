# tests/test_idea_dedup.py
from algua.research.idea_dedup import (
    families_comparable,
    is_collision,
    jaccard,
    signature,
)


def test_signature_normalizes_and_is_order_independent():
    a = signature("Momentum Reversal", "Stocks that fell rebound strongly")
    b = signature("reversal momentum", "strongly rebound that fell stocks")
    assert a == b  # lowercased, stopword-stripped, sorted token set
    assert "the" not in a.split() and "that" not in a.split()  # stopwords removed


def test_jaccard_bounds():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_families_comparable_null_is_failsafe():
    assert families_comparable("mom", "mom") is True
    assert families_comparable("mom", "value") is False
    # NULL/unknown on EITHER side compares against everything (cannot suppress a collision)
    assert families_comparable(None, "value") is True
    assert families_comparable("mom", "unknown") is True
    assert families_comparable(None, None) is True


def test_is_collision_respects_family_and_threshold():
    sig = signature("low volatility anomaly", "low vol names outperform on risk-adjusted basis")
    near = signature("the low volatility anomaly", "low vol stocks outperform risk adjusted")
    far = signature("earnings drift", "prices drift after earnings surprises for weeks")
    # same family + high overlap -> collision
    assert is_collision(sig, "vol", near, "vol") is True
    # different concrete family -> not compared, no collision
    assert is_collision(sig, "vol", near, "momentum") is False
    # null family on the candidate -> compared anyway; low overlap -> no collision
    assert is_collision(sig, None, far, "drift") is False
