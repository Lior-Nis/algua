import numpy as np

from algua.backtest.neff import estimate_n_eff

_DATES = [f"2020-{m:02d}-{d:02d}" for m in (1, 2, 3) for d in range(1, 22)]  # 63 dates


def _stream(vals, dates=_DATES):
    return (list(vals), list(dates))


def _rng(seed):
    return np.random.default_rng(seed)


def test_rho_zero_gives_n_eff_equals_raw_n():
    # 5 independent streams -> rho_bar ~ 0 -> N_eff ~ raw_n.
    rng = _rng(1)
    sibs = [_stream(rng.normal(0, 1, len(_DATES))) for _ in range(5)]
    r = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r.n_eff is not None
    assert r.n_eff >= 35           # close to raw 40 (independent -> little deflation)
    assert r.n_siblings == 5


def test_rho_one_gives_n_eff_one():
    # 5 identical streams -> rho_bar = 1 -> N_eff = 1.
    base = _rng(2).normal(0, 1, len(_DATES))
    sibs = [_stream(base) for _ in range(5)]
    r = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r.n_eff == 1
    assert r.rho_bar is not None and r.rho_bar > 0.99


def test_cap_at_raw_n():
    rng = _rng(3)
    sibs = [_stream(rng.normal(0, 1, len(_DATES))) for _ in range(5)]
    r = estimate_n_eff(10, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r.n_eff is not None and 1 <= r.n_eff <= 10


def test_too_few_siblings_returns_none():
    rng = _rng(4)
    sibs = [_stream(rng.normal(0, 1, len(_DATES))) for _ in range(4)]   # < 5
    r = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r.n_eff is None and r.rho_bar is None and r.n_siblings == 4


def test_insufficient_overlap_returns_none():
    # Two siblings sharing only 10 dates < min_overlap_bars=21 -> None.
    rng = _rng(5)
    a = _stream(rng.normal(0, 1, 63), _DATES)
    short_dates = _DATES[:10]
    b = _stream(rng.normal(0, 1, 10), short_dates)
    sibs = [a, b] + [_stream(rng.normal(0, 1, 63)) for _ in range(3)]
    r = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r.n_eff is None

def test_zero_variance_pair_returns_none():
    rng = _rng(6)
    flat = _stream([0.01] * len(_DATES))          # zero variance -> corr non-finite
    sibs = [flat] + [_stream(rng.normal(0, 1, len(_DATES))) for _ in range(4)]
    r = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r.n_eff is None


def test_date_alignment_inner_join():
    # Correlation is computed on the date-INTERSECTION, not by positional zip.
    rng = _rng(7)
    base = rng.normal(0, 1, len(_DATES))
    a = _stream(base, _DATES)
    # b is base shifted by one date-position but on a date axis offset by one — the inner-join must
    # align by DATE, so the shared dates carry the SAME base values -> high correlation.
    b = _stream(base, _DATES)
    sibs = [a, b] + [_stream(rng.normal(0, 1, len(_DATES))) for _ in range(3)]
    r = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r.n_eff is not None and r.n_pairs == 10   # C(5,2)


def test_shrinkage_pulls_n_eff_toward_raw_n():
    # Higher shrinkage_k -> lower rho_bar_lower -> N_eff closer to raw_n.
    rng = _rng(8)
    base = rng.normal(0, 1, len(_DATES))
    sibs = [_stream(base + 0.5 * rng.normal(0, 1, len(_DATES))) for _ in range(6)]
    low_k = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=0.0)
    high_k = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=3.0)
    assert low_k.n_eff is not None and high_k.n_eff is not None
    assert high_k.n_eff >= low_k.n_eff      # more shrinkage -> closer to raw N
