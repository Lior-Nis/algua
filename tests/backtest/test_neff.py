import math

import numpy as np

from algua.backtest.neff import _pair_correlation, estimate_n_eff

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
    # 5 identical streams -> rho_bar ≈ 1 -> N_eff is minimal (1 or 2).
    # np.corrcoef(x, x) returns 0.9999... (1 ULP below 1.0) due to floating-point arithmetic,
    # so n_eff = 40/(1+39*0.9999...) ≈ 1+eps → ceil = 2.  This is the CORRECT conservative
    # result: with ceil rounding, near-perfectly-correlated siblings give N_eff ≤ 2, not 1.
    base = _rng(2).normal(0, 1, len(_DATES))
    sibs = [_stream(base) for _ in range(5)]
    r = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r.n_eff is not None and r.n_eff <= 2   # ceil(1+fp_epsilon) = 2; ceil is conservative
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


def test_date_alignment_inner_join_discriminates_positional_zip():
    """Date-keyed alignment: streams whose DATE AXES are misaligned positionally but share
    dates carrying matching values must yield high correlation (corr ≈ 1).

    Setup: values V are iid normal; stream a uses D[:-1] (first 62 dates) with values V[:-1];
    stream b uses D[1:] (last 62 dates) with values V[1:].

    The date inner-join paired at each SHARED date gives the SAME value (corr ≈ 1 → N_eff ≈ 1).
    A positional-zip implementation would pair V[0] with V[1], V[1] with V[2], … (corr ≈ autocorr
    of iid ≈ 0 → N_eff ≈ raw_n), so assertion that n_eff == 1 fails under zip.

    We use _pair_correlation directly (focused unit test) and also via estimate_n_eff with enough
    independent fillers to isolate the diagnostic pair.
    """
    rng = _rng(42)
    n = len(_DATES)          # 63
    vals = rng.normal(0, 1, n).tolist()

    # a and b share dates D[1:-1] (61 dates, >>21 overlap) carrying the same value at each date.
    a = (vals[:-1], _DATES[:-1])    # dates D[0]..D[61], values vals[0]..vals[61]
    b = (vals[1:],  _DATES[1:])     # dates D[1]..D[62], values vals[1]..vals[62]
    # At shared dates D[1]..D[61] (61 dates), a maps D[i] -> vals[i] and b maps D[i] -> vals[i].
    # So date-inner-join correlation = corr(vals[1..61], vals[1..61]) = 1.0 exactly.

    rho = _pair_correlation(a, b, min_overlap_bars=21)
    assert rho is not None, "inner-join should produce a finite correlation"
    assert rho > 0.99, f"date-keyed alignment must yield corr ≈ 1, got {rho}"

    # Also exercise via estimate_n_eff with just the two crafted streams and min_siblings=2.
    # The single pair has corr ≈ 1 → rho_bar ≈ 1 → N_eff is minimal (ceil of ~1+fp_eps = 2).
    # A positional-zip impl would compute corr(vals[0..61], vals[1..62]) ≈ 0 → N_eff ≈ raw_n=40.
    # Discriminating threshold: n_eff ≤ 2 passes date-inner-join, fails positional-zip.
    r = estimate_n_eff(40, [a, b], min_siblings=2, min_overlap_bars=21, shrinkage_k=0.0)
    assert r.n_eff is not None
    assert r.n_eff <= 2, (
        f"date-keyed alignment with identical values should yield N_eff≤2; "
        f"got {r.n_eff} (positional-zip would give ~40)"
    )


def test_n_eff_rounds_up_conservative():
    """N_eff is rounded UP (ceil) — the conservative direction.

    A lower N_eff would lower the DSR benchmark SR* (more lenient), so rounding up is
    the anti-lenient choice. We construct a case where the continuous N_eff lands between
    two integers and assert that the returned n_eff equals math.ceil, not math.floor.

    We also assert n_eff >= round(continuous) when round rounds down, i.e. that ceil is not
    less than nearest-integer rounding.
    """
    # Use 5 identical streams: rho_bar = 1.0 -> N_eff = raw_n / raw_n = 1 (exact, no fractional).
    # Instead pick a moderate positive rho to get a fractional continuous N_eff.
    # Build streams with a known target rho_bar: use a 1-factor model.
    # x_i = sqrt(rho)*F + sqrt(1-rho)*eps_i with F,eps iid N(0,1) -> E[corr(x_i,x_j)] = rho.
    rng = _rng(99)
    n = len(_DATES)
    raw_n = 10
    # Target rho = 0.5 -> N_eff = 10/5.5 ≈ 1.818 -> ceil=2, round=2 (not discriminating).
    # Target rho ≈ 0.41: N_eff = 10/4.69 ≈ 2.13 -> ceil=3, round=2 (discriminating).
    # shrinkage_k=0 so rho_lower = rho_mean exactly (no se deflation).
    # We need pairwise rho to average to ~0.41 with high precision. Use a fixed factor loading.
    F = rng.normal(0, 1, n)
    # Loading chosen so that true corr is exactly 0.41 for all pairs.
    # corr(x,y) = lam^2 for x=lam*F+sqrt(1-lam^2)*eps. So lam = sqrt(0.41).
    lam = math.sqrt(0.41)
    eps = [rng.normal(0, 1, n) for _ in range(5)]
    streams = [_stream(lam * F + math.sqrt(1 - lam**2) * e) for e in eps]

    # With 5 streams, shrinkage_k=0 means rho_lower = mean(rhos).
    r = estimate_n_eff(raw_n, streams, min_siblings=5, min_overlap_bars=21, shrinkage_k=0.0)
    assert r.n_eff is not None
    # Compute expected continuous N_eff from the actual rho_bar returned.
    assert r.rho_bar is not None
    continuous = raw_n / (1.0 + (raw_n - 1) * r.rho_bar)
    expected_ceil = min(raw_n, math.ceil(continuous))
    assert r.n_eff == expected_ceil, (
        f"n_eff should be ceil({continuous:.4f})={expected_ceil}, got {r.n_eff}"
    )
    # Belt-and-suspenders: ceil is >= nearest-integer round (conservative direction).
    assert r.n_eff >= round(continuous), (
        f"ceil direction violated: n_eff={r.n_eff} < round({continuous:.4f})={round(continuous)}"
    )


def test_duplicate_dates_in_stream_returns_none():
    """A sibling stream with duplicate dates is invalid; _pair_correlation must return None
    (fail-closed) so that estimate_n_eff returns None (no N_eff evidence)."""
    rng = _rng(10)
    n = len(_DATES)
    good = _stream(rng.normal(0, 1, n))

    # Build a stream with a duplicated date.
    bad_dates = list(_DATES)
    bad_dates[5] = bad_dates[4]          # duplicate: position 4 and 5 share the same date
    bad_vals = rng.normal(0, 1, n).tolist()
    bad_stream = (bad_vals, bad_dates)

    # Test _pair_correlation directly: any pair involving the duplicate-date stream returns None.
    rho = _pair_correlation(bad_stream, good, min_overlap_bars=21)
    assert rho is None, "duplicate-date stream must cause _pair_correlation to return None"

    rho2 = _pair_correlation(good, bad_stream, min_overlap_bars=21)
    assert rho2 is None, "duplicate dates detected regardless of argument order"

    # Test via estimate_n_eff: any bad pair -> None for the whole estimate.
    fillers = [_stream(rng.normal(0, 1, n)) for _ in range(3)]
    sibs = [bad_stream, good] + fillers
    r = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r.n_eff is None, "duplicate-date sibling must cause N_eff to be None (fail-closed)"


def test_shrinkage_pulls_n_eff_toward_raw_n():
    # Higher shrinkage_k -> lower rho_bar_lower -> N_eff closer to raw_n.
    rng = _rng(8)
    base = rng.normal(0, 1, len(_DATES))
    sibs = [_stream(base + 0.5 * rng.normal(0, 1, len(_DATES))) for _ in range(6)]
    low_k = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=0.0)
    high_k = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=3.0)
    assert low_k.n_eff is not None and high_k.n_eff is not None
    assert high_k.n_eff >= low_k.n_eff      # more shrinkage -> closer to raw N
