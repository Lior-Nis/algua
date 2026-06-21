import math

import numpy as np

from algua.backtest.neff import _pair_correlation, estimate_n_eff

# ---------------------------------------------------------------------------
# Helper: compute the OLD σ_ρ/√M reference bound (inline, from pair correlations)
# ---------------------------------------------------------------------------


def _old_rho_lower(rhos: list[float], shrinkage_k: float) -> float:
    """Reference implementation of the OLD clamp(ρ̄ − k·σ_ρ/√M) bound.

    Used in the tighten-property test to assert new rho_bar ≤ old_rho_lower.
    M = pair count (C(n,2)), which the old code used — treating all pairs as independent.
    """
    arr = np.asarray(rhos, dtype=float)
    m = len(arr)
    rho_mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(m)) if m >= 2 else 0.0
    return min(1.0, max(0.0, rho_mean - shrinkage_k * se))

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

    result = _pair_correlation(a, b, min_overlap_bars=21)
    assert result is not None, "inner-join should produce a finite correlation"
    # Q2.2: _pair_correlation now returns (rho, n_overlap); unpack accordingly.
    rho, n_overlap = result
    assert rho > 0.99, f"date-keyed alignment must yield corr ≈ 1, got {rho}"
    assert n_overlap == 61, f"expected 61 common dates, got {n_overlap}"

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


# ---------------------------------------------------------------------------
# Q2.2 Fisher-z effective-N tests (new, replace σ_ρ/√M bound)
# ---------------------------------------------------------------------------


def test_fisher_z_tighten_property_vs_pair_count_bound():
    """HEADLINE INVARIANT (Q2.2): the new Fisher-z rho_bar (ρ̄_lower) must be ≤ the old
    σ_ρ/√M bound (M = pair count) for every point in the synthetic grid.

    This proves: N_eff_new ≥ N_eff_old — the new estimator is never more lenient.

    Grid: n_sib ∈ {5, 8}, correlation level ∈ {low, mid, high}, overlap ∈ {30, 63}.
    The old bound is computed inline via _old_rho_lower() from the same pair correlations.
    """
    shrinkage_k = 1.0
    rng = _rng(42)

    def make_sibs(n_sib: int, factor_loading: float, n_dates: int) -> tuple:
        """1-factor model: x_i = lam*F + sqrt(1-lam^2)*eps_i → E[corr] = lam^2."""
        dates = _DATES[:n_dates]
        F = rng.normal(0, 1, n_dates)
        eps_list = [rng.normal(0, 1, n_dates) for _ in range(n_sib)]
        lam = factor_loading
        streams = [_stream(lam * F + math.sqrt(max(0, 1 - lam**2)) * e, dates)
                   for e in eps_list]
        return streams, dates

    failures = []
    old_bounds_list: list[float] = []
    for n_sib in (5, 8):
        for lam in (0.1, 0.6, 0.9):   # low / mid / high correlation
            for n_dates in (30, 63):
                sibs, _ = make_sibs(n_sib, lam, n_dates)

                # Collect pair correlations the same way the implementation does
                pair_rhos: list[float] = []
                for a, b in __import__("itertools").combinations(sibs, 2):
                    result = _pair_correlation(a, b, min_overlap_bars=21)
                    if result is None:
                        # skip degenerate realisations (extremely unlikely with lam≠0)
                        continue
                    pair_rhos.append(result[0])   # float rho only, NOT the (rho, n_overlap) tuple

                if not pair_rhos:
                    continue

                old_bound = _old_rho_lower(pair_rhos, shrinkage_k)
                old_bounds_list.append(old_bound)
                r = estimate_n_eff(40, sibs,
                                   min_siblings=2,
                                   min_overlap_bars=21,
                                   shrinkage_k=shrinkage_k)

                if r.rho_bar is None:
                    # No estimate possible (degenerate) — skip
                    continue

                if r.rho_bar > old_bound + 1e-12:
                    failures.append(
                        f"n_sib={n_sib} lam={lam} n_dates={n_dates}: "
                        f"new rho_bar={r.rho_bar:.6f} > old_bound={old_bound:.6f} "
                        f"(violates must-tighten)"
                    )

    # Non-vacuity guard: at least some grid points must produce a non-trivially-clamped
    # old_bound < 1.0; if all are 1.0 the test cannot distinguish correct from broken impl.
    assert any(b < 1.0 for b in old_bounds_list), (
        "All old_bounds are 1.0 — grid is degenerate (vacuous test). "
        "Check that pair_rhos contains float rho values, not (rho, n_overlap) tuples."
    )
    assert not failures, "Must-tighten violated:\n" + "\n".join(failures)


def test_pair_correlation_returns_tuple_with_overlap():
    """After Q2.2, _pair_correlation returns (rho, n_overlap) — a 2-tuple.

    The n_overlap field is the count of common dates; it feeds Fisher-z sampling variance.
    On failure paths (short overlap, zero variance, duplicate dates), it must return None.
    """
    rng = _rng(77)
    n = len(_DATES)
    a = _stream(rng.normal(0, 1, n))
    b = _stream(rng.normal(0, 1, n))

    result = _pair_correlation(a, b, min_overlap_bars=21)
    assert result is not None, "_pair_correlation on valid pair should not return None"
    assert isinstance(result, tuple) and len(result) == 2, (
        f"_pair_correlation must return (rho, n_overlap) tuple; got {type(result)}"
    )
    rho, n_overlap = result
    assert math.isfinite(rho) and -1.0 <= rho <= 1.0
    assert isinstance(n_overlap, int) and n_overlap == n  # all dates shared


def test_pair_correlation_none_on_failure_paths_still_returns_none():
    """Failure paths must still return None (not a tuple) — fail-closed preserved."""
    rng = _rng(78)
    n = len(_DATES)
    good = _stream(rng.normal(0, 1, n))
    flat = _stream([0.0] * n)  # zero variance

    # zero-variance pair
    assert _pair_correlation(flat, good, min_overlap_bars=21) is None
    assert _pair_correlation(good, flat, min_overlap_bars=21) is None

    # insufficient overlap
    short_dates = _DATES[:10]
    short = _stream(rng.normal(0, 1, 10), short_dates)
    assert _pair_correlation(good, short, min_overlap_bars=21) is None

    # duplicate dates
    bad_dates = list(_DATES)
    bad_dates[1] = bad_dates[0]
    bad = (rng.normal(0, 1, n).tolist(), bad_dates)
    assert _pair_correlation(good, bad, min_overlap_bars=21) is None


def test_fisher_z_n_overlap_used_for_sampling_variance():
    """The Fisher-z sampling variance uses n_overlap per pair (1/(n_overlap-3)).

    With fewer overlapping dates, sampling variance is higher → SE_z larger → rho_bar lower.
    We demonstrate this directionally: two sibling sets are constructed with the SAME
    correlation structure (same factor loading, same RNG seed draws) but different overlap
    lengths (200 bars vs 25 bars). The short-overlap set must yield rho_bar <= the
    long-overlap set's rho_bar (more uncertainty → more conservative lower bound).

    We use a large n_dates for the long arm (200) so the directional gap is unambiguous even
    in a single realisation, and shrinkage_k=2.0 to amplify SE_z differences.
    """
    lam = 0.7
    n_sib = 5
    shrinkage_k = 2.0

    # Build both arms from the SAME factor/eps draws by using a fixed large date list and
    # simply truncating for the short arm. This keeps the realized pair-rho values close.
    long_dates = [f"2000-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(200)]
    rng_long = _rng(55)
    F_long = rng_long.normal(0, 1, 200)
    eps_long = [rng_long.normal(0, 1, 200) for _ in range(n_sib)]
    sibs_long = [_stream(lam * F_long + math.sqrt(1 - lam**2) * e, long_dates)
                 for e in eps_long]

    # Short arm: take the first 25 values/dates from the same draws.
    short_dates = long_dates[:25]
    sibs_short = [_stream((lam * F_long + math.sqrt(1 - lam**2) * eps_long[i])[:25].tolist(),
                          short_dates)
                  for i in range(n_sib)]

    r_long = estimate_n_eff(
        40, sibs_long, min_siblings=5, min_overlap_bars=21, shrinkage_k=shrinkage_k
    )
    r_short = estimate_n_eff(
        40, sibs_short, min_siblings=5, min_overlap_bars=21, shrinkage_k=shrinkage_k
    )

    # Both should yield a valid estimate (both > min_overlap_bars=21).
    assert r_long.rho_bar is not None, "long-overlap arm should produce a valid rho_bar"
    assert r_short.rho_bar is not None, "short-overlap arm should produce a valid rho_bar"

    # Directional assertion: shorter overlap → larger 1/(n-3) sampling variance → larger SE_z
    # → lower rho_bar lower-bound. Allow a small numerical slack (1e-9) for floating-point.
    assert r_short.rho_bar <= r_long.rho_bar + 1e-9, (
        f"Shorter overlap should yield rho_bar <= long-overlap rho_bar; "
        f"got short={r_short.rho_bar:.6f} > long={r_long.rho_bar:.6f}"
    )


def test_fisher_z_determinism():
    """Fisher-z is closed-form (no RNG); identical inputs must produce identical outputs."""
    rng = _rng(100)
    n = len(_DATES)
    sibs = [_stream(rng.normal(0, 1, n)) for _ in range(5)]

    r1 = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    r2 = estimate_n_eff(40, sibs, min_siblings=5, min_overlap_bars=21, shrinkage_k=1.0)
    assert r1 == r2, f"Determinism violated: {r1} != {r2}"


def test_fisher_z_rho_zero_clamp_gives_n_eff_equals_raw_n():
    """When rho_bar_lower is clamped to 0 (mean correlation ≤ 0), N_eff = raw_n.

    Construction: use exactly 2 siblings (min_siblings=2) that are perfectly anti-correlated
    (corr = -1). With a single pair, rho_mean = -1 < 0 so rho_lower is clamped to 0, and
    the Kish formula gives N_eff = raw_n/(1+0) = raw_n.

    This is fully deterministic — no random fillers, no opaque RNG luck.
    """
    n = len(_DATES)
    # Two perfectly anti-correlated streams: corr(V, -V) = -1 → single pair rho = -1 < 0
    vals = np.linspace(-1, 1, n).tolist()
    s_pos = _stream(vals)
    s_neg = _stream([-v for v in vals])

    r = estimate_n_eff(40, [s_pos, s_neg], min_siblings=2, min_overlap_bars=21, shrinkage_k=0.0)
    assert r.n_eff is not None
    assert r.rho_bar is not None and r.rho_bar == 0.0, (
        f"Anti-correlated pair should give rho_bar=0 (clamped from -1), got {r.rho_bar}"
    )
    assert r.n_eff == 40, f"rho_bar=0 → N_eff=raw_n=40, got {r.n_eff}"


def test_fisher_z_m_eff_equals_n_sib_not_pair_count():
    """M_eff = n_sib (sibling count), NOT C(n_sib, 2) pair count.

    Rationale: pairs sharing a strategy are dependent; the independent information
    scales with the number of strategies (n_sib), not the number of pairs.

    Consequence: for n_sib=5, M_eff=5 not 10. The Fisher-z SE_z is larger than if
    M_eff=10 were used → rho_bar is lower → N_eff is more conservative.

    We probe this indirectly: with n_sib=2 (1 pair) M_eff=2, the dispersion_var=0 (m<2),
    Var(z_bar) = mean(v)/M_eff = (1/(n_overlap-3))/2. This is finite and positive,
    so SE_z > 0 and shrinkage deflates rho_bar below rho_mean.
    """
    rng = _rng(102)
    n = len(_DATES)
    # 2 siblings with known moderate correlation (1-factor, lam=0.7)
    lam = 0.7
    F = rng.normal(0, 1, n)
    eps1 = rng.normal(0, 1, n)
    eps2 = rng.normal(0, 1, n)
    s1 = _stream(lam * F + math.sqrt(1 - lam**2) * eps1)
    s2 = _stream(lam * F + math.sqrt(1 - lam**2) * eps2)

    r_k0 = estimate_n_eff(40, [s1, s2], min_siblings=2, min_overlap_bars=21, shrinkage_k=0.0)
    r_k1 = estimate_n_eff(40, [s1, s2], min_siblings=2, min_overlap_bars=21, shrinkage_k=1.0)

    assert r_k0.rho_bar is not None and r_k1.rho_bar is not None
    # With k=0: rho_bar = tanh(z_bar) ≈ rho_mean; with k=1: rho_bar < rho_bar_k0
    # (SE_z > 0 because M_eff=2 gives finite variance even with 1 pair)
    assert r_k1.rho_bar <= r_k0.rho_bar + 1e-12, (
        f"SE_z should be > 0 with M_eff=n_sib=2; expected rho_bar(k=1) <= rho_bar(k=0), "
        f"got {r_k1.rho_bar:.6f} vs {r_k0.rho_bar:.6f}"
    )


def test_small_overlap_bars_fails_closed_no_zerodivision():
    """A caller passing min_overlap_bars <= 3 must fail closed (None), never ZeroDivisionError on
    the Fisher-z sampling variance 1/(n_overlap-3). Production uses 21; this guards a misuse."""
    dates = [f"2020-01-{i + 1:02d}" for i in range(3)]
    a = ([0.0, 1.0, 2.0], list(dates))
    b = ([0.0, 2.0, 4.0], list(dates))
    for mo in (1, 2, 3):
        r = estimate_n_eff(10, [a, b], min_siblings=2, min_overlap_bars=mo, shrinkage_k=1.0)
        assert r.n_eff is None  # fail-closed, no exception
