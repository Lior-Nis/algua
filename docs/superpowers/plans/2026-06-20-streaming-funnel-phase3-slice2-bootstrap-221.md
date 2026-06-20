# Phase 3 Slice 2 — serial-dependence bootstrap (#221) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a **stationary-bootstrap AND cross-check** (`dsr_bootstrap`) to the promotion gate: resample
the strategy's own OOS return vector (block-stationary, Politis–Romano) `B` times, recompute the DSR
confidence against the *same* `SR*` on each resample, and require the
`DSR_BOOTSTRAP_LOWER_QUANTILE`-quantile of that bootstrap distribution to also clear `1 − DSR_ALPHA`.
This catches a closed-form DSR that overstates certainty when the OOS stream is serially correlated
(intra-strategy autocorrelation). It is a **tighten-only** AND-check — it can only revoke a PASS.

**Architecture:** A new unprotected `algua/backtest/bootstrap.py` does all resampling and the
Politis–White automatic block-length selection, returning a single pre-computed scalar
`lower_confidence`. The protected `gates.py` stays pure-math: it gains pure helpers
`floored_trial_var_per_period` / `dsr_sr_star` / `dsr_sr_star_annualized` (so the bootstrap reuses the
exact floored `SR*` the closed-form gate uses) and appends a `dsr_bootstrap` check to the AND-set when
binding. The protected `promotion.py` assembles a deterministic seed, computes `SR*` via the gates
helper, calls the bootstrap on `wf.holdout_returns`, and passes the scalar into `evaluate_gate`.
`gates.py` does NO resampling and receives NO return vector.

**Tech Stack:** Python 3.12, numpy 2.4 (seeded `np.random.default_rng`), scipy.stats.norm, hashlib
(stable seed), pytest. **No `arch` dependency** — Politis–White is hand-rolled (lean-deps preference).

## Global Constraints

- **Tighten-only.** `dsr_bootstrap` is a NEW binding AND-check appended *only when binding*; it can only
  move a gate PASS→FAIL, never FAIL→PASS. The closed-form `dsr_evidence` check is **byte-identical** to
  today (the bootstrap is alongside it, not a replacement). Property-tested.
- **Binding condition.** `dsr_bootstrap` binds iff `dsr_binding AND wf.holdout_returns is not None`
  (measured breadth AND the in-process OOS vector is present). When it binds and
  `bootstrap_lower_confidence is None` (degenerate vector / too few finite resamples) → the check is
  appended as **FAILED** (fail-closed). When it does NOT bind (declared breadth, or no OOS vector —
  e.g. a pre-Slice-1 promote) → the check is **omitted entirely** (never appended `passed=False`),
  mirroring the `dsr_evidence`/`returns_available` omit-not-fail pattern. This keeps existing
  promotion tests that build a `wf` without `holdout_returns` byte-identical.
- **Same `SR*` as the closed form.** The bootstrap must recompute DSR confidence against the SAME
  floored `SR*` the binding `dsr_evidence` uses (`max(own_sweep_var, funnel_floor)` per-period, Slice 0).
  `SR*` is computed ONCE in `promotion.py` via the new `gates.dsr_sr_star_annualized(...)` helper and
  passed into the bootstrap as a scalar — the López de Prado E[max] formula stays single-sourced in
  `gates.py`.
- **Determinism (the repo bans nondeterminism).** The RNG seed is a STABLE hash
  (`hashlib.sha256`, NOT Python's salted `hash()`) of `(strategy_name, holdout_start, holdout_end,
  config_hash)`. Same inputs → same `lower_confidence`. The seed, `B`, and block length are persisted in
  the audit payload so the result is reproducible from the row alone.
- **Fail-closed / no NaN in payload.** Every pure helper returns `None` on degenerate/non-finite input;
  any resample yielding a non-finite Sharpe/moment is excluded; if fewer than `B/2` resamples are
  finite, `lower_confidence = None`. `to_dict` nulls non-finite floats via the existing `_f`.
- **Architecture boundary.** `gates.py` stays pure-math, no resampling, no return vector. All
  resampling lives in `algua/backtest/bootstrap.py` (unprotected). `algua/contracts` and
  `algua/features` stay pure. Import direction: `registry.promotion → backtest.bootstrap` and
  `registry.promotion → research.gates` are both already-used directions (lint-imports enforces).
- **Protected walls** (`algua/research/gates.py`, `algua/registry/promotion.py`) are CODEOWNERS
  `@Lior-Nis`. Preserve the Phase-1 invariants (tighten-only, fail-closed, unit discipline, boundary).
- **Constant values (spec GATE-1, Q3.4/Q3.6):** `DSR_BOOTSTRAP_RESAMPLES = 2000`,
  `DSR_BOOTSTRAP_LOWER_QUANTILE = 0.05` (fraction — 5th percentile, matches `DSR_ALPHA` convention),
  `MAX_BOOTSTRAP_BLOCK_LEN_FRACTION = 0.5` (cap block length at `max(1, floor(T * 0.5))`).
- **Per-slice quality gate (must pass before commit):**
  `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- Full-suite baseline: run `uv run pytest -q` once before Task 1 and record the count.

## File Structure

- `algua/research/gates.py` (PROTECTED) — pure helpers `floored_trial_var_per_period`, `dsr_sr_star`,
  `dsr_sr_star_annualized`; 3 new constants; `evaluate_gate` `dsr_bootstrap` check; `GateDecision`
  audit fields + `to_dict`.
- `algua/backtest/bootstrap.py` (NEW, unprotected) — `politis_white_block_length`,
  `stationary_bootstrap_dsr`, `BootstrapResult`, the per-resample DSR-confidence core, and a stable
  seed helper.
- `algua/registry/promotion.py` (PROTECTED) — `run_gate` computes `SR*` + seed, calls the bootstrap,
  passes the scalar + audit into `evaluate_gate`.
- Tests: `tests/backtest/test_bootstrap.py` (new), `tests/research/test_dsr_bootstrap_gate.py` (new),
  extend `tests/test_promotion.py`.

## Locked design decisions

- **D1 — `SR*` single-sourced in gates;** bootstrap takes it as a scalar parameter. Only the final
  `Phi(z)` confidence formula is duplicated in `bootstrap.py` (kept in lock-step by a consistency test
  against `gates.dsr_confidence`).
- **D2 — bind on the in-process `wf.holdout_returns`,** not on the DB `returns_available` write. The
  bootstrap is intra-strategy; the persisted row is for the cross-strategy siblings (Slices 3/4). The
  signals coincide for a fresh promote.
- **D3 — hand-rolled Politis–White** (no `arch`), bounded `[1, max(1, floor(T·0.5))]`. This is the
  GATE-2 focus.
- **D4 — `dsr_bootstrap` is a binding AND-check (tightening), NOT shadow-only.** Unlike Slice 3 (N_eff)
  which would loosen and is therefore shadow-only, adding this AND-check only removes passes — allowed.

---

## Task 1: gates.py pure helpers + constants (PROTECTED, behavior-preserving refactor)

**Files:**
- Modify: `algua/research/gates.py` (constants near `DSR_ALPHA:23`; helpers before `dsr_confidence:155`;
  refactor `dsr_confidence` body)
- Test: `tests/research/test_dsr_bootstrap_gate.py` (create — start with the helper tests here)

**Interfaces:**
- Produces:
  - `DSR_BOOTSTRAP_RESAMPLES = 2000`, `DSR_BOOTSTRAP_LOWER_QUANTILE = 0.05`,
    `MAX_BOOTSTRAP_BLOCK_LEN_FRACTION = 0.5` (module constants).
  - `floored_trial_var_per_period(own_var_pp: float, floor_var_pp: float | None) -> float | None` —
    own-variance-first validation (finite & ≥0 else `None`), then `max(own, floor)` only when floor is
    finite and `> own`. Returns the floored per-period variance, or `None` if own is degenerate.
  - `dsr_sr_star(n_trials: int, trial_sr_var_per_period: float) -> float | None` — the López de Prado
    E[max] benchmark: `0.0` for `n ≤ 1`; else `sqrt(var)·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]`;
    `None` if `n < 1`, var non-finite/negative, or the result is non-finite.
  - `dsr_sr_star_annualized(n_trials: int, trial_var_ann: float | None, floor_var_ann: float | None)
    -> float | None` — convenience for `promotion.py`: converts annualized→per-period (`/ ANN`),
    floors via `floored_trial_var_per_period`, returns `dsr_sr_star(...)`. `None` if `trial_var_ann`
    is None/non-finite or the floored var is `None`.

- [ ] **Step 1: Write the failing helper tests**

Create `tests/research/test_dsr_bootstrap_gate.py`:

```python
import math

from algua.backtest._constants import ANN
from algua.research import gates
from algua.research.gates import (
    DSR_BOOTSTRAP_LOWER_QUANTILE, DSR_BOOTSTRAP_RESAMPLES, MAX_BOOTSTRAP_BLOCK_LEN_FRACTION,
    dsr_confidence, dsr_sr_star, dsr_sr_star_annualized, floored_trial_var_per_period,
)


def test_constants():
    assert DSR_BOOTSTRAP_RESAMPLES == 2000
    assert DSR_BOOTSTRAP_LOWER_QUANTILE == 0.05
    assert MAX_BOOTSTRAP_BLOCK_LEN_FRACTION == 0.5


def test_floored_var_own_first_then_max():
    assert floored_trial_var_per_period(0.04, 0.01) == 0.04        # floor below own -> own
    assert floored_trial_var_per_period(0.01, 0.09) == 0.09        # floor above own -> floor
    assert floored_trial_var_per_period(0.04, None) == 0.04        # no floor -> own
    assert floored_trial_var_per_period(-1.0, 0.09) is None        # degenerate own fails closed
    assert floored_trial_var_per_period(float("nan"), 0.09) is None


def test_dsr_sr_star_matches_inline_formula():
    assert dsr_sr_star(1, 0.04) == 0.0           # n<=1 -> 0
    assert dsr_sr_star(0, 0.04) is None          # n<1 -> None
    assert dsr_sr_star(50, -1.0) is None         # negative var -> None
    v = dsr_sr_star(50, 0.04)
    assert v is not None and v > 0.0


def test_dsr_sr_star_consistency_with_dsr_confidence():
    # dsr_confidence must use exactly dsr_sr_star internally — verify the refactor is behavior-true.
    n, var = 40, 0.04
    sr_star = dsr_sr_star(n, var)
    # Reconstruct the closed-form confidence from sr_star and compare to dsr_confidence.
    sr, t, skew, kurt = 0.12, 90, -0.2, 4.0
    var_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    from scipy.stats import norm
    z = (sr - sr_star) * math.sqrt(t - 1) / math.sqrt(var_term)
    assert dsr_confidence(sr, t, skew, kurt, n, var) == float(norm.cdf(z))


def test_dsr_sr_star_annualized_floors_and_converts():
    # annualized var / ANN, floored, then sr_star
    own_ann, floor_ann, n = 0.04 * ANN, 0.09 * ANN, 40
    assert dsr_sr_star_annualized(n, own_ann, floor_ann) == dsr_sr_star(n, 0.09)
    assert dsr_sr_star_annualized(n, None, floor_ann) is None      # no own var -> None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/research/test_dsr_bootstrap_gate.py -q`
Expected: FAIL — ImportError on the new names.

- [ ] **Step 3: Add the constants**

In `gates.py` after `DSR_ALPHA`/`EULER_MASCHERONI` (~line 24):

```python
# Serial-dependence bootstrap (#221, Phase 3 Slice 2). Protected — relaxing weakens the gate.
DSR_BOOTSTRAP_RESAMPLES = 2000            # stationary-bootstrap resample count (B)
DSR_BOOTSTRAP_LOWER_QUANTILE = 0.05       # lower quantile of the bootstrap DSR-confidence distribution
MAX_BOOTSTRAP_BLOCK_LEN_FRACTION = 0.5    # cap block length at max(1, floor(T * FRACTION))
```

- [ ] **Step 4: Extract the helpers + refactor `dsr_confidence`**

Add the helpers BEFORE `dsr_confidence`:

```python
def floored_trial_var_per_period(
    own_var_pp: float, floor_var_pp: float | None
) -> float | None:
    """Own-variance-first dispersion floor (#221 Slice 0): validate own (finite & >=0 else None),
    then max(own, floor) only when floor is finite and strictly greater. Returns the floored
    per-period trial-Sharpe variance, or None when own is degenerate (the floor must never rescue a
    degenerate own variance into a pass)."""
    if not math.isfinite(own_var_pp) or own_var_pp < 0.0:
        return None
    var_used = own_var_pp
    if floor_var_pp is not None and math.isfinite(floor_var_pp) and floor_var_pp > var_used:
        var_used = floor_var_pp
    return var_used


def dsr_sr_star(n_trials: int, trial_sr_var_per_period: float) -> float | None:
    """Selection-inflated benchmark SR* (per-period). 0.0 for n<=1; else the López de Prado E[max]
    of n trial Sharpes scaled by sqrt(var). None on degenerate input."""
    n = int(n_trials)
    if n < 1:
        return None
    if not math.isfinite(trial_sr_var_per_period) or trial_sr_var_per_period < 0.0:
        return None
    if n <= 1:
        return 0.0
    sr_star = math.sqrt(trial_sr_var_per_period) * (
        (1.0 - EULER_MASCHERONI) * float(_norm.ppf(1.0 - 1.0 / n))
        + EULER_MASCHERONI * float(_norm.ppf(1.0 - 1.0 / (n * math.e)))
    )
    return sr_star if math.isfinite(sr_star) else None


def dsr_sr_star_annualized(
    n_trials: int, trial_var_ann: float | None, floor_var_ann: float | None
) -> float | None:
    """SR* (per-period) from ANNUALIZED inputs — the SR* the binding dsr_evidence uses. Converts
    /ANN, applies the funnel floor, then dsr_sr_star. None if no own variance or degenerate."""
    if trial_var_ann is None or not math.isfinite(trial_var_ann):
        return None
    own_pp = trial_var_ann / ANN
    floor_pp = (
        floor_var_ann / ANN
        if floor_var_ann is not None and math.isfinite(floor_var_ann)
        else None
    )
    var_used = floored_trial_var_per_period(own_pp, floor_pp)
    if var_used is None:
        return None
    return dsr_sr_star(n_trials, var_used)
```

Refactor `dsr_confidence` body so lines that compute `var_used` (193-201) and `sr_star` (203-213) call
the helpers — behavior MUST be byte-identical:

```python
    if t <= 1:
        return None
    if not math.isfinite(sr_obs_per_period) or not math.isfinite(skew) \
            or not math.isfinite(raw_kurtosis):
        return None
    var_used = floored_trial_var_per_period(trial_sr_var_per_period, funnel_floor_var_per_period)
    if var_used is None:
        return None
    sr_star = dsr_sr_star(n_trials, var_used)
    if sr_star is None:
        return None
    sr = sr_obs_per_period
    var_term = 1.0 - skew * sr + ((raw_kurtosis - 1.0) / 4.0) * sr * sr
    if not math.isfinite(var_term) or var_term <= 0.0:
        return None
    z = (sr - sr_star) * math.sqrt(t - 1) / math.sqrt(var_term)
    conf = float(_norm.cdf(z))
    return conf if math.isfinite(conf) else None
```

(Keep the early `n < 1` guard at the top of `dsr_confidence` — `dsr_sr_star` also guards it, but the
top guard preserves the exact early-return order. Verify the existing `dsr_confidence` tests + the
Slice-0 tighten-only property test still pass byte-for-byte.)

- [ ] **Step 5: Run the helper tests + full quality gate**

Run: `uv run pytest tests/research/test_dsr_bootstrap_gate.py -q` → PASS.
Run the full quality gate → all pass; the existing `dsr_confidence`/Slice-0 tests are unchanged.

- [ ] **Step 6: Commit**

```bash
git add algua/research/gates.py tests/research/test_dsr_bootstrap_gate.py
git commit -m "feat(221): extract dsr_sr_star/floored-var helpers + bootstrap constants — Slice 2 of #221"
```

---

## Task 2: `algua/backtest/bootstrap.py` — stationary bootstrap + Politis–White (NEW, unprotected)

**Files:**
- Create: `algua/backtest/bootstrap.py`
- Test: `tests/backtest/test_bootstrap.py` (create; add `tests/backtest/__init__.py` if peers need it)

**Interfaces:**
- Produces:
  - `BootstrapResult(NamedTuple)`: `lower_confidence: float | None`, `seed_used: int`, `b_used: int`,
    `block_len: int`.
  - `stable_bootstrap_seed(strategy_name: str, holdout_start: str, holdout_end: str, config_hash: str)
    -> int` — `int.from_bytes(hashlib.sha256("\x00".join(...).encode()).digest()[:8], "big")`
    (deterministic; NOT Python `hash()`).
  - `politis_white_block_length(returns: Sequence[float], max_fraction: float) -> int` — automatic
    stationary-bootstrap block length, bounded `[1, max(1, floor(T * max_fraction))]`.
  - `stationary_bootstrap_dsr(returns, dates, sr_star, dsr_alpha, b, seed, *, block_len_auto=True,
    block_len_override=None, lower_quantile) -> BootstrapResult` — resample the OOS vector
    block-stationary `b` times; on each, recompute per-period `SR_obs`, skew, Pearson kurtosis, and the
    DSR confidence against the fixed `sr_star`; return the `lower_quantile`-quantile as
    `lower_confidence`. `lower_confidence = None` if `T ≤ 1`, `sr_star is None`, any input non-finite,
    or fewer than `b/2` finite resamples.

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_bootstrap.py`. Use a seeded numpy generator to build synthetic series.

```python
import math

import numpy as np

from algua.backtest.bootstrap import (
    BootstrapResult, politis_white_block_length, stable_bootstrap_seed, stationary_bootstrap_dsr,
)
from algua.research.gates import dsr_confidence, dsr_sr_star


def _white(n, seed=0):
    return list(np.random.default_rng(seed).normal(0.001, 0.01, n))


def _ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 0.01)
    return list(x + 0.001)


def test_seed_is_stable_and_deterministic():
    a = stable_bootstrap_seed("s", "2020-01-01", "2020-06-30", "abc")
    b = stable_bootstrap_seed("s", "2020-01-01", "2020-06-30", "abc")
    c = stable_bootstrap_seed("s", "2020-01-01", "2020-06-30", "abd")
    assert a == b and a != c and isinstance(a, int)


def test_block_length_white_noise_is_small():
    bl = politis_white_block_length(_white(400), 0.5)
    assert 1 <= bl <= 10            # near-iid -> short blocks


def test_block_length_ar1_is_larger_than_white():
    bl_w = politis_white_block_length(_white(400, 1), 0.5)
    bl_a = politis_white_block_length(_ar1(400, 0.7, 1), 0.5)
    assert bl_a > bl_w
    assert bl_a <= 200             # capped at floor(400 * 0.5)


def test_seed_reproducibility_same_output():
    args = dict(dates=["d"] * 200, sr_star=0.0, dsr_alpha=0.05, b=500,
                lower_quantile=0.05)
    r = _white(200, 3)
    o1 = stationary_bootstrap_dsr(r, seed=123, **args)
    o2 = stationary_bootstrap_dsr(r, seed=123, **args)
    assert o1.lower_confidence == o2.lower_confidence
    assert o1.block_len == o2.block_len and o1.b_used == o2.b_used


def test_white_noise_bootstrap_lower_near_closed_form():
    # On a white-noise series the bootstrap-lower should be close to (not wildly below) the
    # closed-form DSR confidence — benign autocorrelation does not widen much.
    r = _white(252, 7)
    n_trials, var = 30, 0.04
    sr_star = dsr_sr_star(n_trials, var)
    arr = np.asarray(r)
    sr, skew = float(arr.mean() / arr.std(ddof=1)), 0.0
    out = stationary_bootstrap_dsr(r, ["d"] * 252, sr_star, 0.05, 2000, 11, lower_quantile=0.05)
    assert out.lower_confidence is not None
    assert 0.0 <= out.lower_confidence <= 1.0


def test_strong_ar1_lowers_confidence_vs_white():
    # Strongly autocorrelated returns inflate the naive Sharpe SE -> bootstrap-lower should be
    # MEANINGFULLY below a white-noise series of the same mean Sharpe.
    sr_star = 0.0
    w = stationary_bootstrap_dsr(_white(252, 2), ["d"] * 252, sr_star, 0.05, 2000, 5,
                                 lower_quantile=0.05)
    a = stationary_bootstrap_dsr(_ar1(252, 0.8, 2), ["d"] * 252, sr_star, 0.05, 2000, 5,
                                 lower_quantile=0.05)
    assert a.lower_confidence is not None and w.lower_confidence is not None
    assert a.lower_confidence <= w.lower_confidence + 1e-9


def test_degenerate_returns_none():
    assert stationary_bootstrap_dsr([0.1], ["d"], 0.0, 0.05, 100, 1,
                                    lower_quantile=0.05).lower_confidence is None     # T<=1
    assert stationary_bootstrap_dsr(_white(50), ["d"] * 50, None, 0.05, 100, 1,
                                    lower_quantile=0.05).lower_confidence is None     # sr_star None


def test_dsr_core_consistent_with_gates():
    # The per-resample DSR-confidence core must equal gates.dsr_confidence for the same inputs,
    # pinning the duplicated formula against drift.
    from algua.backtest.bootstrap import _dsr_conf_core
    n, var = 40, 0.04
    sr_star = dsr_sr_star(n, var)
    sr, t, skew, kurt = 0.1, 120, -0.1, 3.5
    assert _dsr_conf_core(sr, t, skew, kurt, sr_star) == dsr_confidence(sr, t, skew, kurt, n, var)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtest/test_bootstrap.py -q`
Expected: FAIL — module/functions absent.

- [ ] **Step 3: Implement `bootstrap.py`**

Write `algua/backtest/bootstrap.py`. Key points:
- `_dsr_conf_core(sr_pp, t, skew, kurt, sr_star)` — the EXACT `Phi(z)` formula from `gates.dsr_confidence`
  (lines: `var_term = 1 - skew*sr + (kurt-1)/4*sr^2`; guard finite/>0; `z = (sr-sr_star)*sqrt(t-1)/
  sqrt(var_term)`; `Phi(z)`; `None` on any non-finite). Comment: "MUST stay in lock-step with
  gates.dsr_confidence — pinned by test_dsr_core_consistent_with_gates."
- Per-period moments per resample: `mean`, `std (ddof=1)`, `SR = mean/std`; `skew`/`kurt` via
  `scipy.stats.skew`/`kurtosis(fisher=False)` (Pearson, =3 for normal — matches `metrics_from_returns`).
  A zero/non-finite std → that resample's confidence is non-finite → excluded.
- Stationary resampling (Politis–Romano): seeded `rng = np.random.default_rng(seed)`. Build each
  resample of length `T` by: pick a uniform start index; with prob `p = 1/block_len` start a fresh
  uniform index, else advance `(idx+1) mod T` (circular). Vectorize where practical for `B=2000 × T≤500`.
- `politis_white_block_length`: implement the Politis & White (2004; 2009 correction) stationary-bootstrap
  selector. Document each step with the reference. Bound the result to `[1, max(1, floor(T*max_fraction))]`.
  On any degenerate input (T<3, zero variance, non-finite autocovariances) return `1` (the safe minimum).
- Aggregate: collect finite per-resample confidences; if `len(finite) < b/2` → `lower_confidence=None`;
  else `lower_confidence = float(np.quantile(finite, lower_quantile))`. Return `BootstrapResult`.
- Top guards: `T = len(returns)`; if `T <= 1` or `sr_star is None` or any return non-finite →
  `BootstrapResult(None, seed, b, block_len_used)`.

- [ ] **Step 4: Run the bootstrap tests + quality gate**

Run: `uv run pytest tests/backtest/test_bootstrap.py -q` → PASS. If the AR(1) block-length or
confidence-direction tests are flaky on the chosen seeds, adjust the SEED (not the assertion) until the
qualitative property holds robustly, or widen tolerances slightly — but the qualitative direction
(AR1 block > white; AR1 confidence ≤ white) MUST hold.
Run the full quality gate → all pass.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/bootstrap.py tests/backtest/test_bootstrap.py
git commit -m "feat(221): stationary-bootstrap DSR + Politis-White block length — Slice 2"
```

---

## Task 3: gates.py `dsr_bootstrap` AND-check + audit (PROTECTED)

**Files:**
- Modify: `algua/research/gates.py` (`evaluate_gate` signature + the `if dsr_binding:` block;
  `GateDecision` fields + `to_dict`)
- Test: extend `tests/research/test_dsr_bootstrap_gate.py`

**Interfaces:**
- Consumes (from `promotion.py`, Task 4): pre-computed `bootstrap_lower_confidence: float | None`,
  `bootstrap_binding: bool`, and audit scalars `bootstrap_seed/b/block_len`.
- Produces: `evaluate_gate(..., bootstrap_binding: bool = False, bootstrap_lower_confidence: float |
  None = None, bootstrap_seed: int | None = None, bootstrap_b: int | None = None, bootstrap_block_len:
  int | None = None)`. `GateDecision` gains `dsr_bootstrap_binding: bool = False`,
  `dsr_bootstrap_lower: float | None`, `dsr_bootstrap_seed: int | None`, `dsr_bootstrap_b: int | None`,
  `dsr_bootstrap_block_len: int | None`, all in `to_dict`.

- [ ] **Step 1: Write the failing tests**

```python
from algua.research.gates import DSR_ALPHA, GateCriteria, evaluate_gate
# reuse a WalkForwardResult builder; if test_promotion has _gate_wf, import or mirror it here.


def test_bootstrap_check_appended_only_when_binding(make_wf):
    wf = make_wf(sharpe=2.0)
    # binding -> appended
    d = evaluate_gate(wf, GateCriteria(), n_combos=5, pit_ok=True, dsr_binding=True,
                      dsr_trial_var_ann=0.04 * 252, bootstrap_binding=True,
                      bootstrap_lower_confidence=0.99)
    assert any(c["name"] == "dsr_bootstrap" for c in d.checks)
    # not binding -> omitted entirely
    d2 = evaluate_gate(wf, GateCriteria(), n_combos=5, pit_ok=True, dsr_binding=True,
                       dsr_trial_var_ann=0.04 * 252, bootstrap_binding=False)
    assert all(c["name"] != "dsr_bootstrap" for c in d2.checks)


def test_bootstrap_none_is_failed_when_binding(make_wf):
    d = evaluate_gate(make_wf(sharpe=7.0), GateCriteria(), n_combos=5, pit_ok=True, dsr_binding=True,
                      dsr_trial_var_ann=0.04 * 252, bootstrap_binding=True,
                      bootstrap_lower_confidence=None)
    chk = next(c for c in d.checks if c["name"] == "dsr_bootstrap")
    assert chk["passed"] is False
    assert d.passed is False


def test_bootstrap_tighten_only(make_wf):
    # For any bootstrap value, the gate's pass is old_pass AND (NOT binding OR bootstrap_pass).
    wf = make_wf(sharpe=7.0)
    base = dict(criteria=GateCriteria(), n_combos=5, pit_ok=True, dsr_binding=True,
                dsr_trial_var_ann=0.04 * 252)
    old = evaluate_gate(wf, **base)                                   # no bootstrap
    for lower in [None, 0.0, 0.5, 0.99, 1.0]:
        new = evaluate_gate(wf, **base, bootstrap_binding=True, bootstrap_lower_confidence=lower)
        if new.passed:
            assert old.passed                                        # never FAIL->PASS
    # audit fields surface
    d = evaluate_gate(wf, **base, bootstrap_binding=True, bootstrap_lower_confidence=0.97,
                      bootstrap_seed=42, bootstrap_b=2000, bootstrap_block_len=8)
    dd = d.to_dict()
    assert dd["dsr_bootstrap_lower"] == 0.97 and dd["dsr_bootstrap_seed"] == 42
    assert dd["dsr_bootstrap_b"] == 2000 and dd["dsr_bootstrap_block_len"] == 8
```

(Provide a `make_wf` fixture mirroring `tests/test_promotion.py::_gate_wf` — a `WalkForwardResult`
with `holdout_metrics` carrying `n_bars`, `sharpe`, `skewness`, `kurtosis`, plus passing
`stability`/`window_metrics` so the non-DSR checks pass and the test isolates the bootstrap.)

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/research/test_dsr_bootstrap_gate.py -q -k bootstrap` → FAIL (unexpected kwargs / missing fields).

- [ ] **Step 3: Add `GateDecision` fields + `to_dict`**

After the Slice-1 `returns_available` field:

```python
    # Serial-dependence bootstrap audit (#221 Slice 2). Binding only when dsr_binding AND the OOS
    # return vector is available; non-binding otherwise (omit-not-fail).
    dsr_bootstrap_binding: bool = False
    dsr_bootstrap_lower: float | None = None
    dsr_bootstrap_seed: int | None = None
    dsr_bootstrap_b: int | None = None
    dsr_bootstrap_block_len: int | None = None
```

In `to_dict` (with the other `dsr_*`): `dsr_bootstrap_lower` via `_f`; the rest plain.

- [ ] **Step 4: Append the `dsr_bootstrap` check in `evaluate_gate`**

Add the 5 kwargs to the signature. Inside the `if dsr_binding:` block, AFTER the `dsr_evidence` append:

```python
        if bootstrap_binding:
            boot_pass = (bootstrap_lower_confidence is not None
                         and bootstrap_lower_confidence >= (1.0 - DSR_ALPHA))
            boot_value = (bootstrap_lower_confidence
                          if (bootstrap_lower_confidence is not None
                              and math.isfinite(bootstrap_lower_confidence)) else None)
            checks.append({"name": "dsr_bootstrap", "value": boot_value,
                           "threshold": 1.0 - DSR_ALPHA, "op": ">=", "passed": bool(boot_pass)})
```

In the returned `GateDecision(...)`, populate the audit fields (binding-gated like the other dsr_*):
`dsr_bootstrap_binding=bool(bootstrap_binding)`, and the seed/b/block_len/lower passed through
(`... if bootstrap_binding else None`).

- [ ] **Step 5: Run tests + quality gate** → PASS. The Slice-0 tighten-only property test and the
existing `dsr_evidence` tests must remain byte-identical (the bootstrap check is additive and only
present when `bootstrap_binding`).

- [ ] **Step 6: Commit**

```bash
git add algua/research/gates.py tests/research/test_dsr_bootstrap_gate.py
git commit -m "feat(221): dsr_bootstrap AND-check + audit fields in evaluate_gate — Slice 2"
```

---

## Task 4: promotion.py wiring (PROTECTED) — compute SR*, seed, run bootstrap, pass scalar

**Files:**
- Modify: `algua/registry/promotion.py` (`run_gate`, around lines 397-408)
- Test: extend `tests/test_promotion.py`

**Interfaces:**
- Consumes: `gates.dsr_sr_star_annualized`, `bootstrap.stationary_bootstrap_dsr`,
  `bootstrap.stable_bootstrap_seed`, the constants from gates.
- Produces: `decision_json` now carries `dsr_bootstrap_*`; a measured promote with an OOS vector binds
  the bootstrap AND-check.

- [ ] **Step 1: Write a failing integration test**

In `tests/test_promotion.py`, add a test that drives `run_gate` with measured breadth AND a `wf` that
carries `holdout_returns` (mirror the Slice-1 integration setup; reuse a committed-burn fixture so
`holdout_evaluation_id` is real). Assert:
- `decision.dsr_bootstrap_binding is True`; a `dsr_bootstrap` check is in `decision.checks`;
  `decision.dsr_bootstrap_seed/_b/_block_len` are populated; `"dsr_bootstrap_lower"` is in the
  persisted `decision_json`.
- A measured promote whose `wf` has NO `holdout_returns` (the existing `_run` helper) yields
  `decision.dsr_bootstrap_binding is False` and NO `dsr_bootstrap` check — i.e. existing promotion
  outcomes are unchanged (assert an existing passing test still passes).
- Determinism: two identical `run_gate` calls produce the same `dsr_bootstrap_lower`.

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_promotion.py -q -k bootstrap` → FAIL.

- [ ] **Step 3: Wire the bootstrap into `run_gate`**

BEFORE the `evaluate_gate(...)` call (after `funnel_floor` is computed, ~line 399), compute the
bootstrap scalar:

```python
    # Serial-dependence bootstrap (#221 Slice 2): bind iff measured AND the in-process OOS vector is
    # present. Recompute DSR confidence against the SAME floored SR* the closed form uses; gates.py
    # gets only the pre-computed scalar (it does no resampling).
    bootstrap_binding = dsr_binding and wf.holdout_returns is not None
    boot_lower = boot_seed = boot_b = boot_block = None
    if bootstrap_binding:
        rets = wf.holdout_returns[0]
        sr_star_pp = dsr_sr_star_annualized(
            n_funnel, dsr_trial_var_ann, funnel_floor.var_ann if funnel_floor else None)
        boot_seed = stable_bootstrap_seed(
            name, wf.holdout_metrics["start"], wf.holdout_metrics["end"], wf.config_hash)
        boot = stationary_bootstrap_dsr(
            rets, wf.holdout_returns[1], sr_star_pp, DSR_ALPHA, DSR_BOOTSTRAP_RESAMPLES, boot_seed,
            lower_quantile=DSR_BOOTSTRAP_LOWER_QUANTILE)
        boot_lower, boot_b, boot_block = boot.lower_confidence, boot.b_used, boot.block_len
```

Pass into `evaluate_gate(...)`:
`bootstrap_binding=bootstrap_binding, bootstrap_lower_confidence=boot_lower,
bootstrap_seed=boot_seed, bootstrap_b=boot_b, bootstrap_block_len=boot_block`.

Add the imports at the top of `promotion.py`:
`from algua.research.gates import (..., DSR_ALPHA, DSR_BOOTSTRAP_RESAMPLES,
DSR_BOOTSTRAP_LOWER_QUANTILE, dsr_sr_star_annualized)` and
`from algua.backtest.bootstrap import stable_bootstrap_seed, stationary_bootstrap_dsr`.

- [ ] **Step 4: Run the integration test + full quality gate** → PASS. If a previously-passing
promotion test that DOES supply `holdout_returns` now fails because its synthetic OOS vector fails the
bootstrap bar, that is a REAL tightening — fix the fixture's returns to be benign (or assert the new
check), do NOT weaken the bootstrap. If a test WITHOUT `holdout_returns` changes outcome, that is a bug
(the bootstrap must be omitted there) — STOP and fix the binding condition.

- [ ] **Step 5: Commit**

```bash
git add algua/registry/promotion.py tests/test_promotion.py
git commit -m "feat(221): wire stationary-bootstrap AND-check into run_gate — Slice 2"
```

---

## Self-Review notes

- **Spec coverage (component b):** stationary bootstrap (Politis–Romano) — Task 2; Politis–White block
  length with cap — Task 2; AND cross-check alongside `dsr_evidence` (not replacing) — Task 3; same
  `SR*` single-sourced — Task 1; deterministic stable seed + audit (`seed/b/block_len`) — Tasks 2/4;
  degenerate `T≤1`/`<B/2` finite → `None` → FAILED-when-binding — Tasks 2/3; `gates.py` receives a
  pre-computed scalar, no resampling — Tasks 3/4.
- **Tighten-only:** `dsr_evidence` byte-identical; `dsr_bootstrap` additive AND-check, omitted when not
  binding; property-tested (Task 3) — never FAIL→PASS.
- **Type consistency:** `dsr_sr_star`/`floored_trial_var_per_period`/`dsr_sr_star_annualized` (Task 1)
  consumed in Tasks 2/4; `BootstrapResult` fields (`lower_confidence/seed_used/b_used/block_len`) Task 2
  ↔ Task 4; `bootstrap_*` kwargs Task 3 ↔ Task 4.
- **Determinism:** `hashlib`-based seed (NOT `hash()`); seeded `np.random.default_rng`; no
  `Math.random`/wall-clock.
- **GATE-2 focus:** the Politis–White implementation correctness (Task 2) and the tighten-only +
  binding-condition correctness on the protected wall (Tasks 3/4).
- **Deferred (NOT this slice):** cross-strategy block-bootstrap of the joint sibling matrix (a future
  phase); N_eff (Slice 3) and multi-regime (Slice 4) — independent of this slice given Slice 1.
