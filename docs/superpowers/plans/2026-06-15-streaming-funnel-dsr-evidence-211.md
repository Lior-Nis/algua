# Streaming-Funnel DSR Evidence Gate (Phase 1, issue #211) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a calibrated per-strategy Deflated-Sharpe-Ratio (DSR) confidence check to the
`backtested → candidate` promotion gate, as a tighten-only AND-check beside the existing
deflated-Sharpe haircut.

**Architecture:** A pure `dsr_confidence(...)` (Bailey–López de Prado) lives in the protected
`gates.py`; `evaluate_gate` adds a binding `dsr_evidence` check when measured trial-Sharpe
dispersion is available. The dispersion is recorded by `sweep()` as a `(count, mean, var)` triple
per search-trial row (schema 23→24) and pooled per-strategy via the exact pooled-sample-variance
formula. Holdout skew/raw-kurtosis come from `metrics_from_returns`. Conversions between annualized
and per-period Sharpe units happen inside `gates.py`.

**Tech Stack:** Python, scipy (`scipy.stats.norm`), numpy/pandas, SQLite, pytest.

**Spec:** `docs/superpowers/specs/2026-06-15-streaming-funnel-multiple-testing-issue-211-design.md`
(read it — it carries the statistical rationale, the GATE-1 corrections, and the deferred phases).

**Protected files (CODEOWNERS @Lior-Nis) — human must approve the merge:** `algua/research/gates.py`,
`algua/registry/promotion.py`. The agent may edit them inside the worktree.

---

## File Structure

| File | Protected | Responsibility for this work |
|---|---|---|
| `algua/backtest/metrics.py` | no | Add `skewness` + raw `kurtosis` to `metrics_from_returns`, NaN-coerced. |
| `algua/research/gates.py` | **yes** | Pure `dsr_confidence`; `EULER_MASCHERONI`/`DSR_ALPHA`; `dsr_evidence` binding check + new `GateDecision` fields. |
| `algua/backtest/sweep.py` | no | Compute `(count, mean, var)` over per-combo ranking Sharpes; new `SweepResult` fields. |
| `algua/registry/db.py` | no | Schema 23→24: three nullable `search_trials` columns + migration. |
| `algua/registry/repository.py` | no | `StrategyRepository` Protocol: new `record_search_trial` params + `pooled_trial_sharpe_var` accessor. |
| `algua/registry/store.py` | no | SQLite impl of the above. |
| `algua/cli/backtest_cmd.py` | no | Pass the triple from `SweepResult` into `record_search_trial`. |
| `algua/registry/promotion.py` | **yes** | `run_gate` assembles DSR inputs + binding decision; threads into `evaluate_gate`. |
| `pyproject.toml` | no | Add explicit `scipy` dependency. |

**Pre-flight (do once before Task 1):** confirm `SCHEMA_VERSION` is still 23 (a concurrent session
may have bumped it). Run: `grep -n "SCHEMA_VERSION = " algua/registry/db.py`. If it is not 23, use
`<current>+1` everywhere this plan says "24" and adjust the migration-test assertion accordingly.

---

## Task 1: Holdout skew + raw kurtosis in `metrics_from_returns`

**Files:**
- Modify: `algua/backtest/metrics.py`
- Test: `tests/test_metrics.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
import pandas as pd

from algua.backtest.metrics import metrics_from_returns


def test_metrics_include_skew_and_raw_kurtosis():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, size=2000))
    m = metrics_from_returns(r)
    assert "skewness" in m and "kurtosis" in m
    # raw (Pearson) kurtosis: ~3 for a normal series, NOT ~0 (excess)
    assert abs(m["skewness"]) < 0.25
    assert 2.5 < m["kurtosis"] < 3.5


def test_metrics_moments_finite_on_degenerate_input():
    # empty, single-element, and constant series must never inject NaN/inf
    for r in (pd.Series([], dtype=float), pd.Series([0.01]), pd.Series([0.01, 0.01, 0.01])):
        m = metrics_from_returns(r)
        assert m["skewness"] == 0.0 and m["kurtosis"] == 0.0
        assert all(np.isfinite(v) for v in m.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -q`
Expected: FAIL — `KeyError: 'skewness'`.

- [ ] **Step 3: Implement**

In `algua/backtest/metrics.py`, add the scipy import near the top imports:

```python
from scipy import stats as _stats
```

Replace the body of `metrics_from_returns` (lines 68–76) with:

```python
    r = returns.dropna()
    if len(r) == 0:
        return {name: 0.0 for name in METRIC_FUNCTIONS} | {
            "sharpe": 0.0, "skewness": 0.0, "kurtosis": 0.0,
        }

    out = {name: fn(r) for name, fn in METRIC_FUNCTIONS.items()}
    ann_vol = out["ann_volatility"]
    excess = out["ann_return"] - risk_free
    out["sharpe"] = float(excess / ann_vol) if ann_vol > 0 else 0.0
    # Moments for the DSR non-normality adjustment (#211). RAW (Pearson) kurtosis (fisher=False):
    # a Gaussian series gives ~3, so the gate's (kurtosis-1)/4 term reduces to 0.5. scipy returns
    # NaN for a single-element or zero-variance series; coerce any non-finite moment to 0.0 so no
    # NaN leaks into holdout_metrics / the JSON gate payload. dsr_confidence's T<=1 guard and the
    # MIN_HOLDOUT_OBSERVATIONS=63 floor ensure these placeholders are never consumed.
    skew = float(_stats.skew(r))
    kurt = float(_stats.kurtosis(r, fisher=False))
    out["skewness"] = skew if math.isfinite(skew) else 0.0
    out["kurtosis"] = kurt if math.isfinite(kurt) else 0.0
    return out
```

Add `import math` to the top of the file if not already present (it is not — add it under
`from __future__ import annotations`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -q`
Expected: PASS.

- [ ] **Step 5: Regression-check the broader metrics consumers**

Run: `uv run pytest tests/ -k "metric or walkforward or backtest" -q`
Expected: PASS (the new keys are additive; `portfolio_metrics` builds its own dict and is unaffected).

- [ ] **Step 6: Commit**

```bash
git add algua/backtest/metrics.py tests/test_metrics.py
git commit -m "feat(211): holdout skew + raw kurtosis in metrics_from_returns (NaN-coerced)"
```

---

## Task 2: Pure `dsr_confidence` + constants in `gates.py` (PROTECTED)

**Files:**
- Modify: `algua/research/gates.py`
- Test: `tests/test_research_gates.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_research_gates.py
import math
import pytest
from algua.research.gates import dsr_confidence, DSR_ALPHA, EULER_MASCHERONI


def test_euler_mascheroni_constant():
    assert EULER_MASCHERONI == pytest.approx(0.5772156649015329)


def test_dsr_n1_collapses_to_psr_against_zero():
    # N<=1 -> SR*=0; PSR for SR_pp=0.1, T=252, normal moments.
    # z = 0.1*sqrt(251)/sqrt(1+0.5*0.1**2) ~= 1.580 -> Phi ~= 0.9429
    c = dsr_confidence(0.1, 252, 0.0, 3.0, 1, 0.04)
    assert c == pytest.approx(0.9429, abs=2e-3)


def test_dsr_high_benchmark_rejects():
    # N=10 with sizeable trial dispersion lifts SR* well above SR_obs -> low confidence
    c = dsr_confidence(0.1, 252, 0.0, 3.0, 10, 0.04)
    assert c is not None and c < 0.5


def test_dsr_monotonic_in_n_and_sharpe():
    base = dsr_confidence(0.15, 252, 0.0, 3.0, 5, 0.04)
    assert dsr_confidence(0.15, 252, 0.0, 3.0, 50, 0.04) < base   # more trials -> stricter
    assert dsr_confidence(0.25, 252, 0.0, 3.0, 5, 0.04) > base    # higher SR -> higher conf


def test_dsr_fail_closed_guards():
    assert dsr_confidence(0.1, 1, 0.0, 3.0, 5, 0.04) is None       # T<=1
    assert dsr_confidence(0.1, 252, 0.0, 3.0, 0, 0.04) is None     # N<1
    assert dsr_confidence(0.1, 252, 0.0, 3.0, 5, -0.01) is None    # negative variance
    assert dsr_confidence(float("nan"), 252, 0.0, 3.0, 5, 0.04) is None
    # denominator <= 0: large negative skew + high SR drives 1 - skew*SR + (k-1)/4*SR^2 negative
    assert dsr_confidence(5.0, 252, -3.0, 3.0, 1, 0.0) is None


def test_dsr_zero_variance_is_psr():
    # trial_sr_var=0 -> SR*=0 -> equals the N=1 PSR value
    assert dsr_confidence(0.1, 252, 0.0, 3.0, 9, 0.0) == pytest.approx(
        dsr_confidence(0.1, 252, 0.0, 3.0, 1, 0.04), abs=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_research_gates.py -k dsr -q`
Expected: FAIL — `ImportError: cannot import name 'dsr_confidence'`.

- [ ] **Step 3: Implement in `algua/research/gates.py`**

Add the scipy import after the existing imports:

```python
from scipy.stats import norm as _norm
```

Add constants after `MIN_HOLDOUT_OBSERVATIONS` (line 18):

```python
# DSR evidence layer (#211, Phase 1). Protected constants — relaxing them weakens the gate.
DSR_ALPHA = 0.05  # require >= 95% confidence the true Sharpe beats the selection-inflated benchmark
EULER_MASCHERONI = 0.5772156649015329  # gamma_E, the DSR expected-max weight (NOT e^-1)
```

Add the pure function (place it just below `sharpe_haircut`):

```python
def dsr_confidence(
    sr_obs_per_period: float,
    t: int,
    skew: float,
    raw_kurtosis: float,
    n_trials: int,
    trial_sr_var_per_period: float,
) -> float | None:
    """Deflated-Sharpe-Ratio confidence (Bailey & López de Prado): the probability — in [0,1],
    NOT a p-value — that the true (per-period) Sharpe exceeds the expected maximum Sharpe of
    ``n_trials`` selections.

        SR* = sqrt(var) * [ (1-gamma_E)*Z^-1(1-1/N) + gamma_E*Z^-1(1-1/(N*e)) ]   for N > 1
        SR* = 0                                                                    for N <= 1
        DSR = Phi( (SR_obs - SR*) * sqrt(T-1) / sqrt(1 - skew*SR_obs + (kurt-1)/4 * SR_obs^2) )

    ``raw_kurtosis`` is Pearson kurtosis (=3 for Gaussian), so the variance term reduces to the
    Lo/Mertens 1 + SR^2/2 for a normal series. Inputs are PER-PERIOD; the caller converts from the
    system's annualized Sharpes. Returns None (fail closed) on any degenerate input."""
    n = int(n_trials)
    if n < 1:                      # invalid breadth
        return None
    if t <= 1:                     # PSR needs sqrt(T-1) > 0; underpowered holdout
        return None
    if not math.isfinite(sr_obs_per_period) or not math.isfinite(skew) \
            or not math.isfinite(raw_kurtosis):
        return None
    if not math.isfinite(trial_sr_var_per_period) or trial_sr_var_per_period < 0.0:
        return None

    sr = sr_obs_per_period
    if n <= 1:
        sr_star = 0.0
    else:
        # E[max] of n trial Sharpes (Gaussian approximation), scaled by the trial-SR spread.
        sr_star = math.sqrt(trial_sr_var_per_period) * (
            (1.0 - EULER_MASCHERONI) * float(_norm.ppf(1.0 - 1.0 / n))
            + EULER_MASCHERONI * float(_norm.ppf(1.0 - 1.0 / (n * math.e)))
        )
    if not math.isfinite(sr_star):
        return None

    var_term = 1.0 - skew * sr + ((raw_kurtosis - 1.0) / 4.0) * sr * sr
    if not math.isfinite(var_term) or var_term <= 0.0:
        return None
    z = (sr - sr_star) * math.sqrt(t - 1) / math.sqrt(var_term)
    conf = float(_norm.cdf(z))
    return conf if math.isfinite(conf) else None
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_research_gates.py -k dsr -q`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/research/gates.py tests/test_research_gates.py
git commit -m "feat(211): pure dsr_confidence (PSR/DSR, Euler-Mascheroni, fail-closed guards)"
```

---

## Task 3: Wire the `dsr_evidence` binding check into `evaluate_gate` (PROTECTED)

**Files:**
- Modify: `algua/research/gates.py` (`GateDecision`, `evaluate_gate`)
- Test: `tests/test_research_gates.py`

**Interface:** `evaluate_gate` gains two keyword args:
`dsr_binding: bool = False` and `dsr_trial_var_ann: float | None = None`. When `dsr_binding` is
True, the gate computes `dsr_confidence` (converting the annualized holdout Sharpe and the
annualized trial variance to per-period via `ANN`) and appends a binding `dsr_evidence` check; a
`None` confidence fails that check closed. When `dsr_binding` is False, no `dsr_evidence` check is
appended (advisory/omitted) — the overall `passed` is unaffected. All DSR fields are recorded on
`GateDecision` either way.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_research_gates.py — assumes the repo's existing WalkForwardResult builder
# helper; if none exists, construct a minimal wf with holdout_metrics carrying the needed keys.
from algua.research.gates import GateCriteria, evaluate_gate


def _wf_with(holdout, stability):
    from algua.backtest.walkforward import WalkForwardResult
    return WalkForwardResult(
        strategy="s", config_hash="c", data_source="d", snapshot_id=None, timeframe="1d",
        seed=None, period={"start": "2020-01-01", "end": "2021-01-01"}, windows=4,
        holdout_frac=0.2, window_metrics=[], holdout_metrics=holdout, stability=stability)


# a passing-on-everything-but-DSR walk-forward
_GOOD_HOLDOUT = {"sharpe": 1.0, "total_return": 0.2, "n_bars": 252, "skewness": 0.0, "kurtosis": 3.0}
_GOOD_STAB = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}


def test_dsr_omitted_when_not_binding_does_not_change_passed():
    wf = _wf_with(_GOOD_HOLDOUT, _GOOD_STAB)
    d = evaluate_gate(wf, GateCriteria(), n_combos=10, pit_ok=True, dsr_binding=False)
    assert d.passed is True
    assert all(c["name"] != "dsr_evidence" for c in d.checks)
    assert d.dsr_binding is False and d.dsr_confidence is None


def test_dsr_binding_can_only_reject():
    wf = _wf_with(_GOOD_HOLDOUT, _GOOD_STAB)
    # huge trial dispersion + many trials -> SR* far above the holdout Sharpe -> DSR fails
    d = evaluate_gate(wf, GateCriteria(), n_combos=500, pit_ok=True,
                      dsr_binding=True, dsr_trial_var_ann=400.0)
    assert d.passed is False
    assert any(c["name"] == "dsr_evidence" and c["passed"] is False for c in d.checks)


def test_dsr_binding_missing_variance_fails_closed():
    wf = _wf_with(_GOOD_HOLDOUT, _GOOD_STAB)
    d = evaluate_gate(wf, GateCriteria(), n_combos=10, pit_ok=True,
                      dsr_binding=True, dsr_trial_var_ann=None)
    assert d.passed is False
    assert any(c["name"] == "dsr_evidence" and c["passed"] is False for c in d.checks)
    assert d.dsr_confidence is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_research_gates.py -k "dsr_omitted or dsr_binding" -q`
Expected: FAIL — `TypeError: evaluate_gate() got an unexpected keyword argument 'dsr_binding'`.

- [ ] **Step 3: Implement**

In `GateDecision` (after line 91, the `pit_override` field), add fields:

```python
    dsr_binding: bool = False
    dsr_confidence: float | None = None
    dsr_skip_reason: str | None = None
    dsr_sr_star: float | None = None
    dsr_n_trials: int | None = None
    dsr_trial_sr_var_ann: float | None = None
    dsr_t: int | None = None
    dsr_skew: float | None = None
    dsr_raw_kurtosis: float | None = None
```

In `GateDecision.to_dict` add these keys to the returned dict (non-finite floats nulled, mirroring
`effective_min_holdout_sharpe`):

```python
            "dsr_binding": self.dsr_binding,
            "dsr_confidence": self.dsr_confidence,
            "dsr_skip_reason": self.dsr_skip_reason,
            "dsr_sr_star": self.dsr_sr_star,
            "dsr_n_trials": self.dsr_n_trials,
            "dsr_trial_sr_var_ann": self.dsr_trial_sr_var_ann,
            "dsr_t": self.dsr_t,
            "dsr_skew": self.dsr_skew,
            "dsr_raw_kurtosis": self.dsr_raw_kurtosis,
```

Add the import of `ANN` (already imported at line 8) — confirm `from algua.backtest._constants import ANN` is present (it is).

In `evaluate_gate`, add the two parameters to the signature:

```python
    dsr_binding: bool = False,
    dsr_trial_var_ann: float | None = None,
```

After the PIT check block (after line 209, before the `return GateDecision(...)`), insert:

```python
    # DSR evidence (#211): a tighten-only AND-check, appended ONLY when binding (measured trial
    # dispersion is available). When not binding it is omitted entirely so `passed` is unchanged.
    # Unit conversion lives here: holdout Sharpe and trial variance are ANNUALIZED; DSR is per-period.
    dsr_conf: float | None = None
    dsr_sr_star: float | None = None
    dsr_skip_reason: str | None = None
    n_for_dsr = n_combos if n_combos is not None else 1
    t_hold = int(wf.holdout_metrics["n_bars"])
    skew = float(wf.holdout_metrics.get("skewness", 0.0))
    raw_kurt = float(wf.holdout_metrics.get("kurtosis", 3.0))
    sr_obs_ann = float(wf.holdout_metrics["sharpe"])
    if dsr_binding:
        var_pp = (dsr_trial_var_ann / ANN) if dsr_trial_var_ann is not None else None
        if var_pp is not None and math.isfinite(var_pp):
            dsr_conf = dsr_confidence(
                sr_obs_ann / math.sqrt(ANN), t_hold, skew, raw_kurt, n_for_dsr, var_pp)
        passed_dsr = dsr_conf is not None and dsr_conf >= (1.0 - DSR_ALPHA)
        checks.append({"name": "dsr_evidence",
                       "value": dsr_conf if (dsr_conf is not None and math.isfinite(dsr_conf)) else None,
                       "threshold": 1.0 - DSR_ALPHA, "op": ">=", "passed": bool(passed_dsr)})
        if dsr_conf is None:
            dsr_skip_reason = "no_dispersion"  # measured sweep exists but stats missing -> fail closed
    else:
        dsr_skip_reason = "no_measured_dispersion"
```

Pass the new fields into the `GateDecision(...)` constructor:

```python
        dsr_binding=bool(dsr_binding),
        dsr_confidence=dsr_conf,
        dsr_skip_reason=dsr_skip_reason,
        dsr_sr_star=dsr_sr_star,
        dsr_n_trials=(n_for_dsr if dsr_binding else None),
        dsr_trial_sr_var_ann=(dsr_trial_var_ann if dsr_binding else None),
        dsr_t=(t_hold if dsr_binding else None),
        dsr_skew=(skew if dsr_binding else None),
        dsr_raw_kurtosis=(raw_kurt if dsr_binding else None),
```

(Note: `dsr_sr_star` stays None in Phase 1's payload unless you also surface it from
`dsr_confidence`; it is declared for forward-compat and audit and may remain None — keep it None to
avoid duplicating the SR* computation outside the pure function.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_research_gates.py -q`
Expected: PASS (all gate tests, old and new).

- [ ] **Step 5: Add the strong tighten-only property test**

```python
# append to tests/test_research_gates.py
import itertools


def test_tighten_only_invariant():
    # new_pass == old_pass AND (not dsr_binding or dsr_pass), over a grid of decisions.
    for sharpe, nbars, binding, var in itertools.product(
            [0.2, 0.6, 1.2], [80, 252], [False, True], [None, 0.0, 4.0, 400.0]):
        holdout = {"sharpe": sharpe, "total_return": 0.1, "n_bars": nbars,
                   "skewness": 0.0, "kurtosis": 3.0}
        stab = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}
        wf = _wf_with(holdout, stab)
        old = evaluate_gate(wf, GateCriteria(), n_combos=20, pit_ok=True, dsr_binding=False)
        new = evaluate_gate(wf, GateCriteria(), n_combos=20, pit_ok=True,
                            dsr_binding=binding, dsr_trial_var_ann=var)
        dsr_check = next((c for c in new.checks if c["name"] == "dsr_evidence"), None)
        dsr_pass = (dsr_check is None) or dsr_check["passed"]
        assert new.passed == (old.passed and ((not binding) or dsr_pass))
```

Run: `uv run pytest tests/test_research_gates.py -k tighten_only -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add algua/research/gates.py tests/test_research_gates.py
git commit -m "feat(211): bind dsr_evidence into evaluate_gate (tighten-only, unit conversion, audit fields)"
```

---

## Task 4: `sweep()` records the trial-Sharpe `(count, mean, var)` triple

**Files:**
- Modify: `algua/backtest/sweep.py` (`SweepResult`, `sweep`)
- Test: `tests/test_sweep.py` (append; create if absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep.py — uses the repo's existing synthetic strategy + provider sweep fixtures.
# If a sweep helper already exists in the test suite, reuse it; otherwise adapt an existing
# test_sweep.py case. The assertion is on the new SweepResult fields only.
def test_sweep_records_trial_sharpe_triple(simple_sweepable_strategy, synthetic_provider, span):
    from algua.backtest.sweep import sweep
    res = sweep(simple_sweepable_strategy, synthetic_provider, span.start, span.end,
                grid={"window": [5, 10, 20]})
    assert res.trial_sharpe_count == 3
    assert res.trial_sharpe_mean is not None
    assert res.trial_sharpe_var_ann is not None and res.trial_sharpe_var_ann >= 0.0


def test_sweep_single_combo_var_zero(simple_sweepable_strategy, synthetic_provider, span):
    from algua.backtest.sweep import sweep
    res = sweep(simple_sweepable_strategy, synthetic_provider, span.start, span.end,
                grid={"window": [10]})
    assert res.trial_sharpe_count == 1
    assert res.trial_sharpe_var_ann == 0.0
```

(If the test suite lacks ready sweep fixtures, model these on the existing `tests/test_sweep.py`
setup — do not invent new fixtures.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_sweep.py -k trial_sharpe -q`
Expected: FAIL — `AttributeError: 'SweepResult' object has no attribute 'trial_sharpe_count'`.

- [ ] **Step 3: Implement**

Add fields to `SweepResult` (after `best` / before `code_hash`, line 157):

```python
    trial_sharpe_count: int = 0
    trial_sharpe_mean: float | None = None
    trial_sharpe_var_ann: float | None = None
```

In `sweep`, after `ranked = _rank_records(records)` (line 329), compute the triple from the
per-combo ranking scores (the Sharpe used for selection), keeping only finite scores:

```python
    # Trial-Sharpe dispersion for the DSR evidence layer (#211): variance of the per-combo
    # ranking Sharpes (annualized), in COMBO order. Finite scores only; ddof=1 for count>=2.
    finite_scores = [r["score"] for r in records if math.isfinite(r["score"])]
    t_count = len(finite_scores)
    if t_count >= 2:
        t_mean = float(np.mean(finite_scores))
        t_var = float(np.var(finite_scores, ddof=1))
    elif t_count == 1:
        t_mean, t_var = float(finite_scores[0]), 0.0
    else:
        t_mean, t_var = None, None
```

(`math` and `np` are already imported in sweep.py — confirm; `numpy` is imported as `np`.)

Pass them into the `SweepResult(...)` constructor (alongside the existing fields):

```python
        trial_sharpe_count=t_count,
        trial_sharpe_mean=t_mean,
        trial_sharpe_var_ann=t_var,
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_sweep.py -k trial_sharpe -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/sweep.py tests/test_sweep.py
git commit -m "feat(211): sweep records per-combo trial-Sharpe (count, mean, var) triple"
```

---

## Task 5: Schema 23→24 + `record_search_trial` stats + pooled-variance accessor

**Files:**
- Modify: `algua/registry/db.py` (`SCHEMA_VERSION`, `_SCHEMA` DDL, `migrate`)
- Modify: `algua/registry/repository.py` (Protocol)
- Modify: `algua/registry/store.py` (impl)
- Test: `tests/test_store.py` / `tests/test_registry_db.py` (append; match existing test file names)

- [ ] **Step 1: Write the failing tests**

```python
# append to the registry store test module
import math


def test_search_trials_records_and_pools_variance(tmp_repo):  # tmp_repo: a fresh SqliteStrategyRepository
    # two sweeps with different means -> pooled variance must exceed the mean of within-sweep vars
    tmp_repo.record_search_trial("s", 3, "{}", trial_sharpe_count=3,
                                 trial_sharpe_mean=0.2, trial_sharpe_var_ann=0.04)
    tmp_repo.record_search_trial("s", 2, "{}", trial_sharpe_count=2,
                                 trial_sharpe_mean=1.2, trial_sharpe_var_ann=0.04)
    pooled = tmp_repo.pooled_trial_sharpe_var("s")
    # exact pooled sample variance:
    # M = (3*0.2 + 2*1.2)/5 = 0.6 ; SSE = (2*0.04 + 3*(0.2-0.6)^2) + (1*0.04 + 2*(1.2-0.6)^2)
    #     = (0.08 + 0.48) + (0.04 + 0.72) = 1.32 ; pooled = 1.32/4 = 0.33
    assert pooled == math.isclose(pooled, 0.33, rel_tol=1e-9) or abs(pooled - 0.33) < 1e-9


def test_pooled_variance_equal_means_matches_naive(tmp_repo):
    tmp_repo.record_search_trial("s", 3, "{}", trial_sharpe_count=3,
                                 trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    tmp_repo.record_search_trial("s", 2, "{}", trial_sharpe_count=2,
                                 trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.10)
    # equal means -> between-sweep term zero -> pooled = ((3-1)*0.04+(2-1)*0.10)/(5-1) = 0.045
    assert abs(tmp_repo.pooled_trial_sharpe_var("s") - 0.045) < 1e-9


def test_pooled_variance_none_when_any_stat_missing(tmp_repo):
    tmp_repo.record_search_trial("s", 3, "{}", trial_sharpe_count=3,
                                 trial_sharpe_mean=0.2, trial_sharpe_var_ann=0.04)
    tmp_repo.record_search_trial("s", 2, "{}")  # old-style row: NULL stats
    assert tmp_repo.pooled_trial_sharpe_var("s") is None


def test_pooled_variance_none_when_no_rows(tmp_repo):
    assert tmp_repo.pooled_trial_sharpe_var("nope") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest <that test module> -k "pooled or records_and_pools" -q`
Expected: FAIL — `TypeError: record_search_trial() got an unexpected keyword argument 'trial_sharpe_count'`.

- [ ] **Step 3: Implement — `db.py`**

Bump `SCHEMA_VERSION = 23` → `24` (line 16; use current+1 if the pre-flight found it already moved).

Add the three columns to the `migrate()` body (after the existing `_add_missing_columns` calls,
before the `user_version` stamp):

```python
    _add_missing_columns(conn, "search_trials", {
        "trial_sharpe_count": "INTEGER",
        "trial_sharpe_mean": "REAL",
        "trial_sharpe_var_ann": "REAL",
    })
```

(The `_SCHEMA` `CREATE TABLE search_trials` is for fresh DBs — also add the three columns there so a
brand-new DB has them without relying on the ALTER path:)

```sql
CREATE TABLE IF NOT EXISTS search_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    n_combos INTEGER NOT NULL,
    grid_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    trial_sharpe_count INTEGER,
    trial_sharpe_mean REAL,
    trial_sharpe_var_ann REAL
);
```

- [ ] **Step 4: Implement — `repository.py` Protocol**

Change the `record_search_trial` signature (line 189) and add the accessor:

```python
    def record_search_trial(
        self, strategy_name: str, n_combos: int, grid_json: str,
        *, trial_sharpe_count: int | None = None,
        trial_sharpe_mean: float | None = None,
        trial_sharpe_var_ann: float | None = None,
    ) -> int:
        """Persist one measured search-breadth row (size + grid + the sweep's trial-Sharpe
        (count, mean, annualized var) for the #211 DSR dispersion); return its row id. Keyed by
        strategy NAME. Stats default to None for callers that record only breadth."""
        ...

    def pooled_trial_sharpe_var(self, strategy_name: str) -> float | None:
        """Exact pooled SAMPLE variance (ddof=1) of the strategy's trial Sharpes across all its
        search_trials rows, via the law of total variance over the per-row (count, mean, var)
        triples. Returns None (fail closed) if there are no rows OR any contributing row has a
        NULL/NaN/inf count/mean/var. ANNUALIZED units (caller converts)."""
        ...
```

- [ ] **Step 5: Implement — `store.py`**

Replace `record_search_trial` (lines 398–407):

```python
    def record_search_trial(
        self, strategy_name: str, n_combos: int, grid_json: str,
        *, trial_sharpe_count: int | None = None,
        trial_sharpe_mean: float | None = None,
        trial_sharpe_var_ann: float | None = None,
    ) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at,"
                " trial_sharpe_count, trial_sharpe_mean, trial_sharpe_var_ann)"
                " VALUES (?,?,?,?,?,?,?)",
                (strategy_name, n_combos, grid_json, _now(),
                 trial_sharpe_count, trial_sharpe_mean, trial_sharpe_var_ann),
            )
        rowid = cur.lastrowid
        assert rowid is not None
        return rowid

    def pooled_trial_sharpe_var(self, strategy_name: str) -> float | None:
        rows = self._conn.execute(
            "SELECT trial_sharpe_count AS n, trial_sharpe_mean AS mean,"
            " trial_sharpe_var_ann AS var FROM search_trials WHERE strategy_name=?",
            (strategy_name,),
        ).fetchall()
        if not rows:
            return None
        triples: list[tuple[int, float, float]] = []
        for r in rows:
            n, mean, var = r["n"], r["mean"], r["var"]
            if n is None or mean is None or var is None:
                return None
            if not (math.isfinite(mean) and math.isfinite(var)) or int(n) < 1 or var < 0.0:
                return None
            triples.append((int(n), float(mean), float(var)))
        total_n = sum(n for n, _, _ in triples)
        if total_n <= 1:
            return 0.0
        grand_mean = sum(n * m for n, m, _ in triples) / total_n
        sse = sum((n - 1) * v + n * (m - grand_mean) ** 2 for n, m, v in triples)
        return sse / (total_n - 1)
```

Add `import math` to `store.py` if not present (check the top of the file; add under the existing
imports if missing).

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest <registry store test module> -k "pooled or records_and_pools" -q`
Expected: PASS.

- [ ] **Step 7: Schema-migration test**

```python
def test_migration_adds_trial_sharpe_columns_idempotent(tmp_path):
    # open twice to prove the ALTER path is idempotent and NULL on pre-existing rows
    from algua.registry.db import connect  # use the repo's actual connect/open helper
    db = tmp_path / "r.db"
    c1 = connect(str(db)); c1.close()
    c2 = connect(str(db))
    cols = {row["name"] for row in c2.execute("PRAGMA table_info(search_trials)")}
    assert {"trial_sharpe_count", "trial_sharpe_mean", "trial_sharpe_var_ann"} <= cols
    assert c2.execute("PRAGMA user_version").fetchone()[0] == 24
    c2.close()
```

(Use the repo's real DB-open helper name — check how other `tests/test_registry_db.py` cases open a
connection; mirror that.)

Run: `uv run pytest <db test module> -k migration_adds_trial_sharpe -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add algua/registry/db.py algua/registry/repository.py algua/registry/store.py tests/
git commit -m "feat(211): schema 24 + trial-Sharpe stats + pooled-sample-variance accessor"
```

---

## Task 6: CLI passes the triple into `record_search_trial`

**Files:**
- Modify: `algua/cli/backtest_cmd.py:205`
- Test: covered by an integration assertion in Task 7; add a focused CLI test if `tests/test_backtest_cmd.py` exists.

- [ ] **Step 1: Implement**

Change `algua/cli/backtest_cmd.py:205` from:

```python
        repo.record_search_trial(name, result.n_combos, json.dumps(result.grid, sort_keys=True))
```

to:

```python
        repo.record_search_trial(
            name, result.n_combos, json.dumps(result.grid, sort_keys=True),
            trial_sharpe_count=result.trial_sharpe_count,
            trial_sharpe_mean=result.trial_sharpe_mean,
            trial_sharpe_var_ann=result.trial_sharpe_var_ann,
        )
```

- [ ] **Step 2: Verify nothing regressed**

Run: `uv run pytest tests/ -k "backtest_cmd or sweep_cmd" -q`
Expected: PASS (or no tests collected — then rely on Task 7's end-to-end coverage).

- [ ] **Step 3: Commit**

```bash
git add algua/cli/backtest_cmd.py
git commit -m "feat(211): record sweep trial-Sharpe stats from the sweep CLI"
```

---

## Task 7: `run_gate` assembles DSR inputs + binding decision (PROTECTED)

**Files:**
- Modify: `algua/registry/promotion.py` (`run_gate`)
- Test: `tests/test_promotion.py` (append)

**Binding decision (the spec's actor-independent rule):** DSR binds iff the strategy has measured
breadth — i.e. `breadth.provenance == "measured"`. Then the pooled dispersion is fetched; if it is
`None` (old rows lacking stats) the binding check fails closed. When provenance is `"declared"`
(human, no sweep) DSR is omitted (`dsr_binding=False`).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_promotion.py — reuse the module's existing promotion harness
# (a repo + a walk-forward result + breadth context). Mirror an existing run_gate test's setup.

def test_run_gate_agent_measured_binds_dsr(promotion_env):
    env = promotion_env  # provides repo, a passing wf, measured BreadthContext, etc.
    env.repo.record_search_trial(env.name, 5, "{}", trial_sharpe_count=5,
                                 trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    outcome = env.run_gate(provenance="measured")
    d = outcome.decision
    assert d.dsr_binding is True
    assert any(c["name"] == "dsr_evidence" for c in d.checks)


def test_run_gate_declared_breadth_omits_dsr(promotion_env):
    env = promotion_env
    outcome = env.run_gate(provenance="declared")
    d = outcome.decision
    assert d.dsr_binding is False
    assert all(c["name"] != "dsr_evidence" for c in d.checks)
    assert d.dsr_skip_reason == "no_measured_dispersion"


def test_run_gate_measured_but_missing_stats_fails_closed(promotion_env):
    env = promotion_env
    env.repo.record_search_trial(env.name, 5, "{}")  # measured row, NULL stats (pre-migration)
    outcome = env.run_gate(provenance="measured")
    d = outcome.decision
    assert d.dsr_binding is True
    assert any(c["name"] == "dsr_evidence" and c["passed"] is False for c in d.checks)
    assert d.passed is False
```

(Adapt `promotion_env`/`run_gate(...)` to the actual fixtures in `tests/test_promotion.py`. The
behaviors asserted — binds on measured, omits on declared, fails-closed on measured-but-missing —
are the contract; wire them to the real harness.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_promotion.py -k "dsr or run_gate_agent or declared_breadth" -q`
Expected: FAIL (DSR not yet threaded; `dsr_binding` always False).

- [ ] **Step 3: Implement**

In `algua/registry/promotion.py`, inside `run_gate`, after `pit_ok = resolve_pit_ok(...)` (line 172)
and before `evaluate_gate(...)`, compute the binding inputs:

```python
    dsr_binding = breadth.provenance == "measured"
    dsr_trial_var_ann = repo.pooled_trial_sharpe_var(name) if dsr_binding else None
```

Pass them into the `evaluate_gate(...)` call (extend the existing call, lines 174–178):

```python
    decision = evaluate_gate(
        wf, criteria, n_combos=breadth.n_funnel, breadth_provenance=breadth.provenance,
        pit_ok=pit_ok, allow_non_pit=allow_non_pit, own_lifetime_combos=breadth.own,
        windowed_total_combos=breadth.windowed_total, funnel_window_days=FUNNEL_WINDOW_DAYS,
        dsr_binding=dsr_binding, dsr_trial_var_ann=dsr_trial_var_ann,
    )
```

The DSR fields already flow into `decision.to_dict()` → the `decision_json` column of the
`gate_evaluations` row (Task 3), so no new `record_gate_evaluation` columns are needed.

- [ ] **Step 4: Add the re-sweep message to the gate reason (optional but helpful)**

In `_gate_reason` (or where the failed-gate message is surfaced), no change is strictly required —
`dsr_evidence` renders as `name=fail` via the existing boolean-check branch. If you want the
re-sweep hint, append when `decision.dsr_binding and decision.dsr_confidence is None`:
`"; dsr_evidence failed: no recorded trial-Sharpe dispersion — re-run `backtest sweep` to record it"`.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_promotion.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add algua/registry/promotion.py tests/test_promotion.py
git commit -m "feat(211): run_gate binds DSR on measured breadth, omits on declared, fails closed on missing stats"
```

---

## Task 8: Explicit scipy dependency + full quality gate

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add scipy to project dependencies**

Inspect the dependencies block: `grep -n "dependencies" pyproject.toml`. Add `scipy` to the runtime
dependency list (pin to a floor matching the installed 1.17.x, e.g. `"scipy>=1.11"`), matching the
existing version-spec style in that file.

- [ ] **Step 2: Sync and run the full quality gate**

```bash
uv sync
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
```

Expected: all green. If `mypy` flags `scipy` stubs, confirm whether the repo already ignores
untyped third-party imports (check `pyproject.toml`/`mypy` config); if not, add
`scipy.*` to the existing ignore-missing-imports section (do NOT broaden beyond scipy).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(211): add explicit scipy dependency for DSR confidence"
```

---

## Self-Review (run before handing off)

**Spec coverage:** PSR/DSR formula (Task 2) ✓; Euler–Mascheroni + N≤1 guard + confidence-not-pvalue
(Task 2) ✓; raw kurtosis + same-series moments (Task 1) ✓; denominator/finiteness/negative-var
guards (Task 2) ✓; tighten-only AND-check + advisory-omit (Task 3) ✓; exact pooled-sample-variance
+ NULL→None fail-closed (Task 5) ✓; `(count, mean, var)` triple from sweep (Task 4) ✓; schema 23→24
(Task 5) ✓; binding actor-independent on measured breadth, fail-closed on missing stats (Task 7) ✓;
audit fields in payload (Task 3) ✓; scipy dep + unit-conversion-in-gates (Tasks 2/3/8) ✓.

**Deferred (NOT in this plan, per spec):** variance floor; full per-combo distribution storage;
LORD++/FDR ledger (Phase 2); dependence calibration (Phase 3); family budgets (Phase 4); haircut
retirement (end-state). File these as follow-up issues from the spec's "Deferred phases".

**Type consistency:** `dsr_confidence(sr_obs_per_period, t, skew, raw_kurtosis, n_trials,
trial_sr_var_per_period)` is called once, from `evaluate_gate`, with per-period conversions applied.
`evaluate_gate(..., dsr_binding, dsr_trial_var_ann)` matches the `run_gate` call. `record_search_trial`
keyword args (`trial_sharpe_count/mean/var_ann`) match across Protocol, store, sweep result, and CLI.
`pooled_trial_sharpe_var(name) -> float | None` matches its `run_gate` use.
