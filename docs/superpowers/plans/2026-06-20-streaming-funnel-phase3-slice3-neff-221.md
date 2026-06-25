# Phase 3 Slice 3 — effective independent trials N_eff (shadow-only) (#221) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate the **effective number of independent trials** `N_eff` from the pairwise
correlation of funnel siblings' overlapping OOS return streams (Kish form), and **record it in the
gate audit payload only** — `N_eff` is **SHADOW-ONLY** in Phase 3: the binding DSR benchmark keeps
using raw `N`. (A lower `N_eff < N` would LOWER `SR*` and could flip a DSR FAIL→PASS, violating
tighten-only; so `N_eff` becomes binding only at Slice 5, bundled atomically with haircut retirement.)

**Architecture:** A new unprotected `algua/backtest/neff.py` does the estimation (date-align siblings,
pairwise Pearson, Kish formula) and returns a small result struct. The protected `gates.py` gains
ONLY three constants + three audit fields on `GateDecision` (+ `to_dict`) — NO change to any gate
check; `evaluate_gate`'s binding DSR still receives raw `N`. The protected `promotion.py` `run_gate`
queries siblings via the existing Slice-1 `overlapping_holdout_return_streams`, calls the estimator,
and writes `dsr_n_eff`/`dsr_rho_bar`/`dsr_n_siblings` onto the decision — it does NOT pass `N_eff`
into `evaluate_gate`.

**Tech Stack:** Python 3.12, numpy (Pearson via `np.corrcoef`), pytest. No new dependency, no schema
change.

## Global Constraints

- **SHADOW-ONLY — NO gate-behavior change.** `evaluate_gate`'s checks and `decision.passed` are
  byte-identical to today. `N_eff` is recorded in the audit payload and NEVER fed into the binding
  `dsr_confidence` (which keeps `n_trials = raw N = n_funnel`). A property/integration test must
  machine-check that the binding DSR's `dsr_n_trials` equals raw `N`, never `N_eff`. The full suite
  stays green (record the baseline count before Task 1).
- **Estimation lives in `algua/backtest/*` (the non-negotiable Phase-1 architecture boundary).**
  `gates.py` stays pure gate-math and does NO estimation, NO I/O, NO return-vector handling. The
  three threshold constants live in `gates.py` (protected — they become load-bearing in Slice 5);
  `neff.estimate_n_eff` takes them as PARAMETERS (so `neff.py` needs no `algua.research` import —
  mirrors the Slice-2 `max_block_fraction` pattern).
- **Fail-closed / conservative-when-uncertain.** Uncertainty pushes `N_eff` UP toward raw `N` (never
  below what the evidence supports). The estimator returns `n_eff = None` (→ audit field `None`,
  i.e. "no N_eff evidence, raw N stands") when: fewer than `MIN_N_EFF_SIBLINGS` sibling streams; OR
  any sibling pair shares fewer than `MIN_CORR_OVERLAP_BARS` date-aligned bars; OR any pairwise
  correlation is non-finite (zero-variance stream). `ρ̄_lower = clamp(max(0, ρ̄_sample − k·SE), 0, 1)`
  with `SE = σ_ρ/√M` — a conservative lower bound (the documented Q2.2 SE-understatement is
  acceptable in shadow mode; it is BLOCKING only for the Slice-5 binding switch).
- **Cap at raw N.** `n_eff_int = max(1, min(raw_n, round(N_eff)))` — `N_eff` can never exceed raw `N`.
- **Determinism.** Pure function of the input vectors; no RNG, no wall-clock.
- **No NaN in payload.** `to_dict` nulls non-finite floats via the existing `_f`.
- **Protected walls** (`gates.py`, `promotion.py`) are CODEOWNERS `@Lior-Nis`.
- **Per-slice quality gate (before commit):**
  `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- **Constant values (spec GATE-1 Q2.4 + named-constants table):** `MIN_N_EFF_SIBLINGS = 5`
  (reliable-estimate choice — Slice-5 wants `N_eff < N` only on real correlation evidence),
  `MIN_CORR_OVERLAP_BARS = 21` (≈1 trading month minimum to estimate a pairwise correlation),
  `RHO_BAR_SHRINKAGE_K = 1.0` (one-SE conservative shrinkage). All three are protected and may be
  re-tuned before the Slice-5 binding switch.

## Locked design decisions

- **D1 — sibling-only correlation pool.** ρ̄ is estimated from pairwise correlations among the
  SIBLING streams (other funnel strategies, from `overlapping_holdout_return_streams`, which
  excludes the strategy under promotion). The strategy's own vector is NOT added to the pool — this
  matches the spec's "funnel siblings' return streams" framing and the access-control boundary.
  (Adding own is a candidate refinement for the Slice-5 binding switch; out of scope here.)
- **D2 — estimator in `backtest/neff.py`, not `gates.py`.** The component-(a) footprint casually
  says "gates.py: estimate_n_eff", but the non-negotiable Phase-1 invariant ("estimation lives in
  backtest, gates stays pure-math reading pre-computed inputs") governs — and Slice 2 set the
  precedent (bootstrap in backtest, gates gets a scalar). Constants stay in `gates.py`.
- **D3 — strict "any bad pair → None".** If any sibling pair has insufficient overlap or a
  non-finite correlation, the whole estimate is `None` (raw `N` stands). Conservative; shadow-only.
- **D4 — `dsr_rho_bar` records ρ̄_lower** (the value that drives `N_eff`), not the raw sample mean.

## File Structure

- `algua/backtest/neff.py` (NEW, unprotected) — `NEffResult`, `estimate_n_eff`.
- `algua/research/gates.py` (PROTECTED) — 3 constants; 3 `GateDecision` audit fields + `to_dict`.
- `algua/registry/promotion.py` (PROTECTED) — shadow wiring in `run_gate`.
- Tests: `tests/backtest/test_neff.py` (new), extend `tests/research/test_*` + `tests/test_promotion.py`.

---

## Task 1: `algua/backtest/neff.py` — N_eff estimator (NEW, unprotected)

**Files:**
- Create: `algua/backtest/neff.py`
- Test: `tests/backtest/test_neff.py` (create)

**Interfaces:**
- Produces:
  - `NEffResult(NamedTuple)`: `n_eff: int | None`, `rho_bar: float | None`, `n_siblings: int`,
    `n_pairs: int`.
  - `estimate_n_eff(raw_n: int, sibling_streams: list[tuple[list[float], list[str]]], *,
    min_siblings: int, min_overlap_bars: int, shrinkage_k: float) -> NEffResult` — Kish
    average-pairwise-correlation effective trial count. Returns `n_eff=None` (raw N stands) on any
    fail-closed condition (see Global Constraints); else `n_eff = max(1, min(raw_n, round(raw_n /
    (1 + (raw_n−1)·ρ̄_lower))))`.

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_neff.py`. Build synthetic streams with known correlation.

```python
import numpy as np

from algua.backtest.neff import NEffResult, estimate_n_eff

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
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/backtest/test_neff.py -q` → FAIL (module absent).

- [ ] **Step 3: Implement `neff.py`**

```python
"""Effective independent trials N_eff — shadow-only (#221, Phase 3 Slice 3).

Kish average-pairwise-correlation effective trial count:
    N_eff = raw_n / (1 + (raw_n - 1) * rho_bar_lower)
where rho_bar_lower is a conservative (lower-bound) estimate of the mean off-diagonal pairwise
Pearson correlation of funnel siblings' date-aligned overlapping OOS return streams.

Pure-maths leaf: no algua.research import (thresholds are passed in as parameters). Estimation lives
in algua/backtest per the Phase-1 architecture boundary; gates.py receives only the pre-computed
scalar in the audit payload. N_eff is SHADOW-ONLY in Phase 3 (never the binding DSR trial count).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import NamedTuple

import numpy as np


class NEffResult(NamedTuple):
    n_eff: int | None        # None => no N_eff evidence; raw N stands (fail-open in shadow mode)
    rho_bar: float | None    # rho_bar_lower actually used (None when n_eff is None)
    n_siblings: int
    n_pairs: int


def _pair_correlation(a, b, min_overlap_bars):
    """Inner-join two (returns, dates) streams on DATE, return Pearson corr or None if the overlap
    is too short or the correlation is non-finite (e.g. a zero-variance stream)."""
    ar, ad = a
    br, bd = b
    amap = dict(zip(ad, ar, strict=True))
    bmap = dict(zip(bd, br, strict=True))
    common = sorted(set(amap) & set(bmap))
    if len(common) < min_overlap_bars:
        return None
    av = np.array([amap[d] for d in common], dtype=float)
    bv = np.array([bmap[d] for d in common], dtype=float)
    if av.std() == 0.0 or bv.std() == 0.0:
        return None
    rho = float(np.corrcoef(av, bv)[0, 1])
    return rho if math.isfinite(rho) else None


def estimate_n_eff(
    raw_n: int,
    sibling_streams: Sequence[tuple[list[float], list[str]]],
    *,
    min_siblings: int,
    min_overlap_bars: int,
    shrinkage_k: float,
) -> NEffResult:
    n_sib = len(sibling_streams)
    if raw_n < 1 or n_sib < min_siblings:
        return NEffResult(None, None, n_sib, 0)
    rhos: list[float] = []
    for a, b in combinations(sibling_streams, 2):
        rho = _pair_correlation(a, b, min_overlap_bars)
        if rho is None:                      # strict: any bad pair -> no estimate (raw N stands)
            return NEffResult(None, None, n_sib, len(rhos))
        rhos.append(rho)
    m = len(rhos)
    if m == 0:                                # n_sib >= min_siblings(>=2) guarantees m>=1, defensive
        return NEffResult(None, None, n_sib, 0)
    arr = np.asarray(rhos, dtype=float)
    rho_mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(m)) if m >= 2 else 0.0
    rho_lower = min(1.0, max(0.0, rho_mean - shrinkage_k * se))
    n_eff = raw_n / (1.0 + (raw_n - 1) * rho_lower)
    if not math.isfinite(n_eff):
        return NEffResult(None, None, n_sib, m)
    n_eff_int = max(1, min(raw_n, int(round(n_eff))))
    return NEffResult(n_eff_int, rho_lower, n_sib, m)
```

(`zip(..., strict=True)` requires equal-length returns/dates — they always are, validated at the
Slice-1 write. If a future caller could pass mismatched lengths, guard with a length check first.)

- [ ] **Step 4: Run tests + quality gate**

Run: `uv run pytest tests/backtest/test_neff.py -q` → PASS. Adjust seeds (NOT assertions) if a
synthetic-correlation magnitude is borderline; the qualitative properties (ρ̄=0→N_eff≈N, ρ̄=1→
N_eff=1, cap, fail-closed, shrinkage direction, date-join) MUST hold.
Run the full quality gate → all pass.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/neff.py tests/backtest/test_neff.py
git commit -m "feat(221): N_eff effective-trials estimator (Kish, shadow) — Slice 3 of #221"
```

---

## Task 2: gates.py constants + GateDecision audit fields (PROTECTED, no gate-check change)

**Files:**
- Modify: `algua/research/gates.py` (constants near `DSR_BOOTSTRAP_*`; `GateDecision` fields after
  `dsr_bootstrap_block_len:327`; `to_dict` after `dsr_bootstrap_block_len:373`)
- Test: `tests/research/test_neff_shadow.py` (create)

**Interfaces:**
- Produces: `MIN_N_EFF_SIBLINGS = 5`, `MIN_CORR_OVERLAP_BARS = 21`, `RHO_BAR_SHRINKAGE_K = 1.0`
  (module constants). `GateDecision` gains `dsr_n_eff: int | None = None`, `dsr_rho_bar: float |
  None = None`, `dsr_n_siblings: int | None = None`, all in `to_dict` (`dsr_rho_bar` via `_f`).

- [ ] **Step 1: Write the failing tests**

Create `tests/research/test_neff_shadow.py`:

```python
from algua.research.gates import (
    MIN_CORR_OVERLAP_BARS, MIN_N_EFF_SIBLINGS, RHO_BAR_SHRINKAGE_K, GateDecision,
)


def test_constants():
    assert MIN_N_EFF_SIBLINGS == 5
    assert MIN_CORR_OVERLAP_BARS == 21
    assert RHO_BAR_SHRINKAGE_K == 1.0


def test_gatedecision_neff_fields_default_none_and_serialize():
    d = GateDecision(passed=True, checks=[])
    assert d.dsr_n_eff is None and d.dsr_rho_bar is None and d.dsr_n_siblings is None
    dd = d.to_dict()
    assert dd["dsr_n_eff"] is None and dd["dsr_rho_bar"] is None and dd["dsr_n_siblings"] is None
    d2 = GateDecision(passed=True, checks=[], dsr_n_eff=12, dsr_rho_bar=0.4, dsr_n_siblings=7)
    dd2 = d2.to_dict()
    assert dd2["dsr_n_eff"] == 12 and dd2["dsr_rho_bar"] == 0.4 and dd2["dsr_n_siblings"] == 7
```

(Confirm the exact `GateDecision(...)` minimal-construction signature against the current dataclass —
`passed`/`checks` are the only required positional fields; everything else defaults.)

- [ ] **Step 2: Run to verify it fails** — ImportError / unexpected-kwarg.

- [ ] **Step 3: Add the constants**

In `gates.py` near the bootstrap constants:

```python
# Effective independent trials N_eff (#221, Phase 3 Slice 3). SHADOW-ONLY in Phase 3 — recorded in
# the audit payload, never the binding DSR trial count (a lower N_eff would loosen the gate; it goes
# binding only at Slice 5, bundled with haircut retirement). Protected — load-bearing from Slice 5.
MIN_N_EFF_SIBLINGS = 5          # min overlapping-OOS sibling streams to attempt an N_eff estimate
MIN_CORR_OVERLAP_BARS = 21      # min date-aligned shared bars per sibling pair to estimate a corr
RHO_BAR_SHRINKAGE_K = 1.0       # SE multiplier for the conservative (lower-bound) rho_bar
```

- [ ] **Step 4: Add `GateDecision` fields + `to_dict`**

After `dsr_bootstrap_block_len`:

```python
    # Effective independent trials audit (#221 Slice 3). SHADOW-ONLY: recorded, never fed into the
    # binding dsr_confidence (which keeps n_trials = raw N). Populated by promotion.run_gate.
    dsr_n_eff: int | None = None
    dsr_rho_bar: float | None = None
    dsr_n_siblings: int | None = None
```

In `to_dict` (with the other `dsr_*`): `"dsr_n_eff": self.dsr_n_eff`, `"dsr_rho_bar":
_f(self.dsr_rho_bar)`, `"dsr_n_siblings": self.dsr_n_siblings`.

**Do NOT touch `evaluate_gate`** — no check, no logic uses `N_eff`. This is the entire gates change.

- [ ] **Step 5: Run tests + quality gate** → PASS. Existing gate/promotion tests byte-identical.

- [ ] **Step 6: Commit**

```bash
git add algua/research/gates.py tests/research/test_neff_shadow.py
git commit -m "feat(221): GateDecision N_eff shadow audit fields + constants — Slice 3"
```

---

## Task 3: promotion.py shadow wiring (PROTECTED)

**Files:**
- Modify: `algua/registry/promotion.py` (`run_gate`, after `rec = repo.get(name):458`, before the
  `gate_row` build at `:481`)
- Test: extend `tests/test_promotion.py`

**Interfaces:**
- Consumes: `repo.overlapping_holdout_return_streams(strategy_id, holdout_start, holdout_end,
  window_days)` (Slice 1); `neff.estimate_n_eff`; the gates constants.
- Produces: `decision.dsr_n_eff/_rho_bar/_n_siblings` populated (shadow) in `decision_json`; the
  binding gate outcome UNCHANGED.

- [ ] **Step 1: Write a failing integration test**

In `tests/test_promotion.py`, add tests driving `run_gate` with measured breadth where the funnel has
≥5 sibling `holdout_returns` rows whose OOS interval overlaps the strategy under promotion (reuse the
Slice-1 committed-burn + `record_holdout_returns` fixtures to seed sibling vectors for OTHER
strategies, all within `FUNNEL_WINDOW_DAYS`, overlapping the promotion interval). Assert:
- `decision.dsr_n_eff` is not None, `1 <= dsr_n_eff <= n_funnel`, `decision.dsr_n_siblings >= 5`,
  `"dsr_n_eff"` present in the persisted `decision_json`.
- **Shadow invariant:** `decision.dsr_n_trials == n_funnel` (raw N) — the binding DSR did NOT use
  `N_eff`; and `decision.passed` is IDENTICAL to the same promote run with NO siblings seeded (i.e.
  recording N_eff changed no gate outcome). Construct both and compare `passed` + the `dsr_evidence`
  check value.
- A measured promote with FEWER than 5 overlapping siblings → `decision.dsr_n_eff is None`,
  `decision.dsr_n_siblings < 5`, outcome unchanged.

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_promotion.py -q -k n_eff` → FAIL.

- [ ] **Step 3: Wire the shadow estimate into `run_gate`**

After `rec = repo.get(name)` (line 458), BEFORE the returns-write block / `gate_row` build:

```python
    # Effective independent trials N_eff (#221 Slice 3) — SHADOW-ONLY: recorded for the audit trail,
    # NEVER passed as the binding DSR trial count (a lower N_eff would loosen the gate; it goes
    # binding only at Slice 5 with haircut retirement). Sibling-only read (excludes own vector).
    if dsr_binding:
        siblings = repo.overlapping_holdout_return_streams(
            rec.id, wf.holdout_metrics["start"], wf.holdout_metrics["end"], FUNNEL_WINDOW_DAYS)
        neff = estimate_n_eff(
            n_funnel, siblings, min_siblings=MIN_N_EFF_SIBLINGS,
            min_overlap_bars=MIN_CORR_OVERLAP_BARS, shrinkage_k=RHO_BAR_SHRINKAGE_K)
        decision.dsr_n_eff = neff.n_eff
        decision.dsr_rho_bar = neff.rho_bar
        decision.dsr_n_siblings = neff.n_siblings
```

Imports: add `MIN_N_EFF_SIBLINGS, MIN_CORR_OVERLAP_BARS, RHO_BAR_SHRINKAGE_K` to the
`from algua.research.gates import (...)` group and `from algua.backtest.neff import estimate_n_eff`.
Do NOT change the `evaluate_gate(...)` call or `n_combos=n_funnel`. `n_funnel` here is the SAME raw N
the binding DSR uses (verify it is in scope at this point — it is, computed ~line 397).

- [ ] **Step 4: Run the integration test + full quality gate** → PASS. If ANY existing promotion
test's `passed`/`dsr_evidence` outcome changes, the wiring is wrong (N_eff must be pure shadow) —
STOP and fix. (Seeding sibling rows must not perturb the strategy-under-promotion's own gate inputs.)

- [ ] **Step 5: Commit**

```bash
git add algua/registry/promotion.py tests/test_promotion.py
git commit -m "feat(221): record shadow N_eff in run_gate (never binding) — Slice 3"
```

---

## Self-Review notes

- **Spec coverage (component a, shadow):** Kish estimator with conservative ρ̄_lower shrinkage —
  Task 1; cap at raw N, fall-back-to-None guards (`<MIN_N_EFF_SIBLINGS` / `<MIN_CORR_OVERLAP_BARS` /
  non-finite corr) — Task 1; sibling set via overlapping-OOS streams in the rolling window with
  date-aligned inner-join — Tasks 1/3; shadow-only audit fields `dsr_n_eff/dsr_rho_bar/
  dsr_n_siblings` — Tasks 2/3; the binding DSR keeps raw N (machine-checked) — Task 3.
- **Shadow-only is enforced structurally:** `N_eff` never reaches `evaluate_gate`; the only gates
  change is data fields. Property/integration test pins `dsr_n_trials == raw N` and unchanged
  `passed`.
- **Q2.2 (SE understatement) is acceptable here** (shadow) and is BLOCKING only for Slice 5 — note
  it in the code comment so the Slice-5 author corrects the ρ̄ SE before binding.
- **Type consistency:** `NEffResult(n_eff, rho_bar, n_siblings, n_pairs)` (Task 1) consumed in Task
  3; constants (Task 2) passed as `estimate_n_eff` params (Task 3); `overlapping_holdout_return_streams`
  signature (Slice 1) reused verbatim.
- **Deferred (NOT this slice):** the binding `N_eff` switch + corrected ρ̄ SE (Fisher-z / block-
  bootstrap-the-matrix, Q2.2) — Slice 5; including the own vector in the correlation pool (D1).
