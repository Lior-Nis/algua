# Phase 3 Slice 4 — multi-regime robustness (#221) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require a promoted strategy to clear a deliberately-relaxed per-regime Sharpe bar in EACH
sufficiently-long market-volatility regime of its OOS holdout — a tighten-only AND-check
(`regime_robustness`) — so a strategy cannot pass on an aggregate Sharpe that one benign regime
carries over a bad one. Regimes are **market-volatility tertiles** computed from a market/benchmark
return series (NOT the strategy's own returns — that would be circular). Plus the spec-mandated
**dominance-audit predeclaration** scaffolding that must land in this slice (the last tightening
slice) so the haircut-retirement audit can accumulate real-traffic data with pre-committed thresholds.

**Architecture:** `walk_forward` computes the benchmark series — the **equal-weighted cross-sectional
daily return** of the strategy's universe (Q4.2 source: `provider.get_bars` → adj_close panel →
`pct_change().mean(axis=1)`, same `provider`/snapshot as the backtest ⇒ PIT-identical) — and attaches
it as `WalkForwardResult.market_returns` (mirroring `holdout_returns`). The protected `gates.py` gains
PURE helpers `regime_splits`/`regime_robustness_check` (rolling-vol tertiles + per-regime Sharpe), a
`regime_robustness` AND-check in `evaluate_gate`, audit fields, the dominance-audit constants, and two
shadow audit fields (`haircut_would_have_blocked`, `phase3_component_mask`). The protected
`promotion.py` threads `wf.market_returns` into `evaluate_gate`.

**Tech Stack:** Python 3.12, numpy/pandas, scipy (none new), pytest. No schema change, no new dep.

## Global Constraints

- **Tighten-only.** The aggregate `holdout_sharpe` check (Phase 1) is **byte-identical**.
  `regime_robustness` is a NEW AND-member, appended ONLY when the market-vol series is available with
  sufficient overlap; it can only move a gate PASS→FAIL. Property-tested.
- **Binding / omit-not-fail.** `regime_robustness` binds iff `wf.market_returns` is present AND
  `wf.holdout_returns` is present AND the post-alignment overlap (holdout dates with a valid trailing
  market-vol) ≥ `MIN_REGIME_OVERLAP_BARS`. When it binds: it is ALWAYS appended (never omitted) —
  `< 2` surviving regimes ⇒ appended **FAILED** (fail-closed; covers the constant-vol case). When it
  does NOT bind (no market series, or insufficient overlap) ⇒ the check is **OMITTED entirely**
  (`regime_method = "unavailable" | "insufficient_overlap"`, `regime_robustness_binding = false`) —
  this keeps every existing promotion test (whose `wf` has no `market_returns`) byte-identical.
- **Regimes from the MARKET series, never the strategy's own returns** (circular/gameable). Tertiles
  by 21-bar trailing realized volatility of the benchmark, date-aligned to the strategy's OOS bars.
  Deterministic tie-break (rank ties → earliest date wins the lower tertile).
- **Zero-vol regime = underpowered (dropped), not a Sharpe=0 pass.** `metrics_from_returns` returns
  `sharpe = 0.0` when `ann_volatility == 0.0`; with `MIN_REGIME_SHARPE = 0.0` that would falsely
  pass. Treat `ann_volatility == 0.0` as `dropped_reason = "zero_vol"` (alongside `"too_short"`).
- **Dominance-audit predeclaration (BLOCKING — must land in this slice).** `gates.py` gains
  CODEOWNERS-protected constants `DOMINANCE_AUDIT_MIN_PROMOTIONS = 30`,
  `DOMINANCE_AUDIT_MIN_WINDOW_DAYS = 90`, `DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS = 0`, and every
  `decision_json` records `haircut_would_have_blocked: bool` + `phase3_component_mask: int`. A
  CI-enforcing test imports all three `DOMINANCE_AUDIT_*` constants and FAILS if any is absent
  (prevents this PR landing without the predeclaration). These are SHADOW/AUDIT only — no gate
  behavior change.
- **Fail-closed / no NaN in payload.** Pure helpers return a "no robustness evidence" result (omit)
  or FAILED, never NaN; `to_dict` nulls non-finite floats via `_f`; `per_regime_sharpes` list is
  null-coerced element-wise.
- **PIT-compliance.** The benchmark series uses the SAME `provider`/`universe_by_date` the backtest
  used — a second read of the same immutable snapshot. No live pull. If the read fails / is empty,
  `market_returns = None` (the check then omits; `walk_forward` must NOT raise).
- **`market_returns` is bulky, not sensitive** (aggregate market data) — but EXCLUDE it from
  `WalkForwardResult.to_dict()` anyway (it's an internal gate input, like `holdout_returns`; keep
  operator output lean).
- **Architecture boundary.** `regime_splits`/`regime_robustness_check` are PURE (deterministic, no
  I/O) — acceptable in `gates.py` (unlike the Slice-2 bootstrap's RNG, which went to backtest). The
  benchmark-series COMPUTATION (data read) lives in `walk_forward` (backtest). `gates.py` gets the
  pre-computed series vectors and does pure math.
- **Per-slice quality gate (before commit):**
  `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- **Constant values (spec GATE-1 Q4.4 + named-constants table):** `N_REGIMES = 3`,
  `MIN_REGIME_OBSERVATIONS = 21`, `MIN_REGIME_SHARPE = 0.0`, `MIN_REGIME_OVERLAP_BARS = 63`,
  `VOL_ROLLING_WINDOW = 21`; dominance constants above; `PHASE3_COMPONENT_MASK = 0b11111` (slices
  0–4 active).
- **Protected walls** (`gates.py`, `promotion.py`) CODEOWNERS `@Lior-Nis`.
- **Review panel:** Codex + OpenCode only (Gemini deauthorized).

## Locked design decisions

- **D1 — benchmark = equal-weighted cross-sectional daily return of the strategy's universe** (user-
  chosen Q4.2 source), computed in `walk_forward` from the same bars, full-period (so the 21-bar
  trailing vol has lookback for every holdout date). Attached as `market_returns`.
- **D2 — pure regime helpers in `gates.py`** (deterministic reduction, no I/O — consistent with
  `dsr_confidence`); the series VECTORS are threaded in by `promotion.py`/read off `wf`.
- **D3 — realized vol = annualized std of trailing-21 daily LOG returns** of the benchmark (spec
  Q4.2). Guard `1 + r > 0` (equal-weighted index daily returns never ≤ −100%; clip defensively).
- **D4 — regime check binds independent of `dsr_binding`** (it is a holdout-robustness check,
  orthogonal to measured-breadth). It binds purely on market-series availability + overlap.
- **D5 — dominance scaffolding ships HERE** (spec: blocking-before-Slice-4-merges). Shadow-only.

## File Structure

- `algua/backtest/walkforward.py` — compute + attach `market_returns`; `to_dict` excludes it.
- `algua/research/gates.py` (PROTECTED) — regime constants + dominance constants; pure
  `regime_splits`/`regime_robustness_check` + `RegimeSlice`/`RegimeRobustnessResult`; `evaluate_gate`
  `regime_robustness` check; GateDecision regime + dominance audit fields + `to_dict`.
- `algua/registry/promotion.py` (PROTECTED) — thread `wf.market_returns` into `evaluate_gate`.
- Tests: `tests/backtest/test_market_returns.py`, `tests/research/test_regime_robustness.py`,
  `tests/research/test_dominance_predeclaration.py`, extend `tests/test_promotion.py`.

---

## Task 1: `walk_forward` benchmark series — `WalkForwardResult.market_returns` (backtest)

**Files:**
- Modify: `algua/backtest/walkforward.py` (`WalkForwardResult` ~line 70-101; `walk_forward` body
  after `build_portfolio` ~line 142-148 and the constructor ~line 193)
- Modify: `algua/backtest/engine.py` — make `_adj_grid` importable (rename to `adj_grid` public, or
  add a thin public wrapper) so `walk_forward` reuses the EXACT pivot the engine uses. (If
  `lint-imports`/style prefers, inline the one-liner instead — but reuse is DRY-preferred.)
- Test: `tests/backtest/test_market_returns.py` (create)

**Interfaces:**
- Produces: `WalkForwardResult.market_returns: tuple[list[float], list[str]] | None = None` — the
  FULL-PERIOD `(equal-weighted cross-sectional daily return, ISO date)` series, or `None` if the
  benchmark read fails / is empty. EXCLUDED from `to_dict()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/backtest/test_market_returns.py`. Reuse the synthetic-provider + loaded-demo-strategy
harness from an existing walk-forward test (e.g. `tests/test_walkforward_segment_noguard.py` or
`tests/test_walkforward_holdout_returns.py`).

```python
import dataclasses

# build a WalkForwardResult directly for the to_dict exclusion test:
from algua.backtest.walkforward import WalkForwardResult, walk_forward


def test_market_returns_excluded_from_to_dict():
    wf = WalkForwardResult(
        strategy="s", config_hash="c", data_source="synthetic", snapshot_id=None, timeframe="1d",
        seed=None, period={"start": "2020-01-01", "end": "2020-12-31"}, windows=4, holdout_frac=0.2,
        window_metrics=[], holdout_metrics={"n_bars": 3}, stability={},
        market_returns=([0.001, -0.002], ["2020-12-30", "2020-12-31"]))
    assert "market_returns" not in wf.to_dict()
    assert wf.market_returns is not None


def test_walk_forward_populates_market_returns(<provider/strategy fixture>):
    wf = walk_forward(<demo strategy, synthetic provider, start, end, ...>)
    assert wf.market_returns is not None
    mr, md = wf.market_returns
    assert len(mr) == len(md) > 0
    assert all(isinstance(d, str) and len(d) == 10 for d in md)   # ISO dates
    # full-period length: market series spans the whole sim, longer than the holdout
    assert len(md) >= wf.holdout_metrics["n_bars"]
    # equal-weighted cross-sectional: for a 1-symbol universe it equals that symbol's daily return;
    # for >1 symbols it is the mean across symbols. (Assert finite, dated, monotonic dates.)
    assert md == sorted(md)
```

(Fill the fixtures from the existing walk-forward test. For the multi-symbol assertion, use a
2-symbol synthetic universe and assert the market return at a date equals the mean of the two
symbols' returns at that date if the harness exposes per-symbol bars; otherwise assert finiteness +
length + date-sortedness.)

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/backtest/test_market_returns.py -q` → FAIL.

- [ ] **Step 3: Implement the field + computation**

Add the field after `holdout_returns` on `WalkForwardResult`:

```python
    # FULL-PERIOD equal-weighted cross-sectional daily return of the universe (#221 Slice 4) — the
    # market/benchmark series for vol-tertile regime labeling. NOT sensitive, but bulky → excluded
    # from to_dict (an internal gate input, like holdout_returns). None if the benchmark read fails.
    market_returns: tuple[list[float], list[str]] | None = None
```

Extend the existing custom `to_dict` exclusion to also drop `market_returns`:
`if f.name not in ("holdout_returns", "market_returns")`.

In `walk_forward`, after `build_portfolio` returns (the `provider`, `start`, `end`,
`universe_by_date`, `strategy` are in scope), compute the benchmark — robustly (never raise):

```python
    market_returns = _market_return_series(strategy, provider, start, end, universe_by_date)
```

with a module-level helper:

```python
def _market_return_series(strategy, provider, start, end, universe_by_date):
    """Equal-weighted cross-sectional daily return of the universe (PIT: same provider/snapshot as
    the backtest). Returns (returns, ISO-dates) over the FULL period, or None on any failure."""
    try:
        symbols = _fetch_symbols(strategy, universe_by_date)   # same symbol set the engine fetches
        bars = provider.get_bars(symbols, start, end, "1d")
        adj = adj_grid(bars)                                   # (dates x symbols) adj_close panel
        xs = adj.pct_change().mean(axis=1).dropna()            # equal-weighted cross-sectional return
        if xs.empty:
            return None
        return ([float(x) for x in xs.to_numpy()],
                [str(idx.date()) for idx in xs.index])
    except Exception:
        return None
```

Import `_fetch_symbols` and `adj_grid` from `algua.backtest.engine` (make them importable). Pass
`market_returns=market_returns` into the `WalkForwardResult(...)` constructor.

- [ ] **Step 4: Run tests + quality gate** → PASS. Confirm no existing walk-forward/backtest test
broke (the new field defaults None and is excluded from `to_dict`; the extra `get_bars` read is
additive). Run the full quality gate.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/walkforward.py algua/backtest/engine.py tests/backtest/test_market_returns.py
git commit -m "feat(221): WalkForwardResult.market_returns equal-weighted benchmark series — Slice 4 of #221"
```

---

## Task 2: pure regime helpers + constants (PROTECTED gates.py, no gate-check change yet)

**Files:**
- Modify: `algua/research/gates.py` (constants near the other Phase-3 constants; pure helpers near
  `dsr_confidence`)
- Test: `tests/research/test_regime_robustness.py` (create — helper unit tests)

**Interfaces:**
- Produces: constants `N_REGIMES = 3`, `MIN_REGIME_OBSERVATIONS = 21`, `MIN_REGIME_SHARPE = 0.0`,
  `MIN_REGIME_OVERLAP_BARS = 63`, `VOL_ROLLING_WINDOW = 21`.
  - `RegimeSlice(NamedTuple)`: `regime_index: int`, `returns: list[float]`, `n_bars: int`,
    `dropped_reason: str | None`  (`None` | `"too_short"` | `"zero_vol"`).
  - `regime_splits(strategy_returns, strategy_dates, market_returns, market_dates, *, n_regimes,
    vol_window) -> tuple[list[RegimeSlice], int]` — returns `(slices, overlap_n)` where `overlap_n`
    is the count of holdout dates that received a valid trailing market-vol label. Computes the
    21-bar trailing realized vol (annualized std of daily LOG returns) of the market over its full
    series, labels each market date that has ≥ `vol_window` lookback, inner-joins to
    `strategy_dates`, assigns the joined dates to `n_regimes` volatility tertiles (rank-based,
    deterministic tie-break by date), and buckets the strategy returns. `overlap_n < anything` and
    empty join → `([], 0)`.
  - `regime_robustness_check(slices, *, min_obs, min_sharpe) -> RegimeRobustnessResult` where
    `RegimeRobustnessResult(NamedTuple)`: `passed: bool`, `n_attempted: int`, `n_surviving: int`,
    `per_regime_sharpes: list[float | None]`. Drops a regime with `< min_obs` bars (`too_short`) or
    `ann_volatility == 0.0` (`zero_vol`); `< 2` survivors ⇒ `passed = False`; else `passed = all
    surviving sharpe ≥ min_sharpe`.

- [ ] **Step 1: Write the failing helper tests**

Create `tests/research/test_regime_robustness.py`. Build synthetic strategy + market series with a
known vol structure.

```python
import numpy as np

from algua.research.gates import (
    MIN_REGIME_OBSERVATIONS, MIN_REGIME_SHARPE, N_REGIMES, RegimeSlice,
    regime_robustness_check, regime_splits,
)


def _dates(n, start=0):
    return [f"2020-{(start+i)//28 % 12 + 1:02d}-{(start+i) % 28 + 1:02d}" for i in range(n)]
    # NOTE: replace with a robust ISO date generator (datetime.date + timedelta) — see Task 1 note.


def test_constants():
    assert N_REGIMES == 3 and MIN_REGIME_OBSERVATIONS == 21 and MIN_REGIME_SHARPE == 0.0


def test_tertiles_assigned_by_market_vol():
    # market: first third low-vol, middle third mid, last third high-vol; strategy: arbitrary.
    n = 90
    md = _robust_dates(n)
    rng = np.random.default_rng(0)
    low = rng.normal(0, 0.002, n // 3); mid = rng.normal(0, 0.01, n // 3); hi = rng.normal(0, 0.03, n - 2*(n//3))
    market = list(np.concatenate([low, mid, hi]))
    strat = list(rng.normal(0.001, 0.01, n))
    slices, overlap = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    assert overlap > 0 and len(slices) == 3
    assert sum(s.n_bars for s in slices) == overlap        # every labeled date bucketed once


def test_constant_vol_fewer_than_two_survivors_fails():
    # near-constant market vol -> all dates land in (effectively) one tertile -> <2 powered regimes.
    n = 90; md = _robust_dates(n)
    market = [0.01] * n                                    # constant returns -> constant (zero) vol
    strat = list(np.random.default_rng(1).normal(0.001, 0.01, n))
    slices, overlap = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    res = regime_robustness_check(slices, min_obs=21, min_sharpe=0.0)
    assert res.passed is False                              # fail-closed


def test_zero_vol_regime_dropped_not_passed():
    s = [RegimeSlice(0, [0.0]*30, 30, None), RegimeSlice(1, [0.01, -0.01]*15, 30, None),
         RegimeSlice(2, [0.02, -0.005]*15, 30, None)]
    res = regime_robustness_check(s, min_obs=21, min_sharpe=0.0)
    # regime 0 is constant -> ann_volatility 0 -> dropped (zero_vol), not a sharpe=0 pass.
    assert res.n_surviving == 2


def test_underpowered_regime_dropped_and_lt2_fails():
    s = [RegimeSlice(0, [0.01,-0.01]*15, 30, None), RegimeSlice(1, [0.01], 1, None),
         RegimeSlice(2, [0.02], 1, None)]
    res = regime_robustness_check(s, min_obs=21, min_sharpe=0.0)
    assert res.n_surviving == 1 and res.passed is False    # only 1 powered -> <2 survivors -> FAIL


def test_all_surviving_clear_floor_passes():
    s = [RegimeSlice(0, list(np.random.default_rng(2).normal(0.01, 0.01, 30)), 30, None),
         RegimeSlice(1, list(np.random.default_rng(3).normal(0.01, 0.01, 30)), 30, None),
         RegimeSlice(2, list(np.random.default_rng(4).normal(0.01, 0.01, 30)), 30, None)]
    res = regime_robustness_check(s, min_obs=21, min_sharpe=0.0)
    assert res.n_surviving == 3 and res.passed is True


def test_deterministic_tie_break():
    # identical market vols -> rank ties broken by date order; two runs identical.
    n = 90; md = _robust_dates(n); market = [0.01]*n
    strat = list(np.random.default_rng(5).normal(0.001, 0.01, n))
    a = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    b = regime_splits(strat, md, market, md, n_regimes=3, vol_window=21)
    assert [s.n_bars for s in a[0]] == [s.n_bars for s in b[0]]
```

(Use a ROBUST ISO date generator `_robust_dates(n)` = `datetime.date(2019,1,1)+timedelta(days=i)`
formatted — NOT the fragile arithmetic — so dates are real, unique, sorted.)

- [ ] **Step 2: Run to verify it fails** — ImportError / missing functions.

- [ ] **Step 3: Add constants**

Near the other Phase-3 constants in `gates.py`:

```python
# Multi-regime robustness (#221, Phase 3 Slice 4). Protected — relaxing weakens the gate.
N_REGIMES = 3                  # market-volatility tertiles (low/medium/high)
MIN_REGIME_OBSERVATIONS = 21   # per-regime power floor (underpowered regimes are dropped)
MIN_REGIME_SHARPE = 0.0        # relaxed per-regime Sharpe bar (paired with zero-vol drop)
MIN_REGIME_OVERLAP_BARS = 63   # min holdout dates with a valid trailing market-vol for the check to bind
VOL_ROLLING_WINDOW = 21        # trailing bars for the benchmark realized-vol estimate
```

- [ ] **Step 4: Implement the pure helpers**

`regime_splits`: (1) build `date→market_return` map; compute the trailing-`vol_window` realized vol
(annualized std of `log(1+r)`, guard `1+r>0`) for each market date with ≥ `vol_window` prior bars;
(2) inner-join the vol-labeled market dates with `strategy_dates` (a dict on strategy
`date→return`); `overlap_n = len(joined)`; (3) rank the joined dates by `(vol, date)` ascending and
split into `n_regimes` contiguous tertiles (deterministic: ties broken by the date key); (4) bucket
each joined strategy return into its regime. Return `(slices, overlap_n)`.

`regime_robustness_check`: for each slice, drop if `n_bars < min_obs` (`too_short`) or
`metrics_from_returns(slice.returns)["ann_volatility"] == 0.0` (`zero_vol`); else compute
`sharpe = metrics_from_returns(slice.returns)["sharpe"]`. `< 2` survivors ⇒ `passed = False`. Else
`passed = all(sharpe ≥ min_sharpe for surviving)`. Record `per_regime_sharpes` aligned to the
attempted regimes (`None` for dropped).

(Use `metrics_from_returns` from `algua.backtest.metrics` — already imported in gates? confirm; it
gives `sharpe` and `ann_volatility` with `sharpe=0.0` at `ann_volatility=0.0`, which is exactly why
the zero-vol drop is needed. numpy/pandas for the rolling vol — `gates.py` may import numpy.)

- [ ] **Step 5: Run helper tests + quality gate** → PASS.

- [ ] **Step 6: Commit**

```bash
git add algua/research/gates.py tests/research/test_regime_robustness.py
git commit -m "feat(221): pure regime-split + per-regime robustness helpers + constants — Slice 4"
```

---

## Task 3: `regime_robustness` AND-check in `evaluate_gate` + audit (PROTECTED)

**Files:**
- Modify: `algua/research/gates.py` (`evaluate_gate` signature + body; `GateDecision` fields + `to_dict`)
- Test: extend `tests/research/test_regime_robustness.py`

**Interfaces:**
- `evaluate_gate(..., market_returns: tuple[list[float], list[str]] | None = None)`. When
  `market_returns` AND `wf.holdout_returns` are present AND `regime_splits` yields `overlap_n ≥
  MIN_REGIME_OVERLAP_BARS`: append a `regime_robustness` check (`passed` = `regime_robustness_check`
  verdict; `< 2` survivors ⇒ FAILED). Else: OMIT the check (no append). `GateDecision` gains
  `regime_method: str | None` (`"vol_tertile" | "unavailable" | "insufficient_overlap"`),
  `n_regimes_attempted: int | None`, `n_regimes_surviving: int | None`, `per_regime_sharpes:
  list[float | None] | None`, `regime_robustness_binding: bool = False`; all in `to_dict`
  (`per_regime_sharpes` element-wise null-coerced).

- [ ] **Step 1: Write failing tests** — extend the test file: a `make_wf` carrying `holdout_returns`
+ a `market_returns` arg produces a `regime_robustness` check; binding+<2 survivors ⇒ FAILED + gate
`passed=False`; `market_returns=None` ⇒ NO `regime_robustness` check (omit) and `regime_method=
"unavailable"`; insufficient overlap ⇒ omit + `regime_method="insufficient_overlap"`; tighten-only
property (for any market series, `new.passed ⇒ old.passed`); audit fields surface in `to_dict`.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Add `GateDecision` fields + `to_dict`** (after the Slice-3 `dsr_n_*` fields).

- [ ] **Step 4: Wire the check into `evaluate_gate`**

Add the `market_returns` param. Compute availability: read `wf.holdout_returns`; if it and
`market_returns` are present, call `regime_splits(...)`. If `overlap_n < MIN_REGIME_OVERLAP_BARS` →
omit (`regime_method = "insufficient_overlap"`, binding False). If `market_returns`/`holdout_returns`
absent → omit (`regime_method = "unavailable"`). Else call `regime_robustness_check`, append
`{"name": "regime_robustness", "value": None, "threshold": MIN_REGIME_SHARPE, "op": ">=", "passed":
result.passed}`, set `regime_method = "vol_tertile"`, `regime_robustness_binding = True`, and
populate the audit fields. (The aggregate `holdout_sharpe` check is untouched.)

- [ ] **Step 5: Run tests + quality gate** → PASS; existing tests (no `market_returns`) byte-identical.

- [ ] **Step 6: Commit**

```bash
git add algua/research/gates.py tests/research/test_regime_robustness.py
git commit -m "feat(221): regime_robustness AND-check + audit in evaluate_gate — Slice 4"
```

---

## Task 4: dominance-audit predeclaration scaffolding (PROTECTED, shadow)

**Files:**
- Modify: `algua/research/gates.py` (3 dominance constants + `PHASE3_COMPONENT_MASK`; `GateDecision`
  `haircut_would_have_blocked` + `phase3_component_mask` fields + `to_dict`; compute them in
  `evaluate_gate`)
- Test: `tests/research/test_dominance_predeclaration.py` (create — the CI-enforcing import test)

**Interfaces:**
- Produces: `DOMINANCE_AUDIT_MIN_PROMOTIONS = 30`, `DOMINANCE_AUDIT_MIN_WINDOW_DAYS = 90`,
  `DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS = 0`, `PHASE3_COMPONENT_MASK = 0b11111`.
  `GateDecision.haircut_would_have_blocked: bool = False`, `phase3_component_mask: int | None = None`;
  both in `to_dict`.

- [ ] **Step 1: Write the failing CI-enforcing + behavior tests**

```python
def test_dominance_audit_constants_predeclared():
    # CI enforcement: this PR cannot land without the predeclared thresholds.
    from algua.research.gates import (
        DOMINANCE_AUDIT_MIN_PROMOTIONS, DOMINANCE_AUDIT_MIN_WINDOW_DAYS,
        DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS,
    )
    assert DOMINANCE_AUDIT_MIN_PROMOTIONS == 30
    assert DOMINANCE_AUDIT_MIN_WINDOW_DAYS == 90
    assert DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS == 0


def test_haircut_would_have_blocked_true_only_when_haircut_is_the_blocker(make_wf):
    # holdout sharpe passes the BASE bar but fails the haircut-inflated bar -> True.
    # (build a wf whose holdout sharpe sits between base and base+haircut with measured n_combos)
    ...

def test_phase3_component_mask_recorded():
    d = evaluate_gate(make_wf(sharpe=7.0), GateCriteria(), n_combos=5, pit_ok=True)
    assert d.phase3_component_mask == 0b11111
    assert "haircut_would_have_blocked" in d.to_dict()
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Add the constants + `GateDecision` fields + `to_dict`.**

- [ ] **Step 4: Compute the shadow fields in `evaluate_gate`**

```python
    sr = float(wf.holdout_metrics["sharpe"])
    haircut_would_have_blocked = bool(
        math.isfinite(effective_holdout_sharpe)
        and sr >= base_holdout_sharpe and sr < effective_holdout_sharpe)
    # (when the effective bar is inf — degenerate holdout — the haircut "blocks" everything; record
    #  True iff the base bar passed, consistent with "the haircut is what blocks it".)
```

Set `phase3_component_mask = PHASE3_COMPONENT_MASK` on every returned `GateDecision`. These are
SHADOW/audit only — they do NOT enter `decision.passed`.

- [ ] **Step 5: Run tests + quality gate** → PASS; gate outcomes byte-identical (audit-only).

- [ ] **Step 6: Commit**

```bash
git add algua/research/gates.py tests/research/test_dominance_predeclaration.py
git commit -m "feat(221): dominance-audit predeclaration (constants + shadow fields) — Slice 4"
```

---

## Task 5: promotion.py threading + integration (PROTECTED)

**Files:**
- Modify: `algua/registry/promotion.py` (`run_gate` — pass `wf.market_returns` into `evaluate_gate`)
- Test: extend `tests/test_promotion.py`

- [ ] **Step 1: Write a failing integration test** — a measured promote with a `wf` carrying both
`holdout_returns` AND a `market_returns` series with a real vol structure and sufficient overlap:
assert `decision.regime_robustness_binding is True`, a `regime_robustness` check present,
`regime_method == "vol_tertile"`, `per_regime_sharpes`/`n_regimes_*` populated, and the fields land
in `decision_json`. A promote whose `wf` has NO `market_returns` → no `regime_robustness` check,
`regime_method == "unavailable"`, outcome unchanged. Tighten-only: a strategy passing the aggregate
but failing a high-vol regime is blocked.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Thread it in** — pass `market_returns=wf.market_returns` into the `evaluate_gate(...)`
call in `run_gate`. (No other change; `evaluate_gate` reads `wf.holdout_returns` itself.)

- [ ] **Step 4: Run integration + full quality gate** → PASS. If an existing promotion test that
supplies `market_returns` flips outcome, that is a real tightening — adjust the fixture's returns to
be regime-robust, do NOT weaken the check. A test WITHOUT `market_returns` changing outcome is a bug
(must omit) — STOP and fix.

- [ ] **Step 5: Commit**

```bash
git add algua/registry/promotion.py tests/test_promotion.py
git commit -m "feat(221): thread market_returns into run_gate for regime robustness — Slice 4"
```

---

## Self-Review notes

- **Spec coverage (component c):** market-vol-tertile regimes from a benchmark series (Tasks 1/2);
  deterministic tie-break + constant-vol→fail-closed + zero-vol drop + underpowered drop + ≥2
  survivors (Task 2); calendar-split fallback = OMIT when vol unavailable/insufficient-overlap
  (Task 3); tighten-only AND alongside the untouched aggregate `holdout_sharpe` (Task 3); PIT
  benchmark from the same snapshot (Task 1). Dominance-audit predeclaration + shadow fields + CI test
  (Task 4). Q4.2 source = equal-weighted cross-sectional universe vol (user-chosen, Task 1).
- **Tighten-only / omit-not-fail** property-tested (Task 3); existing no-`market_returns` tests
  byte-identical.
- **GATE-2 focus:** the `regime_splits` rolling-vol + tertile + date-alignment math (Task 2); the
  tighten-only + binding-condition on the protected wall (Tasks 3/5); the `haircut_would_have_blocked`
  definition correctness (Task 4).
- **Deferred (NOT this slice):** the actual dominance audit + haircut retirement + binding N_eff —
  Slice 5 (separate post-Phase-2 issue).
