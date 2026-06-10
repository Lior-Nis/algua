# Corporate-Action Back-Adjustment Engine (#149) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure, I/O-free corporate-action back-adjustment engine — `(raw OHLC frame, split/dividend events) → (adj_close, adj_factor)` — plus a reverse-split-safe validator that checks a vendor-supplied `adj_close` against an event list.

**Architecture:** One new pure leaf module `algua/data/corpactions.py` (pandas + numpy only, no other `algua` imports). Three public surfaces: the `Split`/`Dividend` discriminated-union event types, `back_adjust(raw, events)`, and `check_adj_close_consistent(...)`. No importer wiring (rides with Databento #150). Full design + rationale: `docs/superpowers/specs/2026-06-10-corporate-action-back-adjustment-engine-issue-149-design.md`.

**Tech Stack:** Python, pandas, numpy, pytest. Frozen dataclasses, `math.isfinite` guards, numpy suffix-product over event boundaries.

**Quality gate (run before each commit):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

### Task 1: Event types — `Split`, `Dividend`, `CorporateAction`

**Files:**
- Create: `algua/data/corpactions.py`
- Test: `tests/test_corpactions.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corpactions.py
import math

import pandas as pd
import pytest

from algua.data.corpactions import Dividend, Split


def _utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_split_and_dividend_construct():
    s = Split(ex_date=_utc("2024-01-03"), ratio=2.0)
    d = Dividend(ex_date=_utc("2024-01-03"), cash=1.5)
    assert s.ratio == 2.0 and d.cash == 1.5


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("inf"), float("nan")])
def test_split_ratio_must_be_finite_positive(ratio):
    with pytest.raises(ValueError, match="ratio"):
        Split(ex_date=_utc("2024-01-03"), ratio=ratio)


@pytest.mark.parametrize("cash", [0.0, -1.0, float("inf"), float("nan")])
def test_dividend_cash_must_be_finite_positive(cash):
    with pytest.raises(ValueError, match="cash"):
        Dividend(ex_date=_utc("2024-01-03"), cash=cash)


def test_tz_naive_ex_date_rejected():
    with pytest.raises(ValueError, match="tz-aware"):
        Split(ex_date=pd.Timestamp("2024-01-03"), ratio=2.0)
    with pytest.raises(ValueError, match="tz-aware"):
        Dividend(ex_date=pd.Timestamp("2024-01-03"), cash=1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_corpactions.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'algua.data.corpactions'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# algua/data/corpactions.py
"""Pure corporate-action back-adjustment engine (#149).

Given a raw OHLC frame plus a typed split/dividend event list for one symbol, produce the
back-adjusted close (`adj_close`) and the cumulative adjustment factor; plus a validator that checks
a vendor-supplied `adj_close` against the same events (reverse-split-safe). No I/O; imports only
pandas + numpy. See docs/superpowers/specs/2026-06-10-corporate-action-back-adjustment-engine-issue-149-design.md.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _check_ex_date(ex_date: pd.Timestamp) -> None:
    if not isinstance(ex_date, pd.Timestamp) or ex_date.tz is None:
        raise ValueError(f"ex_date must be a tz-aware pd.Timestamp, got {ex_date!r}")


@dataclass(frozen=True)
class Split:
    """A forward/reverse split. `ratio` = new shares per old (2.0 = 2:1; 0.1 = 1:10 reverse)."""

    ex_date: pd.Timestamp
    ratio: float

    def __post_init__(self) -> None:
        _check_ex_date(self.ex_date)
        if not math.isfinite(self.ratio) or self.ratio <= 0:
            raise ValueError(f"Split.ratio must be finite and > 0, got {self.ratio!r}")


@dataclass(frozen=True)
class Dividend:
    """An ordinary cash dividend. `cash` = per-share cash in RAW-close (pre-split) price units."""

    ex_date: pd.Timestamp
    cash: float

    def __post_init__(self) -> None:
        _check_ex_date(self.ex_date)
        if not math.isfinite(self.cash) or self.cash <= 0:
            raise ValueError(f"Dividend.cash must be finite and > 0, got {self.cash!r}")


CorporateAction = Split | Dividend
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_corpactions.py -q`
Expected: PASS (all Task-1 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/data/corpactions.py tests/test_corpactions.py
git commit -m "feat(data): corporate-action event types Split/Dividend (#149)"
```

---

### Task 2: The engine — `back_adjust(raw, events)`

**Files:**
- Modify: `algua/data/corpactions.py`
- Test: `tests/test_corpactions.py`

- [ ] **Step 1: Write the failing tests (math, edges, guards)**

Append to `tests/test_corpactions.py`:

```python
import numpy as np

from algua.data.corpactions import back_adjust


def _bars(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"ts": ts, "close": [float(c) for c in closes]})


def _factor(raw, events) -> np.ndarray:
    return back_adjust(raw, events)["adj_factor"].to_numpy()


def _adj(raw, events) -> np.ndarray:
    return back_adjust(raw, events)["adj_close"].to_numpy()


def test_no_events_is_identity():
    raw = _bars([100, 110, 120])
    out = back_adjust(raw, [])
    assert list(out.columns) == ["ts", "adj_close", "adj_factor"]
    np.testing.assert_allclose(out["adj_factor"].to_numpy(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out["adj_close"].to_numpy(), [100, 110, 120])


def test_forward_2for1_split():
    raw = _bars([100, 110, 50, 55])  # ex on bar index 2 (already post-split)
    ev = [Split(ex_date=_utc("2024-01-03"), ratio=2.0)]
    np.testing.assert_allclose(_factor(raw, ev), [0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(_adj(raw, ev), [50, 55, 50, 55])


def test_reverse_1for10_split_scales_history_up():
    raw = _bars([5, 6, 50, 55])  # 1:10 reverse on bar index 2
    ev = [Split(ex_date=_utc("2024-01-03"), ratio=0.1)]
    np.testing.assert_allclose(_factor(raw, ev), [10.0, 10.0, 1.0, 1.0])
    np.testing.assert_allclose(_adj(raw, ev), [50, 60, 50, 55])  # adj/raw > 1 historically


def test_single_dividend_total_return():
    raw = _bars([100, 110, 120, 130])  # ex on bar index 2; P_prev = raw close[1] = 110
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=2.0)]
    m = (110 - 2) / 110
    np.testing.assert_allclose(_factor(raw, ev), [m, m, 1.0, 1.0])


def test_two_dividends_same_ex_date_sum_cash_no_cross_term():
    raw = _bars([100, 110, 120, 130])  # P_prev = 110
    ev = [
        Dividend(ex_date=_utc("2024-01-03"), cash=2.0),
        Dividend(ex_date=_utc("2024-01-03"), cash=3.0),
    ]
    correct = (110 - 5) / 110  # NOT (1 - 2/110)(1 - 3/110)
    np.testing.assert_allclose(_factor(raw, ev)[0], correct)


def test_same_ex_date_split_and_dividend():
    raw = _bars([100, 110, 50, 55])  # P_prev = raw close[1] = 110 (pre-split units)
    ev = [
        Split(ex_date=_utc("2024-01-03"), ratio=2.0),
        Dividend(ex_date=_utc("2024-01-03"), cash=2.0),
    ]
    expected = 0.5 * (110 - 2) / 110
    np.testing.assert_allclose(_factor(raw, ev)[0], expected)


def test_p_prev_is_raw_when_split_sits_between():
    # split on bar 1, dividend on bar 3; P_prev for the dividend is the RAW close[2], not adjusted.
    raw = _bars([100, 50, 55, 60, 65])
    ev = [
        Split(ex_date=_utc("2024-01-02"), ratio=2.0),
        Dividend(ex_date=_utc("2024-01-04"), cash=2.0),
    ]
    div = (55 - 2) / 55
    np.testing.assert_allclose(_factor(raw, ev), [0.5 * div, div, div, 1.0, 1.0])


def test_event_before_first_bar_is_noop():
    raw = _bars([100, 110, 120], start="2024-02-01")
    ev = [Split(ex_date=_utc("2024-01-01"), ratio=2.0)]
    np.testing.assert_allclose(_factor(raw, ev), [1.0, 1.0, 1.0])


def test_event_after_last_bar_is_noop_anchor_preserved():
    raw = _bars([100, 110, 120])
    ev = [Split(ex_date=_utc("2030-01-01"), ratio=2.0)]
    out = back_adjust(raw, ev)
    np.testing.assert_allclose(out["adj_factor"].to_numpy(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out["adj_close"].to_numpy(), [100, 110, 120])  # adj == close (no-op)


def test_dividend_ex_on_non_trading_day_resolves_to_prior_bar():
    # bars Mon/Tue/Thu/Fri; dividend ex on Wed (no bar) scales the two bars strictly before it.
    ts = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"], tz="UTC")
    raw = pd.DataFrame({"ts": ts, "close": [100.0, 110.0, 120.0, 130.0]})
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=2.0)]  # P_prev = close at 01-02 = 110
    m = (110 - 2) / 110
    np.testing.assert_allclose(_factor(raw, ev), [m, m, 1.0, 1.0])


def test_empty_bars_returns_empty_typed_frame():
    raw = pd.DataFrame({"ts": pd.DatetimeIndex([], tz="UTC"), "close": pd.Series([], dtype="float64")})
    out = back_adjust(raw, [Split(ex_date=_utc("2024-01-03"), ratio=2.0)])
    assert list(out.columns) == ["ts", "adj_close", "adj_factor"]
    assert len(out) == 0


def test_non_midnight_ex_date_raises():
    raw = _bars([100, 110, 120])
    ev = [Split(ex_date=pd.Timestamp("2024-01-02 12:00", tz="UTC"), ratio=2.0)]
    with pytest.raises(ValueError, match="midnight"):
        back_adjust(raw, ev)


def test_dividend_ge_prior_close_raises_actionable():
    raw = _bars([100, 110, 120, 130])  # P_prev = 110
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=200.0)]
    with pytest.raises(ValueError, match="liquidating"):
        back_adjust(raw, ev)


def test_raw_frame_guards():
    ev: list = []
    with pytest.raises(ValueError, match="'ts' and 'close'"):
        back_adjust(pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=2, tz="UTC")}), ev)
    # unsorted ts
    ts = pd.DatetimeIndex(["2024-01-02", "2024-01-01"], tz="UTC")
    with pytest.raises(ValueError, match="ascending"):
        back_adjust(pd.DataFrame({"ts": ts, "close": [1.0, 2.0]}), ev)
    # duplicate ts
    ts = pd.DatetimeIndex(["2024-01-01", "2024-01-01"], tz="UTC")
    with pytest.raises(ValueError, match="ascending|unique"):
        back_adjust(pd.DataFrame({"ts": ts, "close": [1.0, 2.0]}), ev)
    # nonpositive / inf close
    with pytest.raises(ValueError, match="finite"):
        back_adjust(_bars([100, 0, 120]), ev)
    with pytest.raises(ValueError, match="finite"):
        back_adjust(_bars([100, float("inf"), 120]), ev)
    # tz-naive ts
    naive = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=2), "close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="tz-aware"):
        back_adjust(naive, ev)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_corpactions.py -q`
Expected: FAIL — `back_adjust` is not defined / `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `algua/data/corpactions.py`:

```python
def back_adjust(raw: pd.DataFrame, events: Iterable[CorporateAction]) -> pd.DataFrame:
    """Back-adjust `raw['close']` for `events`, anchored at the most recent bar.

    Returns a frame `[ts, adj_close, adj_factor]`, one row per input bar in input order, with
    `adj_close = close * adj_factor`. Pure; see the module docstring + design spec.
    """
    if "ts" not in raw.columns or "close" not in raw.columns:
        raise ValueError("raw must have 'ts' and 'close' columns")

    ts = pd.DatetimeIndex(raw["ts"])
    n = len(ts)
    if n > 0 and ts.tz is None:
        raise ValueError("raw['ts'] must be tz-aware (UTC)")
    if ts.tz is not None:
        ts = ts.tz_convert("UTC")
    close = raw["close"].to_numpy(dtype="float64")
    if n > 0:
        if not (ts.is_monotonic_increasing and ts.is_unique):
            raise ValueError("raw['ts'] must be strictly ascending and unique")
        if not np.all(np.isfinite(close)) or np.any(close <= 0):
            raise ValueError("raw['close'] must be finite and > 0")

    # Group events by (UTC, midnight-validated) ex_date: split ratios multiply, dividend cash sums.
    split_ratio_prod: dict[pd.Timestamp, float] = {}
    div_cash_sum: dict[pd.Timestamp, float] = {}
    for ev in events:
        ex = ev.ex_date.tz_convert("UTC")
        if ex != ex.normalize():
            raise ValueError(
                f"ex_date must be UTC midnight (a session date), got {ev.ex_date!r}"
            )
        if isinstance(ev, Split):
            split_ratio_prod[ex] = split_ratio_prod.get(ex, 1.0) * ev.ratio
        else:
            div_cash_sum[ex] = div_cash_sum.get(ex, 0.0) + ev.cash

    # Suffix-product accumulator: A[idx] holds the combined multiplier applied at each ex-date
    # boundary; factor[i] = prod(A[i+1 .. n]). A[0] and A[n] (pre-/post-range no-ops) stay 1.0.
    A = np.ones(n + 1, dtype="float64")
    for ex in set(split_ratio_prod) | set(div_cash_sum):
        idx = int(ts.searchsorted(ex, side="left"))
        if idx == 0 or idx == n:  # pre-range or post-range ex-date -> no-op (no look-ahead)
            continue
        split_mult = 1.0 / split_ratio_prod.get(ex, 1.0)
        cash = div_cash_sum.get(ex, 0.0)
        if cash > 0.0:
            p_prev = close[idx - 1]
            if p_prev - cash <= 0:
                raise ValueError(
                    f"dividend cash {cash} >= prior close {p_prev} on {ex.date()}: if this is a "
                    f"liquidating / return-of-capital distribution, exclude it from the event list; "
                    f"otherwise check data alignment (only ordinary cash dividends are modeled)"
                )
            div_mult = (p_prev - cash) / p_prev
        else:
            div_mult = 1.0
        A[idx] *= split_mult * div_mult

    factor = np.cumprod(A[1:][::-1])[::-1] if n > 0 else np.array([], dtype="float64")
    return pd.DataFrame(
        {"ts": ts, "adj_close": close * factor, "adj_factor": factor}
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_corpactions.py -q`
Expected: PASS (all Task-1 + Task-2 tests).

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. (Fix any ruff/mypy nit before committing — e.g. import order.)

- [ ] **Step 6: Commit**

```bash
git add algua/data/corpactions.py tests/test_corpactions.py
git commit -m "feat(data): back_adjust corporate-action engine (#149)"
```

---

### Task 3: The validator — `check_adj_close_consistent(...)`

**Files:**
- Modify: `algua/data/corpactions.py`
- Test: `tests/test_corpactions.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_corpactions.py`:

```python
from algua.data.corpactions import check_adj_close_consistent


def _series(closes: list[float], start: str = "2024-01-01") -> pd.Series:
    ts = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    return pd.Series([float(c) for c in closes], index=ts)


def _vendor_adj(closes: list[float], events) -> pd.Series:
    raw = _bars(closes)
    out = back_adjust(raw, events)
    return pd.Series(out["adj_close"].to_numpy(), index=pd.DatetimeIndex(out["ts"]))


def test_validator_accepts_consistent_series():
    closes = [100, 110, 120, 130]
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=2.0)]
    check_adj_close_consistent(_series(closes), _vendor_adj(closes, ev), ev)  # no raise


def test_validator_accepts_reverse_split():
    closes = [5, 6, 50, 55]
    ev = [Split(ex_date=_utc("2024-01-03"), ratio=0.1)]
    check_adj_close_consistent(_series(closes), _vendor_adj(closes, ev), ev)  # no raise


def test_validator_rejects_globally_mis_scaled_cents_vs_dollars():
    closes = [100, 110, 120, 130]
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=2.0)]
    vendor_cents = _vendor_adj(closes, ev) * 100.0  # adj in cents, raw in dollars
    with pytest.raises(ValueError, match="anchor|mis-scaled|cents"):
        check_adj_close_consistent(_series(closes), vendor_cents, ev)


def test_validator_rejects_torn_shifted_series():
    closes = [100, 110, 120, 130]
    ev = [Split(ex_date=_utc("2024-01-03"), ratio=2.0)]
    good = _vendor_adj(closes, ev)
    shifted = pd.Series(np.roll(good.to_numpy(), 1), index=good.index)
    with pytest.raises(ValueError):
        check_adj_close_consistent(_series(closes), shifted, ev)


def test_validator_rejects_wrong_magnitude_split():
    closes = [100, 110, 60, 66]
    vendor = _vendor_adj(closes, [Split(ex_date=_utc("2024-01-03"), ratio=3.0)])
    claimed = [Split(ex_date=_utc("2024-01-03"), ratio=2.0)]
    with pytest.raises(ValueError):
        check_adj_close_consistent(_series(closes), vendor, claimed)


def test_validator_low_price_small_dividend_within_tolerance_passes():
    closes = [5.00, 5.10, 5.20, 5.30]
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=0.03)]
    vendor = _vendor_adj(closes, ev).round(2)  # vendor rounds adj to the cent
    check_adj_close_consistent(_series(closes), vendor, ev)  # no false-reject


def test_validator_input_guards():
    closes = [100, 110, 120]
    ev: list = []
    good = _series(closes)
    # mismatched index
    other = _series(closes, start="2025-01-01")
    with pytest.raises(ValueError, match="same index"):
        check_adj_close_consistent(good, other, ev)
    # NaN / inf values
    nan_series = _series([100, float("nan"), 120])
    with pytest.raises(ValueError, match="finite"):
        check_adj_close_consistent(good, nan_series, ev)
    inf_series = _series([100, float("inf"), 120])
    with pytest.raises(ValueError, match="finite"):
        check_adj_close_consistent(inf_series, good, ev)
    # tz-naive index
    naive = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="tz-aware"):
        check_adj_close_consistent(naive, naive, ev)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_corpactions.py -q`
Expected: FAIL — `check_adj_close_consistent` is not defined.

- [ ] **Step 3: Write the implementation**

Append to `algua/data/corpactions.py`:

```python
def check_adj_close_consistent(
    raw_close: pd.Series,
    vendor_adj: pd.Series,
    events: Iterable[CorporateAction],
    *,
    rtol: float = 1e-3,
    atol: float = 5e-3,
) -> None:
    """Assert a vendor-supplied `adj_close` is consistent with `events`. Raise `ValueError` if not.

    Precondition: a FULL symbol series through the vendor's adjustment anchor (the last bar is the
    most recent), not an arbitrary mid-history slice. Reverse-split-safe; a gross-error / mis-units
    detector, not a penny-level dividend-parity certifier. See the design spec.
    """
    for name, series in (("raw_close", raw_close), ("vendor_adj", vendor_adj)):
        if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None:
            raise ValueError(f"{name} must have a tz-aware DatetimeIndex")
    if not raw_close.index.equals(vendor_adj.index):
        raise ValueError("raw_close and vendor_adj must share the same index")
    index = raw_close.index
    if len(index) and not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError("index must be strictly increasing and unique")
    rc = raw_close.to_numpy(dtype="float64")
    va = vendor_adj.to_numpy(dtype="float64")
    if not (np.all(np.isfinite(rc)) and np.all(rc > 0)):
        raise ValueError("raw_close must be finite and > 0")
    if not (np.all(np.isfinite(va)) and np.all(va > 0)):
        raise ValueError("vendor_adj must be finite and > 0")
    if len(index) == 0:
        return

    factor = back_adjust(pd.DataFrame({"ts": index, "close": rc}), events)[
        "adj_factor"
    ].to_numpy(dtype="float64")
    implied = va / rc

    if not math.isclose(implied[-1], 1.0, rel_tol=rtol, abs_tol=atol):
        raise ValueError(
            f"vendor adj_close not anchored at the last bar: adj/raw = {implied[-1]:.6f} != 1.0 "
            f"(globally mis-scaled series, e.g. cents vs dollars?). The validator requires a full "
            f"series through the vendor's adjustment horizon."
        )

    bad = ~np.isclose(factor, implied, rtol=rtol, atol=atol)
    if bad.any():
        dates = [str(d.date()) for d in index[bad]]
        raise ValueError(
            f"vendor adj_close inconsistent with events at {dates[:10]}"
            f"{'...' if len(dates) > 10 else ''}: "
            f"expected factor {np.round(factor[bad][:3], 6).tolist()}, "
            f"got {np.round(implied[bad][:3], 6).tolist()}"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_corpactions.py -q`
Expected: PASS (all Task-1/2/3 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/data/corpactions.py tests/test_corpactions.py
git commit -m "feat(data): check_adj_close_consistent validator (#149)"
```

---

### Task 4: Full quality gate + boundary verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. `lint-imports` must confirm `algua/data/corpactions.py` imports no `algua.*` module beyond what the existing data-layer contracts allow (it imports only `math`, `pandas`, `numpy`).

- [ ] **Step 2: Confirm the module is a pure leaf**

Run: `grep -nE "^(from|import) " algua/data/corpactions.py`
Expected: only `from __future__`, `math`, `collections.abc`, `dataclasses`, `numpy`, `pandas` — **no** `from algua...` import.

- [ ] **Step 3: If anything failed, fix and re-run; otherwise the slice is complete.**

No commit needed if Tasks 1-3 already committed and the gate is green.

---

## Self-review notes (author)

- **Spec coverage:** event types (T1) ✓; engine math incl. split/dividend/grouping/suffix-product/anchor/no-op/non-midnight guard/dividend≥close guard/raw guards (T2) ✓; validator anchor+shape+input guards (T3) ✓; purity/lint-imports (T4) ✓. Deferred items (importer wiring, adj_factor persistence, intraday, special distributions) are intentionally not tasks.
- **Type consistency:** `back_adjust` returns `[ts, adj_close, adj_factor]` (used by validator via `["adj_factor"]`); `Split.ratio`, `Dividend.cash`, `CorporateAction` names consistent across tasks.
- **No placeholders:** every code/test step contains the actual content.
