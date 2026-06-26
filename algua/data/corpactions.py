"""Pure corporate-action back-adjustment engine (#149).

Given a raw OHLC frame plus a typed split/dividend event list for one symbol, produce the
back-adjusted close (`adj_close`) and the cumulative adjustment factor; plus a validator that checks
a vendor-supplied `adj_close` against the same events (reverse-split-safe). No I/O; imports only
pandas + numpy. See docs/superpowers/specs/
2026-06-10-corporate-action-back-adjustment-engine-issue-149-design.md.
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


def back_adjust(raw: pd.DataFrame, events: Iterable[CorporateAction]) -> pd.DataFrame:
    """Back-adjust `raw['close']` for `events`, anchored at the most recent bar.

    Returns a frame `[ts, adj_close, adj_factor]`, one row per input bar in input order, with
    `adj_close = close * adj_factor`. Pure; see the module docstring + design spec.
    """
    if "ts" not in raw.columns or "close" not in raw.columns:
        raise ValueError("raw must have 'ts' and 'close' columns")

    ts = pd.DatetimeIndex(raw["ts"])
    n = len(ts)
    if n > 0 and str(ts.tz) != "UTC":
        raise ValueError("raw['ts'] must be tz-aware UTC")
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
                    f"dividend cash {cash} >= prior close {p_prev} on {ex.date()}: "
                    f"if this is a liquidating / return-of-capital distribution, "
                    f"exclude it from the event list; "
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


def check_adj_close_anchored(
    raw_close: pd.Series,
    vendor_adj: pd.Series,
    *,
    rtol: float = 1e-3,
    atol: float = 5e-3,
) -> None:
    """Structural (NO-events) subset of `check_adj_close_consistent`: assert `vendor_adj` is finite,
    positive, index-aligned with `raw_close`, and ANCHORED at the last bar (adj/raw ≈ 1.0). Catches
    a globally mis-scaled / mis-anchored adjusted series (e.g. cents vs dollars) with no event list.

    PRECONDITION: a FULL symbol series through the vendor's adjustment anchor — the last bar is the
    most recent, where a back-adjusted series has adj_close == raw close. A windowed series whose
    last bar is NOT the anchor (e.g. Alpaca `adjustment=all` over [start, end] with events after
    `end`) would false-positive, so do NOT apply this to such a series. Raises `ValueError`.
    """
    for name, series in (("raw_close", raw_close), ("vendor_adj", vendor_adj)):
        if not isinstance(series.index, pd.DatetimeIndex) or str(series.index.tz) != "UTC":
            raise ValueError(f"{name} must have a tz-aware UTC DatetimeIndex")
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
    implied = va / rc
    if not math.isclose(implied[-1], 1.0, rel_tol=rtol, abs_tol=atol):
        raise ValueError(
            f"vendor adj_close not anchored at the last bar: adj/raw = {implied[-1]:.6f} != 1.0 "
            f"(globally mis-scaled series, e.g. cents vs dollars?). The check requires a full "
            f"series through the vendor's adjustment horizon."
        )


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
    # Structural + anchored checks first (index/finite/positive/anchor), then the event check.
    check_adj_close_anchored(raw_close, vendor_adj, rtol=rtol, atol=atol)
    index = raw_close.index
    if len(index) == 0:
        return
    rc = raw_close.to_numpy(dtype="float64")
    va = vendor_adj.to_numpy(dtype="float64")
    factor = back_adjust(pd.DataFrame({"ts": index, "close": rc}), events)[
        "adj_factor"
    ].to_numpy(dtype="float64")
    implied = va / rc
    bad = ~np.isclose(factor, implied, rtol=rtol, atol=atol)
    if bad.any():
        dates = [str(d.date()) for d in index[bad]]
        raise ValueError(
            f"vendor adj_close inconsistent with events at {dates[:10]}"
            f"{'...' if len(dates) > 10 else ''}: "
            f"expected factor {np.round(factor[bad][:3], 6).tolist()}, "
            f"got {np.round(implied[bad][:3], 6).tolist()}"
        )
