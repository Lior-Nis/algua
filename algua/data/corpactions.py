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
