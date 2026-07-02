"""Training-ONLY feature-scaler fitters (issue #388).

This module derives statistics from data, so it is the ONE place that can fit a scaler. It is kept
OUT of every runtime lane by an import-linter `forbidden` contract (`algua.strategies` and
`algua.backtest` may not import it), which is what makes refit-at-serve structurally impossible:
a strategy literally cannot reach these functions. It is NOT re-exported from
`algua.features.__init__` for the same reason.

A fitter reads ONLY the train frame passed to it and returns an immutable `FrozenScaler` (see
`algua.features.scaling`) whose parameters — and `fit_max_timestamp` — are then embedded, offline,
inside the #376 model artifact bytes. Fit on the TRAIN split; the frozen scaler is applied unchanged
at serve.

Fit-window evidence: each fitter REQUIRES `train.index` be a monotonic-increasing `DatetimeIndex`
and stamps `fit_max_timestamp` from its validated maximum, so the recorded value is a meaningful
last-seen instant derived from the actual fit frame — not a free-form claim. Degenerate columns
(zero variance / fewer than 2 distinct quantile knots) fail closed at fit time rather than produce
a silently broken scaler.

Pure leaf: stdlib + numpy + pandas + `algua.contracts` only (no I/O).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from algua.contracts.model_types import normalize_as_of
from algua.features.scaling import QUANTILE, STANDARD, FrozenScaler, ScalerError


def _validate_fit_frame(train: pd.DataFrame, columns: Sequence[str]) -> str:
    """Validate the train frame and return the canonical `fit_max_timestamp` (UTC ISO). Fails
    closed on an empty frame, a non-`DatetimeIndex`/non-monotonic index (so the stamped timestamp
    is meaningful), or a missing/all-NaN feature column."""
    if not isinstance(columns, (list, tuple)) or len(columns) == 0:
        raise ScalerError("fit requires a non-empty list of columns")
    if len(train) == 0:
        raise ScalerError("fit requires a non-empty train frame")
    if not isinstance(train.index, pd.DatetimeIndex):
        raise ScalerError(
            "fit requires a DatetimeIndex train frame (the fit_max_timestamp evidence is derived "
            f"from it); got index type {type(train.index).__name__}"
        )
    if not train.index.is_monotonic_increasing:
        raise ScalerError("fit requires a monotonic-increasing DatetimeIndex train frame")
    missing = [c for c in columns if c not in train.columns]
    if missing:
        raise ScalerError(f"train frame is missing fit columns {missing}")
    return normalize_as_of(str(train.index.max())).isoformat()


def fit_standard_scaler(train: pd.DataFrame, *, columns: Sequence[str]) -> FrozenScaler:
    """Fit a per-column standard scaler (mean, population std ddof=0) on the TRAIN frame only.

    A zero-variance (degenerate) column has an undefined standard scaler and fails closed rather
    than divide by zero at serve. NaNs are ignored in the mean/std (nanmean/nanstd)."""
    fit_max_timestamp = _validate_fit_frame(train, columns)
    params: dict[str, dict[str, object]] = {}
    for col in columns:
        values = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ScalerError(f"column {col!r}: no finite values to fit a standard scaler")
        mean = float(np.mean(finite))
        std = float(np.std(finite, ddof=0))
        if not np.isfinite(std) or std <= 0.0:
            raise ScalerError(
                f"column {col!r}: zero/degenerate variance — a standard scaler is undefined "
                f"(std={std})"
            )
        params[col] = {"mean": mean, "std": std}
    return FrozenScaler(
        kind=STANDARD,
        columns=tuple(columns),
        params=params,
        fit_max_timestamp=fit_max_timestamp,
    )


def fit_quantile_scaler(
    train: pd.DataFrame, *, columns: Sequence[str], n_quantiles: int = 100
) -> FrozenScaler:
    """Fit a per-column robust quantile scaler on the TRAIN frame only. Stores `n_quantiles` sorted
    quantile knots per column; `transform` maps a value through them to a rank-uniform [0, 1] score
    (linear interpolation, clipped). Robust to outliers vs a standard scaler.

    A column with fewer than 2 DISTINCT finite values has no usable quantile map and fails closed.
    NaNs are ignored when computing the knots."""
    if n_quantiles < 2:
        raise ScalerError(f"n_quantiles must be >= 2, got {n_quantiles}")
    fit_max_timestamp = _validate_fit_frame(train, columns)
    params: dict[str, dict[str, object]] = {}
    grid = np.linspace(0.0, 1.0, num=n_quantiles)
    for col in columns:
        values = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ScalerError(f"column {col!r}: no finite values to fit a quantile scaler")
        knots = np.quantile(finite, grid)
        # Collapse duplicate knots (flat regions) to a strictly-increasing set. `transform`
        # re-derives an equally-spaced [0,1] grid of len(knots), so it stays aligned to whatever
        # count survives here.
        unique_knots = np.unique(knots)
        if unique_knots.size < 2:
            raise ScalerError(
                f"column {col!r}: fewer than 2 distinct quantile knots (degenerate/constant "
                f"column) — a quantile scaler is undefined"
            )
        params[col] = {"knots": [float(k) for k in unique_knots]}
    return FrozenScaler(
        kind=QUANTILE,
        columns=tuple(columns),
        params=params,
        fit_max_timestamp=fit_max_timestamp,
    )
