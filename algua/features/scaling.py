"""Serve-safe PIT feature-scaling seam (issue #388).

The single most common silent ML failure is train/serve skew: a stateful scaler (mean/std,
quantile) fit over the WHOLE series leaks future statistics into a backtest, and then a different
transform runs at serve time. This module is the SERVE-SAFE half of the fix — the ONLY scaling
module a strategy is allowed to import. It contains `FrozenScaler` (already-fitted parameters +
a pure `.transform`) and nothing that can fit. The training-only fitters live in
`algua.features.scaling_fit`, which an import-linter `forbidden` contract keeps OUT of every
runtime lane (`algua.strategies`, `algua.backtest`), so a strategy cannot refit-at-serve.

Composition with the #376 model-artifact seam (the KISS/DRY design):
  A `FrozenScaler` serializes to a plain dict (`to_dict`) that an ML author embeds INSIDE the
  model artifact bytes. Those bytes are already digest-pinned (`ModelRef.digest`), provenance-
  committed (`provenance_digest`), and PIT-guarded (`training_as_of <= first decision bar`) by
  #376 — so the frozen scaler parameters inherit ALL of those guarantees for free, with NO new
  registry and NO new identity mechanism. Changing a scaler parameter changes the artifact bytes,
  hence the digest, hence `config_hash`, hence a prior live approval is invalidated.

Fit-window evidence (the leakage guard):
  A `FrozenScaler` carries `fit_max_timestamp` — the max index timestamp of the frame it was fit
  on, DERIVED at fit time from a validated monotonic `DatetimeIndex` (see `scaling_fit`). At serve
  time the strategy calls `assert_fit_before(scaler, model.version.training_as_of)` BEFORE scoring,
  so a scaler whose fit window ran past the model's PIT cutoff is refused even when #376's own
  `training_as_of <= first decision bar` guard would pass. The chain enforced is
  `fit_max_timestamp <= training_as_of <= first decision bar`.

Trust boundary (stated honestly — identical to #376's): `fit_max_timestamp` is tamper-EVIDENT
(committed by the artifact digest), not a cryptographic proof of which rows produced the fitted
floats — no offline-fit artifact system can prove that, and #376's `training_as_of` shares the
same model. What this seam adds beyond a bare string is (a) the value is derived from a validated
timestamp index, (b) it is re-checkable at load via `assert_fit_before`, and (c) the first-party
fitters are structurally unreachable from serve. A determined author hand-rolling future-aware
numpy inside `signal()` to deliberately leak is the training-pipeline-ownership problem the issue
explicitly defers; it is out of scope for a pure feature library.

This module is a pure leaf: stdlib + numpy + pandas + `algua.contracts` only (no I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from algua.contracts.model_types import normalize_as_of

STANDARD = "standard"
QUANTILE = "quantile"


class ScalerError(Exception):
    """Any feature-scaling failure — a malformed frozen dict, a missing/unknown column at
    transform, or a fit-window that runs past the model's PIT cutoff. Callers fail closed on it."""


@dataclass(frozen=True)
class FrozenScaler:
    """An already-fitted, immutable feature scaler. Constructed ONLY from fitted parameters — it
    has no way to derive statistics from data, so it cannot fit (and therefore cannot leak at
    serve time). Build one via `algua.features.scaling_fit.fit_*` offline; embed `to_dict()` in a
    model artifact; reconstruct with `from_dict()` and apply with `transform()` at serve.

    Parameters
    ----------
    kind:
        `"standard"` (per-column mean/std, ddof=0) or `"quantile"` (per-column rank-uniform map
        via stored quantile knots).
    columns:
        The ordered feature columns this scaler transforms. `transform` requires every one to be
        present (fail closed on a missing column).
    params:
        Per-column fitted parameters. For `standard`: `{col: {"mean": float, "std": float}}`.
        For `quantile`: `{col: {"knots": [float, ...]}}` — `n_quantiles` sorted, strictly
        increasing quantile values mapped to an equally-spaced grid on [0, 1].
    fit_max_timestamp:
        ISO-8601 max index timestamp of the frame the scaler was fit on (the fit-window evidence).
    """

    kind: str
    columns: tuple[str, ...]
    params: dict[str, dict[str, Any]]
    fit_max_timestamp: str

    def __post_init__(self) -> None:
        if self.kind not in (STANDARD, QUANTILE):
            raise ScalerError(f"unknown scaler kind {self.kind!r}")
        if not self.columns:
            raise ScalerError("FrozenScaler requires at least one column")
        missing = [c for c in self.columns if c not in self.params]
        if missing:
            raise ScalerError(f"missing fitted params for columns {missing}")
        # `fit_max_timestamp` must parse to a comparable instant — fail closed on garbage so the
        # PIT evidence can never be a value `assert_fit_before` silently mis-handles.
        try:
            normalize_as_of(self.fit_max_timestamp)
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            raise ScalerError(
                f"fit_max_timestamp {self.fit_max_timestamp!r} is not a valid timestamp"
            ) from exc
        for col in self.columns:
            p = self.params[col]
            if self.kind == STANDARD:
                std = p.get("std")
                if std is None or not np.isfinite(std) or std <= 0.0:
                    raise ScalerError(
                        f"column {col!r}: standard scaler needs a finite positive std, got {std!r}"
                    )
                mean = p.get("mean")
                if mean is None or not np.isfinite(mean):
                    raise ScalerError(f"column {col!r}: standard scaler needs a finite mean")
            else:
                knots = p.get("knots")
                if not isinstance(knots, (list, tuple)) or len(knots) < 2:
                    raise ScalerError(
                        f"column {col!r}: quantile scaler needs >= 2 knots, got {knots!r}"
                    )
                arr = np.asarray(knots, dtype="float64")
                if not np.all(np.isfinite(arr)):
                    raise ScalerError(f"column {col!r}: quantile knots must be finite")
                if not np.all(np.diff(arr) > 0.0):
                    raise ScalerError(
                        f"column {col!r}: quantile knots must be strictly increasing"
                    )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen transform. Pure: uses ONLY the stored parameters — no statistic is
        derived from `df`, so this can never fit or leak. Returns a NEW frame with the scaler's
        `columns` replaced by their scaled values (other columns pass through unchanged). Fails
        closed if any required column is absent."""
        absent = [c for c in self.columns if c not in df.columns]
        if absent:
            raise ScalerError(f"transform input missing required columns {absent}")
        out = df.copy()
        for col in self.columns:
            values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype="float64")
            p = self.params[col]
            if self.kind == STANDARD:
                out[col] = (values - float(p["mean"])) / float(p["std"])
            else:
                knots = np.asarray(p["knots"], dtype="float64")
                grid = np.linspace(0.0, 1.0, num=len(knots))
                # Map through the fitted CDF knots (rank-uniform), clipped to [0, 1]. NaNs stay NaN.
                scaled = np.interp(values, knots, grid, left=0.0, right=1.0)
                scaled = np.where(np.isnan(values), np.nan, scaled)
                out[col] = scaled
        return out

    def to_dict(self) -> dict[str, Any]:
        """Canonical serializable form (embed inside a model artifact). Column order preserved."""
        return {
            "kind": self.kind,
            "columns": list(self.columns),
            "params": {c: dict(self.params[c]) for c in self.columns},
            "fit_max_timestamp": self.fit_max_timestamp,
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> FrozenScaler:
        """Reconstruct from `to_dict()`. Fails closed (`ScalerError`) on a malformed blob so a
        corrupt/forged scaler never silently transforms into an identity/garbage transform."""
        try:
            kind = blob["kind"]
            columns = tuple(blob["columns"])
            params = {c: dict(blob["params"][c]) for c in columns}
            fit_max_timestamp = blob["fit_max_timestamp"]
        except (KeyError, TypeError) as exc:
            raise ScalerError(f"malformed FrozenScaler dict: {exc}") from exc
        return cls(
            kind=kind, columns=columns, params=params, fit_max_timestamp=fit_max_timestamp
        )


def assert_fit_before(scaler: FrozenScaler, training_as_of: str) -> None:
    """Fail closed unless the scaler's fit window ended at or before `training_as_of` — i.e.
    `normalize(fit_max_timestamp) <= normalize(training_as_of)`. This is the load-time enforcement
    that makes the fit-window evidence load-bearing: a strategy MUST call it (with the pinned
    model's `training_as_of`) BEFORE using the scaler, so a scaler fit over post-cutoff (e.g.
    whole-series) data is refused even when #376's `training_as_of <= first decision bar` PIT guard
    passes. Both sides normalize via #376's `normalize_as_of` (bare date => 00:00:00 UTC) so the
    comparison is a single unambiguous UTC instant."""
    fit_max = normalize_as_of(scaler.fit_max_timestamp)
    cutoff = normalize_as_of(training_as_of)
    if fit_max > cutoff:
        raise ScalerError(
            f"scaler fit window ends {scaler.fit_max_timestamp} which is AFTER the model's "
            f"training_as_of {training_as_of} — refusing to serve a scaler that saw data past the "
            f"PIT cutoff (train/serve leakage)"
        )
