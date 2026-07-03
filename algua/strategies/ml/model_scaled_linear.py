"""Reference model-lane strategy with a PIT-fit feature scaler (issue #388).

Demonstrates the train/serve-skew fix end to end, composing the #388 scaling seam with the #376
model-artifact seam:

  1. OFFLINE (not here — a real author fits on a TRAIN split via `algua.features.scaling_fit`, which
     the import-linter forbids in any runtime lane so it can never be reached at serve). The fitted
     `FrozenScaler`'s parameters + `fit_max_timestamp` are what get pinned. Here the scaler is a
     PINNED CONSTANT (`_SCALER`), and `tests/test_model_scaled_linear.py` proves those exact
     parameters are what `fit_standard_scaler` produces on a representative train frame — so the
     embedded constant is verified against the real fitter without importing it into the strategy.
  2. The frozen scaler's `to_dict()` is embedded INSIDE the model artifact bytes alongside the
     linear coefficients. Those bytes are digest-pinned + provenance-committed + PIT-guarded by
     #376 — so the frozen scaler parameters are part of the run identity for free.
  3. AT SERVE, `signal(view, params, model)` deserializes the scaler, ENFORCES
     `assert_fit_before(scaler, model.version.training_as_of)` (fail closed BEFORE scoring — the
     load-time check that makes the fit-window evidence load-bearing), applies the FROZEN transform
     unchanged, and scores. No fitter is reachable from here, so serve can never refit or leak.

The "model" is trivial by design (per-symbol linear coefficients on two price features) — the seam,
not the model, is the point.

To run it: register the artifact, then `uv run algua backtest run model_scaled_linear ...`. The
engine enforces `training_as_of <= first decision bar` (a model that saw future data is refused);
the scaler's own `assert_fit_before` adds the independent `fit_max_timestamp <= training_as_of`
check at signal time.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from algua.contracts.model_types import ModelHandle, ModelRef, compute_provenance_digest
from algua.contracts.types import ExecutionContract
from algua.features.scaling import STANDARD, FrozenScaler, assert_fit_before
from algua.strategies.base import StrategyConfig

_FEATURES = ["return_1d", "return_5d"]
_COEFFICIENTS = {
    "AAPL": [1.0, 0.5],
    "MSFT": [0.8, 0.4],
    "NVDA": [1.2, 0.6],
}
_INTERCEPT = 0.0

TRAINING_SNAPSHOT_ID = "demo-train-2019"
TRAINING_AS_OF = "2020-01-01"  # the model + scaler saw nothing after 2019 -> safe for a 2023 run
HYPERPARAMETERS: dict[str, Any] = {"ridge_alpha": 0.1, "scaler": "standard"}
SEED = 11
EVAL_REPORT: dict[str, Any] = {"train_ic": 0.04}

# The PINNED, already-fitted scaler. Constructed directly (serve-safe — `scaling.py` has no fitter),
# with `fit_max_timestamp` inside the 2019 train window, hence <= TRAINING_AS_OF. These exact
# parameters are what `fit_standard_scaler` yields on the representative train frame in
# `tests/test_model_scaled_linear.py`, which is where the fitter (a training-only module) is
# exercised — never from this strategy module.
_SCALER = FrozenScaler(
    kind=STANDARD,
    columns=tuple(_FEATURES),
    params={
        "return_1d": {"mean": 0.0, "std": 0.01},
        "return_5d": {"mean": 0.0, "std": 0.02},
    },
    fit_max_timestamp="2019-12-31T00:00:00+00:00",
)

# The canonical, immutable model artifact: the fitted scaler + the linear model, serialized once.
_MODEL = {
    "scaler": _SCALER.to_dict(),
    "features": _FEATURES,
    "coefficients": _COEFFICIENTS,
    "intercept": _INTERCEPT,
}
ARTIFACT_BYTES = json.dumps(_MODEL, sort_keys=True).encode()

_DIGEST = hashlib.sha256(ARTIFACT_BYTES).hexdigest()[:16]
_PROVENANCE_DIGEST = compute_provenance_digest(
    digest=_DIGEST,
    training_snapshot_id=TRAINING_SNAPSHOT_ID,
    training_as_of=TRAINING_AS_OF,
    code_hash=None,
    hyperparameters=HYPERPARAMETERS,
    seed=SEED,
    eval_report=EVAL_REPORT,
)

CONFIG = StrategyConfig(
    name="model_scaled_linear",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={},
    construction="equal_weight_positive",
    needs_model=True,
    model_ref=ModelRef(
        name="model_scaled_linear",
        version=1,
        digest=_DIGEST,
        training_as_of=TRAINING_AS_OF,
        provenance_digest=_PROVENANCE_DIGEST,
    ),
    feature_lookback=5,  # return_5d needs a 5-bar warmup window (sizes the walk-forward embargo)
)


def signal(view: pd.DataFrame, params: dict[str, Any], model: ModelHandle) -> pd.Series:
    """Score each symbol with its pinned linear coefficients on SCALER-NORMALIZED 1d/5d returns.

    The model (linear coefficients + the frozen scaler) is deserialized from `model.artifact_bytes`
    in-memory (pure — no file I/O here; the loader already resolved and injected the handle). Before
    scoring, `assert_fit_before(scaler, model.version.training_as_of)` fails closed if the scaler's
    fit window ran past the model's PIT cutoff — the serve-time enforcement of the fit-window
    evidence. The frozen scaler is applied UNCHANGED (fit-on-train, frozen-apply), so there is no
    train/serve skew and no way to refit at serve."""
    blob = json.loads(model.artifact_bytes.decode())
    scaler = FrozenScaler.from_dict(blob["scaler"])
    # Load-time PIT enforcement: refuse a scaler that saw data past the model's training cutoff,
    # even if #376's `training_as_of <= first decision bar` guard already passed.
    assert_fit_before(scaler, model.version.training_as_of)

    coeffs: dict[str, list[float]] = blob["coefficients"]
    intercept = float(blob["intercept"])

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) < 6:  # need t and t-5 for return_5d
        return pd.Series(0.0, index=wide.columns, dtype="float64")

    last = wide.iloc[-1]
    ret_1d = (last / wide.iloc[-2]) - 1.0
    ret_5d = (last / wide.iloc[-6]) - 1.0

    # One row per symbol of the raw features, scaled through the FROZEN transform.
    raw = pd.DataFrame({"return_1d": ret_1d, "return_5d": ret_5d}, index=wide.columns)
    scaled = scaler.transform(raw)

    scores: dict[Any, float] = {}
    for sym in wide.columns:
        c = coeffs.get(str(sym))
        f1 = scaled["return_1d"].get(sym)
        f5 = scaled["return_5d"].get(sym)
        if c is None or f1 is None or f5 is None or pd.isna(f1) or pd.isna(f5):
            scores[sym] = 0.0
            continue
        scores[sym] = intercept + c[0] * float(f1) + c[1] * float(f5)
    return pd.Series(scores, dtype="float64")
