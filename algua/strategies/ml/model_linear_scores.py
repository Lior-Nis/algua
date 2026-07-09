"""Reference model-lane strategy (issue #376): a MINIMAL demonstration of binding a versioned,
PIT-pinned model artifact to `signal()`.

The model artifact is just a JSON blob of per-symbol linear coefficients over a couple of price
features — deliberately trivial so the seam, not the model, is the point. The strategy DESERIALIZES
`model.artifact_bytes` in-memory (pure; no file I/O in this module — the loader already resolved
and injected the handle), computes a score per symbol, and hands it to the construction policy.

Binding contract:
  - `needs_model=True` + a PINNED `model_ref` (name, version, digest, training_as_of,
    provenance_digest). The loader resolves that exact version from the model registry and fails
    closed unless the on-disk artifact + provenance match the pin — so this strategy can never
    silently run a different model than the one its config was validated against.
  - The artifact + training metadata are canonical constants HERE, and the digests in `model_ref`
    are DERIVED from them, so CONFIG is always self-consistent with what
    `register(...ARTIFACT_BYTES...)` produces (see `tests/test_model_reference_strategy.py`).

To actually run it: register the artifact first, e.g. inside `<data_dir>/models`:

    from algua.tracking.mlflow_tracker import log_model  # or algua.models.register directly
    register("model_linear_scores", ARTIFACT_BYTES,
             training_snapshot_id=TRAINING_SNAPSHOT_ID, training_as_of=TRAINING_AS_OF,
             code_hash=None, hyperparameters=HYPERPARAMETERS, seed=SEED, eval_report=EVAL_REPORT)

then `uv run algua backtest run model_linear_scores --start 2023-01-01 --end 2023-06-30`. The
engine enforces `training_as_of <= first decision bar` (a model that saw future data is refused).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from algua.contracts.model_types import (
    ModelHandle,
    ModelRef,
    compute_provenance_digest,
)
from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

# --- the canonical, immutable model artifact + its training provenance ------------------------- #
# A trained "model": per-symbol linear coefficients on (return_1d, return_5d). Trivial by design.
_MODEL = {
    "features": ["return_1d", "return_5d"],
    "coefficients": {
        "AAPL": [1.0, 0.5],
        "MSFT": [0.8, 0.4],
        "NVDA": [1.2, 0.6],
    },
    "intercept": 0.0,
}
ARTIFACT_BYTES = json.dumps(_MODEL, sort_keys=True).encode()

TRAINING_SNAPSHOT_ID = "demo-train-2019"
TRAINING_AS_OF = "2020-01-01"  # the model saw nothing after 2019 -> safe for a 2023 backtest
HYPERPARAMETERS: dict[str, Any] = {"ridge_alpha": 0.1}
SEED = 7
EVAL_REPORT: dict[str, Any] = {"train_ic": 0.03}

# Digest of the artifact bytes (matches the model registry's sha256[:16]).
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

# Provenance marker (additions-only discipline): these bundled seed examples are hand-authored.
# Informational only — read by `algua doctor`'s advisory generated_provenance probe, NOT a trust
# or authorization control.
GENERATED_BY = "human"

CONFIG = StrategyConfig(
    name="model_linear_scores",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={},
    construction="equal_weight_positive",
    needs_model=True,
    model_ref=ModelRef(
        name="model_linear_scores",
        version=1,
        digest=_DIGEST,
        training_as_of=TRAINING_AS_OF,
        provenance_digest=_PROVENANCE_DIGEST,
    ),
    feature_lookback=5,  # return_5d needs a 5-bar warmup window (sizes the walk-forward embargo)
)


def signal(view: pd.DataFrame, params: dict[str, Any], model: ModelHandle) -> pd.Series:
    """Score each symbol with its pinned linear coefficients on 1d/5d returns.

    `view` is the expanding per-bar history in LONG bar-schema form (index=timestamp; columns
    include `symbol`, `adj_close`, …). The model is already bound (deserialize from
    `model.artifact_bytes` — pure, in-memory; no file I/O in this module). A symbol with no
    coefficient row, or too little history for its features, scores 0 (not held)."""
    blob = json.loads(model.artifact_bytes.decode())
    coeffs: dict[str, list[float]] = blob["coefficients"]
    intercept = float(blob["intercept"])

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) < 6:  # need t and t-5 for return_5d
        return pd.Series(0.0, index=wide.columns, dtype="float64")

    last = wide.iloc[-1]
    ret_1d = (last / wide.iloc[-2]) - 1.0
    ret_5d = (last / wide.iloc[-6]) - 1.0

    scores: dict[Any, float] = {}
    for sym in wide.columns:
        c = coeffs.get(str(sym))
        r1 = ret_1d.get(sym)
        r5 = ret_5d.get(sym)
        if c is None or r1 is None or r5 is None or pd.isna(r1) or pd.isna(r5):
            scores[sym] = 0.0
            continue
        scores[sym] = intercept + c[0] * float(r1) + c[1] * float(r5)
    return pd.Series(scores, dtype="float64")
