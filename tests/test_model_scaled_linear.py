"""End-to-end reference: the `model_scaled_linear` strategy (#388) binds a PIT-fit FROZEN scaler
inside the #376 model artifact, enforces the fit-window at serve, and applies the frozen transform
unchanged (no train/serve skew, no refit-at-serve)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import run as run_backtest
from algua.contracts.model_types import ModelHandle
from algua.features.scaling import FrozenScaler, ScalerError
from algua.models import register
from algua.strategies.loader import load_strategy


@pytest.fixture
def registry_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    return tmp_path / "models"


def _register_reference(**over):
    from algua.strategies.ml import model_scaled_linear as m

    kw = dict(
        training_snapshot_id=m.TRAINING_SNAPSHOT_ID,
        training_as_of=m.TRAINING_AS_OF,
        code_hash=None,
        hyperparameters=m.HYPERPARAMETERS,
        seed=m.SEED,
        eval_report=m.EVAL_REPORT,
    )
    kw.update(over)
    return register("model_scaled_linear", m.ARTIFACT_BYTES, **kw)


def test_pinned_scaler_matches_the_real_fitter() -> None:
    """The embedded FROZEN scaler constant is EXACTLY what the training-only fitter produces on a
    representative train frame — verifying the constant against `fit_standard_scaler` WITHOUT the
    strategy module ever importing the (forbidden-in-runtime) fitter."""
    from algua.features.scaling_fit import fit_standard_scaler
    from algua.strategies.ml import model_scaled_linear as m

    # A train frame whose per-column mean/std are exactly the pinned params, ending in 2019.
    idx = pd.date_range("2019-12-27", periods=2, freq="D")  # max = 2019-12-28 <= 2019-12-31
    train = pd.DataFrame({"return_1d": [-0.01, 0.01], "return_5d": [-0.02, 0.02]}, index=idx)
    fitted = fit_standard_scaler(train, columns=m._FEATURES)
    assert fitted.params == m._SCALER.params
    # The pinned fit_max_timestamp is within the train window and <= TRAINING_AS_OF.
    from algua.contracts.model_types import normalize_as_of

    assert normalize_as_of(m._SCALER.fit_max_timestamp) <= normalize_as_of(m.TRAINING_AS_OF)


def test_config_matches_registered_artifact(registry_root) -> None:
    v = _register_reference()
    from algua.strategies.ml import model_scaled_linear as m

    assert v.digest == m.CONFIG.model_ref.digest
    assert v.provenance_digest == m.CONFIG.model_ref.provenance_digest


def test_load_binds_model_handle(registry_root) -> None:
    _register_reference()
    loaded = load_strategy("model_scaled_linear")
    assert loaded.config.needs_model
    assert loaded.model_handle is not None
    assert loaded.model_signal_fn is not None


def test_reference_backtests_with_scaled_features(registry_root) -> None:
    _register_reference()
    loaded = load_strategy("model_scaled_linear")
    result = run_backtest(
        loaded, SyntheticProvider(seed=3), datetime(2023, 1, 1), datetime(2023, 6, 30)
    )
    assert result.strategy == "model_scaled_linear"
    assert result.model_ref is not None
    assert result.model_ref["digest"] == loaded.config.model_ref.digest


def test_signal_refuses_scaler_fit_past_training_cutoff() -> None:
    """A serve-time guard, independent of the #376 PIT guard: swap in an artifact whose scaler was
    fit PAST training_as_of and the signal fails closed BEFORE scoring — the load-time enforcement
    that makes the fit-window evidence load-bearing."""
    import json

    from algua.strategies.ml import model_scaled_linear as m

    # A scaler whose fit window ends in 2023 (after the 2020 training_as_of) — a leak.
    leaky = FrozenScaler(
        kind="standard",
        columns=("return_1d", "return_5d"),
        params={
            "return_1d": {"mean": 0.0, "std": 0.01},
            "return_5d": {"mean": 0.0, "std": 0.02},
        },
        fit_max_timestamp="2023-06-30T00:00:00+00:00",
    )
    blob = json.loads(m.ARTIFACT_BYTES.decode())
    blob["scaler"] = leaky.to_dict()
    tampered_bytes = json.dumps(blob, sort_keys=True).encode()

    # Build a ModelHandle whose version.training_as_of is the pinned 2020 cutoff.
    version = _make_version(training_as_of="2020-01-01")
    handle = ModelHandle(version=version, artifact_bytes=tampered_bytes)

    view = _tiny_view()
    with pytest.raises(ScalerError, match="AFTER the model's training_as_of"):
        m.signal(view, {}, handle)


def _make_version(*, training_as_of: str):
    from algua.contracts.model_types import ModelVersion

    return ModelVersion(
        name="model_scaled_linear",
        version=1,
        digest="deadbeefdeadbeef",
        created_at="2020-01-01T00:00:00+00:00",
        training_snapshot_id="x",
        training_as_of=training_as_of,
        code_hash=None,
        hyperparameters={},
        seed=0,
        eval_report={},
        artifact_path=None,
        provenance_digest="",
    )


def _tiny_view() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=8, freq="B")
    rows = []
    for ts in idx:
        for sym in ("AAPL", "MSFT", "NVDA"):
            rows.append({"timestamp": ts, "symbol": sym, "adj_close": 100.0 + hash((ts, sym)) % 10})
    return pd.DataFrame(rows).set_index("timestamp")
