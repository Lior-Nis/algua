"""End-to-end model lane (#376): the reference `model_linear_scores` strategy loads its pinned
model from the registry, the loader fails closed on a pin mismatch, and the engine enforces the
training_as_of PIT guard + fails closed in walk-forward/sweep."""
from __future__ import annotations

from datetime import datetime

import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
from algua.backtest.engine import run as run_backtest
from algua.backtest.sweep import sweep
from algua.backtest.walkforward import walk_forward
from algua.models import register
from algua.strategies.loader import StrategyNotFound, load_strategy, load_tradable_strategy


@pytest.fixture
def registry_root(tmp_path, monkeypatch):
    """Point the model registry (and everything under data_dir) at a temp dir."""
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    return tmp_path / "models"


def _register_reference(**over):
    from algua.strategies.ml import model_linear_scores as m
    kw = dict(
        training_snapshot_id=m.TRAINING_SNAPSHOT_ID,
        training_as_of=m.TRAINING_AS_OF,
        code_hash=None,
        hyperparameters=m.HYPERPARAMETERS,
        seed=m.SEED,
        eval_report=m.EVAL_REPORT,
    )
    kw.update(over)
    return register("model_linear_scores", m.ARTIFACT_BYTES, **kw)


def test_reference_strategy_config_matches_registered_artifact(registry_root):
    """The strategy's pinned model_ref digests are DERIVED from the canonical artifact + metadata,
    so registering that artifact yields a version the loader accepts."""
    v = _register_reference()
    from algua.strategies.ml import model_linear_scores as m
    assert v.digest == m.CONFIG.model_ref.digest
    assert v.provenance_digest == m.CONFIG.model_ref.provenance_digest


def test_load_reference_strategy_binds_model_handle(registry_root):
    _register_reference()
    loaded = load_strategy("model_linear_scores")
    assert loaded.config.needs_model
    assert loaded.model_handle is not None
    assert loaded.model_signal_fn is not None
    # The handle carries the exact artifact bytes.
    from algua.strategies.ml import model_linear_scores as m
    assert loaded.model_handle.artifact_bytes == m.ARTIFACT_BYTES


def test_loader_fails_closed_when_model_not_registered(registry_root):
    with pytest.raises(StrategyNotFound):
        load_strategy("model_linear_scores")


def test_loader_fails_closed_on_pin_mismatch(registry_root):
    """Register a DIFFERENT artifact as v1 so the on-disk digest no longer matches the pinned ref;
    the loader must refuse to bind (can't silently run a different model)."""
    register("model_linear_scores", b"a-different-model", training_snapshot_id="x",
             training_as_of="2020-01-01", code_hash=None, hyperparameters={}, seed=0,
             eval_report={})
    with pytest.raises(StrategyNotFound, match="does not match"):
        load_strategy("model_linear_scores")


def test_reference_strategy_backtests(registry_root):
    _register_reference()
    loaded = load_strategy("model_linear_scores")
    result = run_backtest(
        loaded, SyntheticProvider(seed=1), datetime(2023, 1, 1), datetime(2023, 6, 30)
    )
    assert result.strategy == "model_linear_scores"
    # Result carries explicit model provenance stamped at the engine boundary.
    assert result.model_ref is not None
    assert result.model_ref["digest"] == loaded.config.model_ref.digest
    assert result.model_ref["version"] == 1
    assert result.to_dict()["model_ref"]["provenance_digest"] == (
        loaded.config.model_ref.provenance_digest
    )


def test_engine_rejects_model_trained_after_backtest_start(registry_root):
    """PIT guard: a model whose training_as_of is AFTER the first decision bar is refused at the
    engine (mandatory boundary), even for a valid, pin-matched model."""
    from algua.contracts.model_types import ModelHandle
    from algua.strategies.base import LoadedStrategy
    from algua.strategies.ml import model_linear_scores as m
    _register_reference()  # v1 (2020) so load_strategy binds cleanly
    # v2: same artifact but trained on 2024 data (a future-leaking model for a 2023 backtest).
    v2 = register("model_linear_scores", m.ARTIFACT_BYTES, training_snapshot_id="late",
                  training_as_of="2024-06-01", code_hash=None, hyperparameters=m.HYPERPARAMETERS,
                  seed=m.SEED, eval_report=m.EVAL_REPORT)
    loaded = load_strategy("model_linear_scores")
    # Build a loaded strategy PINNED to the late v2 (pin matches the registry, so the loader would
    # accept it — the ONLY thing that stops the run is the engine PIT guard).
    late_cfg = loaded.config.model_copy(update={"model_ref": v2.as_ref()})
    late = LoadedStrategy(
        config=late_cfg, model_signal_fn=loaded.model_signal_fn,
        model_handle=ModelHandle(version=v2, artifact_bytes=m.ARTIFACT_BYTES),
        construct_fn=loaded.construct_fn,
    )
    with pytest.raises(BacktestError, match="future data|look-ahead"):
        run_backtest(late, SyntheticProvider(seed=1), datetime(2023, 1, 1), datetime(2023, 6, 30))


def test_walk_forward_fails_closed_on_model(registry_root):
    _register_reference()
    loaded = load_strategy("model_linear_scores")
    with pytest.raises(BacktestError, match="needs_model"):
        walk_forward(loaded, SyntheticProvider(seed=1), datetime(2023, 1, 1), datetime(2023, 6, 30))


def test_sweep_fails_closed_on_model(registry_root):
    _register_reference()
    loaded = load_strategy("model_linear_scores")
    with pytest.raises(BacktestError, match="needs_model"):
        sweep(loaded, SyntheticProvider(seed=1), datetime(2023, 1, 1), datetime(2023, 6, 30),
              grid={"unused": [1]})


def test_needs_model_not_tradable(registry_root):
    _register_reference()
    with pytest.raises(ValueError, match="needs_model"):
        load_tradable_strategy("model_linear_scores")
