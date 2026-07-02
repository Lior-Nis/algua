"""Model lane (#376): StrategyConfig/LoadedStrategy exclusivity, config_hash stability + model
identity, and loader pin-match / fail-closed resolution."""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from algua.contracts.model_types import (
    ModelHandle,
    ModelRef,
    ModelVersion,
    compute_provenance_digest,
)
from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash


def _plain_cfg(**over) -> StrategyConfig:
    base = dict(
        name="s", universe=["A", "B"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        params={}, construction="equal_weight_positive",
    )
    base.update(over)
    return StrategyConfig(**base)


def _plain_loaded(cfg: StrategyConfig) -> LoadedStrategy:
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(dtype="float64"),
        construct_fn=get_construction_policy(cfg.construction),
    )


def _consistent(name="m", version=1, artifact=b"model-bytes", as_of="2020-01-01"):
    """Build a self-consistent (ModelVersion, ModelRef, ModelHandle) trio: the digest is the real
    sha256 of `artifact` and the provenance_digest is reproduced from the version metadata, so it
    passes LoadedStrategy.__post_init__'s bytes+provenance verification."""
    digest = hashlib.sha256(artifact).hexdigest()[:16]
    prov = compute_provenance_digest(
        digest=digest, training_snapshot_id=None, training_as_of=as_of,
        code_hash=None, hyperparameters={}, seed=None, eval_report={},
    )
    mv = ModelVersion(
        name=name, version=version, digest=digest, created_at="", training_snapshot_id=None,
        training_as_of=as_of, code_hash=None, hyperparameters={}, seed=None, eval_report={},
        provenance_digest=prov,
    )
    ref = ModelRef(name=name, version=version, digest=digest, training_as_of=as_of,
                   provenance_digest=prov)
    return mv, ref, ModelHandle(version=mv, artifact_bytes=artifact)


def _ref() -> ModelRef:
    return _consistent()[1]


def test_config_hash_unchanged_for_non_model_strategy():
    """Adding the model lane must NOT churn a non-model strategy's config_hash — the model block
    is folded in ONLY when needs_model=True, so existing live approvals / identity are stable."""
    cfg = _plain_cfg(params={"lookback": 10}, construction="top_k_equal_weight",
                     construction_params={"top_k": 2})
    # This literal is the sha256[:32] of the canonical identity payload WITHOUT any model keys.
    # If a future change folds model/other keys in unconditionally, this pins the regression.
    h = config_hash(_plain_loaded(cfg))
    # Recompute independently to prove the model block is absent from the payload.
    import hashlib
    import json
    from dataclasses import asdict
    payload = json.dumps({
        "name": "s", "universe": ["A", "B"], "params": {"lookback": 10},
        "execution": asdict(cfg.execution), "construction": "top_k_equal_weight",
        "construction_params": {"top_k": 2}, "needs_fundamentals": False, "needs_news": False,
        "feature_lookback": cfg.feature_lookback,
    }, sort_keys=True, allow_nan=False)
    assert h == hashlib.sha256(payload.encode()).hexdigest()[:32]


def test_needs_model_requires_model_ref():
    """needs_model=True without a model_ref fails closed at LoadedStrategy construction."""
    from algua.contracts.model_types import ModelHandle, ModelVersion
    cfg = _plain_cfg(needs_model=True)  # no model_ref
    with pytest.raises(ValueError, match="requires model_ref"):
        LoadedStrategy(
            config=cfg, model_signal_fn=lambda v, p, m: pd.Series(dtype="float64"),
            model_handle=ModelHandle(
                version=ModelVersion(name="m", version=1, digest="d0", created_at="",
                                     training_snapshot_id=None, training_as_of="2020-01-01",
                                     code_hash=None, provenance_digest="p0"),
                artifact_bytes=b"x"),
            construct_fn=get_construction_policy(cfg.construction),
        )


def test_model_ref_forbidden_without_needs_model():
    from algua.contracts.model_types import ModelHandle, ModelVersion
    cfg = _plain_cfg(model_ref=_ref())  # needs_model defaults False
    with pytest.raises(ValueError, match="model_ref set but needs_model is False"):
        LoadedStrategy(
            config=cfg,
            model_signal_fn=lambda v, p, m: pd.Series(dtype="float64"),
            model_handle=ModelHandle(
                version=ModelVersion(name="m", version=1, digest="d0", created_at="",
                                     training_snapshot_id=None, training_as_of="2020-01-01",
                                     code_hash=None, provenance_digest="p0"),
                artifact_bytes=b"x",
            ),
            construct_fn=get_construction_policy(cfg.construction),
        )


def test_config_hash_changes_with_model_ref():
    _, ref1, h1 = _consistent(artifact=b"model-A", version=1)
    _, ref2, h2 = _consistent(artifact=b"model-B", version=2)
    cfg1 = _plain_cfg(needs_model=True, model_ref=ref1)
    cfg2 = _plain_cfg(needs_model=True, model_ref=ref2)

    def _mk(cfg, handle):
        return LoadedStrategy(
            config=cfg, model_signal_fn=lambda v, p, m: pd.Series(dtype="float64"),
            model_handle=handle, construct_fn=get_construction_policy(cfg.construction),
        )
    assert config_hash(_mk(cfg1, h1)) != config_hash(_mk(cfg2, h2))


def test_bound_handle_must_match_pinned_ref():
    """A handle whose version diverges from the pinned model_ref is rejected at construction —
    a caller cannot bind a different model than the config was validated against."""
    _, ref, _ = _consistent()
    cfg = _plain_cfg(needs_model=True, model_ref=ref)
    mismatched = ModelHandle(
        version=ModelVersion(name="m", version=1, digest="DIFFERENT", created_at="",
                             training_snapshot_id=None, training_as_of="2020-01-01",
                             code_hash=None, provenance_digest="p0"),
        artifact_bytes=b"x",
    )
    with pytest.raises(ValueError, match="does not match the pinned model_ref"):
        LoadedStrategy(
            config=cfg, model_signal_fn=lambda v, p, m: pd.Series(dtype="float64"),
            model_handle=mismatched, construct_fn=get_construction_policy(cfg.construction),
        )


def test_bound_handle_bytes_must_hash_to_pinned_digest():
    """Metadata matching the pin is NOT enough: a handle whose version metadata matches model_ref
    but whose artifact_bytes hash to something else is rejected — the exact silent-different-model
    bypass (bind future bytes behind a safe old ModelVersion)."""
    _, ref, _ = _consistent(artifact=b"the-validated-model")
    cfg = _plain_cfg(needs_model=True, model_ref=ref)
    # version metadata matches the pin, but the bytes are a DIFFERENT model.
    forged = ModelHandle(version=_consistent(artifact=b"the-validated-model")[0],
                         artifact_bytes=b"a-different-future-model")
    with pytest.raises(ValueError, match="do not hash to the pinned"):
        LoadedStrategy(
            config=cfg, model_signal_fn=lambda v, p, m: pd.Series(dtype="float64"),
            model_handle=forged, construct_fn=get_construction_policy(cfg.construction),
        )


def test_three_way_exclusivity():
    cfg = _plain_cfg(needs_news=True, needs_model=True, model_ref=_ref())
    with pytest.raises(ValueError, match="at most one"):
        LoadedStrategy(
            config=cfg, news_signal_fn=lambda v, p, n: pd.Series(dtype="float64"),
            model_handle=ModelHandle(
                version=ModelVersion(name="m", version=1, digest="d0", created_at="",
                                     training_snapshot_id=None, training_as_of="2020-01-01",
                                     code_hash=None, provenance_digest="p0"),
                artifact_bytes=b"x"),
            construct_fn=get_construction_policy(cfg.construction),
        )
