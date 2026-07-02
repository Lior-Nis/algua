"""Tests for the PIT-fit feature-scaling seam (issue #388)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algua.features.scaling import (
    QUANTILE,
    STANDARD,
    FrozenScaler,
    ScalerError,
    assert_fit_before,
)
from algua.features.scaling_fit import fit_quantile_scaler, fit_standard_scaler


def _train(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"a": rng.normal(1.0, 2.0, size=n), "b": rng.normal(-3.0, 5.0, size=n)}, index=idx
    )


# --------------------------------------------------------------------------- fit-window evidence

def test_fit_stamps_max_timestamp_from_validated_index() -> None:
    train = _train()
    scaler = fit_standard_scaler(train, columns=["a", "b"])
    assert scaler.fit_max_timestamp == pd.Timestamp(train.index.max(), tz="UTC").isoformat()


def test_fit_rejects_non_datetime_index() -> None:
    train = _train().reset_index(drop=True)  # RangeIndex
    with pytest.raises(ScalerError, match="DatetimeIndex"):
        fit_standard_scaler(train, columns=["a"])


def test_fit_rejects_non_monotonic_index() -> None:
    train = _train()
    shuffled = train.iloc[::-1]  # descending -> not monotonic increasing
    with pytest.raises(ScalerError, match="monotonic"):
        fit_standard_scaler(shuffled, columns=["a"])


def test_fit_rejects_empty_frame() -> None:
    empty = _train().iloc[:0]
    with pytest.raises(ScalerError, match="non-empty"):
        fit_standard_scaler(empty, columns=["a"])


# --------------------------------------------------------------------------- standard scaler

def test_standard_scaler_normalizes_to_zero_mean_unit_std() -> None:
    train = _train()
    scaler = fit_standard_scaler(train, columns=["a", "b"])
    out = scaler.transform(train)
    assert out["a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)
    assert out["b"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_standard_scaler_rejects_degenerate_variance() -> None:
    idx = pd.date_range("2019-01-01", periods=10, freq="B")
    train = pd.DataFrame({"a": [5.0] * 10}, index=idx)
    with pytest.raises(ScalerError, match="degenerate|variance"):
        fit_standard_scaler(train, columns=["a"])


# --------------------------------------------------------------------------- quantile scaler

def test_quantile_scaler_maps_to_unit_interval_and_is_monotone() -> None:
    train = _train()
    scaler = fit_quantile_scaler(train, columns=["a"], n_quantiles=50)
    out = scaler.transform(train)
    assert out["a"].min() >= 0.0
    assert out["a"].max() <= 1.0
    # A monotone transform: sorting by raw preserves the scaled order.
    order = train["a"].sort_values().index
    scaled_sorted = out.loc[order, "a"].to_numpy()
    assert np.all(np.diff(scaled_sorted) >= -1e-12)


def test_quantile_scaler_rejects_constant_column() -> None:
    idx = pd.date_range("2019-01-01", periods=10, freq="B")
    train = pd.DataFrame({"a": [2.0] * 10}, index=idx)
    with pytest.raises(ScalerError, match="distinct|degenerate"):
        fit_quantile_scaler(train, columns=["a"])


# --------------------------------------------------------------------------- frozen round-trip

def test_frozen_round_trip_is_transform_identical() -> None:
    train = _train()
    scaler = fit_standard_scaler(train, columns=["a", "b"])
    restored = FrozenScaler.from_dict(scaler.to_dict())
    test = _train(n=30, seed=99)
    pd.testing.assert_frame_equal(scaler.transform(test), restored.transform(test))


def test_to_dict_is_json_stable() -> None:
    import json

    scaler = fit_standard_scaler(_train(), columns=["a", "b"])
    d1 = json.dumps(scaler.to_dict(), sort_keys=True)
    d2 = json.dumps(FrozenScaler.from_dict(scaler.to_dict()).to_dict(), sort_keys=True)
    assert d1 == d2


def test_from_dict_fails_closed_on_malformed_blob() -> None:
    with pytest.raises(ScalerError, match="malformed"):
        FrozenScaler.from_dict({"kind": STANDARD, "columns": ["a"]})  # no params/timestamp


# --------------------------------------------------------------------------- transform guards

def test_transform_fails_closed_on_missing_column() -> None:
    scaler = fit_standard_scaler(_train(), columns=["a", "b"])
    with pytest.raises(ScalerError, match="missing required columns"):
        scaler.transform(pd.DataFrame({"a": [1.0]}))


def test_transform_leaves_other_columns_untouched() -> None:
    train = _train()
    scaler = fit_standard_scaler(train, columns=["a"])
    df = train.assign(keep=range(len(train)))
    out = scaler.transform(df)
    pd.testing.assert_series_equal(out["keep"], df["keep"])


def test_transform_preserves_nan() -> None:
    scaler = fit_standard_scaler(_train(), columns=["a"])
    df = pd.DataFrame({"a": [np.nan, 1.0]})
    out = scaler.transform(df)
    assert pd.isna(out["a"].iloc[0])


# --------------------------------------------------------------------------- construction guards

def test_frozen_scaler_rejects_unknown_kind() -> None:
    with pytest.raises(ScalerError, match="unknown scaler kind"):
        FrozenScaler(kind="nope", columns=("a",), params={"a": {}}, fit_max_timestamp="2019-01-01")


def test_frozen_scaler_rejects_bad_timestamp() -> None:
    with pytest.raises(ScalerError, match="not a valid timestamp"):
        FrozenScaler(
            kind=STANDARD,
            columns=("a",),
            params={"a": {"mean": 0.0, "std": 1.0}},
            fit_max_timestamp="not-a-date",
        )


def test_frozen_scaler_rejects_nonpositive_std() -> None:
    with pytest.raises(ScalerError, match="positive std"):
        FrozenScaler(
            kind=STANDARD,
            columns=("a",),
            params={"a": {"mean": 0.0, "std": 0.0}},
            fit_max_timestamp="2019-01-01",
        )


# --------------------------------------------------------------------------- assert_fit_before

def test_assert_fit_before_passes_when_fit_within_cutoff() -> None:
    scaler = fit_standard_scaler(_train(), columns=["a"])
    assert_fit_before(scaler, "2020-01-01")  # 2019 fit window <= 2020 cutoff -> ok


def test_assert_fit_before_rejects_fit_past_cutoff() -> None:
    # A whole-series scaler whose fit window ends AFTER the model's training cutoff must be refused,
    # even though the digest checks would pass (the exact train/serve leak #388 closes).
    idx = pd.date_range("2019-01-01", "2023-06-30", freq="B")
    whole_series = pd.DataFrame({"a": np.linspace(0.0, 1.0, len(idx))}, index=idx)
    leaky = fit_standard_scaler(whole_series, columns=["a"])
    with pytest.raises(ScalerError, match="AFTER the model's training_as_of"):
        assert_fit_before(leaky, "2020-01-01")


def test_assert_fit_before_boundary_equal_is_allowed() -> None:
    idx = pd.date_range("2019-12-30", "2020-01-01", freq="D")
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    scaler = fit_standard_scaler(train, columns=["a"])
    assert_fit_before(scaler, "2020-01-01")  # fit_max == cutoff -> allowed (<=)


def test_quantile_kind_constant_used() -> None:
    # sanity: the QUANTILE constant is what the fitter stamps
    scaler = fit_quantile_scaler(_train(), columns=["a"])
    assert scaler.kind == QUANTILE
