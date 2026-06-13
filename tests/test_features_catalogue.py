# tests/test_features_catalogue.py
import inspect

import pytest

from algua.contracts.idea import DataCapability
from algua.features.catalogue import FactorKind, FactorSpec, factor


def test_factor_stamps_spec_and_returns_same_object():
    def raw(x):
        """One-line summary here. More detail ignored."""
        return x

    decorated = factor(kind=FactorKind.MOMENTUM, tags=["a", "b"])(raw)
    assert decorated is raw  # pure annotation, no wrapper
    spec = decorated.__factor_spec__
    assert isinstance(spec, FactorSpec)
    assert spec.name == "raw"
    assert spec.summary == "One-line summary here. More detail ignored."
    assert spec.kind is FactorKind.MOMENTUM
    assert spec.tags == ("a", "b")  # tuple, not list
    assert spec.data_needs == (DataCapability.OHLCV,)  # default
    assert spec.module == "tests.test_features_catalogue"
    assert spec.import_path == "tests.test_features_catalogue:test_factor_stamps_spec_and_returns_same_object.<locals>.raw"  # noqa: E501
    assert spec.signature == str(inspect.signature(raw))
    assert decorated(7) == 7  # behaviour unchanged


def test_explicit_overrides():
    @factor(name="z", summary="explicit", kind=FactorKind.VALUE,
            data_needs=[DataCapability.FUNDAMENTALS])
    def f(a, b):
        return a

    spec = f.__factor_spec__
    assert spec.name == "z"
    assert spec.summary == "explicit"
    assert spec.data_needs == (DataCapability.FUNDAMENTALS,)


def test_missing_summary_and_docstring_fails_closed():
    def nodoc(x):
        return x

    with pytest.raises(ValueError, match="summary"):
        factor()(nodoc)
