# tests/test_features_catalogue.py
import inspect
import sys
from pathlib import Path

import pytest

import algua.features as _featpkg
from algua.contracts.idea import DataCapability
from algua.features import catalogue as _cat
from algua.features.catalogue import (
    FactorKind,
    FactorNotFound,
    FactorSpec,
    all_factors,
    factor,
    filter_factors,
    get_factor,
    load_all_factors,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    _cat._reset_registry()
    yield
    _cat._reset_registry()


def _write_feature_module(stem: str, body: str) -> Path:
    path = Path(_featpkg.__path__[0]) / f"{stem}.py"
    path.write_text(body)
    return path


def _drop_module(stem: str, path: Path) -> None:
    path.unlink(missing_ok=True)
    sys.modules.pop(f"algua.features.{stem}", None)


def all_factors_names():
    return {f.name for f in all_factors()}


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


def test_discovers_decorated_factor_in_a_feature_module():
    path = _write_feature_module(
        "tmp_disc_mod",
        "from algua.features.catalogue import factor, FactorKind\n"
        "@factor(summary='tmp', kind=FactorKind.MOMENTUM, tags=['t'])\n"
        "def tmp_fac(x):\n"
        "    return x\n",
    )
    try:
        reg = load_all_factors()
        assert "tmp_fac" in reg
        assert reg["tmp_fac"].kind is FactorKind.MOMENTUM
    finally:
        _drop_module("tmp_disc_mod", path)


def test_reexport_is_not_double_registered():
    # A module that imports another module's catalogued factor must NOT re-register it.
    defn = _write_feature_module(
        "tmp_defn_mod",
        "from algua.features.catalogue import factor\n"
        "@factor(summary='defined here')\n"
        "def shared_fac(x):\n"
        "    return x\n",
    )
    reexp = _write_feature_module(
        "tmp_reexp_mod",
        "from algua.features.tmp_defn_mod import shared_fac  # re-export\n"
        "from algua.features.catalogue import factor\n"
        "@factor(summary='own')\n"
        "def own_fac(x):\n"
        "    return x\n",
    )
    try:
        reg = load_all_factors()  # must NOT raise duplicate for shared_fac
        assert reg["shared_fac"].module == "algua.features.tmp_defn_mod"
        assert "own_fac" in reg
    finally:
        _drop_module("tmp_reexp_mod", reexp)
        _drop_module("tmp_defn_mod", defn)


def test_duplicate_name_fails_closed():
    a = _write_feature_module(
        "tmp_dup_a",
        "from algua.features.catalogue import factor\n"
        "@factor(summary='a')\n"
        "def clashing(x):\n"
        "    return x\n",
    )
    b = _write_feature_module(
        "tmp_dup_b",
        "from algua.features.catalogue import factor\n"
        "@factor(summary='b')\n"
        "def clashing(x):\n"  # same bare name, different module
        "    return x\n",
    )
    try:
        with pytest.raises(ValueError, match="duplicate factor name"):
            load_all_factors()
    finally:
        _drop_module("tmp_dup_a", a)
        _drop_module("tmp_dup_b", b)


def test_transactional_failed_import_preserves_prior_registry():
    good = _write_feature_module(
        "tmp_good_mod",
        "from algua.features.catalogue import factor\n"
        "@factor(summary='good')\n"
        "def good_fac(x):\n"
        "    return x\n",
    )
    try:
        load_all_factors()  # registry now has good_fac
        bad = _write_feature_module("tmp_bad_mod", "raise RuntimeError('boom')\n")
        try:
            with pytest.raises(RuntimeError, match="boom"):
                load_all_factors()
            # prior registry intact, never half-populated
            assert "good_fac" in all_factors_names()
        finally:
            _drop_module("tmp_bad_mod", bad)
    finally:
        _drop_module("tmp_good_mod", good)


def test_get_factor_unknown_raises():
    with pytest.raises(FactorNotFound):
        get_factor("does_not_exist_factor")


def test_seeded_factors_present_and_filterable():
    reg = load_all_factors()
    assert {"momentum", "zscore"} <= set(reg)

    mom = reg["momentum"]
    assert mom.kind is FactorKind.MOMENTUM
    assert "momentum" in mom.tags
    assert mom.import_path == "algua.features.indicators:momentum"
    assert mom.data_needs == (DataCapability.OHLCV,)

    only_momentum = filter_factors(kind=FactorKind.MOMENTUM)
    assert "momentum" in {f.name for f in only_momentum}
    assert "zscore" not in {f.name for f in only_momentum}

    tagged = filter_factors(tag="cross-sectional")
    assert {"momentum", "zscore"} <= {f.name for f in tagged}
