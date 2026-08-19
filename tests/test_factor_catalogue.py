import pytest

from algua.features.catalogue import FactorKind, factor


def test_standalone_factor_accepts_signal_shaped_fn():
    @factor(standalone=True, summary="ok", kind=FactorKind.MOMENTUM)
    def good(view, params):  # 2 positional-or-keyword args
        return view

    assert good.__factor_spec__.standalone is True


def test_factor_defaults_to_not_standalone():
    @factor(summary="ok")
    def helper(prices, lookback):
        return prices

    assert helper.__factor_spec__.standalone is False


@pytest.mark.parametrize(
    "bad",
    [
        lambda v: v,                       # 1 arg
        lambda v, p, x: v,                 # 3 args
        lambda *a, **k: a,                 # varargs
        lambda v, *a: v,                   # trailing *args
    ],
)
def test_standalone_rejects_non_signal_shape(bad):
    with pytest.raises(ValueError, match="standalone"):
        factor(standalone=True, summary="x")(bad)
