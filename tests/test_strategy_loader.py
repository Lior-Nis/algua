import sys
from pathlib import Path

import pytest

from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import StrategyNotFound, list_strategies, load_strategy


def test_load_bundled_momentum():
    strat = load_strategy("cross_sectional_momentum")
    assert isinstance(strat, LoadedStrategy)
    assert strat.name == "cross_sectional_momentum"


def test_unknown_strategy_raises():
    with pytest.raises(StrategyNotFound):
        load_strategy("does_not_exist")


def test_strategy_missing_compute_weights_raises():
    import algua.strategies.momentum as fam

    mod_path = Path(fam.__path__[0]) / "_tmp_no_weights_fn.py"
    mod_path.write_text(
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='_tmp_no_weights_fn', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'))\n"
    )
    try:
        with pytest.raises(StrategyNotFound, match="compute_weights"):
            load_strategy("_tmp_no_weights_fn")
    finally:
        mod_path.unlink()


def test_list_includes_bundled():
    assert "cross_sectional_momentum" in list_strategies()


def test_load_by_bare_name_across_families(tmp_path):
    # Both bundled strategies resolve by bare name regardless of which family dir they live in.
    # NOTE: after Task 3, both live in different family dirs; this test uses bare names only.
    assert load_strategy("cross_sectional_momentum").name == "cross_sectional_momentum"
    assert load_strategy("fundamentals_earnings_tilt").name == "fundamentals_earnings_tilt"


def test_duplicate_bare_name_across_families_raises():
    """A bare name appearing in two family dirs is a hard, fail-closed error."""
    import algua.strategies as sp
    root = Path(sp.__file__).parent
    fam_a, fam_b = root / "_dupfam_a", root / "_dupfam_b"
    body = (
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "import pandas as pd\n"
        "CONFIG = StrategyConfig(name='dupe', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'))\n"
        "def compute_weights(view, params):\n    return pd.Series(dtype='float64')\n"
    )
    try:
        for fam in (fam_a, fam_b):
            fam.mkdir()
            (fam / "__init__.py").write_text("")
            (fam / "dupe.py").write_text(body)
        with pytest.raises(StrategyNotFound, match="duplicate"):
            load_strategy("dupe")
    finally:
        import shutil
        shutil.rmtree(fam_a, ignore_errors=True)
        shutil.rmtree(fam_b, ignore_errors=True)


def test_loading_one_strategy_does_not_import_siblings():
    """The single-import contract: loading one strategy must not pull a sibling into sys.modules."""
    sys.modules.pop("algua.strategies.fundamentals.fundamentals_earnings_tilt", None)
    load_strategy("cross_sectional_momentum")
    assert "algua.strategies.fundamentals.fundamentals_earnings_tilt" not in sys.modules
