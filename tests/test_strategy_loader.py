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


def test_strategy_missing_signal_raises():
    import algua.strategies.momentum as fam

    # Non-`_` name: `_`-prefixed modules are skipped at discovery (so this must be discoverable to
    # exercise the missing-signal branch). Cleaned up in `finally`.
    mod_path = Path(fam.__path__[0]) / "tmp_no_signal_fn.py"
    mod_path.write_text(
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_no_signal_fn', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
    )
    try:
        with pytest.raises(StrategyNotFound, match="signal"):
            load_strategy("tmp_no_signal_fn")
    finally:
        mod_path.unlink()


def test_list_includes_bundled():
    assert "cross_sectional_momentum" in list_strategies()


def test_loader_resolves_and_binds_construction():
    s = load_strategy("cross_sectional_momentum")
    assert s.config.construction == "top_k_equal_weight"
    assert s.signal_fn is not None and s.signal_panel_fn is not None
    assert callable(s.construct_fn)


def test_loader_rejects_unknown_construction():
    # A module whose CONFIG names a missing policy must fail at load. Non-`_` name in a real
    # family dir so `_index` discovers it (`_`-prefixed modules are skipped). Cleaned up after.
    import algua.strategies.momentum as fam
    mod_path = Path(fam.__path__[0]) / "tmp_bad_policy.py"
    mod_path.write_text(
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_bad_policy', universe=['A'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='nope_not_real')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        with pytest.raises(StrategyNotFound):
            load_strategy("tmp_bad_policy")
    finally:
        mod_path.unlink()


def test_loader_rejects_config_name_mismatch():
    """A module whose CONFIG.name differs from the file/registry stem fails closed (#275),
    so a strategy's identity can't silently fragment across MLflow/docs/registry."""
    import algua.strategies.momentum as fam
    mod_path = Path(fam.__path__[0]) / "tmp_name_mismatch.py"
    mod_path.write_text(
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        # CONFIG.name deliberately != the file stem 'tmp_name_mismatch'
        "CONFIG = StrategyConfig(name='some_other_name', universe=['A'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    return pd.Series(dtype='float64')\n"
    )
    try:
        with pytest.raises(StrategyNotFound, match="must match"):
            load_strategy("tmp_name_mismatch")
    finally:
        mod_path.unlink()


def test_load_by_bare_name_across_families(tmp_path):
    # Both bundled strategies resolve by bare name regardless of which family dir they live in.
    assert load_strategy("cross_sectional_momentum").name == "cross_sectional_momentum"
    assert load_strategy("fundamentals_earnings_tilt").name == "fundamentals_earnings_tilt"


def test_duplicate_bare_name_across_families_raises():
    """A bare name appearing in two family dirs is a hard, fail-closed error (raised during
    discovery, before any module is imported or its contract checked)."""
    import algua.strategies as sp
    root = Path(sp.__file__).parent
    # Non-`_` family dirs: `_`-prefixed dirs are skipped at discovery, so the dup must be in real
    # (discoverable) family dirs. Cleaned up in `finally`.
    fam_a, fam_b = root / "dupfam_a", root / "dupfam_b"
    body = (
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='dupe', universe=['AAA'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n    return pd.Series(dtype='float64')\n"
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


def test_family_init_files_are_empty():
    """Enforce the empty-`__init__` convention the single-import contract relies on: a family
    `__init__.py` that imported a sibling would silently pull extra modules into the code_hash
    closure when any strategy in that family is loaded. Committed family inits must stay empty."""
    import algua.strategies as sp
    root = Path(sp.__file__).parent
    for fam in root.iterdir():
        if not (fam.is_dir() and not fam.name.startswith("_") and (fam / "__init__.py").exists()):
            continue
        body = (fam / "__init__.py").read_text().strip()
        assert body == "", f"family package {fam.name}/__init__.py must be empty, has: {body!r}"


def test_load_tradable_strategy_loads_plain_strategy():
    from algua.strategies.loader import load_tradable_strategy
    s = load_tradable_strategy("cross_sectional_momentum")
    assert s.config.name == "cross_sectional_momentum"


def test_load_tradable_strategy_rejects_fundamentals_strategy(monkeypatch):
    # A needs_fundamentals strategy must be refused by the tradability gate.
    import algua.strategies.loader as loader

    sentinel = object()

    def _fake_load(name):
        return sentinel

    def _boom(strategy):
        assert strategy is sentinel
        raise ValueError("needs_fundamentals: not tradable without a fundamentals lane")

    monkeypatch.setattr(loader, "load_strategy", _fake_load)
    monkeypatch.setattr(loader, "assert_tradable_without_fundamentals", _boom)
    with pytest.raises(ValueError, match="needs_fundamentals"):
        loader.load_tradable_strategy("x")
