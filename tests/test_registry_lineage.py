# tests/test_registry_lineage.py
import sys
from pathlib import Path

import pytest

import algua.strategies.momentum as _momfam
from algua.features.catalogue import FactorNotFound
from algua.registry.db import connect, migrate
from algua.registry.lineage import dependents_of, factors_used_by
from algua.registry.store import SqliteStrategyRepository


@pytest.fixture()
def repo(tmp_path):
    c = connect(tmp_path / "r.db")
    migrate(c)
    return SqliteStrategyRepository(c)


def _write_strategy_using_momentum(stem: str) -> Path:
    path = Path(_momfam.__path__[0]) / f"{stem}.py"
    path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.features.indicators import momentum\n"
        f"CONFIG = StrategyConfig(name='{stem}', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    wide = view.reset_index().pivot(index='timestamp', columns='symbol',\n"
        "        values='adj_close')\n"
        "    return momentum(wide.iloc[-1], 1).dropna()\n"
    )
    return path


def _drop(stem: str, path: Path) -> None:
    path.unlink(missing_ok=True)
    sys.modules.pop(f"algua.strategies.momentum.{stem}", None)


def test_factors_used_by_reports_module_granular():
    path = _write_strategy_using_momentum("tmp_uses_strat")
    try:
        used = {f.name for f in factors_used_by("tmp_uses_strat")}
        # module-granular: importing `momentum` pulls in indicators.py -> both factors reported
        assert {"momentum", "zscore"} <= used
    finally:
        _drop("tmp_uses_strat", path)


def test_cross_sectional_momentum_uses_no_catalogued_factor():
    # The bundled strategy inlines its math and imports no catalogued factor.
    assert factors_used_by("cross_sectional_momentum") == []


def test_dependents_of_lists_registered_importer(repo):
    path = _write_strategy_using_momentum("tmp_dep_strat")
    try:
        repo.add("tmp_dep_strat")
        result = dependents_of(repo, "momentum")
        assert "tmp_dep_strat" in result.dependents
        assert result.unloadable == []
    finally:
        _drop("tmp_dep_strat", path)


def test_dependents_of_buckets_unloadable_registered_strategy(repo):
    repo.add("ghost_strategy")  # registered but no module on disk
    result = dependents_of(repo, "momentum")
    assert any(u["name"] == "ghost_strategy" for u in result.unloadable)
    assert "ghost_strategy" not in result.dependents


def test_dependents_of_unknown_factor_raises(repo):
    with pytest.raises(FactorNotFound):
        dependents_of(repo, "no_such_factor")
