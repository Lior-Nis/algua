# tests/test_registry_lineage.py
import sys
from pathlib import Path

import algua.strategies.momentum as _momfam
from algua.registry.lineage import factors_used_by


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


def test_cross_sectional_momentum_uses_composed_factor():
    # After composition (issue #140), the strategy delegates to xs_trailing_return.
    used = {f.name for f in factors_used_by("cross_sectional_momentum")}
    assert "xs_trailing_return" in used
    # Module-granular: alphas.py imports indicators.py so both momentum + zscore are also reported.
    assert {"momentum", "zscore"} <= used
