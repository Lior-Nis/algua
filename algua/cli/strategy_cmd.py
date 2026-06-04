from __future__ import annotations

import keyword
from pathlib import Path

import typer

from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.strategies.loader import list_strategies

strategy_app = typer.Typer(help="Author and list strategies", no_args_is_help=True)
app.add_typer(strategy_app, name="strategy")

_TEMPLATE = '''\
"""Strategy: {name}. Edit compute_weights to express your cross-sectional logic."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="{name}",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={{"lookback": 60, "top_k": 2}},
)


def compute_weights(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    scores = (wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0).dropna()
    winners = scores.sort_values(ascending=False).head(int(params["top_k"])).index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)
'''


@strategy_app.command("list")
@json_errors()
def list_() -> None:
    """List available strategies as JSON."""
    emit(list_strategies())


@strategy_app.command("new")
@json_errors()
def new(name: str) -> None:
    """Scaffold a new strategy module under algua/strategies/examples/."""
    # `name` becomes both a filesystem path segment and a Python module name, so it must be a
    # safe identifier — rejects path traversal ("../x"), separators/spaces, and non-importable
    # or reserved names ("bad-name", "class").
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f"invalid strategy name {name!r}: must be a valid, non-keyword Python identifier"
        )
    path = Path("algua/strategies/examples") / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"strategy already exists: {path}")
    path.write_text(_TEMPLATE.format(name=name))
    emit({"ok": True, "name": name, "path": str(path)})
