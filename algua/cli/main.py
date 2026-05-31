from __future__ import annotations

from algua.cli import (  # noqa: F401 - imports register subcommands
    backtest_cmd,
    data_cmd,
    registry_cmd,
    research_cmd,
    strategy_cmd,
)
from algua.cli.app import app

__all__ = ["app"]
