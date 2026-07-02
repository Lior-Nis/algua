"""Loader tests for the needs_news lane (issue #132, Task 5).

Fixture mechanics mirror test_strategy_loader.py: write a temp .py file into a real
family dir so the filesystem discovery finds it, load it, clean up in finally.
The momentum family is used as the temp home (same as the other temp-module tests).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import algua.strategies.momentum as _momentum_fam
from algua.strategies.loader import StrategyNotFound, load_strategy

_FAM_PATH = Path(_momentum_fam.__path__[0])

# ---------------------------------------------------------------------------
# Common module-body fragments
# ---------------------------------------------------------------------------
_CONFIG_HEADER = (
    "import pandas as pd\n"
    "from algua.contracts.types import ExecutionContract\n"
    "from algua.strategies.base import StrategyConfig\n"
)

_CONFIG_NEWS = (
    "CONFIG = StrategyConfig(\n"
    "    name='{name}',\n"
    "    universe=['AAPL'],\n"
    "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
    "    construction='equal_weight_positive',\n"
    "    needs_news=True,\n"
    ")\n"
)

_CONFIG_BOTH = (
    "CONFIG = StrategyConfig(\n"
    "    name='{name}',\n"
    "    universe=['AAPL'],\n"
    "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
    "    construction='equal_weight_positive',\n"
    "    needs_fundamentals=True,\n"
    "    needs_news=True,\n"
    ")\n"
)

_SIGNAL_3ARG = (
    "def signal(view, params, news):\n"
    "    return pd.Series(dtype='float64')\n"
)

_SIGNAL_2ARG = (
    "def signal(view, params):\n"
    "    return pd.Series(dtype='float64')\n"
)

_SIGNAL_PANEL = (
    "def signal_panel(bars, params):\n"
    "    return pd.DataFrame()\n"
)


def _write_module(name: str, body: str) -> Path:
    path = _FAM_PATH / f"{name}.py"
    path.write_text(body)
    return path


def _clear_cache(dotted: str) -> None:
    sys.modules.pop(dotted, None)


# ---------------------------------------------------------------------------
# Test 1: needs_news=True with a 3-arg signal loads correctly
# ---------------------------------------------------------------------------
def test_needs_news_3arg_signal_loads():
    name = "tmp_news_3arg"
    dotted = f"algua.strategies.momentum.{name}"
    body = (
        _CONFIG_HEADER
        + _CONFIG_NEWS.format(name=name)
        + _SIGNAL_3ARG
    )
    path = _write_module(name, body)
    _clear_cache(dotted)
    try:
        import importlib
        module = importlib.import_module(dotted)
        loaded = load_strategy(name)
        assert loaded.config.needs_news is True
        assert loaded.news_signal_fn is module.signal
        assert loaded.signal_fn is None
    finally:
        path.unlink(missing_ok=True)
        _clear_cache(dotted)


# ---------------------------------------------------------------------------
# Test 2: needs_news=True with a 2-arg signal raises StrategyNotFound
# ---------------------------------------------------------------------------
def test_needs_news_2arg_signal_raises():
    name = "tmp_news_2arg"
    dotted = f"algua.strategies.momentum.{name}"
    body = (
        _CONFIG_HEADER
        + _CONFIG_NEWS.format(name=name)
        + _SIGNAL_2ARG
    )
    path = _write_module(name, body)
    _clear_cache(dotted)
    try:
        with pytest.raises(
            StrategyNotFound, match="needs_news=True requires signal\\(view, params, news\\)"
        ):
            load_strategy(name)
    finally:
        path.unlink(missing_ok=True)
        _clear_cache(dotted)


# ---------------------------------------------------------------------------
# Test 3: needs_news=True with signal_panel raises StrategyNotFound
# ---------------------------------------------------------------------------
def test_needs_news_with_signal_panel_raises():
    name = "tmp_news_panel"
    dotted = f"algua.strategies.momentum.{name}"
    body = (
        _CONFIG_HEADER
        + _CONFIG_NEWS.format(name=name)
        + _SIGNAL_3ARG
        + _SIGNAL_PANEL
    )
    path = _write_module(name, body)
    _clear_cache(dotted)
    try:
        with pytest.raises(StrategyNotFound, match="signal_panel is not supported with needs_news"):
            load_strategy(name)
    finally:
        path.unlink(missing_ok=True)
        _clear_cache(dotted)


# ---------------------------------------------------------------------------
# Test 4: both needs_news=True AND needs_fundamentals=True raises
# (LoadedStrategy.__post_init__ catches this via "at most one")
# ---------------------------------------------------------------------------
def test_needs_news_and_needs_fundamentals_both_raises():
    name = "tmp_news_and_fund"
    dotted = f"algua.strategies.momentum.{name}"
    body = (
        _CONFIG_HEADER
        + _CONFIG_BOTH.format(name=name)
        + _SIGNAL_3ARG
    )
    path = _write_module(name, body)
    _clear_cache(dotted)
    try:
        with pytest.raises((StrategyNotFound, ValueError), match="at most one"):
            load_strategy(name)
    finally:
        path.unlink(missing_ok=True)
        _clear_cache(dotted)
