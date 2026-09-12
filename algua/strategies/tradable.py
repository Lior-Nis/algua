"""Paper/live tradability guards: a strategy that declares a PIT sidecar lane the lanes cannot
serve yet is refused at every trading load point (carved from strategies/base.py, overlays PR)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algua.strategies.base import LoadedStrategy


def assert_tradable_without_fundamentals(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_fundamentals strategy must NOT run paper/live yet — the as-of
    fundamentals lane is wired only into the backtest engine (issue #132). Called at every trading
    load point so no actor (agent promote OR human raw transition) can run it blind."""
    if strategy.config.needs_fundamentals:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_fundamentals; paper/live fundamentals "
            f"wiring is not built yet (#132 follow-up) — refusing to trade it blind"
        )


def assert_tradable_without_news(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_news strategy must NOT run paper/live yet — the as-of news lane is
    wired only into the backtest engine (issue #132). Called at every trading load point."""
    if strategy.config.needs_news:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_news; paper/live news wiring is not built "
            f"yet (#132 follow-up) — refusing to trade it blind"
        )


def assert_tradable_without_model(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_model strategy must NOT run paper/live yet — the model lane is wired
    only into the `backtest run` engine (issue #376). Called at every trading load point."""
    if strategy.config.needs_model:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_model; paper/live model wiring is not "
            f"built yet (#376 follow-up) — refusing to trade it blind"
        )
