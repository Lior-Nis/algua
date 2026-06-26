"""Schedule-class predicate: should a strategy decide on the current session?

Today every strategy is daily/XNYS, so the only supported class is "1d" (decide every session).
Multi-cadence (weekly, intraday, crypto 24/7) is added here later WITHOUT changing callers — the
future `paper run-all` only asks `is_due(strategy.execution.rebalance_frequency)`. Fail closed on
anything unrecognized so a typo'd or future frequency can never silently tick or silently skip."""
from __future__ import annotations


def is_due(rebalance_frequency: str) -> bool:
    if rebalance_frequency == "1d":
        return True
    raise ValueError(f"unsupported rebalance_frequency {rebalance_frequency!r}")
