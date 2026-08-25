"""Backtest-lane error leaf.

``BacktestError`` is the domain error for the whole backtest lane, not just one module inside it —
it is raised by the PIT masking views, the orchestrator, the delisting path and the windowing
helpers, and it is caught outside the package by ``cli/errors`` (which maps it to the
``backtest_error`` envelope code), ``registry``, ``risk`` and ``evaluation``. Consumers in five
packages should not have to import a masking module to catch it.

Lives in its own leaf rather than in ``engine.py`` or ``pit_view.py`` for the same reason
``execution/errors.py`` holds ``BrokerError`` and ``data/providers/errors.py`` holds its provider
error: the raiser and the catcher must be able to agree on the exception type without either
importing the other's implementation. Defining it in a module that also does work creates an import
cycle the moment that module's own dependencies want to raise it — which is exactly what happened
when the masking views were carved out of ``engine.py``.

Imports nothing from algua, deliberately: an error leaf that pulls in machinery is not a leaf.
"""

from __future__ import annotations


class BacktestError(RuntimeError):
    pass
