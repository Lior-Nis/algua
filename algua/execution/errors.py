"""Broker-neutral error leaf.

``BrokerError`` is the domain error for *any* broker adapter, not just Alpaca. It lives here rather
than in ``alpaca_broker`` so broker-agnostic consumers (``execution/tick_clock``, ``cli/errors``)
can catch it without importing a concrete adapter. Mirrors ``algua/data/providers/errors.py``,
which plays the same role for the data-provider seam.

Imports nothing — keeping this a true leaf is what makes it safe for any layer to depend on.
"""

from __future__ import annotations


class BrokerError(RuntimeError):
    """Any failure talking to the Alpaca trading API — network error, non-2xx status,
    or a malformed/unexpected response. Callers (the CLI, the future loop) catch this so a
    broker hiccup never escapes as a raw traceback."""
