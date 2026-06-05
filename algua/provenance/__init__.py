"""Provenance primitives shared across lanes.

Neutral leaf package (stdlib only, no other ``algua`` imports) so any lane may depend on it
without crossing a module boundary. It holds the ONE implementation of the locked-dependency
hash, which both the backtest reproducibility stamps and the live-approval identity must agree
on — divergent implementations would let an approval and a stamp disagree about what was run.
"""

from __future__ import annotations

from algua.provenance.lockfile import dependency_hash

__all__ = ["dependency_hash"]
