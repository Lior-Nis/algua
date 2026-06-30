from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CycleCounters:
    """Golden-signal counts for one always-on loop cycle.

    Plain int accumulators incremented by ``run-all`` and flushed as a single ``golden_signals``
    log line at cycle end. Pure and single-threaded (the loop ticks strategies sequentially).
    """

    ticks: int = 0
    breaches: int = 0
    flatten_failures: int = 0
    reconcile_deferred: int = 0
    reconcile_halted: int = 0

    def as_fields(self) -> dict[str, int]:
        return {
            "ticks": self.ticks,
            "breaches": self.breaches,
            "flatten_failures": self.flatten_failures,
            "reconcile_deferred": self.reconcile_deferred,
            "reconcile_halted": self.reconcile_halted,
        }
