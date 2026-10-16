"""The whole-account loss breaker TRIP sequence, shared by paper and live run-all (#647).

`algua.risk.book_cycle.evaluate_book_loss_breaker` DECIDES (equity vs the lane's high-water mark and
the prior-session close). This module ACTS on a breach, in the order that is safe when the next step
can fail: engage the persistent global halt FIRST (no trading even if the close then errors), audit,
THEN cancel-all + close-all on the whole account (reaching orphan/dormant/unverified holdings a
per-strategy flatten never would). One body for both lanes, so the rehearsal lane (paper) cannot
lack a wall the real-money lane (live) has -- the gap #647 closed.

Lane differences are data, not branches: the audit action carries the lane prefix the other paper
audit rows use (`paper_` + action), and the high-water mark is the lane's own row.
"""

from __future__ import annotations

from dataclasses import dataclass

from algua.audit.log import append as audit_append
from algua.observability import get_logger
from algua.risk import global_halt
from algua.risk.book_breaker import BookBreach
from algua.risk.book_cycle import evaluate_book_loss_breaker

log = get_logger(__name__)

_AUDIT_ACTION = {"live": "book_circuit_breaker", "paper": "paper_book_circuit_breaker"}


@dataclass(frozen=True)
class BookBreakerTrip:
    """A tripped breaker after the halt was engaged and the close attempted."""

    breach: BookBreach
    liquidation_submitted: bool
    flatten_error: str | None

    def payload(self) -> dict:
        """The envelope keys describing the trip, identical in both lanes."""
        out: dict = {"ok": False,
                     "book_breach": {"kind": self.breach.kind, "detail": self.breach.detail},
                     "global_halt": "set", "liquidation_submitted": self.liquidation_submitted}
        if self.flatten_error is not None:
            out["flatten_error"] = self.flatten_error
        return out


def enforce_book_loss_breaker(conn, broker, *, lane: str) -> BookBreakerTrip | None:
    """Evaluate the breaker for `lane`; on a breach halt, audit and flatten the whole account.

    Returns None when the book is within its limits. The caller emits the envelope and exits
    non-zero on a returned trip (`flatten_error` is set when the close itself failed)."""
    breach = evaluate_book_loss_breaker(conn, broker, lane)
    if breach is None:
        return None
    global_halt.engage(conn, reason=breach.detail, actor="system")
    audit_append(conn, actor="system", action=_AUDIT_ACTION[lane],
                 reason=f"{breach.kind}: {breach.detail}", strategy=None)
    log.error("book_circuit_breaker", extra={"fields": {"lane": lane, "kind": breach.kind}})
    try:
        broker.close_all_positions()
    except Exception as exc:  # noqa: BLE001 — surface + persist halt, never swallow
        log.error("book_flatten_failed", extra={"fields": {"lane": lane}}, exc_info=True)
        return BookBreakerTrip(breach, False, str(exc))
    return BookBreakerTrip(breach, True, None)
