from __future__ import annotations

import sqlite3

from algua.audit.log import append as audit_append
from algua.risk import kill_switch
from algua.risk.limits import RiskBreach


def trip_for_breach(conn: sqlite3.Connection, name: str, exc: RiskBreach) -> None:
    """Trip the kill-switch for a risk breach and write the matching audit row.

    Ordering is INTENTIONAL: mutate (trip) THEN audit. For a trip, fail-safe means the switch is
    persisted (halted) even if the audit write then fails; reversing to audit-first would leave the
    worse state (audited-as-tripped but switch unpersisted). The divergent emit/flatten stays in
    each caller.
    """
    kill_switch.trip(conn, name, reason=exc.detail, actor="system")
    audit_append(conn, actor="system", action="kill_switch_trip",
                 reason=f"{exc.kind}: {exc.detail}", strategy=name)
