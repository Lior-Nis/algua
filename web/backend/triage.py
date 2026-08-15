"""The "does the machine need me?" assembly behind the Now screen (spec: 2026-08-15).

A PURE function of ``(fleet, ops, book)`` — no I/O, no clock beyond the ``now`` passed in — so every
ranking rule is unit-testable without a subprocess, a filesystem, or a browser. Same discipline as
``poller.diff_fleet``.

The ranking exists because the operator's scarce resource is attention, not information. Three
sources feed one list, and they are NOT equally urgent: a stopped machine outranks a stranded
strategy, which outranks an individual unhealthy tick, because the first two are silent failures
that can persist for days while the fleet view stays green — which is exactly what happened on
2026-08-15 (a research loop dead 5 days, and a paper strategy stranded without capital).

Every item carries the ONE fact that makes it actionable. A triage row that says "something is
wrong" without saying what costs a tap and returns nothing.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SEVERITY", "build_triage"]

# Lower sorts first. Deliberately NOT a copy of fleet_health's severity: this ranks across
# CATEGORIES (machine vs. capital vs. strategy), and a dead loop outranks any single strategy.
SEVERITY = {
    "loop_down": 0,  # the machine stopped producing
    "global_halt": 1,  # the whole book is stopped
    "capital_stranded": 2,  # a strategy that cannot trade until a human acts
    "strategy": 3,  # one strategy's tick health
    "queue_wedged": 4,  # work piling up but nothing lost yet
}

# Loop verdicts that mean "not producing". `idle` is excluded on purpose: a never-run operator or an
# empty merge-back queue is a quiet system, not a broken one.
_LOOP_DOWN = frozenset({"failing", "rate_limited", "stale", "unknown"})


def _loop_items(ops: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ops, dict):
        return []
    items: list[dict[str, Any]] = []
    for name, row in (ops.get("loops") or {}).items():
        if not isinstance(row, dict):
            continue
        health = str(row.get("health"))
        if health not in _LOOP_DOWN:
            continue
        # A wedged queue is real but recoverable — the work is still enqueued, nothing is lost —
        # so it ranks below a loop that has stopped producing altogether.
        kind = "queue_wedged" if name == "mergeback" else "loop_down"
        items.append({
            "kind": kind,
            "severity": SEVERITY[kind],
            "title": f"{name} loop {health.replace('_', ' ')}",
            "detail": row.get("detail"),
            "since": row.get("last_ok_at") or row.get("last_run_at"),
            "route": "/research" if name in ("research", "mergeback") else "/fleet",
        })
    return items


def _capital_items(book: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(book, dict):
        return []
    items: list[dict[str, Any]] = []
    for row in book.get("unallocated_operational") or []:
        if not isinstance(row, dict):
            continue
        # "never ticked" vs "stopped ticking" are different stories and the operator needs to know
        # which: the first never started, the second lost its slice mid-life.
        never = row.get("ever_ticked") is False
        items.append({
            "kind": "capital_stranded",
            "severity": SEVERITY["capital_stranded"],
            # Carried so the fleet pass can suppress this strategy's symptom row.
            "strategy": str(row.get("strategy")),
            "title": f"{row.get('strategy')} holds no capital",
            "detail": (
                f"stage {row.get('stage')}, "
                + ("never ticked" if never else "no active allocation")
                + " — the operator loop skips it"
            ),
            "since": row.get("since"),
            "route": "/money",
        })
    return items


def _fleet_items(
    fleet: dict[str, Any] | None, explained: frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    """Per-strategy tick health, skipping any strategy whose cause is already on the list.

    ``explained`` carries the strategies a higher-severity item already diagnoses.
    """
    if not isinstance(fleet, dict):
        return []
    items: list[dict[str, Any]] = []
    if fleet.get("global_halt"):
        items.append({
            "kind": "global_halt",
            "severity": SEVERITY["global_halt"],
            "title": "global halt engaged",
            "detail": "the account-wide kill switch is active; nothing will trade",
            "since": None,
            "route": "/fleet",
        })
        # Under a global halt EVERY row is halted; listing them all would bury the one line that
        # matters. The halt item carries the event (same fan-out rule as the push poller).
        return items
    for row in fleet.get("alerting") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("strategy"))
        if name in explained:
            # Already covered by a row that names the CAUSE. An unallocated strategy is `idle`
            # BECAUSE it holds no slice, so listing both states the symptom under the diagnosis
            # and doubles the apparent workload — the exact noise a triage list must not have.
            continue
        stale = row.get("staleness_sessions")
        health = str(row.get("health"))
        items.append({
            "kind": "strategy",
            "severity": SEVERITY["strategy"],
            "title": f"{name} {health}",
            # Never blank: an item with no detail costs a tap to learn nothing. Fall back to the
            # stage, which at least says whether anything is supposed to be ticking it.
            "detail": (row.get("kill_switch") or {}).get("reason")
            or (f"{stale} sessions since last tick" if isinstance(stale, int) else None)
            or f"stage {row.get('stage')} — no tick evidence",
            "since": None,
            "route": f"/s/{name}",
        })
    return items


def build_triage(
    fleet: dict[str, Any] | None,
    ops: dict[str, Any] | None,
    book: dict[str, Any] | None,
) -> dict[str, Any]:
    """One ranked "needs you" list plus the three headline numbers, worst-first.

    A part that failed to load is passed as ``None`` and contributes nothing rather than raising —
    a degraded source must never blank the whole screen. ``sources`` reports which parts were
    present so the UI can say "this list may be incomplete" instead of implying all-clear.
    """
    capital = _capital_items(book)
    # A stranded strategy's `idle` health is a SYMPTOM of the missing slice, so the capital item
    # suppresses the fleet item for that strategy rather than both appearing.
    explained = frozenset(str(item["strategy"]) for item in capital)
    items = _loop_items(ops) + capital + _fleet_items(fleet, explained)
    items.sort(key=lambda item: (item["severity"], str(item["title"])))

    summary = (fleet or {}).get("summary") or {}
    by_health: dict[str, int] = {}
    for row in (fleet or {}).get("rows") or []:
        if isinstance(row, dict):
            health = str(row.get("health"))
            by_health[health] = by_health.get(health, 0) + 1

    return {
        "items": items,
        "sources": {
            "fleet": fleet is not None,
            "ops": ops is not None,
            "book": book is not None,
        },
        "headline": {
            # Fleet-wide, counted from `rows` — `summary.by_health` covers ALERTING rows only.
            "fleet_ok": by_health.get("ok", 0),
            "fleet_total": summary.get("total"),
            "book_allocated": (book or {}).get("allocated"),
            "book_capacity": (book or {}).get("capacity"),
            "loops_alerting": len((ops or {}).get("alerting") or []),
        },
    }
