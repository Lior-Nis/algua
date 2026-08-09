"""Background poller: cache prewarm (slice D) + web-push alert diffing (slice E).

Each pass force-refreshes the prewarm targets through the SAME
:func:`backend.algua_cli.run_cli` seam the endpoints use (same cache entries,
same TTLs), so endpoint behavior never changes. When — and ONLY when — the
``fleet health`` refresh is genuinely fresh (no CliError, not a stale-cache
echo; R1), the result is diffed against the persisted notify state and
worsening transitions are pushed to the subscribed browsers.

The diff itself (:func:`diff_fleet`) is a pure function of
``(fleet_payload, state, now)`` so every notification rule is unit-testable
without a loop, a filesystem, or a push service.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import UTC, datetime
from typing import Any

from backend import push
from backend.algua_cli import CliError, run_cli

logger = logging.getLogger(__name__)

_DEFAULT_POLL_SECONDS = 600.0

# (argv, ttl_s) — ttl_s mirrors the SERVING endpoint's TTL for the same argv
# (fleet health -> /api/fleet 10s; registry list -> /api/strategies 60s) so a
# poller-written cache entry is exactly what the endpoint would have written.
_FLEET_ARGV = ("fleet", "health")
PREWARM_TARGETS: tuple[tuple[tuple[str, ...], float], ...] = (
    (_FLEET_ARGV, 10.0),
    (("registry", "list"), 60.0),
)

# Mirror of algua/execution/fleet_health.py::_SEVERITY — LOWER int = WORSE health.
_SEVERITY = {"halted": 0, "drift": 1, "stale": 2, "idle": 3, "ok": 4}
_OK_SEVERITY = _SEVERITY["ok"]
# Mirror of algua/execution/fleet_health.py::OPERATIONAL_STAGES — only these
# stages have an operator loop whose health is worth waking a phone for.
_OPERATIONAL_STAGES = frozenset({"live", "paper", "forward_tested"})
_RENOTIFY_AFTER_S = 24 * 60 * 60.0  # R3: per-strategy bad-health notification cooldown


def poll_seconds() -> float:
    """ALGUA_WEB_POLL_SECONDS, parsed defensively: a bad value -> default + warning."""
    raw = os.environ.get("ALGUA_WEB_POLL_SECONDS")
    if raw is None:
        return _DEFAULT_POLL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0.0:
        logger.warning(
            "ALGUA_WEB_POLL_SECONDS=%r is not a positive number; using default %ss",
            raw,
            _DEFAULT_POLL_SECONDS,
        )
        return _DEFAULT_POLL_SECONDS
    return value


def _severity(health: Any) -> int:
    # Unknown health strings rank PAST "ok" (99), the same convention as
    # fleet_health's worst-first sort — they never read as a worsening.
    return _SEVERITY.get(str(health), 99)


def _cooldown_active(last_bad_notified_at: Any, now: datetime) -> bool:
    """True when ANY bad-health notification for this strategy landed within 24h (R3).

    The cooldown is deliberately independent of WHICH bad health was notified —
    keying on the exact health let a stale<->drift flap notify every cycle.
    """
    if last_bad_notified_at is None:
        return False
    try:
        at = datetime.fromisoformat(str(last_bad_notified_at))
    except ValueError:
        return False  # corrupt marker -> notify rather than stay silent
    return (now - at).total_seconds() < _RENOTIFY_AFTER_S


def _halted_bypass(notified: Any, health: str) -> bool:
    """A transition INTO ``halted`` is materially worse: it may pierce the cooldown ONCE.

    "Once" is enforced by ``notified.health`` — set only after an accepted send
    (R5), so an undelivered halted alert still retries, while a delivered one
    puts halted itself under the cooldown.
    """
    if health != "halted":
        return False
    return not (isinstance(notified, dict) and notified.get("health") == "halted")


def _strategy_notification(row: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("strategy"))
    health = str(row.get("health"))
    reason = (row.get("kill_switch") or {}).get("reason")
    body = f"{row.get('stage')} · {health}"
    if reason:
        body += f" — {reason}"
    return {
        "kind": "strategy",
        "strategy": name,
        "health": health,
        "urgency": "high" if health == "halted" else "normal",  # R11
        # renotify rides in the payload for the SW's showNotification options (R7).
        "payload": {
            "title": f"{name} {health}",
            "body": body,
            "tag": f"strategy:{name}",
            "url": f"/s/{name}",
            "renotify": True,
        },
    }


def diff_fleet(
    fleet: dict[str, Any], state: dict[str, Any] | None, *, now: datetime
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure diff of a FRESH fleet-health payload against the persisted notify state.

    Returns ``(notifications, new_state)``. ``new_state`` carries only the
    ALWAYS-committed observed side (R5) — the ``notified``/
    ``last_bad_notified_at``/``notified_at``/``unhealthy``-clear marks are
    applied via :func:`mark_notified` ONLY after ``send_to_all`` accepted
    >= 1, so an undelivered alert retries next cycle.

    Baseline (R2): ``state is None`` (file missing) -> record observed state
    for everything, notify NOTHING. A restart with an existing file diffs
    against persisted state, so transitions during downtime DO notify.

    Cooldown (R3): an operational-stage row whose health is bad (severity < ok)
    is a candidate every cycle, gated by a per-strategy 24h cooldown
    (``last_bad_notified_at``) that is INDEPENDENT of which bad health was
    notified — so a stale<->drift flap alerts once per day, not once per cycle,
    while a persistent condition still re-alerts daily and an unACKed send
    retries (the cooldown commits only via :func:`mark_notified`). The one
    bypass: a transition INTO ``halted`` may pierce the cooldown once; after
    that accepted send the cooldown covers halted too. The cooldown (and the
    ``notified`` debug marker) resets ONLY on recovery to ok, on the row
    leaving the operational stages, or on the row disappearing. Old state
    files without ``last_bad_notified_at`` read as null (no cooldown).

    Fan-out suppression: when the account-wide global halt marks a row halted
    (``row.kill_switch.global_halt`` truthy), the row is NOT a per-strategy
    candidate — the single ``global:halt`` notification carries the event. A
    row tripped for its own reason (``kill_switch.tripped`` with
    ``global_halt`` false) still notifies individually.
    """
    now_iso = now.isoformat()
    rows = fleet.get("rows") or []
    global_halt = bool(fleet.get("global_halt"))
    alerting_count = int((fleet.get("summary") or {}).get("alerting") or 0)
    unhealthy = global_halt or alerting_count > 0  # R6

    baseline = state is None
    prev_strategies: dict[str, Any] = {} if state is None else (state.get("strategies") or {})
    prev_halt: dict[str, Any] = {} if state is None else (state.get("global_halt") or {})
    prev_unhealthy = unhealthy if state is None else bool(state.get("unhealthy"))

    notifications: list[dict[str, Any]] = []
    new_strategies: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("strategy"))
        health = str(row.get("health"))
        sev = _severity(health)
        operational = str(row.get("stage")) in _OPERATIONAL_STAGES
        prev = prev_strategies.get(name)
        prev_sev = _severity(prev.get("observed_health")) if isinstance(prev, dict) else None
        notified = prev.get("notified") if isinstance(prev, dict) else None
        # Migration tolerance: pre-cooldown state files lack the key -> null.
        last_bad_notified_at = (
            prev.get("last_bad_notified_at") if isinstance(prev, dict) else None
        )
        if health == "ok" or not operational:
            # The ONLY cooldown resets: recovery to ok / leaving operational
            # stages (disappearing rows drop out of new_strategies entirely).
            notified = None
            last_bad_notified_at = None
        # Account-wide halt fans out as health=halted on EVERY row; suppress the
        # per-strategy alert — the single global:halt notification carries it. A
        # row tripped for its own reason has global_halt false and still fires.
        account_halted_row = bool((row.get("kill_switch") or {}).get("global_halt"))
        candidate = (
            prev_sev is not None  # a first-ever observation is baseline, never an alert
            and operational
            and sev < _OK_SEVERITY
            and not account_halted_row
        )
        if candidate and (
            not _cooldown_active(last_bad_notified_at, now)
            or _halted_bypass(notified, health)
        ):
            notifications.append(_strategy_notification(row))
        new_strategies[name] = {
            "observed_health": health,
            "observed_at": now_iso,
            "notified": notified,
            "last_bad_notified_at": last_bad_notified_at,
        }

    halt_notified_at = None if baseline else prev_halt.get("notified_at")
    if not global_halt:
        halt_notified_at = None  # true->false: reset the marker
    elif not baseline and halt_notified_at is None:
        # Engaged and never successfully notified (fresh flip OR an unACKed send).
        notifications.append(
            {
                "kind": "global_halt",
                "urgency": "high",
                "payload": {
                    "title": "global halt engaged",
                    "body": "account-wide global halt is active",
                    "tag": "global:halt",
                    "url": "/",
                    "renotify": True,
                },
            }
        )

    if not baseline and prev_unhealthy and not unhealthy:
        # R6: ONE fleet-recovered on the unhealthy True->False transition.
        notifications.append(
            {
                "kind": "fleet_clear",
                "urgency": "normal",
                "payload": {
                    "title": "fleet recovered",
                    "body": "no alerting strategies; global halt clear",
                    "tag": "global:clear",
                    "url": "/",
                    "renotify": False,
                },
            }
        )

    if baseline or unhealthy or not prev_unhealthy:
        stored_unhealthy = unhealthy
    else:
        # True->False clears only when the fleet-recovered send is accepted (R5).
        stored_unhealthy = True

    new_state = {
        "strategies": new_strategies,
        "global_halt": {"observed": global_halt, "notified_at": halt_notified_at},
        "unhealthy": stored_unhealthy,
    }
    return notifications, new_state


def mark_notified(state: dict[str, Any], notification: dict[str, Any], now_iso: str) -> None:
    """Commit the notified side of one notification — ONLY after >=1 accepted send (R5)."""
    kind = notification["kind"]
    if kind == "strategy":
        entry = state["strategies"].get(notification["strategy"])
        if entry is not None:
            # notified.health stays for debug + the halted-bypass once-check;
            # the notification GATE is the last_bad_notified_at cooldown.
            entry["notified"] = {"health": notification["health"], "at": now_iso}
            entry["last_bad_notified_at"] = now_iso
    elif kind == "global_halt":
        state["global_halt"]["notified_at"] = now_iso
    elif kind == "fleet_clear":
        state["unhealthy"] = False


async def _diff_and_notify(fleet: dict[str, Any]) -> None:
    """One diff cycle: load state, diff, send, commit accepted marks, persist."""
    state = push.load_notify_state()
    now = datetime.now(UTC)
    notifications, new_state = diff_fleet(fleet, state, now=now)
    now_iso = now.isoformat()
    for notification in notifications:
        accepted = await push.send_to_all(
            notification["payload"], urgency=notification["urgency"]
        )
        if accepted >= 1:
            mark_notified(new_state, notification, now_iso)
    if new_state != state:
        push.save_notify_state(new_state)


async def poll_loop() -> None:
    """Prewarm the targets forever, sleeping POLL_SECONDS between passes.

    A failed target NEVER kills the loop (CliError -> warning; any other
    exception -> log.exception) and never stops the remaining targets in the
    pass. The fleet-health result is diffed for push alerts ONLY when it is
    genuinely fresh — a CliError or a stale-cache echo skips the diff entirely
    (R1: stale data must never read as a transition OR a recovery), and any
    exception in the diff half is logged and swallowed. Cancellation (server
    shutdown) propagates cleanly.
    """
    interval_s = poll_seconds()
    while True:
        for target_argv, ttl_s in PREWARM_TARGETS:
            try:
                result = await run_cli(*target_argv, ttl_s=ttl_s, force_refresh=True)
            except asyncio.CancelledError:
                raise
            except CliError as exc:
                logger.warning("poller: prewarm of %s failed: %s", " ".join(target_argv), exc)
                continue
            except Exception:
                logger.exception(
                    "poller: unexpected error prewarming %s", " ".join(target_argv)
                )
                continue
            if target_argv != _FLEET_ARGV or result.get("stale"):
                continue  # R1: only a genuinely fresh fleet payload is diffable
            try:
                await _diff_and_notify(result["data"])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("poller: push diff failed")
        await asyncio.sleep(interval_s)
