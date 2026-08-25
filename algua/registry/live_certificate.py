"""Live-wall forward-test certificate re-verification (#124): the evidence precondition of
``paper/forward_tested -> live``.

Split out of ``registry/forward_promotion.py`` (mirroring ``registry/promotion.py`` /
``family_assignment.py`` for the shortlist gate). ``verify_forward_certificate`` re-checks that an
EARNED forward-gate pass is still valid at go-live time — freshness, a clean record since
certification, kill-switch/global-halt state, account continuity, and account hygiene — reusing
the same tick-parsing (``_parse_dt``) and activity-classification (``_classify_activities``) rules
``registry/forward_evidence.py`` uses to assemble the evidence in the first place, imported from
there rather than duplicated. NOT waivable in-band: there is deliberately no flag; a human who
must bypass owns the DB.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from algua.contracts.lifecycle import TransitionError
from algua.registry.forward_evidence import ActivitiesFetch, _classify_activities, _parse_dt
from algua.registry.repository import ArtifactIdentity, StrategyRepository
from algua.research.forward_gates import CERTIFICATE_FRESH_SESSIONS
from algua.risk.global_halt import is_engaged
from algua.risk.kill_switch import is_tripped

if TYPE_CHECKING:
    from algua.registry.forward_promotion import SessionCalendar


def verify_forward_certificate(
    repo: StrategyRepository,
    conn: sqlite3.Connection,
    *,
    name: str,
    strategy_id: int,
    identity: ArtifactIdentity,
    calendar: SessionCalendar,
    now: datetime,
    activities_fetch: ActivitiesFetch,
    account_id_fetch: Callable[[], str],
) -> dict[str, Any]:
    """The evidence precondition of the live wall (#124). NOT waivable in-band: there is
    deliberately no flag; a human who must bypass owns the DB. Selection is the NEWEST row
    pass-or-fail so a newer failed re-evaluation invalidates an older pass.

    Checks, in order, each failing closed with an actionable ``TransitionError``: a certificate
    exists for THIS strategy + the current identity; its newest verdict is a pass; it is fresh
    (at most ``CERTIFICATE_FRESH_SESSIONS`` sessions old); the record since certification is
    clean (no reconcile-failed and no malformed/future-stamped paper ticks after
    ``last_tick_id`` — the gate's own defect rule — no kill-switch trips after ``created_at``,
    kill switch and global halt clear); the broker the live gate queries is the SAME account
    the certificate was earned on (``account_id_fetch`` vs the row — account drift makes
    hygiene continuity unverifiable); and account hygiene re-verified over ``[created_at,
    now]`` with the same activity-classification rules the gate itself uses.
    Returns the certificate summary the human signs against."""
    # Normalize (not just tag) to UTC — same rule as assemble_forward_evidence: `.date()`
    # freshness arithmetic must use the UTC date, not a local one.
    now_utc = now.astimezone(UTC) if now.tzinfo is not None else now.replace(tzinfo=UTC)
    row = repo.latest_forward_gate_row(
        strategy_id, identity.code_hash, identity.config_hash, identity.dependency_hash)
    if row is None:
        raise TransitionError(
            "go-live requires a forward-test certificate for the current "
            "code+config+dependency; run `algua paper promote`")
    if not row["passed"]:
        raise TransitionError(
            f"the newest forward-gate evaluation (id={row['id']}, created_at="
            f"{row['created_at']}) FAILED, which invalidates any prior pass; "
            "re-run `algua paper promote`")
    age = calendar.sessions_between(
        datetime.fromisoformat(row["created_at"]).date(), now_utc.date())
    if age > CERTIFICATE_FRESH_SESSIONS:
        raise TransitionError(
            f"the forward-test certificate is stale: {age} sessions old, max "
            f"{CERTIFICATE_FRESH_SESSIONS}; re-run `algua paper promote` to refresh it")
    ticks_since = conn.execute(
        "SELECT tick_ts, reconcile_ok FROM tick_snapshots WHERE lane='paper' AND strategy_id=?"
        " AND id > ?",
        (strategy_id, row["last_tick_id"] or 0),
    ).fetchall()
    n_bad_ticks = sum(1 for t in ticks_since if not t["reconcile_ok"])
    if n_bad_ticks:
        raise TransitionError(
            f"{n_bad_ticks} reconcile-failed paper tick(s) since certification; investigate,"
            " then re-run `algua paper promote`")
    n_malformed = sum(
        1 for t in ticks_since
        if (tick_dt := _parse_dt(t["tick_ts"])) is None or tick_dt > now_utc)
    if n_malformed:
        raise TransitionError(
            f"{n_malformed} malformed or future-stamped paper tick(s) since certification"
            " (the gate's defective-tick rule); investigate, then re-run `algua paper promote`")
    n_trips = conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE strategy=? AND action='kill_switch_trip'"
        " AND ts >= ?",
        (name, row["created_at"]),
    ).fetchone()[0]
    if n_trips:
        raise TransitionError(
            f"{n_trips} kill-switch trip(s) since certification; investigate, then re-run"
            " `algua paper promote`")
    if is_tripped(conn, name) or is_engaged(conn):
        raise TransitionError(
            "kill switch / global halt engaged; clear it before going live")
    # Account continuity: the hygiene re-check below queries whatever account the CURRENT broker
    # credentials point at, while the certificate's evidence lives on row["account_id"]. If they
    # differ (the operator switched paper accounts after certification), a deposit or manual fill
    # on the certified account would be invisible here — unverifiable continuity fails closed.
    if row["account_id"] is None:
        raise TransitionError(
            "the forward-test certificate records no account_id, so hygiene continuity since "
            "certification is unverifiable; re-run `algua paper promote`")
    try:
        current_account = account_id_fetch()
    except Exception as exc:
        raise TransitionError(
            f"could not verify the broker account id ({exc}); failing closed") from exc
    if current_account != row["account_id"]:
        raise TransitionError(
            f"the broker credentials point at account {current_account!r} but the certificate "
            f"was earned on account {row['account_id']!r}; hygiene continuity since "
            "certification is unverifiable — re-run `algua paper promote` on this account")
    # Alpaca's activities `after` bound is EXCLUSIVE: a capital movement stamped at EXACTLY
    # the certification instant would escape an `after == created_at` window, so widen the
    # start 1s earlier. The overlap errs fail-closed — an extra pre-window movement can only
    # FAIL the wall, never pass it; pre-window FILLs from ANY paper-book strategy's order on
    # this account remain attributable (account-level attribution).
    window_after = (datetime.fromisoformat(row["created_at"]) - timedelta(seconds=1)).isoformat()
    try:
        acts = activities_fetch(window_after, now_utc.isoformat())
    except Exception as exc:
        raise TransitionError(
            f"could not verify account activities since certification ({exc}); "
            "failing closed") from exc
    n_external, n_unattributable = _classify_activities(conn, acts)
    if n_external or n_unattributable:
        raise TransitionError(
            f"account hygiene failed since certification: {n_external} external capital "
            f"flow(s), {n_unattributable} unattributable fill(s); re-run `algua paper promote`")
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "realized_sharpe": row["realized_sharpe"],
        "holdout_sharpe": row["holdout_sharpe"],
        "n_forward_observations": row["n_forward_observations"],
        "n_concurrent_forward": row["n_concurrent_forward"],
    }
