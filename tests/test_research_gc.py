"""Unit tests for the pure lifecycle-GC classifier (#510).

Pure: no filesystem, no DB — FileItem/RegistryEntry/now are built directly.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from algua.research.lifecycle_gc import (
    KEEP_ORPHANED_WITHIN_RETENTION,
    KEEP_PROTECTED_NON_TERMINAL,
    KEEP_RETIRED_WITHIN_RETENTION,
    KEEP_UNTRACKED_MODULE,
    REAP_ORPHANED_REPORT,
    REAP_RETIRED_EXPIRED,
    RETIRED,
    SURFACE_MODULE,
    SURFACE_REPORT,
    Classified,
    FileItem,
    RegistryEntry,
    classify,
    reapable,
    still_reapable,
)

NOW = datetime(2026, 7, 9, tzinfo=UTC)
RETENTION = 30.0


def _iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _mtime(days_ago: float) -> float:
    return (NOW - timedelta(days=days_ago)).timestamp()


def _classify_one(item: FileItem, registry: dict[str, RegistryEntry]) -> Classified:
    result = classify([item], registry, now=NOW, retention_days=RETENTION)
    assert len(result) == 1
    return result[0]


def test_retired_expired_module_is_reapable() -> None:
    item = FileItem("strategies/fam/old.py", "old", SURFACE_MODULE, 100)
    reg = {"old": RegistryEntry(RETIRED, _iso(45))}
    c = _classify_one(item, reg)
    assert c.reapable is True
    assert c.reason == REAP_RETIRED_EXPIRED
    assert c.age_days is not None and c.age_days >= RETENTION
    assert c.stage == RETIRED


def test_retired_expired_report_is_reapable() -> None:
    item = FileItem("kb/reports/old.md", "old", SURFACE_REPORT, 200)
    reg = {"old": RegistryEntry(RETIRED, _iso(60))}
    c = _classify_one(item, reg)
    assert c.reapable is True
    assert c.reason == REAP_RETIRED_EXPIRED


def test_retired_within_retention_is_kept() -> None:
    item = FileItem("strategies/fam/recent.py", "recent", SURFACE_MODULE, 100)
    reg = {"recent": RegistryEntry(RETIRED, _iso(5))}
    c = _classify_one(item, reg)
    assert c.reapable is False
    assert c.reason == KEEP_RETIRED_WITHIN_RETENTION
    assert c.age_days is not None and c.age_days < RETENTION


def test_retired_without_timestamp_is_kept_fail_safe() -> None:
    item = FileItem("strategies/fam/nots.py", "nots", SURFACE_MODULE, 100)
    reg = {"nots": RegistryEntry(RETIRED, None)}
    c = _classify_one(item, reg)
    assert c.reapable is False
    assert c.reason == KEEP_RETIRED_WITHIN_RETENTION
    assert c.age_days is None
    assert c.retired_at is None


def test_non_terminal_dormant_is_kept() -> None:
    item = FileItem("strategies/fam/d.py", "d", SURFACE_MODULE, 100)
    reg = {"d": RegistryEntry("dormant", None)}
    c = _classify_one(item, reg)
    assert c.reapable is False
    assert c.reason == KEEP_PROTECTED_NON_TERMINAL
    assert c.stage == "dormant"


def test_non_terminal_live_is_kept() -> None:
    item = FileItem("strategies/fam/l.py", "l", SURFACE_MODULE, 100)
    reg = {"l": RegistryEntry("live", None)}
    c = _classify_one(item, reg)
    assert c.reapable is False
    assert c.reason == KEEP_PROTECTED_NON_TERMINAL
    assert c.stage == "live"


def test_non_terminal_with_stale_retired_at_still_kept() -> None:
    # A stray retired_at on a non-terminal row must NOT make it reapable —
    # stage guard wins and age is not even computed.
    item = FileItem("strategies/fam/p.py", "p", SURFACE_MODULE, 100)
    reg = {"p": RegistryEntry("paper", _iso(999))}
    c = _classify_one(item, reg)
    assert c.reapable is False
    assert c.reason == KEEP_PROTECTED_NON_TERMINAL
    assert c.age_days is None


def test_orphaned_report_old_mtime_is_reapable() -> None:
    # An orphaned report (no registry row) is reapable ONLY once its own mtime is past the window.
    item = FileItem("kb/reports/ghost.md", "ghost", SURFACE_REPORT, 50, mtime=_mtime(60))
    c = _classify_one(item, {})
    assert c.reapable is True
    assert c.reason == REAP_ORPHANED_REPORT
    assert c.age_days is not None and c.age_days >= RETENTION
    assert c.stage is None
    assert c.retired_at is None


def test_orphaned_report_recent_mtime_is_kept() -> None:
    # A fresh orphaned report is within the retention window — fail-safe keep (never reap on sight).
    item = FileItem("kb/reports/fresh.md", "fresh", SURFACE_REPORT, 50, mtime=_mtime(5))
    c = _classify_one(item, {})
    assert c.reapable is False
    assert c.reason == KEEP_ORPHANED_WITHIN_RETENTION
    assert c.age_days is not None and c.age_days < RETENTION


def test_orphaned_report_no_mtime_is_kept_fail_safe() -> None:
    # No provable timestamp => never reapable.
    item = FileItem("kb/reports/notime.md", "notime", SURFACE_REPORT, 50)
    c = _classify_one(item, {})
    assert c.reapable is False
    assert c.reason == KEEP_ORPHANED_WITHIN_RETENTION
    assert c.age_days is None
    assert c.stage is None
    assert c.retired_at is None


def test_orphaned_report_mtime_boundary_exact_is_reapable() -> None:
    item = FileItem("kb/reports/edge.md", "edge", SURFACE_REPORT, 50, mtime=_mtime(RETENTION))
    c = _classify_one(item, {})
    assert c.reapable is True
    assert c.reason == REAP_ORPHANED_REPORT


def test_untracked_module_no_row_is_kept() -> None:
    item = FileItem("strategies/fam/wip.py", "wip", SURFACE_MODULE, 50)
    c = _classify_one(item, {})
    assert c.reapable is False
    assert c.reason == KEEP_UNTRACKED_MODULE
    assert c.age_days is None
    assert c.stage is None


def test_retention_boundary_exact_is_reapable() -> None:
    item = FileItem("strategies/fam/b.py", "b", SURFACE_MODULE, 100)
    reg = {"b": RegistryEntry(RETIRED, _iso(RETENTION))}
    c = _classify_one(item, reg)
    assert c.reapable is True
    assert c.reason == REAP_RETIRED_EXPIRED
    assert c.age_days == RETENTION


def test_retention_boundary_just_under_is_kept() -> None:
    item = FileItem("strategies/fam/b.py", "b", SURFACE_MODULE, 100)
    # 1 second under the wall.
    just_under = (NOW - timedelta(days=RETENTION) + timedelta(seconds=1)).isoformat()
    reg = {"b": RegistryEntry(RETIRED, just_under)}
    c = _classify_one(item, reg)
    assert c.reapable is False
    assert c.reason == KEEP_RETIRED_WITHIN_RETENTION
    assert c.age_days is not None and c.age_days < RETENTION


def test_tz_naive_retired_at_handled() -> None:
    # Naive ISO ts (no offset) must be stamped UTC and not raise.
    naive = (NOW - timedelta(days=45)).replace(tzinfo=None).isoformat()
    assert "+" not in naive and "Z" not in naive
    item = FileItem("strategies/fam/n.py", "n", SURFACE_MODULE, 100)
    reg = {"n": RegistryEntry(RETIRED, naive)}
    c = _classify_one(item, reg)
    assert c.reapable is True
    assert c.reason == REAP_RETIRED_EXPIRED
    assert c.age_days is not None
    assert abs(c.age_days - 45.0) < 1e-6


def test_classify_preserves_input_order() -> None:
    items = [
        FileItem("a.py", "a", SURFACE_MODULE, 1),
        FileItem("b.md", "b", SURFACE_REPORT, 2),
        FileItem("c.py", "c", SURFACE_MODULE, 3),
    ]
    result = classify(items, {}, now=NOW, retention_days=RETENTION)
    assert [c.path for c in result] == ["a.py", "b.md", "c.py"]


def test_reapable_orders_by_size_then_age_then_path() -> None:
    items = [
        FileItem("small_old.py", "small_old", SURFACE_MODULE, 100),
        FileItem("big.py", "big", SURFACE_MODULE, 500),
        FileItem("small_new.py", "small_new", SURFACE_MODULE, 100),
        FileItem("kept.py", "kept", SURFACE_MODULE, 9999),  # non-terminal -> excluded
    ]
    reg = {
        "small_old": RegistryEntry(RETIRED, _iso(90)),
        "big": RegistryEntry(RETIRED, _iso(40)),
        "small_new": RegistryEntry(RETIRED, _iso(35)),
        "kept": RegistryEntry("live", None),
    }
    classified = classify(items, reg, now=NOW, retention_days=RETENTION)
    ranked = reapable(classified)
    # kept is excluded; big (largest) first; then equal-size pair ordered older-first.
    assert [c.strategy for c in ranked] == ["big", "small_old", "small_new"]


def test_reapable_size_tie_older_age_first() -> None:
    items = [
        FileItem("younger.py", "younger", SURFACE_MODULE, 100),
        FileItem("older.py", "older", SURFACE_MODULE, 100),
    ]
    reg = {
        "younger": RegistryEntry(RETIRED, _iso(40)),
        "older": RegistryEntry(RETIRED, _iso(80)),
    }
    ranked = reapable(classify(items, reg, now=NOW, retention_days=RETENTION))
    assert [c.strategy for c in ranked] == ["older", "younger"]


def test_reapable_empty_when_nothing_reapable() -> None:
    items = [FileItem("wip.py", "wip", SURFACE_MODULE, 10)]
    assert reapable(classify(items, {}, now=NOW, retention_days=RETENTION)) == []


def test_still_reapable_retired_expired_true_when_still_past_retention() -> None:
    entry = RegistryEntry(RETIRED, _iso(90))
    assert still_reapable(
        REAP_RETIRED_EXPIRED, entry, now=NOW, retention_days=RETENTION) is True


def test_still_reapable_retired_expired_false_when_un_retired() -> None:
    """A stage-only recheck would miss this: the entry is simply not RETIRED any more."""
    entry = RegistryEntry("live", None)
    assert still_reapable(
        REAP_RETIRED_EXPIRED, entry, now=NOW, retention_days=RETENTION) is False


def test_still_reapable_retired_expired_false_when_re_retired_resets_clock() -> None:
    """The strategy reads RETIRED again (a stage-equality check alone would pass it), but its
    retired_at is FRESH — the retirement clock reset via an un-retire/re-retire round trip — so it
    must NOT be re-authorized (#510 GATE-2)."""
    entry = RegistryEntry(RETIRED, _iso(1))
    assert still_reapable(
        REAP_RETIRED_EXPIRED, entry, now=NOW, retention_days=RETENTION) is False


def test_still_reapable_retired_expired_false_when_no_entry_or_no_timestamp() -> None:
    assert still_reapable(REAP_RETIRED_EXPIRED, None, now=NOW, retention_days=RETENTION) is False
    assert still_reapable(
        REAP_RETIRED_EXPIRED, RegistryEntry(RETIRED, None), now=NOW,
        retention_days=RETENTION) is False


def test_still_reapable_orphaned_report_true_only_while_still_no_row() -> None:
    assert still_reapable(REAP_ORPHANED_REPORT, None, now=NOW, retention_days=RETENTION) is True


def test_still_reapable_orphaned_report_false_once_registered() -> None:
    """An orphan that got `registry add`ed between classify and move must not be archived."""
    entry = RegistryEntry("idea", None)
    assert still_reapable(
        REAP_ORPHANED_REPORT, entry, now=NOW, retention_days=RETENTION) is False
