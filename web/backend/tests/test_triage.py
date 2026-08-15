"""The Now screen's ranked "needs you" assembly — pure, so every rule is testable directly."""

from __future__ import annotations

from typing import Any

from backend.triage import build_triage


def _fleet(rows: list[dict[str, Any]] | None = None, *, global_halt: bool = False,
           alerting: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else []
    alerting = alerting if alerting is not None else []
    return {
        "ok": not alerting and not global_halt,
        "global_halt": global_halt,
        "alerting": alerting,
        "summary": {"total": len(rows), "alerting": len(alerting), "by_health": {}},
        "rows": rows,
    }


def _row(name: str, health: str = "ok", stage: str = "paper", **extra: Any) -> dict[str, Any]:
    return {"strategy": name, "stage": stage, "health": health,
            "staleness_sessions": 0, "kill_switch": None, **extra}


def _ops(**loops: Any) -> dict[str, Any]:
    base = {"research": {"health": "ok"}, "paper": {"health": "ok"},
            "mergeback": {"health": "idle"}}
    base.update(loops)
    alerting = [n for n, r in base.items()
                if r["health"] in ("failing", "rate_limited", "stale", "unknown")]
    return {"ok": not alerting, "alerting": alerting, "loops": base}


def _book(unallocated: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    return {"ok": not unallocated, "capacity": 64, "allocated": 7,
            "unallocated_operational": unallocated or [], "slices": [], **extra}


def test_a_quiet_system_produces_no_items() -> None:
    result = build_triage(_fleet([_row("a")]), _ops(), _book())
    assert result["items"] == []


def test_an_idle_loop_is_not_an_alert() -> None:
    """A never-run operator or an empty merge-back queue is a quiet system, not a broken one."""
    result = build_triage(_fleet(), _ops(paper={"health": "idle"}), _book())
    assert result["items"] == []


def test_a_dead_loop_outranks_a_stranded_strategy_and_an_unhealthy_tick() -> None:
    """The 2026-08-15 ordering: silent machine failure first, then capital, then one strategy."""
    result = build_triage(
        _fleet(alerting=[_row("momo", "stale")]),
        _ops(research={"health": "rate_limited", "detail": "provider usage limit reached"}),
        _book([{"strategy": "stranded", "stage": "paper", "since": "2026-08-14T12:12:43+00:00",
                "ever_ticked": False}]),
    )
    assert [item["kind"] for item in result["items"]] == [
        "loop_down", "capital_stranded", "strategy",
    ]


def test_a_wedged_queue_ranks_below_a_strategy_because_nothing_is_lost_yet() -> None:
    result = build_triage(
        _fleet(alerting=[_row("momo", "stale")]),
        _ops(mergeback={"health": "stale", "detail": "oldest item outlived several drain cycles"}),
        _book(),
    )
    assert [item["kind"] for item in result["items"]] == ["strategy", "queue_wedged"]


def test_every_item_carries_an_actionable_detail() -> None:
    result = build_triage(
        _fleet(),
        _ops(research={"health": "failing", "detail": "exit 1"}),
        _book([{"strategy": "stranded", "stage": "paper", "since": "...", "ever_ticked": False}]),
    )
    assert all(item["detail"] for item in result["items"])
    stranded = next(i for i in result["items"] if i["kind"] == "capital_stranded")
    assert "never ticked" in stranded["detail"]
    assert "skips it" in stranded["detail"]


def test_global_halt_collapses_the_per_strategy_fan_out() -> None:
    """A halt marks EVERY row halted; listing them all buries the one line that matters."""
    rows = [_row(f"s{n}", "halted") for n in range(5)]
    result = build_triage(_fleet(rows, global_halt=True, alerting=rows), _ops(), _book())
    assert [item["kind"] for item in result["items"]] == ["global_halt"]


def test_a_failed_part_contributes_nothing_and_is_reported_not_hidden() -> None:
    """A degraded source must never render as all-clear."""
    result = build_triage(_fleet(), None, _book())
    assert result["sources"] == {"fleet": True, "ops": False, "book": True}


def test_headline_fleet_ok_counts_every_row_not_just_alerting_ones() -> None:
    """summary.by_health covers ALERTING rows only — the headline must count `rows`."""
    rows = [_row("a"), _row("b"), _row("c", "idle", stage="idea")]
    result = build_triage(_fleet(rows), _ops(), _book())
    assert result["headline"]["fleet_ok"] == 2
    assert result["headline"]["fleet_total"] == 3


def test_headline_survives_every_part_being_absent() -> None:
    result = build_triage(None, None, None)
    assert result["items"] == []
    assert result["headline"]["fleet_ok"] == 0
    assert result["headline"]["book_allocated"] is None


def test_malformed_rows_are_skipped_rather_than_crashing_the_screen() -> None:
    fleet = _fleet(alerting=["not a dict"])  # type: ignore[list-item]
    fleet["rows"] = ["also not a dict"]
    ops = {"loops": {"research": "not a dict"}, "alerting": []}
    result = build_triage(fleet, ops, {"unallocated_operational": [None]})
    assert result["items"] == []
