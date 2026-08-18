"""Tests for `.codex/scripts/mergeback_queue.py` (factory slice 3): the durable, dual-writer
merge-back queue (`data/mergeback-queue.json` + `data/mergeback-queue.lock`).

Imported directly via `importlib` from its repo path (it's a standalone script, not part of the
installed `algua` package — see the module's own docstring for why).
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = REPO_ROOT / ".codex" / "scripts" / "mergeback_queue.py"

_spec = importlib.util.spec_from_file_location("mergeback_queue", _MOD_PATH)
mergeback_queue = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mergeback_queue)


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "queue.json", tmp_path / "queue.lock"


def _ctx(**over):
    """A minimal VALID eval_context recipe (mergeback authoritative intake)."""
    ctx = {"demo": True, "sweep_grid": {"lookback": [10, 20]}, "rank_by": "mean_sharpe"}
    ctx.update(over)
    return ctx


def _item(strategy="strat_a", branch="research-run/1", universe="sp500",
          start="2024-01-01", end="2024-06-01", eval_context=None):
    return dict(strategy=strategy, universe=universe, start=start, end=end, branch=branch,
                eval_context=eval_context if eval_context is not None else _ctx())


# --- enqueue -------------------------------------------------------------------------------------


def test_enqueue_creates_file_and_item(paths):
    queue_path, lock_path = paths
    assert not queue_path.exists()
    result = mergeback_queue.enqueue(queue_path, lock_path, **_item())
    assert result["created"] is True
    assert queue_path.exists()
    data = json.loads(queue_path.read_text())
    key = "strat_a@research-run/1"
    assert key in data["items"]
    item = data["items"][key]
    assert item["status"] == "pending"
    assert item["attempts"] == 0
    assert item["last_attempt_at"] is None
    assert item["last_result"] is None
    assert item["universe"] == "sp500"


def test_enqueue_is_idempotent_leaves_in_flight_item_untouched(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    # Simulate an in-flight attempt via record_attempt (gate_failed, retryable).
    mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1",
        stdout_text=json.dumps({"ok": True, "status": "gate_failed"}),
    )
    before = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert before["status"] == "gate_failed"
    assert before["attempts"] == 1

    # Re-enqueueing the SAME key must not reset attempts/status.
    result = mergeback_queue.enqueue(queue_path, lock_path, **_item())
    assert result["created"] is False
    after = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert after == before


def test_enqueue_leaves_a_terminal_item_untouched_too(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1",
        stdout_text=json.dumps({"ok": True, "status": "promoted_allocated"}),
    )
    before = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert before["status"] == "promoted_allocated"

    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    after = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert after == before


def test_enqueue_distinct_branches_are_distinct_keys(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item(branch="research-run/1"))
    mergeback_queue.enqueue(queue_path, lock_path, **_item(branch="research-run/2"))
    items = json.loads(queue_path.read_text())["items"]
    assert set(items) == {"strat_a@research-run/1", "strat_a@research-run/2"}


# --- select_eligible -------------------------------------------------------------------------


def test_select_eligible_picks_a_pending_item(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    result = mergeback_queue.select_eligible(queue_path, lock_path)
    assert result["key"] == "strat_a@research-run/1"
    assert result["item"]["strategy"] == "strat_a"


def test_select_eligible_returns_none_key_when_nothing_eligible(paths):
    queue_path, lock_path = paths
    result = mergeback_queue.select_eligible(queue_path, lock_path)
    assert result == {"key": None, "item": None}


def test_select_eligible_skips_terminal_items(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1",
        stdout_text=json.dumps({"ok": True, "status": "diff_policy_rejected"}),
    )
    result = mergeback_queue.select_eligible(queue_path, lock_path)
    assert result == {"key": None, "item": None}


def test_select_eligible_exactly_one_of_multiple_pending(paths):
    queue_path, lock_path = paths
    for i in range(5):
        mergeback_queue.enqueue(queue_path, lock_path, **_item(
            strategy=f"strat_{i}", branch="research-run/1"))
    result = mergeback_queue.select_eligible(queue_path, lock_path)
    assert result["key"] is not None
    # Exactly one item, and it's the FIRST one enqueued (FIFO by insertion order).
    assert result["key"] == "strat_0@research-run/1"


def test_select_eligible_gate_failed_under_backoff_window_not_eligible(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1",
        stdout_text=json.dumps({"ok": True, "status": "gate_failed"}),
    )
    # attempts=1 -> backoff window is 10 minutes from `last_attempt_at`, which is "now" — not
    # elapsed yet.
    result = mergeback_queue.select_eligible(
        queue_path, lock_path, backoff_minutes_per_attempt=10)
    assert result == {"key": None, "item": None}


def test_select_eligible_gate_failed_after_backoff_elapses(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1",
        stdout_text=json.dumps({"ok": True, "status": "gate_failed"}),
    )
    # Zero-minute backoff => always eligible immediately.
    result = mergeback_queue.select_eligible(
        queue_path, lock_path, backoff_minutes_per_attempt=0)
    assert result["key"] == "strat_a@research-run/1"


def test_select_eligible_gate_failed_at_attempt_cap_not_eligible(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    for _ in range(3):
        mergeback_queue.record_attempt(
            queue_path, lock_path, key="strat_a@research-run/1",
            stdout_text=json.dumps({"ok": True, "status": "gate_failed"}),
            max_attempts=3,
        )
    item = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert item["status"] == "terminal_failed"
    result = mergeback_queue.select_eligible(
        queue_path, lock_path, max_attempts=3, backoff_minutes_per_attempt=0)
    assert result == {"key": None, "item": None}


# --- classify_attempt (pure) -----------------------------------------------------------------


@pytest.mark.parametrize("status", ["promoted_allocated", "promoted_queued", "already_done"])
def test_classify_success_terminal(status):
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": status}, attempts=0, max_attempts=3)
    assert verdict == {"action": "terminal", "status": status, "attempts": 1,
                       "transient_failures": 0}


@pytest.mark.parametrize("status", ["diff_policy_rejected", "promote_failed"])
def test_classify_hard_failure_terminal(status):
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": status}, attempts=0, max_attempts=3)
    assert verdict == {"action": "terminal", "status": status, "attempts": 1,
                       "transient_failures": 0}


def test_classify_gate_failed_retryable_under_cap():
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": "gate_failed"}, attempts=0, max_attempts=3)
    assert verdict == {"action": "retry", "status": "gate_failed", "attempts": 1,
                       "transient_failures": 0}


def test_classify_gate_failed_exhausted_at_cap():
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": "gate_failed"}, attempts=2, max_attempts=3)
    assert verdict == {"action": "exhausted", "status": "gate_failed", "attempts": 3,
                       "transient_failures": 0}


def test_classify_lock_contention_is_not_an_attempt():
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "ran": False, "reason": "locked"}, attempts=1, max_attempts=3,
        transient_failures=2)
    # Lock contention is a legitimate no-op — it neither burns an attempt nor spends the transient
    # budget (no work was even started, so it carries no signal about the item's health).
    assert verdict == {"action": "lock_contention", "status": None, "attempts": 1,
                       "transient_failures": 2}


def test_classify_unparseable_payload_is_transient_not_an_attempt():
    verdict = mergeback_queue.classify_attempt(None, attempts=1, max_attempts=3)
    assert verdict == {"action": "transient_failure", "status": None, "attempts": 1,
                       "transient_failures": 1}


def test_classify_ok_false_envelope_is_transient_not_an_attempt():
    # e.g. a fail-closed exception out of `paper merge-back` itself (RemoteMovedError etc.), or a
    # lock-run setup failure (git_dir_unresolved/lock_unavailable) — neither is branch content.
    verdict = mergeback_queue.classify_attempt(
        {"ok": False, "error": "boom", "code": "internal"}, attempts=1, max_attempts=3)
    assert verdict == {"action": "transient_failure", "status": None, "attempts": 1,
                       "transient_failures": 1}


def test_classify_unrecognized_status_is_transient_not_an_attempt():
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": "some_future_status"}, attempts=1, max_attempts=3)
    assert verdict == {"action": "transient_failure", "status": None, "attempts": 1,
                       "transient_failures": 1}


# --- the transient-failure budget (#549: a repeating infra failure must not block the queue head) -


def test_classify_transient_failures_accumulate_up_to_the_budget():
    budget = mergeback_queue.MAX_TRANSIENT_FAILURES
    verdict = mergeback_queue.classify_attempt(
        None, attempts=0, max_attempts=3, transient_failures=budget - 2)
    assert verdict["action"] == "transient_failure"
    assert verdict["transient_failures"] == budget - 1


def test_classify_transient_failures_exhaust_the_budget():
    # The drainer selects FIFO, so an item that fails the SAME unclassifiable way every cycle would
    # otherwise sit at the head forever, re-running the (multi-minute) quality gate and starving
    # every item behind it. The budget makes it terminal so the queue drains past it.
    budget = mergeback_queue.MAX_TRANSIENT_FAILURES
    verdict = mergeback_queue.classify_attempt(
        None, attempts=0, max_attempts=3, transient_failures=budget - 1)
    assert verdict == {"action": "transient_exhausted", "status": None, "attempts": 0,
                       "transient_failures": budget}


def test_classify_a_real_outcome_resets_the_transient_streak():
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": "gate_failed"}, attempts=0, max_attempts=3, transient_failures=2)
    assert verdict["transient_failures"] == 0


# --- record_attempt (queue side-effects) --------------------------------------------------------


def test_record_attempt_lock_contention_leaves_item_completely_untouched(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    before = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]

    result = mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1",
        stdout_text=json.dumps({"ok": True, "ran": False, "reason": "locked"}),
    )
    assert result["action"] == "lock_contention"
    after = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert after == before  # not even last_attempt_at moved


def test_record_attempt_transient_failure_leaves_the_attempt_record_untouched(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    before = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]

    result = mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1", stdout_text="not json at all",
    )
    assert result["action"] == "transient_failure"
    after = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    # No real merge-back attempt happened, so the attempt record is untouched — only the
    # consecutive-transient counter (the head-of-line budget) moves.
    assert (before["transient_failures"], after["transient_failures"]) == (0, 1)
    assert {k: v for k, v in after.items() if k != "transient_failures"} == \
           {k: v for k, v in before.items() if k != "transient_failures"}
    assert after["attempts"] == 0


def test_record_attempt_transient_failures_terminal_fail_at_the_budget(paths):
    # #549: an item whose every invocation fails the SAME unclassifiable way (exit 127, a git
    # error, a full disk) sat at the FIFO head forever, re-running the multi-minute quality gate
    # each cycle and starving every item behind it. The budget terminal-fails it so the queue
    # drains past it — and the retained last_result says WHY, for the operator.
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    key = "strat_a@research-run/1"
    payload = {"ok": False, "error": "git commit --no-edit returned 1", "code": "internal"}
    actions = [
        mergeback_queue.record_attempt(
            queue_path, lock_path, key=key, stdout_text=json.dumps(payload))["action"]
        for _ in range(mergeback_queue.MAX_TRANSIENT_FAILURES)
    ]
    assert actions[:-1] == ["transient_failure"] * (mergeback_queue.MAX_TRANSIENT_FAILURES - 1)
    assert actions[-1] == "transient_exhausted"
    item = json.loads(queue_path.read_text())["items"][key]
    assert item["status"] == "terminal_failed"
    assert item["last_result"] == payload
    assert item["attempts"] == 0  # no REAL merge-back attempt ever completed


def test_record_attempt_a_real_outcome_clears_the_transient_streak(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    key = "strat_a@research-run/1"
    mergeback_queue.record_attempt(queue_path, lock_path, key=key, stdout_text="garbage")
    assert json.loads(queue_path.read_text())["items"][key]["transient_failures"] == 1
    mergeback_queue.record_attempt(
        queue_path, lock_path, key=key,
        stdout_text=json.dumps({"ok": True, "status": "gate_failed"}))
    item = json.loads(queue_path.read_text())["items"][key]
    assert item["transient_failures"] == 0
    assert item["status"] == "gate_failed" and item["attempts"] == 1


def test_record_attempt_lock_contention_does_not_spend_the_transient_budget(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    key = "strat_a@research-run/1"
    mergeback_queue.record_attempt(queue_path, lock_path, key=key, stdout_text="garbage")
    for _ in range(mergeback_queue.MAX_TRANSIENT_FAILURES + 2):
        result = mergeback_queue.record_attempt(
            queue_path, lock_path, key=key,
            stdout_text=json.dumps({"ok": True, "ran": False, "reason": "locked"}))
        assert result["action"] == "lock_contention"
    item = json.loads(queue_path.read_text())["items"][key]
    assert item["transient_failures"] == 1
    assert item["status"] == "pending"


def test_transient_exhausted_releases_the_queue_head_to_the_next_item(paths):
    # The end-to-end property the budget exists for: a permanently-failing head item must stop
    # being selected, so the item behind it finally gets a turn.
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    mergeback_queue.enqueue(
        queue_path, lock_path, **{**_item(), "strategy": "strat_b"})
    head, behind = "strat_a@research-run/1", "strat_b@research-run/1"
    for _ in range(mergeback_queue.MAX_TRANSIENT_FAILURES):
        assert mergeback_queue.select_and_reserve(queue_path, lock_path)["key"] == head
        mergeback_queue.record_attempt(queue_path, lock_path, key=head, stdout_text="garbage")
    assert mergeback_queue.select_and_reserve(queue_path, lock_path)["key"] == behind


def test_record_attempt_terminal_success_updates_status_and_result(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    payload = {"ok": True, "status": "promoted_allocated", "strategy": "strat_a"}
    result = mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1", stdout_text=json.dumps(payload),
    )
    assert result["action"] == "terminal"
    item = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert item["status"] == "promoted_allocated"
    assert item["attempts"] == 1
    assert item["last_attempt_at"] is not None
    assert item["last_result"] == payload


def test_record_attempt_missing_key_reports_missing(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    result = mergeback_queue.record_attempt(
        queue_path, lock_path, key="no_such_key", stdout_text=json.dumps({"ok": True}))
    assert result["action"] == "missing_key"


# --- record_attempt: robust JSON extraction from noisy REAL stdout ------------------------------
#
# `paper merge-back` runs the FULL quality gate (pytest/ruff/mypy/lint-imports) with INHERITED
# stdout before printing its own final `emit()` JSON line (see algua/cli/paper_cmd.py's
# `_run_quality_gate`), so on every REAL invocation the drainer's captured stdout is
# "<gate tool noise>\n<final JSON line>" — never just the bare JSON object the mocked-subprocess
# unit tests above use. A parser requiring the WHOLE stdout to be exactly one JSON object returns
# None on every real attempt (the bug this fixes).


def test_record_attempt_parses_last_json_line_amid_real_gate_noise(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    payload = {"ok": True, "status": "promoted_allocated", "strategy": "strat_a"}
    noisy_stdout = (
        "182 passed in 4.2s\n"
        "All checks passed!\n"
        "Success: no issues found\n"
        "Contracts: 24 kept\n"
        + json.dumps(payload)
    )
    result = mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1", stdout_text=noisy_stdout,
    )
    assert result["action"] == "terminal"
    item = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert item["status"] == "promoted_allocated"
    assert item["attempts"] == 1
    assert item["last_result"] == payload


def test_record_attempt_parses_json_object_not_on_the_final_line(paths):
    # Leading tool noise PLUS trailing blank line(s)/newline after the JSON object — the object is
    # not literally the last thing in the buffer either.
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    payload = {"ok": True, "status": "gate_failed"}
    noisy_stdout = "some preamble tool output\nmore noise\n" + json.dumps(payload) + "\n\n"
    result = mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1", stdout_text=noisy_stdout,
    )
    assert result["action"] == "retry"
    item = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert item["status"] == "gate_failed"
    assert item["attempts"] == 1


# --- select_and_reserve: atomic reserve step (TOCTOU fix) ----------------------------------------


def test_select_and_reserve_then_record_attempt_lifecycle(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())

    reserved = mergeback_queue.select_and_reserve(queue_path, lock_path)
    assert reserved["key"] == "strat_a@research-run/1"
    assert reserved["item"]["status"] == "in_progress"
    on_disk = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert on_disk["status"] == "in_progress"
    assert on_disk["reserved_at"] is not None

    # A second reserve attempt finds nothing else eligible (only item is now in_progress, fresh).
    assert mergeback_queue.select_and_reserve(queue_path, lock_path) == {
        "key": None, "item": None,
    }

    outcome = mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1",
        stdout_text=json.dumps({"ok": True, "status": "promoted_allocated"}),
    )
    assert outcome["action"] == "terminal"
    final = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert final["status"] == "promoted_allocated"
    assert final["attempts"] == 1


def test_select_and_reserve_prefers_pending_over_non_stale_in_progress(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(
        queue_path, lock_path, **_item(strategy="strat_a", branch="research-run/1"))
    mergeback_queue.enqueue(
        queue_path, lock_path, **_item(strategy="strat_b", branch="research-run/1"))

    first = mergeback_queue.select_and_reserve(queue_path, lock_path)
    assert first["key"] == "strat_a@research-run/1"  # FIFO: reserved first

    # strat_b is still pending, strat_a is now in_progress (non-stale) -> strat_b must win next.
    second = mergeback_queue.select_and_reserve(queue_path, lock_path)
    assert second["key"] == "strat_b@research-run/1"

    # Nothing left eligible: both items are now in_progress and fresh.
    assert mergeback_queue.select_and_reserve(queue_path, lock_path) == {
        "key": None, "item": None,
    }


def test_stale_in_progress_reservation_becomes_eligible_again(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    mergeback_queue.select_and_reserve(queue_path, lock_path)

    # Fresh reservation: not eligible yet.
    assert mergeback_queue.select_and_reserve(queue_path, lock_path) == {
        "key": None, "item": None,
    }

    # Backdate the reservation past the staleness threshold (simulating a crashed/killed drainer).
    data = json.loads(queue_path.read_text())
    stale_ts = (
        datetime.now(UTC)
        - timedelta(seconds=mergeback_queue.RESERVATION_STALE_SECONDS + 60)
    ).isoformat()
    data["items"]["strat_a@research-run/1"]["reserved_at"] = stale_ts
    queue_path.write_text(json.dumps(data))

    result = mergeback_queue.select_and_reserve(queue_path, lock_path)
    assert result["key"] == "strat_a@research-run/1"
    assert result["item"]["status"] == "in_progress"


def test_select_and_reserve_releases_reservation_on_lock_contention(paths):
    # A benign lock-contention outcome is NOT a real attempt: the reservation must be released
    # back to "pending" immediately, not left wedged `in_progress` until it goes stale.
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    mergeback_queue.select_and_reserve(queue_path, lock_path)

    result = mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1",
        stdout_text=json.dumps({"ok": True, "ran": False, "reason": "locked"}),
    )
    assert result["action"] == "lock_contention"
    item = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert item["status"] == "pending"
    assert item["attempts"] == 0
    assert "reserved_at" not in item

    # Immediately eligible again (no need to wait out RESERVATION_STALE_SECONDS).
    again = mergeback_queue.select_and_reserve(queue_path, lock_path)
    assert again["key"] == "strat_a@research-run/1"


def test_concurrent_select_and_reserve_only_one_wins(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())

    results: list[dict] = []
    results_lock = threading.Lock()

    def _attempt() -> None:
        r = mergeback_queue.select_and_reserve(queue_path, lock_path)
        with results_lock:
            results.append(r)

    threads = [threading.Thread(target=_attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(not t.is_alive() for t in threads)
    winners = [r for r in results if r["key"] is not None]
    assert len(winners) == 1
    assert winners[0]["key"] == "strat_a@research-run/1"
    item = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert item["status"] == "in_progress"


# --- concurrency: enqueue + record_attempt racing must not lose an update -------------------------


def test_concurrent_enqueue_and_record_attempt_no_lost_update(paths, tmp_path):
    # Realistic production contention is exactly TWO writers: the research driver (enqueue) and
    # the drainer (record_attempt) — never more than one of each running at a time. This test
    # races exactly two threads, each performing a SERIES of separate lock acquisitions against
    # the same file, to prove no update is lost under real (not synthetically extreme) contention.
    queue_path, lock_path = paths
    n_strategies = 12
    for i in range(n_strategies):
        mergeback_queue.enqueue(queue_path, lock_path, **_item(
            strategy=f"strat_{i}", branch="research-run/1"))

    corruption_seen: list[Exception] = []
    stop = threading.Event()

    def _watcher():
        # Best-effort: repeatedly read the file while the two writers race and confirm it is
        # ALWAYS valid, fully-formed JSON — never a torn/partial write.
        while not stop.is_set():
            try:
                data = json.loads(queue_path.read_text())
                assert isinstance(data.get("items"), dict)
            except FileNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001
                corruption_seen.append(exc)

    watcher = threading.Thread(target=_watcher, daemon=True)
    watcher.start()

    def _drainer_thread() -> None:
        # Simulates the drainer: record one attempt per pre-enqueued strategy.
        for i in range(n_strategies):
            mergeback_queue.record_attempt(
                queue_path, lock_path, key=f"strat_{i}@research-run/1",
                stdout_text=json.dumps({"ok": True, "status": "promoted_allocated"}),
            )

    def _driver_thread() -> None:
        # Simulates the research driver: enqueue a fresh batch of NEW candidates concurrently.
        for i in range(n_strategies):
            mergeback_queue.enqueue(queue_path, lock_path, **_item(
                strategy=f"new_strat_{i}", branch="research-run/2"))

    t_drainer = threading.Thread(target=_drainer_thread)
    t_driver = threading.Thread(target=_driver_thread)
    t_drainer.start()
    t_driver.start()
    t_drainer.join(timeout=10)
    t_driver.join(timeout=10)
    stop.set()
    watcher.join(timeout=5)

    assert not t_drainer.is_alive() and not t_driver.is_alive()
    assert corruption_seen == []
    data = json.loads(queue_path.read_text())["items"]
    # Every original strategy's record_attempt landed (no lost update)...
    for i in range(n_strategies):
        assert data[f"strat_{i}@research-run/1"]["status"] == "promoted_allocated"
    # ...and every new enqueue landed too.
    for i in range(n_strategies):
        assert f"new_strat_{i}@research-run/2" in data
    assert len(data) == 2 * n_strategies


# --- CLI smoke (the drainer's actual invocation surface) ------------------------------------------


def test_cli_select_shell_format_emits_sourceable_assignments(paths, capsys):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    rc = mergeback_queue.main([
        "select", "--queue", str(queue_path), "--lock", str(lock_path), "--format", "shell",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MERGEBACK_SELECTED=1" in out
    assert "MERGEBACK_STRATEGY=strat_a" in out


def test_cli_select_shell_format_none_selected(paths, capsys):
    queue_path, lock_path = paths
    rc = mergeback_queue.main([
        "select", "--queue", str(queue_path), "--lock", str(lock_path), "--format", "shell",
    ])
    assert rc == 0
    assert "MERGEBACK_SELECTED=0" in capsys.readouterr().out


def test_cli_record_attempt_via_stdin(paths, capsys, monkeypatch):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(
        json.dumps({"ok": True, "status": "already_done"})))
    rc = mergeback_queue.main([
        "record-attempt", "--queue", str(queue_path), "--lock", str(lock_path),
        "--key", "strat_a@research-run/1", "--stdin",
    ])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["action"] == "terminal"
    item = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert item["status"] == "already_done"


# --- eval_context validation matrix (mergeback authoritative intake) ------------------------------


def test_validate_eval_context_canonicalizes_a_valid_recipe():
    out = mergeback_queue.validate_eval_context({
        "snapshot": "bars-2026-08-01", "fundamentals_snapshot": "fund.1",
        "sweep_grid": {"zeta": [1, 2], "alpha": [0.5, 1.5], "construction.top_k": [5, 10]},
        "rank_by": "min_sharpe",
    })
    assert out["snapshot"] == "bars-2026-08-01"
    assert out["fundamentals_snapshot"] == "fund.1"
    assert out["rank_by"] == "min_sharpe"
    # Keys are canonical-sorted; a mixed int list stays int, floats stay float.
    assert list(out["sweep_grid"]) == ["alpha", "construction.top_k", "zeta"]
    assert out["sweep_grid"]["zeta"] == [1, 2]
    assert "demo" not in out


def test_validate_eval_context_widens_mixed_numeric_lists_like_parse_grid():
    out = mergeback_queue.validate_eval_context(
        {"demo": True, "sweep_grid": {"k": [10, 10.5]}})
    assert out["sweep_grid"]["k"] == [10.0, 10.5]
    assert all(type(v) is float for v in out["sweep_grid"]["k"])


@pytest.mark.parametrize("ctx,msg", [
    ("nope", "must be a JSON object"),
    ({"sweep_grid": {"k": [1]}}, "exactly one of demo"),                       # neither source
    ({"demo": True, "snapshot": "s", "sweep_grid": {"k": [1]}}, "exactly one of demo"),
    ({"demo": 1, "sweep_grid": {"k": [1]}}, "JSON literal true"),
    ({"demo": True, "snapshot": None, "sweep_grid": {"k": [1]},
      "windows": 4}, "unknown keys"),                                          # never transported
    ({"demo": True, "sweep_grid": {"k": [1]}, "holdout_frac": 0.2}, "unknown keys"),
    ({"snapshot": "bad id!", "sweep_grid": {"k": [1]}}, "format check"),
    ({"demo": True, "sweep_grid": {"k": [1]}, "rank_by": "sharpe"}, "rank_by"),
    ({"demo": True}, "sweep_grid"),
    ({"demo": True, "sweep_grid": {}}, "sweep_grid"),
    ({"demo": True, "sweep_grid": {"bad key!": [1]}}, "param-name format"),
    ({"demo": True, "sweep_grid": {"k": []}}, "non-empty list"),
    ({"demo": True, "sweep_grid": {"k": [True]}}, "bool"),
    ({"demo": True, "sweep_grid": {"k": [float("nan")]}}, "non-finite"),
    ({"demo": True, "sweep_grid": {"k": [float("inf")]}}, "non-finite"),
    ({"demo": True, "sweep_grid": {"k": ["has space"]}}, "transport-safe"),
    ({"demo": True, "sweep_grid": {"k": ["5"]}}, "transport-safe"),            # numeric-looking str
    ({"demo": True, "sweep_grid": {"k": [{"x": 1}]}}, "not int/float/str"),
    ({"demo": True, "sweep_grid": {"k": list(range(20)), "j": list(range(20))}}, "too large"),
])
def test_validate_eval_context_fails_closed(ctx, msg):
    with pytest.raises(ValueError, match=re.escape(msg)):
        mergeback_queue.validate_eval_context(ctx)


def test_max_sweep_combos_mirrors_the_engine_cap():
    # MAX_SWEEP_COMBOS is a deliberate stdlib-only mirror of the sweep engine's _MAX_COMBOS; this
    # cross-check is the drift alarm the duplication relies on.
    from algua.backtest.sweep import _MAX_COMBOS

    assert mergeback_queue.MAX_SWEEP_COMBOS == _MAX_COMBOS


def test_valid_rank_by_mirrors_the_engine_keys():
    from algua.backtest.sweep import _RANK_KEYS

    assert mergeback_queue.VALID_RANK_BY == frozenset(_RANK_KEYS)


def test_enqueue_rejects_invalid_context_without_touching_the_queue(paths):
    queue_path, lock_path = paths
    with pytest.raises(ValueError):
        mergeback_queue.enqueue(
            queue_path, lock_path, **_item(eval_context={"demo": True}))  # no sweep_grid
    assert not queue_path.exists()


def test_cli_select_shell_format_emits_context_transport_vars(paths, capsys):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item(eval_context={
        "snapshot": "bars-1", "delistings": "delist-1",
        "sweep_grid": {"lookback": [10, 20], "threshold": [0.5, 1.5]},
        "rank_by": "min_sharpe",
    }))
    rc = mergeback_queue.main([
        "select-and-reserve", "--queue", str(queue_path), "--lock", str(lock_path),
        "--format", "shell",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MERGEBACK_DEMO=0" in out
    assert "MERGEBACK_SNAPSHOT=bars-1" in out
    assert "MERGEBACK_DELISTINGS=delist-1" in out
    assert "MERGEBACK_RANK_BY=min_sharpe" in out
    # Floats emit via repr so parse_grid round-trips them exactly; tokens are space-joined.
    assert "MERGEBACK_SWEEP_PARAMS='lookback=10,20 threshold=0.5,1.5'" in out


def test_cli_enqueue_takes_eval_context_json(paths, capsys):
    queue_path, lock_path = paths
    rc = mergeback_queue.main([
        "enqueue", "--queue", str(queue_path), "--lock", str(lock_path),
        "--strategy", "strat_a", "--universe", "sp500",
        "--start", "2024-01-01", "--end", "2024-06-01", "--branch", "research-run/1",
        "--eval-context", json.dumps({"demo": True, "sweep_grid": {"k": [1, 2]}}),
    ])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["created"] is True
    assert result["item"]["eval_context"]["sweep_grid"] == {"k": [1, 2]}


# --- compute_run_branch_name (candidate-keyed research-worktree lifecycle, #555) ------------------


def test_compute_run_branch_name_no_candidates_is_stamp_only():
    assert (mergeback_queue.compute_run_branch_name("20260814-120000", [])
            == "research-run/20260814-120000")


def test_compute_run_branch_name_joins_candidate_names_with_plus():
    assert (mergeback_queue.compute_run_branch_name("20260814-120000", ["strat_a", "strat_b"])
            == "research-run/20260814-120000--strat_a+strat_b")


def test_compute_run_branch_name_caps_at_three_names():
    assert (mergeback_queue.compute_run_branch_name(
        "20260814-120000", ["s1", "s2", "s3", "s4"])
        == "research-run/20260814-120000--s1+s2+s3")


def test_compute_run_branch_name_drops_whole_names_never_truncates():
    long_name = "x" * 60
    out = mergeback_queue.compute_run_branch_name(
        "20260814-120000", [long_name, long_name, long_name])
    # Two 60-char names exceed the 120-char total cap -> the LIST is truncated to one name; the
    # surviving name is intact (never itself truncated).
    assert out == f"research-run/20260814-120000--{long_name}"
    assert len(out) <= 120


def test_compute_run_branch_name_single_max_length_name_always_fits():
    name = "y" * 64  # the strategy-name validator's maximum
    out = mergeback_queue.compute_run_branch_name("20260814-120000", [name])
    assert out == f"research-run/20260814-120000--{name}"
    assert len(out) <= 120


# --- cleanup_branch (outcome-keyed worktree reclaim, #555) ----------------------------------------
#
# Exercised against REAL throwaway git repos + worktrees (never a mock of git), matching the
# diff-tree tests in test_research_run_digest.py. `cleanup_branch` is best-effort BY CONTRACT: it
# must never raise (the drainer must not fail on cleanup), so every scenario asserts on the
# returned dict.


_STAMP = "20260101-000000"


def _init_cleanup_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "t@example.com"),
                 ("config", "user.name", "Test")):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True,
                   capture_output=True)
    return repo


def _add_run_worktree(repo: Path, branch: str, worktree: Path, *, log: bool = True) -> None:
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-b", branch, str(worktree)],
                   check=True, capture_output=True)
    if log:
        (worktree / "research-loop.log").write_text("codex transcript\n", encoding="utf-8")


def test_cleanup_branch_all_terminal_removes_worktree_and_archives_log(paths, tmp_path):
    queue_path, lock_path = paths
    repo = _init_cleanup_repo(tmp_path)
    branch = f"research-run/{_STAMP}--strat_a"  # the candidate suffix is stripped for the path
    worktree = repo / ".runs" / _STAMP
    _add_run_worktree(repo, branch, worktree)

    mergeback_queue.enqueue(queue_path, lock_path, **_item(branch=branch))
    mergeback_queue.record_attempt(
        queue_path, lock_path, key=f"strat_a@{branch}",
        stdout_text=json.dumps({"ok": True, "status": "promoted_allocated"}))

    result = mergeback_queue.cleanup_branch(
        queue_path, lock_path, branch=branch, repo_root=repo)
    assert result["removed"] is True
    assert not worktree.exists()
    # Log archived BEFORE removal, branch slashes -> underscores.
    archived = repo / ".runs" / "logs" / f"research-run_{_STAMP}--strat_a.log"
    assert result["log_archived"] is True
    assert archived.read_text(encoding="utf-8") == "codex transcript\n"
    # The branch itself survives the worktree removal (the authored code persists on it).
    branches = subprocess.run(["git", "-C", str(repo), "branch", "--list", branch],
                              capture_output=True, text=True, check=True).stdout
    assert branch in branches


def test_cleanup_branch_noop_while_a_sibling_item_is_non_terminal(paths, tmp_path):
    queue_path, lock_path = paths
    repo = _init_cleanup_repo(tmp_path)
    branch = f"research-run/{_STAMP}--strat_a+strat_b"
    worktree = repo / ".runs" / _STAMP
    _add_run_worktree(repo, branch, worktree)

    for strategy in ("strat_a", "strat_b"):
        mergeback_queue.enqueue(queue_path, lock_path, **_item(strategy=strategy, branch=branch))
    # Only strat_a reaches terminal; strat_b stays pending on the SAME branch.
    mergeback_queue.record_attempt(
        queue_path, lock_path, key=f"strat_a@{branch}",
        stdout_text=json.dumps({"ok": True, "status": "promoted_allocated"}))

    result = mergeback_queue.cleanup_branch(
        queue_path, lock_path, branch=branch, repo_root=repo)
    assert result["removed"] is False
    assert result["skipped"] == "non_terminal_items"
    assert result["non_terminal"] == [f"strat_b@{branch}"]
    assert worktree.is_dir()  # untouched

    # Once strat_b goes terminal too, the same call reclaims the worktree.
    mergeback_queue.record_attempt(
        queue_path, lock_path, key=f"strat_b@{branch}",
        stdout_text=json.dumps({"ok": True, "status": "gate_failed"}), max_attempts=1)
    result = mergeback_queue.cleanup_branch(
        queue_path, lock_path, branch=branch, repo_root=repo)
    assert result["removed"] is True
    assert not worktree.exists()


def test_cleanup_branch_in_progress_item_counts_as_non_terminal(paths, tmp_path):
    queue_path, lock_path = paths
    repo = _init_cleanup_repo(tmp_path)
    branch = f"research-run/{_STAMP}"
    worktree = repo / ".runs" / _STAMP
    _add_run_worktree(repo, branch, worktree)

    mergeback_queue.enqueue(queue_path, lock_path, **_item(branch=branch))
    mergeback_queue.select_and_reserve(queue_path, lock_path)  # -> in_progress

    result = mergeback_queue.cleanup_branch(
        queue_path, lock_path, branch=branch, repo_root=repo)
    assert result["skipped"] == "non_terminal_items"
    assert worktree.is_dir()


def test_cleanup_branch_missing_worktree_is_a_clean_noop(paths, tmp_path):
    queue_path, lock_path = paths
    repo = _init_cleanup_repo(tmp_path)
    branch = f"research-run/{_STAMP}"
    mergeback_queue.enqueue(queue_path, lock_path, **_item(branch=branch))
    mergeback_queue.record_attempt(
        queue_path, lock_path, key=f"strat_a@{branch}",
        stdout_text=json.dumps({"ok": True, "status": "already_done"}))

    result = mergeback_queue.cleanup_branch(
        queue_path, lock_path, branch=branch, repo_root=repo)
    assert result == {
        "branch": branch, "removed": False, "log_archived": False,
        "skipped": "worktree_absent",
    }


def test_cleanup_branch_falls_back_to_the_legacy_location(paths, tmp_path):
    # Transition support: a pre-#555 run's worktree lives at <repo_root>/../algua-research-<stamp>.
    queue_path, lock_path = paths
    repo = _init_cleanup_repo(tmp_path)
    branch = f"research-run/{_STAMP}"
    legacy = tmp_path / f"algua-research-{_STAMP}"
    _add_run_worktree(repo, branch, legacy)

    mergeback_queue.enqueue(queue_path, lock_path, **_item(branch=branch))
    mergeback_queue.record_attempt(
        queue_path, lock_path, key=f"strat_a@{branch}",
        stdout_text=json.dumps({"ok": True, "status": "promoted_queued"}))

    result = mergeback_queue.cleanup_branch(
        queue_path, lock_path, branch=branch, repo_root=repo)
    assert result["removed"] is True
    assert not legacy.exists()
    assert result["log_archived"] is True  # archived into <repo_root>/.runs/logs/ regardless
    assert (repo / ".runs" / "logs" / f"research-run_{_STAMP}.log").is_file()


def test_cleanup_branch_rm_rf_fallback_for_an_unregistered_dir(paths, tmp_path):
    # A dir at the expected path that git does NOT know as a worktree (already pruned from git's
    # list, dir left behind): `git worktree remove` fails, the rm -rf fallback reclaims it.
    queue_path, lock_path = paths
    repo = _init_cleanup_repo(tmp_path)
    branch = f"research-run/{_STAMP}"
    orphan = repo / ".runs" / _STAMP
    orphan.mkdir(parents=True)
    (orphan / "research-loop.log").write_text("orphan transcript\n", encoding="utf-8")

    result = mergeback_queue.cleanup_branch(
        queue_path, lock_path, branch=branch, repo_root=repo)  # empty queue: nothing non-terminal
    assert result["removed"] is True
    assert not orphan.exists()
    assert result["log_archived"] is True


@pytest.mark.parametrize("garbage", [
    "", "junk", "research-run/../../../etc", "research-run/20260101-000000--bad name!",
    "research-run/not-a-stamp", "main",
])
def test_cleanup_branch_never_raises_on_garbage_branches(paths, tmp_path, garbage):
    queue_path, lock_path = paths
    result = mergeback_queue.cleanup_branch(
        queue_path, lock_path, branch=garbage, repo_root=tmp_path)
    assert result["removed"] is False
    assert result["skipped"] == "not_a_research_run_branch"


def test_cleanup_branch_never_raises_on_a_corrupt_queue_file(paths, tmp_path):
    queue_path, lock_path = paths
    queue_path.write_text("this is not json", encoding="utf-8")
    result = mergeback_queue.cleanup_branch(
        queue_path, lock_path, branch=f"research-run/{_STAMP}", repo_root=tmp_path)
    assert result["removed"] is False
    assert "error" in result


def test_cli_cleanup_branch_emits_json(paths, tmp_path, capsys):
    queue_path, lock_path = paths
    rc = mergeback_queue.main([
        "cleanup-branch", "--queue", str(queue_path), "--lock", str(lock_path),
        "--branch", f"research-run/{_STAMP}", "--repo-root", str(tmp_path),
    ])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["branch"] == f"research-run/{_STAMP}"
    assert result["skipped"] == "worktree_absent"
