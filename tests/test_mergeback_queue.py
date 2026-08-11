"""Tests for `.codex/scripts/mergeback_queue.py` (factory slice 3): the durable, dual-writer
merge-back queue (`data/mergeback-queue.json` + `data/mergeback-queue.lock`).

Imported directly via `importlib` from its repo path (it's a standalone script, not part of the
installed `algua` package — see the module's own docstring for why).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
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


def _item(strategy="strat_a", branch="research-run/1", universe="sp500",
          start="2024-01-01", end="2024-06-01"):
    return dict(strategy=strategy, universe=universe, start=start, end=end, branch=branch)


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
    assert verdict == {"action": "terminal", "status": status, "attempts": 1}


@pytest.mark.parametrize("status", ["diff_policy_rejected", "promote_failed"])
def test_classify_hard_failure_terminal(status):
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": status}, attempts=0, max_attempts=3)
    assert verdict == {"action": "terminal", "status": status, "attempts": 1}


def test_classify_gate_failed_retryable_under_cap():
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": "gate_failed"}, attempts=0, max_attempts=3)
    assert verdict == {"action": "retry", "status": "gate_failed", "attempts": 1}


def test_classify_gate_failed_exhausted_at_cap():
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": "gate_failed"}, attempts=2, max_attempts=3)
    assert verdict == {"action": "exhausted", "status": "gate_failed", "attempts": 3}


def test_classify_lock_contention_is_not_an_attempt():
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "ran": False, "reason": "locked"}, attempts=1, max_attempts=3)
    assert verdict == {"action": "lock_contention", "status": None, "attempts": 1}


def test_classify_unparseable_payload_is_transient_not_an_attempt():
    verdict = mergeback_queue.classify_attempt(None, attempts=1, max_attempts=3)
    assert verdict == {"action": "transient_failure", "status": None, "attempts": 1}


def test_classify_ok_false_envelope_is_transient_not_an_attempt():
    # e.g. a fail-closed exception out of `paper merge-back` itself (RemoteMovedError etc.), or a
    # lock-run setup failure (git_dir_unresolved/lock_unavailable) — neither is branch content.
    verdict = mergeback_queue.classify_attempt(
        {"ok": False, "error": "boom", "code": "internal"}, attempts=1, max_attempts=3)
    assert verdict == {"action": "transient_failure", "status": None, "attempts": 1}


def test_classify_unrecognized_status_is_transient_not_an_attempt():
    verdict = mergeback_queue.classify_attempt(
        {"ok": True, "status": "some_future_status"}, attempts=1, max_attempts=3)
    assert verdict == {"action": "transient_failure", "status": None, "attempts": 1}


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


def test_record_attempt_transient_failure_leaves_item_untouched(paths):
    queue_path, lock_path = paths
    mergeback_queue.enqueue(queue_path, lock_path, **_item())
    before = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]

    result = mergeback_queue.record_attempt(
        queue_path, lock_path, key="strat_a@research-run/1", stdout_text="not json at all",
    )
    assert result["action"] == "transient_failure"
    after = json.loads(queue_path.read_text())["items"]["strat_a@research-run/1"]
    assert after == before
    assert after["attempts"] == 0


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
