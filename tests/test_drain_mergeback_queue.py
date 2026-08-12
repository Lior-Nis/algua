"""Tests for `.codex/scripts/drain-mergeback-queue.sh` (factory slice 3).

Runs the REAL bash script against an isolated queue file, with a tiny STUB standing in for the
`algua` binary (`ALGUA_BIN` env override) so no real git/quality-gate/broker machinery runs. The
stub simulates `algua operator lock-run -- algua paper merge-back ...`'s transparent passthrough:
it prints one canned JSON line and exits 0, exactly like the real `lock-run` does on both the
ran-the-command and the benign-lock-contention paths (see `algua/cli/operator_cmd.py`).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAINER = REPO_ROOT / ".codex" / "scripts" / "drain-mergeback-queue.sh"
_QUEUE_MOD_PATH = REPO_ROOT / ".codex" / "scripts" / "mergeback_queue.py"

_spec = importlib.util.spec_from_file_location("mergeback_queue", _QUEUE_MOD_PATH)
mergeback_queue = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mergeback_queue)


def _make_stub(
    tmp_path: Path, *, response: dict, exit_code: int = 0, marker: Path | None = None,
) -> Path:
    """A fake `algua` binary: on ANY invocation (whether as the outer `lock-run` wrapper or,
    since `ALGUA_BIN` substitutes both occurrences in the drainer's argv, the inner wrapped
    command) it just prints `response` and exits `exit_code` — exactly what a real `lock-run`
    prints on its two observable shapes (transparent passthrough of the wrapped command's JSON,
    or its own benign `{"reason": "locked"}` no-op envelope)."""
    stub = tmp_path / "algua-stub.sh"
    lines = ["#!/usr/bin/env bash"]
    if marker is not None:
        lines.append(f"touch {shlex.quote(str(marker))}")
    lines.append(f"printf '%s' {shlex.quote(json.dumps(response))}")
    lines.append(f"exit {exit_code}")
    stub.write_text("\n".join(lines) + "\n")
    stub.chmod(0o755)
    return stub


def _run_drainer(
    tmp_path: Path, *, algua_bin: Path | None = None, dry_run: bool = False,
    max_attempts: int = 3, backoff_minutes: float = 0.0,
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["ALGUA_MERGEBACK_QUEUE_PATH"] = str(tmp_path / "queue.json")
    env["ALGUA_MERGEBACK_QUEUE_LOCK_PATH"] = str(tmp_path / "queue.lock")
    env["MAX_MERGEBACK_ATTEMPTS"] = str(max_attempts)
    env["MERGEBACK_BACKOFF_MINUTES_PER_ATTEMPT"] = str(backoff_minutes)
    if algua_bin is not None:
        env["ALGUA_BIN"] = str(algua_bin)
    args = ["bash", str(DRAINER)]
    if dry_run:
        args.append("--dry-run")
    return subprocess.run(args, capture_output=True, text=True, env=env, timeout=30, cwd=REPO_ROOT)


def _queue_path(tmp_path: Path) -> Path:
    return tmp_path / "queue.json"


def _lock_path(tmp_path: Path) -> Path:
    return tmp_path / "queue.lock"


def _seed(tmp_path: Path, **kwargs) -> None:
    # Every queue item carries a validated eval_context recipe (mergeback authoritative intake);
    # tests that don't care about its content get a minimal valid one.
    kwargs.setdefault("eval_context", {
        "demo": True, "sweep_grid": {"lookback": [10, 20]}, "rank_by": "mean_sharpe"})
    mergeback_queue.enqueue(_queue_path(tmp_path), _lock_path(tmp_path), **kwargs)


def _items(tmp_path: Path) -> dict:
    return json.loads(_queue_path(tmp_path).read_text())["items"]


# --- no queue file yet: benign no-op --------------------------------------------------------------


def test_no_queue_file_is_a_benign_noop(tmp_path):
    proc = _run_drainer(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "nothing to drain" in proc.stdout


# --- exactly one item drained per invocation, even with multiple pending ------------------------


def test_exactly_one_item_drained_per_invocation(tmp_path):
    for i in range(4):
        _seed(tmp_path, strategy=f"strat_{i}", universe="sp500", start="2024-01-01",
              end="2024-06-01", branch="research-run/1")
    stub = _make_stub(tmp_path, response={"ok": True, "status": "promoted_allocated"})

    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr

    items = _items(tmp_path)
    touched = [k for k, v in items.items() if v["status"] != "pending"]
    assert len(touched) == 1
    # FIFO: the first-enqueued item is the one drained.
    assert touched == ["strat_0@research-run/1"]


# --- terminal success / terminal failure classification ------------------------------------------


@pytest.mark.parametrize("status", ["promoted_allocated", "promoted_queued", "already_done"])
def test_terminal_success_statuses_recorded(tmp_path, status):
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    stub = _make_stub(tmp_path, response={"ok": True, "status": status})
    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr
    item = _items(tmp_path)["s@research-run/1"]
    assert item["status"] == status
    assert item["attempts"] == 1


@pytest.mark.parametrize("status", ["diff_policy_rejected", "promote_failed"])
def test_terminal_failure_statuses_never_retried(tmp_path, status):
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    stub = _make_stub(tmp_path, response={"ok": True, "status": status})
    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr
    item = _items(tmp_path)["s@research-run/1"]
    assert item["status"] == "terminal_failed"
    assert item["attempts"] == 1

    # A second drain cycle must find NOTHING eligible (terminal, never retried).
    proc2 = _run_drainer(tmp_path, algua_bin=stub)
    assert "nothing to drain" in proc2.stdout
    assert _items(tmp_path)["s@research-run/1"]["attempts"] == 1


# --- gate_failed: retry with a cap, flat cap of 3 (no phase info in the merge-back JSON) --------


def test_gate_failed_retries_then_hits_the_flat_cap(tmp_path):
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    stub = _make_stub(tmp_path, response={"ok": True, "status": "gate_failed"})

    for expected_attempts in (1, 2, 3):
        proc = _run_drainer(tmp_path, algua_bin=stub, max_attempts=3, backoff_minutes=0.0)
        assert proc.returncode == 0, proc.stderr
        item = _items(tmp_path)["s@research-run/1"]
        assert item["attempts"] == expected_attempts
        if expected_attempts < 3:
            assert item["status"] == "gate_failed"
        else:
            assert item["status"] == "terminal_failed"

    # Exhausted: a further cycle finds nothing eligible.
    proc_final = _run_drainer(tmp_path, algua_bin=stub, max_attempts=3, backoff_minutes=0.0)
    assert "nothing to drain" in proc_final.stdout


# --- lock contention: benign, NEVER counted as an attempt -----------------------------------------


def test_lock_contention_does_not_increment_attempts(tmp_path):
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    stub = _make_stub(tmp_path, response={"ok": True, "ran": False, "reason": "locked"})

    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr
    item = _items(tmp_path)["s@research-run/1"]
    assert item["status"] == "pending"
    assert item["attempts"] == 0
    assert item["last_attempt_at"] is None


# --- unparseable / hard-failure output: leave pending, do NOT increment attempts -----------------


def test_non_json_stub_output_leaves_item_pending(tmp_path):
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    stub = tmp_path / "algua-stub.sh"
    stub.write_text("#!/usr/bin/env bash\nprintf 'not json at all'\nexit 0\n")
    stub.chmod(0o755)

    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr
    item = _items(tmp_path)["s@research-run/1"]
    assert item["status"] == "pending"
    assert item["attempts"] == 0


# --- dry-run: never invokes the stub at all -------------------------------------------------------


def test_dry_run_never_invokes_algua(tmp_path):
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    marker = tmp_path / "invoked.marker"
    stub = _make_stub(
        tmp_path, response={"ok": True, "status": "promoted_allocated"}, marker=marker)

    proc = _run_drainer(tmp_path, algua_bin=stub, dry_run=True)
    assert proc.returncode == 0, proc.stderr
    assert not marker.exists()
    assert _items(tmp_path)["s@research-run/1"]["status"] == "pending"


# --- reservation (TOCTOU fix): the drainer reserves via select-and-reserve, never a bare select ---


def test_dry_run_is_read_only_never_reserves(tmp_path):
    # `select` (read-only), never `select-and-reserve`, must be used on the dry-run path — a dry
    # run must never flip a real queue item to `in_progress`.
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    stub = _make_stub(tmp_path, response={"ok": True, "status": "promoted_allocated"})

    proc = _run_drainer(tmp_path, algua_bin=stub, dry_run=True)
    assert proc.returncode == 0, proc.stderr
    item = _items(tmp_path)["s@research-run/1"]
    assert item["status"] == "pending"
    assert "reserved_at" not in item


def test_real_drain_reserves_item_in_progress_before_invoking_algua(tmp_path):
    # The stub, while running, cannot observe the reservation directly (it's a single subprocess
    # call), but a lock_contention/transient outcome exercises the release path, and the terminal
    # path proves the item passed THROUGH `in_progress` — asserted indirectly via the item's final
    # `pre_reservation_status` bookkeeping being cleared and status landing on the real outcome.
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    stub = _make_stub(tmp_path, response={"ok": True, "status": "promoted_allocated"})

    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr
    assert "selecting + reserving" in proc.stdout
    item = _items(tmp_path)["s@research-run/1"]
    assert item["status"] == "promoted_allocated"
    assert "pre_reservation_status" not in item
    assert "reserved_at" not in item


def test_lock_contention_releases_reservation_back_to_pending(tmp_path):
    # A benign lock-contention outcome must release the reservation immediately (back to
    # "pending"), not leave the item wedged `in_progress` until RESERVATION_STALE_SECONDS elapses.
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    stub = _make_stub(tmp_path, response={"ok": True, "ran": False, "reason": "locked"})

    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr
    item = _items(tmp_path)["s@research-run/1"]
    assert item["status"] == "pending"
    assert item["attempts"] == 0
    assert "reserved_at" not in item

    # A SECOND drain cycle immediately (no stale wait) still finds and drains the same item.
    proc2 = _run_drainer(tmp_path, algua_bin=stub)
    assert proc2.returncode == 0, proc2.stderr
    assert "selected s@research-run/1" in proc2.stdout


def test_overlapping_drainer_cannot_double_act_on_same_item(tmp_path):
    # Simulates the TOCTOU this fix closes: a second drainer invocation firing while an item is
    # ALREADY reserved (e.g. a prior invocation still mid-attempt) must find nothing eligible,
    # rather than re-selecting and re-running the same item.
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")
    # Reserve the item directly (as if a first drainer invocation is mid-flight).
    mergeback_queue.select_and_reserve(_queue_path(tmp_path), _lock_path(tmp_path))
    assert _items(tmp_path)["s@research-run/1"]["status"] == "in_progress"

    marker = tmp_path / "invoked.marker"
    stub = _make_stub(
        tmp_path, response={"ok": True, "status": "promoted_allocated"}, marker=marker)
    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr
    assert "nothing to drain" in proc.stdout
    assert not marker.exists()  # the second invocation never invoked algua at all
    assert _items(tmp_path)["s@research-run/1"]["status"] == "in_progress"  # untouched


# --- eval_context argv transport (mergeback authoritative intake) ---------------------------------


def test_eval_context_rides_the_invocation_argv(tmp_path):
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1", eval_context={
              "snapshot": "bars-9", "delistings": "delist-3",
              "sweep_grid": {"lookback": [10, 20], "threshold": [0.5, 1.5]},
              "rank_by": "min_sharpe",
          })
    argv_log = tmp_path / "argv.log"
    stub = tmp_path / "algua-stub.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$@\" >> {shlex.quote(str(argv_log))}\n"
        "printf '%s' '{\"ok\": true, \"status\": \"promoted_allocated\"}'\n",
        encoding="utf-8")
    stub.chmod(0o755)

    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr

    args = argv_log.read_text(encoding="utf-8").splitlines()
    # The transported recipe reaches the wrapped `paper merge-back` as discrete argv tokens.
    assert "--snapshot" in args and args[args.index("--snapshot") + 1] == "bars-9"
    assert "--delistings" in args and args[args.index("--delistings") + 1] == "delist-3"
    assert "--rank-by" in args and args[args.index("--rank-by") + 1] == "min_sharpe"
    sweep_values = [args[i + 1] for i, a in enumerate(args) if a == "--sweep-param"]
    assert sweep_values == ["lookback=10,20", "threshold=0.5,1.5"]
    assert "--demo" not in args
    item = _items(tmp_path)["s@research-run/1"]
    assert item["status"] == "promoted_allocated"


def test_demo_context_emits_the_demo_flag(tmp_path):
    _seed(tmp_path, strategy="s", universe="sp500", start="2024-01-01", end="2024-06-01",
          branch="research-run/1")  # _seed's default context is demo: true
    argv_log = tmp_path / "argv.log"
    stub = tmp_path / "algua-stub.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$@\" >> {shlex.quote(str(argv_log))}\n"
        "printf '%s' '{\"ok\": true, \"status\": \"promoted_allocated\"}'\n",
        encoding="utf-8")
    stub.chmod(0o755)

    proc = _run_drainer(tmp_path, algua_bin=stub)
    assert proc.returncode == 0, proc.stderr
    args = argv_log.read_text(encoding="utf-8").splitlines()
    assert "--demo" in args
    assert "--snapshot" not in args
    assert "--sweep-param" in args
