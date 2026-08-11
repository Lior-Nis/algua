"""Tests for the run-research-loop.sh digest/trailer parser (factory slice 3, #536 follow-on).

The trailer parser and the anti-dup reader are Python, EMBEDDED in the bash launcher as heredocs
(the existing pattern this file already used pre-slice-3) rather than a standalone module — so
these tests extract each heredoc's source text directly from the .sh file and run it exactly as
the launcher does (``python3 - <args> <<'PY' ... PY``), via a real subprocess with the same
positional-argv contract. This exercises the ACTUAL code the launcher runs, not a reimplementation
of it, while still being a fast, isolated, no-codex unit test.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / ".codex" / "scripts" / "run-research-loop.sh"
_HEREDOC_RE = re.compile(r"<<'PY'[^\n]*\n(.*?)\n^PY$", re.DOTALL | re.MULTILINE)


def _heredocs() -> list[str]:
    return _HEREDOC_RE.findall(LAUNCHER.read_text(encoding="utf-8"))


# Heredoc 0 = append_digest's trailer parser + enqueue; heredoc 1 = the anti-dup reader;
# heredoc 2 = the sqlite consistent-backup snippet (untouched by this slice, not tested here).
_APPEND_DIGEST_SRC = _heredocs()[0]
_ANTI_DUP_SRC = _heredocs()[1]


def _fence(payload: dict) -> str:
    return "some run report prose\n\n```json\n" + json.dumps(payload) + "\n```\n"


def _run_append_digest(
    tmp_path: Path,
    *,
    trailer: dict | None,
    strategy_names: str = "",
    branch: str = "research-run/20260811-000000",
    stamp: str = "20260811-000000",
    outcome: str = "completed",
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    digest_path = tmp_path / "digest.jsonl"
    report_path = tmp_path / "report.md"
    queue_path = tmp_path / "queue.json"
    queue_lock_path = tmp_path / "queue.lock"
    if trailer is not None:
        report_path.write_text(_fence(trailer), encoding="utf-8")
    else:
        report_path.write_text("no trailer here at all\n", encoding="utf-8")
    argv_tail = [
        str(digest_path), outcome, stamp, branch, "a thesis",
        "0", "0", "12", "1", "0",
        str(report_path), strategy_names, str(REPO_ROOT),
        str(queue_path), str(queue_lock_path),
    ]
    proc = subprocess.run(
        [sys.executable, "-", *argv_tail],
        input=_APPEND_DIGEST_SRC, capture_output=True, text=True, timeout=30,
    )
    return proc, digest_path, queue_path


def _last_digest_row(digest_path: Path) -> dict:
    lines = digest_path.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def _queue_items(queue_path: Path) -> dict:
    if not queue_path.exists():
        return {}
    return json.loads(queue_path.read_text(encoding="utf-8"))["items"]


# --- (a) basic parse: verdict + validated merge_back on a matching strategy ---------------------


def test_parses_verdict_and_validated_merge_back(tmp_path):
    trailer = {
        "hypotheses": [{
            "title": "Momentum thing", "verdict": "candidate-preview-pass",
            "merge_back": {"strategy": "strat_a", "universe": "sp500",
                           "start": "2024-01-01", "end": "2024-06-01"},
        }],
        "preview_gate": {"passed": True, "failed_checks": []},
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr

    row = _last_digest_row(digest_path)
    assert row["hypotheses"] == [{
        "title": "Momentum thing", "verdict": "candidate-preview-pass",
        "merge_back": {"strategy": "strat_a", "universe": "sp500",
                       "start": "2024-01-01", "end": "2024-06-01"},
    }]
    assert row["preview_gate"] == {"passed": True, "failed_checks": []}
    assert row["trailer_parse_error"] is False
    assert row["report"] == "research-run/20260811-000000:kb/research-runs/20260811-000000.md"

    items = _queue_items(queue_path)
    assert set(items) == {"strat_a@research-run/20260811-000000"}
    item = items["strat_a@research-run/20260811-000000"]
    assert item["strategy"] == "strat_a"
    assert item["universe"] == "sp500"
    assert item["start"] == "2024-01-01"
    assert item["end"] == "2024-06-01"
    assert item["branch"] == "research-run/20260811-000000"
    assert item["status"] == "pending"
    assert item["attempts"] == 0
    assert "merge-back queue:" in proc.stdout


# --- (b) strategy cross-check: reject an unrelated strategy name --------------------------------


def test_merge_back_dropped_when_strategy_not_in_run_commit(tmp_path):
    trailer = {
        "hypotheses": [{
            "title": "Sneaky", "verdict": "candidate-preview-pass",
            "merge_back": {"strategy": "not_my_strategy", "universe": "sp500",
                           "start": "2024-01-01", "end": "2024-06-01"},
        }],
        "preview_gate": None,
    }
    # This run's own commit only added `strat_a` — `not_my_strategy` must be rejected.
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr

    row = _last_digest_row(digest_path)
    assert row["hypotheses"] == [{"title": "Sneaky", "verdict": "candidate-preview-pass",
                                   "merge_back": None}]
    assert not queue_path.exists()
    assert "not among this run's own committed" in proc.stdout


# --- (c) same-strategy-twice dedup: first wins ---------------------------------------------------


def test_duplicate_strategy_in_one_run_keeps_first(tmp_path):
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01", "end": "2024-06-01"}
    mb2 = {"strategy": "strat_a", "universe": "nasdaq100", "start": "2024-02-01",
           "end": "2024-07-01"}
    trailer = {
        "hypotheses": [
            {"title": "First", "verdict": "candidate-preview-pass", "merge_back": mb},
            {"title": "Second", "verdict": "candidate-preview-pass", "merge_back": mb2},
        ],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr

    row = _last_digest_row(digest_path)
    assert row["hypotheses"][0]["merge_back"] == mb
    assert row["hypotheses"][1]["merge_back"] is None  # second dropped, first wins
    items = _queue_items(queue_path)
    assert len(items) == 1
    assert items["strat_a@research-run/20260811-000000"]["universe"] == "sp500"
    assert "duplicate merge_back.strategy" in proc.stdout


# --- (d) format-validation rejections ------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("strategy", "not-a-valid-name!"),
        ("universe", "bad universe with spaces"),
        ("start", "01-01-2024"),
        ("end", "2024/06/01"),
    ],
)
def test_format_rejections_drop_candidacy_not_the_run(tmp_path, bad_field, bad_value):
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01", "end": "2024-06-01"}
    mb[bad_field] = bad_value
    trailer = {
        "hypotheses": [{"title": "Bad", "verdict": "candidate-preview-pass", "merge_back": mb}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"][0]["merge_back"] is None
    assert not queue_path.exists()


def test_start_after_end_rejected(tmp_path):
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-06-01", "end": "2024-01-01"}
    trailer = {
        "hypotheses": [
            {"title": "Backwards", "verdict": "candidate-preview-pass", "merge_back": mb}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"][0]["merge_back"] is None
    assert not queue_path.exists()


def test_end_after_today_rejected(tmp_path):
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01", "end": "2099-01-01"}
    trailer = {
        "hypotheses": [{"title": "Future", "verdict": "candidate-preview-pass", "merge_back": mb}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"][0]["merge_back"] is None
    assert not queue_path.exists()


# --- (e) silent-drop-not-abort: one bad + one good hypothesis in the same run --------------------


def test_one_bad_one_good_hypothesis_still_completes_and_enqueues_the_good_one(tmp_path):
    good = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01", "end": "2024-06-01"}
    bad = {"strategy": "strat_b", "universe": "bad universe", "start": "2024-01-01",
           "end": "2024-06-01"}
    trailer = {
        "hypotheses": [
            {"title": "Good one", "verdict": "candidate-preview-pass", "merge_back": good},
            {"title": "Bad one", "verdict": "candidate-preview-pass", "merge_back": bad},
        ],
        "preview_gate": {"passed": True, "failed_checks": []},
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a,strat_b")
    assert proc.returncode == 0, proc.stderr

    row = _last_digest_row(digest_path)
    assert row["trailer_parse_error"] is False
    assert len(row["hypotheses"]) == 2
    assert row["hypotheses"][0]["merge_back"] == good
    assert row["hypotheses"][1]["merge_back"] is None

    items = _queue_items(queue_path)
    assert set(items) == {"strat_a@research-run/20260811-000000"}


# --- (f) discarded/error verdicts never read merge_back at all -----------------------------------


def test_merge_back_ignored_for_non_pass_verdicts(tmp_path):
    trailer = {
        "hypotheses": [
            {"title": "Discarded", "verdict": "discarded",
             "merge_back": {"strategy": "strat_a", "universe": "sp500",
                             "start": "2024-01-01", "end": "2024-06-01"}},
            {"title": "Errored", "verdict": "error", "merge_back": None},
        ],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"][0]["merge_back"] is None
    assert row["hypotheses"][1]["merge_back"] is None
    assert not queue_path.exists()


# --- (g) invalid verdict enum invalidates the WHOLE trailer (strict, like `title`) ---------------


def test_invalid_verdict_value_invalidates_whole_trailer(tmp_path):
    trailer = {
        "hypotheses": [{"title": "Weird", "verdict": "sort-of-passed", "merge_back": None}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"] == []
    assert row["preview_gate"] is None
    assert row["trailer_parse_error"] is True
    assert not queue_path.exists()


# --- (h) legacy bare-string hypothesis entries still parse (verdict/merge_back absent) -----------


def test_bare_string_hypothesis_entries_still_parse(tmp_path):
    trailer = {"hypotheses": ["an old-style bare string hypothesis"], "preview_gate": None}
    proc, digest_path, _queue_path = _run_append_digest(tmp_path, trailer=trailer)
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"] == [
        {"title": "an old-style bare string hypothesis", "verdict": None, "merge_back": None}
    ]
    assert row["trailer_parse_error"] is False


# --- (i) missing verdict key (mid-rollout tolerance) is lenient, not an error ---------------------


def test_missing_verdict_key_is_lenient_not_an_error(tmp_path):
    trailer = {"hypotheses": [{"title": "No verdict field"}], "preview_gate": None}
    proc, digest_path, _queue_path = _run_append_digest(tmp_path, trailer=trailer)
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"] == [{"title": "No verdict field", "verdict": None, "merge_back": None}]
    assert row["trailer_parse_error"] is False


# --- (j) no trailer at all: unaffected pre-existing behavior --------------------------------------


def test_no_trailer_present_yields_empty_hypotheses_and_parse_error(tmp_path):
    proc, digest_path, queue_path = _run_append_digest(tmp_path, trailer=None)
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"] == []
    assert row["preview_gate"] is None
    assert row["trailer_parse_error"] is True
    assert not queue_path.exists()


# --- (k) report path reflects the kb/research-runs/<stamp>.md migration --------------------------


def test_report_field_uses_kb_research_runs_path(tmp_path):
    proc, digest_path, _queue_path = _run_append_digest(
        tmp_path, trailer={"hypotheses": [], "preview_gate": None},
        branch="research-run/20301231-235959", stamp="20301231-235959")
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["report"] == "research-run/20301231-235959:kb/research-runs/20301231-235959.md"
    assert "run-report.md" not in row["report"]


# --- (l) a non-completed outcome never touches the trailer/queue at all ---------------------------


def test_setup_failed_outcome_skips_trailer_and_queue(tmp_path):
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer={"hypotheses": [{"title": "x", "verdict": "discarded"}]},
        outcome="setup_failed", branch="")
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"] == []
    assert row["preview_gate"] is None
    assert row["trailer_parse_error"] is None
    assert row["report"] is None
    assert not queue_path.exists()


# --- anti-dup reader: backward-compat over BOTH bare-string and object-shaped digest lines --------


def _run_anti_dup(digest_path: Path) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-", str(digest_path)],
        input=_ANTI_DUP_SRC, capture_output=True, text=True, timeout=30, check=True,
    )
    return json.loads(proc.stdout)


def test_anti_dup_reads_both_old_bare_string_and_new_object_shaped_lines(tmp_path):
    digest_path = tmp_path / "digest.jsonl"
    old_row = {"hypotheses": ["an old bare-string title"], "preview_gate": None}
    new_row = {
        "hypotheses": [{"title": "a new object-shaped title", "verdict": "discarded",
                         "merge_back": None}],
        "preview_gate": None,
    }
    digest_path.write_text(
        json.dumps(old_row) + "\n" + json.dumps(new_row) + "\n", encoding="utf-8")

    titles = _run_anti_dup(digest_path)

    assert "an old bare-string title" in titles
    assert "a new object-shaped title" in titles
