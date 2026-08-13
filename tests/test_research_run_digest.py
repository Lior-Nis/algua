"""Tests for the run-research-loop.sh digest/trailer parser (factory slice 3, #536 follow-on).

The trailer parser and the anti-dup reader are Python, EMBEDDED in the bash launcher as heredocs
(the existing pattern this file already used pre-slice-3) rather than a standalone module — so
these tests extract each heredoc's source text directly from the .sh file and run it exactly as
the launcher does (``python3 - <args> <<'PY' ... PY``), via a real subprocess with the same
positional-argv contract. This exercises the ACTUAL code the launcher runs, not a reimplementation
of it, while still being a fast, isolated, no-codex unit test.
"""

from __future__ import annotations

import fcntl
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


# Heredoc 0 = append_digest's trailer parser + branch rename + enqueue; heredoc 1 = the anti-dup
# reader; heredoc 2 = the sqlite consistent-backup snippet (not tested here); heredoc 3 = the
# end-of-run enqueued-candidate counter (runs-worktree lifecycle, #555).
_APPEND_DIGEST_SRC = _heredocs()[0]
_ANTI_DUP_SRC = _heredocs()[1]
_COUNT_ENQUEUED_SRC = _heredocs()[3]


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
    git_root: Path | str = "",
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Run the append_digest heredoc with the launcher's exact positional-argv contract.

    ``git_root`` is the trailing argv: where the run branch lives for the candidate-keyed rename.
    Tests default it to "" (rename disabled) — NEVER the real repo, whose shared ref store could
    hold a genuine research-run branch colliding with a test stamp; rename tests pass an isolated
    throwaway repo instead."""
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
        str(queue_path), str(queue_lock_path), str(git_root),
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



def _ec(**over):
    """A minimal VALID eval_context recipe (mergeback authoritative intake) for trailer tests."""
    ec = {"demo": True, "sweep_grid": {"lookback": [10, 20]}, "rank_by": "mean_sharpe"}
    ec.update(over)
    return ec


# --- (a) basic parse: verdict + validated merge_back on a matching strategy ---------------------


def test_parses_verdict_and_validated_merge_back(tmp_path):
    trailer = {
        "hypotheses": [{
            "title": "Momentum thing", "verdict": "candidate-preview-pass",
            "merge_back": {"strategy": "strat_a", "universe": "sp500",
                           "start": "2024-01-01", "end": "2024-06-01",
                           "eval_context": _ec()},
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
                       "start": "2024-01-01", "end": "2024-06-01",
                       "eval_context": _ec()},
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
    # The canonicalized recipe rides the queue item (validated fail-closed at enqueue).
    assert item["eval_context"] == {"demo": True, "rank_by": "mean_sharpe",
                                    "sweep_grid": {"lookback": [10, 20]}}
    assert "merge-back queue:" in proc.stdout


# --- (b) strategy cross-check: reject an unrelated strategy name --------------------------------


def test_merge_back_dropped_when_strategy_not_in_run_commit(tmp_path):
    trailer = {
        "hypotheses": [{
            "title": "Sneaky", "verdict": "candidate-preview-pass",
            "merge_back": {"strategy": "not_my_strategy", "universe": "sp500",
                           "start": "2024-01-01", "end": "2024-06-01",
                           "eval_context": _ec()},
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
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01",
          "end": "2024-06-01", "eval_context": _ec()}
    mb2 = {"strategy": "strat_a", "universe": "nasdaq100", "start": "2024-02-01",
           "end": "2024-07-01", "eval_context": _ec()}
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
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01",
          "end": "2024-06-01", "eval_context": _ec()}
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
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-06-01",
          "end": "2024-01-01", "eval_context": _ec()}
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
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01",
          "end": "2099-01-01", "eval_context": _ec()}
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
    good = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01",
            "end": "2024-06-01", "eval_context": _ec()}
    bad = {"strategy": "strat_b", "universe": "bad universe", "start": "2024-01-01",
           "end": "2024-06-01", "eval_context": _ec()}
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


# --- (m) ADDED-only strategy cross-check: a real git commit, not a reimplementation --------------
#
# `run-research-loop.sh` computes the cross-check set (STRATEGY_MODULE_NAMES) from
# `git diff-tree --no-commit-id --name-only --diff-filter=A -r HEAD -- algua/strategies` — the
# EXACT command copied below, run against a REAL throwaway git repo/commit, never a
# reimplementation of git's diff semantics. A file this run's commit only MODIFIES (a pre-existing
# strategy module) must NOT satisfy the cross-check — that's exactly the "agent nominates an
# unrelated pre-existing strategy" case the check exists to catch, just disguised as a diff instead
# of an untouched file.


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    return repo


def _compute_added_strategy_module_names(repo_dir: Path) -> str:
    """The EXACT `git diff-tree` invocation run-research-loop.sh uses (see its "ADDED-only" comment
    block, near the STRATEGY_FILE_LIST assignment) to build the cross-check set for HEAD — executed
    here against a real git repo/commit, not reimplemented."""
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "diff-tree", "--no-commit-id", "--name-only",
         "--diff-filter=A", "-r", "HEAD", "--", "algua/strategies"],
        capture_output=True, text=True, check=True,
    )
    files = [line for line in result.stdout.splitlines() if line.strip()]
    return ",".join(Path(f).stem for f in files)


def test_git_diff_filter_a_accepts_a_truly_added_strategy_file(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "seed", cwd=repo)

    strat_dir = repo / "algua" / "strategies" / "family"
    strat_dir.mkdir(parents=True)
    (strat_dir / "new_strat.py").write_text("CONFIG = {}\n", encoding="utf-8")
    _git("add", "algua/strategies", cwd=repo)
    _git("commit", "-q", "-m", "research-run: author new_strat", cwd=repo)

    strategy_names = _compute_added_strategy_module_names(repo)
    assert strategy_names == "new_strat"

    trailer = {
        "hypotheses": [{
            "title": "Added strategy", "verdict": "candidate-preview-pass",
            "merge_back": {"strategy": "new_strat", "universe": "sp500",
                           "start": "2024-01-01", "end": "2024-06-01",
                           "eval_context": _ec()},
        }],
        "preview_gate": {"passed": True, "failed_checks": []},
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names=strategy_names)
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"][0]["merge_back"] == {
        "strategy": "new_strat", "universe": "sp500", "start": "2024-01-01", "end": "2024-06-01",
        "eval_context": _ec(),
    }
    items = _queue_items(queue_path)
    assert set(items) == {"new_strat@research-run/20260811-000000"}


def test_git_diff_filter_a_rejects_a_merely_modified_pre_existing_strategy_file(tmp_path):
    repo = _init_repo(tmp_path)
    strat_dir = repo / "algua" / "strategies" / "family"
    strat_dir.mkdir(parents=True)
    (strat_dir / "existing_strat.py").write_text("CONFIG = {}\n", encoding="utf-8")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "seed existing_strat", cwd=repo)

    # THIS run's commit only MODIFIES the pre-existing file — never adds anything.
    (strat_dir / "existing_strat.py").write_text("CONFIG = {}\nEXTRA = 1\n", encoding="utf-8")
    _git("add", "algua/strategies", cwd=repo)
    _git("commit", "-q", "-m", "research-run: tweak existing_strat", cwd=repo)

    strategy_names = _compute_added_strategy_module_names(repo)
    assert strategy_names == ""  # nothing ADDED by this commit

    trailer = {
        "hypotheses": [{
            "title": "Nominates a merely-modified pre-existing strategy",
            "verdict": "candidate-preview-pass",
            "merge_back": {"strategy": "existing_strat", "universe": "sp500",
                           "start": "2024-01-01", "end": "2024-06-01"},
        }],
        "preview_gate": {"passed": True, "failed_checks": []},
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names=strategy_names)
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"][0]["merge_back"] is None  # dropped silently, run still completes
    assert not queue_path.exists()
    assert "not among this run's own committed" in proc.stdout


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


# --- (k) eval_context (mergeback authoritative intake) producer-side rules -----------------------


def test_merge_back_without_eval_context_is_dropped(tmp_path):
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01", "end": "2024-06-01"}
    trailer = {
        "hypotheses": [{"title": "No recipe", "verdict": "candidate-preview-pass",
                        "merge_back": mb}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"][0]["merge_back"] is None
    assert not queue_path.exists()
    assert "eval_context missing" in proc.stdout


@pytest.mark.parametrize("attest", [{"windows": 6}, {"holdout_frac": 0.3},
                                    {"windows": 3, "holdout_frac": 0.2}])
def test_non_default_preview_partition_is_rejected(tmp_path, attest):
    # The producer REFUSES a candidate whose scratch preview deviated from the strict-agent
    # defaults (windows=4, holdout_frac=0.2) — the authoritative run must evaluate the same
    # partition the preview claimed.
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01",
          "end": "2024-06-01", "eval_context": _ec(**attest)}
    trailer = {
        "hypotheses": [{"title": "Repartitioned", "verdict": "candidate-preview-pass",
                        "merge_back": mb}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    row = _last_digest_row(digest_path)
    assert row["hypotheses"][0]["merge_back"] is None
    assert not queue_path.exists()
    assert "strict-agent defaults" in proc.stdout


def test_default_partition_attestation_is_accepted_and_stripped(tmp_path):
    # Declaring the exact defaults is accepted — and the attestation keys are STRIPPED before
    # enqueue (windows/holdout_frac are never transported; the drainer pins the defaults).
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01",
          "end": "2024-06-01", "eval_context": _ec(windows=4, holdout_frac=0.2)}
    trailer = {
        "hypotheses": [{"title": "Defaults", "verdict": "candidate-preview-pass",
                        "merge_back": mb}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    items = _queue_items(queue_path)
    item = items["strat_a@research-run/20260811-000000"]
    assert "windows" not in item["eval_context"]
    assert "holdout_frac" not in item["eval_context"]
    assert item["eval_context"]["sweep_grid"] == {"lookback": [10, 20]}


def test_invalid_eval_context_fails_closed_at_enqueue(tmp_path):
    # A shape-invalid recipe passes the producer's presence check but is rejected FAIL-CLOSED by
    # mergeback_queue.validate_eval_context inside enqueue — loud warning, no queue item.
    mb = {"strategy": "strat_a", "universe": "sp500", "start": "2024-01-01",
          "end": "2024-06-01",
          "eval_context": {"demo": True, "sweep_grid": {}, "rank_by": "mean_sharpe"}}
    trailer = {
        "hypotheses": [{"title": "Empty grid", "verdict": "candidate-preview-pass",
                        "merge_back": mb}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    assert not queue_path.exists()
    assert "failed to enqueue merge-back candidate" in proc.stdout


# --- candidate-keyed branch rename (runs-worktree lifecycle, #555) --------------------------------
#
# The append_digest heredoc renames research-run/<stamp> -> research-run/<stamp>--<candidates>
# BEFORE the digest row lands and the candidates enqueue, so both always record the FINAL branch.
# Exercised against an isolated throwaway git repo passed as the heredoc's git_root argv (see
# _run_append_digest: the default "" disables the rename so no other test can touch a real branch).


def _rename_repo(tmp_path: Path, *, with_branch: bool = True,
                 stamp: str = "20260811-000000") -> Path:
    repo = _init_repo(tmp_path)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "seed", cwd=repo)
    if with_branch:
        _git("branch", f"research-run/{stamp}", cwd=repo)
    return repo


def _branches(repo: Path) -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        capture_output=True, text=True, check=True).stdout
    return set(out.splitlines())


def test_candidates_rename_branch_and_digest_and_queue_carry_final_name(tmp_path):
    repo = _rename_repo(tmp_path)
    trailer = {
        "hypotheses": [
            {"title": "A", "verdict": "candidate-preview-pass",
             "merge_back": {"strategy": "strat_a", "universe": "sp500",
                            "start": "2024-01-01", "end": "2024-06-01", "eval_context": _ec()}},
            {"title": "B", "verdict": "candidate-preview-pass",
             "merge_back": {"strategy": "strat_b", "universe": "sp500",
                            "start": "2024-01-01", "end": "2024-06-01", "eval_context": _ec()}},
        ],
        "preview_gate": {"passed": True, "failed_checks": []},
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a,strat_b", git_root=repo)
    assert proc.returncode == 0, proc.stderr

    final = "research-run/20260811-000000--strat_a+strat_b"
    assert f"run branch renamed: research-run/20260811-000000 -> {final}" in proc.stdout
    # The git branch itself was renamed (old name gone).
    assert final in _branches(repo)
    assert "research-run/20260811-000000" not in _branches(repo)
    # The digest row records the FINAL branch, report ref included.
    row = _last_digest_row(digest_path)
    assert row["branch"] == final
    assert row["report"] == f"{final}:kb/research-runs/20260811-000000.md"
    # Every enqueue used the FINAL branch too.
    items = _queue_items(queue_path)
    assert set(items) == {f"strat_a@{final}", f"strat_b@{final}"}
    assert all(item["branch"] == final for item in items.values())


def test_no_candidates_means_no_rename(tmp_path):
    repo = _rename_repo(tmp_path)
    trailer = {
        "hypotheses": [{"title": "Nope", "verdict": "discarded", "merge_back": None}],
        "preview_gate": {"passed": False, "failed_checks": ["holdout_sharpe_floor"]},
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a", git_root=repo)
    assert proc.returncode == 0, proc.stderr
    assert "run branch renamed" not in proc.stdout
    assert "research-run/20260811-000000" in _branches(repo)  # untouched, stamp-only
    assert _last_digest_row(digest_path)["branch"] == "research-run/20260811-000000"
    assert not queue_path.exists()


def test_rename_failure_keeps_old_branch_everywhere(tmp_path):
    # The branch does NOT exist in git_root -> `git branch -m` fails -> the OLD name is kept and
    # is what the digest row + enqueue record (they always describe the branch that exists).
    repo = _rename_repo(tmp_path, with_branch=False)
    trailer = {
        "hypotheses": [{"title": "A", "verdict": "candidate-preview-pass",
                        "merge_back": {"strategy": "strat_a", "universe": "sp500",
                                       "start": "2024-01-01", "end": "2024-06-01",
                                       "eval_context": _ec()}}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a", git_root=repo)
    assert proc.returncode == 0, proc.stderr
    assert "WARNING: branch rename" in proc.stdout
    row = _last_digest_row(digest_path)
    assert row["branch"] == "research-run/20260811-000000"
    items = _queue_items(queue_path)
    assert set(items) == {"strat_a@research-run/20260811-000000"}


def test_rename_respects_the_name_cap(tmp_path):
    # 4 candidates -> only the first 3 names ride the branch (compute_run_branch_name's cap).
    repo = _rename_repo(tmp_path)
    names = ["strat_a", "strat_b", "strat_c", "strat_d"]
    trailer = {
        "hypotheses": [
            {"title": f"H {n}", "verdict": "candidate-preview-pass",
             "merge_back": {"strategy": n, "universe": "sp500",
                            "start": "2024-01-01", "end": "2024-06-01", "eval_context": _ec()}}
            for n in names
        ],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names=",".join(names), git_root=repo)
    assert proc.returncode == 0, proc.stderr
    final = "research-run/20260811-000000--strat_a+strat_b+strat_c"
    assert final in _branches(repo)
    # ALL FOUR candidates still enqueue — the cap trims the branch NAME, never the queue.
    items = _queue_items(queue_path)
    assert set(items) == {f"{n}@{final}" for n in names}


# --- end-of-run enqueued-candidate counter (heredoc 3): the zero-candidate removal signal ---------


def _run_count_enqueued(queue_path: Path, lock_path: Path, branch: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-", str(queue_path), str(lock_path), branch,
         str(REPO_ROOT / ".codex" / "scripts" / "mergeback_queue.py")],
        input=_COUNT_ENQUEUED_SRC, capture_output=True, text=True, timeout=30, check=True,
    )
    return proc.stdout.strip()


def test_count_enqueued_counts_only_items_on_the_final_branch(tmp_path):
    trailer = {
        "hypotheses": [{"title": "A", "verdict": "candidate-preview-pass",
                        "merge_back": {"strategy": "strat_a", "universe": "sp500",
                                       "start": "2024-01-01", "end": "2024-06-01",
                                       "eval_context": _ec()}}],
        "preview_gate": None,
    }
    proc, _digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    lock_path = tmp_path / "queue.lock"
    assert _run_count_enqueued(queue_path, lock_path, "research-run/20260811-000000") == "1"
    assert _run_count_enqueued(queue_path, lock_path, "research-run/29990101-000000") == "0"


def test_count_enqueued_missing_queue_file_counts_zero(tmp_path):
    assert _run_count_enqueued(
        tmp_path / "absent.json", tmp_path / "absent.lock", "research-run/20260811-000000") == "0"


def test_rename_updates_a_checked_out_worktree_head(tmp_path):
    # The PRODUCTION shape: the run branch is CHECKED OUT in the run worktree when the rename
    # fires from the main repo — git renames the ref AND updates the per-worktree HEAD, which is
    # exactly what the launcher's FINAL_BRANCH re-read (`git -C $WORKTREE branch --show-current`)
    # depends on.
    repo = _rename_repo(tmp_path, with_branch=False)
    worktree = repo / ".runs" / "20260811-000000"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-b", "research-run/20260811-000000",
         str(worktree)],
        check=True, capture_output=True)

    trailer = {
        "hypotheses": [{"title": "A", "verdict": "candidate-preview-pass",
                        "merge_back": {"strategy": "strat_a", "universe": "sp500",
                                       "start": "2024-01-01", "end": "2024-06-01",
                                       "eval_context": _ec()}}],
        "preview_gate": None,
    }
    proc, digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a", git_root=repo)
    assert proc.returncode == 0, proc.stderr

    final = "research-run/20260811-000000--strat_a"
    assert f"run branch renamed: research-run/20260811-000000 -> {final}" in proc.stdout
    head = subprocess.run(
        ["git", "-C", str(worktree), "branch", "--show-current"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert head == final  # the worktree's HEAD followed the rename
    assert _last_digest_row(digest_path)["branch"] == final
    assert set(_queue_items(queue_path)) == {f"strat_a@{final}"}


def test_count_enqueued_prints_minus_one_when_the_queue_lock_is_held(tmp_path):
    # FAIL CLOSED: a QueueLockTimeout (a concurrent drain holding the queue lock) must NOT read as
    # "zero candidates" — that would remove the worktree of a run WITH pending items. The heredoc
    # prints -1 and the launcher's shell keeps the worktree for anything other than a literal "0".
    trailer = {
        "hypotheses": [{"title": "A", "verdict": "candidate-preview-pass",
                        "merge_back": {"strategy": "strat_a", "universe": "sp500",
                                       "start": "2024-01-01", "end": "2024-06-01",
                                       "eval_context": _ec()}}],
        "preview_gate": None,
    }
    proc, _digest_path, queue_path = _run_append_digest(
        tmp_path, trailer=trailer, strategy_names="strat_a")
    assert proc.returncode == 0, proc.stderr
    lock_path = tmp_path / "queue.lock"

    with open(lock_path, "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # another process "owns" the queue for the duration
        out = _run_count_enqueued(queue_path, lock_path, "research-run/20260811-000000")
    assert out == "-1"
    # And once the lock is free again, the same call reports the real (nonzero) count.
    assert _run_count_enqueued(queue_path, lock_path, "research-run/20260811-000000") == "1"
