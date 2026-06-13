"""End-to-end research lifecycle through the real console-script entry point.

Unlike the per-command CliRunner tests, this drives ``algua.cli.main:main`` the way the
installed ``algua`` binary does: each command is a fresh ``main(argv)`` call that emits JSON on
real stdout and exits with a real code. It exercises the cross-command seams (registry stage
flowing from one command into the next, the promotion gate landing in the registry) and the
human-only live wall, end to end, in one offline pass on the synthetic provider.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from algua.cli.main import main

STRATEGY = "cross_sectional_momentum"

# The console script the `[project.scripts]` wiring installs sits next to the venv's python.
ALGUA_BIN = Path(sys.executable).parent / "algua"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "lifecycle.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _run(capsys, *args: str) -> tuple[int, dict]:
    """Invoke the entry point as the installed binary would; return (exit_code, parsed JSON)."""
    with pytest.raises(SystemExit) as exit_info:
        main(list(args))
    out = capsys.readouterr().out
    code = exit_info.value.code or 0
    return code, json.loads(out)


def _stage(capsys) -> str:
    code, payload = _run(capsys, "registry", "show", STRATEGY)
    assert code == 0
    return payload["stage"]


def test_full_research_lifecycle_to_shortlist_and_live_wall(capsys):
    # idea -> backtested: a single backtest run that also registers the strategy.
    code, payload = _run(capsys, "backtest", "run", STRATEGY, "--demo",
                         "--start", "2022-01-01", "--end", "2023-12-31", "--register")
    assert code == 0, payload
    assert _stage(capsys) == "backtested"

    # backtested -> candidate: the promotion gate evaluates and transitions on pass.
    code, payload = _run(capsys, "research", "promote", STRATEGY, "--demo",
                         "--start", "2022-01-01", "--end", "2023-12-31",
                         "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                         "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                         "--n-combos", "9", "--allow-non-pit", "--actor", "human")
    assert code == 0, payload
    assert payload["passed"] is True
    assert payload["promoted"] is True
    assert _stage(capsys) == "candidate"

    # candidate -> paper: an agent may operate the lifecycle up to and including paper.
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "paper", "--actor", "agent", "--reason", "e2e")
    assert code == 0, payload
    assert _stage(capsys) == "paper"

    # paper -> forward_tested: a human must advance to forward_tested before the live gate.
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "forward_tested", "--actor", "human", "--reason", "e2e fwd test")
    assert code == 0, payload
    assert _stage(capsys) == "forward_tested"

    # forward_tested -> live by an agent is the hard wall: fails, stage stays forward_tested.
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "live", "--actor", "agent", "--reason", "should be blocked")
    assert code == 1
    assert payload["ok"] is False
    assert _stage(capsys) == "forward_tested"


# ---------------------------------------------------------------------------
# Issue #125: dormant lifecycle end-to-end CLI tests
# ---------------------------------------------------------------------------

def _to_paper_e2e(capsys):
    """Drive the strategy from idea to paper via the legal chain using CLI transitions.

    Uses human actor for backtested/candidate steps (the raw shortcut route) and an
    agent actor for the candidate->paper step, mirroring the existing e2e pattern.
    """
    code, payload = _run(capsys, "registry", "add", STRATEGY)
    assert code == 0, payload
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "backtested", "--actor", "human", "--reason", "e2e setup")
    assert code == 0, payload
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "candidate", "--actor", "human", "--reason", "e2e setup")
    assert code == 0, payload
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "paper", "--actor", "agent", "--reason", "e2e setup")
    assert code == 0, payload
    assert _stage(capsys) == "paper"


def test_e2e_paper_to_dormant_to_retired(capsys):
    """paper -> dormant -> retired: both transitions succeed and stages are reported correctly."""
    _to_paper_e2e(capsys)

    # paper -> dormant: needs a non-empty reason; agent actor is allowed.
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "dormant", "--actor", "agent", "--reason", "seasonal")
    assert code == 0, payload
    assert _stage(capsys) == "dormant"

    # dormant -> retired: also legal for any actor.
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "retired", "--actor", "agent", "--reason", "done")
    assert code == 0, payload
    assert _stage(capsys) == "retired"


def test_e2e_paper_to_dormant_requires_reason(capsys):
    """Transitioning to dormant without --reason must fail with a non-zero exit and an error
    mentioning 'reason'."""
    _to_paper_e2e(capsys)

    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "dormant", "--actor", "agent")
    assert code != 0, payload
    assert payload.get("ok") is False
    assert "reason" in payload.get("error", "").lower()
    # Stage must be unchanged.
    assert _stage(capsys) == "paper"


def test_e2e_dormant_recovers_to_paper(capsys):
    """A dormant strategy can be reactivated back to paper by any actor."""
    _to_paper_e2e(capsys)

    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "dormant", "--actor", "agent", "--reason", "seasonal")
    assert code == 0, payload
    assert _stage(capsys) == "dormant"

    # dormant -> paper: the back-step edge is allowed.
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "paper", "--actor", "agent", "--reason", "revive")
    assert code == 0, payload
    assert _stage(capsys) == "paper"


@pytest.mark.skipif(not ALGUA_BIN.exists(), reason="algua console script not installed")
def test_installed_console_script_emits_json_and_exit_codes(tmp_path):
    """Smoke the real installed binary: in-process main() can't catch a broken script wiring."""
    env = {**os.environ,
           "ALGUA_DB_PATH": str(tmp_path / "x.db"), "ALGUA_DATA_DIR": str(tmp_path)}

    ok = subprocess.run([str(ALGUA_BIN), "version"],
                        capture_output=True, text=True, env=env)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["name"] == "algua"

    # A bad option must still honor the JSON contract on real stdout with a non-zero exit.
    bad = subprocess.run([str(ALGUA_BIN), "version", "--nope"],
                         capture_output=True, text=True, env=env)
    assert bad.returncode == 1
    assert json.loads(bad.stdout)["ok"] is False
