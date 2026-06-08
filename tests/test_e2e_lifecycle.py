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

    # backtested -> shortlisted: the promotion gate evaluates and transitions on pass.
    code, payload = _run(capsys, "research", "promote", STRATEGY, "--demo",
                         "--start", "2022-01-01", "--end", "2023-12-31",
                         "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                         "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                         "--n-combos", "9", "--allow-non-pit", "--actor", "human")
    assert code == 0, payload
    assert payload["passed"] is True
    assert payload["promoted"] is True
    assert _stage(capsys) == "shortlisted"

    # shortlisted -> paper: an agent may operate the lifecycle up to and including paper.
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "paper", "--actor", "agent", "--reason", "e2e")
    assert code == 0, payload
    assert _stage(capsys) == "paper"

    # paper -> live by an agent is the hard wall: it must fail as JSON, leaving the stage at paper.
    code, payload = _run(capsys, "registry", "transition", STRATEGY,
                         "--to", "live", "--actor", "agent", "--reason", "should be blocked")
    assert code == 1
    assert payload["ok"] is False
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
