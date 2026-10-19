from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "acceptance-report.sh"


def _recording_executable(path: Path, name: str) -> None:
    path.write_text(
        f'#!/usr/bin/env bash\nprintf \'{name}|%s\\n\' "$*" >> "$CALL_LOG"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_acceptance_report_runs_fresh_scenarios_and_generates_dashboard(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _recording_executable(bin_dir / "uv", "uv")
    _recording_executable(bin_dir / "npx", "npx")

    call_log = tmp_path / "calls.log"
    env = os.environ | {
        "CALL_LOG": str(call_log),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "uv|run pytest -m acceptance "
        "--alluredir=.runs/acceptance/allure-results --clean-alluredir",
        "npx|--no-install allure generate .runs/acceptance/allure-results "
        "--output .runs/acceptance/allure-report --clean",
    ]
    assert "Acceptance dashboard:" in completed.stdout
    assert ".runs/acceptance/allure-report/index.html" in completed.stdout
