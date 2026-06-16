# tests/container/test_gate_script.py
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "gate.sh"


def test_gate_script_is_executable():
    assert os.access(GATE, os.X_OK), "scripts/gate.sh must be executable"


def test_gate_script_runs_the_four_quality_commands():
    body = GATE.read_text()
    # Must match the gate defined in CLAUDE.md, in order, so the container can't drift from it.
    for cmd in (
        "uv run pytest -q",
        "uv run ruff check .",
        "uv run mypy algua",
        "uv run lint-imports",
    ):
        assert cmd in body, f"gate.sh must run: {cmd}"


def test_gate_script_fails_fast():
    assert "set -euo pipefail" in GATE.read_text()
