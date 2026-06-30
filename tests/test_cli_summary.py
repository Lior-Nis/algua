"""--summary field projection on the heavy agent-facing commands (#349, context-rot defense)."""

import json

import pytest
from typer.testing import CliRunner

from algua.cli._common import project
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _run(args):
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    return json.loads(res.output)


# --- unit: the shared projection helper ---


def test_project_keeps_listed_and_ok_adds_marker():
    out = project({"ok": True, "a": 1, "b": 2, "c": 3}, keep=("a", "c"))
    assert out == {"ok": True, "a": 1, "c": 3, "summary": True}


def test_project_ignores_absent_keys():
    out = project({"ok": True, "a": 1}, keep=("a", "missing"))
    assert out == {"ok": True, "a": 1, "summary": True}


# --- backtest walk-forward ---


def test_walk_forward_summary_projects():
    base = ["backtest", "walk-forward", "cross_sectional_momentum", "--demo",
            "--start", "2022-01-01", "--end", "2023-12-31"]
    full = _run(base)
    summ = _run(base + ["--summary"])
    # full output unchanged: bulky per-window list present, no marker
    assert "window_metrics" in full and "summary" not in full
    # summary drops the bulk, keeps the decision scalars + marker, values agree
    assert "window_metrics" not in summ
    assert summ["ok"] is True and summ["summary"] is True
    assert summ["stability"] == full["stability"]
    assert summ["strategy"] == full["strategy"]


# --- backtest sweep ---


def test_sweep_summary_projects():
    base = ["backtest", "sweep", "cross_sectional_momentum", "--demo",
            "--start", "2022-01-01", "--end", "2023-12-31", "--param", "lookback=20,40"]
    full = _run(base)
    summ = _run(base + ["--summary"])
    # full output unchanged: the per-combo list + grid present, no marker
    assert "ranked" in full and "grid" in full and "summary" not in full
    # summary drops the bulk, keeps the headline combo + breadth + marker, values agree
    assert "ranked" not in summ and "grid" not in summ
    assert summ["summary"] is True
    assert summ["best"] == full["best"]
    assert "recorded_breadth" in summ  # breadth is stateful/cumulative, so just assert it survives
