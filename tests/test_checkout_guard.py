"""#687: operator lanes refuse a checkout that is not a clean ``main``; ``fleet health`` reports it.

The git probe is INJECTED (no real repo state). Test environments: ``tests/conftest.py`` installs
a clean-main fake as the default probe for the whole suite (autouse), so the suite is independent
of the branch it runs on (worktrees included). There is deliberately no env-var bypass.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from algua.cli.errors import error_code
from algua.cli.main import app
from algua.operator import checkout_guard as cg
from algua.operator.checkout_guard import (
    CheckoutNotCleanMain,
    GitCheckoutProbe,
    checkout_status,
    require_clean_main,
)

runner = CliRunner()


class FakeProbe:
    def __init__(self, branch="main", dirty=False, boom=False):
        self._b, self._d, self._boom = branch, dirty, boom

    def branch(self):
        if self._boom:
            raise subprocess.CalledProcessError(128, "git")
        return self._b

    def tracked_dirty(self):
        return self._d


def _use(monkeypatch, probe):
    monkeypatch.setattr(cg, "default_probe", lambda: probe)


def test_clean_main_passes():
    require_clean_main(FakeProbe())
    assert checkout_status(FakeProbe())["ok"] is True


@pytest.mark.parametrize("probe", [
    FakeProbe(branch="feat/x"), FakeProbe(branch="HEAD"), FakeProbe(dirty=True),
    FakeProbe(boom=True)])
def test_not_clean_main_refused(probe):
    with pytest.raises(CheckoutNotCleanMain):
        require_clean_main(probe)
    assert checkout_status(probe)["ok"] is False


def test_error_code_is_stable():
    assert error_code(CheckoutNotCleanMain("x")) == "checkout_not_clean_main"


@pytest.mark.parametrize("lane", ["paper", "live"])
@pytest.mark.parametrize("flag", [["--snapshot", "x"], ["--refresh"]])
def test_run_all_refused_on_non_main(monkeypatch, lane, flag):
    _use(monkeypatch, FakeProbe(branch="feat/story-1-3b"))
    r = runner.invoke(app, [lane, "run-all", *flag])
    assert r.exit_code == 1
    out = json.loads(r.stdout)
    assert out["ok"] is False and out["code"] == "checkout_not_clean_main"


def test_trade_tick_refused_on_non_main(monkeypatch):
    _use(monkeypatch, FakeProbe(branch="feat/x"))
    r = runner.invoke(app, ["paper", "trade-tick", "anything", "--snapshot", "x"])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["code"] == "checkout_not_clean_main"


@pytest.mark.parametrize("lane", ["paper", "live"])
def test_run_all_refused_on_dirty_main(monkeypatch, lane):
    _use(monkeypatch, FakeProbe(dirty=True))
    r = runner.invoke(app, [lane, "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["code"] == "checkout_not_clean_main"


@pytest.mark.parametrize("lane", ["paper", "live"])
def test_run_all_proceeds_on_clean_main(monkeypatch, lane):
    _use(monkeypatch, FakeProbe())
    r = runner.invoke(app, [lane, "run-all", "--snapshot", "x"])
    assert r.exit_code == 0, r.stdout  # empty registry: a no-op cycle, past the guard


def test_fleet_health_reports_non_main(monkeypatch):
    _use(monkeypatch, FakeProbe(branch="feat/x"))
    r = runner.invoke(app, ["fleet", "health"])
    assert r.exit_code == 1
    out = json.loads(r.stdout)
    assert out["ok"] is False and out["checkout"]["ok"] is False
    assert out["checkout"]["branch"] == "feat/x"


def test_fleet_health_ok_on_clean_main(monkeypatch):
    _use(monkeypatch, FakeProbe())
    r = runner.invoke(app, ["fleet", "health"])
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["checkout"]["ok"] is True


def _git(root: Path, *a: str) -> None:
    subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)


def test_real_probe_against_throwaway_repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("a")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-qm", "i")
    p = GitCheckoutProbe(tmp_path)
    assert (p.branch(), p.tracked_dirty()) == ("main", False)
    (tmp_path / "untracked.txt").write_text("u")
    assert p.tracked_dirty() is False  # untracked files are not tracked modifications
    (tmp_path / "f.txt").write_text("b")
    assert p.tracked_dirty() is True
    _git(tmp_path, "checkout", "-q", "-b", "side")
    assert p.branch() == "side"
    with pytest.raises(CheckoutNotCleanMain):
        require_clean_main(p)
