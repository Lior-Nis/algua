import json
import sys

import pytest

from algua.cli.main import main


def test_bad_option_type_renders_json(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv",
                        ["algua", "backtest", "walk-forward", "x", "--demo", "--windows", "nope"])
    with pytest.raises(SystemExit) as ei:
        main()
    assert ei.value.code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_unknown_option_renders_json(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["algua", "version", "--nope"])
    with pytest.raises(SystemExit) as ei:
        main()
    assert ei.value.code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_version_succeeds_via_main(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["algua", "version"])
    with pytest.raises(SystemExit) as ei:
        main()
    assert ei.value.code == 0
    assert json.loads(capsys.readouterr().out)["name"] == "algua"
