"""Tests for the CLI machine-readable error `code` + json_errors catch-all (issue #337)."""

import json
import sqlite3
import sys

import pytest
import typer

from algua.backtest.engine import BacktestError
from algua.cli.app import emit
from algua.cli.errors import error_code, is_retryable, json_errors
from algua.cli.main import main
from algua.contracts.lifecycle import TransitionError
from algua.data.store import SnapshotNotFound
from algua.live.live_loop import TickHalted
from algua.risk.limits import RiskBreach

# --- error_code: the type-keyed registry resolves the right code ------------------------------

@pytest.mark.parametrize(
    "exc, expected",
    [
        (ValueError("bad"), "invalid_input"),
        (LookupError("missing"), "not_found"),
        (KeyError("k"), "not_found"),  # KeyError IS a LookupError -> generic not_found bucket
        (RuntimeError("boom"), "internal"),  # unexpected/bug-class -> internal
        (AttributeError("attr"), "internal"),
        (TransitionError("wrong stage"), "wrong_stage"),
        (SnapshotNotFound("no snap"), "not_found"),
        (TickHalted("halted"), "tick_halted"),
        (BacktestError("bt"), "backtest_error"),
        (RiskBreach("gross_exposure", "too much"), "risk_breach"),
        (FileNotFoundError("nope"), "file_not_found"),
        (sqlite3.OperationalError("locked"), "db_unavailable"),
    ],
)
def test_error_code_resolves(exc, expected):
    assert error_code(exc) == expected


def test_error_code_specific_beats_generic():
    # TransitionError is a ValueError subclass; the specific code must win over invalid_input.
    assert error_code(TransitionError("x")) == "wrong_stage"
    assert isinstance(TransitionError("x"), ValueError)


# --- is_retryable: transient env failures opt in; everything else is abort-only ---------------

@pytest.mark.parametrize(
    "code, expected",
    [
        ("db_unavailable", True),   # transient SQLite contention -> retry-with-backoff
        ("invalid_input", False),   # deterministic input error -> abort
        ("internal", False),        # bug-class -> abort
        ("wrong_stage", False),
        ("usage_error", False),
        ("aborted", False),
    ],
)
def test_is_retryable(code, expected):
    assert is_retryable(code) is expected


# --- json_errors: catch-all renders ANY exception as the JSON envelope -------------------------


def test_json_errors_marks_db_unavailable_retryable(capsys):
    @json_errors
    def cmd():
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(typer.Exit):
        cmd()
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "db_unavailable"
    assert payload["retryable"] is True


def test_json_errors_marks_value_error_not_retryable(capsys):
    @json_errors
    def cmd():
        raise ValueError("bad input")

    with pytest.raises(typer.Exit):
        cmd()
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": False,
        "error": "bad input",
        "code": "invalid_input",
        "retryable": False,
    }

def test_json_errors_catch_all_renders_undeclared_exception(capsys):
    @json_errors
    def cmd():
        raise RuntimeError("deep pandas-ish failure no author anticipated")

    with pytest.raises(typer.Exit) as ei:
        cmd()
    assert ei.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out)  # valid JSON, not a traceback
    assert payload["ok"] is False
    assert payload["code"] == "internal"  # unanticipated type -> internal, contract intact


def test_json_errors_stamps_specific_code(capsys):
    @json_errors
    def cmd():
        raise TransitionError("requires stage backtested")

    with pytest.raises(typer.Exit):
        cmd()
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": False,
        "error": "requires stage backtested",
        "code": "wrong_stage",
        "retryable": False,
    }


def test_json_errors_reraises_typer_exit_unchanged(capsys):
    @json_errors
    def cmd():
        emit({"ok": False, "custom": "envelope"})
        raise typer.Exit(code=3)

    with pytest.raises(typer.Exit) as ei:
        cmd()
    assert ei.value.exit_code == 3  # control flow preserved, NOT converted to internal
    # The command's own envelope stands; the decorator did not add a `code`.
    assert json.loads(capsys.readouterr().out) == {"ok": False, "custom": "envelope"}


def test_json_errors_reraises_typer_abort():
    @json_errors
    def cmd():
        raise typer.Abort()

    with pytest.raises(typer.Abort):
        cmd()


def test_json_errors_passes_success_through(capsys):
    @json_errors
    def cmd():
        emit({"ok": True, "value": 42})

    cmd()
    assert json.loads(capsys.readouterr().out) == {"ok": True, "value": 42}


# --- main(): arg-parse codes + last-resort catch-all -------------------------------------------

def test_main_usage_error_carries_code(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["algua", "version", "--nope"])
    with pytest.raises(SystemExit) as ei:
        main()
    assert ei.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == "usage_error"


def test_main_imports_every_command_module():
    # Smoke test: importing main pulls in every command module; a leftover `@json_errors(...)`
    # call-form (post-migration) would raise TypeError at import, failing this.
    import algua.cli.main as m

    assert m.app is not None
