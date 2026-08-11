"""Tests for `ALGUA_PAPER_BOOK_CAPACITY` (factory slice 3): the shared wide-book default that
replaces `paper_cmd.py`'s three independently-hardcoded 5/5/8 `--max-concurrent` defaults.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

import algua.cli.paper_cmd as paper_cmd
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.execution.alpaca_broker import AccountState
from algua.registry.db import connect, migrate

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    monkeypatch.delenv("ALGUA_PAPER_BOOK_CAPACITY", raising=False)


class _FakeBroker:
    def __init__(self, equity: float = 100_000.0):
        self._equity = equity

    def account(self) -> AccountState:
        return AccountState(
            equity=self._equity, cash=self._equity, buying_power=self._equity, account_id="t")


def _db_path():
    return get_settings().db_path


# --- the resolver itself --------------------------------------------------------------------------


def test_default_setting_is_64_when_unset():
    assert get_settings().paper_book_capacity == 64


def test_resolve_max_concurrent_none_reads_settings(monkeypatch):
    monkeypatch.setenv("ALGUA_PAPER_BOOK_CAPACITY", "42")
    assert paper_cmd._resolve_max_concurrent(None) == 42


def test_resolve_max_concurrent_explicit_overrides_settings(monkeypatch):
    monkeypatch.setenv("ALGUA_PAPER_BOOK_CAPACITY", "42")
    assert paper_cmd._resolve_max_concurrent(7) == 7


def test_resolve_max_concurrent_rereads_env_fresh_each_call(monkeypatch):
    # Regression guard for the import-time-baking bug this design deliberately avoids: a
    # `typer.Option(get_settings().paper_book_capacity, ...)` default would freeze the value at
    # module-import time and never observe a later env change. `_resolve_max_concurrent` must NOT
    # do that — a `None` explicit value must reflect whatever the env says AT CALL TIME.
    monkeypatch.setenv("ALGUA_PAPER_BOOK_CAPACITY", "10")
    assert paper_cmd._resolve_max_concurrent(None) == 10
    monkeypatch.setenv("ALGUA_PAPER_BOOK_CAPACITY", "99")
    assert paper_cmd._resolve_max_concurrent(None) == 99


# --- `paper intake` reports the resolved capacity directly in its JSON --------------------------


def test_intake_uses_settings_capacity_when_omitted(monkeypatch):
    monkeypatch.setenv("ALGUA_PAPER_BOOK_CAPACITY", "17")
    monkeypatch.setattr(paper_cmd, "_alpaca_broker_from_settings", lambda: _FakeBroker())
    result = runner.invoke(app, ["paper", "intake"])
    assert result.exit_code == 0, result.output
    import json
    payload = json.loads(result.output)
    assert payload["max_concurrent"] == 17


def test_intake_explicit_flag_overrides_settings(monkeypatch):
    monkeypatch.setenv("ALGUA_PAPER_BOOK_CAPACITY", "17")
    monkeypatch.setattr(paper_cmd, "_alpaca_broker_from_settings", lambda: _FakeBroker())
    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "3"])
    assert result.exit_code == 0, result.output
    import json
    payload = json.loads(result.output)
    assert payload["max_concurrent"] == 3


# --- `paper allocate` actually ENFORCES the resolved capacity (count cap) ------------------------


def _seed_one_paper_occupant(conn) -> None:
    """Strategy #1: already at `paper` with an active allocation (consumes 1 of the count cap).
    Inserted via raw SQL — `paper allocate`/`allocate_in_lane` never load the strategy MODULE, only
    the registry row, so this needs no real strategy python file."""
    conn.execute(
        "INSERT INTO strategies(id, name, stage, created_at, updated_at) "
        "VALUES (1, 'occupant', 'paper', '2026-01-01', '2026-01-01')")
    conn.execute(
        "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
        "VALUES (1, 1000.0, ?, 'agent')", (datetime.now(UTC).isoformat(),))


def _seed_unallocated_paper_target(conn) -> None:
    """Strategy #2: at `paper` but with NO allocation yet — a count-INCREASING allocation attempt,
    exactly what the count cap gates."""
    conn.execute(
        "INSERT INTO strategies(id, name, stage, created_at, updated_at) "
        "VALUES (2, 'target', 'paper', '2026-01-01', '2026-01-01')")


def test_allocate_enforces_settings_capacity_when_omitted(monkeypatch):
    monkeypatch.setenv("ALGUA_PAPER_BOOK_CAPACITY", "1")
    monkeypatch.setattr(paper_cmd, "_alpaca_broker_from_settings", lambda: _FakeBroker())
    conn = connect(_db_path())
    migrate(conn)
    _seed_one_paper_occupant(conn)
    _seed_unallocated_paper_target(conn)
    conn.commit()
    conn.close()

    # Cap is 1 (from settings) and it's already consumed by 'occupant' — a SECOND
    # count-increasing allocation for 'target' must be rejected, with NO --max-concurrent given.
    result = runner.invoke(app, ["paper", "allocate", "target", "--capital", "500"])
    assert result.exit_code == 1, result.output
    import json
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["code"] == "allocation_error"


def test_allocate_explicit_flag_overrides_settings_cap(monkeypatch):
    monkeypatch.setenv("ALGUA_PAPER_BOOK_CAPACITY", "1")
    monkeypatch.setattr(paper_cmd, "_alpaca_broker_from_settings", lambda: _FakeBroker())
    conn = connect(_db_path())
    migrate(conn)
    _seed_one_paper_occupant(conn)
    _seed_unallocated_paper_target(conn)
    conn.commit()
    conn.close()

    # An explicit --max-concurrent 2 overrides the settings-derived cap of 1, so the SECOND
    # allocation now fits.
    result = runner.invoke(
        app, ["paper", "allocate", "target", "--capital", "500", "--max-concurrent", "2"])
    assert result.exit_code == 0, result.output


# --- `paper merge-back` threads the same default into its own intake sub-call -------------------


def test_merge_back_command_resolves_capacity_via_the_shared_helper():
    # `merge-back`'s full git/gate/promote/intake flow is exercised end-to-end in
    # test_cli_merge_back.py (heavy stubs: fake git, fake promote, fake intake). This is a
    # wiring check on top of that: confirm the command body actually calls the SAME
    # `_resolve_max_concurrent` helper `intake`/`allocate` use (already proven correct above and
    # in the `--max-concurrent` override test), rather than re-hardcoding a literal default.
    import inspect
    src = inspect.getsource(paper_cmd.merge_back)
    assert "_resolve_max_concurrent(max_concurrent)" in src
