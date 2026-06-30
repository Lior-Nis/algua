"""Wiring tests for #346: structured logging in the always-on paper loop.

Asserts the JSON logger is attached to the `algua` logger and that the cycle emits the expected
structured records to STDERR (never stdout), including the crash-safe `golden_signals` rollup that
must flush in `finally` even when the cycle exits non-zero.
"""
from __future__ import annotations

import json
import logging

import pandas as pd
import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.execution.alpaca_broker import AccountState

runner = CliRunner()

_NAME = "cross_sectional_momentum"
_START = "2022-01-01"
_END = "2023-06-01"


def _to_paper() -> None:
    assert runner.invoke(app, ["backtest", "run", _NAME, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", _NAME, "--to", "candidate",
                               "--actor", "human", "--reason", "ok"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", _NAME, "--to", "paper",
                               "--actor", "agent", "--reason", "paper"]).exit_code == 0


class _CaptureHandler(logging.Handler):
    """Capture formatted JSON records emitted to the `algua` logger."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        # Use the real JSON formatter so we exercise it end-to-end.
        from algua.observability.log import JsonFormatter

        self.records.append(json.loads(JsonFormatter().format(record)))


@pytest.fixture
def capture(monkeypatch):
    """Attach a capture handler directly to the `algua` logger (caplog can't see it because
    the logger sets propagate=False); detach on teardown."""
    handler = _CaptureHandler()
    logger = logging.getLogger("algua")
    logger.addHandler(handler)
    monkeypatch.setenv("ALGUA_LOG_LEVEL", "INFO")
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def _msgs(handler: _CaptureHandler) -> list[str]:
    return [r["msg"] for r in handler.records]


def test_clean_breach_tick_emits_golden_signals_and_correlation(monkeypatch, tmp_path, capture):
    """A clean reconcile that ticks then breaches: ticks==1, breaches==1, flatten_failures==1,
    and every record shares one correlation id."""
    handler = capture
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    class _CleanBroker:
        def clock(self) -> str:
            return "2024-01-15T14:00:00Z"

        def account(self) -> AccountState:
            return AccountState(equity=100_000.0, cash=100_000.0,
                                buying_power=100_000.0, account_id="acct")

        def account_activities_window(self, after: str, until: str) -> list:
            return []

        def get_positions(self) -> pd.Series:
            return pd.Series(dtype=float)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: _CleanBroker())
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: object())
    # Force the tick to report a breach WITH a flatten failure (no real sizing needed).
    monkeypatch.setattr(
        "algua.cli.paper_cmd._run_paper_strategy_tick",
        lambda *a, **k: {"ok": False, "strategy": _NAME, "kind": "drawdown",
                         "flatten_error": "boom"},
    )

    result = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    assert result.exit_code == 1, result.stdout

    assert "cycle_start" in _msgs(handler)
    gs = [r for r in handler.records if r["msg"] == "golden_signals"]
    assert len(gs) == 1
    assert gs[0]["ticks"] == 1 and gs[0]["breaches"] == 1 and gs[0]["flatten_failures"] == 1
    # one correlation id across the whole cycle
    cids = {r.get("correlation_id") for r in handler.records}
    assert len(cids) == 1 and None not in cids


def test_ingest_failure_still_flushes_golden_signals(monkeypatch, tmp_path, capture):
    """Crash-safety: the cycle fails on venue ingest yet golden_signals still flushes (finally),
    and a venue_ingest_failed ERROR carries the exception."""
    from algua.execution.alpaca_broker import BrokerError

    handler = capture
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    class _IngestFailBroker:
        def clock(self) -> str:
            return "2024-01-15T14:00:00Z"

        def account(self) -> AccountState:
            return AccountState(equity=100_000.0, cash=100_000.0,
                                buying_power=100_000.0, account_id="fail-acct")

        def account_activities_window(self, after: str, until: str) -> list:
            raise BrokerError("transport error: 503")

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _IngestFailBroker())
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider",
                        lambda demo, snapshot: object())

    result = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    assert result.exit_code == 1

    msgs = _msgs(handler)
    assert "venue_ingest_failed" in msgs
    assert "golden_signals" in msgs  # flushed in finally despite the failure
    err = next(r for r in handler.records if r["msg"] == "venue_ingest_failed")
    assert err["level"] == "ERROR" and err["exc_type"] == "BrokerError"


def test_logger_writes_nothing_to_stdout(monkeypatch, tmp_path):
    """The command's own JSON envelope is the ONLY thing on stdout; logger output is on stderr.

    This click/typer version keeps stdout and stderr separate; stdout must parse as exactly the
    command envelope (one JSON document), never interleaved with log lines.
    """
    from algua.execution.alpaca_broker import BrokerError

    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()

    class _IngestFailBroker:
        def clock(self) -> str:
            return "2024-01-15T14:00:00Z"

        def account(self) -> AccountState:
            return AccountState(equity=1.0, cash=1.0, buying_power=1.0, account_id="a")

        def account_activities_window(self, after: str, until: str) -> list:
            raise BrokerError("503")

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _IngestFailBroker())
    monkeypatch.setattr("algua.cli.paper_cmd._select_provider", lambda demo, snapshot: object())

    result = runner.invoke(app, ["paper", "trade-tick", _NAME, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    assert result.exit_code == 1
    # stdout is exactly one JSON document (the command envelope) — no log lines leaked in.
    payload = json.loads(result.stdout)
    assert payload["ok"] is False and payload["kind"] == "venue_ingest_failed"
