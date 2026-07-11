import sqlite3
from datetime import UTC, datetime

import pytest

from algua.cli._common import (
    now_iso,
    registry_conn,
    resolve_eval_inputs,
    select_provider,
    utc,
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "c.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def test_utc_stamps_utc():
    dt = utc("2023-01-02")
    assert dt == datetime(2023, 1, 2, tzinfo=UTC)
    assert dt.tzinfo is UTC


def test_now_iso_is_utc_isoformat():
    parsed = datetime.fromisoformat(now_iso())
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == datetime.now(UTC).utcoffset()


def test_registry_conn_migrates_and_closes():
    from algua.registry.store import SqliteStrategyRepository

    with registry_conn() as conn:
        # migrated: a registry write succeeds against a fresh DB
        repo = SqliteStrategyRepository(conn)
        repo.add("demo_strat")
        recs = repo.list_strategies()
        assert any(r.name == "demo_strat" for r in recs)
    # closed on exit: using the connection now raises
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_select_provider_rejects_both_and_neither():
    with pytest.raises(ValueError):
        select_provider(True, "snap")
    with pytest.raises(ValueError):
        select_provider(False, None)


def test_select_provider_demo():
    from algua.backtest._sample import SyntheticProvider

    assert isinstance(select_provider(True, None), SyntheticProvider)


def test_resolve_eval_inputs_returns_quadruple():
    strategy, provider, start_dt, end_dt = resolve_eval_inputs(
        "cross_sectional_momentum", True, None, "2022-01-01", "2022-12-31"
    )
    assert strategy.name == "cross_sectional_momentum"
    assert start_dt == datetime(2022, 1, 1, tzinfo=UTC)
    assert end_dt == datetime(2022, 12, 31, tzinfo=UTC)
    assert provider is not None


def test_breach_payload_shape():
    from algua.cli._common import breach_payload

    p = breach_payload("boom", kind="drawdown", strategy="s")
    assert p == {"ok": False, "kill_switch": "tripped", "error": "boom",
                 "kind": "drawdown", "strategy": "s"}


def test_resolve_drawdown_breaker_defaults_on():
    from algua.cli._common import resolve_drawdown_breaker

    # omitting --max-drawdown resolves to the conservative default-ON setting (0.25), NOT None.
    assert resolve_drawdown_breaker(None, disabled=False) == pytest.approx(0.25)
    # an explicit value is honored as-is.
    assert resolve_drawdown_breaker(0.1, disabled=False) == pytest.approx(0.1)
    # the ONLY off-path is the explicit disable flag.
    assert resolve_drawdown_breaker(None, disabled=True) is None
    assert resolve_drawdown_breaker(0.1, disabled=True) is None


def test_resolve_drawdown_breaker_rejects_misconfigured_default(monkeypatch):
    # a bad env override of the default must fail closed rather than silently disable the breaker.
    from algua.cli import _common
    from algua.config.settings import Settings

    monkeypatch.setattr(_common, "get_settings",
                        lambda: Settings(strategy_max_drawdown_default=1.5))
    with pytest.raises(ValueError):
        _common.resolve_drawdown_breaker(None, disabled=False)


def test_strategy_setup_error_code_is_the_cause_class_name():
    from algua.cli._common import StrategySetupError

    err = StrategySetupError("s1", ValueError("bad config token /secret/path"))
    assert err.code == "ValueError"
    assert err.strategy == "s1"
    assert err.cause.args == ("bad config token /secret/path",)
    # the raw message never leaks into str(err) either.
    assert "secret" not in str(err)


def test_strategy_setup_error_code_sanitizes_unsafe_class_name():
    # #374 GATE-2 MEDIUM: `code` is derived from the raising exception's CLASS NAME, which is
    # normally a safe fixed identifier — but a dynamically-constructed class (built at runtime from
    # untrusted data) could in principle produce an arbitrary string. Since `code` is surfaced
    # verbatim in the JSON envelope and the audit-log `reason` column, anything that doesn't look
    # like a plain identifier must fall back to a fixed classifier rather than being trusted as-is.
    from algua.cli._common import StrategySetupError

    Unsafe = type("../etc/passwd; DROP TABLE strategies;--", (Exception,), {})
    err = StrategySetupError("s1", Unsafe("whatever"))
    assert err.code == "SetupError"
    assert "DROP TABLE" not in err.code


def test_strategy_setup_error_code_rejects_overlong_class_name():
    from algua.cli._common import StrategySetupError

    Unsafe = type("X" * 200, (Exception,), {})
    err = StrategySetupError("s1", Unsafe("whatever"))
    assert err.code == "SetupError"
