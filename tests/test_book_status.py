"""Book-level capital rollup (`algua book status`).

The condition this exists to catch: `paper -> dormant` atomically releases the slice and
`dormant -> paper` restores the STAGE but not the capital, so a strategy can sit in an operational
stage the operator loop skips forever, surfacing only as an unexplained `idle`.
"""

from contextlib import closing
from datetime import UTC, datetime

from algua.config.settings import get_settings
from algua.contracts.lifecycle import Stage
from algua.execution.book import book_status
from algua.execution.order_state import record_tick_snapshot
from algua.registry.allocations import allocate_locked
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository


def _conn():
    conn = connect(get_settings().db_path)
    migrate(conn)
    return conn


def _register(conn, name, stage=Stage.PAPER):
    repo = SqliteStrategyRepository(conn)
    repo.add(name)
    if stage is not Stage.IDEA:
        conn.execute("UPDATE strategies SET stage = ? WHERE name = ?", (stage.value, name))
        conn.commit()
    return repo.get(name)


def _allocate(conn, rec, capital=1000.0):
    allocate_locked(conn, rec.id, capital, "agent", account_equity=1_000_000.0)
    conn.commit()


def test_an_operational_strategy_without_a_slice_is_flagged(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "algua.db"))
    with closing(_conn()) as conn:
        _register(conn, "stranded", Stage.PAPER)
        payload = book_status(conn, capacity=8)
    assert payload["ok"] is False
    assert [row["strategy"] for row in payload["unallocated_operational"]] == ["stranded"]
    assert payload["unallocated_operational"][0]["ever_ticked"] is False


def test_a_non_operational_strategy_without_a_slice_is_not_flagged(tmp_path, monkeypatch) -> None:
    """A backtested or retired strategy holding no capital is CORRECT, not a fault."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "algua.db"))
    with closing(_conn()) as conn:
        _register(conn, "researching", Stage.BACKTESTED)
        _register(conn, "done", Stage.RETIRED)
        payload = book_status(conn, capacity=8)
    assert payload["unallocated_operational"] == []
    assert payload["ok"] is True


def test_slices_carry_capital_and_last_tick_equity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "algua.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "funded", Stage.PAPER)
        _allocate(conn, rec, 1571.42)
        tick_ts = datetime(2026, 8, 14, 20, 30, tzinfo=UTC).isoformat()
        record_tick_snapshot(
            conn, rec.name, tick_ts=tick_ts, decision_ts=tick_ts, equity=1600.5,
            peak_equity=1600.5, positions={}, n_submitted=0, reconcile_ok=True, lane="paper",
            strategy_id=rec.id, code_hash="c", config_hash="cfg", dependency_hash="d",
            account_id="acct", cash=1600.5, clock_source="broker",
        )
        payload = book_status(conn, capacity=8)
    assert payload["allocated"] == 1
    assert payload["count_headroom"] == 7
    assert payload["sum_allocations"] == 1571.42
    slice_ = payload["slices"][0]
    assert slice_["strategy"] == "funded"
    assert slice_["capital"] == 1571.42
    assert slice_["last_equity"] == 1600.5


def test_a_funded_strategy_is_not_reported_as_stranded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "algua.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "funded", Stage.PAPER)
        _allocate(conn, rec)
        payload = book_status(conn, capacity=8)
    assert payload["unallocated_operational"] == []
    assert payload["ok"] is True


def test_a_revoked_allocation_makes_the_strategy_stranded_again(tmp_path, monkeypatch) -> None:
    """This is the dormant round-trip in miniature: the slice is released, the stage is not."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "algua.db"))
    with closing(_conn()) as conn:
        rec = _register(conn, "benched", Stage.PAPER)
        _allocate(conn, rec)
        conn.execute(
            "UPDATE strategy_allocations SET revoked_ts = ? WHERE strategy_id = ?",
            (datetime.now(UTC).isoformat(), rec.id),
        )
        conn.commit()
        payload = book_status(conn, capacity=8)
    assert [row["strategy"] for row in payload["unallocated_operational"]] == ["benched"]
    assert payload["slices"] == []


def test_capacity_headroom_never_goes_negative(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "algua.db"))
    with closing(_conn()) as conn:
        for n in range(3):
            _allocate(conn, _register(conn, f"s{n}", Stage.PAPER))
        payload = book_status(conn, capacity=1)
    assert payload["allocated"] == 3
    assert payload["count_headroom"] == 0
