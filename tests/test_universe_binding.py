"""Tests for the gate-universe binding resolver (#559): the paper tick's operational universe is
resolved from the newest PASSING gate row's universe_name, never the module's CONFIG template."""
from __future__ import annotations

import pytest

from algua.data.store import DataStore
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.universe_binding import (
    SOURCE_CONFIG_LEGACY,
    SOURCE_GATE,
    resolve_operational_universe,
)

_CONFIG_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


def _repo(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def _gate_row(repo, name: str, *, passed: bool = True, universe_name: str | None = None) -> int:
    sid = repo.get(name).id
    return repo.record_gate_evaluation(
        sid, passed=passed, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=63, min_holdout_observations=63, code_hash="c", config_hash="cfg",
        dependency_hash="d", data_source="SyntheticProvider", snapshot_id=None,
        period_start="2023-01-01", period_end="2023-12-31", holdout_frac=0.2, actor="agent",
        decision_json="{}", universe_name=universe_name)


def _ingest(store: DataStore, universe: str, symbols: list[str], effective_date: str) -> None:
    store.ingest_universe(universe=universe, symbols=symbols, effective_date=effective_date,
                          as_of=f"{effective_date}T00:00:00Z", source="test")


def test_gate_source_resolves_current_membership(tmp_path):
    repo = _repo(tmp_path)
    repo.add("s")
    _gate_row(repo, "s", universe_name="liquid3")
    store = DataStore(tmp_path)
    _ingest(store, "liquid3", ["AAPL", "MSFT", "GOOGL"], "2020-01-01")
    symbols, source = resolve_operational_universe(
        repo._conn, tmp_path, "s", _CONFIG_UNIVERSE)
    assert source == SOURCE_GATE
    assert symbols == ["AAPL", "GOOGL", "MSFT"]  # the GATED universe, not the CONFIG template


def test_gate_source_uses_membership_as_of_today(tmp_path):
    """As-of semantics: the CURRENT membership is the greatest effective_date <= today; a
    future-dated snapshot never leaks in, and an older snapshot is superseded."""
    repo = _repo(tmp_path)
    repo.add("s")
    _gate_row(repo, "s", universe_name="rolling")
    store = DataStore(tmp_path)
    _ingest(store, "rolling", ["AAPL", "MSFT"], "2020-01-01")
    _ingest(store, "rolling", ["MSFT", "GOOGL"], "2021-01-01")   # supersedes 2020
    _ingest(store, "rolling", ["TSLA"], "2999-01-01")            # future: must not apply yet
    symbols, source = resolve_operational_universe(
        repo._conn, tmp_path, "s", _CONFIG_UNIVERSE)
    assert source == SOURCE_GATE
    assert symbols == ["GOOGL", "MSFT"]


def test_gate_source_newest_passing_row_wins(tmp_path):
    """The binding follows the NEWEST passing row (a re-gated strategy re-binds); failing rows
    never bind."""
    repo = _repo(tmp_path)
    repo.add("s")
    _gate_row(repo, "s", universe_name="old_u")
    _gate_row(repo, "s", universe_name="new_u")
    _gate_row(repo, "s", passed=False, universe_name="failed_u")  # newest but FAILED: ignored
    store = DataStore(tmp_path)
    _ingest(store, "new_u", ["IBM", "ORCL"], "2020-01-01")
    symbols, source = resolve_operational_universe(
        repo._conn, tmp_path, "s", _CONFIG_UNIVERSE)
    assert source == SOURCE_GATE
    assert symbols == ["IBM", "ORCL"]


def test_legacy_null_universe_name_falls_back_to_config(tmp_path):
    """A pre-v39 passing row (universe_name NULL) falls back to CONFIG.universe with the loud
    config_legacy source (the caller warns)."""
    repo = _repo(tmp_path)
    repo.add("s")
    _gate_row(repo, "s", universe_name=None)
    symbols, source = resolve_operational_universe(
        repo._conn, tmp_path, "s", _CONFIG_UNIVERSE)
    assert source == SOURCE_CONFIG_LEGACY
    assert symbols == _CONFIG_UNIVERSE


def test_no_passing_gate_row_fails_closed(tmp_path):
    """No passing gate row at all -> LookupError: an unpromoted strategy has no business
    ticking. A FAILING row does not rescue it."""
    repo = _repo(tmp_path)
    repo.add("s")
    with pytest.raises(LookupError, match="no passing gate"):
        resolve_operational_universe(repo._conn, tmp_path, "s", _CONFIG_UNIVERSE)
    _gate_row(repo, "s", passed=False, universe_name="u")
    with pytest.raises(LookupError, match="no passing gate"):
        resolve_operational_universe(repo._conn, tmp_path, "s", _CONFIG_UNIVERSE)


def test_gate_universe_with_no_effective_membership_fails_closed(tmp_path):
    """A gated universe whose only snapshot is future-dated resolves to NO current membership —
    fail closed rather than trading an empty/undefined universe."""
    repo = _repo(tmp_path)
    repo.add("s")
    _gate_row(repo, "s", universe_name="future_only")
    store = DataStore(tmp_path)
    _ingest(store, "future_only", ["TSLA"], "2999-01-01")
    with pytest.raises(LookupError, match="no membership"):
        resolve_operational_universe(repo._conn, tmp_path, "s", _CONFIG_UNIVERSE)
