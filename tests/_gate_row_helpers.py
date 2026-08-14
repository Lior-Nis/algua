"""Shared test helper: seed a PASSING gate_evaluations row for a strategy (#559).

Since the paper tick binds its operational universe to the newest passing gate row
(`resolve_operational_universe`), a test strategy that reaches the tick path needs a passing gate
row on record. `universe_name=None` seeds a legacy pre-v39 row: the binding falls back to the
module's CONFIG.universe (`config_legacy`) — the pre-#559 behaviour most tick tests assume.
"""
from __future__ import annotations

from contextlib import closing

from algua.config.settings import get_settings
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository


def seed_passing_gate(name: str, *, universe_name: str | None = None) -> int:
    """Insert a passing gate row for `name` in the settings-resolved registry DB."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        sid = repo.get(name).id
        return repo.record_gate_evaluation(
            sid, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
            funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
            holdout_n_bars=63, min_holdout_observations=63, code_hash="c", config_hash="cfg",
            dependency_hash="d", data_source="SyntheticProvider", snapshot_id=None,
            period_start="2023-01-01", period_end="2023-12-31", holdout_frac=0.2, actor="agent",
            decision_json="{}", universe_name=universe_name)
