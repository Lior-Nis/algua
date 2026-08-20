"""Backtest-returns context: ``backtest_returns``.

The daily return series used for return-correlation family clustering (#222), operated by
``algua/registry/store/backtest_returns.py``, with its two #524 append-only triggers. The shared
rationale for those triggers is the comment block above the family triggers in ``family.py``.
"""
from __future__ import annotations

SCHEMA = """
-- v26 (#222): backtest_returns stores daily return series for return-correlation clustering.
CREATE TABLE IF NOT EXISTS backtest_returns (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name  TEXT NOT NULL,
    period_start   TEXT NOT NULL,
    period_end     TEXT NOT NULL,
    returns_json   BLOB NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_backtest_returns_strategy ON backtest_returns(strategy_name);
CREATE TRIGGER IF NOT EXISTS trg_backtest_returns_append_only_upd BEFORE UPDATE ON backtest_returns
  BEGIN SELECT RAISE(ABORT, 'backtest_returns is append-only (#524)'); END;
CREATE TRIGGER IF NOT EXISTS trg_backtest_returns_append_only_del BEFORE DELETE ON backtest_returns
  BEGIN SELECT RAISE(ABORT, 'backtest_returns is append-only (#524)'); END;
"""
