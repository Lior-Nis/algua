"""``BacktestReturnsLedger`` — backtest return-series persistence (#222, Task 7)."""
from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from algua.registry.store._util import _now

if TYPE_CHECKING:
    import pandas as pd


class BacktestReturnsLedgerMixin:
    _conn: sqlite3.Connection

    def persist_backtest_returns(
        self,
        strategy_name: str,
        period_start: str,
        period_end: str,
        returns: pd.Series,
    ) -> int:
        """Persist a backtest return series as JSON [[date_str, float], ...]. Returns row id."""
        pairs = [
            [idx.isoformat() if hasattr(idx, "isoformat") else str(idx), float(v)]
            for idx, v in returns.items()
        ]
        blob = json.dumps(pairs)
        now = _now()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO backtest_returns"
                " (strategy_name, period_start, period_end, returns_json, created_at)"
                " VALUES (?,?,?,?,?)",
                (strategy_name, period_start, period_end, blob, now),
            )
        rowid = cur.lastrowid
        assert rowid is not None
        return rowid

    def load_backtest_returns(self, strategy_name: str) -> pd.Series | None:
        """Load the most recent return series for a strategy, or None."""
        import pandas as pd

        row = self._conn.execute(
            "SELECT returns_json FROM backtest_returns WHERE strategy_name = ?"
            " ORDER BY created_at DESC, id DESC LIMIT 1",
            (strategy_name,),
        ).fetchone()
        if row is None:
            return None
        pairs = json.loads(row["returns_json"])
        if not pairs:
            return None
        dates, values = zip(*pairs, strict=True)
        return pd.Series(values, index=pd.to_datetime(dates), dtype=float)
