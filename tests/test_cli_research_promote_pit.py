"""Tests: `research promote` accepts --fundamentals/--news-snapshot and the PIT funnel is
unblocked end-to-end to candidate for BOTH lanes (Task 6, #132)."""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.store import DataStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _seed(data_dir):
    """Seed bars + a news snapshot + a fundamentals snapshot over AAPL/MSFT/NVDA.

    Uses ~330 calendar-day bars so the holdout (0.2 frac) clears the 63-observation power floor
    (a hard gate constant, not CLI-relaxable) and the two walk-forward windows each get >5 bars.

    Returns (bars_snapshot_id, news_snapshot_id, fundamentals_snapshot_id).
    """
    store = DataStore(data_dir)
    n = 330
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    rows = [[t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
            for s in ["AAPL", "MSFT", "NVDA"] for t in idx]
    bars = pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low", "close",
                                       "adj_close", "volume"])
    end_iso = idx[-1].date().isoformat()
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                             end=end_iso, as_of="2026-01-01T00:00:00Z", source="t", frame=bars)
    news = pd.DataFrame([
        {"source": "reuters", "article_id": "art-001", "symbols": ["AAPL", "MSFT"],
         "published_at": pd.Timestamp("2025-01-06T12:00:00Z"),
         "knowable_at": pd.Timestamp("2025-01-06T12:00:00Z"),
         "headline": "AAPL and MSFT report strong quarter"},
        {"source": "reuters", "article_id": "art-002", "symbols": ["NVDA"],
         "published_at": pd.Timestamp("2025-01-07T09:00:00Z"),
         "knowable_at": pd.Timestamp("2025-01-07T09:00:00Z"),
         "headline": "NVDA sets new GPU record"},
        {"source": "reuters", "article_id": "art-003", "symbols": ["AAPL"],
         "published_at": pd.Timestamp("2025-01-08T15:00:00Z"),
         "knowable_at": pd.Timestamp("2025-01-08T15:00:00Z"),
         "headline": "AAPL unveils new product line"},
    ])
    nrec = store.ingest_news(provider="reuters", as_of="2025-02-01T00:00:00Z", frame=news)
    funds = pd.DataFrame(
        [["AAPL", "2024-12-31", "eps_diluted", 5.0, "2024-12-31T00:00:00Z", "v"]],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"])
    frec = store.ingest_fundamentals(provider="v", symbols=["AAPL", "MSFT", "NVDA"],
                                     as_of="2025-01-01T00:00:00Z", source="v", frame=funds)
    return brec.snapshot_id, nrec.snapshot_id, frec.snapshot_id


_RELAX = ["--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
          "--min-pct-positive", "0", "--min-window-sharpe", "-100",
          "--windows", "2", "--n-combos", "9", "--allow-non-pit", "--actor", "human"]
_END = (pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=329)).date().isoformat()
_WINDOW = ["--start", "2025-01-01", "--end", _END]


def _stage(name):
    return json.loads(runner.invoke(app, ["registry", "show", name]).stdout)["stage"]


def _backtest_to_backtested(name, *snapshot_args):
    return runner.invoke(app, ["backtest", "run", name, *snapshot_args, "--register", *_WINDOW])


def _holdout_count(tmp_path):
    conn = sqlite3.connect(tmp_path / "r.db")
    try:
        return conn.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone()[0]
    finally:
        conn.close()


def _latest_gate_snapshots(tmp_path):
    conn = sqlite3.connect(tmp_path / "r.db")
    try:
        return conn.execute(
            "SELECT fundamentals_snapshot, news_snapshot FROM gate_evaluations "
            "ORDER BY id DESC LIMIT 1").fetchone()
    finally:
        conn.close()


def test_news_snapshot_on_plain_strategy_is_misuse(tmp_path):
    bid, nid, _fid = _seed(tmp_path)
    assert _backtest_to_backtested("cross_sectional_momentum", "--snapshot", bid).exit_code == 0
    r = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                            "--snapshot", bid, "--news-snapshot", nid, *_WINDOW, *_RELAX])
    assert r.exit_code != 0, r.stdout
    assert "needs_news" in r.stdout


def test_needs_news_without_snapshot_fails_closed_before_reservation(tmp_path):
    bid, _nid, _fid = _seed(tmp_path)
    assert _backtest_to_backtested(
        "news_coverage_tilt", "--snapshot", bid, "--news-snapshot", _nid).exit_code == 0
    # Promote WITHOUT --news-snapshot: must fail closed.
    r = runner.invoke(app, ["research", "promote", "news_coverage_tilt",
                            "--snapshot", bid, *_WINDOW, *_RELAX])
    assert r.exit_code != 0, r.stdout
    assert "needs_news" in r.stdout
    # The fail-closed must precede any holdout reservation: zero rows.
    assert _holdout_count(tmp_path) == 0


def test_news_lane_unblocked_through_funnel(tmp_path):
    # #132 slice 4: a needs_news strategy is no longer blocked at the PIT gate. `research promote`
    # threads the news snapshot through walk_forward + the holdout into the gate evaluation and
    # records it in provenance. Reaching `candidate` ALSO requires passing the orthogonal #221
    # regime-robustness gate, which this flat synthetic fixture cannot (no vol structure → tertiles
    # collapse, fail-closed by design; that gate is covered by main's own regime tests). So we
    # prove the PIT lane is unblocked THROUGH the funnel — gated only by regime-robustness.
    bid, nid, _fid = _seed(tmp_path)
    assert _backtest_to_backtested(
        "news_coverage_tilt", "--snapshot", bid, "--news-snapshot", nid).exit_code == 0
    r = runner.invoke(app, ["research", "promote", "news_coverage_tilt",
                            "--snapshot", bid, "--news-snapshot", nid, *_WINDOW, *_RELAX,
                            "--new-family", "news_fam"])  # #222: seed the first family (human)
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    # The PIT snapshot flowed into the emitted payload (#132 GATE-2 Fix 1) AND the gate-row
    # provenance — i.e. all the way through walk_forward + the holdout into the recorded evaluation.
    assert payload["news_snapshot"] == nid
    assert payload["fundamentals_snapshot"] is None
    fund_snap, news_snap = _latest_gate_snapshots(tmp_path)
    assert news_snap == nid and fund_snap is None
    # It got PAST the (removed) PIT block, family governance, and breadth, into the holdout + gate
    # evaluation (a holdout was reserved/burned). The sole remaining blocker is the PIT-unrelated
    # #221 regime-robustness gate — proving the funnel is unblocked for the PIT lane.
    assert _holdout_count(tmp_path) >= 1
    assert payload["promoted"] is False
    failing = [c["name"] for c in payload.get("checks", []) if not c["passed"]]
    assert failing == ["regime_robustness"], failing


def test_promote_with_wrong_kind_snapshot_errors_before_reservation(tmp_path):
    """A wrong-kind PIT snapshot (a FUNDAMENTALS id passed to --news-snapshot of a needs_news
    strategy) must fail closed BEFORE the holdout reservation, so a deterministic operator typo
    can never strand a pending reservation row (#132 GATE-2 Fix 2)."""
    bid, _nid, fid = _seed(tmp_path)
    assert _backtest_to_backtested(
        "news_coverage_tilt", "--snapshot", bid, "--news-snapshot", _nid).exit_code == 0
    r = runner.invoke(app, ["research", "promote", "news_coverage_tilt",
                            "--snapshot", bid, "--news-snapshot", fid,  # wrong kind
                            *_WINDOW, *_RELAX])
    assert r.exit_code != 0, r.stdout
    # The explicit pre-reservation guard (names the CLI flag), NOT the downstream read_news error.
    assert "--news-snapshot" in r.stdout
    assert "not 'news'" in r.stdout
    # The fail-closed must precede any holdout reservation: zero rows.
    assert _holdout_count(tmp_path) == 0


def test_fundamentals_lane_unblocked_through_funnel(tmp_path):
    # #132 slice 4: a needs_fundamentals strategy is no longer blocked at the PIT gate; the snapshot
    # threads through walk_forward + the holdout into the gate-row provenance. Reaching `candidate`
    # also needs the orthogonal #221 regime-robustness gate (unreachable on this flat fixture — see
    # the news-lane test). Here we prove the PIT lane is unblocked THROUGH the funnel.
    bid, _nid, fid = _seed(tmp_path)
    assert _backtest_to_backtested(
        "fundamentals_earnings_tilt", "--snapshot", bid,
        "--fundamentals-snapshot", fid).exit_code == 0
    r = runner.invoke(app, ["research", "promote", "fundamentals_earnings_tilt",
                            "--snapshot", bid, "--fundamentals-snapshot", fid, *_WINDOW, *_RELAX,
                            "--new-family", "fund_fam"])  # #222: seed the first family (human)
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["fundamentals_snapshot"] == fid
    assert payload["news_snapshot"] is None
    fund_snap, news_snap = _latest_gate_snapshots(tmp_path)
    assert fund_snap == fid and news_snap is None
    assert _holdout_count(tmp_path) >= 1
    assert payload["promoted"] is False
    failing = [c["name"] for c in payload.get("checks", []) if not c["passed"]]
    assert failing == ["regime_robustness"], failing
