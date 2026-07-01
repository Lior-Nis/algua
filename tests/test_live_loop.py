from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.execution.alpaca_broker import BrokerError, TickSnapshot
from algua.execution.live_ledger import (
    delete_paper_venue_order,
    record_paper_venue_order,
)
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
from algua.registry.db import connect, migrate
from algua.risk.limits import RiskBreach
from algua.strategies.base import LoadedStrategy, StrategyConfig

DATES = [datetime(2023, 1, d, tzinfo=UTC) for d in (2, 3, 4)]


def _bars(symbol_prices):
    rows = []
    for sym, prices in symbol_prices.items():
        for ts, px in zip(DATES, prices, strict=True):
            rows.append({"timestamp": ts, "symbol": sym, "open": px, "high": px,
                         "low": px, "close": px, "adj_close": px, "volume": 1000})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


class _FakeProvider:
    def __init__(self, bars):
        self._bars = bars

    def get_bars(self, symbols, start, end, timeframe):
        return self._bars


class _FakeBroker:
    def __init__(self, positions=None, equity=100_000.0):
        self._positions = pd.Series(positions or {}, dtype="float64")
        self._equity = equity
        self.submitted = []
        self.client_order_ids = []
        self.cancels = 0
        self.snapshots = 0

    def get_positions(self):
        return self._positions

    def cancel_open_orders(self):
        self.cancels += 1

    def snapshot(self, universe):
        self.snapshots += 1
        syms = set(universe) | set(self._positions.index)
        qtys = {s: float(self._positions.get(s, 0.0)) for s in syms}
        # market value priced at $1/share for simplicity (qty == market value)
        return TickSnapshot(equity=self._equity, market_values=dict(qtys), qtys=qtys)

    def submit_sized(self, intent, snap, client_order_id=None, reserve=None):
        if intent.symbol not in snap.qtys:
            raise BrokerError(f"{intent.symbol} not in universe")
        self.submitted.append(intent)
        self.client_order_ids.append(client_order_id)
        return f"order-{len(self.submitted)}"


def _identity(scores, view, params):
    """Test-local construction: the injected scores ARE the target weights, so the precise vector
    under test reaches the risk rails unchanged."""
    return scores


def _strategy(weights, warmup_bars=0):
    cfg = StrategyConfig(
        name="cfg", universe=sorted(weights),
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1,
                                     warmup_bars=warmup_bars),
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    return LoadedStrategy(
        config=cfg, signal_fn=lambda view, params: pd.Series(weights), construct_fn=_identity
    )


class _NoopBroker(_FakeBroker):
    """submit_sized always reports 'noop' (no order reaches the venue), like a sub-min-notional
    delta or a position already at target. It never POSTs — mirrors alpaca_broker returning the
    sentinel before /v2/orders."""

    def submit_sized(self, intent, snap, client_order_id=None, reserve=None):
        self.client_order_ids.append(client_order_id)
        return "noop"


def test_run_tick_fires_on_noop_when_submit_sized_noops():
    # #311: when submit_sized reports noop/skipped, run_tick must invoke on_noop(intent, coid) so
    # the paper lane can retract its crash-safe before_submit phantom — and must NOT call submit.
    broker = _NoopBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    noops: list[tuple[str, str | None]] = []
    submits: list[str] = []
    hooks = TickHooks(
        client_order_id_for=lambda s, ts, sym: f"{s}-{sym}",
        before_submit=lambda intent, coid: None,
        on_noop=lambda intent, coid: noops.append((intent.symbol, coid)),
        on_submitted=lambda rec_: submits.append(rec_.symbol),
    )
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars),
                      DATES[0], DATES[-1], hooks=hooks)
    assert noops == [("AAA", "cfg-AAA")]      # on_noop got the intent + deterministic coid
    assert submits == []                      # a noop is never a real submit
    assert result.submitted == []             # nothing recorded as sent


def test_run_tick_does_not_fire_on_noop_on_real_submit():
    # #311: a genuine submit (real order id) must NOT trigger on_noop — only the noop/skip branch.
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    noops: list[str] = []
    hooks = TickHooks(on_noop=lambda intent, coid: noops.append(intent.symbol))
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars),
                      DATES[0], DATES[-1], hooks=hooks)
    assert noops == []
    assert len(result.submitted) == 1


def test_record_paper_venue_order_returns_true_only_on_fresh_insert(tmp_path):
    # #311: the caller relies on the return value to know whether IT created the row this attempt
    # (safe to retract on a noop) vs. a pre-existing row (a prior run may have POSTed — preserve).
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert record_paper_venue_order(conn, "s", "AAA", "buy", None, "c-1", strategy_id=1) is True
    # same client_order_id again -> INSERT OR IGNORE ignores -> not a fresh insert
    assert record_paper_venue_order(conn, "s", "AAA", "buy", None, "c-1", strategy_id=1) is False
    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM paper_venue_orders WHERE client_order_id = 'c-1'"
    ).fetchone()
    assert rows["n"] == 1  # still exactly one row


def test_delete_paper_venue_order_removes_null_row_but_not_backfilled(tmp_path):
    # #311: delete retracts a phantom intent (NULL broker_order_id) but the belt-and-suspenders
    # guard leaves any row that already carries a broker id (a real accepted order) untouched.
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    record_paper_venue_order(conn, "s", "AAA", "buy", None, "c-null", strategy_id=1)
    record_paper_venue_order(conn, "s", "BBB", "buy", None, "c-real", strategy_id=1)
    conn.execute("UPDATE paper_venue_orders SET broker_order_id = 'o-9' "
                 "WHERE client_order_id = 'c-real'")
    conn.commit()

    delete_paper_venue_order(conn, "c-null")   # phantom -> gone
    delete_paper_venue_order(conn, "c-real")   # backfilled -> preserved by the NULL guard

    remaining = {
        r["client_order_id"]
        for r in conn.execute("SELECT client_order_id FROM paper_venue_orders").fetchall()
    }
    assert remaining == {"c-real"}


def test_run_tick_submits_target_and_cancels_first():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert broker.cancels == 1
    assert broker.snapshots == 1  # #20: ONE snapshot per tick, not per symbol
    assert len(result.submitted) == 1 and result.submitted[0]["symbol"] == "AAA"
    assert result.submitted[0]["order_id"] == "order-1"
    assert result.decision_ts == DATES[-1]


def test_run_tick_exits_dropped_symbol():
    broker = _FakeBroker(positions={"BBB": 10.0})  # held but not in target
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [50.0, 50.0, 50.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    syms = {o["symbol"]: o["target_weight"] for o in result.submitted}
    assert syms["BBB"] == 0.0  # exit order for the dropped name


def test_run_tick_warmup_not_met_submits_nothing():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})  # 3 sessions
    result = run_tick(_strategy({"AAA": 1.0}, warmup_bars=5), broker, _FakeProvider(bars),
                      DATES[0], DATES[-1])
    assert result.submitted == [] and broker.submitted == []


def _bars_n(symbol, n_sessions):
    """Daily bars for one symbol over `n_sessions` distinct closed sessions, flat at $100."""
    dates = [datetime(2023, 2, d, tzinfo=UTC) for d in range(1, n_sessions + 1)]
    rows = [{"timestamp": ts, "symbol": symbol, "open": 100.0, "high": 100.0,
             "low": 100.0, "close": 100.0, "adj_close": 100.0, "volume": 1000} for ts in dates]
    return pd.DataFrame(rows).set_index("timestamp").sort_index(), dates


def test_run_tick_warmup_boundary_matches_backtest_paper_semantic():
    """PIN live's warm-up boundary to the reconciled backtest/paper semantic (#1).

    "warmup_bars = N" holds the first N sessions flat and FIRST DECIDES on session index N
    (the bar that sees N+1 sessions of history) — the SAME boundary as the backtest loop
    (`if i < warmup: continue`, first decision at i == N) and the paper loop
    (`if bars_seen <= warmup: continue`, first decision at session index N). See
    test_decision_parity.test_warmup_means_the_same_number_of_flat_bars_in_both_paths.

    Regression guard against the historical off-by-one: live formerly decided once `nunique()
    reached N` (deciding on session index N-1, one bar early). It must now block while
    `nunique() <= N` and first decide at `nunique() == N+1`.
    """
    N = 5

    # Exactly N closed sessions: still warm-up — the tick must NOT decide (no snapshot, no orders).
    bars_n, dates_n = _bars_n("AAA", N)
    broker = _FakeBroker()
    after = datetime(2023, 2, N + 1, tzinfo=UTC)  # all N sessions are fully closed
    result = run_tick(_strategy({"AAA": 1.0}, warmup_bars=N), broker, _FakeProvider(bars_n),
                      dates_n[0], dates_n[-1], now=after)
    assert result.target_weights == {} and result.submitted == []
    assert broker.snapshots == 0  # warm-up path takes no sizing snapshot / makes no decision

    # N+1 closed sessions: warm-up satisfied — the tick decides on session index N (the latest
    # closed session, the same bar the backtest/paper first evaluate).
    bars_n1, dates_n1 = _bars_n("AAA", N + 1)
    broker = _FakeBroker()
    after = datetime(2023, 2, N + 2, tzinfo=UTC)
    result = run_tick(_strategy({"AAA": 1.0}, warmup_bars=N), broker, _FakeProvider(bars_n1),
                      dates_n1[0], dates_n1[-1], now=after)
    assert result.decision_ts == dates_n1[N]  # session index N (0-based)
    assert len(result.submitted) == 1 and result.submitted[0]["symbol"] == "AAA"


class _DedupBroker(_FakeBroker):
    """Mimics Alpaca's client_order_id de-dup: a resubmit carrying the SAME (non-None)
    client_order_id maps back to the SAME broker_order_id (one logical order), as it does after a
    transient retry or a full tick replay. A MISSING client_order_id gets a FRESH broker id every
    time (no dedup) — exactly as Alpaca would, so the test fails if the deterministic id ever stops
    being threaded through the loop."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._by_coid = {}
        self._counter = 0

    def submit_sized(self, intent, snap, client_order_id=None, reserve=None):
        if intent.symbol not in snap.qtys:
            raise BrokerError(f"{intent.symbol} not in universe")
        self.client_order_ids.append(client_order_id)
        if client_order_id is not None and client_order_id in self._by_coid:
            return self._by_coid[client_order_id]      # dedup: same order, same broker id
        self._counter += 1
        oid = f"order-{self._counter}"
        if client_order_id is not None:                # None -> no mapping stored -> never dedups
            self._by_coid[client_order_id] = oid
        self.submitted.append(intent)
        return oid


def test_live_replay_record_paper_venue_order_is_idempotent(tmp_path):
    # #166 gap 3 / #249: the LIVE replay equivalent of order_state idempotency. A crash/retry
    # replays the same tick: the deterministic client_order_id makes Alpaca return the SAME
    # broker_order_id, and on_submitted -> record_paper_venue_order must INSERT OR IGNORE (on the
    # client_order_id UNIQUE index) so the replay leaves exactly one paper_venue_orders row (not a
    # duplicate). Covers the live loop wiring, not just the bare DB writer.
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    broker = _DedupBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    strat = _strategy({"AAA": 1.0})
    now = datetime(2023, 1, 5, tzinfo=UTC)   # all three sessions fully closed -> stable decision_ts
    hooks = TickHooks(
        client_order_id_for=lambda s, ts, sym: f"{s}-{sym}",  # deterministic across the replay
        on_submitted=lambda o: record_paper_venue_order(
            conn, strat.name, o.symbol, o.side, o.target_weight,
            o.client_order_id, strategy_id=1,
        ),
    )

    for _ in range(2):  # run the SAME tick twice (replay)
        run_tick(strat, broker, _FakeProvider(bars), DATES[0], DATES[-1], now=now, hooks=hooks)

    # The deterministic id was threaded on BOTH runs — the precondition for Alpaca's dedup. If the
    # loop ever stopped sending it, _DedupBroker would mint two distinct ids -> two rows -> failure.
    assert broker.client_order_ids == ["cfg-AAA", "cfg-AAA"]
    rows = conn.execute(
        "SELECT client_order_id FROM paper_venue_orders WHERE strategy = ?", (strat.name,)
    ).fetchall()
    assert len(rows) == 1 and rows[0]["client_order_id"] == "cfg-AAA"  # one row, not duplicated


def test_run_tick_live_snapshot_equity_flows_into_order_notional(monkeypatch):
    # #166 gap 4: the ledger-backed sizing equity supplied via live_snapshot must reach order
    # SIZING in the loop, not just the snapshot/drawdown checks. A buy is sized off the derealized
    # 8k equity, and the account-equity path (broker.snapshot) is NOT consulted. Uses the REAL
    # AlpacaPaperBroker sizing path so the notional is genuinely computed, not stubbed.
    from algua.execution import alpaca_broker as ab
    from algua.execution.alpaca_broker import AlpacaPaperBroker

    class _Resp:
        def __init__(self, code, payload):
            self.status_code = code
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

    class _RecordingRequests:
        def __init__(self):
            self.posted = []

        def get(self, url, headers=None, timeout=None):
            return _Resp(200, [])

        def post(self, url, headers=None, json=None, timeout=None):
            self.posted.append(json)
            return _Resp(201, {"id": "o1"})

        def delete(self, url, headers=None, timeout=None):
            return _Resp(200, [])

    fake = _RecordingRequests()
    monkeypatch.setattr(ab, "requests", fake)
    broker = AlpacaPaperBroker(api_key="k", api_secret="s")
    # Guard: the live path must size off the hook's snapshot, never fall back to account equity.
    monkeypatch.setattr(broker, "snapshot", lambda universe: (_ for _ in ()).throw(
        AssertionError("run_tick must use live_snapshot, not broker.snapshot()")))

    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    # Derealized ledger snapshot: equity 8k (e.g. min(10k allocation, 8k NAV)), flat book.
    sizing_snap = TickSnapshot(equity=8_000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    hooks = TickHooks(live_snapshot=lambda b: (sizing_snap, 8_000.0))
    run_tick(_strategy({"AAA": 0.5}), broker, _FakeProvider(bars), DATES[0], DATES[-1], hooks=hooks)
    # 0.5 * 8000 derealized = 4000.00 — NOT sized off any 10k allocation / account equity.
    assert fake.posted[0]["notional"] == "4000.00"


def test_run_tick_gross_breach_raises():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0]})
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0, "BBB": 1.0}), broker, _FakeProvider(bars),
                 DATES[0], DATES[-1])
    assert ei.value.kind == "gross_exposure"


def test_run_tick_drops_partial_current_session_bar():
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})  # sessions Jan 2,3,4 2023
    # now = Jan 4 -> that session is "today" (possibly partial); decide on Jan 3 instead.
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                      now=datetime(2023, 1, 4, 12, 0, tzinfo=UTC))
    assert result.decision_ts == DATES[1]


def test_run_tick_persists_each_order_immediately_with_client_order_id():
    # #18: on_submitted fires per accepted order (before the next submit), carrying a deterministic
    # client_order_id passed through to the broker.
    broker = _FakeBroker(positions={"BBB": 10.0})
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [50.0, 50.0, 50.0]})
    persisted = []
    hooks = TickHooks(
        client_order_id_for=lambda strat, ts, sym: f"{strat}-{sym}",
        on_submitted=persisted.append,
    )
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                      hooks=hooks)
    assert {r.symbol for r in persisted} == {o["symbol"] for o in result.submitted}
    assert all(isinstance(r, SubmittedOrder) for r in persisted)
    assert all(c is not None for c in broker.client_order_ids)  # coid threaded to the broker


def test_run_tick_halts_before_submit_when_switch_trips():
    # #21: should_halt() true => abort before any cancel/submit.
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(should_halt=lambda: True)
    with pytest.raises(TickHalted):
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks)
    assert broker.cancels == 0 and broker.submitted == []  # nothing sent


def test_run_tick_drawdown_breach_halts_before_trading():
    # #27: equity below the persisted peak by more than max_drawdown trips before any order.
    broker = _FakeBroker(equity=80_000.0)
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(peak_equity=100_000.0)
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks, max_drawdown=0.1)  # 20% drop > 10%
    assert ei.value.kind == "drawdown"
    assert broker.cancels == 0 and broker.submitted == []


@pytest.mark.parametrize("equity", [0.0, -500.0, float("nan")])
def test_run_tick_non_positive_equity_breaches_before_trading(equity):
    # #162: a non-usable sizing denominator must trip BEFORE the mv/equity division. equity == 0
    # ZeroDivisions mid-tick; equity < 0 silently flips every current weight's sign and trades
    # against inverted holdings; NaN (e.g. a bad mark) poisons every weight to a silent no-op and
    # slips a bare `<= 0` guard. Hold a position so the (dangerous) sign-flip path is exercised.
    broker = _FakeBroker(positions={"AAA": 10.0}, equity=equity)
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert ei.value.kind == "non_positive_equity"
    assert broker.cancels == 0 and broker.submitted == []  # halted before any cancel/order


def test_run_tick_reconcile_mismatch_raises():
    # #18/#249: venue_belief disagrees with the broker's pre-submit book — halts the tick.
    broker = _FakeBroker(positions={"AAA": 10.0})
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(venue_belief=lambda: {"AAA": 999.0})  # belief says far more than broker holds
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks)
    assert ei.value.kind == "reconcile"


def test_run_tick_reconcile_empty_db_vs_held_broker_raises():
    # #18/#249 drift: the belief is empty while the broker still holds positions.
    # Supplying the hook (even empty) must reconcile and halt, not skip on empty.
    broker = _FakeBroker(positions={"AAA": 10.0})
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(venue_belief=lambda: {})  # belief says flat, broker holds 10 -> drift
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks)
    assert ei.value.kind == "reconcile"
    assert broker.cancels == 0 and broker.submitted == []  # halted before any order


def test_run_tick_no_reconcile_hook_does_not_compare():
    # No venue_belief hook supplied: the loop must not attempt reconcile (pure decide+submit path)
    # and trades normally.
    broker = _FakeBroker(positions={"AAA": 10.0})
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert result.reconcile_ok is True
    assert broker.cancels == 1


def test_run_tick_halts_after_cancel_before_submit_when_switch_trips():
    # #21: the kill-switch trips after cancel is already in flight. The second should_halt() check
    # (after cancel, before the submit loop) must abort before any order is sent.
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    calls = {"n": 0}

    def _halt():
        calls["n"] += 1
        return calls["n"] >= 2  # first check (pre-cancel) passes; second (post-cancel) trips

    hooks = TickHooks(should_halt=_halt)
    with pytest.raises(TickHalted):
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks)
    assert broker.cancels == 1  # cancel ran
    assert broker.submitted == []  # but nothing was submitted


def test_run_tick_realized_gross_breach_trips_before_submit():
    # #27: the broker book is already over the realized gross limit BEFORE this tick. The realized
    # check must trip/flatten before any new order is sent, not after.
    broker = _FakeBroker(positions={"AAA": 100_000.0, "BBB": 100_000.0}, equity=100_000.0)
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0]})
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert ei.value.kind == "gross_exposure_realized"
    assert broker.cancels == 0 and broker.submitted == []  # tripped before cancel + submit


def test_run_tick_ratchets_peak_equity():
    broker = _FakeBroker(equity=120_000.0)
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                      hooks=TickHooks(peak_equity=100_000.0))
    assert result.peak_equity == 120_000.0  # new high


def test_run_tick_result_carries_equity():
    broker = _FakeBroker(equity=123_456.0)  # snapshot() returns TickSnapshot(equity=self._equity)
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1])
    assert result.equity == 123_456.0  # the tick's snapshot equity is surfaced on the result


def test_run_tick_should_halt_aborts_between_orders():
    from algua.live.live_loop import TickHalted, TickHooks
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0]})
    calls = {"n": 0}

    def should_halt():
        calls["n"] += 1
        # False for pre-loop checks (x2) + 1st order; True before the 2nd.
        return calls["n"] > 3

    hooks = TickHooks(should_halt=should_halt)
    with pytest.raises(TickHalted):
        run_tick(_strategy({"AAA": 0.5, "BBB": 0.5}), broker, _FakeProvider(bars),
                 DATES[0], DATES[-1], hooks=hooks)
    assert len(broker.submitted) == 1  # only the first order went out (adapt attr name if needed)


def test_run_tick_uses_cancel_hook_when_supplied():
    from algua.live.live_loop import TickHooks
    broker = _FakeBroker()
    called = {"scoped": 0, "account_wide": 0}
    broker.cancel_open_orders = lambda: called.__setitem__("account_wide",
                                                           called["account_wide"] + 1)
    hooks = TickHooks(cancel=lambda: called.__setitem__("scoped", called["scoped"] + 1))
    run_tick(_strategy({"AAA": 0.5}), broker, _FakeProvider(_bars({"AAA": [100.0, 100.0, 100.0]})),
             DATES[0], DATES[-1], hooks=hooks)
    assert called == {"scoped": 1, "account_wide": 0}  # the hook replaced the account-wide cancel


def test_run_tick_live_snapshot_sizes_off_hook_and_drawdowns_off_nav(monkeypatch):
    from algua.execution.alpaca_broker import TickSnapshot
    from algua.live.live_loop import RiskBreach, TickHooks
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    # ledger snapshot: allocation 10k as equity, flat; NAV 6k vs peak 10k -> 40% drawdown trips
    snap = TickSnapshot(equity=10_000.0, market_values={"AAA": 0.0}, qtys={"AAA": 0.0})
    hooks = TickHooks(live_snapshot=lambda b: (snap, 6_000.0), peak_equity=10_000.0)
    with pytest.raises(RiskBreach):  # NAV 6k vs peak 10k = 40% drawdown
        run_tick(_strategy({"AAA": 0.5}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=hooks, max_drawdown=0.2)


def test_run_tick_live_positions_on_warmup_early_return():
    from algua.live.live_loop import TickHooks
    broker = _FakeBroker()
    # 3 bars, warmup_bars=5 -> nunique() (3) <= warmup (5) -> warmup early-return
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(live_positions=lambda: {"AAA": 7.0})
    res = run_tick(_strategy({"AAA": 0.5}, warmup_bars=5), broker, _FakeProvider(bars),
                   DATES[0], DATES[-1], hooks=hooks)
    assert res.positions_before == {"AAA": 7.0}  # ledger positions, not broker's


def test_run_tick_threads_reserve_buy_to_submit_sized():
    from algua.live.live_loop import TickHooks
    seen = {}
    broker = _FakeBroker()
    orig = broker.submit_sized
    broker.submit_sized = lambda intent, snap, coid=None, reserve=None: (
        seen.__setitem__("reserve", reserve) or orig(intent, snap, coid))
    hooks = TickHooks(reserve_buy=lambda sym, n: n)
    run_tick(_strategy({"AAA": 0.5}), broker, _FakeProvider(_bars({"AAA": [100.0, 100.0, 100.0]})),
             DATES[0], DATES[-1], hooks=hooks)
    assert seen["reserve"] is hooks.reserve_buy   # run_tick forwarded the hook


def test_before_submit_fires_before_submit():
    # #249: before_submit hook fires with (intent, coid) BEFORE broker.submit_sized is called;
    # the ordering guarantee is what makes crash-safe intent recording possible.
    calls = []
    broker = _FakeBroker()
    orig_submit = broker.submit_sized
    broker.submit_sized = lambda intent, snap, coid=None, reserve=None: (
        calls.append(("submit", coid)) or orig_submit(intent, snap, coid)
    )
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    hooks = TickHooks(
        client_order_id_for=lambda s, t, sym: "cid",
        before_submit=lambda intent, coid: calls.append(("before", coid)),
    )
    run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
             hooks=hooks)
    assert calls and calls[0] == ("before", "cid")         # before_submit fired first
    assert ("submit", "cid") in calls                      # broker submit also ran
    assert calls.index(("before", "cid")) < calls.index(("submit", "cid"))  # strict ordering


def test_reconcile_tolerates_fractional_residual():
    # #249: venue_belief {AAA: 5.0}; broker snapshot qtys {AAA: 5.0 + 4e-7} -> within 1e-6 ->
    # no breach. Floating-point residuals from fill arithmetic must not trip a false positive.
    broker = _FakeBroker(positions={"AAA": 5.0 + 4e-7}, equity=100_000.0)
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    result = run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                      hooks=TickHooks(venue_belief=lambda: {"AAA": 5.0}))
    assert result.reconcile_ok is True


def test_reconcile_trips_on_unexplained_holding():
    # #249: venue_belief {} (nothing attributed); broker holds {AAA: 5.0} -> drift > tol ->
    # RiskBreach('reconcile'). Supplying the hook even with an empty belief must reconcile.
    broker = _FakeBroker(positions={"AAA": 5.0}, equity=100_000.0)
    bars = _bars({"AAA": [100.0, 100.0, 100.0]})
    with pytest.raises(RiskBreach) as ei:
        run_tick(_strategy({"AAA": 1.0}), broker, _FakeProvider(bars), DATES[0], DATES[-1],
                 hooks=TickHooks(venue_belief=lambda: {}))
    assert ei.value.kind == "reconcile"
