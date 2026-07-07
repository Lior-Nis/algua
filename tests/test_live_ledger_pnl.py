from algua.execution.live_ledger import LedgerKind, position_pnl, strategy_cash_credit
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "s.db")
    migrate(conn)
    return conn


def _fill(conn, aid, strategy, symbol, qty, price, ts="2026-06-06T00:00:00+00:00"):
    conn.execute(
        "INSERT INTO live_fills"
        "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, ts),
    )
    conn.commit()


def _activity(conn, aid, type_, symbol, amount, ts="2026-06-06T00:00:00+00:00"):
    conn.execute(
        "INSERT INTO live_activities(activity_id, type, symbol, amount, ts, raw) "
        "VALUES (?,?,?,?,?,?)",
        (aid, type_, symbol, amount, ts, "{}"),
    )
    conn.commit()


def test_cash_credit_single_holder_gets_full_amount(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0)          # only s1 traded AAA
    _activity(conn, "d1", "DIV", "AAA", 25.0)
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 25.0


def test_cash_credit_splits_long_credit_by_long_shares(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0)          # long 10
    _fill(conn, "f2", "s2", "AAA", 30.0, 100.0)          # long 30
    _activity(conn, "d1", "DIV", "AAA", 40.0)            # +40 credit split across LONGS 10:30
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 10.0
    assert strategy_cash_credit(conn, "s2", LedgerKind.LIVE) == 30.0


def test_short_position_is_debited_on_dividend(tmp_path):
    # #437: a short owes the dividend — the broker books a NEGATIVE DIV amount, which must land as a
    # NEGATIVE credit (a debit) on the short's NAV, never a positive credit.
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", -20.0, 100.0)         # short 20
    _activity(conn, "d1", "DIV", "AAA", -50.0)           # short-dividend debit
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == -50.0


def test_offsetting_book_both_sides_signed_and_nonzero(tmp_path):
    # #437: an offsetting long/short book must NOT cancel. The broker books the long credit and the
    # short debit as two separate signed rows; each side is attributed independently, so the long is
    # credited (+) and the short debited (-) — both nonzero, oppositely signed. The old net-pooling
    # split collapsed these to ~0.
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s_long", "AAA", 100.0, 100.0)     # long 100
    _fill(conn, "f2", "s_short", "AAA", -100.0, 100.0)   # short 100
    _activity(conn, "d1", "DIV", "AAA", 200.0)           # +200 long-side credit
    _activity(conn, "d2", "DIV", "AAA", -200.0)          # -200 short-side debit
    assert strategy_cash_credit(conn, "s_long", LedgerKind.LIVE) == 200.0
    assert strategy_cash_credit(conn, "s_short", LedgerKind.LIVE) == -200.0


def test_credit_is_deterministic_under_later_unrelated_fills(tmp_path):
    # #437: a computed historical credit must not drift as unrelated FUTURE fills accumulate. The
    # entitled share base is the position as of the dividend's own date, so fills stamped after the
    # ex/activity date never change a past dividend's attribution.
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0, ts="2026-06-01T00:00:00+00:00")
    _activity(conn, "d1", "DIV", "AAA", 30.0, ts="2026-06-02T00:00:00+00:00")
    before = strategy_cash_credit(conn, "s1", LedgerKind.LIVE)
    assert before == 30.0
    # later, unrelated fills (after the dividend date), including a second strategy piling into AAA
    _fill(conn, "f2", "s1", "AAA", 90.0, 110.0, ts="2026-07-01T00:00:00+00:00")
    _fill(conn, "f3", "s2", "AAA", 500.0, 110.0, ts="2026-07-01T00:00:00+00:00")
    after = strategy_cash_credit(conn, "s1", LedgerKind.LIVE)
    assert after == before                                # unchanged — bounded to the dividend date


def test_cash_credit_excludes_non_dividend_and_untraded(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0)          # s1 only traded AAA
    _activity(conn, "i1", "INT", None, 5.0)              # symbol-less account cash -> excluded
    _activity(conn, "j1", "JNLC", "AAA", 99.0)           # non-DIV symbol cash -> excluded
    _activity(conn, "d1", "DIV", "BBB", 7.0)             # symbol s1 never traded -> excluded
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 0.0


def test_same_day_fill_after_dividend_is_entitled(tmp_path):
    # #437 (GATE-2 finding #1): the entitlement bound is DELIBERATELY date-level, not
    # full-timestamp, because broker DIV rows carry a date-only `date` field. A fill placed the SAME
    # DAY the dividend posts — even one stamped strictly AFTER the DIV row's own instant — shares
    # its date and is therefore treated as entitled. This asserts that DISCLOSED behavior
    # (design-doc limitation #3) so it can't regress silently.
    conn = _conn(tmp_path)
    _activity(conn, "d1", "DIV", "AAA", 30.0, ts="2026-06-06")   # date-only DIV, as brokers book it
    # the ONLY holder's buy is stamped later the same day than the DIV's date bound
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0, ts="2026-06-06T15:30:00+00:00")
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 30.0


def test_dividend_with_null_ts_is_residual(tmp_path):
    # #437 (GATE-2 finding #2): a DIV row with a NULL ts carries no trustworthy entitlement window,
    # so it fails closed to an unattributed account-level residual (credit 0.0) rather than being
    # silently zero-credited by an empty-string fallback OR crediting an unbounded all-fills window.
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0)
    _activity(conn, "d1", "DIV", "AAA", 30.0, ts=None)
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 0.0


def test_dividend_with_malformed_ts_is_residual(tmp_path):
    # #437 (GATE-2 finding #2): a non-ISO-parseable ts is the dangerous case — as a raw string it
    # sorts BELOW every real fill date ("not-a-date" vs "2026-…"), which a naive substr comparison
    # would read as "every fill is on-or-before the dividend" and credit the whole book. The parse
    # guard fails it closed to an account-level residual (credit 0.0).
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0)
    _activity(conn, "d1", "DIV", "AAA", 30.0, ts="not-a-date")
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 0.0


def test_dividend_with_valid_date_prefix_malformed_suffix_is_residual(tmp_path):
    # #437 (GATE-2 finding #2, follow-up): a ts whose first 10 chars are a valid date but whose
    # SUFFIX is garbage ("2026-06-06 garbage") must NOT be accepted on its clean prefix. The guard
    # validates the WHOLE string (date-only OR full ISO datetime), never a prefix slice, so this
    # fails closed to an account-level residual (credit 0.0) exactly like a wholly-malformed ts.
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0, ts="2026-06-06T00:00:00+00:00")
    _activity(conn, "d1", "DIV", "AAA", 30.0, ts="2026-06-06 garbage")
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 0.0
    # a genuine full ISO datetime with the same date IS accepted (bounds to its date) — proving the
    # residual above is the malformed suffix, not the date itself.
    _activity(conn, "d2", "DIV", "AAA", 30.0, ts="2026-06-06T13:00:00+00:00")
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 30.0


def test_fill_with_malformed_ts_is_excluded_from_entitled_base(tmp_path):
    # #437 (GATE-2): the entitlement guard must be applied to each FILL's fill_ts too, not just the
    # DIV row's ts. A fill with an empty/malformed fill_ts sorts BELOW every real date under a raw
    # `substr(fill_ts,1,10) <= as_of` comparison, so it would ALWAYS satisfy '<=' and be counted as
    # entitled — inflating (or fabricating) the base. It must instead be EXCLUDED, exactly like
    # a malformed DIV ts fails closed. Here the only fill for AAA has an empty fill_ts, so the
    # entitled base is empty and the DIV stays an unattributed account-level residual (credit 0.0).
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0, ts="")           # un-datable fill_ts
    _activity(conn, "d1", "DIV", "AAA", 30.0, ts="2026-06-06")
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 0.0
    # a wholly non-ISO fill_ts is likewise excluded (not silently on-or-before the dividend)
    _fill(conn, "f2", "s1", "AAA", 10.0, 100.0, ts="not-a-date")
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 0.0
    # control: a well-formed same-symbol fill on-or-before the dividend IS entitled and gets it all
    _fill(conn, "f3", "s1", "AAA", 10.0, 100.0, ts="2026-06-05T00:00:00+00:00")
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 30.0


def test_non_finite_dividend_amount_fails_closed(tmp_path):
    # #437 (GATE-2, codex #3): a corrupt (non-finite) DIV amount must not propagate inf/nan into NAV
    # and the persisted drawdown peak — it fails closed to no credit.
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0)
    _activity(conn, "d1", "DIV", "AAA", float("inf"))
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 0.0
    _activity(conn, "d2", "DIV", "AAA", float("nan"))
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 0.0


def test_long_then_partial_close_realizes():
    # buy 10@100, buy 10@110 -> avg 105, qty 20; sell 5@120 -> realized 5*(120-105)=75
    fills = [(10.0, 100.0), (10.0, 110.0), (-5.0, 120.0)]
    r = position_pnl(fills, mark=120.0)
    assert r.qty == 15.0
    assert r.avg_cost == 105.0
    assert r.realized == 75.0
    assert r.unrealized == 15.0 * (120.0 - 105.0)


def test_flip_long_to_short():
    # buy 10@100; sell 15@120 -> close 10 (realize 10*(120-100)=200), open short 5@120
    fills = [(10.0, 100.0), (-15.0, 120.0)]
    r = position_pnl(fills, mark=130.0)
    assert r.qty == -5.0
    assert r.avg_cost == 120.0
    assert r.realized == 200.0
    # short unrealized = (avg-mark)*|qty| = (120-130)*5 = -50; (mark-avg)*qty == (130-120)*-5 == -50
    assert r.unrealized == -50.0


def test_short_then_cover():
    # sell 10@100 (short); buy 4@90 -> realize 4*(100-90)=40 covering short
    fills = [(-10.0, 100.0), (4.0, 90.0)]
    r = position_pnl(fills, mark=95.0)
    assert r.qty == -6.0
    assert r.avg_cost == 100.0
    assert r.realized == 40.0


def test_flat_is_zero():
    r = position_pnl([], mark=100.0)
    assert r.qty == 0.0 and r.realized == 0.0 and r.unrealized == 0.0
