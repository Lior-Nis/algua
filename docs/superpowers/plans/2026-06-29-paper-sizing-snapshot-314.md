# Per-strategy paper sizing snapshot + virtual NAV (#314) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give paper strategies a per-strategy attributed sizing snapshot + virtual NAV by generalizing `build_live_sizing_snapshot` over `LedgerKind`, so the multi-tenant driver (#316) can size each strategy off its own book.

**Architecture:** Generalize the live sizing-snapshot builder by adding a `kind: LedgerKind` parameter (positions via `believed_positions(conn, strategy, kind)`; the per-symbol PnL fills read from the kind's fills table via a new `fills_table(kind)` accessor). Keep thin `build_live_sizing_snapshot` / `build_paper_sizing_snapshot` aliases (the codebase's existing `believed_positions`/`paper_believed_positions` idiom). Purely additive: live caller + live tests untouched; nothing wired into `run_tick`.

**Tech Stack:** Python 3.12, SQLite (`live_fills` / `paper_venue_fills` from #249), pandas, pytest.

## Global Constraints

- Run everything via `uv run ...`. Quality gate green before every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Ruff line length ≤ 100 columns.
- `git add` only the named files — never `git add -A` (untracked WIP exists in the tree).
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Behavior-preserving for the live lane:** `algua/cli/live_cmd.py:139` keeps calling `build_live_sizing_snapshot` unchanged, and `tests/test_live_sizing.py` must stay green unchanged.
- `fills_table(kind)` returns a controlled module-level constant; f-string interpolation of it in SQL is acceptable and intended.
- Branch: `paper-sizing-snapshot-314` (already created off `main`, which has #249).

---

## File structure

- `algua/execution/live_ledger.py` — **Modify.** Add `fills_table(kind: LedgerKind) -> str`.
- `algua/execution/live_sizing.py` — **Modify.** Generalize the builder to `build_sizing_snapshot(..., kind)`; add `build_live_sizing_snapshot` + `build_paper_sizing_snapshot` aliases.
- `tests/test_live_ledger_ledgerkind.py` — **Modify.** Add a `fills_table` test.
- `tests/test_paper_sizing.py` — **Create.** Paper sizing mirror + live/paper parity test.
- `tests/test_live_sizing.py` — **Unchanged** (regression guard; routes through the alias).

---

### Task 1: `fills_table(kind)` accessor

**Files:**
- Modify: `algua/execution/live_ledger.py`
- Test: `tests/test_live_ledger_ledgerkind.py`

**Interfaces:**
- Consumes: existing `LedgerKind` and the module-private `_TABLES` mapping in `live_ledger.py`.
- Produces: `fills_table(kind: LedgerKind) -> str` returning the kind's fills table name (`"live_fills"` for LIVE, `"paper_venue_fills"` for PAPER).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_live_ledger_ledgerkind.py`:

```python
def test_fills_table_returns_per_kind_table():
    from algua.execution.live_ledger import LedgerKind, fills_table
    assert fills_table(LedgerKind.LIVE) == "live_fills"
    assert fills_table(LedgerKind.PAPER) == "paper_venue_fills"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_live_ledger_ledgerkind.py::test_fills_table_returns_per_kind_table -v`
Expected: FAIL — `ImportError: cannot import name 'fills_table'`.

- [ ] **Step 3: Add the accessor**

In `algua/execution/live_ledger.py`, add this function (place it just after the `_TABLES` mapping is defined):

```python
def fills_table(kind: LedgerKind) -> str:
    """The fills table name for a ledger kind (live_fills / paper_venue_fills). Lets callers read a
    kind's fills without importing the private _TABLES mapping."""
    return _TABLES[kind].fills
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_live_ledger_ledgerkind.py::test_fills_table_returns_per_kind_table -v`
Expected: PASS.

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check algua/execution/live_ledger.py tests/test_live_ledger_ledgerkind.py
uv run mypy algua/execution/live_ledger.py
git add algua/execution/live_ledger.py tests/test_live_ledger_ledgerkind.py
git commit -m "feat(execution): fills_table(kind) accessor on the ledger #314

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Generalize the sizing snapshot by `LedgerKind` + paper alias

**Files:**
- Modify: `algua/execution/live_sizing.py`
- Test: `tests/test_paper_sizing.py` (create); `tests/test_live_sizing.py` (existing — must stay green unchanged)

**Interfaces:**
- Consumes: `fills_table(kind)` (Task 1); `LedgerKind`, `believed_positions(conn, strategy, kind)`, `position_pnl(fills, mark)` from `live_ledger`.
- Produces:
  - `build_sizing_snapshot(conn, strategy, allocation, bars, universe, *, kind: LedgerKind) -> tuple[SizingSnapshot, float]` — the generalized core.
  - `build_live_sizing_snapshot(conn, strategy, allocation, bars, universe) -> tuple[SizingSnapshot, float]` — alias, `kind=LedgerKind.LIVE`.
  - `build_paper_sizing_snapshot(conn, strategy, allocation, bars, universe) -> tuple[SizingSnapshot, float]` — alias, `kind=LedgerKind.PAPER`.
  - `SizingSnapshot` (fields `equity`, `market_values`, `qtys`) and `LiveSizingError` unchanged.

- [ ] **Step 1: Write the failing paper tests**

Create `tests/test_paper_sizing.py` (mirrors `tests/test_live_sizing.py` but writes the paper ledger; plus a live/paper parity test):

```python
import pandas as pd
import pytest

from algua.execution import live_sizing as S
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "s.db")
    migrate(conn)
    return conn


def _paper_fill(conn, aid, strategy, symbol, qty, price):
    conn.execute(
        "INSERT INTO paper_venue_fills"
        "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def _live_fill(conn, aid, strategy, symbol, qty, price):
    conn.execute(
        "INSERT INTO live_fills"
        "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def _bars(close_by_symbol):
    rows = []
    for sym, closes in close_by_symbol.items():
        for i, c in enumerate(closes):
            rows.append({"timestamp": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=i),
                         "symbol": sym, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_paper_nav_and_equity_above_allocation(tmp_path):
    conn = _conn(tmp_path)
    _paper_fill(conn, "a1", "s1", "AAA", 10.0, 100.0)        # long 10 @100
    bars = _bars({"AAA": [100.0, 110.0]})                    # mark 110 -> +100 -> NAV 10_100
    snap, nav = S.build_paper_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                              universe=["AAA"])
    assert nav == 10_100.0
    assert snap.equity == 10_000.0                           # min(allocation, NAV)
    assert snap.qtys["AAA"] == 10.0
    assert snap.market_values["AAA"] == 10.0 * 110.0


def test_paper_equity_derisks_below_allocation(tmp_path):
    conn = _conn(tmp_path)
    _paper_fill(conn, "a1", "s1", "AAA", 10.0, 100.0)
    bars = _bars({"AAA": [100.0, 50.0]})                     # mark 50 -> -500 -> NAV 9_500
    snap, nav = S.build_paper_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                              universe=["AAA"])
    assert nav == 9_500.0
    assert snap.equity == 9_500.0


def test_paper_held_symbol_missing_mark_fails_closed(tmp_path):
    conn = _conn(tmp_path)
    _paper_fill(conn, "a1", "s1", "ZZZ", 5.0, 10.0)          # held ZZZ, no bar -> fail closed
    bars = _bars({"AAA": [100.0]})
    with pytest.raises(S.LiveSizingError, match="mark"):
        S.build_paper_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars, universe=["AAA"])


def test_paper_nav_collapse_fails_closed(tmp_path):
    conn = _conn(tmp_path)
    _paper_fill(conn, "a1", "s1", "AAA", 10.0, 100.0)
    bars = _bars({"AAA": [100.0, 1.0]})                      # NAV 100 - 990 < 0
    with pytest.raises(S.LiveSizingError, match="non-positive"):
        S.build_paper_sizing_snapshot(conn, "s1", allocation=100.0, bars=bars, universe=["AAA"])


def test_live_paper_parity_identical_fills(tmp_path):
    # Identical fills in each lane's own table must yield identical snapshot + NAV.
    conn = _conn(tmp_path)
    _live_fill(conn, "L1", "s1", "AAA", 10.0, 100.0)
    _paper_fill(conn, "P1", "s1", "AAA", 10.0, 100.0)
    bars = _bars({"AAA": [100.0, 110.0]})
    live_snap, live_nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                                       universe=["AAA"])
    paper_snap, paper_nav = S.build_paper_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                                          universe=["AAA"])
    assert live_nav == paper_nav
    assert live_snap == paper_snap
```

- [ ] **Step 2: Run paper tests to verify they fail**

Run: `uv run pytest tests/test_paper_sizing.py -v`
Expected: FAIL — `AttributeError: module 'algua.execution.live_sizing' has no attribute 'build_paper_sizing_snapshot'`.

- [ ] **Step 3: Generalize the builder + add aliases**

In `algua/execution/live_sizing.py`: (a) update the import to add `fills_table`; (b) rename the existing `build_live_sizing_snapshot` to `build_sizing_snapshot` with a keyword-only `kind` parameter, replacing the two live-hardcoded references; (c) add the two aliases.

Change the import line:

```python
from algua.execution.live_ledger import LedgerKind, believed_positions, fills_table, position_pnl
```

Replace the whole `build_live_sizing_snapshot` function with the generalized core plus aliases:

```python
def build_sizing_snapshot(
    conn: sqlite3.Connection,
    strategy: str,
    allocation: float,
    bars: pd.DataFrame,
    universe: list[str],
    *,
    kind: LedgerKind,
) -> tuple[SizingSnapshot, float]:
    held = believed_positions(conn, strategy, kind)  # {symbol: signed qty}, nonzero only
    marks = _latest_marks(bars)
    symbols = set(universe) | set(held)

    nav = allocation
    market_values: dict[str, float] = {}
    qtys: dict[str, float] = {}
    for sym in symbols:
        qty = held.get(sym, 0.0)
        qtys[sym] = qty
        mark = marks.get(sym)
        if qty != 0.0 and (mark is None or mark <= 0.0):
            raise LiveSizingError(
                f"{strategy}: held symbol {sym!r} has no usable mark (got {mark!r}) — refusing to "
                "size on a fail-closed mark"
            )
        market_values[sym] = qty * (mark or 0.0)
        if qty != 0.0:
            fills = [
                (float(r["qty"]), float(r["price"]))
                for r in conn.execute(
                    f"SELECT qty, price FROM {fills_table(kind)} WHERE strategy = ? AND symbol = ? "
                    "ORDER BY fill_ts, id",
                    (strategy, sym),
                )
            ]
            assert mark is not None and mark > 0.0       # guard above already raised otherwise
            pnl = position_pnl(fills, mark=mark)
            nav += pnl.realized + pnl.unrealized

    equity = min(allocation, nav)
    if equity <= 0.0:
        # A non-positive sizing denominator would ZeroDivision / invert weights in run_tick — fail
        # closed (skip the strategy) rather than size off it (codex C1 review).
        raise LiveSizingError(f"{strategy}: NAV {nav:.2f} leaves a non-positive sizing equity")
    return SizingSnapshot(equity=equity, market_values=market_values, qtys=qtys), nav


def build_live_sizing_snapshot(
    conn: sqlite3.Connection,
    strategy: str,
    allocation: float,
    bars: pd.DataFrame,
    universe: list[str],
) -> tuple[SizingSnapshot, float]:
    """The live-lane sizing snapshot (alias over build_sizing_snapshot with LedgerKind.LIVE)."""
    return build_sizing_snapshot(conn, strategy, allocation, bars, universe, kind=LedgerKind.LIVE)


def build_paper_sizing_snapshot(
    conn: sqlite3.Connection,
    strategy: str,
    allocation: float,
    bars: pd.DataFrame,
    universe: list[str],
) -> tuple[SizingSnapshot, float]:
    """The paper-lane sizing snapshot (alias over build_sizing_snapshot with LedgerKind.PAPER)."""
    return build_sizing_snapshot(conn, strategy, allocation, bars, universe, kind=LedgerKind.PAPER)
```

- [ ] **Step 4: Run paper tests + live regression**

Run: `uv run pytest tests/test_paper_sizing.py tests/test_live_sizing.py -v`
Expected: PASS — all paper tests pass, and the existing live tests pass unchanged.

- [ ] **Step 5: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add algua/execution/live_sizing.py tests/test_paper_sizing.py
git commit -m "feat(execution): build_sizing_snapshot(kind) + paper alias — per-strategy paper NAV #314

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §3 `build_sizing_snapshot(..., kind)` + live/paper aliases → Task 2; `fills_table(kind)` accessor → Task 1; §4 paper tests + parity test → Task 2 Step 1; live regression → Task 2 Step 4 (test_live_sizing unchanged). §5 non-goals respected (no run_tick/CLI/tick_snapshots/reconcile/gate/schema change). §6 risk (live behavior-preserving) → live caller + test_live_sizing untouched.
- **Placeholder scan:** none — every code step has full code.
- **Type consistency:** `build_sizing_snapshot(conn, strategy, allocation, bars, universe, *, kind: LedgerKind) -> tuple[SizingSnapshot, float]` defined in Task 2 and the two aliases call it with `kind=` exactly; `fills_table(kind: LedgerKind) -> str` defined in Task 1 and used in Task 2's PnL query; `SizingSnapshot`/`LiveSizingError` names unchanged; the paper test inserts the same `paper_venue_fills` columns confirmed present from #249.
