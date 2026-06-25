# Survivorship-free 30-year backtests (#212) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 30-year backtests survivorship-bias-free — a bulk PIT-constituents importer, a delisting-aware exit overlay (forced liquidation on confirmed delisting records; fail-closed on unexplained terminal gaps; NaN-poison kill), and minimal delisting-record ingestion.

**Architecture:** Reuse the existing PIT universe machinery (`ingest_universe`/`read_universe`/`_members_as_of`). Add a pure constituents transformer feeding `ingest_universe`; a pure execution-grid overlay `apply_delisting_exits` applied inside `simulate()` right before vectorbt `from_orders`; and a delistings snapshot type mirrored on the universe ingest path. The overlay never invents an exit: it only forces a liquidation for a *held* position past its last real bar, and only with a confirmed record (else fail-closed), with a human-only relaxation. A runtime post-sim check backstops the target-state detection proxy.

**Tech Stack:** Python 3.12, pandas, vectorbt, typer (CLI), pytest. Quality gate (run between tasks): `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

**Design spec:** `docs/superpowers/specs/2026-06-15-survivorship-free-backtests-212-design.md` (GATE-1 approved).

---

## File structure

- **Create** `algua/data/constituents.py` — pure transformer: parse/validate constituent intervals → minimal membership timeline. No I/O.
- **Create** `algua/backtest/delisting.py` — `DelistingRecord`, `DelistingExitError`, pure `apply_delisting_exits`. No I/O.
- **Modify** `algua/data/models.py` — add `Dataset.DELISTINGS`, `Kind.DELISTING`.
- **Modify** `algua/data/manifest.py` — `append_if_absent(..., conflict_check=None)` hook runs under the flock.
- **Modify** `algua/data/store.py` — `ingest_universe(..., require_immutable=False)`, `ingest_delistings`, `read_delistings`.
- **Modify** `algua/backtest/engine.py` — `simulate(..., delisting_records=None, assume_terminal_last_close=False)`: apply overlay, conditional `call_seq="auto"`, runtime post-sim guarantee; thread through `run`.
- **Modify** `algua/backtest/walkforward.py`, `algua/backtest/sweep.py` — thread the two new params.
- **Modify** `algua/cli/_common.py` — `resolve_delisting_inputs`.
- **Modify** `algua/cli/data_cmd.py` — `data import-universe`, `data import-delistings`.
- **Modify** `algua/cli/backtest_cmd.py`, `algua/cli/research_cmd.py` — `--delistings`, human-only `--assume-terminal-last-close`.
- **Tests** under `tests/` mirroring each module.

Tasks are ordered so each produces independently-testable, committable software. Component C ingestion (Tasks 8–9) is built before the CLI thread (Task 10) but after the engine overlay (Tasks 5–7) so the overlay is provable in isolation first.

---

## Task 1: Constituents transformer (pure)

**Files:**
- Create: `algua/data/constituents.py`
- Test: `tests/data/test_constituents.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_constituents.py
from datetime import date

import pytest

from algua.data.constituents import (
    ConstituentInterval,
    constituents_to_snapshots,
    parse_constituents_rows,
)


def test_open_interval_is_a_survivor():
    ivs = [ConstituentInterval("AAPL", date(1998, 1, 2), None)]
    assert constituents_to_snapshots(ivs) == [(date(1998, 1, 2), frozenset({"AAPL"}))]


def test_closed_interval_drops_at_drop_date_exclusive():
    ivs = [ConstituentInterval("ENRN", date(1998, 1, 2), date(2001, 11, 28))]
    tl = constituents_to_snapshots(ivs)
    assert tl == [
        (date(1998, 1, 2), frozenset({"ENRN"})),
        (date(2001, 11, 28), frozenset()),
    ]


def test_simultaneous_add_and_drop_is_one_snapshot():
    ivs = [
        ConstituentInterval("A", date(2000, 1, 1), date(2005, 1, 1)),
        ConstituentInterval("B", date(2005, 1, 1), None),
    ]
    tl = constituents_to_snapshots(ivs)
    assert tl == [
        (date(2000, 1, 1), frozenset({"A"})),
        (date(2005, 1, 1), frozenset({"B"})),
    ]


def test_re_addition_two_intervals():
    ivs = [
        ConstituentInterval("XYZ", date(2005, 3, 1), date(2009, 6, 15)),
        ConstituentInterval("XYZ", date(2012, 1, 1), None),
    ]
    tl = constituents_to_snapshots(ivs)
    assert tl == [
        (date(2005, 3, 1), frozenset({"XYZ"})),
        (date(2009, 6, 15), frozenset()),
        (date(2012, 1, 1), frozenset({"XYZ"})),
    ]


def test_no_op_change_dates_collapsed():
    # B drops same day A adds-back equivalent membership? Construct a redundant change date.
    ivs = [
        ConstituentInterval("A", date(2000, 1, 1), None),
        ConstituentInterval("B", date(2000, 1, 1), date(2001, 1, 1)),
        ConstituentInterval("B", date(2001, 1, 1), None),  # re-add same day it drops
    ]
    tl = constituents_to_snapshots(ivs)
    # membership never actually changes on 2001-01-01 ({A,B} -> {A,B}); collapsed away.
    assert tl == [(date(2000, 1, 1), frozenset({"A", "B"}))]


def test_overlapping_intervals_rejected():
    ivs = [
        ConstituentInterval("A", date(2000, 1, 1), date(2003, 1, 1)),
        ConstituentInterval("A", date(2002, 1, 1), None),
    ]
    with pytest.raises(ValueError, match="overlap"):
        constituents_to_snapshots(ivs)


def test_add_after_drop_rejected():
    with pytest.raises(ValueError, match="add_date.*<=.*drop_date|add_date must be"):
        parse_constituents_rows([{"symbol": "A", "add_date": "2005-01-01", "drop_date": "2004-01-01"}])


def test_zero_length_interval_rejected():
    with pytest.raises(ValueError, match="zero-length|add_date == drop_date"):
        parse_constituents_rows([{"symbol": "A", "add_date": "2005-01-01", "drop_date": "2005-01-01"}])


def test_symbols_normalized_before_dedup():
    rows = [
        {"symbol": " aapl ", "add_date": "1998-01-02", "drop_date": ""},
        {"symbol": "AAPL", "add_date": "1998-01-02", "drop_date": ""},  # dup after normalize
    ]
    ivs = parse_constituents_rows(rows)
    assert ivs == [ConstituentInterval("AAPL", date(1998, 1, 2), None)]


def test_malformed_row_rejected():
    with pytest.raises(ValueError):
        parse_constituents_rows([{"symbol": "", "add_date": "1998-01-02", "drop_date": ""}])
    with pytest.raises(ValueError):
        parse_constituents_rows([{"symbol": "A", "add_date": "not-a-date", "drop_date": ""}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_constituents.py -q`
Expected: FAIL (`ModuleNotFoundError: algua.data.constituents`).

- [ ] **Step 3: Write the implementation**

```python
# algua/data/constituents.py
"""Pure constituents transformer: (symbol, add_date, drop_date) intervals → a minimal
point-in-time membership timeline (one membership set per change date).

Convention: ``add_date`` inclusive, ``drop_date`` exclusive (empty drop = open / still a
member). No I/O — the CLI feeds the result to ``DataStore.ingest_universe`` one snapshot per
change date.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ConstituentInterval:
    symbol: str
    add_date: date
    drop_date: date | None  # None = open (still a member)


def _norm_symbol(raw: str) -> str:
    s = raw.strip().upper()
    if not s:
        raise ValueError("symbol must not be empty")
    return s


def parse_constituents_rows(rows: list[dict[str, str]]) -> list[ConstituentInterval]:
    """Canonicalize + validate raw CSV rows into intervals (symbol normalized BEFORE checks).

    Rejects: empty symbol, unparseable dates, ``add_date >= drop_date`` (covers add>drop and the
    degenerate zero-length ``add == drop``). Exact-duplicate rows are de-duplicated.
    """
    seen: set[tuple[str, date, date | None]] = set()
    out: list[ConstituentInterval] = []
    for row in rows:
        symbol = _norm_symbol(str(row.get("symbol", "")))
        add_date = date.fromisoformat(str(row["add_date"]).strip())
        drop_raw = str(row.get("drop_date", "") or "").strip()
        drop_date = date.fromisoformat(drop_raw) if drop_raw else None
        if drop_date is not None and add_date >= drop_date:
            raise ValueError(
                f"{symbol}: add_date {add_date} must be < drop_date {drop_date} "
                f"(zero-length / add_date == drop_date intervals are rejected)"
            )
        key = (symbol, add_date, drop_date)
        if key in seen:
            continue
        seen.add(key)
        out.append(ConstituentInterval(symbol, add_date, drop_date))
    return out


def constituents_to_snapshots(
    intervals: list[ConstituentInterval],
) -> list[tuple[date, frozenset[str]]]:
    """Intervals → minimal membership timeline. Rejects overlapping intervals per symbol;
    collapses consecutive no-op change dates (membership unchanged)."""
    by_symbol: dict[str, list[ConstituentInterval]] = defaultdict(list)
    for iv in intervals:
        by_symbol[iv.symbol].append(iv)
    for symbol, ivs in by_symbol.items():
        ivs.sort(key=lambda i: i.add_date)
        for prev, nxt in zip(ivs, ivs[1:]):
            # prev.drop_date None => open interval that can never be followed by another.
            if prev.drop_date is None or nxt.add_date < prev.drop_date:
                raise ValueError(f"{symbol}: overlapping membership intervals")

    change_dates = sorted(
        {iv.add_date for iv in intervals}
        | {iv.drop_date for iv in intervals if iv.drop_date is not None}
    )

    timeline: list[tuple[date, frozenset[str]]] = []
    prev_members: frozenset[str] | None = None
    for d in change_dates:
        members = frozenset(
            iv.symbol
            for iv in intervals
            if iv.add_date <= d and (iv.drop_date is None or d < iv.drop_date)
        )
        if members == prev_members:
            continue  # no-op change date — collapse
        timeline.append((d, members))
        prev_members = members
    return timeline
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_constituents.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/data/constituents.py tests/data/test_constituents.py
git commit -m "feat(212): pure constituents->membership-timeline transformer"
```

---

## Task 2: Delistings dataset enums + manifest conflict hook + immutable universe ingest

**Files:**
- Modify: `algua/data/models.py` (add enum values)
- Modify: `algua/data/manifest.py:51` (`append_if_absent` conflict hook)
- Modify: `algua/data/store.py:247` (`ingest_universe` gains `require_immutable`)
- Test: `tests/data/test_universe_immutable.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_universe_immutable.py
import pytest

from algua.data.store import DataStore


def _ingest(store, syms, eff):
    return store.ingest_universe(
        universe="SP500", symbols=syms, effective_date=eff,
        as_of="2026-01-01T00:00:00+00:00", source="test", require_immutable=True,
    )


def test_same_date_same_membership_is_idempotent(tmp_path):
    store = DataStore(tmp_path)
    a = _ingest(store, ["AAPL", "MSFT"], "2000-01-01")
    b = _ingest(store, ["MSFT", "AAPL"], "2000-01-01")  # same set, different order
    assert a.snapshot_id == b.snapshot_id  # content-hash dedup


def test_same_date_different_membership_rejected_before_write(tmp_path):
    store = DataStore(tmp_path)
    _ingest(store, ["AAPL", "MSFT"], "2000-01-01")
    before = store.data_dir.joinpath("manifest.jsonl").read_text()
    with pytest.raises(ValueError, match="immutab|conflict|differs"):
        _ingest(store, ["AAPL", "GOOG"], "2000-01-01")
    after = store.data_dir.joinpath("manifest.jsonl").read_text()
    assert before == after  # rejected import left the manifest unmutated


def test_non_immutable_path_unaffected(tmp_path):
    store = DataStore(tmp_path)
    store.ingest_universe(
        universe="U", symbols=["A"], effective_date="2000-01-01",
        as_of="2026-01-01T00:00:00+00:00", source="test",
    )  # require_immutable defaults False — no conflict checking
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_universe_immutable.py -q`
Expected: FAIL (`ingest_universe() got an unexpected keyword argument 'require_immutable'`).

- [ ] **Step 3a: Add the enum values**

In `algua/data/models.py`, extend the two StrEnums:

```python
class Dataset(StrEnum):
    """Dataset routing key — the manifest `dataset` field and snapshot path component."""

    BARS = "bars"
    UNIVERSES = "universes"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"
    DELISTINGS = "delistings"


class Kind(StrEnum):
    """Snapshot `kind` — the provenance of a snapshot's payload."""

    BARS = "bars"
    UNIVERSE = "universe"
    FILE = "file"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"
    DELISTING = "delisting"
```

- [ ] **Step 3b: Add the conflict hook to `append_if_absent`**

In `algua/data/manifest.py`, change the signature and run the hook under the lock, after the snapshot_id dedup and before the append:

```python
    def append_if_absent(
        self,
        rec: SnapshotRecord,
        *,
        conflict_check: "Callable[[list[SnapshotRecord], SnapshotRecord], None] | None" = None,
    ) -> SnapshotRecord:
```

Add the import at the top of the file:

```python
from collections.abc import Callable
```

Inside the `try:` block, replace the dedup loop with one that keeps the parsed list and invokes the hook:

```python
            raw = self.path.read_bytes() if self.path.exists() else b""
            committed = self._committed_prefix(raw)
            committed_records = self._parse_committed(committed.decode("utf-8"))
            for existing in committed_records:
                if existing.snapshot_id == rec.snapshot_id:
                    return existing
            if conflict_check is not None:
                conflict_check(committed_records, rec)  # raises to abort before any write
            self._clean_stale_repair_temps()
            if committed != raw:
                self._repair(committed)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            fsync_dir(self.path.parent)
            return rec
```

- [ ] **Step 3c: Thread `require_immutable` through `ingest_universe`**

In `algua/data/store.py`, `ingest_universe` (line 247): add the parameter and a conflict check passed to `_ingest_parquet`. First extend `_ingest_parquet` to forward a `conflict_check`:

```python
    def _ingest_parquet(
        self,
        *,
        metadata: SnapshotMetadata,
        frame: pd.DataFrame,
        filename: str,
        conflict_check=None,
    ) -> SnapshotRecord:
        payload = frame_to_parquet_bytes(frame)
        content_hash = sha256_bytes(payload)
        snapshot_id = _snapshot_id(metadata, content_hash)

        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing

        relative_path = Path("snapshots") / metadata.dataset / snapshot_id / filename
        write_bytes_snapshot(payload, self.data_dir, relative_path)
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(frame),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet",
        )
        return self.manifest.append_if_absent(rec, conflict_check=conflict_check)
```

Note: the early `manifest.find` is a fast-path; the authoritative same-id dedup and the conflict
check both run under the lock inside `append_if_absent`, so the payload write before a *conflict*
rejection is an orphaned blob (no manifest row) — acceptable (consistent with how a crash between
write and append already behaves; no GC in scope).

Then `ingest_universe`:

```python
    def ingest_universe(
        self,
        *,
        universe: str,
        symbols: list[str],
        effective_date: str,
        as_of: str,
        source: str,
        provider: str = "local",
        source_metadata: dict[str, str] | None = None,
        require_immutable: bool = False,
    ) -> SnapshotRecord:
        clean_symbols = normalize_symbols(symbols)
        frame = pd.DataFrame(
            {"effective_date": effective_date, "universe": universe, "symbol": clean_symbols}
        )
        metadata = _metadata(
            dataset=Dataset.UNIVERSES.value,
            provider=provider,
            symbols=clean_symbols,
            start=effective_date,
            end=effective_date,
            as_of=as_of,
            source=source,
            kind=Kind.UNIVERSE.value,
            universe=universe,
            source_metadata=source_metadata,
        )

        conflict_check = None
        if require_immutable:
            def conflict_check(committed, rec):  # noqa: E306 — closure over universe/effective_date
                for other in committed:
                    if (
                        other.dataset == Dataset.UNIVERSES.value
                        and other.metadata.universe == universe
                        and other.metadata.start == effective_date
                        and other.content_hash != rec.content_hash
                    ):
                        raise ValueError(
                            f"universe {universe!r} already has a DIFFERENT membership on "
                            f"{effective_date} (immutable; corrections require a new name)"
                        )

        return self._ingest_parquet(
            metadata=metadata, frame=frame, filename="universe.parquet",
            conflict_check=conflict_check,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_universe_immutable.py -q`
Expected: PASS (3 tests). Also run `uv run pytest tests/data -q` to confirm no regression in existing ingest tests.

- [ ] **Step 5: Commit**

```bash
git add algua/data/models.py algua/data/manifest.py algua/data/store.py tests/data/test_universe_immutable.py
git commit -m "feat(212): delistings enums + manifest conflict hook + immutable universe ingest"
```

---

## Task 3: `data import-universe` CLI

**Files:**
- Modify: `algua/cli/data_cmd.py` (new command; imports transformer)
- Test: `tests/cli/test_import_universe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_import_universe.py
import json
from datetime import date

from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.store import DataStore


def _write_csv(path, text):
    path.write_text(text)
    return path


def test_import_universe_builds_timeline(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    csv = _write_csv(
        tmp_path / "c.csv",
        "symbol,add_date,drop_date\n"
        "AAPL,1998-01-02,\n"
        "ENRN,1998-01-02,2001-11-28\n",
    )
    res = CliRunner().invoke(app, ["data", "import-universe", "SP", "--file", str(csv)])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["snapshots_written"] == 2  # 1998-01-02 and 2001-11-28

    timeline = DataStore(tmp_path).read_universe("SP")
    eff = {s.effective_date: s.symbols for s in timeline}
    assert eff[date(1998, 1, 2)] == frozenset({"AAPL", "ENRN"})
    assert eff[date(2001, 11, 28)] == frozenset({"AAPL"})


def test_import_universe_same_name_correction_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    csv1 = _write_csv(tmp_path / "a.csv", "symbol,add_date,drop_date\nAAPL,2000-01-01,\n")
    assert CliRunner().invoke(app, ["data", "import-universe", "U", "--file", str(csv1)]).exit_code == 0
    csv2 = _write_csv(tmp_path / "b.csv", "symbol,add_date,drop_date\nMSFT,2000-01-01,\n")
    res = CliRunner().invoke(app, ["data", "import-universe", "U", "--file", str(csv2)])
    assert res.exit_code != 0
    assert "immutab" in res.output.lower() or "different membership" in res.output.lower()
```

(Confirm the data-dir env var name used by the test harness — grep `get_settings` / existing `tests/cli` for the fixture; reuse whatever existing CLI tests use to point `DataStore` at `tmp_path`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_import_universe.py -q`
Expected: FAIL (no such command `import-universe`).

- [ ] **Step 3: Add the command**

In `algua/cli/data_cmd.py` add (near `ingest_universe`), reusing the existing helpers `_store()`, `emit`, `ok`, `now_iso`, `json_errors`, and importing the transformer:

```python
import csv as _csv

from algua.data.constituents import constituents_to_snapshots, parse_constituents_rows


@data_app.command("import-universe")
@json_errors(ValueError, LookupError, FileNotFoundError)
def import_universe(
    universe: str,
    file: Path = typer.Option(..., "--file", help="constituents CSV: symbol,add_date,drop_date"),
    as_of: str = typer.Option(None, "--as-of", help="point-in-time ISO datetime"),
    source: str = typer.Option("bulk-import", "--source"),
) -> None:
    """Bulk-import a constituents CSV into the universe-snapshot timeline (one snapshot per
    change date). Universes are IMMUTABLE: a same-date membership conflict aborts before any
    write — corrections require a new universe name."""
    with file.expanduser().open(newline="") as fh:
        rows = list(_csv.DictReader(fh))
    intervals = parse_constituents_rows(rows)
    timeline = constituents_to_snapshots(intervals)
    stamp = as_of or now_iso()
    store = _store()
    written = 0
    symbols_seen: set[str] = set()
    for effective_date, members in timeline:
        symbols_seen.update(members)
        store.ingest_universe(
            universe=universe,
            symbols=sorted(members) if members else [],
            effective_date=effective_date.isoformat(),
            as_of=stamp,
            source=source,
            require_immutable=True,
        )
        written += 1
    emit(ok({
        "universe": universe,
        "snapshots_written": written,
        "change_dates": [d.isoformat() for d, _ in timeline],
        "symbols_seen": sorted(symbols_seen),
    }))
```

Note: an empty-membership change date passes `symbols=[]`. `normalize_symbols` raises on an empty
list. Decide in implementation: either (a) record an explicit "empty universe" sentinel snapshot, or
(b) — simpler and chosen here — relax `ingest_universe` to accept an empty membership for the
DELISTING-of-everyone edge by storing an empty `symbol` column. Pick (b): in `ingest_universe`,
when `symbols` is empty, skip `normalize_symbols` and store an empty frame
(`pd.DataFrame({"effective_date": [...], "universe": [...], "symbol": []})`), and have
`read_universe` already yields `frozenset()` for it. Add a unit test for the empty-membership round-trip.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_import_universe.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/data_cmd.py algua/data/store.py tests/cli/test_import_universe.py
git commit -m "feat(212): data import-universe bulk constituents importer"
```

---

## Task 4: Delisting-exit overlay (pure)

**Files:**
- Create: `algua/backtest/delisting.py`
- Test: `tests/backtest/test_delisting_overlay.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_delisting_overlay.py
from datetime import date

import numpy as np
import pandas as pd
import pytest

from algua.backtest.delisting import (
    DelistingExitError,
    DelistingRecord,
    apply_delisting_exits,
)


def _grid(cols):  # cols: dict[str, list]; 4 daily bars
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(cols, index=idx)


def test_record_construction_rejects_nonpositive_price():
    with pytest.raises(ValueError, match="terminal_price"):
        DelistingRecord(date(2020, 1, 2), 0.0, "test")
    with pytest.raises(ValueError, match="terminal_price"):
        DelistingRecord(date(2020, 1, 2), float("nan"), "test")


def test_held_with_record_forces_exit_at_terminal_price():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.0, 0.0]})
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 5.0, "vendor")]}
    adj_x, w_x, forced = apply_delisting_exits(adj, w, recs)
    T = adj.index[1]  # B's last valid bar
    assert w_x.loc[T, "B"] == 0.0 and (w_x.loc[T:, "B"] == 0.0).all()
    assert adj_x.loc[T, "B"] == 5.0                      # realized at terminal price
    assert (adj_x.loc[adj_x.index > T, "B"] == 5.0).all()  # NaN tail killed
    assert forced == [{"symbol": "B", "bar": T.isoformat(),
                       "terminal_price": 5.0, "source": "vendor"}]
    # A untouched
    assert (w_x["A"] == w["A"]).all() and adj_x["A"].equals(adj["A"])


def test_held_without_record_fails_closed():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.0, 0.0]})
    with pytest.raises(DelistingExitError, match="no delisting record"):
        apply_delisting_exits(adj, w, None)


def test_held_without_record_relaxation_realizes_last_close():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.0, 0.0]})
    adj_x, w_x, forced = apply_delisting_exits(adj, w, None, assume_terminal_last_close=True)
    T = adj.index[1]
    assert (w_x.loc[T:, "B"] == 0.0).all()
    assert forced == [{"symbol": "B", "bar": T.isoformat(),
                       "terminal_price": 11.0, "source": "assumed_last_close"}]


def test_not_held_only_nan_killed_no_error():
    # B's bars end early but the strategy already exited at bar 0 (weight 0 at its last bar).
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, np.nan, np.nan, np.nan]})
    w = _grid({"A": [1.0, 1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0, 0.0]})
    adj_x, w_x, forced = apply_delisting_exits(adj, w, None)
    assert forced == []
    T = adj.index[0]
    assert (adj_x.loc[adj_x.index > T, "B"] == 10.0).all()  # killed, position 0 anyway


def test_integrity_bars_after_resolved_delisting_fails_closed():
    # Record says B delisted on bar 1, but B has bars through bar 3.
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, 12, 13]})
    w = _grid({"A": [0.5] * 4, "B": [0.5] * 4})
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 5.0, "vendor")]}
    with pytest.raises(DelistingExitError, match="after stated delisting"):
        apply_delisting_exits(adj, w, recs)


def test_period_ends_on_delisting_applies_terminal_price():
    # B trades through the panel end AND a record dates the delisting at the last bar.
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, 12, 20]})
    w = _grid({"A": [0.5] * 4, "B": [0.5] * 4})
    recs = {"B": [DelistingRecord(date(2020, 1, 4), 5.0, "vendor")]}
    adj_x, w_x, forced = apply_delisting_exits(adj, w, recs)
    T = adj.index[3]  # == panel_end
    assert adj_x.loc[T, "B"] == 5.0 and w_x.loc[T, "B"] == 0.0
    assert forced and forced[0]["bar"] == T.isoformat()


def test_record_after_panel_end_skipped():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5] * 4, "B": [0.5, 0.5, 0.0, 0.0]})
    recs = {"B": [DelistingRecord(date(2021, 6, 1), 5.0, "vendor")]}  # far future
    # B's bars end early & held, the future record is not applicable -> fail closed.
    with pytest.raises(DelistingExitError, match="no delisting record"):
        apply_delisting_exits(adj, w, recs)


def test_two_records_same_terminal_bar_ambiguous():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5] * 4, "B": [0.5, 0.5, 0.0, 0.0]})
    # both 2020-01-02 (Thu) and a hypothetical weekend date resolve to bar index 1
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 5.0, "v1"),
                  DelistingRecord(date(2020, 1, 2), 6.0, "v2")]}
    with pytest.raises(DelistingExitError, match="ambiguous"):
        apply_delisting_exits(adj, w, recs)


def test_symbol_never_traded_skipped():
    adj = _grid({"A": [10, 11, 12, 13], "B": [np.nan, np.nan, np.nan, np.nan]})
    w = _grid({"A": [1.0] * 4, "B": [0.0] * 4})
    adj_x, w_x, forced = apply_delisting_exits(adj, w, None)
    assert forced == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_delisting_overlay.py -q`
Expected: FAIL (`ModuleNotFoundError: algua.backtest.delisting`).

- [ ] **Step 3: Write the implementation**

```python
# algua/backtest/delisting.py
"""Delisting-aware exit overlay — pure transform on the (timestamp × symbol) execution grid,
applied right before vectorbt `from_orders`. It NEVER invents an exit: it forces a liquidation
only for a position HELD past its last real bar, and only with a confirmed delisting record
(else fail-closed, unless the human-only relaxation). It always kills the post-delisting NaN
tail so `0 × NaN` cannot poison group equity. See the #212 design spec.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class DelistingRecord:
    delisting_date: date
    terminal_price: float  # per-share terminal proceeds, in adj_close units; strictly > 0
    source: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.terminal_price) or self.terminal_price <= 0:
            raise ValueError(
                "terminal_price must be finite and > 0 (zero-proceeds write-off deferred)"
            )


class DelistingExitError(Exception):
    """Fail-closed condition in the delisting-exit overlay (engine translates to BacktestError)."""


def _resolve_bar(index: pd.DatetimeIndex, d: date) -> pd.Timestamp | None:
    """Greatest bar whose date is <= d (as-of in the panel's own index). None if d precedes
    the first bar. Calendar-free — uses only the panel index."""
    eligible = [ts for ts in index if ts.date() <= d]
    return eligible[-1] if eligible else None


def apply_delisting_exits(
    adj: pd.DataFrame,
    weights_eff: pd.DataFrame,
    records: Mapping[str, list[DelistingRecord]] | None = None,
    *,
    assume_terminal_last_close: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (adj_exec, weights_exec, forced_exits). See module docstring + spec."""
    records = records or {}
    adj_exec = adj.copy()
    weights_exec = weights_eff.copy()
    forced_exits: list[dict] = []
    if len(adj.index) == 0:
        return adj_exec, weights_exec, forced_exits

    panel_end = adj.index[-1]
    panel_end_date = panel_end.date()

    for c in adj.columns:
        col = adj[c]
        T = col.last_valid_index()
        if T is None:
            continue  # never traded in this panel
        first_bar = col.first_valid_index()
        sym_records = list(records.get(c, []))

        # Integrity: any record dated within the panel whose resolved bar has REAL bars after it.
        for r in sym_records:
            if r.delisting_date > panel_end_date:
                continue
            d_bar = _resolve_bar(adj.index, r.delisting_date)
            if d_bar is None:
                continue
            later = col.loc[col.index > d_bar]
            if bool(later.notna().any()):
                raise DelistingExitError(
                    f"{c}: bars exist after stated delisting {r.delisting_date.isoformat()} "
                    f"(resolved bar {d_bar.date().isoformat()})"
                )

        # Applicable record: in [first_bar, panel_end] and resolving exactly to T.
        candidates = [
            r
            for r in sym_records
            if first_bar.date() <= r.delisting_date <= panel_end_date
            and _resolve_bar(adj.index, r.delisting_date) == T
        ]
        if len(candidates) >= 2:
            raise DelistingExitError(
                f"{c}: {len(candidates)} delisting records resolve to the same terminal bar "
                f"{T.date().isoformat()} (ambiguous terminal valuation)"
            )
        record = candidates[0] if candidates else None
        held = bool(weights_eff.loc[T, c] != 0)
        ends_early = T < panel_end

        if record is not None and held:
            weights_exec.loc[T:, c] = 0.0
            adj_exec.loc[T, c] = record.terminal_price
            forced_exits.append(
                {
                    "symbol": c,
                    "bar": T.isoformat(),
                    "terminal_price": float(record.terminal_price),
                    "source": record.source,
                }
            )
        elif ends_early and held and record is None:
            if not assume_terminal_last_close:
                raise DelistingExitError(
                    f"{c}: held position past its last bar {T.date().isoformat()} with no "
                    f"delisting record (provide one or pass --assume-terminal-last-close)"
                )
            weights_exec.loc[T:, c] = 0.0
            forced_exits.append(
                {
                    "symbol": c,
                    "bar": T.isoformat(),
                    "terminal_price": float(col.loc[T]),
                    "source": "assumed_last_close",
                }
            )

        # NaN-poison kill (whenever the column has a dead tail). adj_exec.loc[T, c] is the
        # realized price (overridden above when a record applied), inert at a 0 position.
        if ends_early:
            adj_exec.loc[adj_exec.index > T, c] = adj_exec.loc[T, c]

    return adj_exec, weights_exec, forced_exits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_delisting_overlay.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/delisting.py tests/backtest/test_delisting_overlay.py
git commit -m "feat(212): pure delisting-aware exit overlay"
```

---

## Task 5: Wire the overlay into `simulate()` + runtime guarantee

**Files:**
- Modify: `algua/backtest/engine.py` (`simulate` signature, overlay call, conditional `call_seq`, runtime check; `run` threading)
- Test: `tests/backtest/test_delisting_simulate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_delisting_simulate.py
import numpy as np
import pandas as pd
import pytest

from algua.backtest.engine import BacktestError, simulate


class _FakeProvider:
    """Returns a fixed bars frame in algua's long bar schema for the requested symbols."""

    def __init__(self, frame):
        self._frame = frame

    def get_bars(self, symbols, start, end, timeframe):
        return self._frame[self._frame["symbol"].isin(set(symbols))]


def _bars():
    # B delists after 2020-01-02 (no bars on the 3rd/4th); A survives.
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    for ts, a in zip(idx, [10, 11, 12, 13]):
        rows.append({"timestamp": ts, "symbol": "A", "open": a, "high": a, "low": a,
                     "close": a, "adj_close": a, "volume": 100})
    for ts, b in zip(idx[:2], [10, 11]):
        rows.append({"timestamp": ts, "symbol": "B", "open": b, "high": b, "low": b,
                     "close": b, "adj_close": b, "volume": 100})
    return pd.DataFrame(rows).set_index("timestamp")
```

The two integration tests below depend on a strategy that holds A+B equally each bar; reuse the
project's existing equal-weight test strategy/fixture (grep `tests/backtest` for the helper that
builds a `LoadedStrategy` with a fixed two-symbol universe — e.g. the one used by current
`simulate`/`build_portfolio` tests). Name it `equal_weight_strategy(["A", "B"])` below.

```python
def test_delisting_exit_realizes_and_keeps_equity_finite(equal_weight_strategy):
    from datetime import datetime, UTC
    from algua.backtest.delisting import DelistingRecord
    strat = equal_weight_strategy(["A", "B"])
    provider = _FakeProvider(_bars())
    recs = {"B": [DelistingRecord(__import__("datetime").date(2020, 1, 2), 5.0, "vendor")]}
    pf, weights, forced = simulate(
        strat, provider, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 4, tzinfo=UTC),
        delisting_records=recs,
    )
    assert forced and forced[0]["symbol"] == "B"
    returns = pf.returns()
    assert np.isfinite(returns.fillna(0.0)).all()         # no NaN poison
    positions = pf.assets()
    T = pd.Timestamp("2020-01-02", tz="UTC")
    assert (positions["B"].loc[positions.index >= T].abs() < 1e-9).all()  # B liquidated


def test_held_into_gap_without_record_raises_backtest_error(equal_weight_strategy):
    from datetime import datetime, UTC
    strat = equal_weight_strategy(["A", "B"])
    provider = _FakeProvider(_bars())
    with pytest.raises(BacktestError, match="delisting record|residual"):
        simulate(strat, provider, datetime(2020, 1, 1, tzinfo=UTC),
                 datetime(2020, 1, 4, tzinfo=UTC))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_delisting_simulate.py -q`
Expected: FAIL (`simulate() got an unexpected keyword argument 'delisting_records'`).

- [ ] **Step 3: Modify `simulate()`**

In `algua/backtest/engine.py`: add the imports near the top:

```python
from algua.backtest.delisting import DelistingExitError, DelistingRecord, apply_delisting_exits
```

(Define `POSITION_EPS = 1e-9` as a module constant near the other constants.)

Extend the signature (keyword-only additions) and replace the tail of `simulate` (the
`weights_eff`/`from_orders`/`return` block, lines ~603-613):

```python
def simulate(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
    delisting_records: Mapping[str, list[DelistingRecord]] | None = None,
    assume_terminal_last_close: bool = False,
) -> tuple[vbt.Portfolio, pd.DataFrame]:
```

```python
    lag = strategy.execution.decision_lag_bars
    weights_eff = weights.shift(lag).fillna(0.0)

    try:
        adj_exec, weights_exec, forced_exits = apply_delisting_exits(
            adj, weights_eff, delisting_records,
            assume_terminal_last_close=assume_terminal_last_close,
        )
    except DelistingExitError as exc:
        raise BacktestError(str(exc)) from exc

    # call_seq="auto" (sells before buys under cash_sharing) only when a forced exit needs the
    # same-bar liquidation cash — keeps non-delisting backtests bit-identical to today.
    call_seq = "auto" if forced_exits else None
    pf = vbt.Portfolio.from_orders(
        close=adj_exec,
        size=weights_exec,
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,
        freq="1D",
        **({"call_seq": call_seq} if call_seq is not None else {}),
    )

    # Runtime guarantee: every forced-exit symbol must hold zero AFTER its forced bar, and
    # group returns must be finite. Backstops the target-state `held` proxy + call_seq="auto".
    if forced_exits:
        positions = pf.assets()
        for fe in forced_exits:
            sym = fe["symbol"]
            bar = pd.Timestamp(fe["bar"])
            after = positions[sym].loc[positions.index >= bar]
            if bool((after.abs() > POSITION_EPS).any()):
                raise BacktestError(
                    f"delisting exit for {sym} left a residual position after {fe['bar']}"
                )
        returns = pf.returns()
        if not bool(np.isfinite(returns.fillna(0.0)).all()):
            raise BacktestError("non-finite returns after delisting exits")

    return pf, weights_exec, forced_exits
```

**`simulate` now returns a 3-tuple** `(pf, weights_eff, forced_exits)`. `build_portfolio` is its
alias, so update EVERY unpack site in one go (this task), keeping the suite green:
- `walkforward.py:130` `pf, _weights = build_portfolio(...)` → `pf, _weights, _forced = build_portfolio(...)`
- `sweep.py` worker call site (`build_portfolio`/`simulate`) → unpack three.
- existing tests that do `pf, weights = simulate(...)` / `build_portfolio(...)` — grep
  `tests/` for `= simulate(` and `= build_portfolio(` and add the third unpack var.

Then thread the two params through `run` (engine.py:621), capturing forced_exits (used for
provenance in Task 10):

```python
    pf, weights_eff, _forced_exits = simulate(...)
```

and add the same keyword-only params to `run` so callers can pass them:

```python
def run(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    seed: int | None = None,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    universe_name: str | None = None,
    universe_snapshots: list[dict[str, str]] | None = None,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
    delisting_records: Mapping[str, list[DelistingRecord]] | None = None,
    delisting_snapshot: str | None = None,
    assume_terminal_last_close: bool = False,
) -> BacktestResult:
    pf, weights_eff = simulate(
        strategy, provider, start, end,
        universe_by_date=universe_by_date, fundamentals_provider=fundamentals_provider,
        news_provider=news_provider, delisting_records=delisting_records,
        assume_terminal_last_close=assume_terminal_last_close,
    )
```

(Provenance — `forced_exits`/`delisting_snapshot` into `BacktestResult` — is Task 10. For now `run`
accepts the params and `delisting_snapshot` is unused except to pass through to provenance later;
mypy: annotate it now to avoid a churn later.)

**Note (import cycle):** `delisting.py` must NOT import from `engine.py` (engine imports delisting at
module top). The overlay raises `DelistingExitError`; engine translates. Verified: `delisting.py`
imports only `pandas`/stdlib.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_delisting_simulate.py -q`
Then the regression guard: `uv run pytest tests/backtest -q` (the conditional `call_seq` means
existing no-delisting backtests must be unchanged — all existing tests stay green). If any existing
backtest assertion shifts, STOP and surface it (the spec requires this be explicit, not silent).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/engine.py tests/backtest/test_delisting_simulate.py
git commit -m "feat(212): apply delisting overlay in simulate + runtime guarantee"
```

---

## Task 6: Thread the params through `walk_forward` and `sweep`

**Files:**
- Modify: `algua/backtest/walkforward.py:105` (`walk_forward` signature → `build_portfolio` call)
- Modify: `algua/backtest/sweep.py` (`sweep` + the pool-worker meta → `simulate`/`build_portfolio` call)
- Test: `tests/backtest/test_delisting_walkforward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/backtest/test_delisting_walkforward.py
from datetime import UTC, date, datetime

import numpy as np

from algua.backtest.delisting import DelistingRecord
from algua.backtest.walkforward import walk_forward
from tests.backtest.test_delisting_simulate import _FakeProvider, _bars  # reuse fixtures


def test_walk_forward_threads_delisting_records(equal_weight_strategy):
    strat = equal_weight_strategy(["A", "B"])
    provider = _FakeProvider(_bars())
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 5.0, "vendor")]}
    wf = walk_forward(
        strat, provider, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 4, tzinfo=UTC),
        windows=1, delisting_records=recs,
    )
    # The single full-period sim ran cleanly (no NaN poison) -> finite holdout metrics.
    assert np.isfinite(wf.holdout["sharpe"])
```

(Confirm `walk_forward`'s return shape — `wf.holdout` etc. — against `walkforward.py`; adapt the
assertion to whatever metric dict it exposes. The point is only that it runs end-to-end with records.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_delisting_walkforward.py -q`
Expected: FAIL (`walk_forward() got an unexpected keyword argument 'delisting_records'`).

- [ ] **Step 3: Thread the params**

In `algua/backtest/walkforward.py`, add to `walk_forward`'s keyword-only params (mirroring
`universe_by_date`):

```python
    delisting_records: "Mapping[str, list[DelistingRecord]] | None" = None,
    assume_terminal_last_close: bool = False,
```

Add the import:

```python
from algua.backtest.delisting import DelistingRecord
```

And pass them into the `build_portfolio(...)` call (line ~130; 3-tuple unpack after Task 5):

```python
    pf, _weights, _forced = build_portfolio(
        strategy, provider, start, end,
        universe_by_date=universe_by_date,
        delisting_records=delisting_records,
        assume_terminal_last_close=assume_terminal_last_close,
    )
```

In `algua/backtest/sweep.py`: thread the same two params through `sweep` (line 273) and the
module-level pool-worker meta/`_run_one` (the function that calls `build_portfolio`/`simulate` per
combo — around lines 176-191). Add them to the meta tuple/struct that is pickled to workers (they
are simple data: a dict of lists of frozen dataclasses + a bool — picklable), and forward into the
`simulate`/`build_portfolio` call. Holdout never crosses the worker boundary (unchanged).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_delisting_walkforward.py -q`
Then: `uv run pytest tests/backtest -q` (no regression).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/walkforward.py algua/backtest/sweep.py tests/backtest/test_delisting_walkforward.py
git commit -m "feat(212): thread delisting records through walk_forward + sweep"
```

---

## Task 7: Delisting-record ingestion (store)

**Files:**
- Modify: `algua/data/store.py` (`ingest_delistings`, `read_delistings`)
- Test: `tests/data/test_delistings_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_delistings_store.py
from datetime import date

import pandas as pd
import pytest

from algua.backtest.delisting import DelistingRecord
from algua.data.store import DataStore


def test_ingest_and_read_delistings_roundtrip(tmp_path):
    store = DataStore(tmp_path)
    frame = pd.DataFrame(
        {"symbol": ["ENRN", "WCOM", "ENRN"],
         "delisting_date": ["2001-11-28", "2002-07-01", "1990-01-01"],
         "delisting_value": [0.25, 0.10, 3.0]}
    )
    store.ingest_delistings(frame=frame, as_of="2026-01-01T00:00:00+00:00", source="vendor")
    recs = store.read_delistings()
    assert set(recs) == {"ENRN", "WCOM"}
    assert len(recs["ENRN"]) == 2  # two events for one symbol -> list
    enrn_dates = {r.delisting_date for r in recs["ENRN"]}
    assert enrn_dates == {date(2001, 11, 28), date(1990, 1, 1)}
    assert all(isinstance(r, DelistingRecord) for r in recs["WCOM"])


def test_ingest_delistings_rejects_nonpositive_value(tmp_path):
    store = DataStore(tmp_path)
    frame = pd.DataFrame(
        {"symbol": ["X"], "delisting_date": ["2001-01-01"], "delisting_value": [0.0]}
    )
    with pytest.raises(ValueError, match="terminal_price|> 0|zero-proceeds"):
        store.ingest_delistings(frame=frame, as_of="2026-01-01T00:00:00+00:00", source="v")


def test_ingest_delistings_rejects_duplicate_event(tmp_path):
    store = DataStore(tmp_path)
    frame = pd.DataFrame(
        {"symbol": ["X", "X"], "delisting_date": ["2001-01-01", "2001-01-01"],
         "delisting_value": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        store.ingest_delistings(frame=frame, as_of="2026-01-01T00:00:00+00:00", source="v")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_delistings_store.py -q`
Expected: FAIL (`AttributeError: 'DataStore' object has no attribute 'ingest_delistings'`).

- [ ] **Step 3: Implement `ingest_delistings` + `read_delistings`**

In `algua/data/store.py` (after `ingest_universe`), reusing `_metadata`, `_ingest_parquet`,
`normalize_symbols`, and the `DelistingRecord` dataclass (import it):

```python
from algua.backtest.delisting import DelistingRecord
```

WARNING — import direction: `algua.backtest` already imports from `algua.data` (providers), so
importing `algua.backtest.delisting` into `algua.data.store` risks a cycle and crosses a module
boundary import-linter may flag. To stay clean, do NOT import the backtest dataclass here. Instead
`read_delistings` returns the SAME shape using a *local* construction: import `DelistingRecord`
lazily inside `read_delistings` ONLY, or — preferred — move `DelistingRecord` to a neutral home.
**Decision:** keep `DelistingRecord` in `algua/backtest/delisting.py` (its consumer), and have
`read_delistings` import it lazily inside the method body (`from algua.backtest.delisting import
DelistingRecord`) so `algua.data` has no module-level dependency on `algua.backtest`. Verify with
`uv run lint-imports` after this task; if the contract still complains, relocate `DelistingRecord`
to `algua/contracts/` (pure) and re-import from both — adjust Task 4 accordingly.

```python
    def ingest_delistings(
        self,
        *,
        frame: pd.DataFrame,
        as_of: str,
        source: str,
        provider: str = "local",
    ) -> SnapshotRecord:
        """Persist a point-in-time delistings snapshot: columns symbol, delisting_date,
        delisting_value (per-share terminal price in adj_close units, strictly > 0).

        Fails closed on value <= 0 / non-finite (zero-proceeds write-off deferred) and on a
        duplicate (symbol, delisting_date) event."""
        required = {"symbol", "delisting_date", "delisting_value"}
        if not required.issubset(frame.columns):
            raise ValueError(f"delistings frame must have columns {sorted(required)}")
        clean = frame.copy()
        clean["symbol"] = [s.strip().upper() for s in clean["symbol"].astype(str)]
        clean["delisting_date"] = [
            date.fromisoformat(str(d).strip()).isoformat() for d in clean["delisting_date"]
        ]
        clean["delisting_value"] = clean["delisting_value"].astype(float)
        for v in clean["delisting_value"]:
            if not (v > 0) or not math.isfinite(v):
                raise ValueError(
                    "delisting_value must be finite and > 0 (zero-proceeds write-off deferred)"
                )
        dup = clean.duplicated(subset=["symbol", "delisting_date"]).any()
        if bool(dup):
            raise ValueError("duplicate (symbol, delisting_date) delisting event")
        symbols = normalize_symbols(list(clean["symbol"]))
        metadata = _metadata(
            dataset=Dataset.DELISTINGS.value,
            provider=provider,
            symbols=symbols,
            start=min(clean["delisting_date"]),
            end=max(clean["delisting_date"]),
            as_of=as_of,
            source=source,
            kind=Kind.DELISTING.value,
        )
        return self._ingest_parquet(
            metadata=metadata, frame=clean.reset_index(drop=True), filename="delistings.parquet"
        )

    def read_delistings(self, as_of: str | None = None):
        """Point-in-time delistings read: the latest DELISTINGS snapshot with metadata.as_of <=
        `as_of` (or the latest overall when `as_of is None`). Returns
        {symbol: list[DelistingRecord]} (multiple events per symbol allowed)."""
        from algua.backtest.delisting import DelistingRecord

        records = self.manifest.list_records(Dataset.DELISTINGS.value)
        if as_of is not None:
            records = [r for r in records if r.metadata.as_of <= as_of]
        if not records:
            return {}
        latest = max(records, key=lambda r: r.metadata.as_of)
        frame = pd.read_parquet(self.data_dir / latest.data_path)
        out: dict[str, list[DelistingRecord]] = {}
        for row in frame.itertuples(index=False):
            out.setdefault(str(row.symbol), []).append(
                DelistingRecord(
                    delisting_date=date.fromisoformat(str(row.delisting_date)),
                    terminal_price=float(row.delisting_value),
                    source=str(latest.metadata.source),
                )
            )
        return out
```

(Ensure `import math` is present at the top of `store.py`; add it if missing.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_delistings_store.py -q && uv run lint-imports`
Expected: PASS + import contracts green.

- [ ] **Step 5: Commit**

```bash
git add algua/data/store.py tests/data/test_delistings_store.py
git commit -m "feat(212): delisting-record ingest/read (PIT delistings snapshot)"
```

---

## Task 8: `data import-delistings` CLI

**Files:**
- Modify: `algua/cli/data_cmd.py` (new command)
- Test: `tests/cli/test_import_delistings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_import_delistings.py
import json

from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.store import DataStore


def test_import_delistings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    csv = tmp_path / "d.csv"
    csv.write_text("symbol,delisting_date,delisting_value\nENRN,2001-11-28,0.25\n")
    res = CliRunner().invoke(
        app, ["data", "import-delistings", "--file", str(csv), "--source", "vendor"]
    )
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["ok"] is True
    recs = DataStore(tmp_path).read_delistings()
    assert recs["ENRN"][0].terminal_price == 0.25


def test_import_delistings_rejects_nonpositive(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    csv = tmp_path / "d.csv"
    csv.write_text("symbol,delisting_date,delisting_value\nX,2001-01-01,0\n")
    res = CliRunner().invoke(app, ["data", "import-delistings", "--file", str(csv)])
    assert res.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_import_delistings.py -q`
Expected: FAIL (no such command).

- [ ] **Step 3: Add the command**

In `algua/cli/data_cmd.py`:

```python
@data_app.command("import-delistings")
@json_errors(ValueError, LookupError, FileNotFoundError)
def import_delistings(
    file: Path = typer.Option(..., "--file", help="CSV: symbol,delisting_date,delisting_value"),
    as_of: str = typer.Option(None, "--as-of", help="point-in-time ISO datetime"),
    source: str = typer.Option("vendor", "--source"),
) -> None:
    """Import a delistings CSV (terminal price per share, adj_close units, strictly > 0) as one
    point-in-time delistings snapshot."""
    frame = pd.read_csv(file.expanduser())
    rec = _store().ingest_delistings(frame=frame, as_of=as_of or now_iso(), source=source)
    emit(ok({"snapshot": rec.to_dict()}))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_import_delistings.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/data_cmd.py tests/cli/test_import_delistings.py
git commit -m "feat(212): data import-delistings CLI"
```

---

## Task 9: `resolve_delisting_inputs` + CLI flags (with human-only relaxation)

**Files:**
- Modify: `algua/cli/_common.py` (`resolve_delisting_inputs`)
- Modify: `algua/cli/backtest_cmd.py` (`run`, `walk_forward_cmd`, `sweep` commands: `--delistings`, `--assume-terminal-last-close`)
- Modify: `algua/cli/research_cmd.py` (`--delistings`; `--assume-terminal-last-close` is human-only — the agent `research promote` path must REJECT it)
- Test: `tests/cli/test_delistings_flag.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/cli/test_delistings_flag.py
import json

from typer.testing import CliRunner

from algua.cli.main import app


def _seed_delistings(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("symbol,delisting_date,delisting_value\nB,2020-01-02,5.0\n")
    CliRunner().invoke(app, ["data", "import-delistings", "--file", str(csv), "--source", "v"])


def test_backtest_run_accepts_delistings_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    _seed_delistings(tmp_path)
    # Use the demo/synthetic provider path + a registered demo strategy as existing run tests do;
    # assert the command resolves --delistings without error (exit_code 0) and JSON ok.
    res = CliRunner().invoke(app, [
        "backtest", "run", "<demo-strategy-name>", "--demo",
        "--start", "2020-01-01", "--end", "2020-03-01", "--delistings", "ANY",
    ])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["ok"] is True


def test_research_promote_rejects_human_only_relaxation(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    res = CliRunner().invoke(app, [
        "research", "promote", "<strategy>", "--universe", "U",
        "--start", "2020-01-01", "--end", "2020-12-31",
        "--assume-terminal-last-close",
    ])
    assert res.exit_code != 0
    assert "human" in res.output.lower() or "not allowed" in res.output.lower()
```

(Fill `<demo-strategy-name>`/`<strategy>` from the existing CLI test fixtures. The first test only
needs the flag to thread; the synthetic provider won't trigger a real delisting, so the map is
resolved-but-unused — that is the intended "resolves without error" check.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cli/test_delistings_flag.py -q`
Expected: FAIL (unknown option `--delistings`).

- [ ] **Step 3a: Add `resolve_delisting_inputs`**

In `algua/cli/_common.py`:

```python
def resolve_delisting_inputs(
    delistings_name: str | None, end_dt: datetime
) -> tuple["Mapping[str, list] | None", list[dict[str, str]] | None]:
    """Resolve the opt-in delisting records for a backtest-family command.

    `None` (no `--delistings`) => no records (overlay still kills NaN tails and fails closed on a
    held-into-gap-without-record). Otherwise reads the latest delistings snapshot as-of `end_dt`
    and returns ({symbol: [DelistingRecord,...]}, provenance)."""
    if delistings_name is None:
        return None, None
    store = DataStore(get_settings().data_dir)
    records = store.read_delistings(as_of=end_dt.isoformat())
    if not records:
        raise ValueError(
            f"--delistings {delistings_name!r}: no delistings snapshot effective on or before "
            f"{end_dt.date().isoformat()}"
        )
    provenance = [{"name": delistings_name, "symbols": str(len(records))}]
    return records, provenance
```

(The `delistings_name` is a logical handle for provenance/UX; today there is a single delistings
table per store, so any non-null name selects "the latest as-of snapshot". If multiple named
delisting sets are needed later, key them like universes — out of scope now; document in `--help`.)

- [ ] **Step 3b: Wire the flags into `backtest_cmd.py`**

For `run`, `walk_forward_cmd`, and `sweep` add:

```python
    delistings: str = typer.Option(
        None, "--delistings",
        help="delistings snapshot handle (survivorship-free: realize held delisted names)"),
    assume_terminal_last_close: bool = typer.Option(
        False, "--assume-terminal-last-close",
        help="HUMAN-ONLY: realize a held-into-gap name with no record at its last close"),
```

After `resolve_universe_inputs(...)`:

```python
    delisting_records, delisting_prov = resolve_delisting_inputs(delistings, end_dt)
```

and pass `delisting_records=delisting_records, delisting_snapshot=delistings,
assume_terminal_last_close=assume_terminal_last_close` into `run_backtest`/`walk_forward`/`sweep`.
(`run_backtest` is `engine.run`, already threaded in Task 5; `walk_forward`/`sweep` threaded in Task 6.)

- [ ] **Step 3c: Human-only relaxation in `research_cmd.py`**

`research promote` (and any agent-actor path) must reject `--assume-terminal-last-close`. Mirror the
existing human-only relaxation pattern used for `--allow-non-pit`/`--n-combos` (grep
`research_cmd.py` for how those raise for the agent actor). Add the flag and:

```python
    if assume_terminal_last_close:
        raise ValueError(
            "--assume-terminal-last-close is human-only (an agent must supply delisting records); "
            "a held-into-gap name without a record fails closed for the agent path"
        )
```

Add `--delistings` (allowed for the agent) and thread `delisting_records` into the `walk_forward`
call there, exactly like `universe_by_date`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_delistings_flag.py -q`
Then full gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/_common.py algua/cli/backtest_cmd.py algua/cli/research_cmd.py tests/cli/test_delistings_flag.py
git commit -m "feat(212): --delistings flag + human-only --assume-terminal-last-close"
```

---

## Task 10: Provenance — surface `forced_exits` + `delisting_snapshot` in result JSON

**Files:**
- Modify: `algua/backtest/engine.py` (`simulate` returns forced_exits to `run`; `run` stamps `BacktestResult`)
- Modify: `algua/backtest/result.py` (add the provenance fields to the result dataclass + `to_dict`)
- Test: `tests/backtest/test_delisting_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/backtest/test_delisting_provenance.py
from datetime import UTC, date, datetime

from algua.backtest.delisting import DelistingRecord
from algua.backtest.engine import run
from tests.backtest.test_delisting_simulate import _FakeProvider, _bars


def test_forced_exits_in_result_provenance(equal_weight_strategy):
    strat = equal_weight_strategy(["A", "B"])
    provider = _FakeProvider(_bars())
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 5.0, "vendor")]}
    result = run(
        strat, provider, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 4, tzinfo=UTC),
        delisting_records=recs, delisting_snapshot="vendor-2026",
    )
    d = result.to_dict()
    assert d["delisting_snapshot"] == "vendor-2026"
    assert d["forced_exits"] and d["forced_exits"][0]["symbol"] == "B"
    assert d["forced_exits"][0]["terminal_price"] == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_delisting_provenance.py -q`
Expected: FAIL (`KeyError: 'forced_exits'` or `run` doesn't capture them).

- [ ] **Step 3: Thread forced_exits to the result**

`simulate` already returns the 3-tuple `(pf, weights_eff, forced_exits)` (Task 5). Here, just have
`run` capture forced_exits (currently discarded as `_forced_exits` in Task 5) and pass it + the
`delisting_snapshot` handle into the result:

```python
    pf, weights_eff, forced_exits = simulate(...)
    ...
    return BacktestResult(
        ...,
        delisting_snapshot=delisting_snapshot,
        forced_exits=forced_exits,
    )
```

Also update Task 6's `walk_forward`/`sweep` unpack sites if not already 3-tuple (they were migrated
in Task 5). In `algua/backtest/result.py`, add the two optional fields to the `BacktestResult`
dataclass (default `None`/`[]`) and include them in `to_dict()`:

```python
    delisting_snapshot: str | None = None
    forced_exits: list[dict] = field(default_factory=list)
```

```python
    # in to_dict():
    "delisting_snapshot": self.delisting_snapshot,
    "forced_exits": self.forced_exits,
```

(Check the existing `BacktestResult`/`to_dict` shape and match its style — it already carries
`universe_name`/`universe_snapshots`; add these alongside.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_delisting_provenance.py -q`
Then full gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/engine.py algua/backtest/result.py algua/backtest/walkforward.py algua/backtest/sweep.py tests/backtest/test_delisting_provenance.py
git commit -m "feat(212): surface forced_exits + delisting_snapshot in result provenance"
```

---

## Task 11: Docs — CLAUDE.md command surface + bar-schema note

**Files:**
- Modify: `CLAUDE.md` (Command surface: add `data import-universe`, `data import-delistings`; note the delisting-aware exit + human-only relaxation)

- [ ] **Step 1: Add the command-surface bullets**

Add under "Command surface" in `CLAUDE.md`:

```markdown
- `uv run algua data import-universe NAME --file constituents.csv` — bulk-import a PIT constituents
  CSV (symbol,add_date,drop_date; add inclusive, drop exclusive; multiple rows/symbol for
  re-additions) into the universe-snapshot timeline (one snapshot per change date). Universes are
  IMMUTABLE — a same-date membership conflict aborts before any write (corrections need a new name).
- `uv run algua data import-delistings --file delistings.csv` — import per-symbol terminal prices
  (symbol,delisting_date,delisting_value; value = per-share terminal proceeds in adj_close units,
  strictly > 0) as a PIT delistings snapshot. Backtests opt in with `--delistings NAME`: a held name
  whose bars end mid-backtest is realized at its terminal price; a held-into-gap name WITHOUT a
  record fails closed (`--assume-terminal-last-close` realizes at last close, HUMAN-ONLY).
```

- [ ] **Step 2: Verify the gate is green**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(212): document import-universe/import-delistings + delisting-aware exits"
```

---

## Self-review checklist (run before handoff to GATE-2)

- **Spec coverage:** A (import-universe) → Tasks 1,3. Delisting overlay B → Tasks 4,5; threading → 6; runtime guarantee → 5; call_seq → 5. C (ingestion) → Tasks 7,8; threading/flags → 9; human-only relaxation → 9; provenance → 10. Immutability-under-lock → 2. Zero-price fail-closed → 4 (`__post_init__`) + 7 (ingest). Ambiguous/integrity/boundary → 4. Docs → 11. ✅
- **Placeholder scan:** the only `<...>` are fixture names (demo strategy, equal-weight helper) the implementer fills from existing `tests/` — flagged inline with how to find them, not silent TODOs.
- **Type consistency:** `apply_delisting_exits` returns `(adj, weights, forced_exits)`; `simulate`/`build_portfolio` return `(pf, weights, forced_exits)` from Task 5 onward (all unpack sites — `run`, `walkforward.py:130`, `sweep.py`, existing tests — migrated in Task 5; Task 10 only consumes the already-captured forced_exits). `DelistingRecord(delisting_date, terminal_price, source)` consistent across overlay/store/CLI. `records: Mapping[str, list[DelistingRecord]]` consistent.
- **Open implementation decisions flagged for the executor:** (1) `DelistingRecord` home vs import-linter — lazy import in `store.read_delistings`, relocate to `contracts/` only if `lint-imports` complains; (2) empty-membership universe snapshot handling in `ingest_universe` (Task 3 chose option (b): store an empty `symbol` column); (3) `simulate` returns a 3-tuple from Task 5 — all unpack sites migrate there in one commit to keep the suite green.
