# Phase 3 Slice 1 — return-stream persistence (#221) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist exactly one out-of-sample (OOS) per-period return vector per holdout burn into a new
`holdout_returns` table, surface it as a SENSITIVE field on `WalkForwardResult`, and expose ONLY a
sibling-only cross-strategy read — the heavy shared prerequisite for Slices 2/3/4 (bootstrap, N_eff,
multi-regime). **No gate-behavior change — pure plumbing.**

**Architecture:** `walk_forward` already computes the per-period `returns` series and discards it; we
capture the holdout segment `(returns, bar_dates)` onto `WalkForwardResult.holdout_returns`
(SENSITIVE — excluded from `to_dict`). The burn (`committed_at`) stays at `on_peek` unchanged; the
return vector is written in a SEPARATE, subsequent transaction inside the protected `run_gate` after
`walk_forward` returns (the two events cannot share a transaction). A `UNIQUE(holdout_evaluation_id)`
makes the write idempotent/reconcilable. Access control is application-layer: NO CLI accessor, NO
"get my own vector" API — the only blob-reading method is a sibling-only cross-strategy read.

**Tech Stack:** Python 3.12, sqlite3 (WAL, `foreign_keys=ON`), numpy float64 blobs, pandas, pytest.

## Global Constraints

- **No gate-behavior change.** Slice 1 is plumbing: `evaluate_gate`'s pass/fail math is byte-identical.
  The only `GateDecision` addition is a non-binding `returns_available: bool` audit field. The full
  suite stays green (run it to learn the baseline count before Task 1).
- **Schema bump 26 → 27.** The spec text says "24 → 25" but predates Phase 2 (#220, schema 26) and
  #222 (kept 26). The CURRENT `SCHEMA_VERSION` is **26**; this slice bumps it to **27**. A brand-new
  table needs only its DDL added to `_SCHEMA` (run via `executescript(_SCHEMA)` on every `migrate`) plus
  the version bump — NO `_add_missing_columns`, NO bespoke migration code.
- **Grain = per-strategy-holdout, NOT per-combo.** Exactly ONE vector per holdout burn (the event that
  writes a `holdout_evaluations` row). `sweep()` must persist NOTHING (its per-combo holdouts never
  leave the process — single-use discipline).
- **Two-transaction crash-safety.** The burn `committed_at` update happens at `on_peek` (unchanged).
  The `holdout_returns` INSERT is a SEPARATE later transaction in `run_gate`. A committed burn with a
  missing returns row is a RECOVERABLE inconsistency (the vector is deterministic from re-running the
  same walk-forward); `UNIQUE(holdout_evaluation_id)` makes a reconciliation safe. Never test
  "committed_at stays NULL after a burn" — after a successful burn it is NOT NULL.
- **Access control (application-layer; SQLite has no row-level ACL).**
  1. NO CLI accessor exposes `returns_blob`/`bar_dates_blob` for any strategy.
  2. NO "get my own vector" API on `StrategyRepository`. The ONLY method reading `returns_blob` is the
     sibling-only `overlapping_holdout_return_streams(strategy_id, ...)`, which NEVER returns the
     requesting strategy's own vector.
  3. `promotion.py` is the only writer/read-coordinator. `gates.py` receives NO return vectors and does
     NO DB reads (it stays pure-math).
  4. `data inspect`/`data verify`/any export must NOT surface `holdout_returns` (they read the data
     manifest, not the registry DB — keep it that way; add nothing).
- **Fail-closed validation at the write.** Assert `len(returns) == len(bar_dates) == n_bars` and that
  `n_bars == wf.holdout_metrics["n_bars"]`; assert the passed `strategy_id` matches the
  `holdout_evaluations.strategy_id` for the given `holdout_evaluation_id` (read BEFORE the write tx).
  On read-back, assert `len(np.frombuffer(blob, float64)) == n_bars`.
- **Blob encoding.** `returns_blob = np.asarray(returns, dtype=np.float64).tobytes()`;
  `bar_dates_blob = "\n".join(bar_dates).encode("utf-8")`. Round-trip: `np.frombuffer(blob,
  dtype=np.float64)` and `blob.decode("utf-8").split("\n")`.
- **Pure modules stay pure:** `algua/contracts`, `algua/features` — no I/O. `gates.py` stays pure-math.
- **Per-slice quality gate (must pass before commit):**
  `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- **Protected walls** (`algua/registry/promotion.py`, and `gates.py` if touched) are CODEOWNERS
  `@Lior-Nis` — Slice 1 touches `promotion.py` (the returns-write wiring) and `gates.py` only for the
  `returns_available` audit field on `GateDecision`.

## Locked design decisions (resolved from the spec + current code)

- **D1 — schema 26→27** (see Global Constraints).
- **D2 — include the sibling-read accessor in Slice 1.** The spec's own Slice 1 access-control test
  requires "only the sibling-read accessor exists," so `overlapping_holdout_return_streams` ships here
  (Task 4), even though its first consumer is Slice 3.
- **D3 — the returns write lives in `run_gate`** (protected orchestrator), which gains an optional
  `holdout_evaluation_id: int | None = None` keyword param. Production (`research_cmd.py`) passes the
  `reservation_id`; the two `tests/test_promotion.py` helpers pass nothing (None ⇒ no write, keeps them
  green). The write happens unconditionally on every burn (pass OR fail) — the burn already committed.
- **D4 — harden `finalize_holdout_reservation` with `strategy_id`** in the WHERE predicate (add a
  `strategy_id: int` param); update its sole caller (the `on_peek` lambda in `research_cmd.py`).
- **D5 — `returns_available`** audit field on `GateDecision` (informational in Slice 1; Slices 2–4 use
  it to omit-not-fail bootstrap/regime checks for pre-Slice-1 promotions).
- **D6 — `WalkForwardResult.to_dict` explicitly drops `holdout_returns`** (custom `to_dict`, not
  `dataclasses.asdict`).

## File Structure

- `algua/registry/db.py` — `holdout_returns` table + 3 indexes in `_SCHEMA`; `SCHEMA_VERSION = 27`.
- `algua/backtest/walkforward.py` — `WalkForwardResult.holdout_returns` SENSITIVE field; populate it;
  custom `to_dict` that excludes it.
- `algua/registry/store.py` — `record_holdout_returns(...)`; `overlapping_holdout_return_streams(...)`;
  `finalize_holdout_reservation(..., strategy_id)` hardening.
- `algua/registry/repository.py` — Protocol decls for the two new methods + the `finalize` signature.
- `algua/cli/research_cmd.py` — pass `strategy_id` to `finalize_holdout_reservation`; pass
  `reservation_id` to `run_gate`.
- `algua/registry/promotion.py` (PROTECTED) — `run_gate` writes the returns row + records
  `returns_available`.
- `algua/research/gates.py` (PROTECTED) — `GateDecision.returns_available` field + `to_dict` entry.
- Tests: `tests/registry/test_holdout_returns.py` (new — schema, write, read, access-control);
  `tests/test_walkforward.py` or a new `tests/backtest/test_walkforward_holdout_returns.py` (field +
  to_dict); extend `tests/test_promotion.py` (integration).

---

## Task 1: `holdout_returns` table + schema bump 26→27 (`db.py`)

**Files:**
- Modify: `algua/registry/db.py` (`_SCHEMA` — add table after `holdout_evaluations` block ~line 166;
  `SCHEMA_VERSION = 26` → `27` at line 16)
- Test: `tests/registry/test_holdout_returns.py` (create)

**Interfaces:**
- Produces: a `holdout_returns` table with `UNIQUE(holdout_evaluation_id)`, FK → `holdout_evaluations(id)`
  and `strategies(id)`, columns `id, holdout_evaluation_id, strategy_id, holdout_start, holdout_end,
  n_bars, returns_blob, bar_dates_blob, created_at`; indexes `ux_holdout_returns_eval` (UNIQUE),
  `ix_holdout_returns_strategy`, `ix_holdout_returns_interval`.

- [ ] **Step 1: Write the failing schema test**

Create `tests/registry/test_holdout_returns.py`. Reuse the peer fixture pattern
(`from algua.registry.db import connect, migrate`; `from algua.registry.store import
SqliteStrategyRepository` — confirm the exact names against `tests/registry/test_factor_ledger.py`).

```python
from algua.registry.db import SCHEMA_VERSION, connect, migrate


def test_schema_version_is_27():
    assert SCHEMA_VERSION == 27


def test_holdout_returns_table_and_indexes_exist(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(holdout_returns)")}
    assert cols == {
        "id", "holdout_evaluation_id", "strategy_id", "holdout_start", "holdout_end",
        "n_bars", "returns_blob", "bar_dates_blob", "created_at",
    }
    idx = {r["name"] for r in conn.execute("PRAGMA index_list(holdout_returns)")}
    assert "ux_holdout_returns_eval" in idx          # UNIQUE(holdout_evaluation_id)
    # confirm UNIQUE is enforced
    uniq = [r for r in conn.execute("PRAGMA index_list(holdout_returns)")
            if r["name"] == "ux_holdout_returns_eval"]
    assert uniq and uniq[0]["unique"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/registry/test_holdout_returns.py -q`
Expected: FAIL — `SCHEMA_VERSION == 26` and no `holdout_returns` table.

- [ ] **Step 3: Add the table to `_SCHEMA` and bump the version**

In `algua/registry/db.py` add, immediately after the `holdout_evaluations` block (~line 166, before the
`gate_evaluations` comment):

```sql
-- holdout_returns persists EXACTLY ONE out-of-sample per-period return vector per holdout burn
-- (#221 Slice 1) — the heavy shared prerequisite for Phase-3 Slices 2/3/4 (bootstrap, N_eff,
-- multi-regime). Grain is per-strategy-holdout, NOT per-combo: persisting per-combo vectors would
-- re-open the single-use best-of-N surface sweep() is built to prevent. The FK ties each vector to
-- the burn that produced it; UNIQUE(holdout_evaluation_id) prevents double-writes and makes a
-- reconciliation job (re-running the deterministic walk-forward) safe. SENSITIVE: no CLI accessor and
-- no "get my own vector" API may read returns_blob — only the sibling-only cross-strategy read.
CREATE TABLE IF NOT EXISTS holdout_returns (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    holdout_evaluation_id INTEGER NOT NULL REFERENCES holdout_evaluations(id),
    strategy_id           INTEGER NOT NULL REFERENCES strategies(id),
    holdout_start         TEXT    NOT NULL,   -- OOS interval identity (mirrors #192 / #205)
    holdout_end           TEXT    NOT NULL,
    n_bars                INTEGER NOT NULL,   -- length of stored vector; equals holdout_metrics n_bars
    returns_blob          BLOB    NOT NULL,   -- float64 per-period OOS returns, np.tobytes()
    bar_dates_blob        BLOB    NOT NULL,   -- ISO-8601 bar dates, UTF-8 newline-delimited
    created_at            TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_holdout_returns_eval ON holdout_returns(holdout_evaluation_id);
CREATE INDEX IF NOT EXISTS ix_holdout_returns_strategy  ON holdout_returns(strategy_id);
CREATE INDEX IF NOT EXISTS ix_holdout_returns_interval  ON holdout_returns(holdout_start, holdout_end);
```

Change `SCHEMA_VERSION = 26` to `SCHEMA_VERSION = 27` (line 16). Confirm `migrate()` already runs
`conn.executescript(_SCHEMA)` (it does — no migration code needed for a new table).

- [ ] **Step 4: Run to verify pass + quality gate**

Run: `uv run pytest tests/registry/test_holdout_returns.py -q` → PASS.
Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` → all pass.

- [ ] **Step 5: Commit**

```bash
git add algua/registry/db.py tests/registry/test_holdout_returns.py
git commit -m "feat(221): holdout_returns table + schema 26->27 — Slice 1 of #221"
```

---

## Task 2: `WalkForwardResult.holdout_returns` SENSITIVE field + `to_dict` exclusion

**Files:**
- Modify: `algua/backtest/walkforward.py` (`WalkForwardResult` ~line 70; `to_dict` ~line 91;
  `walk_forward` body ~line 163)
- Test: `tests/backtest/test_walkforward_holdout_returns.py` (create; if `tests/backtest/` has no
  `__init__.py` and peers live at `tests/`, place it at `tests/test_walkforward_holdout_returns.py` —
  match the existing walk-forward test location, e.g. `tests/test_walkforward_segment_noguard.py`).

**Interfaces:**
- Consumes: `returns` (pd.Series, datetime index) and `holdout = (start_i, end_i)` already computed in
  `walk_forward`.
- Produces: `WalkForwardResult.holdout_returns: tuple[list[float], list[str]] | None = None` (the
  `(per-period returns, ISO bar dates)` pair). `to_dict()` NEVER includes it.

- [ ] **Step 1: Write the failing tests**

Create the test. Build a `WalkForwardResult` directly (don't run a full backtest) to test the
`to_dict` exclusion in isolation, plus one end-to-end check that `walk_forward` populates the field.

```python
import dataclasses

from algua.backtest.walkforward import WalkForwardResult


def _minimal_wf(**over):
    base = dict(
        strategy="s", config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None, period={"start": "2020-01-01", "end": "2020-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[], holdout_metrics={"n_bars": 3},
        stability={}, holdout_returns=([0.1, -0.2, 0.05], ["2020-12-29", "2020-12-30", "2020-12-31"]),
    )
    base.update(over)
    return WalkForwardResult(**base)


def test_to_dict_excludes_holdout_returns():
    d = _minimal_wf().to_dict()
    assert "holdout_returns" not in d
    # the rest of the payload is unchanged
    assert d["holdout_metrics"] == {"n_bars": 3}


def test_holdout_returns_defaults_to_none():
    wf = WalkForwardResult(
        strategy="s", config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None, period={"start": "2020-01-01", "end": "2020-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[], holdout_metrics={"n_bars": 3}, stability={})
    assert wf.holdout_returns is None
    assert "holdout_returns" not in wf.to_dict()
```

Add a `walk_forward`-level test that mirrors an existing walk-forward test's setup (synthetic provider +
a loaded demo strategy — copy the harness from `tests/test_walkforward_segment_noguard.py`) and asserts:
`wf.holdout_returns is not None`; `len(wf.holdout_returns[0]) == len(wf.holdout_returns[1]) ==
wf.holdout_metrics["n_bars"]`; the dates are ISO `YYYY-MM-DD` strings and match the holdout segment.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_walkforward_holdout_returns.py -q`
Expected: FAIL — `WalkForwardResult.__init__` got an unexpected keyword `holdout_returns` / field absent.

- [ ] **Step 3: Add the field + custom `to_dict` + populate in `walk_forward`**

In `WalkForwardResult` (after `universe_snapshots` ~line 89), add:

```python
    # SENSITIVE — stronger than holdout_metrics: the raw OOS return vector lets a researcher identify
    # which days their strategy failed and tune a later strategy to exploit the same holdout window.
    # NEVER serialized: to_dict() excludes it; only research promote persists it (in promotion.run_gate),
    # and only the sibling-only store read may ever surface it cross-strategy. (#221 Slice 1)
    holdout_returns: tuple[list[float], list[str]] | None = None
```

Replace `to_dict` (do NOT rely on `dataclasses.asdict` for this type going forward):

```python
    def to_dict(self) -> dict[str, Any]:
        # holdout_returns is SENSITIVE and MUST NOT appear in any serialized payload (#221 Slice 1).
        d = dataclasses.asdict(self)
        d.pop("holdout_returns", None)
        return d
```

In `walk_forward`, right after `holdout_metrics = _segment_record(returns, holdout[0], holdout[1])`
(~line 163), capture the OOS vector + dates (same date format as `_segment_record`: `str(ts.date())`):

```python
    holdout_seg = returns.iloc[holdout[0]:holdout[1]]
    holdout_returns = (
        [float(x) for x in holdout_seg.to_numpy()],
        [str(idx.date()) for idx in holdout_seg.index],
    )
```

and pass `holdout_returns=holdout_returns` into the `WalkForwardResult(...)` constructor.

- [ ] **Step 4: Run to verify pass + quality gate**

Run: `uv run pytest tests/test_walkforward_holdout_returns.py -q` → PASS.
Run the full quality gate → all pass. (Any other test that asserts on `walk_forward(...).to_dict()`
keys must still pass — `holdout_returns` is absent from `to_dict`, so they are unaffected.)

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/walkforward.py tests/test_walkforward_holdout_returns.py
git commit -m "feat(221): WalkForwardResult.holdout_returns SENSITIVE field (to_dict excludes) — Slice 1"
```

---

## Task 3: store write path + `finalize_holdout_reservation` strategy_id hardening

**Files:**
- Modify: `algua/registry/store.py` (`finalize_holdout_reservation` ~line 536; add
  `record_holdout_returns` near the holdout methods)
- Modify: `algua/registry/repository.py` (Protocol: update `finalize_holdout_reservation` signature ~line
  278; add `record_holdout_returns`)
- Modify: `algua/cli/research_cmd.py` (the `on_peek` lambda ~line 164 — pass `strategy_id`)
- Test: extend `tests/registry/test_holdout_returns.py`

**Interfaces:**
- Consumes: `holdout_evaluations` rows (id, strategy_id, committed burn).
- Produces:
  - `finalize_holdout_reservation(self, reservation_id: int, *, config_hash: str, strategy_id: int) ->
    None` — WHERE predicate now also matches `strategy_id` (caller-bug defense). Raises if rowcount != 1.
  - `record_holdout_returns(self, holdout_evaluation_id: int, strategy_id: int, *, holdout_start: str,
    holdout_end: str, returns: list[float], bar_dates: list[str]) -> int` — separate transaction;
    validates lengths + strategy_id match BEFORE the write; encodes float64 + UTF-8 blobs; returns the
    new row id. Raises `ValueError` on a length mismatch, a strategy_id mismatch, or a `UNIQUE`
    violation (already written).

- [ ] **Step 1: Write the failing tests**

Extend `tests/registry/test_holdout_returns.py`. Build a repo, register a strategy, `reserve_holdout`
→ `finalize_holdout_reservation(..., strategy_id=sid)` (a committed burn), then `record_holdout_returns`.
Cover:

```python
import numpy as np
import pytest


def test_record_and_read_back_round_trip(repo_with_burn):
    repo, sid, hid, (h_start, h_end) = repo_with_burn  # committed burn fixture
    rets = [0.01, -0.02, 0.005]
    dates = ["2020-12-29", "2020-12-30", "2020-12-31"]
    rid = repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                      returns=rets, bar_dates=dates)
    assert rid > 0
    row = repo._conn.execute(
        "SELECT n_bars, returns_blob, bar_dates_blob FROM holdout_returns WHERE id=?", (rid,)
    ).fetchone()
    assert row["n_bars"] == 3
    assert list(np.frombuffer(row["returns_blob"], dtype=np.float64)) == pytest.approx(rets)
    assert row["bar_dates_blob"].decode("utf-8").split("\n") == dates


def test_length_mismatch_raises(repo_with_burn):
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    with pytest.raises(ValueError):
        repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                    returns=[0.1, 0.2], bar_dates=["2020-12-31"])


def test_strategy_id_mismatch_raises(repo_with_burn, other_strategy_id):
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    with pytest.raises(ValueError):
        repo.record_holdout_returns(hid, other_strategy_id, holdout_start=h_start, holdout_end=h_end,
                                    returns=[0.1], bar_dates=["2020-12-31"])


def test_unique_prevents_double_write(repo_with_burn):
    repo, sid, hid, (h_start, h_end) = repo_with_burn
    repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                returns=[0.1], bar_dates=["2020-12-31"])
    with pytest.raises(ValueError):  # UNIQUE(holdout_evaluation_id) -> second write rejected
        repo.record_holdout_returns(hid, sid, holdout_start=h_start, holdout_end=h_end,
                                    returns=[0.2], bar_dates=["2020-12-31"])


def test_finalize_requires_matching_strategy_id(repo_pending_reservation, other_strategy_id):
    repo, sid, rid = repo_pending_reservation  # a PENDING (committed_at IS NULL) reservation
    with pytest.raises(ValueError):            # wrong strategy_id -> rowcount 0 -> raise
        repo.finalize_holdout_reservation(rid, config_hash="c", strategy_id=other_strategy_id)
    # correct strategy_id commits the burn
    repo.finalize_holdout_reservation(rid, config_hash="c", strategy_id=sid)
    committed = repo._conn.execute(
        "SELECT committed_at FROM holdout_evaluations WHERE id=?", (rid,)).fetchone()
    assert committed["committed_at"] is not None
```

Add the fixtures (`repo_with_burn`, `repo_pending_reservation`, `other_strategy_id`) in this test file,
building on the schema fixture from Task 1: register a strategy via the repo's `add(...)` (match the
real signature — see `tests/registry/test_factor_ledger.py` or `test_promotion.py` for how strategies
are added), `reserve_holdout(...)` for the interval, and `finalize_holdout_reservation(...,
strategy_id=sid)` to commit a burn. Use a simple ISO interval like `("2020-12-29", "2020-12-31")`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/registry/test_holdout_returns.py -q`
Expected: FAIL — `record_holdout_returns` missing; `finalize_holdout_reservation` has no `strategy_id`.

- [ ] **Step 3: Harden `finalize_holdout_reservation`**

```python
    def finalize_holdout_reservation(
        self, reservation_id: int, *, config_hash: str, strategy_id: int
    ) -> None:
        with self._conn:  # UPDATE + guard commit together or roll back
            cur = self._conn.execute(
                "UPDATE holdout_evaluations SET committed_at = ?, config_hash = ?"
                " WHERE id = ? AND strategy_id = ? AND committed_at IS NULL",
                (_now(), config_hash, reservation_id, strategy_id),
            )
            if cur.rowcount != 1:
                raise ValueError(
                    f"holdout reservation {reservation_id} is missing, already committed, "
                    f"or strategy_id mismatch")
```

- [ ] **Step 4: Add `record_holdout_returns`**

Place it after `release_holdout_reservation`. Validate BEFORE opening the write transaction (avoids
holding a lock for a read):

```python
    def record_holdout_returns(
        self, holdout_evaluation_id: int, strategy_id: int, *,
        holdout_start: str, holdout_end: str,
        returns: list[float], bar_dates: list[str],
    ) -> int:
        """Persist ONE OOS return vector for a committed holdout burn (#221 Slice 1). Separate
        transaction from the burn (the burn committed at on_peek). UNIQUE(holdout_evaluation_id)
        makes a re-run reconciliation safe. Validation is fail-closed and happens before the write."""
        n_bars = len(returns)
        if n_bars != len(bar_dates):
            raise ValueError(
                f"holdout_returns length mismatch: {n_bars} returns vs {len(bar_dates)} dates")
        # strategy_id must match the burn row (caller-bug defense; read before the write tx).
        row = self._conn.execute(
            "SELECT strategy_id FROM holdout_evaluations WHERE id = ?",
            (holdout_evaluation_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"holdout_evaluation {holdout_evaluation_id} does not exist")
        if int(row["strategy_id"]) != int(strategy_id):
            raise ValueError(
                f"strategy_id {strategy_id} does not match holdout_evaluation "
                f"{holdout_evaluation_id} (strategy_id {row['strategy_id']})")
        returns_blob = np.asarray(returns, dtype=np.float64).tobytes()
        bar_dates_blob = "\n".join(bar_dates).encode("utf-8")
        try:
            with self._conn:
                cur = self._conn.execute(
                    "INSERT INTO holdout_returns"
                    "(holdout_evaluation_id, strategy_id, holdout_start, holdout_end, n_bars,"
                    " returns_blob, bar_dates_blob, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (holdout_evaluation_id, strategy_id, holdout_start, holdout_end, n_bars,
                     returns_blob, bar_dates_blob, _now()),
                )
        except sqlite3.IntegrityError as e:  # UNIQUE(holdout_evaluation_id) double-write
            raise ValueError(
                f"holdout_returns already written for holdout_evaluation "
                f"{holdout_evaluation_id}") from e
        rowid = cur.lastrowid
        assert rowid is not None
        return rowid
```

Ensure `import numpy as np` and `import sqlite3` are present in `store.py` (add if missing). Add Protocol
declarations in `repository.py` for both `record_holdout_returns` and the updated
`finalize_holdout_reservation` signature.

- [ ] **Step 5: Update the `on_peek` caller in `research_cmd.py`**

At ~line 164, the burn lambda must pass `strategy_id`. The strategy id is `repo.get(name).id` (already
used at the `reserve_holdout` call). Capture it once and reuse:

```python
        sid = repo.get(name).id
        reservation_id, reused = repo.reserve_holdout(
            sid, data_source=data_source, snapshot_id=snapshot_id, ...)
        ...
                on_peek=lambda cfg: repo.finalize_holdout_reservation(
                    reservation_id, config_hash=cfg, strategy_id=sid),
```

(Keep the existing `repo.get(name).id` call shape; just hoist it to `sid` so both the reserve and the
finalize use the same id.)

- [ ] **Step 6: Run to verify pass + quality gate**

Run: `uv run pytest tests/registry/test_holdout_returns.py -q` → PASS.
Run the full quality gate → all pass (the `finalize_holdout_reservation` signature change is covered by
its only caller in `research_cmd.py` and any test that called it — grep `finalize_holdout_reservation`
in `tests/` and update those call sites to pass `strategy_id=`).

- [ ] **Step 7: Commit**

```bash
git add algua/registry/store.py algua/registry/repository.py algua/cli/research_cmd.py \
        tests/registry/test_holdout_returns.py
git commit -m "feat(221): persist OOS return vector at burn + finalize strategy_id guard — Slice 1"
```

---

## Task 4: sibling-only cross-strategy read accessor + access-control tests

**Files:**
- Modify: `algua/registry/store.py` (add `overlapping_holdout_return_streams`)
- Modify: `algua/registry/repository.py` (Protocol decl)
- Test: extend `tests/registry/test_holdout_returns.py`

**Interfaces:**
- Produces: `overlapping_holdout_return_streams(self, strategy_id: int, holdout_start: str, holdout_end:
  str, window_days: int) -> list[tuple[list[float], list[str]]]` — returns date-aligned SIBLING vectors
  (other strategies' OOS returns over a partially-overlapping interval), NEVER the requesting strategy's
  own vector. The ONLY method that reads `returns_blob`.

- [ ] **Step 1: Write the failing tests**

```python
def test_sibling_read_excludes_own_vector(repo_with_two_strategy_burns):
    # strategy A and sibling B both have overlapping-interval holdout_returns rows.
    repo, a_id, b_id, interval, window = repo_with_two_strategy_burns
    streams = repo.overlapping_holdout_return_streams(a_id, interval[0], interval[1], window)
    # B's vector is returned; A's own is NOT.
    assert len(streams) == 1
    assert all(isinstance(v, tuple) and len(v) == 2 for v in streams)


def test_singleton_funnel_returns_empty(repo_with_burn_and_returns):
    # only the requesting strategy has a vector -> it is its own sibling -> empty.
    repo, sid, interval, window = repo_with_burn_and_returns
    assert repo.overlapping_holdout_return_streams(sid, interval[0], interval[1], window) == []


def test_disjoint_interval_excluded(repo_with_two_strategy_burns_disjoint):
    repo, a_id, b_id, a_interval, window = repo_with_two_strategy_burns_disjoint
    # B's OOS interval does not overlap A's -> not a sibling for this query.
    assert repo.overlapping_holdout_return_streams(a_id, a_interval[0], a_interval[1], window) == []
```

Build the fixtures by writing `holdout_returns` rows for two strategies (reuse `record_holdout_returns`)
with overlapping vs disjoint intervals and recent `holdout_evaluations.created_at`. Confirm the returned
vectors decode correctly (float list + ISO date list).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/registry/test_holdout_returns.py -q -k overlapping or sibling`
Expected: FAIL — `overlapping_holdout_return_streams` missing.

- [ ] **Step 3: Implement the accessor**

The window filter JOINs `holdout_returns` → `holdout_evaluations` on `holdout_evaluation_id` and filters
by `holdout_evaluations.created_at` (burn time), NOT `holdout_returns.created_at`. Self-exclusion is
`hr.strategy_id != ?`. Interval overlap is the standard test `hr.holdout_start <= ? AND ? <=
hr.holdout_end`.

```python
    def overlapping_holdout_return_streams(
        self, strategy_id: int, holdout_start: str, holdout_end: str, window_days: int
    ) -> list[tuple[list[float], list[str]]]:
        """SIBLING-ONLY cross-strategy read (#221 Slice 1 access control): returns OTHER strategies'
        OOS return vectors whose holdout interval overlaps [holdout_start, holdout_end], burned within
        the trailing window_days. NEVER returns the requesting strategy's own vector. This is the ONLY
        method that reads returns_blob. The caller (promotion.run_gate) is trusted to pass the correct
        strategy_id so self-exclusion holds."""
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        rows = self._conn.execute(
            "SELECT hr.n_bars AS n_bars, hr.returns_blob AS rb, hr.bar_dates_blob AS db "
            "FROM holdout_returns hr JOIN holdout_evaluations he"
            " ON hr.holdout_evaluation_id = he.id "
            "WHERE hr.strategy_id != ? AND he.created_at >= ?"
            "  AND hr.holdout_start <= ? AND ? <= hr.holdout_end",
            (strategy_id, cutoff, holdout_end, holdout_start),
        ).fetchall()
        out: list[tuple[list[float], list[str]]] = []
        for r in rows:
            vec = np.frombuffer(r["rb"], dtype=np.float64)
            if len(vec) != int(r["n_bars"]):
                raise ValueError(
                    f"corrupt holdout_returns blob: {len(vec)} floats != n_bars {r['n_bars']}")
            dates = r["db"].decode("utf-8").split("\n")
            out.append(([float(x) for x in vec], dates))
        return out
```

Add the Protocol declaration in `repository.py`. Note in a comment that `holdout_returns` uses
`strategy_id` (FK) while `search_trials` uses `strategy_name` — the asymmetry is intentional.

- [ ] **Step 4: Run to verify pass + quality gate**

Run: `uv run pytest tests/registry/test_holdout_returns.py -q` → PASS.
Run the full quality gate → all pass.

- [ ] **Step 5: Commit**

```bash
git add algua/registry/store.py algua/registry/repository.py tests/registry/test_holdout_returns.py
git commit -m "feat(221): sibling-only overlapping holdout-return-streams read — Slice 1"
```

---

## Task 5: `run_gate` returns-write wiring + `returns_available` audit (PROTECTED)

**Files:**
- Modify: `algua/registry/promotion.py` (`run_gate` ~line 352 — add the param + the write + audit)
- Modify: `algua/research/gates.py` (`GateDecision.returns_available` field ~line 232 + `to_dict`)
- Modify: `algua/cli/research_cmd.py` (pass `reservation_id` into `run_gate` ~line 176)
- Test: extend `tests/test_promotion.py`

**Interfaces:**
- Consumes: `WalkForwardResult.holdout_returns` (Task 2); `repo.record_holdout_returns` (Task 3).
- Produces: `run_gate(..., holdout_evaluation_id: int | None = None)` — when provided AND
  `wf.holdout_returns` is present, writes the returns row (separate tx) and sets
  `decision.returns_available = True`; otherwise `returns_available = False` and NO write.
  `GateDecision.returns_available: bool = False` in `to_dict`.

- [ ] **Step 1: Write the failing tests**

Extend `tests/test_promotion.py`. The existing `_run`/`_run_measured` helpers call `run_gate` with no
`holdout_evaluation_id` — assert they now yield `decision.returns_available is False` and write NO
`holdout_returns` row. Then add an integration test that drives the real burn+write: reserve a holdout,
finalize it (committed burn → a real `holdout_evaluation_id`), build a `wf` carrying `holdout_returns`,
call `run_gate(..., holdout_evaluation_id=hid)`, and assert a `holdout_returns` row exists for `hid`,
`decision.returns_available is True`, and `"returns_available"` is in `decision.to_dict()`. Also assert
the write happens on a FAILED gate (use inputs that fail the gate) — the burn happened, so the vector
persists regardless of pass/fail. (Reuse the burn fixture pattern from `tests/registry/
test_holdout_returns.py`; the strategy id is `repo.get(_GATE_NAME).id`.)

```python
def test_run_gate_no_holdout_id_is_returns_unavailable(tmp_path):
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    outcome = _run(repo, _breadth(repo, "measured"))
    assert outcome.decision.returns_available is False
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_returns").fetchone()["c"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_promotion.py -q -k returns_available`
Expected: FAIL — `GateDecision` has no `returns_available`; `run_gate` has no `holdout_evaluation_id`.

- [ ] **Step 3: Add `returns_available` to `GateDecision`**

In `gates.py` `GateDecision` (with the other audit fields ~line 232) add `returns_available: bool =
False`, and in `to_dict` add `"returns_available": self.returns_available,`. This is the ONLY gate
change in Slice 1 — it is a non-binding audit field; the pass/fail math is untouched.

- [ ] **Step 4: Wire the write into `run_gate`**

Add the keyword param to `run_gate`'s signature: `holdout_evaluation_id: int | None = None`. After
`evaluate_gate(...)` returns `decision` (and before the FDR block), write the returns and set the audit
flag:

```python
    # Persist the OOS return vector for this burn (#221 Slice 1) — separate transaction from the burn
    # (which committed at on_peek). Written on EVERY burn (pass or fail): the holdout was revealed, so
    # the vector exists and funnel siblings may use it. gates.py never sees the vector; promotion is the
    # sole writer. A missing row for a committed burn is a recoverable inconsistency (UNIQUE guards a
    # re-run). returns_available feeds Slices 2-4 (omit-not-fail for pre-Slice-1 promotions).
    returns_available = False
    if holdout_evaluation_id is not None and wf.holdout_returns is not None:
        rets, bar_dates = wf.holdout_returns
        repo.record_holdout_returns(
            holdout_evaluation_id, rec.id,
            holdout_start=wf.holdout_metrics["start"], holdout_end=wf.holdout_metrics["end"],
            returns=rets, bar_dates=bar_dates)
        returns_available = True
    decision.returns_available = returns_available
```

`rec` is `repo.get(name)` — confirm it is already fetched in `run_gate` before this point (it is: the
identity/record block computes `rec = repo.get(name)`); if the write needs to precede that fetch, hoist
`rec = repo.get(name)` up. Use `wf.holdout_metrics["start"]`/`["end"]` for the interval (the same ISO
dates `_segment_record` stored). The write MUST occur before `decision.to_dict()` is serialized into
`gate_row["decision_json"]` so the persisted audit record carries `returns_available`.

- [ ] **Step 5: Pass `reservation_id` into `run_gate` from `research_cmd.py`**

At ~line 176:

```python
        outcome = run_gate(
            repo, wf, name=name, actor=actor_enum, criteria=criteria, breadth=breadth,
            universe_name=universe, universe_snapshots=universe_prov,
            period_start=start_dt.date(), period_end=end_dt.date(), holdout_frac=holdout_frac,
            data_source=data_source, snapshot_id=snapshot_id, allow_non_pit=allow_non_pit,
            holdout_evaluation_id=reservation_id,
            reason_suffix=("; holdout_reuse=" + _HOLDOUT_REUSE_OVERRIDE) if reused else "")
```

- [ ] **Step 6: Run to verify pass + quality gate**

Run: `uv run pytest tests/test_promotion.py -q` → PASS.
Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` → all pass.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/promotion.py algua/research/gates.py algua/cli/research_cmd.py \
        tests/test_promotion.py
git commit -m "feat(221): write OOS vector in run_gate + returns_available audit — Slice 1"
```

---

## Task 6 (process): `sweep()` persists nothing — regression test

**Files:**
- Test: extend `tests/registry/test_holdout_returns.py` (or the existing sweep test module)

The spec requires `sweep()` to produce NO `holdout_returns` rows (single-use discipline preserved by
persisting only at the one burn point). Add a test: run a small `sweep()` (reuse the harness from the
existing sweep test, e.g. `tests/test_sweep*.py`) against a repo, then assert
`SELECT COUNT(*) FROM holdout_returns == 0`. If `sweep()` does not touch the registry DB at all (likely),
this is a cheap guard that documents the invariant. Commit:

```bash
git commit -am "test(221): assert sweep() persists no holdout_returns rows — Slice 1"
```

---

## Self-Review notes

- **Spec coverage:** `holdout_returns` table + indexes + schema bump (T1); SENSITIVE
  `WalkForwardResult.holdout_returns` + `to_dict` exclusion (T2); two-transaction write at the burn +
  `finalize` strategy_id hardening + write-time validation + UNIQUE (T3); sibling-only read + access
  control "only the sibling-read accessor exists" (T4); `returns_available` audit + `run_gate` wiring +
  `sweep()` persists nothing (T5/T6). The partial-write/reconciliation scenario is covered by the
  UNIQUE double-write test (T3).
- **No gate-behavior change:** the only `gates.py` change is the non-binding `returns_available` field;
  `evaluate_gate` math is untouched — assert existing promotion tests' pass/fail are unchanged.
- **Schema number:** 26→27 (NOT the spec's stale 24→25). Stated in Global Constraints + T1.
- **run_gate callers:** production (`research_cmd`) passes `holdout_evaluation_id`; the two
  `tests/test_promotion.py` helpers pass nothing (None ⇒ no write) — backward compatible.
- **Access control:** no CLI accessor, no get-own-vector API, sibling-read self-excludes; `data
  inspect`/`verify` untouched (they read the data manifest, not the registry DB).
- **Type consistency:** `holdout_returns: tuple[list[float], list[str]] | None` (T2) is consumed in T5;
  `record_holdout_returns`/`overlapping_holdout_return_streams`/`finalize_holdout_reservation(...,
  strategy_id)` signatures consistent across store + repository + callers.
- **Deferred (NOT this slice):** any consumer of the persisted vectors (bootstrap, N_eff, multi-regime)
  — Slices 2/3/4; the `returns_available` omit-not-fail binding logic lands with those consumers.
