# `forward_tested` Stage + Forward-Test Evidence Gate Implementation Plan (#124)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `forward_tested` lifecycle stage between `paper` and `live`, reached (for an agent) only by a passing forward-test evidence gate over identity-bound wall-clock paper ticks, with the human live-signature wall additionally demanding a fresh forward certificate.

**Architecture:** Mirrors `research promote`: pure criteria in new protected `algua/research/forward_gates.py`; evidence assembly + orchestration + certificate verification in new protected `algua/registry/forward_promotion.py`; single-use identity-matched token in a new `forward_gate_evaluations` table consumed atomically by `transitions.py`. Evidence = `tick_snapshots` rows newly stamped with lane/identity/account/clock provenance (schema v21). Spec: `docs/superpowers/specs/2026-06-10-forward-tested-stage-gate-issue-124-design.md` — read it before starting any task.

**Tech Stack:** Python 3.12, sqlite3, typer CLI, pandas, exchange_calendars (via `algua.calendar.market_calendar.MarketCalendar`), requests (Alpaca REST), pytest.

**Quality gate between tasks:** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

**Conventions that bind every task:**
- TDD: failing test first, then implementation, then green, then commit.
- NO compat shims, NO default-parameter escape hatches on the new stamped columns — new writer params are required keywords; legacy rows stay NULL and are inadmissible.
- All timestamps ISO-8601 UTC strings in the DB (existing convention).
- `Actor`/`Stage` come from `algua.contracts.lifecycle`.

---

### Task 1: Lifecycle contract — new stage + edges

**Files:**
- Modify: `algua/contracts/lifecycle.py` (protected)
- Test: `tests/test_lifecycle.py` (exists — extend), plus fix any test asserting `paper -> live`

- [ ] **Step 1: Write failing tests** (append to `tests/test_lifecycle.py`):

```python
def test_forward_tested_edges():
    assert can_transition(Stage.PAPER, Stage.FORWARD_TESTED)
    assert can_transition(Stage.FORWARD_TESTED, Stage.LIVE)
    assert can_transition(Stage.FORWARD_TESTED, Stage.PAPER)
    assert can_transition(Stage.FORWARD_TESTED, Stage.RETIRED)  # derived retire edge

def test_paper_to_live_removed_for_everyone():
    assert not can_transition(Stage.PAPER, Stage.LIVE)

def test_live_demotion_still_lands_at_paper():
    assert can_transition(Stage.LIVE, Stage.PAPER)
    assert not can_transition(Stage.LIVE, Stage.FORWARD_TESTED)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_lifecycle.py -q` — expect FAIL (`FORWARD_TESTED` undefined).

- [ ] **Step 3: Implement** in `algua/contracts/lifecycle.py`: add `FORWARD_TESTED = "forward_tested"` between `PAPER` and `LIVE` in `Stage`, and replace the `PAPER`/add `FORWARD_TESTED` entries:

```python
    Stage.PAPER: {Stage.FORWARD_TESTED, Stage.CANDIDATE},
    Stage.FORWARD_TESTED: {Stage.LIVE, Stage.PAPER},
    Stage.LIVE: {Stage.PAPER},
```

- [ ] **Step 4: Run the full suite** `uv run pytest -q`. Tests that previously transitioned `paper -> live` (live-gate, store, transitions, CLI tests) now fail on the removed edge: fix each by inserting a human raw step `paper -> forward_tested` before the live transition (e.g. `transition_strategy(repo, name, Stage.FORWARD_TESTED, Actor.HUMAN, "test setup")`). Do NOT weaken any assertion — the live wall behavior itself is unchanged. (Live-wall certificate enforcement arrives in Task 11; until then `forward_tested -> live` works exactly as `paper -> live` did.)

- [ ] **Step 5: Full gate green, commit** `git commit -m "feat(124): forward_tested lifecycle stage; paper->live edge removed"`

---

### Task 2: Schema v21 — stamped tick provenance + forward_gate_evaluations

**Files:**
- Modify: `algua/registry/db.py`
- Test: `tests/test_db.py` (or the existing migration test module — find with `grep -rl "SCHEMA_VERSION" tests/`)

- [ ] **Step 1: Failing test** — create a v20-shaped DB (just run current `migrate()` against a tmp file before editing, or insert a legacy `tick_snapshots` row post-migrate) and assert the new columns exist and legacy rows read back NULL:

```python
def test_v21_adds_tick_provenance_and_forward_gate_table(tmp_path):
    conn = connect(tmp_path / "r.db"); migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(tick_snapshots)")}
    assert {"lane", "code_hash", "config_hash", "dependency_hash", "strategy_id",
            "account_id", "cash", "clock_source", "recorded_at"} <= cols
    ocols = {r["name"] for r in conn.execute("PRAGMA table_info(paper_orders)")}
    assert "strategy_id" in ocols
    fcols = {r["name"] for r in conn.execute("PRAGMA table_info(forward_gate_evaluations)")}
    assert {"strategy_id", "passed", "realized_sharpe", "holdout_sharpe", "first_tick_id",
            "last_tick_id", "n_concurrent_forward", "consumed", "created_at"} <= fcols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 21
```

- [ ] **Step 2: Run it** — FAIL (no such columns).

- [ ] **Step 3: Implement** in `db.py`:
  - `SCHEMA_VERSION = 21`.
  - Append to `_SCHEMA` the `forward_gate_evaluations` DDL exactly as in the spec's Token-table list (every column listed there; `passed`/`consumed` INTEGER, metrics REAL, counts INTEGER, hashes/actor/decision_json/created_at TEXT; `strategy_id INTEGER NOT NULL REFERENCES strategies(id)`), plus `CREATE INDEX IF NOT EXISTS ix_forward_gate_strategy ON forward_gate_evaluations(strategy_id);`
  - In `migrate()`, follow the existing `_add_missing_columns` pattern (db.py ~366-379) to add to `tick_snapshots`: `lane TEXT`, `code_hash TEXT`, `config_hash TEXT`, `dependency_hash TEXT`, `strategy_id INTEGER`, `account_id TEXT`, `cash REAL`, `clock_source TEXT`, `recorded_at TEXT`; and to `paper_orders`: `strategy_id INTEGER`. (SQLite ALTER TABLE can't add CHECK constraints — the `lane`/`clock_source` value discipline is enforced by the writers, Task 5; note this in a comment.)

- [ ] **Step 4: Test green; full gate; commit** `git commit -m "feat(124): schema v21 — tick provenance columns + forward_gate_evaluations"`

---

### Task 3: Alpaca broker — clock, account id, windowed exhaustive activities

**Files:**
- Modify: `algua/execution/alpaca_broker.py`
- Test: `tests/test_alpaca_broker.py` (exists — follow its requests-mocking pattern)

- [ ] **Step 1: Failing tests** — clock returns the venue timestamp; account() carries the account id; `account_activities_window` paginates exhaustively and raises on malformed pages:

```python
def test_clock_returns_broker_timestamp(...):  # mock GET /v2/clock -> {"timestamp": "2026-06-11T14:00:00-04:00", ...}
def test_account_carries_id(...):              # mock GET /v2/account -> {..., "id": "abc-123"} ; assert .account_id == "abc-123"
def test_activities_window_paginates(...):     # two pages of 100 + final short page -> concatenated, oldest-first
def test_activities_window_fails_closed(...):  # non-list body, or page item missing "id" while full -> BrokerError
```

- [ ] **Step 2: Run** — FAIL (methods missing).

- [ ] **Step 3: Implement** in `_AlpacaBroker`:

```python
@dataclass(frozen=True)
class AccountState:
    equity: float
    cash: float
    buying_power: float
    account_id: str = ""   # new, populated by account(); "" only in legacy constructions
```

```python
    def account(self) -> AccountState:
        data = self._read(self._get("/v2/account"), "/v2/account")
        account_id = str(data.get("id") or "")
        if not account_id:
            raise BrokerError("alpaca /v2/account: missing account id")
        return AccountState(equity=..., cash=..., buying_power=..., account_id=account_id)

    def clock(self) -> str:
        """The venue's own wall-clock timestamp (GET /v2/clock). Evidence ticks use THIS as
        tick_ts so a skewed local clock cannot fabricate session spread (#124)."""
        data = self._read(self._get("/v2/clock"), "/v2/clock")
        ts = data.get("timestamp") if isinstance(data, dict) else None
        if not ts:
            raise BrokerError("alpaca /v2/clock: missing timestamp")
        return str(ts)

    _ACTIVITIES_PAGE = 100

    def account_activities_window(self, after: str, until: str) -> list[dict[str, Any]]:
        """ALL account activities in (after, until], oldest-first, exhaustively paginated.
        Raises BrokerError on ANY failure or malformed page — the forward gate treats the
        account as unverifiable and FAILS, never passes, on partial history (#124)."""
        out: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            path = (f"/v2/account/activities?after={after}&until={until}"
                    f"&direction=asc&page_size={self._ACTIVITIES_PAGE}")
            if page_token:
                path += f"&page_token={page_token}"
            data = self._read(self._get(path), path)
            if not isinstance(data, list):
                raise BrokerError(f"alpaca activities: expected list, got {type(data).__name__}")
            for item in data:
                if not isinstance(item, dict):
                    raise BrokerError(f"alpaca activities: malformed item {item!r}")
            out.extend(data)
            if len(data) < self._ACTIVITIES_PAGE:
                return out
            token = str(data[-1].get("id") or "")
            if not token:
                raise BrokerError("alpaca activities: full page without item id; "
                                  "cannot paginate exhaustively — failing closed")
            page_token = token
```

  Normalize `tick_ts` storage: callers convert the venue timestamp to UTC ISO via `pd.Timestamp(ts).tz_convert("UTC").isoformat()` (done in Task 5, not here).

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): broker clock/account-id/exhaustive windowed activities"`

---

### Task 4: MarketCalendar — session arithmetic

**Files:**
- Modify: `algua/calendar/market_calendar.py`
- Test: `tests/test_market_calendar.py`

- [ ] **Step 1: Failing tests** (known XNYS dates: 2024-07-04 is a holiday, 2024-07-06/07 a weekend):

```python
def test_session_on_or_before():
    cal = MarketCalendar()
    assert cal.session_on_or_before(date(2024, 7, 5)) == date(2024, 7, 5)
    assert cal.session_on_or_before(date(2024, 7, 4)) == date(2024, 7, 3)
    assert cal.session_on_or_before(date(2024, 7, 7)) == date(2024, 7, 5)

def test_sessions_between():
    cal = MarketCalendar()
    assert cal.sessions_between(date(2024, 7, 3), date(2024, 7, 3)) == 0
    assert cal.sessions_between(date(2024, 7, 3), date(2024, 7, 5)) == 1   # holiday skipped
    assert cal.sessions_between(date(2024, 7, 3), date(2024, 7, 8)) == 2   # weekend skipped
    assert cal.sessions_between(date(2024, 7, 8), date(2024, 7, 3)) == -2  # signed
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement:**

```python
    def session_on_or_before(self, day: date) -> date:
        """The trading session containing `day`, or the latest session before it."""
        return self._cal.date_to_session(pd.Timestamp(day), direction="previous").date()

    def sessions_between(self, a: date, b: date) -> int:
        """Signed count of trading sessions from session-of(a) to session-of(b): 0 when both
        map to the same session, positive when b is later. Non-session days map backward
        (session_on_or_before) first."""
        sa, sb = self.session_on_or_before(a), self.session_on_or_before(b)
        lo, hi = (sa, sb) if sb >= sa else (sb, sa)
        n = len(self._cal.sessions_in_range(pd.Timestamp(lo), pd.Timestamp(hi))) - 1
        return n if sb >= sa else -n
```

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): calendar session arithmetic for forward-gate windows"`

---

### Task 5: Stamped writers — trade-tick provenance + stage acceptance

**Files:**
- Modify: `algua/execution/order_state.py` (`record_tick_snapshot`, `record_submitted_order`, `latest_tick_snapshot`)
- Modify: `algua/cli/paper_cmd.py` (`trade_tick`, `_load_gated_strategy`)
- Modify: `algua/cli/live_cmd.py` (its `record_tick_snapshot` call site)
- Test: `tests/test_order_state.py`, `tests/test_cli_paper.py`

- [ ] **Step 1: Failing tests** — `record_tick_snapshot` requires and persists the new fields; `_load_gated_strategy` accepts `FORWARD_TESTED`:

```python
def test_record_tick_snapshot_stamps_provenance(conn):
    record_tick_snapshot(conn, "s", tick_ts="2026-06-11T14:00:00+00:00", decision_ts=None,
                         equity=1.0, peak_equity=None, positions={}, n_submitted=0,
                         reconcile_ok=True, lane="paper", strategy_id=7, code_hash="c",
                         config_hash="g", dependency_hash="d", account_id="acct", cash=1.0,
                         clock_source="broker")
    row = conn.execute("SELECT * FROM tick_snapshots").fetchone()
    assert row["lane"] == "paper" and row["strategy_id"] == 7 and row["clock_source"] == "broker"
    assert row["recorded_at"]  # writer-stamped
```

  CLI test: monkeypatch the broker as the existing trade-tick tests do; assert the persisted row carries `lane='paper'`, the strategy's registry id, identity hashes equal to `compute_artifact_hashes(name)`, the mocked account id/cash, and `clock_source='broker'`; a second test makes `clock()` raise `BrokerError` and asserts `clock_source='local'`. A third registers a strategy at `forward_tested` (human raw transition) and asserts `trade-tick` still runs.

- [ ] **Step 2: Run** — FAIL (unexpected keyword args).

- [ ] **Step 3: Implement:**
  - `record_tick_snapshot(...)` gains REQUIRED keyword params `lane: str, strategy_id: int, code_hash: str, config_hash: str, dependency_hash: str | None, account_id: str, cash: float, clock_source: str`; asserts `lane in ("paper", "live")` and `clock_source in ("broker", "local")`; INSERT includes all new columns plus `recorded_at=datetime.now(UTC).isoformat()`.
  - `record_submitted_order(...)` gains required `strategy_id: int` keyword, written to the new column.
  - `latest_tick_snapshot` SELECT keeps working (add the new fields to the returned dict).
  - In `paper_cmd.trade_tick`, before `run_tick`: `identity = compute_artifact_hashes(name)` (import from `algua.registry.approvals`), `acct = broker.account()`. After the tick, replace the snapshot call:

```python
        try:
            raw_ts = broker.clock()
            tick_ts = pd.Timestamp(raw_ts).tz_convert("UTC").isoformat()
            clock_source = "broker"
        except BrokerError:
            tick_ts, clock_source = datetime.now(UTC).isoformat(), "local"
        record_tick_snapshot(
            conn, name, tick_ts=tick_ts, decision_ts=...,  # unchanged args elided
            lane="paper", strategy_id=rec.id, code_hash=identity.code_hash,
            config_hash=identity.config_hash, dependency_hash=identity.dependency_hash,
            account_id=acct.account_id, cash=acct.cash, clock_source=clock_source,
        )
```

    (`rec` = the registry record — `_load_gated_strategy` currently loads it internally; have it return `(strategy, rec)` or fetch `rec = SqliteStrategyRepository(conn).get(name)` alongside. Pass `strategy_id=rec.id` to the `_persist` order hook too.)
  - `live_cmd`'s call site: same stamping with `lane="live"`.
  - `_load_gated_strategy`: stage check becomes `rec.stage not in (Stage.PAPER, Stage.FORWARD_TESTED)` with an updated error message.

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): trade-tick stamps lane/identity/account/clock provenance"`

---

### Task 6: `forward_gates.py` — pure criteria (protected, NEW)

**Files:**
- Create: `algua/research/forward_gates.py`
- Test: `tests/test_forward_gates.py` (NEW)

- [ ] **Step 1: Failing table-driven tests.** Build a helper returning a fully-passing `ForwardEvidence`, then one test per check flipping exactly one field across the boundary:

```python
def passing_evidence(**over):
    base = dict(n_return_observations=63, session_coverage=0.95, realized_sharpe=0.8,
                realized_vol=0.10, realized_max_drawdown=0.10, holdout_sharpe=1.0,
                n_reconcile_failures=0, n_defective_ticks=0, kill_switch_tripped=False,
                global_halt_engaged=False, n_kill_trips_in_window=0, single_tenant_ok=True,
                activities_ok=True, n_external_cash_flows=0, n_unattributable_fills=0,
                staleness_sessions=2)
    return ForwardEvidence(**(base | over))

# pass case; then each of:
# n_return_observations=62 -> fail ; session_coverage=0.89 -> fail
# realized_sharpe just under max(0.5*holdout, 0.3): holdout_sharpe=1.0, realized_sharpe=0.49 -> fail
# holdout_sharpe=0.4 (bar becomes the 0.3 floor): realized_sharpe=0.31 -> pass, 0.29 -> fail
# holdout_sharpe=None -> fail (no qualified backtest row)
# realized_vol=0.01 -> fail ; realized_max_drawdown=0.26 -> fail
# any of n_reconcile_failures/n_defective_ticks/n_kill_trips_in_window = 1 -> fail
# kill_switch_tripped/global_halt_engaged True -> fail ; single_tenant_ok False -> fail
# activities_ok False, or n_external_cash_flows/n_unattributable_fills = 1 -> fail
# staleness_sessions=6 -> fail ; staleness_sessions=None -> fail (no admissible ticks)
# non-finite realized_sharpe (float('nan')) -> fail closed
```

- [ ] **Step 2: Run** — FAIL (module missing). **Step 3: Implement** `algua/research/forward_gates.py`. Module docstring states it is CODEOWNERS-protected and why. Contents: protected module constants and dataclasses exactly as named in the spec —

```python
MIN_FORWARD_OBSERVATIONS = 63        # symmetric with MIN_HOLDOUT_OBSERVATIONS (returns, not equity points)
MIN_SESSION_COVERAGE = 0.9
DEGRADATION_FACTOR = 0.5
SHARPE_FLOOR = 0.3
MIN_FORWARD_VOL = 0.02               # annualized; below this Sharpe is undefined -> fail closed
MAX_FORWARD_DRAWDOWN = 0.25
MAX_STALENESS_SESSIONS = 5
FORWARD_TOKEN_TTL_DAYS = 7           # consumable-token freshness (transitions.py)
CERTIFICATE_FRESH_SESSIONS = 10      # live-wall certificate freshness (forward_promotion.py)

@dataclass
class ForwardGateCriteria:           # one field per constant above except the last two
    ...

@dataclass(frozen=True)
class ForwardEvidence:               # exactly the fields used in the tests above
    ...

@dataclass
class ForwardGateDecision:
    passed: bool
    checks: list[dict[str, Any]]
    def to_dict(self) -> dict[str, Any]: ...
```

  `evaluate_forward_gate(ev, criteria) -> ForwardGateDecision` builds `checks` in spec order. Mirror `gates.py`'s finite-guard discipline: a non-finite metric value records `value=None, passed=False` (fail closed, JSON-clean). Metric checks carry `{name, value, op, threshold, passed}`; boolean checks `{name, passed, detail}`. The performance check with `holdout_sharpe=None` is a failed check with `detail="no qualified backtest gate row for current identity"`. `passed = all(c["passed"] for c in checks)`.

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): forward-gate criteria evaluator (protected)"`

---

### Task 7: Evidence assembly (forward_promotion.py part 1, protected, NEW)

**Files:**
- Create: `algua/registry/forward_promotion.py`
- Test: `tests/test_forward_promotion.py` (NEW)

- [ ] **Step 1: Failing tests** against a seeded tmp DB (use `migrate()` + direct INSERTs via `record_tick_snapshot`). Use a **fake calendar** (`sessions_between`/`session_on_or_before`/`sessions_in_range` over weekdays) and a **fake activities_fetch** so tests are hermetic. Cover, minimum: admissible filtering per dimension (lane, clock_source, identity, account NULL, stale decision session > 2, future tick_ts), last-tick-per-decision-session collapse, return/coverage math on a known equity path, the integrity universe catching a reconcile-failed *inadmissible* row, staleness, single-tenant violation, sibling-on-other-account incrementing `n_concurrent_forward`, activities classification (CSD fails, DIV passes, unattributable FILL fails, BrokerError -> `activities_ok=False`), holdout-row qualification (passing+pit_ok+non-override+identity-matched found; failing row or drifted identity -> None).

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** in `algua/registry/forward_promotion.py`:

```python
EXTERNAL_CAPITAL_TYPES = frozenset({"CSD", "CSW", "TRANS", "JNLC", "JNLS", "ACATC", "ACATS"})

@dataclass(frozen=True)
class AssembledEvidence:
    evidence: ForwardEvidence
    first_tick_id: int | None
    last_tick_id: int | None
    first_tick_ts: str | None
    last_tick_ts: str | None
    account_id: str | None
    n_concurrent_forward: int
    excluded: dict[str, int]      # per-filter exclusion counts for the CLI payload

def assemble_forward_evidence(conn, *, strategy_id: int, name: str,
                              identity: ArtifactIdentity, calendar, now: datetime,
                              activities_fetch) -> AssembledEvidence:
```

  Assembly algorithm (each numbered item maps to a spec clause):
  1. `rows = SELECT id, tick_ts, decision_ts, equity, reconcile_ok, clock_source, code_hash, config_hash, dependency_hash, account_id, recorded_at FROM tick_snapshots WHERE lane='paper' AND strategy_id=? ORDER BY id` .
  2. Partition into admissible/excluded in Python, counting exclusions per filter, in this order: `clock_source != 'broker'` → `local_clock`; identity mismatch (any of the three hashes, NULL never matches) → `identity_drift`; `account_id` NULL → `legacy_null`; unparseable or future `tick_ts` (`> now`) → `bad_tick_ts`; NULL `decision_ts` → `no_decision` (excluded from observations only); `not (0 <= calendar.sessions_between(decision_date, tick_date) <= 2)` → `stale_decision`.
  3. Observations: key admissible ticks by `calendar.session_on_or_before(decision_ts.date())`; keep the max-`id` row per session; sort by session; `equity` series → `returns = equity.pct_change().dropna()`; `m = metrics_from_returns(returns)` (import from `algua.backtest.metrics` — reuse, don't reimplement); `realized_sharpe=m["sharpe"]`, `realized_vol=m["ann_volatility"]`, `realized_max_drawdown=abs(m["max_drawdown"])`; `n_return_observations=len(returns)`.
  4. Coverage: `len(sessions_with_observation) / len(calendar.sessions_in_range(first_obs_session, last_obs_session))` (0.0 when no observations).
  5. Integrity universe: every row from step 1 with `id >= first_admissible_id`; `n_reconcile_failures = count(reconcile_ok == 0)`; `n_defective_ticks = count(unparseable-or-future tick_ts)`.
  6. `kill_switch_tripped = kill_switch.is_tripped(conn, name)`; `global_halt_engaged = global_halt.is_engaged(conn)`; `n_kill_trips_in_window = SELECT COUNT(*) FROM audit_log WHERE strategy=? AND action='kill_switch_trip' AND ts >= first_admissible_recorded_at`.
  7. Single-tenant: `SELECT COUNT(*) FROM tick_snapshots WHERE lane='paper' AND account_id=? AND strategy_id != ? AND recorded_at >= ? AND recorded_at <= ?` over `[first_admissible_recorded_at, now]` → `single_tenant_ok = (count == 0)`.
  8. `n_concurrent_forward = SELECT COUNT(DISTINCT strategy) FROM tick_snapshots WHERE lane='paper' AND recorded_at >= ? AND recorded_at <= ?` (any admissibility, any account — spec).
  9. Activities: `acts = activities_fetch(first_tick_ts, now_iso)` inside try/except `Exception` → `activities_ok=False`. Classify: `activity_type in EXTERNAL_CAPITAL_TYPES` → `n_external_cash_flows += 1`; `activity_type == "FILL"` → attributable iff `SELECT 1 FROM paper_orders WHERE strategy_id=? AND broker_order_id=?` matches `act["order_id"]`; else `n_unattributable_fills += 1`. Everything else (DIV/INT/FEE/...) passes.
  10. Staleness: `calendar.sessions_between(last_admissible_tick_date, now_date)`; `None` when no admissible ticks.
  11. `holdout_sharpe = qualified_holdout_sharpe(conn, strategy_id, identity)`:

```python
def qualified_holdout_sharpe(conn, strategy_id, identity) -> float | None:
    """RAW measured holdout Sharpe from the newest QUALIFIED backtest gate row: passed=1,
    pit_ok=1, pit_override=0, identity == current. None -> the forward gate fails closed."""
    if identity.dependency_hash is None:
        return None
    row = conn.execute(
        "SELECT decision_json FROM gate_evaluations WHERE strategy_id=? AND passed=1"
        " AND pit_ok=1 AND pit_override=0 AND code_hash=? AND config_hash=?"
        " AND dependency_hash=? ORDER BY id DESC LIMIT 1",
        (strategy_id, identity.code_hash, identity.config_hash, identity.dependency_hash),
    ).fetchone()
    if row is None:
        return None
    checks = json.loads(row["decision_json"]).get("checks", [])
    vals = [c.get("value") for c in checks if c.get("name") == "holdout_sharpe"]
    return float(vals[0]) if vals and vals[0] is not None else None
```

  Skip empty-evidence early: zero admissible ticks → evidence with zeros, `staleness_sessions=None`, all derived checks fail naturally.

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): forward-evidence assembly (protected)"`

---

### Task 8: Store/repository — forward-token records + CAS

**Files:**
- Modify: `algua/registry/store.py`, `algua/registry/repository.py`
- Test: `tests/test_store.py` (or wherever `record_gate_evaluation` tests live)

- [ ] **Step 1: Failing tests:**

```python
# record_forward_gate_evaluation returns row id; persists pass AND fail rows
# find_consumable_forward_gate_evaluation: newest passing agent unconsumed identity-matched
#   row within TTL; returns None for: failed row, consumed row, human row, identity drift,
#   NULL dependency probe, created_at older than ttl
# latest_forward_gate_row: newest row PASS OR FAIL for (strategy_id, identity); None on drift
# apply_transition(consume_forward_gate_id=...): consumes atomically; rowcount!=1 -> TransitionError
#   and the stage does NOT advance; both consume params non-None -> ValueError
# consume-time predicate recheck: token recorded under identity A, apply_transition passed
#   identity B (code_hash arg differs) -> TransitionError, stage unchanged
# CAS: two repos over the same DB; repo2 applies a transition after repo1 read rec; repo1's
#   apply_transition raises TransitionError mentioning "concurrent"
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement:**
  - `record_forward_gate_evaluation(strategy_id, *, passed, n_forward_observations, min_forward_observations, session_coverage, realized_sharpe, holdout_sharpe, degradation_factor, sharpe_floor, realized_vol, min_forward_vol, realized_max_drawdown, max_forward_drawdown, first_tick_id, last_tick_id, first_tick_ts, last_tick_ts, max_staleness_sessions, n_reconcile_failures, n_concurrent_forward, account_id, code_hash, config_hash, dependency_hash, actor, decision_json) -> int` — mirrors `record_gate_evaluation` (store.py:411).
  - `find_consumable_forward_gate_evaluation(strategy_id, code_hash, config_hash, dependency_hash, *, now: str, ttl_days: int) -> int | None` — the `gate_evaluations` query shape plus `AND created_at >= ?` computed as `(datetime.fromisoformat(now) - timedelta(days=ttl_days)).isoformat()`; NULL dependency probe returns None.
  - `latest_forward_gate_row(strategy_id, code_hash, config_hash, dependency_hash) -> dict | None` — newest row regardless of `passed`/`consumed` (the live wall's pass-or-fail selection); returns the full row as a dict; NULL dependency probe → None.
  - `apply_transition` gains `consume_forward_gate_id: int | None = None`; raises `ValueError` if both consume ids are non-None; the forward consume UPDATE re-checks the FULL predicate set (the shortlist consume keeps its current shape — tightening it is out of scope):

```python
            if consume_forward_gate_id is not None:
                cutoff = (datetime.now(UTC) - timedelta(days=FORWARD_TOKEN_TTL_DAYS)).isoformat()
                cur = self._conn.execute(
                    "UPDATE forward_gate_evaluations SET consumed=1"
                    " WHERE id=? AND strategy_id=? AND passed=1 AND actor='agent' AND consumed=0"
                    " AND code_hash=? AND config_hash=? AND dependency_hash=? AND created_at>=?",
                    (consume_forward_gate_id, rec.id, code_hash, config_hash,
                     dependency_hash, cutoff))
                if cur.rowcount != 1:
                    raise TransitionError(
                        f"forward gate evaluation {consume_forward_gate_id} is not a consumable"
                        f" agent token for this strategy+identity (consumed, missing, drifted,"
                        f" or expired)")
```

    (For this recheck `transitions.py` must pass the recomputed identity through `code_hash/config_hash/dependency_hash` — Task 9 does that. Import `FORWARD_TOKEN_TTL_DAYS` from `algua.research.forward_gates`.)
  - CAS: the stage UPDATE becomes

```python
            cur = self._conn.execute(
                "UPDATE strategies SET stage = ?, updated_at = ? WHERE id = ? AND stage = ?",
                (to.value, now, rec.id, from_stage.value))
            if cur.rowcount != 1:
                raise TransitionError(
                    f"concurrent transition detected for {rec.name!r}: stage is no longer"
                    f" {from_stage.value!r} (another session moved it); re-read and retry")
```

  - Add all three new methods + the new `apply_transition` param to the `StrategyRepository` Protocol in `repository.py` with docstrings.

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): forward-gate token store + CAS stage updates (protected)"`

---

### Task 9: transitions.py — scoped gate branches + forward token validation

**Files:**
- Modify: `algua/registry/transitions.py`
- Test: `tests/test_transitions.py`

- [ ] **Step 1: Failing tests:**

```python
# agent PAPER -> CANDIDATE back-step succeeds WITHOUT any gate token (scoping fix)
# agent BACKTESTED -> CANDIDATE still demands the shortlist token (regression)
# agent PAPER -> FORWARD_TESTED with a fresh passing token: succeeds, token consumed
# agent PAPER -> FORWARD_TESTED without a token: TransitionError mentioning `algua paper promote`
# human PAPER -> FORWARD_TESTED raw transition: succeeds with no token, and the transition row
#   records the recomputed identity hashes (audit pinning)
# second consume attempt (re-promote after demote without a new gate run): TransitionError
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** in `transition_strategy`:

```python
    if target == Stage.LIVE:
        identity = _validate_live_gate(...)            # unchanged here; Task 11 extends it
        code_hash, config_hash, dependency_hash = identity
    elif (rec.stage is Stage.BACKTESTED and target == Stage.CANDIDATE
          and transition_actor is not Actor.HUMAN):
        # Wall D, scoped to the FORWARD edge: the PAPER -> CANDIDATE back-step is free for any
        # actor (re-entry to candidate from below always re-runs the research gate).
        consume_gate_id = _validate_shortlist_gate(repo=repo, name=name, strategy_id=rec.id)
    elif rec.stage is Stage.PAPER and target == Stage.FORWARD_TESTED:
        identity = _compute_hashes(name)
        code_hash, config_hash, dependency_hash = identity
        if transition_actor is not Actor.HUMAN:
            consume_forward_gate_id = _validate_forward_gate(
                repo=repo, strategy_id=rec.id, identity=identity)
```

  (`code_hash`/etc. set for the human raw path too — that is the audit pinning; `apply_transition` already writes them to the transition row.)

```python
def _validate_forward_gate(*, repo: StrategyRepository, strategy_id: int,
                           identity: ArtifactIdentity) -> int:
    """Newest fresh passing AGENT forward token matching the current identity, or raise.
    Consumption (with full predicate recheck) happens inside apply_transition."""
    from algua.research.forward_gates import FORWARD_TOKEN_TTL_DAYS
    gate_id = repo.find_consumable_forward_gate_evaluation(
        strategy_id, identity.code_hash, identity.config_hash, identity.dependency_hash,
        now=datetime.now(UTC).isoformat(), ttl_days=FORWARD_TOKEN_TTL_DAYS)
    if gate_id is None:
        raise TransitionError(
            "transition to forward_tested requires a fresh passing forward-gate evaluation for"
            " the current code+config+dependency; run `algua paper promote`")
    return gate_id
```

  Pass `consume_forward_gate_id` through to `repo.apply_transition`.

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): source-scoped gate branches + forward token validation (protected)"`

---

### Task 10: Orchestration — preflight + run_forward_gate (forward_promotion.py part 2)

**Files:**
- Modify: `algua/registry/forward_promotion.py`
- Test: `tests/test_forward_promotion.py`

- [ ] **Step 1: Failing tests:**

```python
# preflight: SYSTEM actor refused; stage CANDIDATE/LIVE refused; PAPER and FORWARD_TESTED accepted
# guard_forward_relaxations: agent passing sharpe_floor=0.2 (relaxing) -> ValueError;
#   agent passing sharpe_floor=0.5 (tightening) -> ok; same per flag in its direction
#   (max_drawdown: agent 0.30 refused, 0.20 ok); human may relax anything
# run_forward_gate (fakes for calendar + activities): records a row on PASS and on FAIL;
#   on pass from PAPER transitions to FORWARD_TESTED and consumes the just-minted token;
#   on pass from FORWARD_TESTED records the row, promoted=False, stage unchanged;
#   decision_json round-trips; n_concurrent_forward + first/last tick ids persisted
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement:**

```python
def guard_forward_relaxations(actor: Actor, criteria: ForwardGateCriteria) -> None:
    """Each threshold has a strict direction; an agent may only move it stricter (#124)."""
    if actor is Actor.HUMAN:
        return
    defaults = ForwardGateCriteria()
    higher_is_stricter = ("min_forward_observations", "min_session_coverage",
                          "degradation_factor", "sharpe_floor", "min_forward_vol")
    lower_is_stricter = ("max_forward_drawdown", "max_staleness_sessions")
    relaxed = [f for f in higher_is_stricter if getattr(criteria, f) < getattr(defaults, f)]
    relaxed += [f for f in lower_is_stricter if getattr(criteria, f) > getattr(defaults, f)]
    if relaxed:
        raise ValueError(
            f"forward-gate relaxation requires --actor human: {', '.join(sorted(relaxed))}")

def forward_promotion_preflight(repo, name, *, actor: Actor,
                                criteria: ForwardGateCriteria) -> StrategyRecord:
    if actor not in (Actor.AGENT, Actor.HUMAN):
        raise ValueError(f"paper promote requires --actor agent or human, got {actor.value}")
    guard_forward_relaxations(actor, criteria)
    rec = repo.get(name)
    if rec.stage not in (Stage.PAPER, Stage.FORWARD_TESTED):
        raise TransitionError(
            f"paper promote requires stage paper or forward_tested, got {rec.stage.value}")
    if rec.stage is Stage.PAPER:
        validate_transition(rec.stage, Stage.FORWARD_TESTED)
    return rec

@dataclass
class ForwardPromotionOutcome:
    decision: ForwardGateDecision
    promoted: bool
    assembled: AssembledEvidence

def run_forward_gate(repo, conn, *, name, actor, criteria, calendar, now,
                     activities_fetch) -> ForwardPromotionOutcome:
    rec = repo.get(name)
    identity = compute_artifact_hashes(name)
    asm = assemble_forward_evidence(conn, strategy_id=rec.id, name=name, identity=identity,
                                    calendar=calendar, now=now,
                                    activities_fetch=activities_fetch)
    decision = evaluate_forward_gate(asm.evidence, criteria)
    repo.record_forward_gate_evaluation(rec.id, passed=decision.passed, ...,  # every column,
                                        actor=actor.value,                    # from asm/criteria
                                        decision_json=json.dumps(decision.to_dict(),
                                                                 sort_keys=True))
    promoted = False
    if decision.passed and rec.stage is Stage.PAPER:
        transition_strategy(repo, name, Stage.FORWARD_TESTED, actor,
                            _forward_gate_reason(decision))
        promoted = True
    return ForwardPromotionOutcome(decision=decision, promoted=promoted, assembled=asm)
```

  `_forward_gate_reason` mirrors `promotion._gate_reason`. `activities_fetch` is a `Callable[[str, str], list[dict]]` (after_iso, until_iso) — the CLI binds it to the broker in Task 12; tests pass lambdas.

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): forward-gate orchestration — preflight + run + re-evaluation (protected)"`

---

### Task 11: The live wall demands the certificate

**Files:**
- Modify: `algua/registry/forward_promotion.py` (`verify_forward_certificate`)
- Modify: `algua/registry/transitions.py` (`_validate_live_gate` + injection param)
- Modify: `algua/cli/registry_cmd.py` (challenge issuance surfaces the certificate summary; builds the default activities fetch)
- Test: `tests/test_forward_promotion.py`, `tests/test_transitions.py` (live-wall cases), CLI test

- [ ] **Step 1: Failing tests** (fakes throughout):

```python
# verify_forward_certificate happy path returns a summary dict (realized_sharpe, created_at, ...)
# no row for current identity -> TransitionError("no forward-gate evaluation...")
# newest row is FAILED (an older PASS exists) -> TransitionError (newer-failed invalidation)
# created_at 11 sessions ago -> TransitionError (stale certificate)
# certificate from an identity-identical SIBLING strategy_id is not found (strategy_id binding)
# reconcile-failed paper tick with id > last_tick_id -> TransitionError (clean-since)
# kill_switch_trip audit row after created_at -> TransitionError
# kill switch tripped now / global halt engaged -> TransitionError
# activities over [created_at, now]: CSD -> TransitionError; unattributable FILL -> TransitionError;
#   fetch raising -> TransitionError (fail closed)
# transitions: forward_tested -> live with human+approval but NO certificate -> TransitionError
# registry_cmd challenge issuance output includes {"forward_certificate": {...}}
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement:**

```python
def verify_forward_certificate(repo, conn, *, name: str, strategy_id: int,
                               identity: ArtifactIdentity, calendar, now: datetime,
                               activities_fetch) -> dict[str, Any]:
    """The evidence precondition of the live wall (#124). NOT waivable in-band."""
    row = repo.latest_forward_gate_row(strategy_id, identity.code_hash,
                                       identity.config_hash, identity.dependency_hash)
    if row is None:
        raise TransitionError(
            "go-live requires a forward-test certificate for the current code+config+dependency;"
            " run `algua paper promote`")
    if not row["passed"]:
        raise TransitionError(
            "the newest forward-gate evaluation for the current identity FAILED"
            f" (id {row['id']}, {row['created_at']}); the prior pass is invalidated —"
            " re-run `algua paper promote`")
    age = calendar.sessions_between(
        datetime.fromisoformat(row["created_at"]).date(), now.date())
    if age > CERTIFICATE_FRESH_SESSIONS:
        raise TransitionError(
            f"forward-test certificate is stale ({age} sessions old, max"
            f" {CERTIFICATE_FRESH_SESSIONS}); re-run `algua paper promote` to refresh")
    # clean record since certification (anchored on ungameable row ids)
    bad = conn.execute(
        "SELECT COUNT(*) FROM tick_snapshots WHERE lane='paper' AND strategy_id=?"
        " AND id > ? AND reconcile_ok=0", (strategy_id, row["last_tick_id"] or 0)).fetchone()[0]
    if bad:
        raise TransitionError(f"{bad} reconcile-failed paper tick(s) since certification")
    trips = conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE strategy=? AND action='kill_switch_trip'"
        " AND ts >= ?", (name, row["created_at"])).fetchone()[0]
    if trips:
        raise TransitionError(f"{trips} kill-switch trip(s) since certification")
    if kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn):
        raise TransitionError("kill switch / global halt engaged")
    hygiene = _classify_activities(conn, strategy_id,
                                   _fetch_or_fail(activities_fetch,
                                                  row["created_at"], now.isoformat()))
    if hygiene.n_external_cash_flows or hygiene.n_unattributable_fills:
        raise TransitionError("account hygiene violated since certification "
                              f"({hygiene.n_external_cash_flows} external cash flow(s), "
                              f"{hygiene.n_unattributable_fills} unattributable fill(s))")
    return {"id": row["id"], "created_at": row["created_at"],
            "realized_sharpe": row["realized_sharpe"], "holdout_sharpe": row["holdout_sharpe"],
            "n_forward_observations": row["n_forward_observations"],
            "n_concurrent_forward": row["n_concurrent_forward"]}
```

  (`_classify_activities` is the Task-7 classification factored to be reusable; `_fetch_or_fail` wraps the fetch and converts any exception to `TransitionError` — partial history fails closed.)
  - `transitions.transition_strategy` gains `forward_certificate_verifier: Callable[..., dict] | None = None`; `_validate_live_gate` calls it (or the default, lazily built like `_default_approval_verifier` — the default imports `forward_promotion`, builds a `MarketCalendar()`, an `AlpacaPaperBroker` from settings for `activities_fetch`, and needs the sqlite conn: get it from the repo (`SqliteStrategyRepository` exposes its connection — add a `conn` property if absent)). Tests inject a fake.
  - `registry_cmd.transition` live path: call the verifier BEFORE `issue_challenge` and merge the returned summary into the emitted challenge payload under `"forward_certificate"`. (The transition path re-verifies inside `_validate_live_gate` — issuance-time is UX, transition-time is the wall.)

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): live wall demands a fresh forward certificate (protected)"`

---

### Task 12: CLI — `algua paper promote`

**Files:**
- Modify: `algua/cli/paper_cmd.py`
- Test: `tests/test_cli_paper.py`

- [ ] **Step 1: Failing tests** (CliRunner + monkeypatched broker/calendar fakes, as the existing paper CLI tests do):

```python
# happy path at PAPER: exit 0, payload has promoted=True, decision checks, excluded counts
# failing gate: exit 1, row still recorded (query forward_gate_evaluations), promoted absent/False
# at FORWARD_TESTED: exit 0 on pass with promoted=False
# agent passing --sharpe-floor 0.2: refused (relaxation); --sharpe-floor 0.5 accepted
# wrong stage (candidate): refusal mentioning stage
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** in `paper_cmd.py`:

```python
@paper_app.command("promote")
@json_errors(ValueError, LookupError, BrokerError)
def promote(
    name: str,
    actor: str = typer.Option("agent", "--actor"),
    degradation_factor: float = typer.Option(DEGRADATION_FACTOR, "--degradation-factor"),
    sharpe_floor: float = typer.Option(SHARPE_FLOOR, "--sharpe-floor"),
    min_observations: int = typer.Option(MIN_FORWARD_OBSERVATIONS, "--min-observations"),
    min_coverage: float = typer.Option(MIN_SESSION_COVERAGE, "--min-coverage"),
    min_vol: float = typer.Option(MIN_FORWARD_VOL, "--min-vol"),
    max_drawdown: float = typer.Option(MAX_FORWARD_DRAWDOWN, "--max-drawdown"),
    max_staleness: int = typer.Option(MAX_STALENESS_SESSIONS, "--max-staleness"),
) -> None:
    """Forward-test evidence gate: evaluate this strategy's wall-clock paper evidence and
    promote paper -> forward_tested on pass (at forward_tested: re-evaluate, refreshing the
    live-wall certificate, no stage change). The paper-side analog of `research promote`."""
    criteria = ForwardGateCriteria(min_forward_observations=min_observations, ...)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = forward_promotion_preflight(repo, name, actor=Actor(actor), criteria=criteria)
        broker = _alpaca_broker_from_settings()
        outcome = run_forward_gate(
            repo, conn, name=name, actor=Actor(actor), criteria=criteria,
            calendar=MarketCalendar(), now=datetime.now(UTC),
            activities_fetch=broker.account_activities_window)
        audit_append(conn, actor=actor, action="paper_promote",
                     reason=("pass" if outcome.decision.passed else "fail"), strategy=name)
    payload = {"strategy": name, "passed": outcome.decision.passed,
               "promoted": outcome.promoted, "decision": outcome.decision.to_dict(),
               "excluded_ticks": outcome.assembled.excluded,
               "n_concurrent_forward": outcome.assembled.n_concurrent_forward}
    emit(ok(payload) if outcome.decision.passed else payload | {"status": "fail"})
    if not outcome.decision.passed:
        raise typer.Exit(1)
```

  (Match the existing emit/exit idiom in `research_cmd.promote` — read it first and mirror its failure-payload shape exactly.)

- [ ] **Step 4: Green; full gate; commit** `git commit -m "feat(124): algua paper promote CLI"`

---

### Task 13: CODEOWNERS + docs + skills

**Files:**
- Modify: `CODEOWNERS`, `CLAUDE.md`, `docs/agent/operating.md`
- Modify: `~/.claude/skills/...` — NO. Project skills only: check `.claude/skills/` in-repo; the `operating-algua` and `run-the-research-loop` skills live wherever `grep -rl "shortlisted\|candidate" .claude` finds them; also `kb/` lifecycle notes (`grep -rln "paper -> live\|paper → live" kb/ docs/`).

- [ ] **Step 1:** CODEOWNERS — append:

```
/algua/research/forward_gates.py      @Lior-Nis   # forward-test gate criteria (#124)
/algua/registry/forward_promotion.py  @Lior-Nis   # forward-test evidence assembly + certificate
```

- [ ] **Step 2:** CLAUDE.md — lifecycle line becomes `idea -> backtested -> candidate -> paper -> forward_tested -> live -> retired`; golden rule "up to and including paper" becomes "up to and including forward_tested"; add the `paper promote` command to the command surface with one sentence on the gate + the human-only relaxation flags; note the `forward_tested -> live` certificate requirement on the go-live bullet.

- [ ] **Step 3:** `docs/agent/operating.md` + skills + kb: update stage enumerations and the agent-reach rule wherever stated (grep for `shortlisted`, `candidate`, `paper -> live`, "up to and including paper").

- [ ] **Step 4:** Full gate; commit `git commit -m "docs(124): lifecycle/golden-rule/skill updates for forward_tested"`

---

### Task 14: Final verification

- [ ] Full quality gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` — all green.
- [ ] Smoke the CLI surface end-to-end against a tmp DB (`ALGUA_DB_PATH`): `registry add x` → raw human transitions to paper → `paper promote x --actor agent` fails closed (zero admissible ticks) with the excluded-tick accounting in the payload; exit code 1.
- [ ] `git log --oneline main..HEAD` — every task committed; no stray files (`git status`).

## Self-review notes (spec-coverage map)

- Lifecycle table → Task 1. Evidence model + admissibility + integrity universe → Tasks 2, 5, 7.
- Criteria 1-8 → Tasks 6, 7. `n_concurrent_forward` recorded → Tasks 7, 8, 10.
- Two-phase + re-evaluation → Task 10. Token + TTL + mutual exclusion + consume recheck + CAS → Tasks 8, 9.
- Gate-branch scoping (`paper -> candidate` unbrick) → Task 9. Live-wall certificate (all five clauses) + challenge summary → Task 11.
- CLI + tightening directions + excluded-tick accounting → Task 12. CODEOWNERS/docs → Task 13.
- Known limitations need no code; deferred list stays deferred.
