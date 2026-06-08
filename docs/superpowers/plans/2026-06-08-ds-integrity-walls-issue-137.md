# DS-integrity walls (Issue 137) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three DS-integrity holes (funnel-level multiple-testing deflation, PIT-by-default, minimum-sample floor) plus the gate-bypass hole underneath them, enforced in CODEOWNERS-protected code so the autonomous agent cannot promote past the gate.

**Architecture:** `research/gates.py` stays pure (math + policy constants + the boolean PIT check + the floor spec). A NEW protected `registry/promotion.py` orchestrates in two phases — `promotion_preflight` (relaxation guard + stage-legality + breadth, BEFORE the holdout is peeked) and `run_gate` (evaluate + record + transition, AFTER `walk_forward`). A NEW `gate_evaluations` table is both the durable audit record and the single-use, agent-only token the shortlist transition consumes atomically — making `transition_strategy` the single enforcement chokepoint for `BACKTESTED→SHORTLISTED` (agent actor), mirroring the live gate.

**Tech Stack:** Python 3.12, uv, Typer CLI, sqlite3, pandas, pytest. Spec: `docs/superpowers/specs/2026-06-08-ds-integrity-walls-issue-137-design.md`.

**Quality gate:** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## Slicing (plan-review correction)

Walls A–D + the CLI are mutually dependent: `evaluate_gate` gains a **required** `pit_ok`, the agent
shortlist becomes **gated**, and the **minting path** must exist — so they cannot be committed as
independent green sub-steps. Three slices, each leaving the FULL gate green:

- **Slice 1 (Task 1):** inert data layer — `gate_evaluations` table + store methods + protocol.
- **Slice 2 (Tasks 2A–2D):** the gate — gates.py + promotion.py + transitions.py + CLI + **all test
  reconciliation**. The suite is intentionally RED between 2A and 2D; run the full gate + the single
  commit only at the end of 2D.
- **Slice 3 (Task 3):** housekeeping — root `CODEOWNERS` + CLAUDE.md.

Note: `TransitionError` is a subclass of `ValueError` (`algua/contracts/lifecycle.py:40`), so the
CLI's existing `@json_errors(ValueError, ...)` already wraps it — no decorator change needed.

---

## Task 1 (Slice 1): inert data layer

**Files:**
- Modify: `algua/registry/db.py` (table DDL), `algua/registry/store.py` (methods + `apply_transition` consume param), `algua/registry/repository.py` (Protocol)
- Test: `tests/test_registry_db.py`, `tests/test_registry_store.py`

### 1.1 — `gate_evaluations` table

- [ ] **Step 1: failing test** — add to `tests/test_registry_db.py`:

```python
def test_gate_evaluations_table_exists(tmp_path):
    from algua.registry.db import connect, migrate
    conn = connect(tmp_path / "g.db")
    migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(gate_evaluations)").fetchall()}
    assert {
        "id", "strategy_id", "passed", "n_funnel", "own_lifetime_combos",
        "windowed_total_combos", "funnel_window_days", "breadth_provenance", "pit_ok",
        "pit_override", "holdout_n_bars", "min_holdout_observations", "code_hash", "config_hash",
        "dependency_hash", "data_source", "snapshot_id", "period_start", "period_end",
        "holdout_frac", "actor", "decision_json", "consumed", "created_at",
    } <= cols
    conn.close()
```

- [ ] **Step 2:** `uv run pytest tests/test_registry_db.py::test_gate_evaluations_table_exists -v` → FAIL.

- [ ] **Step 3: implement** — in `algua/registry/db.py`, append after the `holdout_evaluations` table + index (inside the same migrate SQL script):

```sql
-- gate_evaluations records every promotion-gate evaluation (pass AND fail) for the audit trail,
-- AND is the single-use, AGENT-ONLY token the BACKTESTED->SHORTLISTED transition consumes (the
-- shortlist gate, mirroring the live gate: trust the gate record, not the stage flag). A passing
-- AGENT row is minted by `research promote` (via the protected registry.promotion orchestrator)
-- stamped with the artifact identity recomputed by approvals.compute_artifact_hashes; the
-- transition consumes THAT row's id, in the same transaction as the stage change. A human/override
-- promote writes an actor='human' row that is NEVER an agent-consumable token (audit only). FK into
-- strategies(id) — relational state, not an audit snapshot.
CREATE TABLE IF NOT EXISTS gate_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    passed INTEGER NOT NULL,
    n_funnel INTEGER NOT NULL,
    own_lifetime_combos INTEGER NOT NULL,
    windowed_total_combos INTEGER NOT NULL,
    funnel_window_days INTEGER NOT NULL,
    breadth_provenance TEXT NOT NULL,
    pit_ok INTEGER NOT NULL,
    pit_override INTEGER NOT NULL DEFAULT 0,
    holdout_n_bars INTEGER NOT NULL,
    min_holdout_observations INTEGER NOT NULL,
    code_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    dependency_hash TEXT,
    data_source TEXT NOT NULL,
    snapshot_id TEXT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    holdout_frac REAL NOT NULL,
    actor TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    consumed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_gate_evaluations_strategy ON gate_evaluations(strategy_id);
```

> Follow the surrounding tables' migration convention (`CREATE TABLE IF NOT EXISTS`; if `db.py`
> bumps `PRAGMA user_version`, do the same — match what neighboring tables do).

- [ ] **Step 4:** rerun → PASS.

### 1.2 — store methods + `apply_transition` consume param

- [ ] **Step 1: failing tests** — add to `tests/test_registry_store.py` (reuse/define a `repo` fixture: `connect(tmp_path/"s.db")` → `migrate` → `SqliteStrategyRepository(conn)`):

```python
from algua.contracts.lifecycle import Actor, Stage, TransitionError


def _record_pass(repo, sid, *, actor="agent", code="c0", config="cfg0", dep="dep0"):
    return repo.record_gate_evaluation(
        sid, passed=True, n_funnel=9, own_lifetime_combos=9, windowed_total_combos=9,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=80, min_holdout_observations=63, code_hash=code, config_hash=config,
        dependency_hash=dep, data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2023-12-31", holdout_frac=0.2, actor=actor,
        decision_json="{}")


def test_windowed_search_combos_sums_recent(repo):
    repo.record_search_trial("alpha", 4, "{}")
    repo.record_search_trial("beta", 5, "{}")
    assert repo.windowed_search_combos(window_days=90) == 9


def test_find_consumable_matches_agent_passing_identity(repo):
    rec = repo.add("alpha")
    gid = _record_pass(repo, rec.id)
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") == gid


def test_find_consumable_ignores_human_failing_and_mismatch(repo):
    rec = repo.add("alpha")
    _record_pass(repo, rec.id, actor="human")            # human row is not a token
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") is None
    repo.record_gate_evaluation(  # failing agent row
        rec.id, passed=False, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=80, min_holdout_observations=63, code_hash="c0", config_hash="cfg0",
        dependency_hash="dep0", data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2023-12-31", holdout_frac=0.2, actor="agent",
        decision_json="{}")
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") is None
    gid = _record_pass(repo, rec.id)                     # passing agent row, but...
    assert repo.find_consumable_gate_evaluation(rec.id, "BAD", "cfg0", "dep0") is None  # identity
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", None) is None     # NULL dep
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") == gid


def test_apply_transition_consumes_token_atomically(repo):
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    rec = repo.get("alpha")
    gid = _record_pass(repo, rec.id)
    out = repo.apply_transition(rec, Stage.SHORTLISTED, Actor.AGENT, "go", consume_gate_id=gid)
    assert out.stage == Stage.SHORTLISTED
    # token consumed (single-use)
    assert repo.find_consumable_gate_evaluation(rec.id, "c0", "cfg0", "dep0") is None


def test_apply_transition_bad_token_rolls_back(repo):
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    rec = repo.get("alpha")
    with pytest.raises(TransitionError):
        repo.apply_transition(rec, Stage.SHORTLISTED, Actor.AGENT, "go", consume_gate_id=999999)
    assert repo.get("alpha").stage == Stage.BACKTESTED  # stage unchanged (rolled back)
```

- [ ] **Step 2:** `uv run pytest tests/test_registry_store.py -k "windowed or find_consumable or apply_transition_consumes or apply_transition_bad" -v` → FAIL.

- [ ] **Step 3: implement in `algua/registry/store.py`**

3a. Imports: change `from datetime import UTC, datetime` to `from datetime import UTC, datetime, timedelta`; and add `TransitionError` to the lifecycle import: `from algua.contracts.lifecycle import Actor, Stage, TransitionError`.

3b. Extend `apply_transition` with the consume param (add to signature and body):

```python
    def apply_transition(
        self,
        rec: StrategyRecord,
        to: Stage,
        actor: Actor,
        reason: str | None = None,
        code_hash: str | None = None,
        config_hash: str | None = None,
        dependency_hash: str | None = None,
        consume_gate_id: int | None = None,
    ) -> StrategyRecord:
        from_stage = rec.stage
        now = _now()
        with self._conn:  # consume + UPDATE + INSERT commit together or not at all
            if consume_gate_id is not None:
                # Single-use, atomic with the stage change: flipping the token, the stage UPDATE,
                # and the transition INSERT all live in this one transaction. If the token row was
                # already consumed or is missing, raise so the whole transition rolls back — the
                # stage can never advance on a vanished token, nor a token be spent without the
                # stage advancing.
                cur = self._conn.execute(
                    "UPDATE gate_evaluations SET consumed=1"
                    " WHERE id=? AND strategy_id=? AND passed=1 AND actor='agent' AND consumed=0",
                    (consume_gate_id, rec.id))
                if cur.rowcount != 1:
                    raise TransitionError(
                        f"gate evaluation {consume_gate_id} is not a consumable agent token for "
                        f"this strategy (already consumed, missing, or mismatched)")
            self._conn.execute(
                "UPDATE strategies SET stage = ?, updated_at = ? WHERE id = ?",
                (to.value, now, rec.id),
            )
            self._conn.execute(
                "INSERT INTO stage_transitions"
                "(strategy_id, from_stage, to_stage, actor, reason, code_hash, config_hash,"
                " dependency_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (rec.id, from_stage.value, to.value, actor.value, reason,
                 code_hash, config_hash, dependency_hash, now),
            )
        return self.get(rec.name)
```

3c. Append the three new methods:

```python
    def windowed_search_combos(self, window_days: int) -> int:
        """Sum of ``n_combos`` across ALL strategies' search_trials recorded within the trailing
        ``window_days`` (funnel-wide search effort for Wall A). ISO-8601 UTC timestamps compare
        lexicographically in chronological order, so a string `>=` on created_at is correct."""
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(n_combos), 0) AS total FROM search_trials WHERE created_at >= ?",
            (cutoff,),
        ).fetchone()
        return int(row["total"])

    def record_gate_evaluation(
        self,
        strategy_id: int,
        *,
        passed: bool,
        n_funnel: int,
        own_lifetime_combos: int,
        windowed_total_combos: int,
        funnel_window_days: int,
        breadth_provenance: str,
        pit_ok: bool,
        pit_override: bool,
        holdout_n_bars: int,
        min_holdout_observations: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        actor: str,
        decision_json: str,
    ) -> int:
        """Persist one gate evaluation (pass or fail) and return its row id. A passing AGENT row is
        the single-use token the shortlist transition consumes."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO gate_evaluations"
                "(strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos,"
                " funnel_window_days, breadth_provenance, pit_ok, pit_override, holdout_n_bars,"
                " min_holdout_observations, code_hash, config_hash, dependency_hash, data_source,"
                " snapshot_id, period_start, period_end, holdout_frac, actor, decision_json,"
                " consumed, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
                (strategy_id, int(passed), n_funnel, own_lifetime_combos, windowed_total_combos,
                 funnel_window_days, breadth_provenance, int(pit_ok), int(pit_override),
                 holdout_n_bars, min_holdout_observations, code_hash, config_hash, dependency_hash,
                 data_source, snapshot_id, period_start, period_end, holdout_frac, actor,
                 decision_json, _now()),
            )
        rowid = cur.lastrowid
        assert rowid is not None
        return rowid

    def find_consumable_gate_evaluation(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
    ) -> int | None:
        """Return the id of the most-recent AGENT passing unconsumed gate row whose identity matches
        the recomputed current (code, config, dependency), or None. The ``actor='agent'`` filter
        means a human/override promote's audit row is never an agent-consumable token. A NULL
        ``dependency_hash`` matches nothing — fail-closed, mirroring has_valid_approval."""
        if dependency_hash is None:
            return None
        row = self._conn.execute(
            "SELECT id FROM gate_evaluations WHERE strategy_id=? AND passed=1 AND consumed=0"
            " AND actor='agent' AND code_hash=? AND config_hash=? AND dependency_hash=?"
            " ORDER BY id DESC LIMIT 1",
            (strategy_id, code_hash, config_hash, dependency_hash),
        ).fetchone()
        return int(row["id"]) if row is not None else None
```

3d. In `algua/registry/repository.py`: add the new methods to the `StrategyRepository` Protocol (signatures + one-line docstrings, body `...`), and add `consume_gate_id: int | None = None` to the `apply_transition` Protocol signature.

- [ ] **Step 4:** rerun the 1.2 tests → PASS.

- [ ] **Step 5: full gate + commit (Slice 1)**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/registry/db.py algua/registry/store.py algua/registry/repository.py \
        tests/test_registry_db.py tests/test_registry_store.py
git commit -m "feat(registry): gate_evaluations table + windowed breadth + atomic agent-token consume"
```

---

## Task 2 (Slice 2): the gate

> The suite is intentionally RED between 2A and 2D. Run the full gate + commit ONCE, at the end of 2D.

### Task 2A — `gates.py` pure changes

**Files:** Modify `algua/research/gates.py`; Test `tests/test_research_gates.py`.

- [ ] **Step 1: failing tests** — add to `tests/test_research_gates.py`:

```python
from algua.research.gates import (  # extend the existing import
    FUNNEL_WINDOW_DAYS,
    MIN_HOLDOUT_OBSERVATIONS,
    effective_funnel_breadth,
)


def test_constants_defaults():
    assert FUNNEL_WINDOW_DAYS == 90
    assert MIN_HOLDOUT_OBSERVATIONS == 63


def test_effective_funnel_breadth_is_max():
    assert effective_funnel_breadth(own_lifetime=10, windowed_total=3) == 10
    assert effective_funnel_breadth(own_lifetime=3, windowed_total=10) == 10
    assert effective_funnel_breadth(own_lifetime=0, windowed_total=0) == 0


_LAX = dict(min_holdout_sharpe=-100, min_holdout_return=-100, min_pct_positive_windows=0,
            min_window_sharpe=-100)


def test_min_holdout_observations_fails_closed_below_floor():
    d = evaluate_gate(_wf(n_bars=10), GateCriteria(**_LAX), n_combos=1, pit_ok=True)
    floor = next(c for c in d.checks if c["name"] == "min_holdout_observations")
    assert floor["passed"] is False and d.passed is False


def test_min_holdout_observations_passes_at_floor():
    d = evaluate_gate(_wf(n_bars=63), GateCriteria(**_LAX), n_combos=1, pit_ok=True)
    floor = next(c for c in d.checks if c["name"] == "min_holdout_observations")
    assert floor["passed"] is True


def test_pit_required_fails_closed():
    d = evaluate_gate(_wf(), GateCriteria(**_LAX), n_combos=1, pit_ok=False)
    pit = next(c for c in d.checks if c["name"] == "pit_required")
    assert pit["passed"] is False and pit["override"] is None and d.passed is False


def test_pit_override_passes_and_flags():
    d = evaluate_gate(_wf(), GateCriteria(**_LAX), n_combos=1, pit_ok=False, allow_non_pit=True)
    pit = next(c for c in d.checks if c["name"] == "pit_required")
    assert pit["passed"] is True and pit["override"] == "non_pit" and d.pit_override is True


def test_pit_ok_passes_clean():
    d = evaluate_gate(_wf(), GateCriteria(**_LAX), n_combos=1, pit_ok=True)
    pit = next(c for c in d.checks if c["name"] == "pit_required")
    assert pit["passed"] is True and d.pit_ok is True and d.pit_override is False
```

- [ ] **Step 2:** run those → FAIL (ImportError / unexpected kw `pit_ok`).

- [ ] **Step 3: implement** in `algua/research/gates.py`:

3a. Constants (after imports):

```python
# Funnel-level multiple-testing window (Wall A). Protected constant, not an agent-tunable knob
# (relaxing it would weaken the gate). Rolling window keeps the bar bounded; the wait-out-the-window
# trade-off is accepted and auditable via search_trials.created_at.
FUNNEL_WINDOW_DAYS = 90

# Minimum holdout sample (Wall C). A holdout with fewer observations is underpowered and fails
# closed — complements the 1/sqrt(T) haircut, which is ZERO at N=1. ~one trading quarter. Protected.
MIN_HOLDOUT_OBSERVATIONS = 63
```

3b. Pure combine (after `sharpe_haircut`):

```python
def effective_funnel_breadth(own_lifetime: int, windowed_total: int) -> int:
    """Effective funnel breadth fed to the haircut (Wall A): ``max`` of this strategy's LIFETIME
    recorded breadth and the funnel-wide breadth recorded in the rolling window (``windowed_total``
    INCLUDES this strategy's own windowed sweeps, so no double-count, no name-exclusion subtlety).
    An *effective funnel-breadth policy*, NOT a literal independent-trial count. A lone hypothesis
    with no siblings has ``windowed_total <= own_lifetime`` ⇒ returns ``own_lifetime`` ⇒ identical to
    the prior per-strategy behavior (no regression)."""
    return max(int(own_lifetime), int(windowed_total))
```

3c. `GateCriteria`: add field

```python
    min_window_sharpe: float = 0.0        # the worst window's Sharpe must be >= this
    min_holdout_observations: int = MIN_HOLDOUT_OBSERVATIONS  # Wall C: power floor, fails closed
```

3d. `GATE_SPECS`: append

```python
    GateSpec("min_holdout_observations", "holdout", "n_bars", "min_holdout_observations", ">="),
```

3e. `GateDecision`: add fields + emit in `to_dict`:

```python
    own_lifetime_combos: int | None = None
    windowed_total_combos: int | None = None
    funnel_window_days: int | None = None
    pit_ok: bool | None = None
    pit_override: bool = False
```
```python
            "own_lifetime_combos": self.own_lifetime_combos,
            "windowed_total_combos": self.windowed_total_combos,
            "funnel_window_days": self.funnel_window_days,
            "pit_ok": self.pit_ok,
            "pit_override": self.pit_override,
```

3f. `evaluate_gate`: new signature + append the boolean `pit_required` check + pass the new fields:

```python
def evaluate_gate(
    wf: WalkForwardResult,
    criteria: GateCriteria,
    *,
    n_combos: int | None = None,
    breadth_provenance: str | None = None,
    pit_ok: bool,
    allow_non_pit: bool = False,
    own_lifetime_combos: int | None = None,
    windowed_total_combos: int | None = None,
    funnel_window_days: int | None = None,
) -> GateDecision:
```

After the `for spec in GATE_SPECS:` loop, before building the decision:

```python
    # PIT precondition (Wall B): boolean, not a metric comparison. pit_ok is computed by the
    # protected promotion orchestrator (presence + coverage). Non-PIT fails closed unless a human
    # passed allow_non_pit (recorded as an audited override).
    pit_passed = bool(pit_ok or allow_non_pit)
    pit_override = bool((not pit_ok) and allow_non_pit)
    checks.append({"name": "pit_required", "passed": pit_passed,
                   "pit_ok": bool(pit_ok), "override": "non_pit" if pit_override else None})
```

And the returned `GateDecision(...)` gains `own_lifetime_combos=own_lifetime_combos,
windowed_total_combos=windowed_total_combos, funnel_window_days=funnel_window_days,
pit_ok=bool(pit_ok), pit_override=pit_override`.

- [ ] **Step 4:** run the 2A tests → PASS.

- [ ] **Step 5: reconcile `tests/test_research_gates.py`** (do NOT commit yet). `pit_ok` is now a
  required kwarg and two new checks (`min_holdout_observations`, `pit_required`) exist:
  - Add `pit_ok=True` to EVERY existing `evaluate_gate(...)` call (inventory at the time of writing:
    lines 28, 76, 87, 94, 102, 103, 110, 111, 119, 126, 133, 139, 145, 153, 163, 174, 179, 187 —
    but re-grep `evaluate_gate(` to catch all; the degenerate-holdout tests at 76/87 keep
    `pit_ok=True`, they assert on the holdout-Sharpe `inf` path which is unaffected).
  - `test_all_thresholds_met_passes` (≈line 28) asserts the EXACT check-name set
    `{holdout_sharpe, holdout_return, pct_positive_windows, min_window_sharpe}` — add the two new
    names: `{..., "min_holdout_observations", "pit_required"}`.
  - `test_gate_checks_are_table_driven` (≈line 182): `pit_required` is appended OUTSIDE
    `GATE_SPECS`, so change its assertion to:
    ```python
    names_from_eval = {c["name"] for c in evaluate_gate(_wf(), GateCriteria(), pit_ok=True).checks}
    assert names_from_eval == names_from_table | {"pit_required"}
    ```
  - After editing, run `uv run pytest tests/test_research_gates.py -v` to convergence (all green).

### Task 2B — protected `registry/promotion.py`

**Files:** Create `algua/registry/promotion.py`; Test `tests/test_promotion.py`.

- [ ] **Step 1: failing tests** — add `tests/test_promotion.py`:

```python
from datetime import date

import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.promotion import (
    guard_agent_relaxations,
    promotion_preflight,
    resolve_pit_ok,
)
from algua.registry.store import SqliteStrategyRepository


def _repo(tmp_path):
    conn = connect(tmp_path / "p.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_guard_agent_relaxations_refuses_agent():
    for kw in (dict(declared_combos=9, allow_holdout_reuse=False, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=True, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=False, allow_non_pit=True)):
        with pytest.raises(ValueError, match="human"):
            guard_agent_relaxations(Actor.AGENT, **kw)


def test_guard_allows_clean_agent_and_any_human():
    guard_agent_relaxations(Actor.AGENT, declared_combos=None, allow_holdout_reuse=False,
                            allow_non_pit=False)
    guard_agent_relaxations(Actor.HUMAN, declared_combos=9, allow_holdout_reuse=True,
                            allow_non_pit=True)


def test_resolve_pit_ok_requires_coverage():
    cover = [{"snapshot_id": "u1", "effective_date": "2021-06-01"}]
    late = [{"snapshot_id": "u1", "effective_date": "2022-06-01"}]
    assert resolve_pit_ok("sp", cover, date(2022, 1, 1)) is True
    assert resolve_pit_ok("sp", late, date(2022, 1, 1)) is False
    assert resolve_pit_ok(None, None, date(2022, 1, 1)) is False


@pytest.mark.parametrize("stages", [
    (),                                              # idea
    (Stage.BACKTESTED, Stage.SHORTLISTED),           # shortlisted
    (Stage.BACKTESTED, Stage.SHORTLISTED, Stage.PAPER),  # paper (PAPER->SHORTLISTED is legal!)
])
def test_preflight_refuses_non_backtested_source(tmp_path, stages):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.record_search_trial("alpha", 4, "{}")  # measured breadth present (so stage is the refusal)
    for s in stages:
        rec = repo.apply_transition(rec, s, Actor.HUMAN, "setup")
    with pytest.raises(TransitionError, match="backtested"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False)
    # Refused in preflight => no gate row, no holdout burn.
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


def test_preflight_refuses_agent_without_measured_breadth(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    with pytest.raises(ValueError, match="search breadth"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False)


def test_preflight_resolves_measured_funnel_breadth(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("alpha", 4, "{}")
    repo.record_search_trial("beta", 10, "{}")  # sibling effort raises the funnel bar
    ctx = promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                              allow_holdout_reuse=False, allow_non_pit=False)
    assert ctx.own == 4 and ctx.windowed_total == 14 and ctx.n_funnel == 14
    assert ctx.provenance == "measured"
```

- [ ] **Step 2:** `uv run pytest tests/test_promotion.py -v` → FAIL (module missing).

- [ ] **Step 3: implement `algua/registry/promotion.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from algua.backtest.walkforward import WalkForwardResult
from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.repository import StrategyRepository
from algua.registry.transitions import transition_strategy
from algua.research.gates import (
    FUNNEL_WINDOW_DAYS,
    GateCriteria,
    GateDecision,
    effective_funnel_breadth,
    evaluate_gate,
)


def guard_agent_relaxations(
    actor: Actor,
    *,
    declared_combos: int | None,
    allow_holdout_reuse: bool,
    allow_non_pit: bool,
) -> None:
    """Every gate RELAXATION (declared breadth, holdout reuse, non-PIT) requires a human actor. An
    agent passing any is refused — the agent only ever sees the strict gate. Call EARLY (pre-peek)."""
    if actor is Actor.HUMAN:
        return
    if declared_combos is not None or allow_holdout_reuse or allow_non_pit:
        raise ValueError(
            "gate relaxation requires --actor human: --n-combos (declared breadth), "
            "--allow-holdout-reuse and --allow-non-pit are human-only. For an agent, breadth must "
            "be measured (run `backtest sweep`), the holdout fresh, and the universe PIT."
        )


def resolve_pit_ok(
    universe_name: str | None,
    universe_snapshots: list[dict[str, str]] | None,
    period_start: date,
) -> bool:
    """Wall B: PIT-valid iff a universe was used AND its earliest membership snapshot is effective on
    or before the backtest start (coverage, not mere presence)."""
    if universe_name is None or not universe_snapshots:
        return False
    earliest = min(date.fromisoformat(s["effective_date"]) for s in universe_snapshots)
    return earliest <= period_start


@dataclass
class BreadthContext:
    n_funnel: int
    own: int
    windowed_total: int
    provenance: str


def promotion_preflight(
    repo: StrategyRepository,
    name: str,
    *,
    actor: Actor,
    declared_combos: int | None,
    allow_holdout_reuse: bool,
    allow_non_pit: bool,
) -> BreadthContext:
    """Pre-peek phase — runs BEFORE walk_forward, so every hard refusal happens before the holdout
    is touched and before any gate row is minted: (1) relaxations-need-human; (2) stage legality
    (BACKTESTED -> SHORTLISTED must be legal — never mint a passing token for an illegal source
    stage); (3) breadth resolution (refuse "no measured breadth" here)."""
    guard_agent_relaxations(actor, declared_combos=declared_combos,
                            allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit)
    rec = repo.get(name)
    # Source stage MUST be exactly BACKTESTED. validate_transition alone is too permissive here:
    # PAPER -> SHORTLISTED is a legal back-step, so promoting from `paper` would otherwise pass
    # preflight, burn the holdout, and mint a token. Require backtested explicitly, then validate.
    if rec.stage is not Stage.BACKTESTED:
        raise TransitionError(
            f"research promote requires stage backtested, got {rec.stage.value}")
    validate_transition(rec.stage, Stage.SHORTLISTED)  # TransitionError (a ValueError) if illegal
    measured = repo.total_search_combos(name)
    windowed_total = repo.windowed_search_combos(FUNNEL_WINDOW_DAYS)
    if measured > 0:
        own, provenance = measured, "measured"
    elif declared_combos is not None:  # human-only path (already guarded above)
        own, provenance = declared_combos, "declared"
    else:
        raise ValueError(
            f"no recorded search breadth for {name!r}; run `algua backtest sweep {name} ...` "
            f"(records breadth). Declaring via --n-combos requires --actor human."
        )
    return BreadthContext(effective_funnel_breadth(own, windowed_total), own, windowed_total,
                          provenance)


@dataclass
class PromotionOutcome:
    decision: GateDecision
    promoted: bool


def run_gate(
    repo: StrategyRepository,
    wf: WalkForwardResult,
    *,
    name: str,
    actor: Actor,
    criteria: GateCriteria,
    breadth: BreadthContext,
    universe_name: str | None,
    universe_snapshots: list[dict[str, str]] | None,
    period_start: date,
    period_end: date,
    holdout_frac: float,
    data_source: str,
    snapshot_id: str | None,
    allow_non_pit: bool,
    reason_suffix: str,
) -> PromotionOutcome:
    """Post-walk phase: resolve PIT, evaluate, record the gate_evaluations row (pass AND fail), and
    on pass transition BACKTESTED->SHORTLISTED (which consumes the just-minted agent token). Identity
    is recomputed via compute_artifact_hashes(name) — the SAME function the shortlist gate matches
    against (NOT wf.code_hash, which is git-HEAD-based and would never match)."""
    pit_ok = resolve_pit_ok(universe_name, universe_snapshots, period_start)
    holdout_n_bars = int(wf.holdout_metrics["n_bars"])
    decision = evaluate_gate(
        wf, criteria, n_combos=breadth.n_funnel, breadth_provenance=breadth.provenance,
        pit_ok=pit_ok, allow_non_pit=allow_non_pit, own_lifetime_combos=breadth.own,
        windowed_total_combos=breadth.windowed_total, funnel_window_days=FUNNEL_WINDOW_DAYS,
    )
    identity = compute_artifact_hashes(name)
    rec = repo.get(name)
    repo.record_gate_evaluation(
        rec.id, passed=decision.passed, n_funnel=breadth.n_funnel,
        own_lifetime_combos=breadth.own, windowed_total_combos=breadth.windowed_total,
        funnel_window_days=FUNNEL_WINDOW_DAYS, breadth_provenance=breadth.provenance,
        pit_ok=bool(decision.pit_ok), pit_override=bool(decision.pit_override),
        holdout_n_bars=holdout_n_bars, min_holdout_observations=criteria.min_holdout_observations,
        code_hash=identity.code_hash, config_hash=identity.config_hash,
        dependency_hash=identity.dependency_hash, data_source=data_source, snapshot_id=snapshot_id,
        period_start=period_start.isoformat(), period_end=period_end.isoformat(),
        holdout_frac=holdout_frac, actor=actor.value,
        decision_json=json.dumps(decision.to_dict(), sort_keys=True),
    )
    promoted = False
    if decision.passed:
        transition_strategy(repo, name, Stage.SHORTLISTED, actor,
                            _gate_reason(decision) + reason_suffix)
        promoted = True
    return PromotionOutcome(decision=decision, promoted=promoted)


def _gate_reason(decision: GateDecision) -> str:
    """Human-readable gate summary. Metric checks render value/op/threshold; boolean checks (e.g.
    pit_required) render name=pass|fail."""
    parts: list[str] = []
    for c in decision.checks:
        if "value" in c and c.get("value") is not None and c.get("threshold") is not None:
            parts.append(f"{c['name']}={c['value']:.4g}{c['op']}{c['threshold']:.4g}")
        else:
            parts.append(f"{c['name']}={'pass' if c['passed'] else 'fail'}")
    breadth = (
        f"; funnel_breadth={decision.n_combos}({decision.breadth_provenance}"
        f"; own={decision.own_lifetime_combos}, windowed={decision.windowed_total_combos}"
        f", window={decision.funnel_window_days}d)"
        f"; min_holdout_sharpe={decision.base_min_holdout_sharpe:.4g}"
        f"->{decision.effective_min_holdout_sharpe:.4g}"
        if decision.n_combos is not None else ""
    )
    return "gate pass: " + ", ".join(parts) + breadth
```

- [ ] **Step 4:** `uv run pytest tests/test_promotion.py -v` → PASS.

### Task 2C — shortlist gate in `transitions.py`

**Files:** Modify `algua/registry/transitions.py`; Test `tests/test_shortlist_gate.py`.

- [ ] **Step 1: failing tests** — add `tests/test_shortlist_gate.py`:

```python
import pytest

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy

_IDENT = type("I", (), {"code_hash": "c0", "config_hash": "cfg0", "dependency_hash": "dep0"})


def _repo(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def _backtested(repo, name="alpha"):
    rec = repo.add(name)
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    return repo.get(name)


def _token(repo, sid, *, actor="agent"):
    return repo.record_gate_evaluation(
        sid, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=80, min_holdout_observations=63, code_hash="c0", config_hash="cfg0",
        dependency_hash="dep0", data_source="SyntheticProvider", snapshot_id=None,
        period_start="2022-01-01", period_end="2023-12-31", holdout_frac=0.2, actor=actor,
        decision_json="{}")


def test_agent_shortlist_refused_without_token(tmp_path, monkeypatch):
    repo = _repo(tmp_path); _backtested(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _IDENT())
    with pytest.raises(TransitionError, match="gate"):
        transition_strategy(repo, "alpha", Stage.SHORTLISTED, Actor.AGENT, "try")


def test_agent_shortlist_consumes_token_single_use(tmp_path, monkeypatch):
    repo = _repo(tmp_path); rec = _backtested(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _IDENT())
    _token(repo, rec.id)
    assert transition_strategy(repo, "alpha", Stage.SHORTLISTED, Actor.AGENT, "ok").stage \
        == Stage.SHORTLISTED
    repo.apply_transition(repo.get("alpha"), Stage.BACKTESTED, Actor.AGENT, "back")
    with pytest.raises(TransitionError, match="gate"):
        transition_strategy(repo, "alpha", Stage.SHORTLISTED, Actor.AGENT, "again")


def test_human_token_not_consumable_by_agent(tmp_path, monkeypatch):
    repo = _repo(tmp_path); rec = _backtested(repo)
    monkeypatch.setattr("algua.registry.transitions._compute_hashes", lambda name: _IDENT())
    _token(repo, rec.id, actor="human")  # human audit row is not an agent token
    with pytest.raises(TransitionError, match="gate"):
        transition_strategy(repo, "alpha", Stage.SHORTLISTED, Actor.AGENT, "try")


def test_human_shortlist_exempt(tmp_path):
    repo = _repo(tmp_path); _backtested(repo)
    assert transition_strategy(repo, "alpha", Stage.SHORTLISTED, Actor.HUMAN, "manual").stage \
        == Stage.SHORTLISTED
```

- [ ] **Step 2:** `uv run pytest tests/test_shortlist_gate.py -v` → FAIL.

- [ ] **Step 3: implement in `algua/registry/transitions.py`**

In `transition_strategy`, after computing `target`/`transition_actor`/`rec` and `validate_transition`,
replace the `if target == Stage.LIVE:` block so it also handles shortlist and threads `consume_gate_id`:

```python
    code_hash: str | None = None
    config_hash: str | None = None
    dependency_hash: str | None = None
    consume_gate_id: int | None = None
    if target == Stage.LIVE:
        identity = _validate_live_gate(
            repo=repo, name=name, strategy_id=rec.id, actor=transition_actor,
            approval_verifier=approval_verifier,
        )
        code_hash, config_hash, dependency_hash = identity
    elif target == Stage.SHORTLISTED and transition_actor is not Actor.HUMAN:
        # Wall D: an agent reaches shortlisted ONLY by consuming a fresh, identity-matched,
        # single-use gate token (minted by `research promote`). Humans are exempt.
        consume_gate_id = _validate_shortlist_gate(repo=repo, name=name, strategy_id=rec.id)
    return repo.apply_transition(
        rec=rec, to=target, actor=transition_actor, reason=reason,
        code_hash=code_hash, config_hash=config_hash, dependency_hash=dependency_hash,
        consume_gate_id=consume_gate_id,
    )
```

Add the validator (mirrors `_validate_live_gate`; returns the id so the consume is atomic in
`apply_transition`):

```python
def _validate_shortlist_gate(*, repo: StrategyRepository, name: str, strategy_id: int) -> int:
    """Return the id of a fresh passing AGENT gate token whose recomputed identity matches the
    strategy's current code+config+dependency, or raise. Consumption itself happens inside
    apply_transition, in one transaction with the stage change."""
    identity = _compute_hashes(name)
    gate_id = repo.find_consumable_gate_evaluation(
        strategy_id, identity.code_hash, identity.config_hash, identity.dependency_hash)
    if gate_id is None:
        raise TransitionError(
            "transition to shortlisted requires a fresh passing gate evaluation for the current "
            "code+config+dependency; run `algua research promote` (no matching gate record found)"
        )
    return gate_id
```

- [ ] **Step 4:** `uv run pytest tests/test_shortlist_gate.py -v` → PASS.

### Task 2D — CLI rewire + `--allow-non-pit` + reconcile all existing tests

**Files:** Modify `algua/cli/research_cmd.py`; Tests `tests/test_cli_research.py` (+ reconcile others).

- [ ] **Step 1: failing tests** — add to `tests/test_cli_research.py`:

```python
def test_agent_promote_demo_refuses_relaxation():
    assert _backtest_to_backtested().exit_code == 0
    r = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31", "--n-combos", "9"])
    assert r.exit_code != 0
    assert "human" in r.stdout.lower()


def test_human_promote_demo_overrides_shortlists():
    assert _backtest_to_backtested().exit_code == 0
    r = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                            "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                            "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["promoted"] is True and p["n_funnel"] == 9 and p["pit_override"] is True
    assert _stage() == "shortlisted"


def test_agent_promote_blocked_without_pit():
    assert _backtest_to_backtested().exit_code == 0
    assert _sweep().exit_code == 0
    r = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                            "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["promoted"] is False
    assert next(c for c in p["checks"] if c["name"] == "pit_required")["passed"] is False
```

- [ ] **Step 2:** run those → FAIL.

- [ ] **Step 3: rewire `algua/cli/research_cmd.py`**

3a. Replace `from algua.research.gates import GateCriteria, GateDecision, evaluate_gate` with:
```python
from algua.research.gates import GateCriteria
from algua.registry.promotion import promotion_preflight, run_gate
```

3b. Add the option to `promote` (beside `--allow-holdout-reuse`):
```python
    allow_non_pit: bool = typer.Option(
        False, "--allow-non-pit",
        help="HUMAN-ONLY override: promote a non-PIT (survivorship-biased) backtest. Audited. "
             "Agents may not pass this.",
    ),
```

3c. Keep the existing CLI-level argument validations that run BEFORE the `with registry_conn()`
block (`if n_combos is not None and n_combos < 1: raise ValueError("--n-combos must be >= 1 ...")`
and the `--min-pct-positive` range check). They must stay ahead of the preflight so
`--n-combos 0` still yields "must be >= 1" (not the relaxation-needs-human error) — this preserves
`test_promote_rejects_bad_n_combos` / `test_promote_rejects_out_of_range_pct_positive`.

Then replace the body from the `with registry_conn()` block onward so the PREFLIGHT runs before
`walk_forward`, and `run_gate` replaces the old breadth/evaluate/transition steps:

```python
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        repo.get(name)  # StrategyNotFound -> JSON error before any work
        # PREFLIGHT (pre-peek): relaxations-need-human + stage legality + breadth. Refuses here,
        # before walk_forward touches the holdout.
        breadth = promotion_preflight(
            repo, name, actor=actor_enum, declared_combos=n_combos,
            allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit)
        # Holdout-reuse pre-check — BEFORE walk_forward (unchanged).
        overlap = repo.overlapping_holdout_evaluations(
            repo.get(name).id, data_source=data_source, snapshot_id=snapshot_id,
            period_start=period_start, period_end=period_end, holdout_frac=holdout_frac)
        if overlap and not allow_holdout_reuse:
            raise ValueError(
                f"holdout already consumed for {name!r}: an overlapping out-of-sample window was "
                f"already evaluated. Use fresh out-of-sample data, or --allow-holdout-reuse "
                f"(--actor human) to override and accept the statistical cost."
            )
        reused = bool(overlap and allow_holdout_reuse)
        wf = walk_forward(strategy, provider, start_dt, end_dt, windows=windows,
                          holdout_frac=holdout_frac, universe_by_date=universe_by_date,
                          universe_name=universe, universe_snapshots=universe_prov)
        repo.record_holdout_evaluation(
            repo.get(name).id, data_source=data_source, snapshot_id=snapshot_id,
            period_start=period_start, period_end=period_end, holdout_frac=holdout_frac,
            config_hash=wf.config_hash, reused=reused)
        outcome = run_gate(
            repo, wf, name=name, actor=actor_enum, criteria=criteria, breadth=breadth,
            universe_name=universe, universe_snapshots=universe_prov,
            period_start=start_dt.date(), period_end=end_dt.date(), holdout_frac=holdout_frac,
            data_source=data_source, snapshot_id=snapshot_id, allow_non_pit=allow_non_pit,
            reason_suffix=("; holdout_reuse=" + _HOLDOUT_REUSE_OVERRIDE) if reused else "")
        decision, promoted = outcome.decision, outcome.promoted

    payload: dict[str, Any] = {
        **decision.to_dict(),
        "n_funnel": decision.n_combos,
        "strategy": name,
        "promoted": promoted,
        "config_hash": wf.config_hash,
        "snapshot_id": wf.snapshot_id,
        "holdout": wf.holdout_metrics,
        "stability": wf.stability,
        "universe_name": wf.universe_name,
        "universe_snapshots": wf.universe_snapshots,
    }
    if reused:
        payload["holdout_reuse"] = _HOLDOUT_REUSE_OVERRIDE
    emit(ok(payload))
```

3d. Delete the now-unused local `_resolve_breadth` and `_gate_reason` from `research_cmd.py`. Keep
the `_HOLDOUT_REUSE_OVERRIDE` constant. Ensure `criteria` is still built from the `--min-*` flags
(its `min_holdout_observations` takes the protected default 63 — there is no CLI flag for it).

- [ ] **Step 4:** run the 2D new tests → PASS.

- [ ] **Step 5: reconcile ALL existing tests** that drive `research promote` or transition to
  `shortlisted`. Two behavior changes drive this: (i) a non-PIT (demo) run is not promotable, and
  (ii) `--n-combos`/`--allow-holdout-reuse` are human-only. Apply this **rule** and re-run the
  suite to convergence — do NOT rely on the inventory line numbers being current; re-grep
  (`grep -n '"research", "promote"' tests/test_cli_research.py`) and treat each invocation:

  **Rule per `research promote` invocation:**
  - **Demo (non-PIT) run** (`--demo`, no `--universe`): add `--actor human --allow-non-pit`. Keep
    any `--n-combos`/`--allow-holdout-reuse` (now valid under human).
  - **PIT run** (`--universe ...`) that uses `--n-combos`/`--allow-holdout-reuse`: add
    `--actor human` (PIT already satisfied, so NO `--allow-non-pit`).
  - **Payload assertions:** `payload["n_combos"]` → `payload["n_funnel"]`
    (`measured_breadth_wins` ⇒ `4`; `two_sweeps_accumulate` ⇒ `8`).
  - **Leave AS-IS** (they assert the new strict-agent refusals, which still hold):
    `test_promote_refuses_with_no_breadth`, `test_promote_rejects_bad_n_combos`,
    `test_promote_rejects_out_of_range_pct_positive`,
    `test_promote_with_universe_refuses_with_no_breadth_before_walkforward`.
  - **`test_promote_from_idea_is_json_error`:** still a JSON error, but now from the preflight
    stage check — assert `payload["ok"] is False` and (if it checks a message) `"backtested"`.

  **Inventory (test_cli_research.py, re-grep to confirm):** demo+`--n-combos`/`_PASS` clusters at
  ~33, 52, 65, 79, 104, 158, 170, 187, 204, 228, 244, 358, 406 → human (+`--allow-non-pit` for the
  demo ones); PIT clusters at ~307, 330, 384 → `--actor human`.

  **Other files:**
  - `tests/test_cli_sweep.py:68`, `tests/test_e2e_lifecycle.py:57` — demo promote with `--n-combos`:
    add `--allow-non-pit --actor human`. (Note `test_e2e_lifecycle` then continues `shortlisted →
    paper`; that agent step is unaffected — only the shortlist edge is gated.)
  - `tests/test_cli_paper.py:24` — `registry transition ... --to shortlisted`: add `--actor human`.
  - `tests/test_registry_approvals.py:28`, `tests/test_registry_store.py:66` — change the
    `Actor.AGENT` transition into `shortlisted` to `Actor.HUMAN` (scaffolding reaching a later
    stage; not testing the gate).

  Run `uv run pytest -q` repeatedly, fixing per the rule until green.

- [ ] **Step 6: full gate + commit (Slice 2)**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/research/gates.py algua/registry/promotion.py algua/registry/transitions.py \
        algua/cli/research_cmd.py tests/test_research_gates.py tests/test_promotion.py \
        tests/test_shortlist_gate.py tests/test_cli_research.py tests/test_cli_sweep.py \
        tests/test_e2e_lifecycle.py tests/test_cli_paper.py tests/test_registry_approvals.py \
        tests/test_registry_store.py
git commit -m "feat(gates): funnel deflation, PIT-by-default, min-sample floor + agent shortlist gate (#137)"
```

---

## Task 3 (Slice 3): housekeeping

**Files:** Modify root `CODEOWNERS`, `CLAUDE.md`.

- [ ] **Step 1:** append to the **root** `CODEOWNERS` (NOT `.github/CODEOWNERS` — the repo's owners
  file is at the root):

```
/algua/registry/promotion.py    @Lior-Nis   # promotion policy (breadth/PIT/floor + shortlist gate)
```

- [ ] **Step 2:** update `CLAUDE.md` `research promote` line: agents must use `--universe` (PIT
  required), breadth must be measured (`backtest sweep`), the holdout-Sharpe bar is deflated by
  funnel breadth, a min-holdout-observations floor (63) applies, and `BACKTESTED→SHORTLISTED` for an
  agent now requires a passing `research promote` (no raw `registry transition` shortcut).

- [ ] **Step 3: full gate** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
  Confirm `lint-imports` is "0 broken" (only `registry/promotion.py` imports `research.gates`/
  `backtest.walkforward` — both allowed directions; `research` still imports no `registry`).

- [ ] **Step 4: commit**

```bash
git add CODEOWNERS CLAUDE.md
git commit -m "chore(gates): protect promotion.py; document agent gate requirements (#137)"
```

---

## Self-review notes

- **Spec coverage:** Wall A → 2A (combine) + 1.2 (windowed query) + 2B (resolution). Wall B → 2A
  (check) + 2B (`resolve_pit_ok` coverage) + 2D (`--allow-non-pit`). Wall C → 2A. Wall D → 1.2
  (token store + atomic consume), 2C (transition gate), 2B (mint). Relaxations-need-human → 2B + 2D.
  Audit table → 1.1, 1.2, 2B. CODEOWNERS → Task 3.
- **Codex plan-review fixes folded:** C1 exact-id/agent-only/atomic token (1.2 `find_consumable` +
  `apply_transition(consume_gate_id)`, 2C threads the id); C2 stage-legality preflight (2B
  `promotion_preflight` + `test_preflight_refuses_illegal_source_stage`); C3 breadth refusal pre-walk
  (2B preflight, called before `walk_forward` in 2D); H atomic consume (1.2); H slicing (3 slices,
  Slice 2 single commit); H root CODEOWNERS (Task 3); M wider test reconciliation (2A Step 5, 2D
  Step 5); L ruff (`promotion.py` stdlib `json`, no unused imports).
- **Identity consistency:** gate row written with `compute_artifact_hashes(name)` (2B) and matched by
  `_validate_shortlist_gate`→`_compute_hashes`→`compute_artifact_hashes` (2C). Same function;
  `wf.code_hash` deliberately NOT used.
- **Deferred (spec §9, not this plan):** durable family/hypothesis id (#126/#122), sweep reservation
  ledger, full PIT provenance certification, active-observation floor, base-threshold relaxation flags.
```
