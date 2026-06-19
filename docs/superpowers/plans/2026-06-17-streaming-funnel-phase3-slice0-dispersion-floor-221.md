# Phase 3 Slice 0 — funnel-wide dispersion floor (#221) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Floor the DSR's own-sweep trial-Sharpe variance from below with a funnel-wide
cross-strategy trial-Sharpe variance, removing the documented low-dispersion leniency
(`trial_sr_var → 0 ⇒ SR* → 0 ⇒ DSR collapses to plain PSR`) — a purely tightening change with
NO schema bump and NO migration.

**Architecture:** A new unprotected read-accessor `funnel_trial_sharpe_var(window_days)` in
`store.py` pools each strategy's own `search_trials` `(count, mean, var)` triples FIRST
(anti-gaming: one vote per strategy regardless of combo count), then takes the MEAN across
strategies active in the rolling window. The pure `dsr_confidence` gate in `gates.py` (PROTECTED)
gains a `funnel_floor_var_per_period` parameter and applies `max(own, floor)` before the `SR*`
calculation. The protected orchestrator `promotion.py` wires the accessor result into the gate and
records audit fields. Floor unavailability (`< MIN_FUNNEL_FLOOR_STRATEGIES`) is fail-open →
Phase-1 behavior.

**Tech Stack:** Python 3.12, sqlite3, scipy.stats, pytest, dataclasses, NamedTuple.

## Global Constraints

- **Tighten-only.** `max(own, floor) ≥ own` ⇒ `SR*` can only rise ⇒ DSR confidence can only fall ⇒
  the gate can only move PASS→FAIL, never FAIL→PASS. Property-tested over a generated grid.
- **Fail-open floor, fail-closed gate.** The floor accessor returns `None` (var) when fewer than
  `MIN_FUNNEL_FLOOR_STRATEGIES` finite per-strategy variances exist; the gate then uses own-sweep
  var (Phase-1 behavior). A strategy with ANY NULL/NaN/inf `(count, mean, var)` row is EXCLUDED
  from the floor (its per-strategy pooled var is `None`), never silently skipped row-by-row.
- **Unit discipline.** The floor variance is ANNUALIZED at rest (matches `trial_sharpe_var_ann`);
  conversion to per-period (`/ ANN`) happens once inside `gates.py`, co-located with the existing
  `dsr_trial_var_ann / ANN` conversion.
- **Architecture boundary.** `gates.py` stays pure-math reading pre-computed scalars. All DB reads
  live in `store.py`. `algua/contracts` and `algua/features` remain pure.
- **No schema version bump.** Only `CREATE INDEX IF NOT EXISTS ix_search_trials_created_at` is
  added (an index addition needs no `SCHEMA_VERSION` bump — schema stays 26).
- **Protected walls** (`algua/research/gates.py`, `algua/registry/promotion.py`) are CODEOWNERS
  `@Lior-Nis`. Every change to them must preserve the four Phase-1 invariants above.
- **Per-slice quality gate (must pass before commit):**
  `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- Constant value (from spec GATE-1): `MIN_FUNNEL_FLOOR_STRATEGIES = 5`.

---

## File Structure

- `algua/registry/store.py` (modify) — extract a pure pooling helper from `pooled_trial_sharpe_var`;
  add `funnel_trial_sharpe_var(window_days) → FunnelFloor`. Add the `created_at` index to the schema
  via `db.py`.
- `algua/registry/repository.py` (modify) — add `FunnelFloor` NamedTuple + Protocol declaration of
  `funnel_trial_sharpe_var`.
- `algua/registry/db.py` (modify) — add `CREATE INDEX IF NOT EXISTS ix_search_trials_created_at`.
- `algua/research/gates.py` (PROTECTED, modify) — `MIN_FUNNEL_FLOOR_STRATEGIES` constant;
  `funnel_floor_var_per_period` param on `dsr_confidence`; `dsr_funnel_floor_var_ann` param on
  `evaluate_gate`; three audit fields on `GateDecision` + `to_dict`.
- `algua/registry/promotion.py` (PROTECTED, modify) — call the accessor when `dsr_binding`, pass the
  var into `evaluate_gate`, record the floor audit fields.
- `tests/registry/test_funnel_floor.py` (create) — accessor unit + anti-gaming tests.
- `tests/research/test_dsr_dispersion_floor.py` (create) — gate `max(own, floor)` algebra +
  tighten-only property test + integration.

---

## Task 1: Pure pooling helper + funnel-wide accessor (`store.py`, `db.py`, `repository.py`)

**Files:**
- Modify: `algua/registry/store.py` (extract helper near `pooled_trial_sharpe_var:425`; add
  `funnel_trial_sharpe_var` after `windowed_search_combos:555`)
- Modify: `algua/registry/repository.py` (add `FunnelFloor` NamedTuple near `FdrStreamState:28`;
  add Protocol method after `pooled_trial_sharpe_var:231`)
- Modify: `algua/registry/db.py` (add index after `ix_search_trials_strategy:135`)
- Test: `tests/registry/test_funnel_floor.py` (create)

**Interfaces:**
- Consumes: existing `search_trials` columns `strategy_name, created_at, trial_sharpe_count,
  trial_sharpe_mean, trial_sharpe_var_ann`; `self._conn` (sqlite3 row-factory connection);
  module-level `datetime, UTC, timedelta, math` (already imported in `store.py`).
- Produces:
  - `FunnelFloor(var_ann: float | None, n_strategies: int, n_total_rows: int)` — NamedTuple in
    `repository.py`. `var_ann` is the mean of per-strategy pooled variances (annualized), or `None`
    when `n_strategies < MIN_FUNNEL_FLOOR_STRATEGIES`. `n_strategies` counts strategies with a
    FINITE per-strategy pooled var; `n_total_rows` is the total `search_trials` rows pooled across
    those finite strategies (audit: stale-history visibility).
  - `_pool_trial_sharpe_var(triples: list[tuple[int, float, float]]) → float | None` — module-level
    pure helper in `store.py`; pooled sample variance (ddof=1) of `(n, mean, var)` triples;
    `None` for empty input, `0.0` for `Σn ≤ 1`.
  - `StrategyRepository.funnel_trial_sharpe_var(self, window_days: int) → FunnelFloor` — Protocol
    method.

- [ ] **Step 1: Write the failing accessor tests**

Create `tests/registry/test_funnel_floor.py`. Use the existing in-memory store fixture pattern
(check `tests/registry/` for how peers build a `SqliteStrategyRepository` / `StrategyStore`; mirror
the exact constructor and the `record_search_trial(name, n_combos, grid_json, trial_sharpe_count=,
trial_sharpe_mean=, trial_sharpe_var_ann=)` signature). The window selects STRATEGIES (any row in
window), then pools ALL of that strategy's rows.

```python
import math

from algua.registry.repository import FunnelFloor


def _add_trials(repo, name, triples, *, created_at=None):
    # triples: list of (count, mean, var_ann). Each becomes one search_trials row.
    for count, mean, var in triples:
        repo.record_search_trial(
            name, count, "{}",
            trial_sharpe_count=count, trial_sharpe_mean=mean, trial_sharpe_var_ann=var,
        )


def test_floor_none_below_min_strategies(make_repo):
    repo = make_repo()
    # 4 strategies < MIN_FUNNEL_FLOOR_STRATEGIES (5) -> None var, but counts still reported.
    for i in range(4):
        _add_trials(repo, f"s{i}", [(10, 1.0, 0.25)])
    floor = repo.funnel_trial_sharpe_var(90)
    assert isinstance(floor, FunnelFloor)
    assert floor.var_ann is None
    assert floor.n_strategies == 4
    assert floor.n_total_rows == 4


def test_floor_is_mean_of_per_strategy_pooled_var(make_repo):
    repo = make_repo()
    # 5 strategies, each a SINGLE row whose per-strategy pooled var == its var_ann
    # (Σn==total over one row group; pooled sample var of one (n,mean,var) group == var).
    vars_ann = [0.10, 0.20, 0.30, 0.40, 0.50]
    for i, v in enumerate(vars_ann):
        _add_trials(repo, f"s{i}", [(10, 1.0, v)])
    floor = repo.funnel_trial_sharpe_var(90)
    assert floor.n_strategies == 5
    assert floor.var_ann == _pytest_approx(sum(vars_ann) / 5)


def test_anti_gaming_one_vote_per_strategy(make_repo):
    repo = make_repo()
    # One family runs 100 near-duplicate combos (low dispersion); 4 others run 1 each.
    # The big family must contribute ONE vote (its per-strategy var), not 100-count domination.
    _add_trials(repo, "big", [(1, 1.0, 0.0)] * 100)   # 100 rows, per-strategy pooled var ~0
    for i in range(4):
        _add_trials(repo, f"small{i}", [(10, 1.0, 0.40)])
    floor = repo.funnel_trial_sharpe_var(90)
    assert floor.n_strategies == 5
    # mean of [~0, 0.40, 0.40, 0.40, 0.40] == 0.32, NOT pulled toward 0 by the 100 big rows.
    assert floor.var_ann == _pytest_approx((0.0 + 0.40 * 4) / 5, abs=1e-9)


def test_all_null_strategy_excluded(make_repo):
    repo = make_repo()
    for i in range(5):
        _add_trials(repo, f"good{i}", [(10, 1.0, 0.30)])
    # A strategy with a NULL stat row -> per-strategy pooled var None -> excluded entirely.
    repo.record_search_trial("bad", 10, "{}")  # no trial_sharpe_* -> NULLs
    floor = repo.funnel_trial_sharpe_var(90)
    assert floor.n_strategies == 5  # 'bad' excluded
    assert floor.var_ann == _pytest_approx(0.30)


def test_window_selects_strategy_then_pools_all_its_rows(make_repo, monkeypatch):
    # A strategy with one row in-window pools ALL its rows (including out-of-window rows).
    # A strategy with NO row in-window is not selected at all. See implementation note on
    # injecting created_at; if no per-row created_at control exists, assert via window_days
    # large-enough/zero behavior instead (zero-day window selects nothing -> None).
    repo = make_repo()
    for i in range(5):
        _add_trials(repo, f"s{i}", [(10, 1.0, 0.30)])
    floor_zero = repo.funnel_trial_sharpe_var(0)  # cutoff = now -> rows written 'now' may or may
    # not be >= cutoff; if flaky, prefer a negative-ish guarantee: a far-future-only window.
    # The robust assertion: a window of -1 days selects nothing.
    floor_neg = repo.funnel_trial_sharpe_var(-1)
    assert floor_neg.n_strategies == 0
    assert floor_neg.var_ann is None
```

Use `pytest.approx` (import it; the `_pytest_approx` placeholder above is `pytest.approx`). If the
existing test-suite has a shared `make_repo` fixture, reuse it; otherwise add a local fixture in
this file that constructs the same store peers use (in-memory sqlite). Confirm the
`record_search_trial` kwargs against `store.py:407`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/registry/test_funnel_floor.py -q`
Expected: FAIL — `ImportError: cannot import name 'FunnelFloor'` / `AttributeError:
funnel_trial_sharpe_var`.

- [ ] **Step 3: Add the `FunnelFloor` NamedTuple + Protocol method in `repository.py`**

Near the other NamedTuples (`FdrStreamState:28`):

```python
class FunnelFloor(NamedTuple):
    """Funnel-wide cross-strategy trial-Sharpe dispersion floor (#221 Slice 0). ``var_ann`` is the
    MEAN of per-strategy pooled trial-Sharpe variances (annualized) across strategies active in the
    rolling window, or ``None`` when fewer than ``MIN_FUNNEL_FLOOR_STRATEGIES`` finite per-strategy
    variances exist (fail-open -> Phase-1 behavior). ``n_strategies``/``n_total_rows`` are audit
    counts (the latter spans stale history, since the window SELECTS strategies but pools all of
    each selected strategy's rows)."""
    var_ann: float | None
    n_strategies: int
    n_total_rows: int
```

In the `StrategyRepository` Protocol, after `pooled_trial_sharpe_var:231`:

```python
    def funnel_trial_sharpe_var(self, window_days: int) -> FunnelFloor:
        """Per-strategy pooling FIRST, then mean across strategies active in the trailing
        ``window_days`` (anti-gaming: one vote per strategy regardless of combo count)."""
        ...
```

- [ ] **Step 4: Add the `created_at` index in `db.py`**

After `ix_search_trials_strategy:135`:

```sql
CREATE INDEX IF NOT EXISTS ix_search_trials_created_at ON search_trials(created_at);
```

(No `SCHEMA_VERSION` bump — `CREATE INDEX IF NOT EXISTS` is idempotent and additive.)

- [ ] **Step 5: Extract the pure pooling helper + implement the accessor in `store.py`**

Refactor `pooled_trial_sharpe_var:425` to delegate to a new module-level pure helper, then add the
funnel accessor. The helper holds the exact pooled-sample-variance formula already at
`store.py:444-449`:

```python
def _pool_trial_sharpe_var(triples: list[tuple[int, float, float]]) -> float | None:
    """Exact pooled SAMPLE variance (ddof=1) of trial Sharpes from ``(count, mean, var)`` triples.
    ``None`` for empty input; ``0.0`` for total count <= 1. Callers must pre-validate each triple
    (finite mean/var, count >= 1, var >= 0); this helper assumes clean triples."""
    if not triples:
        return None
    total_n = sum(n for n, _, _ in triples)
    if total_n <= 1:
        return 0.0
    grand_mean = sum(n * m for n, m, _ in triples) / total_n
    sse = sum((n - 1) * v + n * (m - grand_mean) ** 2 for n, m, v in triples)
    return sse / (total_n - 1)


def _validated_triples(rows) -> list[tuple[int, float, float]] | None:
    """Validate raw (n, mean, var) DB rows. Returns None (fail closed) if ANY row has a
    NULL/NaN/inf/negative stat — NULL rows are NEVER silently skipped."""
    triples: list[tuple[int, float, float]] = []
    for r in rows:
        n, mean, var = r["n"], r["mean"], r["var"]
        if n is None or mean is None or var is None:
            return None
        if not (math.isfinite(mean) and math.isfinite(var)) or int(n) < 1 or var < 0.0:
            return None
        triples.append((int(n), float(mean), float(var)))
    return triples
```

Rewrite `pooled_trial_sharpe_var` to reuse them (behavior byte-identical to today, including the
empty-rows `None` and the all-validation `None`):

```python
    def pooled_trial_sharpe_var(self, strategy_name: str) -> float | None:
        rows = self._conn.execute(
            "SELECT trial_sharpe_count AS n, trial_sharpe_mean AS mean,"
            " trial_sharpe_var_ann AS var FROM search_trials WHERE strategy_name=?",
            (strategy_name,),
        ).fetchall()
        if not rows:
            return None
        triples = _validated_triples(rows)
        if triples is None:
            return None
        return _pool_trial_sharpe_var(triples)
```

Add the funnel accessor after `windowed_search_combos:564`:

```python
    def funnel_trial_sharpe_var(self, window_days: int) -> FunnelFloor:
        """Per-strategy pooling FIRST (anti-gaming: one vote per strategy regardless of combo
        count), then MEAN across strategies with at least one search_trials row in the trailing
        ``window_days``. A selected strategy pools ALL its rows (the window selects strategies, it
        does NOT slice rows). A strategy with any NULL/NaN/inf stat row is excluded. Returns
        FunnelFloor(None, ...) when fewer than MIN_FUNNEL_FLOOR_STRATEGIES finite variances exist
        (fail-open -> Phase-1 behavior). ISO-8601 UTC timestamps sort lexically, so a string `>=`
        on created_at is chronological."""
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        # SELECT all rows of every strategy that has at least one in-window row. The window
        # filters which STRATEGIES are eligible; pooling then uses every row of each.
        rows = self._conn.execute(
            "SELECT strategy_name AS name, trial_sharpe_count AS n, trial_sharpe_mean AS mean,"
            " trial_sharpe_var_ann AS var FROM search_trials WHERE strategy_name IN"
            " (SELECT DISTINCT strategy_name FROM search_trials WHERE created_at >= ?)",
            (cutoff,),
        ).fetchall()
        by_strategy: dict[str, list] = {}
        for r in rows:
            by_strategy.setdefault(r["name"], []).append(r)
        per_strategy_vars: list[float] = []
        total_rows = 0
        for name_rows in by_strategy.values():
            triples = _validated_triples(name_rows)
            if triples is None:
                continue  # excluded: a NULL/non-finite stat in any of this strategy's rows
            var_s = _pool_trial_sharpe_var(triples)
            if var_s is None or not math.isfinite(var_s):
                continue
            per_strategy_vars.append(var_s)
            total_rows += len(name_rows)
        n_strategies = len(per_strategy_vars)
        from algua.research.gates import MIN_FUNNEL_FLOOR_STRATEGIES
        if n_strategies < MIN_FUNNEL_FLOOR_STRATEGIES:
            return FunnelFloor(None, n_strategies, total_rows)
        return FunnelFloor(sum(per_strategy_vars) / n_strategies, n_strategies, total_rows)
```

Import `FunnelFloor` at the top of `store.py` (it imports from `repository`/its own module — match
the existing import of `FdrStreamState`/`StrategyRecord` there). The `MIN_FUNNEL_FLOOR_STRATEGIES`
import is function-local to avoid a `store → gates` module-level import cycle; confirm
`gates → store` is not imported at module load (it is not — `gates.py` is pure-math). If a
module-level import is clean per `lint-imports`, prefer it; otherwise keep it function-local.

- [ ] **Step 6: Run the accessor tests to verify they pass**

Run: `uv run pytest tests/registry/test_funnel_floor.py -q`
Expected: PASS. Fix the `created_at`-window test per the note in Step 1 if the `make_repo` fixture
gives no per-row timestamp control — the `-1` day window is the robust always-empty assertion.

- [ ] **Step 7: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass. (`pooled_trial_sharpe_var` refactor must not change any existing test outcome.)

- [ ] **Step 8: Commit**

```bash
git add algua/registry/store.py algua/registry/repository.py algua/registry/db.py \
        tests/registry/test_funnel_floor.py
git commit -m "feat(221): funnel-wide trial-Sharpe dispersion-floor accessor — Slice 0 of #221"
```

---

## Task 2: Apply the floor in the pure gate (`gates.py`, PROTECTED)

**Files:**
- Modify: `algua/research/gates.py` (constant near `DSR_ALPHA:23`; `dsr_confidence:148` param;
  `GateDecision:208` fields + `to_dict:239`; `evaluate_gate:313` param + wiring)
- Test: `tests/research/test_dsr_dispersion_floor.py` (create)

**Interfaces:**
- Consumes: `dsr_confidence(sr_obs_per_period, t, skew, raw_kurtosis, n_trials,
  trial_sr_var_per_period)` (existing); `ANN` from `algua.backtest._constants`.
- Produces:
  - `MIN_FUNNEL_FLOOR_STRATEGIES: int = 5` (module constant, read by `store.py`).
  - `dsr_confidence(..., funnel_floor_var_per_period: float | None = None)` — new trailing param;
    applies `var_used = max(own, floor)` before `SR*`. Default `None` ⇒ Phase-1 behavior.
  - `evaluate_gate(..., dsr_funnel_floor_var_ann: float | None = None)` — new keyword param.
  - `GateDecision.dsr_funnel_floor_var_ann: float | None`,
    `GateDecision.dsr_funnel_floor_n_strategies: int | None`,
    `GateDecision.dsr_funnel_floor_n_total_rows: int | None` — audit fields, in `to_dict`.

- [ ] **Step 1: Write the failing gate-algebra + tighten-only tests**

Create `tests/research/test_dsr_dispersion_floor.py`:

```python
import math

from algua.backtest._constants import ANN
from algua.research import gates
from algua.research.gates import MIN_FUNNEL_FLOOR_STRATEGIES, dsr_confidence


def test_min_funnel_floor_strategies_value():
    assert MIN_FUNNEL_FLOOR_STRATEGIES == 5


def test_floor_below_own_leaves_confidence_unchanged():
    base = dict(sr_obs_per_period=0.10, t=120, skew=0.0, raw_kurtosis=3.0, n_trials=50,
                trial_sr_var_per_period=0.04)
    own = dsr_confidence(**base)
    floored = dsr_confidence(**base, funnel_floor_var_per_period=0.01)  # floor < own
    assert own is not None and floored is not None
    assert floored == own  # max(0.04, 0.01) == 0.04 -> no change


def test_floor_above_own_lowers_confidence():
    base = dict(sr_obs_per_period=0.10, t=120, skew=0.0, raw_kurtosis=3.0, n_trials=50,
                trial_sr_var_per_period=0.01)
    own = dsr_confidence(**base)
    floored = dsr_confidence(**base, funnel_floor_var_per_period=0.09)  # floor > own
    assert own is not None and floored is not None
    assert floored < own  # higher SR* -> lower confidence


def test_floor_none_is_phase1_behavior():
    base = dict(sr_obs_per_period=0.10, t=120, skew=0.0, raw_kurtosis=3.0, n_trials=50,
                trial_sr_var_per_period=0.04)
    assert dsr_confidence(**base, funnel_floor_var_per_period=None) == dsr_confidence(**base)


def test_tighten_only_property_over_grid():
    # For every (own_var, floor_var): the floored confidence is <= the un-floored one (never up).
    thresh = 1.0 - gates.DSR_ALPHA
    for own in [0.0, 0.005, 0.01, 0.04, 0.09, 0.16]:
        for floor in [None, 0.0, 0.005, 0.04, 0.09, 0.25]:
            base = dict(sr_obs_per_period=0.12, t=90, skew=-0.2, raw_kurtosis=4.0,
                        n_trials=40, trial_sr_var_per_period=own)
            old = dsr_confidence(**base)
            new = dsr_confidence(**base, funnel_floor_var_per_period=floor)
            if old is None or new is None:
                continue
            assert new <= old + 1e-12
            old_pass = old >= thresh
            new_pass = new >= thresh
            # tighten-only: a pass can only be revoked, never created.
            assert not (new_pass and not old_pass)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/research/test_dsr_dispersion_floor.py -q`
Expected: FAIL — `ImportError: cannot import name 'MIN_FUNNEL_FLOOR_STRATEGIES'` /
`TypeError: unexpected keyword 'funnel_floor_var_per_period'`.

- [ ] **Step 3: Add the constant**

In `gates.py` after `DSR_ALPHA`/`EULER_MASCHERONI:23-24`:

```python
# Funnel-wide dispersion floor (#221, Phase 3 Slice 0). Min finite per-strategy trial-Sharpe
# variances needed to form a meaningful cross-strategy floor. Below this, the floor is unavailable
# and the DSR falls back to own-sweep variance (Phase-1 behavior). Protected — raising it weakens
# the floor's availability; the floor can only ever TIGHTEN the gate, so its absence is conservative.
MIN_FUNNEL_FLOOR_STRATEGIES = 5
```

- [ ] **Step 4: Apply the floor inside `dsr_confidence`**

Add the trailing parameter and the `max` BEFORE the existing finite/negative guard so the guard
still validates the value actually used:

```python
def dsr_confidence(
    sr_obs_per_period: float,
    t: int,
    skew: float,
    raw_kurtosis: float,
    n_trials: int,
    trial_sr_var_per_period: float,
    funnel_floor_var_per_period: float | None = None,
) -> float | None:
```

In the body, replace the variance validation/use. After the existing
`if not math.isfinite(sr_obs...)` block and BEFORE the
`if not math.isfinite(trial_sr_var_per_period)...` check, floor the variance:

```python
    var_used = trial_sr_var_per_period
    # Funnel-wide dispersion floor (#221 Slice 0): max(own, floor) can only RAISE SR* -> lower
    # confidence -> tighten-only. A None/non-finite floor falls back to own (Phase-1 behavior).
    if funnel_floor_var_per_period is not None and math.isfinite(funnel_floor_var_per_period) \
            and funnel_floor_var_per_period > var_used:
        var_used = funnel_floor_var_per_period
    if not math.isfinite(var_used) or var_used < 0.0:
        return None
```

Then replace every subsequent use of `trial_sr_var_per_period` in the `SR*` computation with
`var_used` (the `math.sqrt(trial_sr_var_per_period)` term at `gates.py:183`). Update the docstring
to note the optional floor.

- [ ] **Step 5: Add audit fields to `GateDecision` + `to_dict`**

In `GateDecision` (after `dsr_raw_kurtosis:228`):

```python
    dsr_funnel_floor_var_ann: float | None = None
    dsr_funnel_floor_n_strategies: int | None = None
    dsr_funnel_floor_n_total_rows: int | None = None
```

In `to_dict` (alongside the other `dsr_*` keys, using `_f` for the float):

```python
            "dsr_funnel_floor_var_ann": _f(self.dsr_funnel_floor_var_ann),
            "dsr_funnel_floor_n_strategies": self.dsr_funnel_floor_n_strategies,
            "dsr_funnel_floor_n_total_rows": self.dsr_funnel_floor_n_total_rows,
```

- [ ] **Step 6: Wire the floor through `evaluate_gate`**

Add the keyword param to `evaluate_gate`:

```python
    dsr_binding: bool = False,
    dsr_trial_var_ann: float | None = None,
    dsr_funnel_floor_var_ann: float | None = None,
```

In the `if dsr_binding:` block, convert the annualized floor to per-period (mirroring
`dsr_trial_var_ann / ANN`) and pass it through:

```python
    if dsr_binding:
        var_pp = (dsr_trial_var_ann / ANN) if dsr_trial_var_ann is not None else None
        floor_pp = (
            dsr_funnel_floor_var_ann / ANN
            if dsr_funnel_floor_var_ann is not None and math.isfinite(dsr_funnel_floor_var_ann)
            else None
        )
        if var_pp is not None and math.isfinite(var_pp):
            dsr_conf = dsr_confidence(
                sr_obs_ann / math.sqrt(ANN), t_hold, skew, raw_kurt, n_for_dsr, var_pp,
                funnel_floor_var_per_period=floor_pp)
        ...
```

In the returned `GateDecision(...)`, populate the audit fields (only when binding, mirroring the
other `dsr_*` fields):

```python
        dsr_funnel_floor_var_ann=(dsr_funnel_floor_var_ann if dsr_binding else None),
```

(The `n_strategies`/`n_total_rows` audit values are passed in by `promotion.py` in Task 3; for the
pure gate, accept them as two more optional `evaluate_gate` kwargs
`dsr_funnel_floor_n_strategies: int | None = None`,
`dsr_funnel_floor_n_total_rows: int | None = None` and forward them to `GateDecision` when binding.
Add those two kwargs to the signature in this step too.)

- [ ] **Step 7: Run the gate tests + quality gate**

Run: `uv run pytest tests/research/test_dsr_dispersion_floor.py -q`
Expected: PASS.
Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass (existing `dsr_confidence` / `evaluate_gate` callers unaffected — new params
default to `None`).

- [ ] **Step 8: Commit**

```bash
git add algua/research/gates.py tests/research/test_dsr_dispersion_floor.py
git commit -m "feat(221): apply funnel dispersion floor in DSR gate (max(own,floor)) — Slice 0"
```

---

## Task 3: Wire the accessor into the orchestrator (`promotion.py`, PROTECTED)

**Files:**
- Modify: `algua/registry/promotion.py` (`run_gate:195-201`)
- Test: extend `tests/research/test_dsr_dispersion_floor.py` with an integration test (or add to the
  existing promotion test module — locate it via `grep -rl "def run_gate\|run_gate(" tests`).

**Interfaces:**
- Consumes: `repo.funnel_trial_sharpe_var(FUNNEL_WINDOW_DAYS) → FunnelFloor`;
  `evaluate_gate(..., dsr_funnel_floor_var_ann=, dsr_funnel_floor_n_strategies=,
  dsr_funnel_floor_n_total_rows=)` from Task 2.
- Produces: `decision_json` now carries `dsr_funnel_floor_*` fields on every measured-breadth gate.

- [ ] **Step 1: Write a failing integration test**

In `tests/research/test_dsr_dispersion_floor.py`, add a test that drives `run_gate` (or the full
`research promote` path the existing promotion tests use) with a real repo where the funnel has
dispersion but the strategy under promotion ran a single low-dispersion combo, and asserts the
floor is recorded and tightens the outcome vs an isolated funnel. Mirror the existing promotion
integration test setup (find it: `grep -rl "run_gate\|research_promote\|evaluate_gate" tests`).

```python
def test_floor_recorded_and_tightens_in_run_gate(promotion_fixture):
    # Build a funnel of >=5 strategies with real trial-Sharpe dispersion, plus the strategy under
    # promotion whose own sweep is a single near-duplicate combo (own_var ~ 0). Assert:
    #  - decision.dsr_funnel_floor_var_ann is not None and > 0
    #  - decision.dsr_funnel_floor_n_strategies >= 5
    #  - the same strategy in an isolated funnel (<5 siblings) records var_ann None (Phase-1).
    ...
```

Flesh this out against the actual promotion test harness in the repo (reuse its fixtures; do not
invent a new one). If no end-to-end `run_gate` fixture exists, assert at the `evaluate_gate` level
with a hand-built `FunnelFloor` and a stub repo whose `funnel_trial_sharpe_var` returns it.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/research/test_dsr_dispersion_floor.py -q -k run_gate`
Expected: FAIL (floor fields are `None`/absent — accessor not yet called).

- [ ] **Step 3: Call the accessor in `run_gate` and pass it through**

At `promotion.py:195`, after computing `dsr_trial_var_ann`:

```python
    dsr_trial_var_ann = repo.pooled_trial_sharpe_var(name) if dsr_binding else None
    funnel_floor = repo.funnel_trial_sharpe_var(FUNNEL_WINDOW_DAYS) if dsr_binding else None
    decision = evaluate_gate(
        wf, criteria, n_combos=breadth.n_funnel, breadth_provenance=breadth.provenance,
        pit_ok=pit_ok, allow_non_pit=allow_non_pit, own_lifetime_combos=breadth.own,
        windowed_total_combos=breadth.windowed_total, funnel_window_days=FUNNEL_WINDOW_DAYS,
        dsr_binding=dsr_binding, dsr_trial_var_ann=dsr_trial_var_ann,
        dsr_funnel_floor_var_ann=(funnel_floor.var_ann if funnel_floor else None),
        dsr_funnel_floor_n_strategies=(funnel_floor.n_strategies if funnel_floor else None),
        dsr_funnel_floor_n_total_rows=(funnel_floor.n_total_rows if funnel_floor else None),
    )
```

- [ ] **Step 4: Run the integration test + quality gate**

Run: `uv run pytest tests/research/test_dsr_dispersion_floor.py -q`
Expected: PASS.
Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add algua/registry/promotion.py tests/research/test_dsr_dispersion_floor.py
git commit -m "feat(221): wire funnel dispersion floor into promotion run_gate — Slice 0"
```

---

## Self-Review notes

- **Spec coverage:** Component (d) footprint — `store.py` accessor (Task 1), `gates.py` floor +
  constant + audit fields (Task 2), `promotion.py` wiring + audit (Task 3), `created_at` index
  (Task 1). Per-strategy-pooling-first anti-gaming (Task 1 test). Tighten-only property (Task 2
  test). Integration (Task 3 test). All four spec test bullets covered.
- **No migration / no schema bump:** only an index addition (Task 1 Step 4) — confirmed.
- **Type consistency:** `FunnelFloor(var_ann, n_strategies, n_total_rows)` defined in Task 1,
  consumed in Task 3; `funnel_floor_var_per_period` param name consistent Task 2 ↔ its caller in
  `evaluate_gate`; `dsr_funnel_floor_var_ann` consistent across `GateDecision`/`evaluate_gate`/
  `promotion.py`.
- **Deferred (NOT in this slice):** the optional family-level dedup hardening (spec "Optional Slice
  0 hardening") is explicitly out — file as a follow-up only if a family-flood attack appears in the
  audit trail.
