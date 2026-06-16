# Factor funnel-FDR accounting — Issue #219

Slice E of #140: persist a factor-evaluation ledger and wire funnel-wide multiple-testing
correction into `algua factor eval`, so the reported IC t-stat is honest about search breadth.

## Why

`algua factor eval` (slice B, #215) reports a raw IC t-stat with `fdr_corrected: false`. Each
factor evaluation is a hypothesis test; evaluating many factors inflates false positives. The
factor catalogue + lineage make the hypothesis surface queryable, so the inputs for FDR
accounting now exist.

Key insight: a factor's IR (`mean_ic / ic_std`) is a Sharpe ratio of the per-timestamp IC
series. So the #211 DSR evidence machinery (plus the #137 sqrt(2·ln N) breadth haircut) both
apply directly — no annualization needed; IC is already per-period.

Design decisions (confirmed with operator):
1. **Correction = breadth haircut + DSR** (mirror #211): tighten-only AND-check.
2. **Report-only + record blast-radius**: no change to research-promote / gates.py.

## Scope

- `factor_evaluations` ledger (DB table, new)
- IC skew/kurtosis moments (for DSR non-normality adjustment)
- Pure correction math module (`algua/research/factor_fdr.py`)
- Repository ledger methods
- CLI wiring in `eval_factor`
- Docs/skill updates

NOT in scope: strategy research-promote, gates.py/promotion.py, other slice-B deferrals
(`__factor__:` name ban, `--track`, single bar-fetch refactor, PIT IC-panel masking, vectorized IC).

## TDD Tasks

### Task 1 — Ledger schema (RED → GREEN → commit) ✅ TDD

**File:** `algua/registry/db.py`

Add `factor_evaluations` table to `_SCHEMA`, bump `SCHEMA_VERSION` 24 → 25.

Schema (brand-new table → `CREATE TABLE IF NOT EXISTS` is sufficient; no `_add_missing_columns`):

```sql
CREATE TABLE IF NOT EXISTS factor_evaluations (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name               TEXT    NOT NULL,
    import_path               TEXT    NOT NULL,
    code_hash                 TEXT    NOT NULL,
    hypothesis_hash           TEXT    NOT NULL,   -- dedup key: identity + window + params
    period_start              TEXT    NOT NULL,
    period_end                TEXT    NOT NULL,
    horizon                   INTEGER NOT NULL,
    params_json               TEXT    NOT NULL DEFAULT '{}',
    construction              TEXT    NOT NULL,
    construction_params_json  TEXT    NOT NULL DEFAULT '{}',
    n_obs                     INTEGER,
    mean_ic                   REAL,
    ic_ir                     REAL,
    t_stat                    REAL,
    ic_skew                   REAL,
    ic_kurtosis               REAL,
    n_dependents              INTEGER NOT NULL DEFAULT 0,
    data_source               TEXT    NOT NULL,   -- 'demo', 'snapshot', 'provider'
    snapshot_id               TEXT,
    actor                     TEXT    NOT NULL DEFAULT 'agent',
    created_at                TEXT    NOT NULL,
    -- Correction columns (NULL until finalize_factor_evaluation):
    n_hypotheses              INTEGER,
    dsr_confidence            REAL,
    significant               INTEGER            -- 0/1/NULL; NULL = not yet finalized
);
CREATE INDEX IF NOT EXISTS ix_factor_evaluations_factor
    ON factor_evaluations (factor_name);
CREATE INDEX IF NOT EXISTS ix_factor_evaluations_created
    ON factor_evaluations (created_at);
CREATE INDEX IF NOT EXISTS ix_factor_evaluations_hypothesis
    ON factor_evaluations (hypothesis_hash, created_at);
```

Test: v24 DB without table → `migrate()` → table exists, `user_version == 25`, idempotent.

### Task 2 — IC higher moments (RED → GREEN → commit) ✅ TDD

**File:** `algua/backtest/factor_eval.py` :: `factor_ic`

Add `ic_skew` and `ic_kurtosis` (raw/Pearson, `fisher=False`) to the returned dict.
Computed via `scipy.stats.skew` / `kurtosis(fisher=False)`, non-finite → `0.0` (same
pattern as `metrics.py:85-88`). `n < 2` branch: `None` for both.
Leave `fdr_corrected: False` (the CLI layer flips it after correction).

Test in `tests/test_factor_ic.py`: known IC series → check skew/kurt; `n<2` → None;
Gaussian-ish series → raw kurtosis ≈ 3.

### Task 3 — Pure correction math (RED → GREEN → commit) ✅ TDD

**File:** `algua/research/factor_fdr.py` (new)

Three functions:

```python
def breadth_benchmark_t(n_hypotheses: int) -> float:
    """sqrt(2·ln(max(N,1))): 0 at N=1, monotone. The expected-max inflator."""

def trial_ir_variance(irs: Iterable[float | None]) -> float | None:
    """Sample variance (ddof=1) of finite IRs; None if <2 finite values (fail-closed)."""

def correct_factor_ic(
    *, t_stat: float | None, ir: float | None, n_obs: int,
    ic_skew: float | None, ic_kurtosis: float | None,
    n_hypotheses: int, trial_ir_var: float | None,
    alpha: float = DSR_ALPHA,
) -> dict[str, Any]:
    """Returns the `fdr` block for the factor eval JSON.

    `significant` = breadth_significant AND (not dsr_binding OR dsr_significant).
    DSR binds only when trial_ir_var is not None AND n_hypotheses >= 2.
    When not binding, dsr_confidence/dsr_significant are None and skip_reason explains why.
    """
```

Reuse from `algua.research.gates`: `dsr_confidence`, `DSR_ALPHA`, `FUNNEL_WINDOW_DAYS`,
`effective_funnel_breadth`.

Output block shape:
```json
{
  "n_hypotheses": 7,
  "breadth_benchmark_t": 1.9939,
  "breadth_significant": true,
  "dsr_binding": true,
  "dsr_confidence": 0.97,
  "dsr_significant": true,
  "dsr_skip_reason": null,
  "significant": true,
  "fdr_corrected": true
}
```

Tests in `tests/test_factor_fdr.py`:
- `breadth_benchmark_t(1) == 0.0`, monotone in N.
- `t_stat=None` → `significant: false`, `fdr_corrected: true`.
- `trial_ir_var=None` → `dsr_binding: false`, reason set, verdict = breadth only.
- N<2 → `dsr_binding: false`.
- Binding case: hand-computed `dsr_confidence` matches.
- AND-check: DSR fail flips breadth-pass → fail; DSR pass on breadth-fail → stays fail.

### Task 4 — Repository ledger methods (RED → GREEN → commit) ✅ TDD

**Files:** `algua/registry/repository.py`, `algua/registry/store.py`

Protocol + impl:

```python
def record_factor_evaluation(
    self, *, factor_name, import_path, code_hash, hypothesis_hash,
    period_start, period_end, horizon, params_json, construction,
    construction_params_json, n_obs, mean_ic, ic_ir, t_stat,
    ic_skew, ic_kurtosis, n_dependents, data_source, snapshot_id,
    actor, created_at,
) -> int:
    """INSERT raw row (correction cols NULL). Returns row id."""

def factor_hypothesis_breadth(
    self, factor_name: str, window_days: int
) -> tuple[int, int]:
    """(own_lifetime, windowed_total) via COUNT(DISTINCT hypothesis_hash)."""

def windowed_factor_irs(self, window_days: int) -> list[float]:
    """Latest IR per distinct hypothesis_hash within window, finite only."""

def finalize_factor_evaluation(
    self, id: int, n_hypotheses: int,
    dsr_confidence: float | None, significant: bool,
) -> None:
    """UPDATE correction cols."""
```

Breadth queries:
- `own_lifetime`: `SELECT COUNT(DISTINCT hypothesis_hash) FROM factor_evaluations WHERE factor_name=?`
- `windowed_total`: `SELECT COUNT(DISTINCT hypothesis_hash) FROM factor_evaluations WHERE created_at >= ?`
  (cutoff = `now(UTC) - window_days`, ISO string compare — mirrors `windowed_search_combos`).
  Self included (INSERT then query).
- `windowed_factor_irs`: group by `hypothesis_hash`, pick latest `created_at`, return `ic_ir`
  where finite.

Tests in `tests/test_factor_ledger.py` (mirror `test_shortlist_gate.py::_repo(tmp_path)`):
- INSERT → breadth includes self (own=1, windowed=1).
- Same `hypothesis_hash` repeated → distinct count = 1 (dedup).
- Different `hypothesis_hash` for same factor → own=2.
- Old `created_at` outside window → windowed omits it.
- `windowed_factor_irs` deduplicates to latest-per-hash.
- `finalize_factor_evaluation` writes correction cols; reading back shows them.

### Task 5 — CLI wiring (RED → GREEN → commit) ✅ TDD

**File:** `algua/cli/factor_cmd.py` :: `eval_factor`

After `evaluate_factor(...)`:
1. Compute `code_hash` and `hypothesis_hash` (pure helper `_factor_hashes(spec, params,
   horizon, construction, construction_params, start, end) -> tuple[str, str]`).
   - `code_hash`: SHA-256 of `inspect.getsource(load_factor_callable(spec))`.
   - `hypothesis_hash`: SHA-256 of canonical JSON of `(import_path, code_hash, sorted
     params items, horizon, construction, sorted construction_params items, start_iso, end_iso)`.
   Same params/window = same hash (dedup); different = new hypothesis.
2. Get `n_dependents = len(dependents_of(repo, name).dependents)`.
3. Get `data_source` and `snapshot_id` from `provider`.
4. `row_id = repo.record_factor_evaluation(...)`.
5. `own, windowed = repo.factor_hypothesis_breadth(name, FUNNEL_WINDOW_DAYS)`;
   `N = effective_funnel_breadth(own, windowed)`;
   `trial_ir_var = trial_ir_variance(repo.windowed_factor_irs(FUNNEL_WINDOW_DAYS))`.
6. `fdr = correct_factor_ic(t_stat=ic["t_stat"], ir=ic["ir"], n_obs=ic["n_obs"],
   ic_skew=ic["ic_skew"], ic_kurtosis=ic["ic_kurtosis"], n_hypotheses=N,
   trial_ir_var=trial_ir_var)`.
7. `repo.finalize_factor_evaluation(row_id, N, fdr["dsr_confidence"], fdr["significant"])`.
8. Build output: `result_dict = result.to_dict()`, `result_dict["ic"]["fdr_corrected"] = True`,
   `result_dict["fdr"] = fdr`, `result_dict["n_dependents"] = n_dependents`, `emit(ok(result_dict))`.

Add `--actor TEXT` option (default `"agent"`) for ledger attribution.
Open `registry_conn()` context manager wrapping steps 2–7.

Tests in `tests/test_cli_factor.py` (CliRunner + ALGUA_DB_PATH autouse, `--demo`):
- Single eval → JSON has `fdr` block, `fdr_corrected: true` in both `ic` and `fdr`,
  `n_dependents` present.
- `ic["fdr_corrected"]` is `true` (CLI overrides the False set by `factor_ic`).
- Underpowered run (n_obs < 2, so t_stat=None) → `fdr.significant == false`.
- Two different factors → second eval sees `n_hypotheses >= 2`.
- Re-run identical factor/params/window → `n_hypotheses` does NOT grow (dedup).
- `dsr_binding` becomes true once ≥2 distinct hypotheses with finite IRs exist.

### Task 6 — Docs / spec / skill updates (commit) ✅

- `CLAUDE.md`: note `factor eval` is now FDR-corrected and writes the `factor_evaluations`
  ledger (still ephemeral re: registry/holdout/gate — only the eval ledger). Add
  `algua factor eval --actor agent` mention.
- `docs/superpowers/specs/2026-06-15-factor-standalone-eval-140-design.md`: flip slice-E
  deferral notes to "implemented in #219".
- `interpret-results` skill (`kb/skills/` or `.claude/skills/`): update factor-eval caveat.
- `AGENTS.md`: add note that `factor_evaluations` is the audit ledger; factor FDR is
  report-only (no gate to consume).

## Gate (before PR)

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
```

## Verification (end-to-end CLI smoke)

```bash
# Single eval → fdr block
uv run algua factor eval <factor> --symbols AAPL,MSFT --construction equal_weight --demo \
  | python -c "import sys,json; d=json.load(sys.stdin); print(d['fdr'])"

# Re-run identical → n_hypotheses stays 1
# Second different factor → n_hypotheses = 2, dsr_binding flips true
# Blast radius
uv run algua factor dependents <factor>
```
