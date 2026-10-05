# Story 2.1 live qualification contract

Normative companion to [SPEC.md](SPEC.md). Baseline: `e875b6d` (line references below are to that
tree). "Row" means a `gate_evaluations` row (research gate) or a `forward_gate_evaluations` row
(forward evaluation; the newest one for a deployment is its certificate).

## 1. The closed vocabulary (CAP-1)

### 1.1 Module

New module `algua/registry/relaxations.py`. Pure: no I/O, no sqlite. It imports only
`algua.contracts.lifecycle.Actor`, `algua.research.gates.GateCriteria` and
`algua.research.forward_gates.ForwardGateCriteria`. It defines, in this order:

```python
UNRECORDED = "unrecorded"   # the predicate's reading of a NULL or unreadable set; never stored

RESEARCH_FLAG_RELAXATIONS = (
    "agent_walls_waived", "allow_holdout_reuse", "allow_non_pit", "assume_terminal_last_close",
    "declared_breadth", "demo_data", "new_family")
RESEARCH_THRESHOLDS = (      # every GateCriteria field; every one is higher-is-stricter
    "min_holdout_observations", "min_holdout_return", "min_holdout_sharpe",
    "min_pct_positive_windows", "min_window_sharpe")
FORWARD_HIGHER_IS_STRICTER = (
    "degradation_factor", "forward_sharpe_confidence", "min_forward_observations",
    "min_forward_vol", "min_session_coverage", "sharpe_floor")
FORWARD_LOWER_IS_STRICTER = ("max_forward_drawdown", "max_staleness_sessions")

RESEARCH_VOCABULARY: frozenset[str]   # the 7 flags + "threshold:" + each RESEARCH_THRESHOLDS field (12)
FORWARD_VOCABULARY: frozenset[str]    # "threshold:" + each forward field (8)

RESEARCH_PROMOTE_INPUTS: Mapping[str, str | None]   # §1.4
PAPER_PROMOTE_INPUTS: Mapping[str, str | None]      # §1.4

def research_relaxations(
    *, actor: Actor, demo: bool, n_combos: int | None, allow_holdout_reuse: bool,
    allow_non_pit: bool, assume_terminal_last_close: bool, new_family: str | None,
    criteria: GateCriteria,
) -> tuple[str, ...]: ...
def forward_relaxations(criteria: ForwardGateCriteria) -> tuple[str, ...]: ...
def encode_relaxations(tokens: tuple[str, ...]) -> str: ...
def decode_relaxations(text: str | None, vocabulary: frozenset[str]) -> tuple[str, ...] | None: ...
def guard_agent_relaxations(actor, *, declared_combos, allow_holdout_reuse, allow_non_pit) -> None
```

- `guard_agent_relaxations` moves here verbatim from `algua/registry/promotion.py:37-54` (its
  body and message are unchanged; `promotion.py` imports it from here). This is the
  `promotion.py` carve (§9).
- Both mappings are `types.MappingProxyType` (read-only).
- Both set functions return `tuple(sorted(set(...)))`; `()` means unrelaxed.

### 1.2 The direction rule

A threshold is **not** a relaxation exactly when its value is provably at least as strict as the
protected default: `value >= default` for a higher-is-stricter field, `value <= default` for a
lower-is-stricter field. Everything else is a relaxation, including a non-finite value
(`NaN` compares false both ways). The protected default is the dataclass default of
`GateCriteria()` / `ForwardGateCriteria()`, read at call time (never a copied constant).
`+inf` on a higher-is-stricter field is stricter, not a relaxation (the evaluator fails that check
closed, which is stricter still).

### 1.3 Research and forward sets

`research_relaxations` returns the sorted union of:

| token | predicate (the arguments as passed) |
|---|---|
| `agent_walls_waived` | `actor is Actor.HUMAN` (the effective, authenticated actor) |
| `declared_breadth` | `n_combos is not None` (given, even when measured breadth then wins) |
| `allow_holdout_reuse` | `allow_holdout_reuse is True` (given, whether or not an overlap existed) |
| `allow_non_pit` | `allow_non_pit is True` (given, whether or not the universe was PIT) |
| `assume_terminal_last_close` | `assume_terminal_last_close is True` |
| `new_family` | `new_family is not None and actor is Actor.HUMAN` |
| `demo_data` | `demo is True` |
| `threshold:<f>` | for each `f` in `RESEARCH_THRESHOLDS`: `not (getattr(criteria, f) >= getattr(GateCriteria(), f))` |

`forward_relaxations` returns the sorted union of `threshold:<f>` for each
`f` in `FORWARD_HIGHER_IS_STRICTER` with `not (value >= default)` and each `f` in
`FORWARD_LOWER_IS_STRICTER` with `not (value <= default)`.

`encode_relaxations(tokens)` raises `ValueError` unless `tokens == tuple(sorted(set(tokens)))`
and every token is in `RESEARCH_VOCABULARY | FORWARD_VOCABULARY`; it returns
`json.dumps(list(tokens), separators=(",", ":"))` (`"[]"` for the empty tuple).

`decode_relaxations(text, vocabulary)` returns `None` when `text is None`, when `json.loads`
fails, when the value is not a list of `str`, when any token is outside `vocabulary`, or when
`encode_relaxations(tuple(value)) != text` (not canonical); otherwise the tuple. `None` is read
by the predicate as `unrecorded`.

### 1.4 Every option, classified (AC2)

`RESEARCH_PROMOTE_INPUTS` covers the union of the `research promote` Typer parameters
(`algua/cli/research_cmd.py:32-95`) and `promote_task`'s parameters
(`algua/registry/promote_run.py:103-114`, also the body of `research run-all` and the merge-back
seam). `PAPER_PROMOTE_INPUTS` covers the `paper promote` Typer parameters
(`algua/cli/paper_cmd.py:1157-1198`). Value `None` means "not a relaxation input".

| command | parameter (`--option`) | protected default | relaxation? | token | predicate |
|---|---|---|---|---|---|
| research | `name` (argument) | - | no | - | - |
| research | `start`, `end` | - | no: the evaluation window | - | - |
| research | `demo` (`--demo`) | `False` | **yes** | `demo_data` | `demo is True` |
| research | `snapshot`, `fundamentals_snapshot`, `news_snapshot` | - | no: data identity | - | - |
| research | `universe` | - | no: absence fails `pit_required` unless `allow_non_pit` | - | - |
| research | `windows` | `4` | no: evaluation design input, not a wall | - | - |
| research | `holdout_frac` | `0.2` | no: the binding 63-observation floor still applies | - | - |
| research | `min_holdout_sharpe` | `0.5` | **yes** if looser | `threshold:min_holdout_sharpe` | `not (v >= 0.5)` |
| research | `min_holdout_return` | `0.0` | **yes** if looser | `threshold:min_holdout_return` | `not (v >= 0.0)` |
| research | `min_pct_positive` | `0.6` | **yes** if looser | `threshold:min_pct_positive_windows` | `not (v >= 0.6)` |
| research | `min_window_sharpe` | `0.0` | **yes** if looser | `threshold:min_window_sharpe` | `not (v >= 0.0)` |
| research | (no option) `GateCriteria.min_holdout_observations` | `63` | **yes** if looser | `threshold:min_holdout_observations` | `not (v >= 63)`; `promote_task` never sets it |
| research | `n_combos` (`--n-combos`) | `None` | **yes** | `declared_breadth` | `n_combos is not None` |
| research | `allow_holdout_reuse` | `False` | **yes** | `allow_holdout_reuse` | flag given |
| research | `allow_non_pit` | `False` | **yes** | `allow_non_pit` | flag given |
| research | `delistings` | - | no: absence fails closed on a held-into-gap name | - | - |
| research | `assume_terminal_last_close` | `False` | **yes** | `assume_terminal_last_close` | flag given |
| research | `actor` (`--actor`) | `agent` | **yes** for human | `agent_walls_waived` | effective actor is human |
| research | `actor_signature` | - | no: authentication | - | - |
| research | `new_family` (`--new-family`) | `None` | **yes** for human | `new_family` | given and effective actor human |
| research | `summary` (CLI only) | - | no: output projection | - | - |
| research | `reload`, `attempt_token` (`promote_task` only) | - | no: worker hygiene, merge-back idempotency | - | - |
| paper | `name` (argument), `actor`, `actor_signature` | - | no (§1.5) | - | - |
| paper | `degradation_factor` | `0.5` | **yes** if looser | `threshold:degradation_factor` | `not (v >= 0.5)` |
| paper | `sharpe_floor` | `0.3` | **yes** if looser | `threshold:sharpe_floor` | `not (v >= 0.3)` |
| paper | `min_observations` | `63` | **yes** if looser | `threshold:min_forward_observations` | `not (v >= 63)` |
| paper | `min_coverage` | `0.9` | **yes** if looser | `threshold:min_session_coverage` | `not (v >= 0.9)` |
| paper | `min_vol` | `0.02` | **yes** if looser | `threshold:min_forward_vol` | `not (v >= 0.02)` |
| paper | `max_drawdown` | `0.25` | **yes** if looser | `threshold:max_forward_drawdown` | `not (v <= 0.25)` |
| paper | `max_staleness` | `5` | **yes** if looser | `threshold:max_staleness_sessions` | `not (v <= 5)` |
| paper | `forward_sharpe_confidence` | `0.95` | **yes** if looser | `threshold:forward_sharpe_confidence` | `not (v >= 0.95)` |

### 1.5 Why the actor rows differ

A human `research promote` skips walls an agent run enforces with no flag at all: the
reproducible-source guard (`promotion.py:126`), the cost floor (`:152`), the declared
feature-lookback (`:168`), the gated-universe subset (`:206`), and the seeded, rate-capped agent
NOVEL/PARENTAGE family path (`family_assignment.py:194-256`). The code calls these exemptions
"exploration", so a human research row is never at its protected defaults: it records
`agent_walls_waived`. A human who wants a live-eligible research gate runs `research promote
--actor agent`, which needs no signature. The forward gate has no actor-dependent wall besides
the thresholds (`forward_promotion.py:47-70`, `:164-180`), so a human `paper promote` at default
thresholds records `[]`. `--new-family` is ignored for an agent (`family_assignment.py:194-238`),
so an agent passing it records nothing.

### 1.6 The agent guard reuses the table

`guard_forward_relaxations` (`forward_promotion.py:47-70`) keeps its confidence check and its
human early return, and replaces its own direction lists with `relaxed =
forward_relaxations(criteria)`, raising
`ValueError("forward-gate relaxation requires --actor human: " + ", ".join(t.removeprefix("threshold:") for t in relaxed))`.
For every finite input the refusal set, order and message are byte-identical to today (stripping a
common prefix preserves sort order). The one deliberate delta: an agent's `NaN` float threshold,
which today evaluates to a guaranteed fail-closed row, is now refused at preflight like any other
relaxation.

## 2. Schema v49 (this section is the protected schema review)

New context fragment `algua/registry/db/relaxations.py`; `SCHEMA_VERSION` 48 → 49
(`algua/registry/db/constants.py:36`, with a v49 comment line). Like the v48 tick link, nothing is
added to the base `CREATE TABLE` text: the column reaches fresh and existing registries through
the same guarded ALTER.

```python
RELAXATION_TABLES = ("gate_evaluations", "forward_gate_evaluations")
RELAXATIONS_COLUMN = {"relaxations_json": (
    "TEXT CHECK (relaxations_json IS NULL OR CASE WHEN json_valid(relaxations_json)"
    " THEN json_type(relaxations_json) = 'array' AND json(relaxations_json) = relaxations_json"
    " ELSE 0 END)")}
```

```sql
CREATE TRIGGER IF NOT EXISTS trg_gate_evaluations_relaxations_recorded
BEFORE INSERT ON gate_evaluations
BEGIN
    SELECT RAISE(ABORT, 'gate_evaluations.relaxations_json is required on every new row')
    WHERE NEW.relaxations_json IS NULL;
    SELECT RAISE(ABORT, 'gate_evaluations.relaxations_json must be a canonical JSON array')
    WHERE NOT (CASE WHEN json_valid(NEW.relaxations_json)
                    THEN json_type(NEW.relaxations_json) = 'array'
                         AND json(NEW.relaxations_json) = NEW.relaxations_json
                    ELSE 0 END);
    SELECT RAISE(ABORT, 'gate_evaluations.relaxations_json holds a token outside the research vocabulary')
    WHERE EXISTS (SELECT 1 FROM json_each(NEW.relaxations_json) e
                  WHERE e.type <> 'text' OR e.value NOT IN (
                      'agent_walls_waived', 'allow_holdout_reuse', 'allow_non_pit',
                      'assume_terminal_last_close', 'declared_breadth', 'demo_data', 'new_family',
                      'threshold:min_holdout_observations', 'threshold:min_holdout_return',
                      'threshold:min_holdout_sharpe', 'threshold:min_pct_positive_windows',
                      'threshold:min_window_sharpe'));
    SELECT RAISE(ABORT, 'gate_evaluations.relaxations_json must be sorted and unique')
    WHERE EXISTS (SELECT 1 FROM json_each(NEW.relaxations_json) a
                  JOIN json_each(NEW.relaxations_json) b ON b.key = a.key + 1
                  WHERE NOT (a.value < b.value));
END;

CREATE TRIGGER IF NOT EXISTS trg_gate_evaluations_relaxations_immutable
BEFORE UPDATE OF relaxations_json ON gate_evaluations
BEGIN SELECT RAISE(ABORT, 'gate_evaluations.relaxations_json is immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_forward_gate_evaluations_relaxations_recorded
BEFORE INSERT ON forward_gate_evaluations
BEGIN
    SELECT RAISE(ABORT, 'forward_gate_evaluations.relaxations_json is required on every new row')
    WHERE NEW.relaxations_json IS NULL;
    SELECT RAISE(ABORT, 'forward_gate_evaluations.relaxations_json must be a canonical JSON array')
    WHERE NOT (CASE WHEN json_valid(NEW.relaxations_json)
                    THEN json_type(NEW.relaxations_json) = 'array'
                         AND json(NEW.relaxations_json) = NEW.relaxations_json
                    ELSE 0 END);
    SELECT RAISE(ABORT, 'forward_gate_evaluations.relaxations_json holds a token outside the forward vocabulary')
    WHERE EXISTS (SELECT 1 FROM json_each(NEW.relaxations_json) e
                  WHERE e.type <> 'text' OR e.value NOT IN (
                      'threshold:degradation_factor', 'threshold:forward_sharpe_confidence',
                      'threshold:max_forward_drawdown', 'threshold:max_staleness_sessions',
                      'threshold:min_forward_observations', 'threshold:min_forward_vol',
                      'threshold:min_session_coverage', 'threshold:sharpe_floor'));
    SELECT RAISE(ABORT, 'forward_gate_evaluations.relaxations_json must be sorted and unique')
    WHERE EXISTS (SELECT 1 FROM json_each(NEW.relaxations_json) a
                  JOIN json_each(NEW.relaxations_json) b ON b.key = a.key + 1
                  WHERE NOT (a.value < b.value));
END;

CREATE TRIGGER IF NOT EXISTS trg_forward_gate_evaluations_relaxations_immutable
BEFORE UPDATE OF relaxations_json ON forward_gate_evaluations
BEGIN SELECT RAISE(ABORT, 'forward_gate_evaluations.relaxations_json is immutable'); END;
```

- The four statements are module constants executed in the order above as
  `RELAXATION_STATEMENTS`; each constant is the statement text above without its trailing `;`
  (the v48 constants' form), and each is idempotent. Lines over 100 columns carry
  `# noqa: E501`, as the v48 fragment does.
- `migrate()` (`algua/registry/db/migrate.py`) runs, after the v48 tick-link block and before the
  `user_version` stamp:
  ```python
  for table in RELAXATION_TABLES:
      _add_missing_columns(conn, table, RELAXATIONS_COLUMN)
  classify_unrecorded_gate_rows(conn)          # §3; one-time
  for statement in RELAXATION_STATEMENTS:
      conn.execute(statement)
  ```
  Hard orderings: the ALTER precedes the classification and the triggers; the classification
  precedes the immutability triggers. The module docstring of `migrate.py` names them.
- `json1` is built into SQLite since 3.38 and present on the runtime's 3.45.1 (rehearsed).
  `ALTER TABLE ... ADD COLUMN` with this CHECK is accepted and checked against existing rows (all
  NULL, so it passes) on SQLite ≥ 3.37.
- `BEFORE UPDATE OF relaxations_json` fires whenever the column is named in a `SET`, even with an
  unchanged value. No production statement names it after insert: the only gate-table UPDATEs are
  the `consumed` flips (`store/base.py:89`, `:106`) and the FDR relabels (`db/gate.py:135`,
  `:171`), which stay legal (rehearsed).
- No delete trigger: no code deletes from either table, a research row anchoring a deployment is
  already undeletable by its foreign key, and a database writer is outside the accident threat
  model (Story 1.3d precedent).
- v49 is forward-only. v48 code on a v49 registry cannot write a gate row (the recorded trigger
  refuses it), so research and paper promotion fail closed. The rollback runbook (in
  `deploy/systemd/README.md`, next to the v48 note): stop the timers, drop the two
  `_relaxations_recorded` triggers, run v48 code; rows it writes stay NULL, which the v49 predicate
  reads as `unrecorded` after the roll-forward.
- Schema fingerprint (`tests/test_registry_db.py:12-13`): object count 137 → 141 (four triggers);
  the digest changes. `test_schema_version_is_current` asserts 49.

## 3. One-time classification of pre-existing rows (CAP-3)

`classify_unrecorded_gate_rows(conn)` lives in `algua/registry/db/relaxations.py`. For each table:

1. If `trg_<table>_relaxations_immutable` exists in `sqlite_master`, skip the table (the one-time
   guard).
2. Otherwise read `SELECT * FROM <table> WHERE relaxations_json IS NULL ORDER BY id`. If no row,
   skip. Only then import, lazily, `algua.registry.relaxations`, `GateCriteria`,
   `ForwardGateCriteria`, `FORWARD_SHARPE_CONFIDENCE` and `Actor` (so an ordinary
   `registry_conn()` never imports the research stack).
3. Derive each row's set (rules below). A `KeyError`, `TypeError`, `ValueError` or
   `OverflowError` while deriving means the row stays NULL. Otherwise execute
   `UPDATE <table> SET relaxations_json=? WHERE id=? AND relaxations_json IS NULL` with
   `encode_relaxations(tokens)`. No other column is touched.

Numbers below are "a JSON number that is not a bool and not null"; `decision_json` is parsed with
Python's `json.loads` (which accepts the `NaN`/`Infinity` literals older rows may contain).
"Check T" means the one element of `decision_json["checks"]` whose `name` is T; zero or several
such elements leave the row NULL.

**Research row.** Stays NULL unless `actor = 'agent'`. Then:
- `demo`: `data_source = 'StoreBackedProvider'` → `False`; `'SyntheticProvider'` → `True`; any
  other value → NULL (`select_provider`, `algua/evaluation/inputs.py:31-39`, yields only these
  two; anything else predates the reproducible-source wall).
- `n_combos`: `breadth_provenance = 'measured'` → `None`; `'declared'` → `own_lifetime_combos`;
  other → NULL.
- `allow_non_pit = bool(pit_override)`; `allow_holdout_reuse`, `assume_terminal_last_close` are
  `False` and `new_family` is `None` (each refused for an agent: `promotion.py:37-54`,
  `promote_run.py:132-137`, ignored: `family_assignment.py`).
- `criteria = GateCriteria(min_holdout_sharpe=decision["base_min_holdout_sharpe"],
  min_holdout_return=<check holdout_return>.threshold,
  min_pct_positive_windows=<check pct_positive_windows>.threshold,
  min_window_sharpe=<check min_window_sharpe>.threshold,
  min_holdout_observations=row.min_holdout_observations)`. A `null` threshold (the evaluator nulls
  a non-finite threshold) leaves the row NULL: `+inf` and `NaN` are indistinguishable.
- `tokens = research_relaxations(actor=Actor.AGENT, demo=..., n_combos=...,
  allow_holdout_reuse=False, allow_non_pit=..., assume_terminal_last_close=False,
  new_family=None, criteria=criteria)`.

**Forward row.** Stays NULL unless `actor = 'agent'` and `decision_json` contains a check
`realized_sharpe_lcb` (the significance wall exists on the row: introduced with
`forward_sharpe_confidence` in #432, whose agent guard refused a looser confidence from the start).
Then `criteria = ForwardGateCriteria(min_forward_observations, degradation_factor, sharpe_floor,
min_forward_vol, max_forward_drawdown, max_staleness_sessions` from the row's columns,
`min_session_coverage=<check session_coverage>.threshold`,
`forward_sharpe_confidence=FORWARD_SHARPE_CONFIDENCE)` and `tokens = forward_relaxations(criteria)`.
A human forward row stays NULL: its confidence is recorded nowhere on a passing row.

**Concurrency.** Two processes migrating at once are safe: the second's UPDATEs carry
`relaxations_json IS NULL`, so after the first commits they match no row and the immutability
trigger never fires (a zero-row UPDATE fires no row trigger; rehearsed). The worst case is the
existing retryable `db_unavailable`.

**Rehearsal (2026-10-05, prototype of this section against a backup of
`file:/home/liornisimov/Projects/algua/data/algua.db?mode=ro`, user_version 48).**

| table | rows | → `[]` | → other set | stay NULL (`unrecorded`) |
|---|---|---|---|---|
| `gate_evaluations` | 21 (all `actor='agent'`, `StoreBackedProvider`, `measured`, `pit_override=0`, default thresholds) | 21 | 0 | 0 |
| `forward_gate_evaluations` | 0 | 0 | 0 | 0 |

A second run was a no-op; every other column of every row was byte-identical before and after;
an UPDATE of the column, an INSERT without it, `[ ]`, `["zzz"]`, `[1]`, an unsorted pair and a
duplicate were each refused with the trigger messages above; `[]` and a canonical relaxed set were
accepted; the existing `migrate()` ran cleanly over the v49 objects. A synthetic v48 registry
(built by today's `migrate()`) classified: agent default → `[]`; agent `min_holdout_sharpe` 0.3 →
`["threshold:min_holdout_sharpe"]`; agent stricter 0.9/0.7 → `[]`; agent `SyntheticProvider` →
`["demo_data"]`; agent `NaN` base → `["threshold:min_holdout_sharpe"]`; agent `declared` +
`pit_override` → `["allow_non_pit","declared_breadth"]`; agent looser window and return →
`["threshold:min_holdout_return","threshold:min_window_sharpe"]`; human, `{}` decision, null
threshold, missing check, unknown provider → NULL; forward agent with LCB check (default or
stricter) → `[]`; forward agent without LCB, forward human, null coverage → NULL. The implementer
re-runs this rehearsal with the real migration on a fresh backup at implementation time and records
the counts in the story (production gains rows while the research timers run; every agent
`StoreBackedProvider` row at default thresholds must read `[]`).

## 4. Recording seam (CAP-2)

Research, one production writer chain: `research promote` (`research_cmd.py:110`),
`research run-all` (`research_batch_cmd.py:63`) and the merge-back seam (`paper_cmd.py:552`) all
call `promote_task`.

- `promote_task` (`promote_run.py`), immediately after `authenticate_actor` returns
  (`:195-216`), computes
  `relaxations = research_relaxations(actor=actor_enum, demo=demo, n_combos=n_combos,
  allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit,
  assume_terminal_last_close=assume_terminal_last_close, new_family=new_family,
  criteria=criteria)` and passes `relaxations=relaxations` to `run_gate` (`:291`). The
  `run_context` dict (`:199-215`) is not touched.
- `run_gate` (`promotion.py:266`) gains the required keyword-only parameter
  `relaxations: tuple[str, ...]` (after `reason_suffix`) and adds
  `"relaxations_json": encode_relaxations(relaxations)` to `gate_row` (`:416-450`). Like
  `actor`, `criteria` and `allow_non_pit`, it trusts its one production caller.
- `record_gate_with_fdr_and_maybe_promote` (`store/gate.py:289`) appends `relaxations_json` to
  its INSERT (`:386`) as `gate_row["relaxations_json"]` (a missing key raises before the INSERT).
- `record_gate_evaluation` (`store/gate.py:51`; production callers: none, only tests and
  `scripts/seed_runs_dev.py:426`) gains the required keyword-only parameter
  `relaxations_json: str` (after `universe_name`) and appends it to its INSERT (`:85`).
- The `GateLedger` Protocol (`repository.py:562`) gains the same parameter.

Forward, one INSERT behind both record paths:

- `run_forward_gate` (`forward_promotion.py:190`) adds
  `"relaxations_json": encode_relaxations(forward_relaxations(criteria))` to `gate_row`
  (`:221-244`), computed from the very `criteria` it evaluates; `paper promote`
  (`paper_cmd.py:1247`) is unchanged.
- `record_forward_gate_evaluation` (`store/forward_gate.py:18`) and
  `_insert_forward_gate_row_locked` (`:73`) gain the required keyword-only parameter
  `relaxations_json: str` (after `decision_json`), appended to the INSERT (`:108`);
  `record_forward_pass_and_promote` (`:129`) forwards it through `**gate_row`. The
  `ForwardGateLedger` Protocol gains the parameter.

The store layer imports nothing from the vocabulary; the triggers validate what it writes.
`research promote` and `paper promote` JSON output is unchanged.

## 5. The qualification predicate (CAP-4)

New module `algua/registry/live_qualification.py`.

```python
CertificateVerifier = Callable[[StrategyRepository, str, int, ArtifactIdentity], dict[str, Any]]

@dataclass(frozen=True)
class LiveQualification:
    deployment_id: int | None
    research_gate_id: int | None
    research: tuple[str, ...]      # () unrelaxed; ("unrecorded",) when unknowable
    certificate_id: int | None
    certificate: tuple[str, ...]
    @property
    def relaxations(self) -> tuple[str, ...]:      # sorted union; () means qualified
        ...

class LiveQualificationRelaxed(TransitionError):
    qualification: LiveQualification

def live_qualification_relaxations(
    conn: sqlite3.Connection, *, strategy_id: int, certificate_id: object,
) -> LiveQualification: ...

def verify_live_qualification(
    repo: StrategyRepository, name: str, strategy_id: int, identity: ArtifactIdentity,
    certificate_verifier: CertificateVerifier,
) -> dict[str, Any]: ...
```

`live_qualification_relaxations` (pure reads, no writes):

1. `cid = certificate_id` if it is an `int` and not a `bool`, else `None`.
2. `deployment = SqliteStrategyRepository(conn).active_deployment(strategy_id)`. If `None`:
   return `LiveQualification(None, None, ("unrecorded",), cid, ("unrecorded",))`. This is what
   refuses the legacy cohort, which has no deployment (`live_certificate.py:91-95` keeps selecting
   its certificate but can no longer authorize go-live).
3. Research: `SELECT relaxations_json FROM gate_evaluations WHERE id=? AND strategy_id=?` with
   `deployment.research_gate_id`. Missing row, NULL, or `decode_relaxations(...,
   RESEARCH_VOCABULARY) is None` → `("unrecorded",)`; else the decoded tuple.
4. Certificate: if `cid is None` → `("unrecorded",)`. Else
   `SELECT relaxations_json FROM forward_gate_evaluations WHERE id=? AND strategy_id=? AND deployment_id=?`
   with `(cid, strategy_id, deployment.id)`; missing row, NULL, or undecodable with
   `FORWARD_VOCABULARY` → `("unrecorded",)`; else the decoded tuple.
5. Return `LiveQualification(deployment.id, deployment.research_gate_id, research, cid, certificate)`.

`verify_live_qualification`:

1. `certificate = certificate_verifier(repo, name, strategy_id, identity)` (it raises its own
   `TransitionError`s exactly as today).
2. `conn = getattr(repo, "connection", None)`; `None` raises
   `TransitionError("live qualification needs a sqlite-backed repository")`.
3. `q = live_qualification_relaxations(conn, strategy_id=strategy_id,
   certificate_id=certificate.get("id") if isinstance(certificate, dict) else None)`.
4. If `q.relaxations`: raise `LiveQualificationRelaxed(q)`.
5. Return `{**certificate, "deployment_id": q.deployment_id, "research_gate_id":
   q.research_gate_id, "research_relaxations": [], "certificate_relaxations": []}` (the lists are
   `list(q.research)` / `list(q.certificate)`, necessarily empty here).

`LiveQualificationRelaxed` message (exact):
`go-live refused: live qualification needs every gate at its protected default; research gate
{R} -> {RT}; forward certificate {C} -> {CT}` where `R`/`C` are the ids or `none`, and `RT`/`CT`
are the comma-space-joined tokens or `unrelaxed`. Example: `... research gate 12 ->
declared_breadth, threshold:min_holdout_sharpe; forward certificate 31 -> unrelaxed`.

Code and envelope: `algua/cli/errors.py::_registry` gains
`(LiveQualificationRelaxed, "live_qualification_relaxed")` immediately before
`(TransitionError, "wrong_stage")`; the code is not in `RETRYABLE_CODES`. The envelope keeps its
four keys. `docs/contracts/cli-error-envelope.md` gains the row
`| LiveQualificationRelaxed | live_qualification_relaxed |` above the `TransitionError` row and
one sentence: go-live refused because the deployment's research gate or forward certificate was
recorded with a relaxation or has no recorded set; not retryable; re-qualify at protected
defaults (a fresh agent `research promote` and paper deployment for a research relaxation, a fresh
default `paper promote` for a certificate relaxation).

Call sites, the only two:

- **Issuance** (`registry_cmd.py:216-217`): replace the verifier call with
  `certificate = live_qualification.verify_live_qualification(` /
  `    repo, name, rec.id, identity, transitions._default_forward_certificate_verifier())`
  (two lines; `from algua.registry import live_gate, live_qualification, transitions` on line
  22). It runs before `live_gate.issue_challenge` (`:218`), so a refusal writes no
  `live_challenges` row. The emitted `forward_certificate` is the returned dict, so the signer
  sees `deployment_id`, `research_gate_id` and both empty sets. The challenge bytes are unchanged.
- **Completion** (`transitions.py:142-143` in `_validate_live_gate`): replace the verifier call
  with `verify_live_qualification(repo, name, strategy_id, identity,
  forward_certificate_verifier or _default_forward_certificate_verifier())`, imported lazily
  inside `_validate_live_gate` (the module's existing pattern). It runs after the actor check,
  the frozen refusal and the identity computation and before the approval verifier, so before
  `ssh-keygen` (`live_gate.verify_pending`, reached through the CLI closure at
  `registry_cmd.py:235-244`) and before challenge consumption inside `apply_transition`. An
  injected `forward_certificate_verifier` replaces only step 1.

The predicate reads rows that cannot change (the column is trigger-immutable; a deployment's
`research_gate_id` is trigger-immutable, `db/deployment.py:47-55`); the active deployment changes
only with a stage change, which the go-live stage CAS detects.

## 6. No raw way in (CAP-6, AC10, #682)

In `transitions.py`, add next to `_REVOKE_ON_EXIT`:

```python
_GATE_COMMAND_EDGES: dict[tuple[Stage, Stage], str] = {
    (Stage.BACKTESTED, Stage.CANDIDATE): "research promote",
    (Stage.PAPER, Stage.FORWARD_TESTED): "paper promote",
}
```

and in `transition_strategy`, directly after the `candidate -> paper` intake refusal (`:49-51`)
and before `validate_transition`:

```python
command = _GATE_COMMAND_EDGES.get((rec.stage, target))
if command is not None:
    raise TransitionError(
        f"{rec.stage.value} -> {target.value} is reachable only through `algua {command}`; "
        "no actor may take it as a raw transition")
```

- Exception `TransitionError`, code `wrong_stage`, not retryable: the same class and code as the
  existing intake refusal and the 1.3d frozen forward refusal. Messages:
  ``backtested -> candidate is reachable only through `algua research promote`; no actor may take it as a raw transition``
  and ``paper -> forward_tested is reachable only through `algua paper promote`; no actor may take it as a raw transition``.
- Applies to every actor (agent, human, system) and every caller of `transition_strategy` (the CLI
  `registry transition`, `registry_cmd.py:249`, and any programmatic caller).
- Deleted as unreachable: the `backtested -> candidate` token branch (`:75-84`), the
  `paper -> forward_tested` branch (`:85-93`), `_validate_shortlist_gate` (`:160-172`),
  `_validate_forward_gate` (`:175-195`), and the forward-edge half of
  `refuse_frozen_deployment` (`:198-218`): it loses its `target` parameter and only raises
  `FrozenLiveUnsupported`; its two callers (`registry_cmd.py:208`, `transitions.py:138`) drop
  the argument. `apply_transition` is called without `consume_gate_id`/`consume_forward_gate_id`.
- Deleted because their only production callers were those helpers:
  `find_consumable_gate_evaluation` (`store/gate.py:104-123`, Protocol `repository.py:599-610`)
  and `find_consumable_forward_gate_evaluation` (`store/forward_gate.py:170-196`, Protocol
  `repository.py:744-758`). The consume parameters of `apply_transition` stay (non-goal).
- Unchanged: the gate commands' own stage moves (`store/gate.py:454`,
  `store/forward_gate.py:163`), every back-step (`paper -> candidate`, `candidate -> backtested`,
  `backtested -> idea`, `forward_tested -> paper`, `live -> paper`, `live -> dormant`,
  `dormant -> paper`), `idea -> backtested`, `paper -> dormant`, every `-> retired`, the
  `candidate -> paper` intake refusal and the go-live wall.

## 7. Entry-point inventory

Every path that writes a gate row, reaches go-live, or enters `candidate` / `forward_tested`.

| # | path | file:line (baseline) | covered by |
|---|---|---|---|
| W1 | research row INSERT, atomic record-and-promote | `store/gate.py:386` in `record_gate_with_fdr_and_maybe_promote` `:289` ← `promotion.run_gate` `:476` ← `promote_task` `promote_run.py:291` | §4 records `research_relaxations`; recorded trigger |
| W1a | `research promote` CLI | `research_cmd.py:110` | via W1 |
| W1b | `research run-all` batch worker | `research_batch_cmd.py:63` (keys `:76-81`) | via W1; AC2 table test covers its keys |
| W1c | merge-back strict-agent promote seam | `paper_cmd.py:552` (drainer: `.opencode/scripts/drain-mergeback-queue.sh:100` may add `--demo`) | via W1; `--demo` records `demo_data` |
| W2 | research row INSERT, plain writer | `store/gate.py:85` in `record_gate_evaluation` `:51`; callers: tests, `scripts/seed_runs_dev.py:426` | required `relaxations_json` parameter; recorded trigger |
| W3 | forward row INSERT, single statement | `store/forward_gate.py:108` in `_insert_forward_gate_row_locked` `:73` | recorded trigger |
| W3a | failing row / refresh at `forward_tested` | `record_forward_gate_evaluation` `store/forward_gate.py:18` ← `forward_promotion.py:262` | §4 `forward_relaxations(criteria)` |
| W3b | passing row + `paper -> forward_tested` | `record_forward_pass_and_promote` `store/forward_gate.py:129` ← `forward_promotion.py:258` | §4 |
| W3c | `paper promote` CLI | `paper_cmd.py:1247` (only caller of `run_forward_gate`) | via W3a/W3b |
| W4 | v49 classification UPDATE (both tables) | `db/relaxations.py` (new) via `migrate()` | §3; runs before the immutability triggers exist |
| W5 | raw SQL INSERT (tests, a DB writer) | tests listed in §11 | recorded trigger refuses a missing or malformed set |
| L1 | go-live challenge issuance | `registry_cmd.py:203-228`: actor `:204`, stage `:207`, frozen `:208`, identity `:215`, verifier `:216-217`, `live_gate.issue_challenge` `:218` (`live_gate.py:45`) | §5 `verify_live_qualification` replaces `:216-217`, before `:218` |
| L2 | go-live completion (CLI) | `registry_cmd.py:230-255` → `transition_strategy` `:249` → `_validate_live_gate` `transitions.py:116` (verifier `:142-143`, approval `:144-153` → `_verify` `registry_cmd.py:235-244` → `live_gate.verify_pending` `live_gate.py:132`) → `apply_transition` (consume + `live_authorizations`) | §5 at `:142-143`, before the approval verifier |
| L3 | programmatic `transition_strategy(..., LIVE, HUMAN)`, default or injected verifiers, `has_valid_approval` (`approvals.py:211`) | `transitions.py:36`, `:65-74` | same `_validate_live_gate`; an injected verifier replaces only the certificate step |
| L4 | `verify_forward_certificate` direct caller | only `transitions._default_forward_certificate_verifier` `transitions.py:268` | wrapped by §5 at both call sites |
| L5 | store primitive `apply_transition(rec, LIVE, HUMAN, live_authorization=...)` | `store/crud.py:238`; production caller: only `transitions.py:97` | structural test (§10) pins the single production caller |
| L6 | legacy-cohort certificate branch | `live_certificate.py:91-95` | no active deployment → predicate `unrecorded` |
| L7 | trade-time `verify_live_authorization`, `live run-all` | `live_gate.py:166` | not a go-live path (Story 2.3) |
| C1 | `research promote` → `candidate` | `store/gate.py:454` (inside W1's transaction) | the only way in |
| C2 | raw `backtested -> candidate` | `transitions.py:75-84` today | §6 refused for every actor |
| C3 | raw `paper -> candidate` back-step | `transitions.py` generic path | unchanged (allowed; revokes allocation) |
| C4 | store primitive `apply_transition(..., CANDIDATE)` | `store/crud.py:238`, production caller only `transitions.py:97` | structural test |
| C5 | `shortlisted -> candidate` stage rename (one-time migration) | `db/core.py:75` | not a transition; unchanged |
| F1 | `paper promote` → `forward_tested` | `store/forward_gate.py:163` (inside W3b's transaction) | the only way in |
| F2 | raw `paper -> forward_tested` | `transitions.py:85-93` today | §6 refused for every actor |
| F3 | store primitive `apply_transition(..., FORWARD_TESTED)` | as C4 | structural test |
| X | other stage writers (`idea -> backtested`, intake) | `mergeback_intake.py:233-249`, `evaluation/backtest_run.py:105`, `store/deployment.py:208` | not candidate/forward_tested/live; unchanged |

`operator/mergeback.py:532` `run_gate()` is the merge-back quality gate, not `promotion.run_gate`.

## 8. Exploration unchanged (CAP-5, AC8)

Claims, each with its proof:

1. **Challenge bytes.** `build_actor_challenge` and `canonical_run_context` are not edited and the
   `run_context` dicts (`promote_run.py:199-215`, `paper_cmd.py:1235-1241`) are not touched.
   New golden tests pin, at baseline values:
   - `research promote NAME --demo --n-combos 7 --allow-holdout-reuse --allow-non-pit
     --assume-terminal-last-close --new-family fam-x --min-holdout-sharpe 0.3 --actor human` on a
     strategy at `backtested` prints a challenge whose `run_context=` line is exactly
     `{"allow_holdout_reuse":true,"allow_non_pit":true,"assume_terminal_last_close":true,"demo":true,"end":"2023-12-31","holdout_frac":0.2,"min_holdout_return":0.0,"min_holdout_sharpe":0.3,"min_pct_positive":0.6,"min_window_sharpe":0.0,"n_combos":7,"new_family":"fam-x","start":"2023-01-01","windows":4}`
     (full payload asserted line by line, as `test_frozen_promotion_refusal.py:413-435` does);
   - `paper promote NAME --degradation-factor 0.4 --min-observations 40 --max-staleness 9 --actor
     human` on a working-tree deployment prints `run_context=`
     `{"degradation_factor":0.4,"forward_sharpe_confidence":0.95,"max_drawdown":0.25,"max_staleness":9,"min_coverage":0.9,"min_observations":40,"min_vol":0.02,"sharpe_floor":0.3}`;
   - the existing default-threshold golden (`test_frozen_promotion_refusal.py:407-410`) stays.
   The go-live signed payload (`live_gate.py:34-41`) is not edited.
2. **Verdicts, tokens and stage moves.** The set is computed from inputs only and is an input to
   nothing but the INSERT. Proofs: (a) an inertness test runs `run_gate` twice on identical
   fixtures in two fresh registries, with `relaxations=()` and with
   `("agent_walls_waived","declared_breadth")`, and requires every column except `id`,
   `created_at` and `relaxations_json`, the `runs` row (except its ids and timestamps) and the stage
   history to be equal; (b) the
   same for `run_forward_gate` with `forward_relaxations` monkeypatched to return
   `("threshold:sharpe_floor",)`; (c) every pre-existing test of a relaxed human research or paper
   promotion passes with assertion changes limited to added relaxation assertions.
3. **Anchoring.** A signed human research row with `allow_non_pit` (recorded
   `["agent_walls_waived","allow_non_pit"]`) is admitted by `intake_candidate_to_paper`
   (`store/deployment.py:124`) exactly as today; a relaxed human `paper promote` moves
   `paper -> forward_tested` and a second one refreshes at `forward_tested`; both rows record
   their sets.

## 9. Placement, protection, size pins, import boundaries

| module | change | size (baseline → bound) |
|---|---|---|
| `algua/registry/relaxations.py` | new (§1, plus moved `guard_agent_relaxations`) | new, < 300 |
| `algua/registry/live_qualification.py` | new (§5) | new, < 300 |
| `algua/registry/db/relaxations.py` | new (§2 DDL, §3 classifier) | new, < 300 |
| `algua/registry/gate_fail_capture.py` | new: `capture_gate_fail_experience` moved verbatim from `promote_run.py:54-100` with its imports (`write_experience_note`, `now_iso`, the `negative_results` names, `get_settings`) | new, < 300 |
| `algua/registry/promote_run.py` | carve out the capture; add §4 (import, 4-line call, `relaxations=` kwarg) | 348 (pin 348) → about 301; lower the pin to the exact result |
| `algua/registry/promotion.py` | carve out `guard_agent_relaxations`; add one import line, one parameter, one `gate_row` entry | 542 (pin 542) → about 526; lower the pin |
| `algua/registry/store/gate.py` | delete `find_consumable_gate_evaluation`; add the parameter and the two INSERT columns | 594 (pin 594) → about 577; lower the pin |
| `algua/registry/repository.py` | delete the two `find_consumable_*` Protocol methods; add two parameters | 965 (pin 965) → about 938; lower the pin |
| `algua/registry/store/forward_gate.py` | delete `find_consumable_forward_gate_evaluation`; add the parameter and column | 217 → about 192 |
| `algua/registry/forward_promotion.py` | guard reuses the table; `gate_row` entry; one import | 280 → about 276 (stays under 300) |
| `algua/registry/transitions.py` | §5 completion call, §6 | 277 → about 235 |
| `algua/registry/db/migrate.py`, `db/constants.py` | v49 block and version | small |
| `algua/cli/registry_cmd.py` | §5 issuance (line count unchanged; pin 446, size 444) | unchanged |
| `algua/cli/errors.py` | one import, one registry entry | 181 → 183 |
| `algua/research/forward_gates.py` (pin 392), `research/gates.py` (544), `cli/paper_cmd.py` (1409), `forward_evidence.py`, `live_certificate.py` | not edited | - |

- Every pin of a module that shrinks is lowered to its exact post-change line count in the same
  commit, or deleted if the module falls below 300 lines (`tests/test_module_size_ratchet.py`); no
  pin is raised and no new module reaches 300 lines.
- CODEOWNERS gains `/algua/registry/relaxations.py`, `/algua/registry/live_qualification.py`,
  `/algua/registry/db/relaxations.py` (also covered by the `db/` rule, named for the evidence
  set), `/algua/registry/gate_fail_capture.py` (it receives the registry connection inside
  `promote_task`) and `/algua/cli/registry_cmd.py` (the go-live issuance check and completion
  closure; precedent: `paper_cmd.py`, `research_cmd.py`). All five join
  `INTEGRITY_CRITICAL_MODULES` (`tests/test_repo_hygiene.py:237`).
- Import boundaries: no `algua/contracts` change. The new modules import `algua.research`
  (allowed: "research never imports registry" is the only research contract), `algua.contracts`,
  `algua.registry.store` / `repository`; none imports `algua.live` ("registry stays off the live
  lane") or `algua.cli`. `db/relaxations.py` imports the vocabulary lazily (§3). `uv run
  lint-imports` passes with no new exemption.

## 10. Tests and mutation checks

New files: `tests/test_relaxations.py`, `tests/test_relaxation_schema.py`,
`tests/test_live_qualification.py`, `tests/test_raw_gate_edges.py` (replaces
`tests/test_shortlist_gate.py`), `tests/test_exploration_unchanged.py`, and the helper
`tests/_live_qualification_helpers.py`:

- `qualified_live_world(conn, name, *, identity, research_relaxations="[]",
  certificate_relaxations="[]") -> (strategy_id, deployment_id, research_gate_id, certificate_id)`:
  registers the strategy, writes the research row with `record_gate_evaluation`, a working-tree
  `deployment_artifacts` + `strategy_deployments` pair by raw SQL (the
  `test_frozen_promotion_refusal.py:125-150` pattern), a passing deployment-bound certificate with
  `record_forward_gate_evaluation(consumable=False, deployment_id=...)`, and sets
  `stage='forward_tested'` directly.
- `insert_unrecorded(conn, table, **cols)`: drops `trg_<table>_relaxations_recorded`, inserts
  with NULL, re-creates the trigger from the `db/relaxations.py` constant in a `finally` (the
  `force_legacy_strategy` precedent).
- `seed_stage(repo, name, to)`: test scaffolding through the store primitive
  (`repo.apply_transition(repo.get(name), to, Actor.HUMAN, reason="test setup")`), the
  replacement for raw `--to candidate` / `--to forward_tested` set-up steps.

Coverage required:

- Vocabulary: each flag token on and off; each threshold at default, stricter, looser and `NaN`,
  for both directions; `+inf`; output sorted and unique; `agent_walls_waived` and `new_family`
  only for a human; `encode`/`decode` round trip and every `decode` rejection.
- AC2: the click parameters of `research promote` and `paper promote` (via
  `typer.main.get_command`) and `inspect.signature(promote_task)` equal the table keys exactly
  (union for research); `research_batch_cmd._ALLOWED_KEYS["promote"]` is a subset; every non-None
  value is in the vocabulary; every vocabulary token except
  `threshold:min_holdout_observations` is reachable from a table entry.
- Tie tests: the DDL token lists parsed from the trigger text equal `RESEARCH_VOCABULARY` /
  `FORWARD_VOCABULARY`; `RESEARCH_THRESHOLDS` equals `dataclasses.fields(GateCriteria)` and every
  `GATE_SPECS` op is `>=` or `>`; the two forward lists partition
  `dataclasses.fields(ForwardGateCriteria)`; §2's DDL is pinned to the module verbatim.
- Recording: one row per writer (W1 pass and fail for agent and human, W1b, W1c, W2, W3a fail,
  W3a refresh, W3b) carries the expected set; an agent row that tightens a threshold records `[]`.
- Schema: INSERT without the column, with each malformed form and out-of-vocabulary token, and any
  UPDATE naming the column are refused with the §2 messages; `consumed` and FDR updates still work.
- Migration: the §3 synthetic cases; idempotence; other columns byte-identical; a second
  `migrate()` with a still-NULL classifiable row (inserted via `insert_unrecorded`) leaves it NULL
  and does not abort; a v48 registry migrates to v49 (column, four triggers, version).
- Predicate: no deployment; research NULL; research relaxed; certificate id missing, not an int,
  a `bool`, another strategy's, another deployment's, NULL, relaxed; both clean.
- Ceremony: issuance refused writes no `live_challenges` row and emits the code; completion
  refused leaves the pending challenge unconsumed and never calls `ssh-keygen`
  (`live_gate.verify_signature` armed to raise); a valid human signature over a fresh challenge
  is still refused when either row is relaxed or unrecorded (the CLI will not issue a challenge
  for such a world, so the test creates the pending challenge with `live_gate.issue_challenge`
  directly, or records a newer relaxed certificate after a clean issuance); an injected verifier (CLI
  monkeypatch seam and `forward_certificate_verifier=`) cannot skip the predicate; the issued
  challenge shows `research_gate_id`, `deployment_id` and both empty sets; the envelope is
  `{"ok": false, "error": <message>, "code": "live_qualification_relaxed", "retryable": false}`.
- AC10: both edges × agent, human, system refused with the §6 message and no stage row; the
  `paper -> candidate` back-step and every other raw edge unchanged; a structural test scans
  `algua/` and requires `transitions.py` to be the only production caller of `.apply_transition(`
  and `{store/gate.py, store/forward_gate.py, store/deployment.py, store/crud.py}` the only callers
  of `._apply_transition_locked(`.
- AC8: §8.

Mutation checks (break, see a named test fail, restore byte for byte): each flag predicate in
`research_relaxations`; `>=` → `>` (the at-default case); moving one forward field to the other
direction list; each of the four statements of each recorded trigger; each immutability trigger;
the §3 one-time guard; the `relaxations_json IS NULL` clause of the classification UPDATE; the
human / unknown-provider / missing-LCB rules; each predicate branch (steps 1-4); the
`q.relaxations` refusal; removing the issuance call; removing the completion call; moving the
predicate inside the default verifier; each `_GATE_COMMAND_EDGES` entry; the `errors.py` registry
entry.

## 11. Known churn

Tests (fixture churn only, unless named): every `record_gate_evaluation` caller passes
`relaxations_json` (25 sites, incl. `tests/_gate_row_helpers.py:23`, `tests/_frozen_paper_world.py:280`);
`record_forward_gate_evaluation` / `record_forward_pass_and_promote` callers and their `gate_row`
dicts (`test_forward_certificate.py`, `test_registry_store.py` ×5, `test_shortlist_gate.py`); `run_gate` callers pass `relaxations=()`
(`test_promotion.py` ×11, `research/test_dsr_dispersion_floor.py` ×6); raw INSERTs into the gate
tables that run after `migrate()` add the column (`test_forward_certificate.py:118`,
`test_registry_store.py:1066,1101`, `test_forward_promotion.py:658,728`, `test_cli_paper.py:1478`,
`test_cli_governance.py:83,162`, `_frozen_evidence_helpers.py:71`,
`registry/test_novel_family_seed_524.py:491`, `test_cli_registry_gates.py:57,79`,
`test_cli_merge_back.py:115`, `registry/test_gate_attempt_token.py:33`, and in
`test_registry_db.py` / `test_db_migrations.py` only where the INSERT follows a v49 `migrate()`);
the schema fingerprint and version in `test_registry_db.py`; `guard_agent_relaxations` import in
`test_promotion.py:13`.

Raw forward edges (replace with `seed_stage` or the real gate command):
`test_cli_registry.py:78,88`; `test_cli_live.py:97,865,1210,1433`; `test_cli_paper.py:78,760,1383,1639`;
`test_forward_certificate.py:507-516`; `test_observability_wiring.py:34`;
`test_paper_run_all.py:93`; `test_paper_venue_reconcile.py:100`; `test_e2e_lifecycle.py:141,169`;
`test_registry_approvals.py:47,53`; `test_registry_store.py:126,128`;
`test_forward_promotion.py:976`; `test_frozen_paper_cli.py:557` and
`test_frozen_promotion_refusal.py:645,668,771` (the expected refusal becomes §6's message);
`test_shortlist_gate.py` (deleted; its back-step test moves to `test_raw_gate_edges.py`).

Deleted finders: `test_registry_store.py:324-353,701` and `test_forward_promotion.py:935,957,997`
read `consumed`/`actor` by SQL instead; the finder-semantics tests are deleted.

Go-live tests that relied on the legacy cohort or a stub certificate id now use
`qualified_live_world` and stub the verifier with its certificate id: `test_cli_registry.py`
(signed ceremony, no-pending-challenge, preallocation, second live strategy),
`test_registry_approvals.py` (`_to_live`), `test_book_exit_revoke.py:300-335`,
`test_registry_store.py` (go-live cases near `:119-142`), `test_forward_certificate.py:398-469`.
Frozen go-live tests (`test_frozen_paper_cli.py:562`, `test_frozen_promotion.py:376`,
`test_frozen_promotion_refusal.py`) keep `frozen_live_unsupported`, which still fires first.

Code and docs: `scripts/seed_runs_dev.py:426`; `CLAUDE.md` (the raw-shortlist sentence at
`:122-123`, the token-gated paragraph at `:203-204`, the go-live bullet); `docs/agent/operating.md`
(live gate steps); `docs/agent/research-lifecycle.md` §6; `docs/architecture.md` (`:33-34` raw
edge, the paper→live wall bullet); `docs/contracts/cli-error-envelope.md` (§5);
`deploy/systemd/README.md` (v49 forward-only note); docstrings naming the deleted finders
(`store/forward_gate.py:143-144`, `store/gate.py:82,399`, `repository.py:593`).
