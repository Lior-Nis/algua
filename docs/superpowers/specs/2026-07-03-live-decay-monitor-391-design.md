# Live performance-decay monitoring + recertification signal (#391)

## Problem

A strategy's forward-test certificate (`forward_gate_evaluations`) is checked exactly ONCE, at the
`forward_tested -> live` transition (`verify_forward_certificate`, forward_promotion.py). Once a
strategy is LIVE, nothing ever re-runs the realized-vs-certified comparison. `live run-all`
re-verifies authorization and reconciles broker positions every cycle but never asks "does this
live strategy's realized return distribution still meet its certified holdout baseline?". A
decayed-edge strategy therefore keeps trading undetected. SR 11-7 requires continuous
outcomes-analysis of a deployed model against its design objectives, not a one-shot promotion gate.

The live data already exists: `tick_snapshots WHERE lane='live'` records per-session equity, and
`forward_gate_evaluations` records the certified `holdout_sharpe` / `realized_sharpe` / `created_at`
for the current identity. No new persistence is needed to compute a decay read.

## Scope (this change)

ADVISORY monitoring ONLY. A new standalone read-only module + its own CLI subcommand that:

1. Reads the newest forward-test certificate (pass-or-fail) for the strategy's CURRENT identity;
   a non-passing newest row invalidates any prior pass (same rule as the live wall) -> baseline
   `None` -> `unknown`.
2. Computes a realized LIVE Sharpe / vol / drawdown / session-coverage from `lane='live'` ticks
   recorded SINCE the certificate `created_at`, over the SAME admissibility filters the forward
   gate uses.
3. Compares realized live Sharpe against the certified decay bar
   `max(DEGRADATION_FACTOR * holdout_sharpe, SHARPE_FLOOR)` — the SAME wall the promotion gate uses,
   imported (not redefined) from `algua.research.forward_gates`.
4. Surfaces a verdict (`ok` / `decay_warn` / `insufficient_data` / `unknown`) plus a
   `recert_needed` flag when the certificate is older than `CERTIFICATE_FRESH_SESSIONS`.

Explicitly OUT of scope (this change gates/persists/transitions NOTHING):

- No auto-halt / kill-switch / demotion. The verdict is a re-audition prompt, not enforcement.
- No schema migration / new table. Decay is computed from existing tick + certificate rows.
- No edits to the CODEOWNERS-protected certificate files (`forward_promotion.py`,
  `forward_gates.py`) or the live order path (`live_cmd.py run-all`, `fleet_health.py`). Read the
  certificate; do not touch it. (The issue's "auto-halt / demote / TTL-enforce" recommendation is a
  protected-path follow-up, deferred.)

## Non-goals / deferred

- Wiring the decay read into `live run-all` (touches protected `live_cmd.py`) — deferred.
- A `live_returns` persisted table + session TTL enforcement (schema + protected promotion) —
  deferred; congested SCHEMA_VERSION.
- Coupling decay to the `monitoring drift` leading indicator (#343). Kept separate to avoid
  re-running a heavy signal-panel eval; the two commands are complementary and reported as such.

## Architecture

### `algua/monitoring/decay.py` (new, PURE — no I/O, no SQL)

A pure evaluator mirroring the shape of `algua/monitoring/drift.py`. It receives already-fetched
inputs and returns a JSON-clean report; the CLI does all the SQL.

```
@dataclass(frozen=True)
class CertifiedBaseline:
    holdout_sharpe: float | None      # design-objective baseline from the certificate
    certified_realized_sharpe: float | None
    created_at: str                   # certificate timestamp (ISO)
    age_sessions: int | None          # sessions from created_at to now (None if uncomputable)

@dataclass
class DecayReport:
    verdict: str                      # ok | decay_warn | insufficient_data | unknown
    recert_needed: bool
    checks: list[dict]                # named, JSON-clean, mirrors forward_gate check rows
    n_live_observations: int
    realized_sharpe: float | None
    realized_vol: float | None
    realized_max_drawdown: float | None
    decay_bar: float | None
    def to_dict(self) -> dict: ...

def decay_report(
    live_returns: pd.Series,          # daily simple returns from admissible lane='live' equity
    session_coverage: float,          # decided sessions / trading sessions in the live window
    n_inadmissible_ticks: int,        # live ticks dropped by the admissibility filters
    baseline: CertifiedBaseline | None,
    *,
    min_observations: int,            # default MIN_FORWARD_OBSERVATIONS (63)
    min_session_coverage: float,      # default MIN_SESSION_COVERAGE (0.9)
    recert_stale_sessions: int,       # default CERTIFICATE_FRESH_SESSIONS (10)
) -> DecayReport: ...
```

Verdict logic (fail-closed; a false "healthy" is forbidden). Evaluated in order — the FIRST
non-`ok` condition wins, so `ok` requires every fail-closed guard to pass:

- `baseline is None` (no certificate, or newest is non-passing) -> `unknown`, recert_needed=True.
- `baseline.holdout_sharpe` is None or non-finite -> `unknown` (no design objective to compare).
- fewer than `min_observations` usable live returns -> `insufficient_data` (never `ok`).
- `session_coverage < min_session_coverage` -> `insufficient_data` (sparse ticking inflates the
  annualized daily Sharpe; same coverage floor the forward gate enforces).
- realized Sharpe non-finite -> `insufficient_data` (a NaN/inf never clears the bar).
- otherwise compute `decay_bar = max(DEGRADATION_FACTOR * holdout_sharpe, SHARPE_FLOOR)` with the
  same finite-guard as the gate (each operand finite BEFORE max, else fail closed); verdict is
  `ok` iff `realized_sharpe >= decay_bar` else `decay_warn`.
- `recert_needed = baseline is None OR age_sessions is None OR age_sessions > recert_stale_sessions`.

`n_inadmissible_ticks` is reported (a non-zero count is a data-hygiene signal) but is advisory —
inadmissible ticks are simply excluded from the return series, exactly as the forward gate excludes
them; they never contribute to a false `ok` because only admissible equity feeds the metrics.

PIT / no look-ahead: `live_returns` are backward-looking realized returns from already-recorded
equity ticks, restricted to ticks AFTER the certificate `created_at` (the window start is the
certification instant, so a certificate refreshed mid-live never mixes pre-refresh returns into the
comparison) and filtered by the gate's admissibility rules (future/malformed `tick_ts`, identity
drift, stale `decision_ts` all dropped). There is no forward label anywhere. The module never reads
the future.

### `algua/cli/monitoring_cmd.py` (append a `decay` subcommand — additive, non-protected)

`monitoring decay <name>` (advisory, exit 0 even on `decay_warn`):

1. `identity = compute_artifact_hashes(name)`; resolve `strategy_id` via the repository.
2. `row = repo.latest_forward_gate_row(strategy_id, *identity)`; if `row is not None and
   row["passed"]`, build `CertifiedBaseline` from its `holdout_sharpe` / `realized_sharpe` /
   `created_at`, computing `age_sessions` against `now` with the same UTC-normalized session
   arithmetic the certificate verifier uses. A missing or non-passing newest row -> baseline None
   (a failed re-eval invalidates a prior pass, same rule as the live wall).
3. Read `lane='live'` ticks for the strategy. Reuse the forward-gate admissibility helpers
   (`_inadmissible_reason`, `_parse_dt` imported read-only from `forward_promotion.py` — NOT
   edited) to drop future/malformed-`tick_ts`, identity-drifted, and stale-`decision_ts` ticks;
   additionally drop ticks whose `tick_ts <= certificate.created_at` (window starts at the
   certification instant). Key the last admissible tick per decision session, build the equity
   series -> `pct_change().dropna()` daily returns (the same construction as
   `assemble_forward_evidence` step 3). Compute session coverage over the admissible window with
   the shared `SessionCalendar`. Count dropped ticks. Metrics via the shared `metrics_from_returns`.
4. `report = decay_report(...)`; emit `ok(report.to_dict() + {strategy, certificate_age_sessions,
   ...})`.

`--min-observations` and `--recert-stale-sessions` are advisory CLI knobs (this gates nothing, so
they are not walls — but they default to the protected constants `MIN_FORWARD_OBSERVATIONS` /
`CERTIFICATE_FRESH_SESSIONS` so the advisory read is calibrated to the real gate). Session coverage
uses `MIN_SESSION_COVERAGE`.

Reusing `_inadmissible_reason` / `_parse_dt` is a READ-ONLY import from the protected
`forward_promotion.py`; the module is imported, never modified, so CODEOWNERS is not triggered.

## Testing

- Pure `decay.py`: ok / decay_warn boundary at the bar; insufficient_data below min-obs; unknown on
  missing/non-finite baseline; NaN realized Sharpe -> insufficient_data (never ok); recert_needed
  on stale/missing certificate; JSON-clean `to_dict` (no NaN/inf leak).
- CLI: end-to-end over a seeded registry DB with live ticks + a passing certificate -> ok; decayed
  live equity -> decay_warn, exit 0; no certificate -> unknown; too-few ticks -> insufficient_data.

## Review focus (Codex GATE-1/GATE-2)

- Is the decay metric PIT (no look-ahead)? — realized returns are backward-looking; cert precedes.
- Does it fail closed / mark `unknown` vs a false `ok`? — every missing/degenerate input -> unknown
  or insufficient_data, never ok.
- Does it accidentally gate/demote? — it must NOT; exit 0 always, no transitions, no writes.
