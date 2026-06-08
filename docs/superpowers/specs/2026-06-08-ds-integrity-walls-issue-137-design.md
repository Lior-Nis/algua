# DS-integrity walls — funnel-level deflation + PIT-by-default + min-sample floor (Issue 137)

**Status:** design — pending sign-off
**Issue:** #137
**Author/operator:** Lior-Nis
**Design review:** GATE 1 (Codex, `codex exec -s read-only`) run on the original 3-tweak design;
findings folded in below (this revised design is the result).

---

## 1. Context & premise

algua has a correctly-placed DS-integrity spine: `t→t+1` anti-look-ahead engine, opt-in
point-in-time (PIT) universe, a **deflated-Sharpe haircut** `√(2·ln N)·√ANN/√T` that raises the
holdout-Sharpe bar by within-strategy search breadth `N` (fails closed `inf` on a zero-length
holdout), single-use holdout burn, walk-forward stability, reproducibility stamps. These live in
CODEOWNERS-protected files so a human must approve any merge that weakens them.

Issue 137 closes three remaining holes — funnel-level (cross-strategy) multiple testing,
survivorship-biased default universe, and the missing minimum-sample floor — **in code**, because
the adversary is the autonomous research agent itself.

**GATE 1 surfaced a fourth, deeper hole that the three walls rest on (verified):** the agent can
skip `research promote` entirely and self-promote via
`registry transition <name> --to shortlisted --actor agent`. `transition_strategy` only enforces a
gate for `Stage.LIVE`; `BACKTESTED → SHORTLISTED` is an ordinary allowed transition. Without
closing this, all three new walls are **advisory**. This spec closes it too (decision: in-scope
for 137).

## 2. Threat model & invariants

**Adversary:** the autonomous agent, driving the same CLI, a multiple-testing machine. "Correct" =
the agent cannot promote an overfit / survivorship-biased / underpowered strategy past the gate by
exploiting an accounting gap, and cannot relax the walls itself, without a human in the loop and an
audited trail.

**Invariants that must NOT weaken (carried from today + added):**
- `N=1` single pre-registered hypothesis → zero haircut (`ln 1 = 0`).
- Degenerate holdout (`T ≤ 0`) → fails closed (`inf` effective bar).
- `evaluate_gate` stays **pure** (no I/O); the math/policy live in CODEOWNERS-protected files.
- A lone hypothesis with no siblings must reduce **exactly** to today's per-strategy behavior (no
  regression).
- **New:** the agent cannot reach `SHORTLISTED` without a fresh, artifact-matched, single-use
  gate-pass record.
- **New (unifying principle):** every gate **relaxation** — declared (vs measured) breadth,
  non-PIT, holdout reuse — requires `--actor human`. The agent only ever sees the strict gate.

## 3. The four walls

### Wall A — Funnel-level (cross-strategy) breadth deflation  *(Gap 1)*

Today the haircut deflates by one strategy's own combos. An agent running N different hypotheses,
each pre-registered as `combos=1`, runs an N-way search with **zero** deflation. Fix: change *what
N is* fed to the (unchanged) haircut.

**Effective funnel breadth (decision: `max`, not additive sum):**
```
N_funnel = max(own_lifetime_combos, windowed_total_combos)
```
- `own_lifetime_combos = total_search_combos(name)` — lifetime, unchanged from today.
- `windowed_total_combos` = Σ `n_combos` over **all** `search_trials` with
  `created_at ≥ now − FUNNEL_WINDOW_DAYS` (includes this strategy's own recent sweeps).
- `FUNNEL_WINDOW_DAYS = 90` — a **protected constant**, not an agent-tunable CLI flag.

**Framing (Codex correction):** this is an **effective funnel-breadth policy**, *not* a literal
"number of independent trials" — mixing horizons has no single coherent error budget, so we do not
claim DSR exactness. The `max` form is the coherent conservative rule: the bar is at least your own
lifetime breadth and at least the recent funnel-wide breadth. `windowed_total` **includes own**, so
there is no double-counting and no free-text-name exclusion subtlety.

**Reduces to today:** a lone hypothesis with no siblings has
`windowed_total ≤ own_lifetime` ⇒ `N_funnel = own_lifetime` ⇒ identical to current behavior.

**Accepted trade-off (documented):** a rolling window lets an agent *wait out* old search effort.
Lifetime-cumulative is unreachable, so the window is the deliberate trade; decay is auditable via
`search_trials.created_at`.

### Wall B — PIT-by-default (survivorship enforcement)  *(Gap 2)*

PIT universe is opt-in; the default static symbol list is survivorship-biased. The gate now
**refuses a non-PIT run as not-promotable**, fail-closed, with a human-only audited override.

**`pit_ok` is computed in the protected policy layer** (it needs the universe timeline, which is
I/O), then passed into the pure `evaluate_gate` as a boolean:
```
pit_ok = (universe was used)  AND  (earliest effective_date ≤ backtest start)   # coverage, not mere presence
```
The **coverage** clause (Codex MEDIUM) closes the "single snapshot back-dated to the end of the
window" hole partway: a universe must at least cover the start of the period. (Full provenance
certification — trusted source, as-of constraints, certification bit — is **deferred** to a
data-lane issue; see §9.)

- If `not pit_ok and not allow_non_pit` → `pit_required` check fails closed (not promotable).
- If `not pit_ok and allow_non_pit` (human only) → passes, stamped `override="non_pit"` (audited).
- `pit_required` is boolean, so `_gate_reason` is generalized to render boolean checks as
  `name=pass|fail`.

### Wall C — Minimum-sample floor  *(Gap 3)*

`GateCriteria` gains `min_holdout_observations` (**default 63 ≈ one trading quarter**; a protected
constant, no agent CLI knob to lower it). Implemented as one declarative `GATE_SPECS` row:
```
GateSpec("min_holdout_observations", "holdout", "n_bars", "min_holdout_observations", ">=")
```
A strategy whose holdout `n_bars < 63` fails closed — complementing the `1/√T` haircut, which only
partly defends short holdouts and is **zero** at `N=1`. (Engine is weight-based; no discrete trade
count exists, so observations/`n_bars` is the floor metric. An "active/non-flat observations" floor
is **deferred** — see §9.)

### Wall D — The shortlist transition gate  *(closes C1; ties A–C together)*

`BACKTESTED → SHORTLISTED` for `--actor agent` now requires a **fresh, artifact-matched,
single-use gate-pass record** — exactly mirroring the live gate (which trusts an unrevoked approval
row, not the stage flag). The new `gate_evaluations` table is that record.

- `research promote` runs the gate (in the protected policy layer), writes a `gate_evaluations`
  row (on **both** pass and fail, for audit), and on pass calls `transition_strategy(... SHORTLISTED)`.
- `transition_strategy` becomes the **single enforcement point** for both live and shortlist: for
  target `SHORTLISTED` with `actor == AGENT`, it requires + **consumes** an unconsumed `passed=1`
  `gate_evaluations` row whose recomputed `(code_hash, config_hash, dependency_hash)` match the
  strategy's current artifact identity. No match → `TransitionError`. (Same hash-recompute defense
  as the live gate: pass the gate, then edit the strategy, and the stale record won't match.)
- `actor == HUMAN` is **exempt** (manual ops, back-steps, demo) — humans are trusted; the agent is
  the adversary.

Because only `research promote` can mint a passing row and it consumes the row in the same
transaction, a raw `registry transition --to shortlisted --actor agent` finds no unconsumed record
and is refused. The ungated path is closed for the agent without removing the human's manual lever.

## 4. Architecture

**Purity-preserving placement (Codex CRITICAL C2):** policy that needs I/O must not sit in the
unprotected CLI (`research_cmd.py`), or an agent could weaken it in a PR that never trips
code-owner review. New split:

- `algua/research/gates.py` *(CODEOWNERS, stays **pure** and registry-free)* — the haircut math,
  `GateCriteria` (+ `min_holdout_observations`), `GATE_SPECS` (+ the floor spec), `evaluate_gate`
  (+ `pit_ok`/`allow_non_pit` args, + the `pit_required` boolean check), `GateDecision`
  (+ funnel/PIT audit fields), the protected constants `FUNNEL_WINDOW_DAYS=90`,
  `MIN_HOLDOUT_OBSERVATIONS=63`, and the pure combine `effective_funnel_breadth(own_lifetime,
  windowed_total) -> int`.
- **NEW** `algua/registry/promotion.py` *(add to CODEOWNERS)* — the protected **policy
  orchestrator**: resolves funnel breadth (calls the protected repo queries + the pure combine),
  resolves `pit_ok` (presence + coverage), enforces the **relaxations-need-human** rule, calls the
  pure `evaluate_gate`, records the `gate_evaluations` row, and drives the transition.
  **Placement rationale:** the orchestrator needs repo I/O, and `algua.research` is deliberately
  **registry-free** (the stated convention: "backtest/research/live stay registry-free"). So it
  lives in `registry/` beside `transitions.py`/`approvals.py` (the live-gate policy) and imports
  the **pure** `research.gates` — a benign `registry → research.gates` direction (gates imports no
  registry ⇒ no cycle). `research_cmd.py` shrinks to: run `walk_forward` + resolve the universe
  (the heavy backtest/data I/O), hand the `WalkForwardResult` + universe metadata to
  `registry.promotion`, emit JSON. It holds **no policy**.

  **Two-phase, pre-peek ordering (critical):** promotion splits into `promotion_preflight(...)`
  and `run_gate(...)`. The preflight runs **before `walk_forward`** and enforces everything that
  must refuse *before the holdout is peeked/burned*: the relaxations-need-human guard, the
  **stage-legality check** (strategy must be `BACKTESTED` and `BACKTESTED -> SHORTLISTED` legal —
  so a passing token is never minted for an illegal source stage), and breadth resolution
  (refuse "no measured breadth" here, not after the burn). `run_gate` runs **after** the walk:
  resolves `pit_ok`, evaluates, records the `gate_evaluations` row (pass/fail), and on pass
  transitions. This preserves the existing pre-peek breadth-refusal guarantee (there is already a
  test asserting no holdout row is written on a no-breadth refusal).
- `algua/registry/store.py` *(CODEOWNERS)* — `windowed_search_combos(window_days)`,
  `record_gate_evaluation(...) -> int`, `find_consumable_gate_evaluation(strategy_id, code_hash,
  config_hash, dependency_hash) -> int | None` (read-only), and `apply_transition(...,
  consume_gate_id)` (consumes the token atomically with the stage change).
- `algua/registry/transitions.py` *(CODEOWNERS)* — `_validate_shortlist_gate` (mirror of
  `_validate_live_gate`), wired into `transition_strategy` for `SHORTLISTED`.
- `algua/registry/db.py` — the `gate_evaluations` table (one migration).
- `algua/contracts/lifecycle.py` *(CODEOWNERS)* — unchanged edges; the new agent-gate is enforced
  in `transitions.py`, not by removing the `BACKTESTED → SHORTLISTED` edge (humans still use it).

## 5. Data model — `gate_evaluations`

One row per gate evaluation (pass **and** fail). Protected (written only by the policy layer).
FK to `strategies(id)` — relational state, not an audit snapshot.

```sql
CREATE TABLE IF NOT EXISTS gate_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    passed INTEGER NOT NULL,
    n_funnel INTEGER NOT NULL,
    own_lifetime_combos INTEGER NOT NULL,
    windowed_total_combos INTEGER NOT NULL,
    funnel_window_days INTEGER NOT NULL,
    breadth_provenance TEXT NOT NULL,          -- 'measured' | 'declared'
    pit_ok INTEGER NOT NULL,
    pit_override INTEGER NOT NULL DEFAULT 0,    -- non-PIT accepted under human override
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
    decision_json TEXT NOT NULL,               -- full GateDecision payload
    consumed INTEGER NOT NULL DEFAULT 0,        -- single-use: set when a transition consumes it
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_gate_evaluations_strategy ON gate_evaluations(strategy_id);
```

**Consume rule (single-use, artifact-matched, agent-only, atomic):**
- `find_consumable_gate_evaluation(strategy_id, code, config, dep) -> int | None` — **read-only** —
  returns the id of the most-recent `passed=1 AND consumed=0 AND actor='agent'` row whose
  `(code_hash, config_hash, dependency_hash)` equal the **recomputed** current identity. A `NULL`
  `dependency_hash` matches nothing (fail-closed, mirroring the live gate). The `actor='agent'`
  filter means a **human/override** promote's audit row is **never** an agent-consumable token —
  closing the "human leftover row consumed by a later raw agent transition" hole.
- `apply_transition(..., consume_gate_id: int | None = None)` — when `consume_gate_id` is set,
  the stage UPDATE, the `stage_transitions` INSERT, and `UPDATE gate_evaluations SET consumed=1
  WHERE id=? AND consumed=0` all run in **one transaction**; if the token row didn't flip
  (`rowcount != 1`) the whole transition rolls back. So a token can never be consumed without the
  stage advancing, nor the stage advance on a vanished token.

`transition_strategy` (for `SHORTLISTED` + agent) calls `find_consumable...` to get the id, refuses
if `None`, and passes that exact id to `apply_transition` — so it consumes **the row this promotion
minted**, not merely "some matching row."

**Identity source (critical):** the row's `code_hash`/`config_hash`/`dependency_hash` are written
from `approvals.compute_artifact_hashes(name)` — the SAME function the shortlist (and live) gate
recomputes at transition time — **not** from `wf.code_hash`. `wf.code_hash` is git-HEAD-based
(`backtest/stamps.py`) and differs from the first-party-source-closure hash
`compute_artifact_hashes` derives; using it would make the match never succeed.

## 6. CLI surface changes (`research promote`)

- **ADD** `--allow-non-pit` — human-only audited override (Wall B).
- **NO** new `--funnel-window-days` / `--min-holdout-observations` flags — both are protected
  constants (Wall A/C); removing the knobs removes the bypass surface (Codex C3).
- **`--n-combos` (declared breadth) and `--allow-holdout-reuse` and `--allow-non-pit` are honored
  only with `--actor human`.** If `--actor agent` passes any of them → `ValueError`
  ("gate relaxation requires --actor human"). For the agent, breadth must be **measured**
  (`own_lifetime > 0`), PIT is required, holdout reuse is refused.
- Payload/`gate_reason` gain the funnel breakdown (`n_funnel`, `own_lifetime_combos`,
  `windowed_total_combos`, `funnel_window_days`) and the PIT verdict/override.

## 7. Build order (slices)

Three slices, each leaving the **full gate green** (plan-review correction: Walls A–D + the CLI are
mutually dependent — `pit_ok` becomes a required arg, the agent shortlist becomes gated, and the
minting path must exist — so they cannot be committed as independent green sub-steps; they land as
one coherent slice):

1. **Inert data layer** — `gate_evaluations` table (db.py) + store methods
   (`windowed_search_combos`, `record_gate_evaluation`, `find_consumable_gate_evaluation`,
   `apply_transition(consume_gate_id=...)`) + repository.py protocol. No behavior change; green.
2. **The gate (coherent)** — gates.py pure changes (constants, `effective_funnel_breadth`,
   `min_holdout_observations` spec, `pit_required` check, `GateDecision` fields), protected
   `registry/promotion.py` (preflight + run_gate), the shortlist gate in `transitions.py`, the
   `research_cmd.py` rewire (+ `--allow-non-pit`), **and** reconciliation of every affected test
   (see §10). One green commit.
3. **Housekeeping** — root `CODEOWNERS` += `promotion.py`; CLAUDE.md help text.

## 8. CODEOWNERS

Add to the **root** `CODEOWNERS` (not `.github/CODEOWNERS` — the repo's owners file is at the root):
`/algua/registry/promotion.py   @Lior-Nis   # promotion policy (breadth/PIT/floor + shortlist gate)`.

## 9. Deferred (tracked) follow-ups — Codex findings out of scope for 137

- **Durable family/hypothesis identity** (Codex HIGH): search breadth keyed by free-text
  `strategy_name` doesn't bind aliases/renames/copies. The `max(own_lifetime, windowed_total)` form
  removes the *double-count* subtlety, but cross-name family identity belongs to **#126/#122**
  (family metadata). Document the residual alias-splitting risk there.
- **Sweep reservation ledger / burst race** (Codex HIGH): record trial reservations before
  evaluation. Not a live threat under the **single sequential** agent loop; revisit if promotion
  ever runs concurrently.
- **Full PIT provenance certification** (Codex HIGH): trusted-source / as-of-constraint /
  certification bit beyond the coverage check. Data-lane issue (Codex owns that lane).
- **Active / non-flat observation floor** (Codex MEDIUM): reject do-nothing portfolios that satisfy
  `n_bars` with flat days. Refinement on top of Wall C.
- **Pre-existing relaxation flags** (`--min-holdout-sharpe` etc.) let an agent lower the *base*
  bar — the same C3 shape as the new flags, but pre-existing and broader than 137. Flag for a
  follow-up that applies the relaxations-need-human rule to the base thresholds too.

## 10. Testing

- **Wall A:** lone hypothesis ⇒ `N_funnel == own_lifetime` (no regression); a sibling sweep inside
  the window raises the bar; outside the window does not; `max` (not sum) verified.
- **Wall B:** non-PIT run not promotable; PIT with `earliest_effective > start` fails coverage;
  `--allow-non-pit` works only for human; provenance stamped.
- **Wall C:** `n_bars < 63` fails closed; `≥ 63` passes; interaction with the degenerate-holdout
  `inf` path unchanged.
- **Wall D:** raw `registry transition --to shortlisted --actor agent` refused with no record;
  `research promote` pass mints+consumes a record and transitions; a record stops matching after the
  strategy's code/config changes; human raw-transition still works.
- **Relaxations-need-human:** agent passing `--n-combos` / `--allow-holdout-reuse` /
  `--allow-non-pit` → `ValueError`.
- **Audit:** `gate_evaluations` row written on both pass and fail with the full breakdown.
- **Stale-pass regression (Codex C2):** `research promote` from an illegal source stage (e.g.
  `idea`) with full human overrides leaves **zero** `gate_evaluations` rows and burns no holdout
  (preflight refuses before any write).
- **Pre-peek breadth (Codex C3):** agent promote with no measured breadth writes **no** holdout
  row (refusal precedes `walk_forward`) — guarded by the existing
  `test_promote_with_universe_refuses_with_no_breadth_before_walkforward`.
- **Human leftover not a token (Codex C1):** a human/override promote's passing audit row is
  `actor='human'`; a later raw `registry transition --to shortlisted --actor agent` after a
  back-step is **refused** (the human row is not an agent-consumable token).
- **Atomic consume (Codex H):** consume + transition commit together (failure-injection: a forced
  transition-insert failure leaves the token unconsumed and the stage unchanged).
- **Existing-test reconciliation:** `test_research_gates` (add `pit_ok=True`; `test_gate_checks_are
  _table_driven` now expects `names_from_eval == table_names | {"pit_required"}`),
  `test_cli_research`, `test_cli_sweep`, `test_e2e_lifecycle`, `test_cli_paper`,
  `test_registry_approvals`, `test_registry_store` — scaffolding that agent-transitions to
  `shortlisted` switches to `--actor human` (humans are exempt; these reach a later stage, they
  don't test the gate).

## 11. Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` — all green.
