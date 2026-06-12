# `forward_tested` lifecycle stage + forward-test evidence gate (#124)

**Date:** 2026-06-10
**Issue:** #124
**Status:** GATE-1 APPROVED (5 panel rounds: Codex + Gemini Flash + GLM-5.1; all accepted findings folded)

## Problem

The lifecycle gives backtest evidence two stages (`backtested` = ran, `candidate` = passed a
measurable gate) but forward/paper evidence only one (`paper` = running). There is no stage that
means "the forward test **passed**." `paper -> live` is gated solely by the human go-live
signature — an *authorization* wall with no *evidence* gate in front of it. The human is asked to
sign with nothing measurable behind the ask.

## Decision summary

Add a `forward_tested` stage between `paper` and `live`, reached **only** (for an agent) by a
passing run of a new forward-test evidence gate — the paper-side analog of `research promote`.
The human signature wall moves intact to `forward_tested -> live` and now always has evidence in
front of it. A bare flag with no gate was explicitly rejected (issue #124): a stage must mean
"passed measurable criteria" or it is forgeable ceremony.

Settled with the operator (2026-06-10): stage name `forward_tested`; performance criterion =
degradation bound + absolute floor; minimum evidence window = 63 daily return observations from
decided paper trading sessions; the agent **can** pass the gate autonomously (the golden rule
extends to "up to and including `forward_tested`"; live remains human-only).

## Lifecycle changes (`algua/contracts/lifecycle.py`, protected)

```python
class Stage(StrEnum):
    ...
    PAPER = "paper"
    FORWARD_TESTED = "forward_tested"
    LIVE = "live"
```

New `_LIVE_TRANSITIONS`:

| From | To |
|---|---|
| `PAPER` | `FORWARD_TESTED`, `CANDIDATE` |
| `FORWARD_TESTED` | `LIVE`, `PAPER` |
| `LIVE` | `PAPER` (unchanged) |

- The `PAPER -> LIVE` edge is **removed**. `LIVE` is reachable only from `FORWARD_TESTED`.
- Demotion from live still lands at `PAPER`: a strategy pulled from live must re-earn forward
  evidence before going back.
- `FORWARD_TESTED -> PAPER` is open to any actor: re-entry to `forward_tested` always requires a
  fresh full gate pass, so demotion gains an agent nothing.
- A human may raw-transition `paper -> forward_tested` without a token (same human exemption as
  `candidate`); the transition records the recomputed identity hashes for audit. This exemption is
  harmless **because the live wall itself demands a forward certificate** (below) — a raw-parked
  strategy still cannot reach `live` without evidence.
- `-> RETIRED` stays derived; the new stage gains it automatically by construction.
- Existing strategies at `PAPER` keep their stage and simply gain the new edge; their legacy ticks
  are inadmissible (below), so they re-qualify under the new stamping. No data migration. The
  human raw transition is the escape hatch for anything that was already near go-live.

## Evidence model — identity-bound, wall-clock, paper-lane ticks

The only admissible evidence is `tick_snapshots` rows written by the wall-clock paper lane
(`paper trade-tick`). `paper run` (historical replay through SimBroker) writes orders/fills but
**no** tick snapshots; equity comes from the real Alpaca paper account over real elapsed time.

`tick_snapshots` gains (schema **v21**, all stamped by the writer, never caller-supplied):

| Column | Purpose |
|---|---|
| `lane` (`CHECK (lane IN ('paper','live'))`) | `live trade-tick` shares the table; the gate counts `lane='paper'` only, so a demoted strategy can't count old live ticks. Hardcoded per CLI command, not a parameter. |
| `code_hash`, `config_hash`, `dependency_hash` | Identity at tick time, from `compute_artifact_hashes(name)`. The gate counts only ticks matching the **current recomputed** identity — code/config drift restarts the evidence clock (same recompute-don't-trust-rows philosophy as the live gate). |
| `strategy_id` | FK provenance; name reuse after delete/recreate can't adopt old rows. |
| `account_id` | The Alpaca paper account id (from the account GET already made each tick) — anchors the single-tenant and cash-flow checks. |
| `cash` | Account cash at tick time — audit data for the cash-flow check. |
| `clock_source` (`CHECK (clock_source IN ('broker','local'))`) | `tick_ts` is taken from the Alpaca clock endpoint (broker server time) when available, else local time flagged `'local'`. Only `'broker'`-clocked ticks are admissible — a skewed local clock can't fabricate session spread. |
| `recorded_at` | Writer-stamped insertion time. Window bounds for the integrity/clean-since universes use the DB-assigned row `id` and `recorded_at`, never `tick_ts` — you cannot bound a universe by the very timestamp whose quality you're auditing. |

`paper_orders` gains `strategy_id` (v21, stamped by the wall-clock writer) so broker-fill
attribution (below) binds to registry identity, not a reusable free-text name.

Legacy rows (NULL in any new column) never count — fail closed. No backfill.

**Admissibility** (a tick is an *observation* iff ALL hold): `lane='paper'`,
`clock_source='broker'`, identity hashes = current recomputed identity, `strategy_id` matches,
`decision_ts` freshness bound holds, and `tick_ts` is not in the future of the gate run.

**Freshness bound, precisely:** let `S(ts)` be the trading session containing `ts` (or the
previous session if `ts` falls outside any session) via the market calendar. A tick is admissible
only if `0 <= sessions_between(S(decision_ts), S(tick_ts)) <= 2`. This kills the
historical-window attack where `trade-tick --end <past>` manufactures old decision dates in
minutes: honest ticks decide on the latest closed bar, so their decision session trails the
wall-clock session by at most a data-staleness session or two.

**Daily series:** observations are keyed by **decision session**; the last admissible tick per
decision session wins. Equity series -> daily simple returns -> annualized Sharpe via the shared
`ANN`-based formula in `algua.backtest.metrics` (same units as the holdout Sharpe; reuse, don't
reimplement).

**The integrity universe is wider than the observation set** (so a bad tick cannot hide by being
inadmissible): integrity checks run over **all** `lane='paper'` ticks for this `strategy_id` —
any identity, any clock source, any `decision_ts` quality — bounded by **row id / `recorded_at`**
between the first admissible observation's row and the gate run. Inside that row set, any
reconcile failure, malformed or future `tick_ts`, or other evidence-quality defect fails the
gate. The evaluation row records `first_tick_id` / `last_tick_id` so the certificate's
clean-since check (below) has an ungameable anchor.

## Gate criteria (v1)

Evaluated by a pure function `evaluate_forward_gate(...) -> ForwardGateDecision` in a new
protected module `algua/research/forward_gates.py` (mirrors `research/gates.py`). Protected
module-level defaults; an agent may tighten any threshold, never relax — relaxation flags are
human-only, exactly like `--allow-non-pit` on `research promote`.

| # | Check | Default | Rationale |
|---|---|---|---|
| 1 | Window floor: daily **return** observations ≥ `min_forward_observations` | **63** | Counted in returns (N+1 distinct decision-session equity points yield N returns), so the floor is genuinely symmetric with `MIN_HOLDOUT_OBSERVATIONS`; underpowered windows fail closed. |
| 2 | Coverage: decided sessions ≥ `min_session_coverage` of the trading sessions in [first, last] admissible tick | **0.9** | Sparse ticking lumps multi-day returns into "daily" observations and inflates the annualized mean (and permits selective omission). Coverage forces an essentially daily series so the `ANN` math is honest. |
| 3 | Performance: `realized_sharpe >= max(degradation_factor * holdout_sharpe, sharpe_floor)` | factor **0.5**, floor **0.3** | `holdout_sharpe` is the RAW measured holdout Sharpe (the check value, not the post-haircut bar) from the newest `gate_evaluations` row with `passed=1` (explicitly in the query), `pit_ok=1`, `pit_override=0`, and identity hashes equal to the current recomputed identity. No such row ⇒ fail closed (a re-coded strategy needs a fresh `research promote` first; a passing row also implies holdout ≥ the 0.5 base bar, so the degradation term is well-behaved). Floor 0.3 keeps a barely-positive forward run from passing on noise. Stored as a dedicated column for audit. |
| 4 | Volatility floor: annualized realized vol ≥ `min_forward_vol` | **0.02** | Near-zero vol makes Sharpe undefined/explosive; a do-nothing strategy must not pass. Fail closed, mirroring `sharpe_haircut`'s inf-on-degenerate behavior. |
| 5 | Drawdown: realized max drawdown over the admissible series ≤ `max_forward_drawdown` | **0.25** | The kill-switch breaker is optional per tick invocation and resettable, so the gate measures drawdown itself from the evidence series. |
| 6 | Integrity | — | `reconcile_ok=1` and well-formed timestamps for **every** tick in the integrity universe (defined above — all paper-lane ticks for the strategy in the window, regardless of admissibility); per-strategy kill switch and global halt currently clear; **no kill-switch trip audit events** for the strategy within the window (a tripped-then-resumed forward test is a failed forward test). |
| 7 | Account hygiene | — | Single-tenant: no other strategy has `lane='paper'` ticks on the same `account_id` overlapping the window (sibling strategies contaminate account-level equity). Account activities: the gate queries the broker's account-activities endpoint (read-only, added to `AlpacaPaperBroker`) with **exhaustive pagination — partial or failed activity history fails the gate, never passes it**. External capital movements (Alpaca types `CSD`, `CSW`, `TRANS`, `JNLC`, `JNLS`, `ACATC`, `ACATS`) inside the window fail. `DIV`/`INT`/`FEE` are legitimate position-attributable flows and pass (the backtest's `adj_close` includes dividends, so excluding them would bias the comparison). Every `FILL` activity in the window must reconcile to one of this strategy's persisted paper orders by broker order id **and `strategy_id`** (added to `paper_orders` in v21) — unattributable trading on the account (manual/API orders, untracked siblings, name-reuse adoption) fails closed. |
| 8 | Recency: newest admissible tick within `max_staleness_sessions` of the gate run | **5 trading sessions** | Calendar days false-fail over long weekends; the strategy must still be actively forward testing when promoted. |

**Recorded, not yet enforced:** `n_concurrent_forward` — the number of distinct strategies with
**any** `lane='paper'` ticks overlapping the window (any account, any admissibility — a strategy
that ran and failed still inflated the family-wise error rate). Promoting the best of N
concurrent forward tests is a multiple-testing problem (the forward analog of Wall A); v1 records
the breadth in the decision row and the spec flags it to the operator. Enforcement (a deflation
or a fleet cap) is a follow-up tied to the funnel-breadth machinery.

## Mechanics — mirrors `research promote`

**Orchestration** in a new protected `algua/registry/forward_promotion.py`:

1. `forward_promotion_preflight(...)` — hard refusals before any work: actor must be AGENT or
   HUMAN; stage must be `PAPER` **or `FORWARD_TESTED`** (re-evaluation, below); relaxation flags
   require HUMAN.
2. `run_forward_gate(...)` — recompute artifact identity; assemble admissible ticks; build the
   daily series; locate the qualified backtest gate row; run the broker activities check;
   `evaluate_forward_gate`; **record the evaluation row (pass and fail)**; on pass from `PAPER`,
   atomically transition `PAPER -> FORWARD_TESTED` via `transition_strategy`.

**Re-evaluation at `FORWARD_TESTED` (certificate refresh):** running `paper promote` on a
strategy already at `FORWARD_TESTED` re-runs the full gate over the updated evidence window and
records a new evaluation row **without a stage change**. This is the certificate-refresh path —
the live wall (below) demands a *recent* passing evaluation, and a fresh full re-run (Sharpe,
coverage, integrity, hygiene — everything) is the only honest way to refresh it. CLI exit 0 on
pass, with `"promoted": false` distinguishing re-evaluation from promotion.

**Token table** — new `forward_gate_evaluations` (schema v21): `strategy_id` FK, `passed`,
`n_forward_observations`, `min_forward_observations`, `session_coverage`, `realized_sharpe`,
`holdout_sharpe`, `degradation_factor`, `sharpe_floor`, `realized_vol`, `min_forward_vol`,
`realized_max_drawdown`, `max_forward_drawdown`, `first_tick_id`, `last_tick_id`,
`first_tick_ts`, `last_tick_ts`, `max_staleness_sessions`, `n_reconcile_failures`,
`n_concurrent_forward`, `account_id`, `code_hash`, `config_hash`, `dependency_hash`, `actor`,
`decision_json` (full `ForwardGateDecision`), `consumed` (default 0), `created_at`.

**Token consumption** — `transitions.py` gains a third branch, the exact shape of Wall D, and
**both gate branches become source-stage-scoped**:

```python
elif rec.stage == Stage.BACKTESTED and target == Stage.CANDIDATE and actor is not HUMAN:
    consume_gate_id = _validate_shortlist_gate(...)
elif rec.stage == Stage.PAPER and target == Stage.FORWARD_TESTED and actor is not HUMAN:
    consume_forward_gate_id = _validate_forward_gate(...)
```

Scoping matters: today the shortlist gate fires on ANY agent transition **to** `CANDIDATE`,
which makes the pre-existing `PAPER -> CANDIDATE` back-step agent-impossible (an agent at
`PAPER` can never mint a shortlist token — `promotion_preflight` requires `BACKTESTED`).
Back-steps are free for any actor; the gates guard the *forward* edges only. Re-entry to
`CANDIDATE` or `FORWARD_TESTED` from below always requires a fresh gate pass.

`_validate_forward_gate` finds the newest **unconsumed passing AGENT** row whose stored identity
matches the current recomputed identity **and whose `created_at` is within a 7-day TTL** (a
passing evaluation is not an indefinitely bankable asset — mirrors the live challenge's expiry,
guards the recorded-but-not-yet-consumed path). `apply_transition` takes
`consume_gate_id` / `consume_forward_gate_id` parameters and **raises if both are non-None**
(mutual exclusivity enforced, not assumed). The consuming `UPDATE` **re-checks the full predicate
set** — `strategy_id`, `passed=1`, `actor='agent'`, `consumed=0`, the current identity hashes,
and the TTL — not just the row id (the current `gate_evaluations` consume trusts the earlier
lookup; the forward path must not copy that validate-then-consume gap). Consumption happens in
the same transaction as the stage change (single-use).

**Compare-and-swap stage updates** — `apply_transition`'s stage UPDATE becomes
`WHERE id = ? AND stage = ?` with a `rowcount == 1` check, raising a `TransitionError` whose
message distinguishes a concurrent-modification race from an illegal transition: two sessions
sharing the SQLite DB can both validate against a stale stage today and silently overwrite each
other. This hardens every transition, not just the new one (cross-link: #164 multi-process
concurrency tests).

**The live wall now demands the certificate** — `_validate_live_gate` (and challenge issuance,
for UX) additionally requires a **forward certificate**, selected as **the newest
`forward_gate_evaluations` row (pass or fail, consumed or not) with `strategy_id` = THIS
strategy's id AND identity hashes equal to the current recomputed identity** (identity alone is
not enough — two registry strategies with identical code/config/deps must not share or poison
each other's certificates) — which must satisfy ALL of: `passed=1` (a newer *failed*
re-evaluation invalidates any older pass — the system cannot learn "this no longer passes" and
still present the stale pass for signature); `created_at` within **10 trading sessions** of the
live transition; a clean record since the certificate (no reconcile-failed or malformed
paper-lane tick rows after `last_tick_id`, no kill-switch trip audit events after `created_at`;
kill switch and global halt clear); and **account hygiene re-checked over
`[created_at, live gate run]`** — the same exhaustive broker-activities query with the same
external-capital-movement and FILL-attribution rules, bound to the same `strategy_id` and the
certificate's `account_id` (a deposit or unattributable trade between certification and
signature invalidates the certificate). This closes the holes the stage flag
alone leaves open:
(a) code/config drift after promotion — the stage stays `forward_tested` but the certificate no
longer matches, so the human cannot sign the new artifact live without fresh evidence; (b) stale
evidence — a strategy parked at `forward_tested` must refresh via re-evaluation (above) before
go-live, and the re-run re-checks *everything*, not just recency; (c) evidence that went bad
after the pass — a reconcile failure or trip between certification and signature invalidates the
certificate. Not waivable in-band (a human who truly must bypass owns the DB; there is
deliberately no flag). The go-live challenge output includes the certificate summary (decision
values + evaluation date) so the human signs with the evidence in front of them. Consequently
the paper CLI (`trade-tick`, `show`, `kill`, `resume`, `flatten` — everything `_load_gated_strategy`
guards) accepts strategies at `PAPER` **or** `FORWARD_TESTED`: evidence accumulation continues
while awaiting signature, and identity drift at `forward_tested` simply yields inadmissible ticks
and a failed certificate check — fail closed, no auto-demotion machinery.

**CLI** — `uv run algua paper promote <name>` in `paper_cmd.py`. Emits the full decision JSON
(including a `promoted` boolean and, on refusal, which admissibility filters dropped how many
ticks — e.g. a broker-clock outage shows up as excluded local-clocked ticks, not a mystery);
exit 0 only on a passing gate. Audit-logged like `research promote`. Threshold flags with their
tightening direction (agent-legal values are on the *stricter* side of the default; the relaxing
direction refuses a non-human actor):

| Flag | Default | Stricter means |
|---|---|---|
| `--degradation-factor` | 0.5 | higher |
| `--sharpe-floor` | 0.3 | higher |
| `--min-observations` | 63 | higher |
| `--min-coverage` | 0.9 | higher |
| `--min-vol` | 0.02 | higher |
| `--max-drawdown` | 0.25 | lower |
| `--max-staleness` | 5 sessions | lower |

**CODEOWNERS** — add `/algua/research/forward_gates.py` and `/algua/registry/forward_promotion.py`.
`lifecycle.py`, `transitions.py`, `store.py`, `gates.py`, `promotion.py` are already protected.

## Touched-surface inventory

- `algua/contracts/lifecycle.py` — new stage + edges (protected).
- `algua/registry/db.py` — schema v21: `tick_snapshots` new columns; `forward_gate_evaluations`
  table; idempotent `_add_missing_columns` migration.
- `algua/execution/order_state.py` — `record_tick_snapshot` and `record_submitted_order` gain
  the new stamped params (`strategy_id` on orders).
- `algua/execution/alpaca_broker.py` — read-only clock + account-activities endpoints.
- `algua/cli/paper_cmd.py` — `trade-tick` stamps lane/identity/account/clock; accepts
  `FORWARD_TESTED` stage; new `paper promote` command.
- `algua/cli/live_cmd.py` — `trade-tick` stamps the same columns with `lane='live'`.
- `algua/research/forward_gates.py` — NEW, protected: criteria + decision dataclasses.
- `algua/registry/forward_promotion.py` — NEW, protected: preflight + orchestration + evidence
  assembly.
- `algua/registry/transitions.py` — forward-gate token branch; forward-certificate check in
  `_validate_live_gate` (protected).
- `algua/registry/store.py` / `repository.py` — record/find/consume forward gate rows; CAS stage
  update (protected).
- `algua/registry/live_gate.py` — challenge issuance surfaces the certificate (protected).
- `CODEOWNERS` — two new protected paths.
- Docs/skills: `CLAUDE.md` (lifecycle line, golden rules "up to and including forward_tested",
  command surface), `docs/agent/operating.md`, `operating-algua` + `run-the-research-loop` skills,
  kb lifecycle notes if they enumerate stages.

## Testing

- Lifecycle: new edges allowed, `paper -> live` rejected for ALL actors, derived retire edge,
  round-trips, demotion targets; `paper -> candidate` back-step works for an AGENT (gate scoping).
- Admissibility (table-driven): each filter independently excludes (lane, clock_source, identity
  drift, strategy_id mismatch, decision session beyond 2 sessions of tick session, future
  tick_ts, legacy NULLs); session boundary cases around weekends/holidays.
- Integrity universe: a reconcile-failed tick that is *inadmissible* (stale decision_ts, local
  clock, drifted identity) still fails the gate.
- Gate criteria (pure, table-driven): each check passes/fails independently; boundary values
  (exactly 63 obs, coverage exactly 0.9, Sharpe exactly at the bar, vol at floor, DD at cap);
  fail-closed paths (no qualified backtest row — incl. a failing row with `pit_ok=1`, zero
  admissible ticks, all-legacy ticks, identity drift mid-window splits evidence).
- Account hygiene: sibling strategy on same account fails; sibling on different account passes
  but increments `n_concurrent_forward` (which also counts failed/inadmissible siblings); mocked
  activities: a `CSD` deposit fails, `DIV`/`FEE` pass, an unattributable `FILL` fails, a
  pagination error fails (never passes).
- Token mechanics: single-use, identity-matched, TTL-expired token refused, agent-only
  requirement (human exempt at this edge), atomic consumption, failing rows unconsumable,
  both-tokens-passed raises, consume-time predicate recheck (identity drift between lookup and
  consume refuses).
- CAS: concurrent stale-stage transition attempt raises (distinguishable message) instead of
  overwriting.
- Live wall: `forward_tested -> live` requires human + signature + matching fresh certificate;
  a certificate belonging to an identity-identical sibling strategy is refused (strategy_id
  binding);
  drift after promotion blocks go-live; certificate older than 10 sessions blocks go-live;
  re-evaluation refreshes it without a stage change (`promoted: false`); a NEWER FAILED
  re-evaluation invalidates an older pass; a reconcile failure, kill-trip, deposit, or
  unattributable fill after certification blocks go-live; `paper -> live` impossible; challenge
  output includes certificate summary.
- CLI: `paper promote` happy path, each refusal (with excluded-tick accounting in the payload),
  JSON shape; relaxing-direction flag values refuse agents, tightening values accepted;
  `trade-tick` stamps all new columns; every paper command runs at `FORWARD_TESTED`.
- Migration: v20 DB with existing tick rows migrates; legacy rows excluded from evidence.

## Known limitations (documented, accepted for v1)

- **No slippage realism**: paper fills carry no modeled-slippage comparison (no slippage model
  exists anywhere — SimBroker and the backtest engine both fill at next open). The realized
  Sharpe is gross of realistic execution costs; `paper promote` output says so.
- **Sharpe over 63 obs is a screen, not proof**: the gate is an operational-forward-smoke +
  performance screen. The combination (coverage + vol floor + DD + integrity + hygiene) is what
  makes it evidence; the Sharpe alone would be noise.
- **Platform upgrades restart clocks**: `dependency_hash` covers the first-party closure, so any
  `algua.*` change invalidates in-flight evidence for all paper strategies. Correct (fail-closed)
  but operationally heavy; revisit if it bites.
- **A broker clock outage blocks evidence accumulation**: ticks recorded while the Alpaca clock
  endpoint is down are `clock_source='local'` and inadmissible. The tick still trades; only its
  evidentiary value is lost. `paper promote` output surfaces the excluded-tick counts so the
  cause is visible.

## Explicitly deferred (follow-up issues, not this PR)

- **Realized-vs-modeled slippage check** — model first, then the gate check.
- **Turnover / tracking-error bounds** — belongs with the portfolio-construction layer (#141).
- **Forward-fleet selection deflation** — `n_concurrent_forward` is recorded from day one;
  enforcement (deflation or a cap) joins the funnel-breadth machinery later.
- **`paper_sessions` table** — every session-ish fact (boundaries, identity, account, coverage)
  is derivable from the stamped per-tick rows; a second mutable record is a consistency liability
  with no current consumer.
- **Backtest<->paper decision-parity check inside the gate** — the `on_decision` parity seam
  exists; wiring a replay comparison into the gate is a separate evidence axis.
