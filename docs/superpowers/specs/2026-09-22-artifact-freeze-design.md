# Artifact freeze: making the forward-evidence clock runnable

**Status:** design, approved in principle 2026-09-22. Not yet planned or implemented.

## The problem

`paper -> forward_tested` needs roughly 250-500 daily observations under ONE unchanged artifact
identity — one to two years of elapsed time. A tick counts only if its stored
`(code_hash, config_hash, dependency_hash)` equals the strategy's identity **as recomputed at
evaluation time**; anything else is dropped as `identity_drift`
(`algua/registry/forward_evidence.py:98-116`).

Measured over the 180 days to 2026-09-22:

| What moves the identity | Frequency |
|---|---|
| `uv.lock` (any dependency at all) | 15 of 15 lockfile commits |
| A strategy's first-party import closure (14 modules for `cross_sectional_momentum`) | 24 distinct days |

The identity moves every 5-7 days. The counter needs 250-500 days of it not moving, so it resets
roughly fifty times faster than it fills. `forward_gate_evaluations` is empty: no strategy has ever
accumulated enough surviving evidence to be evaluated at all, independent of whether any of them is
any good.

This is a DIFFERENT problem from the LCB wall. The LCB says the sample must be large. This says a
sample cannot be accumulated. Lowering the Sharpe bar would not help, because the ticks are discarded
before anything computes a Sharpe.

The identity rule itself is correct and is not in question. "The thing that traded is the thing that
was gated" is what stops a year of evidence being claimed for code rewritten last week. The defect is
that the system CHECKS the invariant without doing anything to KEEP it — so the check fires
constantly, on changes that had nothing to do with the strategy.

## The decision

Freeze the deployed artifact per strategy, and evaluate evidence against a **deployment epoch**
rather than against whatever is currently checked out.

Concretely: when a strategy is adopted into paper, mint an immutable, content-addressed deployment
artifact. The supervisor executes that artifact — not the working tree — for as long as the
deployment is active. Its identity therefore cannot move, and the clock runs.

### The seam: a DB-less, broker-less planner

Only the decision is frozen:

```
exact artifact + exact inputs (bars, positions, equity, universe) -> target weights + order intents
```

`run_tick` currently computes the decision and then crosses straight into cancellation and broker
submission (`algua/live/live_loop.py:364` cancel, `:371-396` submit). That crossing is the extraction point.

**Frozen (runs from the artifact):** strategy load and signal, construction, overlays, capacity cap,
gross utilization, decision timing, per-strategy risk semantics that decide whether or what to trade,
target weights and order intents.

**Current supervisor (never frozen):** the registry DB and its migrations, broker credentials,
cancel and submit, fill ingestion, reconciliation, shared snapshot refresh, book-level breakers, the
account buying-power pool, kill switches, audit and tick recording.

### Why the supervisor must stay single and current

Two findings make "run `paper run-all` inside 18 frozen worktrees" unsafe, and they are the reason
this design is not the simpler one first proposed:

1. **A frozen checkout must never open the registry DB.** Every `registry_conn()` runs THAT
   checkout's migration code, and migrations create tables, drop tables, alter columns and rewrite
   `user_version` (`algua/registry/db/migrate.py`). Eighteen stale migration paths against one
   authority is a corruption hazard, not an isolation win.
2. **`paper run-all` is account-wide.** One broker, one ingest, one shared union snapshot, one
   reconcile, one buying-power pool, then a loop over strategies (`algua/cli/paper_cmd.py:986-1122`).
   Eighteen independent run-alls would each reconcile the same account and each keep a private idea
   of buying power. They would not even serialize: the operator lock is per-worktree git directory
   (`algua/cli/operator_cmd.py:78`).

So: ONE supervisor dispatching N isolated planner calls.

## What is in the frozen set

Derived from the code, not assumed.

| Element | Treatment | Why |
|---|---|---|
| Strategy + first-party decision closure | Frozen | It is the decision |
| Resolved `StrategyConfig` | Frozen | Already the whole of `config_hash` (`strategies/base.py:350`) |
| Numerical / runtime dependency set | Frozen | numpy..vectorbt can change a number |
| Python interpreter, ABI, platform | Frozen AND recorded | `pyproject.toml` pins only `>=3.12`; a lockfile is universal across markers, so the same lock selects different variants on different interpreters |
| Planner protocol version | Stamped | The supervisor/planner contract must be versioned or a protocol change is invisible |
| Bars / universe / calendar inputs | NOT frozen — resolved outside and passed in, recorded per tick | Bars are already content-addressed and the tick already records `snapshot_id` (`execution/order_state.py:198`); universe membership is deliberately as-of-today (`registry/universe_binding.py:30`) |
| Registry DB | Shared authority, unreachable from the planner | See above |
| Broker, credentials, account state | Supervisor only | |
| `.env` | NOT copied into the artifact | It holds DB/data paths, credentials, exchange and risk controls, and is re-read on every call (`config/settings.py:21,48,61,158`). Behaviour-affecting settings become deployment fields; credentials and paths stay operational |

A venv may be SHARED between deployments only when keyed on the complete environment fingerprint
(dependency digest + interpreter + ABI + platform), mounted read-only, and exposing no `algua`
installation bound to a mutable checkout. Note `gate_runner` is not precedent for sharing: it
provisions a separate environment per throwaway worktree and scrubs `VIRTUAL_ENV`,
`UV_PROJECT_ENVIRONMENT` and `PYTHONPATH` (`operator/gate_runner.py:56,104`).

## Records

- **`deployment_artifact`** — content-addressed tree or wheel, resolved config, environment digest,
  interpreter/ABI/platform, planner protocol version.
- **`strategy_deployment`** — strategy, artifact id, research gate id, activated_at, retired_at,
  superseded_by.
- **Every tick** additionally records `deployment_id`.
- **Forward gate** evaluates ONE deployment epoch, never "whatever is current".

## Fix policy: strict

**Any decision-affecting fix requires a new deployment and resets that strategy's clock.** There is
no compatible-patch escape hatch.

The rejected alternatives and why:

- A human "this cannot change decisions" label is a loophole with no proof obligation.
- The tighten-only precedent does not generalise. Overlays may not add symbols, raise absolute
  weights or flip signs (`portfolio/overlays.py:69`), but they still deliberately change weights
  (`:92`). Safer is not the same strategy.
- Replaying history proves nothing about future inputs. Equality over a finite sample is not
  equality of functions.

Evidence may survive a change only when mechanically established: the change lies outside the frozen
planner boundary, or normalized executable content and environment are unchanged, or an exhaustive
domain proof shows identical outputs. In this system the third is rare.

For an urgent safety defect: halt the strategy, deploy the fix, accept the reset.

This is deliberately strict, and the cost is real — a genuine bug in shared construction code costs a
strategy its whole accumulated clock. Revisit only with a concrete proof obligation, never with a
label.

## Migrating the existing 18

They are already identity-drifted (PRs #656 and #657 both moved the identity), so nothing is lost by
re-qualifying, and there is no honest way to grandfather them: a changed artifact has no
exact-matching qualified research gate, so `qualified_holdout_sharpe` returns nothing
(`forward_evidence.py:158`) and the forward evaluator fails a recoded strategy without one
(`research/forward_gates.py:262`).

Order of preference:

1. Select today's vetted artifact, re-run research qualification, then open a deployment epoch.
2. Reuse an older already-qualified artifact only if reconstructible, free of known defects, and
   safe to resume from current state.
3. Human-authorised holdout reuse only when genuinely fresh OOS data is unavailable.

The legal route back is `paper -> candidate -> backtested` (`contracts/lifecycle.py:30`), since
`research promote` requires exactly `BACKTESTED` (`registry/promotion.py:136`). This needs a controlled
migration command, not ad hoc stage manipulation. Prefer a fresh OOS interval: the holdout ledger
rejects any overlapping interval regardless of provenance (`registry/store/holdout.py:35,61`), and
reuse is human-only and signature-bound (`registry/promote_run.py:185,225`).

## Defects to fix alongside

**1. Back-crediting (anti-gaming, exists today, independent of this design).**
`forward_evidence` selects `WHERE lane='paper' AND strategy_id=?` with no lower bound
(`forward_evidence.py:201-207`). Any prior tick matching the adopted identity is credited, whenever
it happened — so an artifact can be adopted AFTER seeing part of its forward performance. Fix: an
explicit epoch start; no back-crediting before activation.

**2. Cosmetic source churn.** `code_hash` hashes raw `inspect.getsource()` (`approvals.py:102`), so a
comment or reformat invalidates every prior approval and resets every clock. Fix: normalize before
hashing.

**3. (Proposed, NOT a defect — needs its own review.)** `code_hash` covers the entire construction
and overlays modules, so editing ANY policy invalidates EVERY strategy including ones that never use
it. The docstring shows this is deliberate (`approvals.py:19-23`). Narrowing it to the selected
callables is defensible — a policy a strategy does not use cannot change its decisions — but it is a
WALL CHANGE and must be reviewed as one, separately. It is listed here because it is a large part of
the 24-days-per-180 churn, not because it is agreed.

## Rejected alternatives

- **Cohort freeze (one epoch for the whole lane).** Advancing it for any one strategy resets all
  eighteen clocks, so nothing can ever be fixed without paying the full reset. The evidence model is
  already per-strategy and needs no such coupling.
- **Per-strategy venvs.** ~1.2GB each, ~22GB for eighteen on a disk at 94%, addressing the SMALLER
  resetter (6 days vs 24) and mostly duplicating: strategies adopted in the same window share a
  lockfile.
- **A smarter semantic hash instead of freezing.** Attempted in PR #658 and parked after three review
  rounds. Exact semantic reachability across dynamic imports, package data, numerical libraries and
  stateful execution is not provable by this hashing machinery. Normalization is worth doing for
  diagnostics and churn (defects 2 and 3), but it cannot be the safety boundary.
- **Recording the identity at promotion time and comparing ticks against THAT, with no freeze.**
  Considered seriously because it would be far cheaper. It does not work: ticks honestly stamp the
  identity of the code that ACTUALLY RAN, so if execution still follows the moving checkout the new
  ticks still mismatch and accumulation still stops. Stamping the promotion identity while running
  different code would make provenance false. Accepting multiple tick identities mixes distinct
  economic return regimes into one sample. The recorded identity must resolve to executable immutable
  content — which is the freeze.

## Live

Live must execute the same artifact that earned the evidence. A correction to an earlier premise: a
moving checkout does NOT silently keep trading under a stale certificate — `live_gate` recomputes the
identity, requires an exact-matching authorization and re-verifies the signature against the trust
anchor (`registry/live_gate.py:166-192`). Checkout movement causes a live OUTAGE, not unauthorized
trading. Freezing removes the outage and makes the certificate mean something durable.

The go-live ceremony changes from signing the ambient checkout's identity to signing a
`deployment_id` plus manifest digest, with the same artifact promoted from paper to live — no
rebuild, no re-freeze during go-live.

## Decomposition

This is too large for one implementation plan. Sequenced so each slice is independently useful and
independently reviewable:

1. **The two defects** (back-crediting epoch bound, cosmetic source normalization). Shippable now,
   valuable with or without the freeze, and the epoch bound is a prerequisite for slice 3 anyway.
2. **The planner seam.** Extract decision-from-execution inside `run_tick` behind a versioned
   contract, still running in-process. No artifacts, no worktrees, no behaviour change — a pure
   refactor with parity tests against the current path.
3. **Deployment records + epoch evaluation.** `deployment_artifact`, `strategy_deployment`,
   `deployment_id` on ticks, and the forward gate evaluating one epoch. Still executing from the
   working tree, so the records are recorded and enforced before anything is frozen.
4. **Frozen execution.** Mint the artifact, run the planner from it, supervisor dispatch.
5. **Migration of the 18**, as its own controlled command.
6. **Live** ceremony onto `deployment_id`.
7. Separately and only if reviewed as a wall change: narrowing `code_hash` to the selected
   construction/overlay callables (defect 3 above).

## Open questions

- Artifact representation: git worktree at a pinned commit, an exported tree, or a built wheel.
- Planner transport: subprocess with a JSON contract is the obvious first cut; the protocol version
  must be stamped either way.
- Garbage collection: when a deployment is retired, what is reclaimed and when.
- Whether the supervisor should refuse to start when any active deployment's artifact is missing
  (fail closed) or skip that strategy (degrade). Probably fail closed.
