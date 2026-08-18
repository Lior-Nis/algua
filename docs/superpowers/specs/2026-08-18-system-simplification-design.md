# System Simplification — Design Spec

Date: 2026-08-18
Status: approved-pending-review
Owner: Lior (design partner: Claude)

## 1. Goal and context

Algua's next growth axes (Modeling ML/DL/Agentic, multi-asset venues, MLflow/VPS/dataset
infra, Ideas engine + KB) are blocked less by missing features than by extension cost:
god-files, hand-rolled one-off mechanisms, domain logic living in CLI handlers, and
missing plugin seams. This spec defines a strangler refactor that makes the codebase
generically extensible under clean-code / SOLID / YAGNI / KISS / DRY, without changing
what the system *does*.

Locked decisions (from the 2026-08-18 grilling session):

- **Order:** simplification first. Modeling/Ideas, multi-asset, and infra axes are
  deferred; their Todoist tasks carry seam-context notes.
- **Approach:** strangler refactor — sequenced behavior-verified stages. No big-bang,
  no dual-path compat cruft.
- **Freedom:** nothing is frozen. CLI JSON shapes, DB schema, and all code are
  touchable. The system is non-operational and is halted for the duration (stage 0).
  The only hard rule: no regressions per stage.
- **Deletion:** allowed, case-by-case; the kill-list below is the approved set.
- **Target shape:** pure domain core + small `Protocol` adapters + explicit name→factory
  registries. No DI framework, no class hierarchies for their own sake.
- **Delivery:** review-gated pipeline (issues → subagent worktrees → multi-model review
  → green CI). Characterization tests added first wherever a carve touches thin coverage.
- **Success = all three:**
  1. *Extension-cost test* (primary): a representative feature lands as one new module
     plus a registration line, with no god-file edits.
  2. *Structural budget*: per-module LOC ratchet enforced in CI.
  3. *Agent-context test*: a fresh agent implements a backlog item from
     `docs/architecture.md` + the modules it touches; validated by dispatching the
     MLflow backlog task at the end.
- **Seam probes** (seams are sized against these; the features are NOT built now):
  A) an ML strategy with a training step + model artifact; B) one non-equity venue
  (24/7 calendar, no delistings); C) a second experiment tracker + a versioned-dataset
  provider; D) a new idea source + a second knowledge-base consumer.

## 2. Ground truth (audit findings this design rests on)

- The repo's recurring failure mode is **"pure half extracted, orchestration half left
  behind"**: `registry/intake.py` vs `paper_cmd._run_intake`; `research/lifecycle_gc.py`
  vs the ~290-line gc filesystem layer in `research_cmd.py`; `risk/book_*.py` vs
  `live_cmd._evaluate_book_loss_breaker`/`_build_book_exposure`;
  `research/family_audit.py` vs its CLI pipeline. Only
  `research/forward_gates.py` ↔ `registry/forward_promotion.py` got the pairing right.
- `registry/repository.py` already declares the Protocol seams (`GateLedger`,
  `HoldoutLedger`, `FamilyGraph`, …) that the 2,416-line `SqliteStrategyRepository`
  implements as one class — the `store.py` cut list is pre-written.
- Two `importlib` escape hatches (`paper_cmd.py` → `research_cmd.promote_task`,
  → `backtest_cmd.sweep_task`) exist solely to satisfy the CLI independence
  import-linter contract; they mark the highest-leverage extraction.
- 8 hand-rolled flock implementations, 7 atomic-write implementations (1 partial shared
  helper in `data/files.py`), 2 identical HTTP retry loops, and 2 line-for-line-parallel
  signed-challenge stacks (`live_gate.py` / `human_actor.py`).
- Two seams are already model extensions: the `data/providers` / `data/importers`
  name→factory registries (~1 line to extend) and the `LedgerKind` table-map in
  `execution/live_ledger.py`. The refactor generalizes these two patterns.
- `algua/operator/` is the reference package architecture (pure jobs/schedule/locks
  outside the CLI).
- Safety-critical ordering is encoded in comments at the code sites
  (audit-before-mutate, kill-switch-before-authorization, peak-ratchet-after-equity-
  validation, burn-on-peek/release-on-failure, the `#524 R9-H4` accepted residual).
  **These comments are the spec and move verbatim with their code.**

## 3. Kill-list (approved case-by-case)

DELETE (severable; git history preserves recovery):

| Item | ~LOC | Notes |
|---|---|---|
| `algua/shadow/` + `cli/shadow_cmd.py` + `shadow_evaluations` table | 440 | Zero core dependents. |
| `algua/monitoring/` (drift + decay) + `cli/monitoring_cmd.py` | 910 | Gates nothing, persists nothing. Rebuilt properly under the Modeling axis when live. |
| `research gc` (`research/lifecycle_gc.py` + CLI fs layer incl. hardened archiver) | 700 | Removes the repo's most complex bespoke atomic-move code. |
| `research family-audit` (`research/family_audit.py` + CLI) | 290 | Keeps `research/clustering.py` (core NOVEL classification). |
| `research dormant-sweep` | 110 | Advisory screen over an empty pool. |
| `registry/family_budget.py` | 63 | Zero importers; dead. |
| Factor eval layer: `cli/factor_cmd.py`, `backtest/factor_eval.py`, `registry/lineage.py`, `research/factor_fdr.py`, `factor_evaluations` table | 750 | KEEPS `features/catalogue.py` + `features/alphas.py`/`indicators.py` (the `@factor` decorator is load-bearing for `cross_sectional_momentum`). |
| `research pbo` (`research/cscv.py` + CLI) and collapse `sweep_with_matrix` back into `sweep()` | 380 | Rebuildable from git if the overfit signal is wanted again. |

Plus each item's tests (~1,500 test LOC total).

KEEP (explicitly protected):

- **Idea pool** (`cli/idea_cmd.py`, `registry/ideas.py`, `research/ideas.py`,
  `research/idea_dedup.py`, `contracts/idea.py`, `ideas` table) — substrate for the
  Ideas-engine axis. `DataCapability` stays importable by `features/catalogue.py` and
  `data/capabilities.py`.
- **Knowledge sync** (`algua/knowledge/`) — actively used Obsidian vault; gets a
  Protocol seam (§5) instead of deletion.
- **Frozen LORD++ FDR machinery** — structurally welded into the atomic promote
  transaction. Only the *safe subset* dies, during the store.py carve (§6):
  `research/fdr_lord.py` and the provably-dead `if fdr_binding:` branches
  (`promotion.py` unconditionally sets `fdr_binding=False` / skip `stats_advisory`).
  The ledger schema and recorded rows remain.
- **Family governance** — IS core; changes what `research promote` accepts. Untouched
  semantics.
- **Advisory statistical stack** (DSR/haircut/regime/bootstrap/n_eff/IR) — recorded on
  every gate run; kept. (Deleting it would edit core gate code for no verdict change.)
- **`audit/`**, **`observability/`** — money-path audit trail and structured logging.

## 4. Shared primitives — `algua/primitives/` (new leaf package)

Pure stdlib; imports nothing from `algua`; importable by everything (import-linter
contract). Dissolves `models/registry.py`'s documented reason for hand-duplicating
helpers ("must stay an import leaf" — a leaf may import another leaf).

1. **`primitives/flock.py`** — one lock context-manager replacing the 8 flock
   implementations (`data/manifest.py`, `data/staging.py`, `models/registry.py`,
   `operator/gitops.py`, `knowledge/sync.py`, `operator/schedule.py` ×2,
   `backtest/core_budget.py`). Parameterized on exactly the axes they differ on:
   blocking vs `LOCK_NB`, ENOLCK policy (fail-closed default; KB-sync's degrade-open
   becomes an explicit opt-in), optional JSON holder metadata in the lock body.
2. **`primitives/atomic_io.py`** — `data/files.py`'s fsync/rename/`write_bytes_atomic`
   + `append_if_absent` promoted here; the 4 surviving duplicates deleted
   (`knowledge/sync.py`, `operator/schedule.py`, `models/registry.py`,
   `data/manifest.py::_repair`).
3. **`primitives/retry.py`** — one exponential-backoff helper; unifies the two HTTP
   retry clones (`execution/alpaca_broker.py`, `data/providers/alpaca.py`).
4. **`registry/challenges.py`** — one challenge lifecycle (issue / find-pending /
   consume nonce under `BEGIN IMMEDIATE`) parameterized by table + signature namespace;
   replaces the twin stacks in `registry/live_gate.py` and `registry/human_actor.py`.
   `verify_signature` remains the single ssh-keygen seam.

Non-goals (deliberate, KISS): no generic ledger/append framework; no unification of the
merge-back journal with the intake started/complete markers (crash-safety-critical,
recently reviewed, working); gate/forward token consumption keeps per-ledger semantics;
no new append-only DB triggers for convention-only tables.

## 5. Seams and registries

Pattern: small Protocol + name→factory registry + config field — the shape
`data/providers` and `LedgerKind` already prove.

1. **Broker.** The 12 fine-grained Protocols in `contracts/types.py` stand. Add
   `execution/broker_factory.py` (name→factory, config-selected) replacing the ~6
   private `_alpaca_*_from_settings()` functions in `paper_cmd.py` / `live_cmd.py` /
   `execution/lane_exit.py`. `registry/transitions.py` stops constructing a broker
   inside the registry layer — construction is injected (as the merge-back saga already
   does). Extract a broker-neutral error leaf (`execution/errors.py` or
   `contracts`) so `execution/tick_clock.py` and `cli/errors.py` stop importing
   `alpaca_broker` (both self-documented as debt).
2. **Calendar.** One public `TradingCalendar` Protocol in `contracts`; delete the three
   private near-duplicate Protocols (`operator/loop_health.py`,
   `execution/fleet_health.py`, `registry/forward_promotion.py`). One factory honoring
   `settings.exchange` replaces the ~10 hardcoded `MarketCalendar()` constructions.
   **Non-goal:** the `1d = UTC-midnight` rail stays baked (schema validators,
   importers, corpactions) — un-baking it IS the multi-asset axis. This seam only
   localizes the future change.
3. **Experiment tracker.** Wire the existing-but-dead `ExperimentTracker` Protocol:
   `tracking/factory.py` selects MLflow or a no-op from settings; the ~5 direct
   `log_*` call sites take the Protocol. (Closes the PR#110 tracker-DI deferral;
   probe C's "second tracker" becomes one module + one registry entry.)
4. **Serving data provider.** `cli/_common.select_provider`'s if/else becomes the same
   registry pattern (probe C's versioned-dataset provider = module + entry).
5. **Knowledge sink.** One `KnowledgeSink` Protocol over the ~10 swallowed best-effort
   sync call sites; the Obsidian vault implementation is the sole impl (probe D's
   second consumer = second impl). Best-effort/swallowed semantics preserved.
6. **ML strategies (probe A).** Not designed now. Prep is incidental: `models/registry.py`
   loses its duplication excuse (§4), strategy loading stays pure.

Left alone (already extensible): ingest provider/importer registries, alerts
(injectable runner + `alert_cmd` config), paper venue `LedgerKind` map.

## 6. God-file carving

End-state rule: every `*_cmd.py` is thin — parse options → call one domain function →
`emit()`.

1. **`data/store.py`** → `data/store/` per-dataset modules (`bars`, `universe`,
   `delistings`, `fundamentals`, `news`, `identity`; streamed bars ingest its own
   module). `DataStore` becomes a facade over the three existing collaborators.
2. **`registry/store.py`** → one implementation module per `repository.py` Protocol
   (crud, transitions, approvals, breadth, holdout, gates, forward-gates, family-graph,
   returns; the factor ledger died in §3). `record_gate_with_fdr_and_maybe_promote`
   (the 290-line transaction script) moves to the promotion layer; the FDR safe subset
   (§3) dies here; the `store.py → research.gates` constants import is fixed.
   `repository.py` is NOT split; its value objects move to `registry/types.py`.
3. **`registry/db.py`** → declarative schema per bounded context + `connect` +
   `migrate` + `backfills`, preserving the idempotent-bootstrap contract exactly.
4. **CLI → domain** (completes the half-finished extractions):
   - `research_cmd.promote_task` → `registry/promote_run.py` (with the sweep-task body
     extraction, kills both `importlib` escapes; the holdout burn-on-peek /
     release-on-failure saga moves intact with its comments).
   - `paper_cmd._run_intake` → `registry/intake.py` (joins its own pure helpers).
   - `live_cmd._evaluate_book_loss_breaker` + `_build_book_exposure` →
     `risk/book_cycle.py` (joins `book_breaker`/`book_equity`/`book_limits`).
   - kill/resume + resume-all peak-rebase policy → `risk/peaks.py`.
   - `live_cmd._live_account_equity` raw HTTP → `execution/alpaca_broker.py`.
   - `promotion.py` family classification (≈lines 106–325) →
     `registry/family_assignment.py`.
   - `forward_promotion.py`: tick admissibility/evidence → `registry/forward_evidence.py`;
     `verify_forward_certificate` → `registry/live_certificate.py`. Preserve the
     preflight → guard → run_gate → reason structural symmetry with `promotion.py`.
   - `operator_cmd` payload helpers → `operator/driver_payload.py`; `_run_session`
     decision tree → `operator/session_runner.py` (emit callback injected).
5. **Paper/live unification (final carve).** The near-duplicate `_run_strategy_tick`
   (`paper_cmd` ≈856–1028 / `live_cmd` ≈163–313) and `run_all` cycles
   (`paper_cmd` ≈1134–1357 / `live_cmd` ≈438–729) merge into `algua/live/`
   (`tick.py`, `cycle.py`) parameterized by lane, with ONE shared breach-routing
   table (dark-feed → halt-no-flatten vs economic → trip+flatten). Deliberately
   different lane details become explicit parameters, not copies.
6. **`backtest/engine.py`** → `backtest/pit_view.py` (sidecar shape + as-of masking),
   `backtest/decision_path.py` (canonical/fast dual path + parity guard incl.
   `verify_signal_panel_parity`), `backtest/execution_model.py` (costs/fill-price);
   `simulate`+`run` remain a ~200-line orchestrator. `holdout_window` moves to a
   windowing home alongside walk-forward.
7. **`research/gates.py`** — `GateDecision.to_dict` → `research/gate_serialization.py`;
   the four advisory-check builders move onto the declarative `GateSpec` pattern the
   binding checks already use; the post-#335 re-export shim in `gates.py` is removed
   once call sites are updated (no compat cruft).

## 7. Enforcement and validation

- **Import-linter additions:** `primitives` leaf contract; cli→domain one-way layering;
  existing purity fences extended to new modules; the CLI independence contract loses
  its `importlib` workarounds.
- **Structural ratchet test** (pattern precedent: the #277 AST data-wall test): CI test
  asserting per-module LOC budgets with a shrink-only allowlist. New god-files cannot
  appear; carved files cannot regrow.
- **`docs/architecture.md`:** a one-page module map (what each package does, how to add
  a provider/broker/calendar/tracker/sink/strategy/command). CLAUDE.md updated to point
  at it; stale CLAUDE.md command references (deleted advisory commands) removed.
- **Acceptance:** dispatch a fresh agent on the MLflow backlog task; it must land as
  new module(s) + registration without editing core files, reading only
  `docs/architecture.md` + touched modules.

## 8. Stage sequence

Each stage ships via the review-gated pipeline; quality gates
(`pytest`, `ruff`, `mypy`, `lint-imports`) green on every PR; characterization tests
added first where a carve touches thin coverage.

| # | Stage | Depends on |
|---|---|---|
| 0 | Freeze ops: stop merge-back drain + research-loop launchers; clean tree | — |
| 1 | Kill-list deletions (~3 grouped PRs: shadow+monitoring; gc+family-audit+dormant-sweep+family_budget; factor layer+pbo) | 0 |
| 2 | `algua/primitives/` + migrate flock/atomic/retry sites; `registry/challenges.py` | 1 |
| 3 | `data/store.py` per-dataset carve (warm-up) | 2 |
| 4 | `registry/store.py` per-Protocol carve + `db.py` split + FDR safe-subset deletion | 2 |
| 5 | Seam factories: broker + neutral errors, calendar, tracker, serving provider, KnowledgeSink | 2 |
| 6 | CLI → domain extractions (promote_task first) | 4, 5 |
| 7 | `engine.py` + `gates.py` cuts | 2 |
| 8 | Paper/live tick + cycle unification | 6 |
| 9 | Ratchets + `docs/architecture.md` + CLAUDE.md refresh + MLflow acceptance probe | all |

## 9. Risks and mitigations

- **Regression in safety-critical orderings** — comments move verbatim; the five named
  orderings get characterization tests before their code moves; multi-model review on
  every PR.
- **Schema changes** — only deletions (kill-list tables) and none required by carving;
  `migrate()`'s idempotent-bootstrap contract preserved exactly.
- **Scope creep into deferred axes** — the §5 non-goals (UTC-midnight rail, ML
  pipeline) are binding; any discovered need upgrades the path explicitly rather than
  sneaking in.
- **Concurrent-session interference** — ops frozen (stage 0); the standing
  git-add-scoped / branch discipline applies.
