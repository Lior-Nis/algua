---
stepsCompleted:
  - step-01-document-discovery
  - step-02-prd-analysis
  - step-03-epic-coverage-validation
  - step-04-ux-alignment
  - step-05-epic-quality-review
  - step-06-final-assessment
filesIncluded:
  - docs/development/specs/spec-story-1-3c-frozen-paper-execution/ (SPEC, companion, decision log)
  - docs/development/stories/1-3c-execute-frozen-planners-in-paper.md
  - docs/development/stories/1-3-*.md and 1-3d-*.md (parent, next story)
  - docs/development/sprint-change-proposal-2026-09-25.md, docs/development/epics.md
  - docs/superpowers/specs/2026-09-22-artifact-freeze-design.md
  - docs/development/specs/spec-story-1-3b-artifact-environment-contract/
assessmentScope: Story 1.3c
baseline: 216c8ec
---

# Implementation Readiness Assessment Report

**Date:** 2026-09-30 · **Project:** Algua — Story 1.3c · **Assessor:** independent reviewer (Claude),
BMAD method, non-interactive defaults. Code claims checked at `216c8ec`; experiments ran in scratch.

**Discovery:** inputs as listed above, with no duplicates. UX does not apply (JSON of existing
commands). PRD/epic coverage is as in the 1.3b report (FR4, FR6, FR9–FR10, NFR1–NFR8).

## Requirements Traceability

| AC | Contract | Status | AC | Contract | Status |
|---|---|---|---|---|---|
| 1 atomic intake | §1 | covered (m3) | 7 teardown | §6 | covered (m1) |
| 2 fresh phases | §3 | covered (M3) | 8 tenant failure | §8 | plumbing open (M5) |
| 3 transport | §4 | covered (M1) | 9 singular supervisor | §10 lint | unsatisfiable (M2) |
| 4 launch | §5 | covered (M6) | 10 parity | §10 tests | harness open (M7) |
| 5 read-only dir | §4, SPEC | covered | 11 promotion refusal | §9 | covered (M4 order) |
| 6 validation | §7 | covered (m4) | 12 repo contracts | §10 | pins bite (M3, m2) |

No clause contradicts an AC, the sprint change proposal or 1.3d. The contract departs from the
parent in three acceptable ways: no lease, no invocation-directory crash sweep, and no `snapshot_id`
in the v1 request (m5). 1.3d AC1/AC8 need byte digests (Arrow bytes are deterministic) and a stable
failure JSON (open, M5).

## Code Feasibility Verification

| Claim | Verdict | Evidence |
|---|---|---|
| `run_intake` → `intake_candidate_to_paper`, one `BEGIN IMMEDIATE`, byte-verified descriptor | Holds | `registry/intake.py:196-216`; `store/deployment.py:131-198`; `store/artifacts.py:51-62` |
| 1.3b `prepare_frozen_artifact(repo,name,repo_root,store_root)`, `verify_frozen_artifact(repo,digest,store_root)` | Holds | `artifact_preparation.py:91-193`; `artifact_verification.py:26-46` |
| View = `StrategyConfig` validated from recorded `resolved_config` | **Fails** | B1 |
| Gate-universe overlay as in `prepare_paper_runtime` | Holds | `registry/paper_runtime.py:39-50` |
| Closed-bar logic extractable as a pure helper | Holds | `live/planner_early.py:222-236` (needs only `warmup_bars`) |
| `build_intents` / `validate_decision_weights` reusable by the supervisor | Holds | `planner_decision.py:34-45` (sorted, pure); `risk/limits.py:209` |
| Belief resolved before Phase B keeps the outcome | Holds for paper | drawdown before belief `planner_late.py:141-146`; no paper belief hook `paper_cmd.py:722-737` |
| 11 `RiskBreach` kinds; `DARK_FEED_KINDS` | Holds | 11 literals found; `risk/limits.py:47` |
| Planner failure codes form a closed vocabulary | Partial | about 20 literals, no constant (m7) |
| `check_mark_freshness` text depends on hash seed | Holds | set iteration `planner_early.py:196,244`, `planner_late.py:179`; `risk/limits.py:193-205` |
| Arrow IPC (pyarrow 23.0.1) is lossless | Verified | `Table.from_pandas` turns NaN into null; `pa.array(…, from_pandas=False)`: 0 nulls, NaN/±inf/−0/subnormal bit-equal, re-encoded bytes identical, `bars_digest` reproduced, empty frame OK |
| `-I -B` (no `-S`) loads venv site-packages | Verified | `.venv` python: `isolated=1 no_site=0 safe_path=True`; `PYTHONPATH` and cwd absent; fresh `uv venv --relocatable` + BOOTSTRAP resolves `algua` from the bundle. 1.3b certified site startup (`planner_environment_startup.py:1-16`) though its probe runs `-I -S` |
| §10 lint contract satisfiable | **Fails** | M2 |
| Fits size pins | Tight | `live_loop.py` 412/418, `paper_cmd.py` 1409/1409, `live_cmd.py` 749/749 |

## Findings

### BLOCKER

**B1 — The supervisor view cannot be built for any strategy.** `StrategyConfig` rejects
`execution` given as a mapping (`strategies/base.py:90-103`: the #344 guard against dict coercion
bypassing the `ExecutionContract`/`CapacityLimit` `__post_init__` rails). The recorded config is
`model_dump(mode="json")` (`artifact_preparation.py:122-123`), so `execution` is always a dict:
21 of 21 loadable strategies were rejected. As written, every frozen tenant becomes
`frozen_content_unsupported`, and the implementer is left to design a protected safety decoder.
**Fix:** specify it. Use exact JSON types (no bool as a number), build `CapacityLimit(**…)` and
`ExecutionContract(**…)` through their constructors, then `StrategyConfig`. Require
`model_dump(mode="json") == resolved_config` and `config_hash` equal to the descriptor's; anything
else is `frozen_content_unsupported`. This round-tripped 21 of 21. Also name the view type, because
`run_tick` and the hooks expect a `LoadedStrategy` (M3).

### MAJOR

**M1 — Wrong object for `resolved_config`.** §4 ships it "gate universe overlaid". The planner needs
the *recorded* template-universe config: `_validate_early` re-hashes with
`resolved_data["universe"]` against the descriptor `config_hash` (`planner_early.py:116-137`), and
`config_hash` includes `universe` (`base.py:366-368`). An overlaid object therefore fails with
`strategy_identity_mismatch` for every tenant whose PIT gate universe differs from its `CONFIG`.
**Fix:** send the recorded object, or omit it and read it from the bundle. `gate_universe` carries the
overlay. The child sets `resolved_config_json` to the planner-canonical serialization of the recorded
object (`planner_binding.py:44-48`).

**M2 — The §10 lint contract is unsatisfiable.** The child must load strategies through
`load_tradable_strategy` for parity, but `strategies/loader.py:220 → algua.models →
models/registry.py:58 → algua.config.settings`. The import is lazy, yet a probe `lint-imports` run
breaks on exactly this chain. §4 also mandates "1.3b `canonical_json`" and a `frozen-planner` v1
check, but both `canonical_json` and `FROZEN_WIRE` live in `algua.registry.artifact_contract`
(`:12-16,67-71`), which the contract forbids. **Fix:** put the wire constants and the canonical
encoder in `frozen_wire`, pinned by a golden-bytes test against 1.3b. Then either add one justified
`ignore_imports` edge (`algua.models.registry -> algua.config.settings`), or check direct imports
only and add a runtime test that no `algua.{registry,config,data,execution,cli}` module is loaded
after both phases.

**M3 — The `run_tick` seam is unspecified in a pinned, shared module.** `run_tick` takes a
`LoadedStrategy` and calls `phase_a`/`phase_a_closed_bars`/`phase_b` directly, with the three-step
belief handshake (`live/live_loop.py:248-351`). It has 6 lines of headroom and is shared with live.
**Fix:** name the seam, for example a planner port on `TickHooks` whose in-process default keeps
today's calls byte-for-byte, with the frozen port in `frozen_dispatch`. State what `strategy` object
a frozen tenant passes (the B1 view), and carve `live_loop.py` rather than raising its pin.

**M4 — The working-tree coupling is not enumerated.** A frozen tenant reaches checkout code at:
`registry/gating.py:27` (`load_gated_strategy`, called from `paper_cmd.py:849`, `:994` and `:1132`);
`cli/lane_refresh.py:67` (`build_cycle_plan`, shared with live);
`registry/deployment_runtime.py:17-18` (`require_tick_deployment` runs
`verify_working_tree_manifest`, which rejects any frozen row at `registry/deployment.py:218-236`, and
`identity_loader=compute_artifact_hashes`); and `registry/forward_promotion.py:115-116`, which hashes
the checkout *before* reading the deployment and so fails before `frozen_qualification_pending` when
the module is gone. **Fix:** list these sites and give one routing function (for example
`frozen_runtime.resolve_tenant`) that returns the view plus descriptor identity for frozen tenants
and today's tuple otherwise, keeping the kill-switch and halt gates. Move the §9 check ahead of
`compute_artifact_hashes`. Make `resolve_tick` refuse frozen with `frozen_live_unsupported` and give
paper a frozen-aware sibling, so `live_cmd.py:158`, which is at its pin, stays untouched.

**M5 — How the failure code reaches the JSON is open.** The error registry is keyed by type
(`cli/errors.py:11-90`), and `StrategySetupError.code` is the cause's class name
(`cli/_common.py:57-66`). "One failure type carrying the code" would therefore surface as
`FrozenTenantFailure`. Separately, anything escaping `run_tick` aborts the cycle
(`paper_cmd.py:686-688,751`), yet child faults arise inside it. **Fix:** make `error_code()` and
`StrategySetupError` honor a `code` attribute on that one type. Declare it the only exception
isolated from inside `run_tick`, and only before cancel. Fix the `run-all` entry shape now, because
1.3d AC8 freezes it: `{"ok": false, "strategy", "kind": "frozen_tenant_failure", "error": code,
"deployment_id"}`.

**M6 — `frozen_content_unsupported` is undetectable.** On a protocol, entry-point or identity
mismatch the child only exits nonzero (§5), and stderr is never interpreted (§8). Those cases become
`frozen_exit_abnormal`, contradicting §8. **Fix:** before launch, the supervisor reads
`_algua/protocol.json` from the verified bundle, checks wire/boundary versions, and confirms that
`algua/live/frozen_child.py` is in the inventory. Make the entry-point name and its exit semantics
normative wire-v1 surface, since future supervisors must call old bundles.

**M7 — The parity harness is undefined.** 1.3a fixtures are `SimpleNamespace` closures that cannot
cross processes (`tests/test_planner_parity.py:19-89`). Its breach cases use a belief hook
(`:150-160`), which frozen paper cannot reach and would reorder. `export_source` uses Git `HEAD`, so
an uncommitted `frozen_child` is missing. The dev `.venv` editable `.pth` puts the checkout on
`sys.path` under `-I -B` (observed); real provisioning is about 1 GB. **Fix:** write fixture
strategy modules into a test bundle copied from working-tree `algua/`; run the repo interpreter under
the exact §5 argv/env; stub the verified-content resolver; assert every loaded `algua.*` file is
under `bundle_root`; use no belief hook (reconcile N/A); add one opt-in real prepare→verify→dispatch.

### MINOR

- **m1** Order teardown as `waitid(WNOWAIT)` → `killpg` → reap (killing after a reap risks PGID reuse).
- **m2** `frozen_wire` and `frozen_dispatch` will likely exceed 300 lines. Allow protected
  `*_json`/`*_arrow`-style splits and extend the lint contract to them.
- **m3** Intake: name the refusal key (`refused: [{strategy, code}]`); inject prepare-and-verify
  plus `repo_root`/`store_root` into `run_intake` so its suites avoid uv and Git; decide whether
  `intake_candidate_to_paper` enforces `frozen`; delete `prepare_working_tree_deployment`. A refused
  candidate re-runs `uv sync` every intake (`artifact_preparation.py:146-166`) and merge-back reports
  it as `promoted_queued` (`operator/mergeback.py:365-372`).
- **m4** `request_id` is per tick and shared by A and B (the binding includes it,
  `planner_binding.py:111`); say so. Define parity as equality of canonical encodings plus the
  effect trace, because the wire sorts mappings while in-process state keeps broker order.
- **m5** Record that `snapshot_id` (parent AC6) stays on the tick row and in the 1.3d attempt record
  (freeze design line 99), not in the v1 wire.
- **m6** `paper run NAME` (`paper_cmd.py:242`) replays checkout code and can trip a frozen tenant's
  kill switch. Refuse it for frozen deployments.
- **m7** Add a named constant for the planner input/binding codes. **m8** Verify content in
  preflight (`prepare_paper_book`), before the plan, refresh and reconcile.

### NOTE

- **N1** The frozen interpreter is system `/usr/bin/python3.12` (3.12.3, `pyvenv.cfg home=/usr/bin`).
  A patch upgrade fails *all* frozen tenants closed until migration; recovery is re-intake with a
  new epoch. Add a runbook note or an apt hold.
- **N2** Each admission provisions about 1 GB, fsyncs it and hashes it twice, inside `operator.lock`
  under merge-back (drainer budget 3600 s). Each run-all re-hashes each distinct environment. At
  124 s per hung tenant, the worst case exceeds `algua-paper.service` `TimeoutStartSec=2400` at
  about 20 frozen tenants (capacity 64).
- **N3** A child `stale_marks` result engages the book-wide halt (`paper_cmd.py:764-772`), as in
  process; but a buggy immutable artifact re-halts the book every cycle until it is retired.
- **N4** The box checkout the operator runs is on `feat/story-1-3b-frozen-artifacts`. Merge-back
  fails closed there, and a manual `paper intake` would freeze unmerged runtime code. Develop only
  in the worktree.
- **N5** This worktree's `epics.md`, `sprint-status.yaml` and `README.md` already say 1.3c passed
  readiness; hold them until the conditions below are met. (Cold child imports: about 0.7 s.)

## Risk to Existing Behaviour

- **Legacy tenants** (all 18 on the box, `deployment_id: null`): the M3/M4 seams run through their
  tick, gating and plan paths, and sorting `check_mark_freshness` changes breach text in paper *and*
  live. Pin current traces before refactoring. Until a new admission, the change is otherwise inert.
  The live lane already refuses frozen rows (`DeploymentError`); only its code and placement change.
- **Merge-back → intake:** safe (clean `main` required; `assert_clean_head` is stricter; an outage
  gives the retryable `frozen_environment_unavailable` and `promoted_queued`). The cost is N2.

## Summary and Verdict

**Overall readiness: NOT READY.** One blocker (B1: the view mechanism fails for every strategy) and
seven cheap MAJOR amendments that otherwise force improvisation on protected, size-pinned trading
surfaces. All fixes are contract text; none changes scope, authority or the 1.3d split. Once B1 and
M1–M7 are applied, with MINORs folded in as convenient, the story can start test-first without a
full re-review. Begin with the codec, the B1 decoder and the M3 port, proving the in-process default
byte-identical. Readiness never authorizes deployment, trading or capital use.

## Resolution (2026-09-30)

All BLOCKER and MAJOR findings were applied to the normative companion
(`specs/spec-story-1-3c-frozen-paper-execution/frozen-execution-contract.md`) and recorded in its
decision log: B1 (§2 strict decoder, `FrozenStrategyView`), M1 (§4 recorded config on the wire),
M2 (§4 shared pure encoder, §10 import contract), M3 (§3 planner port), M4 (§2
`resolve_paper_tenant`, §9 refusal order), M5 (§8 `FrozenTenantFailure` and entry shape), M6 (§5
pre-launch checks and exit codes), M7 (§10 parity harness). MINOR items m1–m8 are folded into §§1–10;
NOTEs N1–N3 are recorded in §§2, 8 and 10. N4 is an operational action outside the story, and N5 is
resolved by this update. As this review specified, no further full review is required.

**Updated verdict: READY.**
