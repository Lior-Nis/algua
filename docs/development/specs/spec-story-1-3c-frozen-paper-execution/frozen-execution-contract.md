# Story 1.3c frozen execution contract

Normative companion to [SPEC.md](SPEC.md). Terms: *supervisor* is the current `algua` process
running `paper trade-tick`/`run-all`/`intake`; *child* is one fresh planner process; *tenant* is one
strategy with an active deployment; *content* is a Story 1.3b bundle plus environment; the
*recorded config* is the descriptor's `resolved_config` object exactly as stored.

## 1. Admission (CAP-1)

- `run_intake` admits every new candidate as `frozen`; working-tree admission and
  `prepare_working_tree_deployment` are removed. `intake_candidate_to_paper` refuses a non-frozen
  descriptor. Existing `working_tree` deployments keep their descriptors and tick path unchanged.
- Per candidate, outside any write transaction: run the Story 1.3b preparation (`qualify → export →
  publish bundle/environment → record descriptor`) for the current clean `HEAD`, then verify the
  recorded descriptor offline with the Story 1.3b verifier. Only then call
  `intake_candidate_to_paper` with that exact descriptor; its existing single `BEGIN IMMEDIATE`
  transaction creates the deployment, allocation and `paper` transition and byte-verifies the
  descriptor row. `run_intake` receives prepare-and-verify, `repo_root` and `store_root` as
  injectable parameters so its tests need neither uv nor Git.
- Outcomes, reported per candidate in the intake JSON; refusals appear under
  `refused: [{"strategy": ..., "code": ...}]`:

| Condition | Effect on this candidate | Remaining queue |
|---|---|---|
| Admitted | deployment + allocation + `paper` | continue |
| `frozen_environment_unavailable` (retryable, 1.3b) | none; stays `candidate`; refused entry | stop, remain queued |
| Any other preparation/verification refusal (1.3b codes) | none; stays `candidate`; refused entry | continue |
| Existing `CountCapReached` / `AllocationError` / `TransitionError` | unchanged semantics | unchanged |

- `paper merge-back` reaches intake through `run_intake` and inherits this behavior; a refused
  candidate is reported by merge-back as today's `promoted_queued` and re-prepared on the next
  intake (environment objects are reused once published).

## 2. Tenant resolution and the supervisor view (CAP-2, CAP-4)

- One registry function, `resolve_paper_tenant(conn, name)`, replaces every checkout strategy load
  on the paper tick path: `load_gated_strategy` in `trade-tick`, `prepare_paper_book` and the
  `run-all` tenant loop, and the strategy read in `build_cycle_plan`. For `working_tree` and the
  legacy cohort it returns today's `LoadedStrategy` path unchanged. For `frozen` it parses the
  recorded descriptor with the Story 1.3b parser, never recomputes identity from the checkout, and
  returns a `FrozenTenant`: deployment/artifact ids, descriptor, content locators and the view.
- The view (`FrozenStrategyView`) exposes exactly what the supervisor reads today: `name`,
  `universe` (the gate universe), `config` and `execution`. It is decoded strictly from the
  recorded config: exact JSON types (no boolean as a number), `CapacityLimit` and
  `ExecutionContract` built through their constructors, then `StrategyConfig`; the decoded config
  must dump back to exactly the recorded object and hash to the descriptor's `config_hash`. The gate
  universe is overlaid afterwards, as `prepare_paper_runtime` does today. Any decoding failure is
  `frozen_content_unsupported`. Ticks are stamped with the descriptor's code/config/dependency
  hashes.
- `prepare_paper_book` verifies each distinct bundle and environment offline with the Story 1.3b
  verifier before planning, refresh and reconciliation, caching the verdict per digest for the rest
  of the process. There is no lease: Phase 1 never removes published content; any future garbage
  collection must add one. A shared environment's failure fails each tenant on it individually.
- The live lane refuses frozen rows inside `resolve_tick` with `frozen_live_unsupported`;
  `cli/live_cmd.py` is not changed. `paper run NAME` (checkout replay) refuses a frozen deployment
  with the same code.
- Operational note: frozen environments bind the host's base interpreter (today `/usr/bin/python3`
  3.12). A patch upgrade fails every frozen tenant closed with `frozen_content_unavailable`;
  recovery is re-admission into a new epoch by the controlled-migration story.

## 3. The planner port (CAP-2)

- `TickHooks` gains `planner`: an object with `phase_a(early)`, `closed_bars(early)` and
  `phase_b(late)`. When unset, `run_tick` uses an in-process adapter that makes exactly today's
  calls (`phase_a`, `phase_a_closed_bars`, `phase_b`) on the `LoadedStrategy`, so working-tree,
  legacy and live ticks are unchanged. `run_tick` keeps its three-step venue-belief handshake; it is
  refactored only enough to call the port, without raising its size pin.
- The frozen port (supervisor side of the dispatcher) implements the same three methods:
  - `phase_a` launches one child for Phase A.
  - `closed_bars` computes closed bars on the supervisor with the pure helper extracted from
    `prepare_early` (behavior unchanged) and checks that their decision timestamp equals the
    child's `snapshot_required.decision_ts`; a mismatch is `frozen_result_invalid`.
  - `phase_b` answers a *pending* venue belief with `VenueBeliefRequired` **without launching a
    child** (Phase B would return it anyway after the same drawdown check that the next call
    repeats), and launches one child for a resolved belief.
- So a decision tick launches at most two children, never concurrently, and none is alive while the
  supervisor acquires late values. Phase A and Phase B share the tick's `request_id` (the Phase A
  binding includes it).

## 4. Invocation directory and wire files

- Per child, the supervisor creates a private directory under `<data_dir>/frozen/invocations/`
  (mode `0700`), writes `request.json` and `bars.arrow`, fsyncs them, then seals files `0444` and the
  directory `0555` before launch. The child accepts no output files; stdout is the only result
  channel. The supervisor removes the directory after the child is reaped. A crash may leave an
  inert directory holding only bounded input; no sweep is added.
- `request.json` is canonical JSON (the Story 1.3b encoder: NFC, sorted keys, compact,
  `allow_nan=False`) of exactly these fields; unknown or missing fields are rejected by the child:

```text
wire                {"name": "frozen-planner", "version": 1}
phase               "a" | "b"
request_id          32 lowercase hex, shared by both phases of one tick
strategy_name       str
deployment_id, artifact_id                           int
manifest_digest, bundle_digest, environment_digest   64 lowercase hex
early               {boundary_version: 1, config_hash, resolved_config (the recorded config, its own
                     universe untouched), now, timeframe: "1d", calendar_code, early_positions,
                     gate_universe, max_drawdown, bars: {file: "bars.arrow", rows, bars_digest}}
late                null for phase "a"; for phase "b": {phase_a_binding, captured}
```

- The gate universe travels only in `gate_universe`. The child serializes `resolved_config` with
  the planner's own canonical form into `EarlyPlannerInput.resolved_config_json`, so Phase A's
  existing universe re-hash against `config_hash` and its gate-universe checks run exactly as in
  process.
- Encodings: every float is its exact `float.hex()` string (NaN/inf in captured or marked values
  cross losslessly; JSON carries no non-finite number); timestamps are UTC ISO-8601 with
  microseconds and `+00:00`; symbol/value mappings are lists of `[symbol, value]` pairs sorted by
  symbol with unique symbols; `gate_universe` keeps its given order; `captured` carries every
  `CapturedStrategyState` field, with `venue_belief` as `{"kind": "disabled"}` or
  `{"kind": "enabled", "quantities": [...]}` (a pending belief never crosses the wire).
- `bars.arrow` is an uncompressed Arrow IPC file with exactly the columns `timestamp`
  (`timestamp[ns, tz=UTC]`), `symbol` (`string`), `open`, `high`, `low`, `close`, `adj_close`,
  `volume` (`float64`), in that order, no nulls, rows in the frame's order. Float arrays are built
  directly from NumPy values, never through `Table.from_pandas`, which turns NaN into null. The
  decoder rebuilds the `timestamp`-named UTC index and must reproduce the request's `bars_digest`
  (the Story 1.3a logical digest). Pickle and object payloads are never produced or accepted.
- The canonical JSON encoder and the frozen-wire name/version constants move from
  `algua/registry/artifact_contract.py` to one pure `algua/contracts` module that both the registry
  and the child import; there are no copies. Existing importers are updated, and the canonical bytes
  stay pinned by Story 1.3b's golden vectors.

## 5. Launch and the child

- Before launch, the supervisor checks that the bundle holds `_algua/protocol.json` naming
  `frozen-planner` version 1 and planner boundary version 1, and holds
  `algua/live/frozen_child.py`; otherwise `frozen_content_unsupported`.
- argv, `shell=False`: `[<env>/bin/python, "-I", "-B", "-c", BOOTSTRAP, <bundle_root>,
  <invocation_dir>]`. `BOOTSTRAP` is a fixed constant that prepends `<bundle_root>` to `sys.path`,
  imports `algua.live.frozen_child` and calls `main(invocation_dir)`. The entry-point module name,
  `main` and the exit codes below are part of wire version 1, so later supervisors can run older
  bundles. `-I` excludes the user site, `PYTHON*` variables and the working directory from import
  roots; ordinary site processing loads only the environment's own site-packages (Story 1.3b).
- `cwd` is the bundle root; stdin is `/dev/null`; descriptors other than stdout/stderr are closed;
  the child starts in a new session (its own process group).
- The environment is a replacement, never a filtered copy: `PATH=<env>/bin`, `HOME=/nonexistent`,
  `LANG=C.UTF-8`, `LC_ALL=C.UTF-8`, `TZ=UTC`, nothing else.
- The child loads the strategy with the bundle's own `load_tradable_strategy`, requires the loaded
  `CONFIG` to dump to the request's `resolved_config` and the request's digests and strategy name to
  match its bundle, runs the phase and writes one JSON document to stdout. Before writing, it checks
  that every loaded `algua.*` module's file lies under `<bundle_root>`.
- Exit codes: `0` result on stdout; `3` the child refused its bundle, protocol, identity or module
  origin (`frozen_content_unsupported`); anything else, or a signal, is `frozen_exit_abnormal`.

## 6. Limits (protected constants, wire version 1)

| Bound | Value | On violation |
|---|---:|---|
| Timeout per phase | 60 s | `frozen_timeout` |
| `request.json` | 256 KiB | `frozen_request_too_large`, before launch |
| `bars.arrow` | 256 MiB | `frozen_request_too_large`, before launch |
| Stdout | 1 MiB | `frozen_output_exceeded` |
| Stderr capture | 64 KiB | truncated, flagged, never a failure by itself |
| JSON nesting | 16 levels | `frozen_result_invalid` |
| JSON collection size | 10,000 elements | `frozen_result_invalid` |
| Persisted sanitized diagnostic | 8 KiB | truncated |
| Grace before process-group SIGKILL | 2 s | — |

- Teardown order: on timeout or stdout overflow, `SIGTERM` the child's process group, wait up to
  the grace, then `SIGKILL` the group. After the child exits, observe it with `waitid(WNOWAIT)`,
  `SIGKILL` its process group while the PGID is still held, then reap, so no descendant survives
  and no reused PGID is signalled. A new stdlib-only primitive
  `algua/primitives/contained_process.py` provides this; `run_bounded` is unchanged.

## 7. Result validation

Stdout must be exactly one UTF-8 JSON document within the limits: no duplicate keys, no trailing
data, no non-finite number, no boolean where a number is expected. It must echo `wire`, `phase` and
`request_id`, and carry one `result` whose `kind` is permitted for the phase:

| Phase | Permitted `result.kind` | Supervisor mapping |
|---|---|---|
| a | `early_no_decision` (`no_bars`/`warming`, `state`) | `EarlyNoDecision` |
| a | `snapshot_required` (`decision_ts`, `warming`, `phase_a_binding`) | `SnapshotRequired` |
| a, b | `risk_failure` (`risk_kind`, `detail`) | `PlannerRiskFailure` → existing `RiskBreach` handling |
| a, b | `planner_rejected` (`code`, `detail`) | tenant failure `frozen_planner_rejected` |
| b | `late_no_decision` (`warming`, `state`) | `LateNoDecision` |
| b | `decision` (`state`, `intents`) | `Decision` |

- `state` carries every `PlannerState` field with the §4 encodings. `intents` are
  `{symbol, side: "buy"|"sell", target_weight, decision_ts}`.
- `risk_kind` must be in `RISK_BREACH_KINDS`, a new named constant beside `DARK_FEED_KINDS` holding
  exactly the current vocabulary (`drawdown`, `gross_exposure`, `gross_exposure_realized`,
  `long_only`, `max_weight_per_symbol`, `non_finite_weight`, `non_positive_equity`,
  `out_of_universe`, `reconcile`, `stale_marks`, `unvaluable_marks`); dark-feed classification is
  derived by the supervisor. `planner_rejected.code` must be in `PLANNER_FAILURE_CODES`, a new named
  constant holding the Story 1.3a planner input and binding codes.
- Child-supplied `detail` is reduced to printable ASCII without control characters and truncated to
  8 KiB before it reaches a kill-switch reason, audit row or JSON output.
- For `decision`, the supervisor checks: every timestamp equals Phase A's `decision_ts`; symbols are
  unique and inside the gate universe or current holdings; intents equal `build_intents`
  recomputed from the decision's target weights and the current weights the supervisor derives
  from its own captured values with the planner's formula (market value / sizing equity); then it
  reruns `validate_decision_weights` with the view's execution contract. Any mismatch is
  `frozen_result_invalid`.
- `check_mark_freshness` lists offenders in sorted order so breach text is identical across fresh
  processes (a message-order change only, affecting paper and live alike).

## 8. Failure taxonomy and classification (CAP-3)

| Code | Cause |
|---|---|
| `frozen_content_unavailable` | descriptor/bundle/environment missing or failing offline verification |
| `frozen_content_unsupported` | protocol/wire/boundary version, missing entry point, child exit `3`, config the strict decoder rejects |
| `frozen_request_too_large` | request metadata or bars over their bound |
| `frozen_launch_failed` | the interpreter could not be started |
| `frozen_timeout` | the phase exceeded 60 s |
| `frozen_exit_abnormal` | any other nonzero exit, or termination by signal |
| `frozen_output_exceeded` | stdout over 1 MiB |
| `frozen_result_invalid` | any §7 violation, or a closed-bar timestamp mismatch |
| `frozen_planner_rejected` | the child's planner refused its input or binding |
| `frozen_live_unsupported` | a frozen deployment reached the live lane or `paper run` |

- All are raised as one exception type, `FrozenTenantFailure(code, deployment_id, diagnostic)`,
  defined with the dispatcher. The diagnostic holds exit status, signal, truncation flags and a
  bounded sanitized stderr head; raw stderr is never persisted or interpreted.
- It is the only exception isolated from inside `run_tick`, and the port raises it only from phase
  dispatch, which precedes any cancel, submit or downstream hook. `_run_paper_strategy_tick` turns
  it into a `StrategySetupError` carrying the code and `deployment_id`. `StrategySetupError` and the
  CLI error registry read the `code` attribute of `FrozenTenantFailure` instead of its class name.
- `trade-tick` exits nonzero with `{"ok": false, "code": <code>, ...}`. `run-all` appends
  `{"ok": false, "strategy": <name>, "kind": "setup_error", "error": <code>, "deployment_id": <id>}`
  (today's shape plus `deployment_id`; Story 1.3d relies on it), audits it as today, and continues
  valid siblings.
- `KeyboardInterrupt`, `SystemExit`, SQLite errors, global halt and account-wide reconciliation or
  book-risk failures remain systemic and are never caught as tenant failures.
- A child `risk_failure` keeps today's `RiskBreach` semantics (kill switch, dark-feed global halt,
  scoped flatten), so breach effect traces match the in-process planner. As in process, a defective
  immutable artifact that keeps reporting stale marks re-halts the book each cycle until retired.

## 9. Promotion and live refusal (CAP-5)

- `paper promote NAME` and the forward gate check the active deployment first, before actor
  authentication or any checkout identity hashing. A `frozen` deployment exits nonzero with
  `frozen_qualification_pending` before gate evaluation, token minting or stage change.
- `frozen_qualification_pending` and the §8 codes are registered as stable, non-retryable codes.

## 10. Placement, protection, cost and evidence

- New modules: `algua/primitives/contained_process.py`, `algua/live/frozen_wire.py` (schemas,
  codec, limits), `algua/live/frozen_dispatch.py` (the frozen port and `FrozenTenantFailure`),
  `algua/live/frozen_child.py` (child entry) and `algua/registry/frozen_runtime.py`
  (`resolve_paper_tenant`, the strict decoder, the verification cache). A module that would pass the
  300-line ratchet floor is split by concern (for example `frozen_wire_json.py`,
  `frozen_wire_arrow.py`). Every new module is added to root `CODEOWNERS` and the repository-hygiene
  protected set.
- An import-linter contract forbids `algua.live.frozen_child` and the `frozen_wire*` modules from
  importing `algua.registry`, `algua.data`, `algua.cli`, `algua.execution` or `algua.operator`.
  (Strategy loading legitimately reaches `algua.config` and `algua.models`.)
- Size-pinned modules do not grow: `cli/paper_cmd.py` and `live/live_loop.py` changes are paid for
  by carving; `cli/live_cmd.py` is untouched.
- Cost: an admission provisions about 1 GB once per distinct environment key; each supervisor
  process re-verifies each distinct environment once. A hung child costs at most
  2 × (60 s + 2 s) per tick. These are recorded, not new limits.
- Parity harness: tests build a bundle by copying the working tree's `algua/` plus fixture strategy
  modules and the generated `_algua/` files, launch the repository interpreter with the exact §5
  argv and environment, and rely on the child's module-origin check (a dev editable install must not
  leak the checkout). Parity means equal canonical encodings of every result plus equal effect
  traces against the in-process planner on the normal, early and breach fixtures, including one
  breach case per risk kind the fixtures exercise. One opt-in test runs real prepare → verify →
  dispatch against a provisioned environment.
- Tests also cover: codec round trips (NaN/±inf/−0, empty frames, order, digests) and every
  rejection; argv/environment and teardown (timeout, signal, overflow, stray grandchild);
  each §8 code's zero-effect isolation in `trade-tick` and `run-all`; mutable-checkout independence
  and restart resolution; intake atomicity and refusals; the promotion, live and `paper run`
  refusals; and unchanged working-tree and legacy traces, pinned before refactoring.
- Out of scope: invocation evidence records, frozen forward evaluation and removing the promotion
  block (Story 1.3d); `snapshot_id` stays on the tick row and in 1.3d's attempt record, not in the
  v1 wire; migration (controlled-migration story).
