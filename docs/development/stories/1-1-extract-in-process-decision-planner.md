# Story 1.1: Extract the in-process decision planner

Status: ready-for-dev

Prepared: 2026-09-24. Baseline: `c4e8c8f409a8233cb5bc2a1ffe4f9ddba6bd8875`.
Epic: 1. Requirements: FR2–FR3; enabling work for FR4–FR7, not their completion.
Constraints: NFR2–NFR6. Upstream: #661, artifact-freeze design slice 2.

## Story

As Algua's operator,
I want a versioned, in-process decision planner separated from broker and registry effects,
so that immutable execution can be introduced without changing strategy behavior or giving the
planner operational authority.

## Scope and authority

This is a behavior-preserving extraction, not a deployed artifact or a new execution mode.
Retain the current supervisor and its staged checks. The mandatory extraction is the shared
`decide`/`build_intents` computation behind a small explicit surface. Extract value-only preparation
only where that seam requires it; do not eagerly read operational inputs to manufacture a single
all-in-one call. Staged pure helpers are acceptable.

No database/schema changes, deployment records, subprocess/IPC, artifact packaging, environment
provisioning, migration command, hourly execution, approval-policy/hashing-algorithm changes or
live activation. Incidental identity changes must be reported as required by AC7.
No new dependency or lockfile update. Do not change thresholds, error policy or permission rules.
Any necessary safety-invariant change is a separately reviewed proposal, not part of this story.

The complete future frozen boundary additionally covers strategy loading, configuration,
decision timing and per-strategy risk semantics. This first in-process seam does not claim to
freeze them or sandbox arbitrary authored strategy code. Explicitly record any preparation/risk
logic still supervisor-side so the frozen-execution story cannot mistake this for completeness.

## Acceptance criteria

1. **Shared decision computation.** Given equivalent strategy state, universe-only closed bars,
   current weights and a decision timestamp, when the planner is called, then it returns the
   same validated weights and deterministically ordered `OrderIntent` values as today's
   `paper_loop.decide`/`build_intents`. Both `run_tick` and simulated paper operation use the
   extracted implementation; do not retain a second production decision algorithm.
2. **Explicit in-process contract.** Given the planner entry point, when called, then its inputs
   and result are typed and its supported protocol version is explicit. An unsupported version
   is rejected before strategy computation and before cancellation/submission. Reuse existing
   `OrderIntent` and strategy interfaces; do not invent a serialization framework.
3. **No operational authority.** Given planner-owned modules, when their dependencies and calls
   are inspected/tested, then they do not acquire provider/broker data, read settings or the clock,
   open the registry, invoke hooks or submit/cancel orders. Receive captured values only (plus the
   already-loaded strategy); any required calendar choice comes from the supervisor. A loaded
   strategy is executable code, not a security sandbox: preserve the documented threat model.
4. **Preparation and early-return parity.** Given flat/held warm-up, empty history, inherited
   holdings or partial-session rows, when a tick runs, then decision eligibility, input history,
   snapshot call count, returned metadata and exceptions match the baseline. Held positions are
   checked before early returns; held symbols do not inflate universe warm-up history. Non-daily
   timeframes remain rejected before provider or venue access.
5. **Risk and reconciliation parity.** Given stale/missing/nonfinite/nonpositive marks, invalid
   sizing equity, drawdown, venue discrepancies or excessive realized gross, when processed,
   then existing breach kinds and ordering remain unchanged and no order is sent. Preserve NAV
   as distinct from sizing equity, held-warming drawdown/peak, and `None` versus empty venue belief.
6. **Effect-order parity.** Given a valid decision or a halt arriving during execution, when the
   supervisor runs, then cancellation, halt checks before/after cancel and before each order,
   client IDs, before-submit callbacks, reservation, immediate accepted-order persistence and
   noop cleanup retain their current order and semantics. Sizing remains execution-owned.
7. **Identity and evidence honesty.** Given representative registered strategy configurations,
   when before/after first-party closure and artifact identities are inspected, then the review
   records any differences without rewriting stored approvals, ticks or qualification evidence.
   Parity tests do not authorize carrying evidence across changed identities. Existing identity
   algorithms and live/human gates remain untouched.
8. **Regression evidence.** Given the extraction, when characterization, focused and full gates
   run, then they pass without weaker assertions, type suppressions, import exemptions or raised
   module-size ceilings. Characterization checks values, exception kinds and event ordering,
   not only final target weights. CLI JSON and simulation's next-bar fill timing remain intact.

## Tasks / Subtasks

- [ ] Characterize the baseline before extracting (AC 1, 4–6, 8).
  - [ ] Add focused tests in `tests/test_planner_parity.py` for normal decisions, dropped symbols,
    empty/flat/held warm-up, divergent held-read versus later snapshot, stale marks and risk failures.
  - [ ] Record provider/snapshot/belief/cancel/submit/hook event traces; assert the exact ordering,
    permitted earlier reads and snapshot counts, and absence of cancellation, submission or
    downstream hooks after the failing stage. Prove these tests pass against the pre-refactor path.
  - [ ] Add failing tests for the new versioned surface and authority boundary (AC 2–3).
- [ ] Extract the shared decision surface (AC 1–3).
  - [ ] Create `algua/live/planner.py`; move actual `decide` and `build_intents` computation out of
    `paper_loop.py`, not a wrapper importing its `SimBroker` dependency.
  - [ ] Define small typed request/result values and a protocol-version check. Use existing
    types and pandas objects in-process; no promise of wire compatibility or deep immutability.
  - [ ] Extract value-only preparation into a focused sibling module only where needed; preserve
    staged supervisor calls. Do not pass `SizingSnapshot`, hooks or callback-based data getters.
  - [ ] Add `tests/test_planner.py`: normal/rejected weights, sorted intents, zero targets for
    removed holdings, unsupported version, equivalent-input repeatability and boundary checks.
- [ ] Wire both existing paths (AC 1, 4–6).
  - [ ] Update `live_loop.py` to call the planner after existing pre-decision checks. Keep
    acquisition, reconciliation, cancellation, sizing/submission and persistence supervisor-owned.
  - [ ] Update `paper_loop.py` to use the shared decision implementation; preserve public helper
    compatibility where callers rely on it and preserve simulation's next-bar fill logic.
  - [ ] Preserve existing `live_loop.decide` observation points where feasible. If tests must
    observe the new entry point, read those files completely and retain behavioral assertions.
- [ ] Validate the boundary and identity implications (AC 3, 7–8).
  - [ ] Add a structural dependency test in `tests/test_planner.py` covering planner-owned helper
    modules as well as the entry point; check transitive first-party dependencies for operational
    imports. Do not treat strategy callbacks as proof of sandbox isolation.
  - [ ] Inspect representative closure membership and before/after identities through the existing
    read-only computation API. Report results; never update the DB to hide identity drift.
  - [ ] Lower/remove the `live_loop.py` pin in `tests/test_module_size_ratchet.py` if needed after
    extraction; new source modules stay below 300 lines. Do not raise the current 418-line pin.
  - [ ] Run focused regressions and then the full gate below. Record actual commands/results.
  - [ ] Obtain independent code review and resolve findings before marking implemented/reviewed.
  - [ ] List timing, freshness and pre-decision risk logic still supervisor-side in completion
    notes, identifying what later frozen execution must cover without changing its semantics now.

## Dev Notes

### Current files and exact responsibilities

| File | Current state | Change and preservation requirement |
|---|---|---|
| `algua/live/live_loop.py` — UPDATE | `run_tick` interleaves input acquisition, mark checks, warm-up, snapshot valuation, reconciliation and effects | Replace decision computation with planner calls; preserve every operational stage and `TickResult` shape |
| `algua/live/paper_loop.py` — UPDATE | `decide` validates target weights; `build_intents` orders rebalance deltas; `run_paper` fills next bar using `SimBroker` | Move actual shared helpers, preserve compatible imports and simulation behavior |
| `tests/test_module_size_ratchet.py` — conditional UPDATE | Pins live loop at 418 lines; new modules must be under 300; stale pins fail | Only ratchet down/remove its pin as the extraction warrants |
| `algua/live/planner.py` — NEW | No planner surface exists | Small versioned decision boundary; no operational imports |
| `tests/test_planner.py`, `tests/test_planner_parity.py` — NEW | Existing suites cover many individual cases | Add direct contract checks and end-to-end characterization without duplicating production implementation |

Use a separate small leaf DTO/preparation module only if the implementation actually requires it.
Do not put a live-specific DTO in the frozen bar schema or enlarge `contracts/types.py` for convenience.

### Ordering that cannot move

1. Reject unsupported timeframe before any provider/venue call.
2. Read held positions, fetch `universe ∪ held`, sort and remove current/future-date bars.
3. Validate held marks before empty/warm-up returns or snapshot acquisition.
4. Derive decision timestamp and warm-up from universe-only history.
5. Acquire the snapshot only when held or past warm-up.
6. Validate sizing equity, check drawdown, invoke venue-belief reconciliation, check realized gross.
7. Return valuation and peak metadata when held but warming; flat warming does not read a snapshot.
8. Check consumed marks against snapshot holdings plus universe; then compute the decision.
9. Preserve pre-cancel, post-cancel and per-order halt checks and all submission/persistence hooks.

Do not merge the earlier held-position read with the later snapshot: they can differ legitimately.
The snapshot's sizing denominator is not necessarily the live-account NAV used for drawdown.
Latest close and timestamp must come from the same row; `groupby.last()` can silently backfill NaN.
An empty venue belief is a real reconciliation input; `None` means the hook is absent.

`assert_marks_usable` currently calls settings-backed `get_calendar()`. Keep its operational
calendar selection supervisor-side; moving it unchanged would make the planner implicitly depend
on current settings. This story does not silently repair the pre-existing calendar-purity issue.
`execution/live_sizing.py` imports ledger/SQLite machinery: use captured fields, not an import of
its `SizingSnapshot` type into the pure surface.

Existing tests monkeypatch `live_loop.decide`. Compatibility must preserve their ability to
assert universe-only inputs; deleting such checks is not a valid extraction.

### Identity and release caveat

`registry/approvals.py::_merged_closure_for` roots identity in strategy signal, construction and
overlays; it does not establish universal coverage of the live-loop/planner orchestration.
Moving helpers may change some closures even with equivalent outputs. Neither unchanged hashes
nor comprehensive planner hash coverage may be assumed. Do not expand/narrow this algorithm in
the refactor. Deployment-manifest binding and actual isolation belong to later reviewed stories.

No existing deployed process should be restarted as part of story preparation or implementation.
A merge is not a live release. Use the existing reviewed release process; if runtime rollback is
needed, stop new exposure and follow the current recovery procedure rather than editing identity
records or transplanting evidence. This story has no migration to reverse.

### Verification

Keep the repository's Python pin and locked dependencies. This refactor introduces no external
API or technology/version decision requiring a dependency upgrade or new integration research.

Focused regression command:

```bash
uv run pytest -q tests/test_planner.py tests/test_planner_parity.py tests/test_live_loop.py tests/test_paper_loop.py tests/test_lane_parity.py tests/test_live_sizing.py tests/test_module_size_ratchet.py
```

Full gate, sequentially (pytest creates temporary source fixtures):

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

### References

- [Canonical PRD](../../PRD.md), §§5–7, 10, 15, 24–26.
- [Current architecture](../../architecture.md) and [reconciliation](../../vision-reconciliation.md).
- [Approved epic outcomes and requirements](../epics.md).
- [Artifact-freeze design](../../superpowers/specs/2026-09-22-artifact-freeze-design.md), seam,
  frozen-set table, strict fix policy and decomposition slice 2.
- [Issue #661](https://github.com/Lior-Nis/algua/issues/661), parent implementation direction.
- Repository `AGENTS.md`, `CLAUDE.md`, `docs/agent/operating.md` and frozen
  `docs/contracts/bar-schema.md` remain binding during implementation.

## Dev Agent Record

### Agent Model Used

To be recorded by the implementing agent.

### Debug Log References

No implementation run yet. Preparation included a separate read-only planner-boundary review.

### Completion Notes List

- Owner approved the artifact-freeze-first scope, both epic outcomes and preparation of this story.
- Story context prepared using BMAD create-story; current code and regression hazards inspected.
- `ready-for-dev` means this bounded refactor is specified, not that it is implemented, deployed,
  independently code-reviewed or a full-epic implementation-readiness assessment has passed.
- No prior story in this epic. Later deployment, migration and live-policy stories remain unprepared.

### File List

Implementation changes: none. The paths above are planned, not a claim of completed edits.
