---
title: 'Research strategy board'
type: 'feature'
created: '2026-10-19'
status: 'done'
baseline_commit: '76ebfd486e062b93fe12b2facbc086a6c011789a'
context:
  - 'docs/PRD.md'
  - 'docs/architecture.md'
  - 'docs/superpowers/specs/2026-08-23-strategy-run-tracking-design.md'
  - 'docs/superpowers/specs/2026-08-27-monitor-human-facing-redesign-design.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Research currently ranks gate runs, so one strategy can occupy multiple rows while
registered strategies with no gate evidence disappear. The operator needs a complete bird's-eye
view of the strategy population without turning exploratory comparison into a promotion signal.

**Approach:** Replace the displayed ranked-run list with a compact, mobile-first strategy board:
exactly one row per registered strategy, joined to that strategy identity's latest gate run. Keep
the scatter and idea pool; expose a versioned scalar-only CLI/API projection and preserve the
existing read-only drill-down.

## Boundaries & Constraints

**Always:** Select latest gate evidence by `(created_at DESC, id DESC)`, never best metric; join by
`strategy_id`, not reusable name; retain strategies with no gate; keep nullable metrics nullable;
label OOS and walk-forward samples; keep lifecycle order as the default; preserve the JSON CLI seam,
read-only monitor, 390px no-overflow invariant, stage filtering, and holdout-series secrecy.

**Ask First:** Any use of board order in research promotion; any database/schema change, historical
backfill, lifecycle write, global shell-width change, or display of new evidence beyond existing
allowlisted run scalars.

**Never:** Compute a composite score, recommend a winner, attach a legacy name-only run to a
strategy, select the best run, coerce missing evidence to zero, expose a per-bar holdout vector, or
add actions that promote, retire, allocate, trade, or otherwise mutate system state.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Complete population | Registered strategies with zero, one, or many gate runs | One row each; latest identity-bound gate or `null` | Malformed/missing evidence stays unavailable |
| Reused name | Old name-only or different-identity run exists | Run is not attached | Render no gate evidence |
| Equal timestamps | Two gate runs share `created_at` | Higher run ID is latest | Deterministic across reads |
| Exploratory sort | User chooses an evidence metric | Same representative runs, nullable values last | Label comparison as exploratory |
| Empty registry | No registered strategies | Directed empty state | Successful empty payload |
| Backend failure | CLI read fails | Existing error/retry treatment | Never render a false empty board |

</frozen-after-approval>

## Code Map

- `algua/registry/strategy_board.py` -- new pure read projection; owns latest-per-identity semantics.
- `algua/cli/runs_cmd.py` -- thin `algua runs board` JSON command.
- `web/backend/main.py` -- cached `/api/strategy-board` CLI adapter.
- `web/frontend/src/types.ts` -- explicit versioned board response types.
- `web/frontend/src/components/StrategyBoard.tsx` -- filtering, comparison order, compact rows.
- `web/frontend/src/screens/Research.tsx` -- board placement and removal of `RunList`.
- `web/frontend/src/theme.css` -- board-specific responsive row marks within the existing shell.
- `tests/registry/`, `tests/test_cli_runs.py`, `web/backend/tests/`, frontend component tests -- contract and UI evidence.

## Tasks & Acceptance

**Execution:**
- [x] Add projection tests first, then implement a schema-versioned scalar payload containing all
  registered strategies and only their latest identity-bound gate row.
- [x] Expose the projection through `runs board` and `/api/strategy-board`; test JSON/error/cache
  behavior without adding a second data-access path.
- [x] Build `StrategyBoard`, wire it into Research, and cover filters, ordering, navigation, nulls,
  failures, accessibility, word budget, and 390px overflow.
- [x] Rehearse read-only against a copy of current production data and run all root/web gates.

**Acceptance Criteria:**
- Given multiple gate attempts for one strategy, when the board loads or changes comparison order,
  then exactly the latest attempt represents it and the chosen run ID never changes.
- Given registered strategies with and without evidence, when the projection is read, then each
  identity appears exactly once and missing evidence is explicit.
- Given a metric comparison, when rows are ordered, then finite values use the declared direction,
  nulls are last, ties are deterministic, and the UI states that sorting is exploratory only.
- Given a 390px viewport, when the populated Research route renders, then every row remains readable,
  keyboard navigable, and the document has no horizontal overflow.
- Given any board request, when inspecting CLI/API output, then no return series, raw gate decision,
  lifecycle action, or sensitive holdout payload is present.

## Spec Change Log

## Design Notes

Keep the existing true-black, Inter/IBM Plex Mono, Electric-for-data language. The distinctive mark
is one dense evidence row: name and stage first, then a short metric rail and degradation sparkline.
Do not introduce a wider desktop shell, card grid, hover-only content, or a second visual system.

## Verification

**Commands:**
- `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- `uv run --project web pytest web/backend/tests -q`
- `cd web/frontend && npm run check && npm run build`

## Implementation Evidence

**Status: implemented and verified.** The feature was exercised through its focused tests, the
complete repository gates, the production frontend build, the 390px browser check, and a read-only
rehearsal against an SQLite backup of the current operational registry.

Changed files:
- `algua/registry/strategy_board.py` (new) — pure `strategy_board_payload`; one SQL read joins each
  `strategies` row to its newest `kind='gate'` run by `strategy_id` (`created_at DESC, id DESC`);
  malformed stored values become `null`; lifecycle → case-insensitive name → id ordering.
- `algua/cli/runs_cmd.py` — thin `runs board` command (no new data-access path).
- `web/backend/main.py` — `GET /api/strategy-board` → `run_cli("runs", "board", ttl_s=60.0)`,
  no query parameters reach argv.
- `web/frontend/src/{types.ts,api.ts}` — `StrategyBoard*` types; `clearFetchCache()` test seam.
- `web/frontend/src/components/StrategyBoard.tsx` (new), `screens/Research.tsx`, `theme.css` —
  board replaces the obsolete `RunList`; the unused component and its dedicated tests were removed.
- `web/frontend/src/fixtures/{steady-state,index}.ts` — `BOARD` fixture (14 strategies, 3 with no
  gate, one nullable metric) served at `/api/strategy-board`.
- Tests: `tests/registry/test_strategy_board.py`, `tests/test_cli_runs.py`,
  `web/backend/tests/test_api.py`, `components/StrategyBoard.test.tsx`, `screens/Research*.test.tsx`,
  `fixtures/index.test.ts`, `invariants/word-budget.test.tsx`.

Verification evidence (2026-10-19):
- Root gate: `6290 passed, 5 skipped`; Ruff clean; mypy clean across 359 source files;
  29 import contracts kept, 0 broken.
- Web backend: `266 passed`.
- Frontend: `39` files / `184` tests passed; TypeScript clean; production build contains no
  fixtures; all five demo routes had `scrollWidth=390` at a 390px viewport.
- Research word budget was measured at 292 words against the populated fixture, then pinned to a
  146-word floor and a 322-word ceiling (measured + 10%); the 12-mark floor remains enforced.
- Operational-data rehearsal: a SQLite online backup of the current 22-strategy registry produced
  22 unique board rows, all with identity-bound gate evidence, and no forbidden series, raw
  decision, action, or recommendation keys. The source database was not modified.
- Comparison axes are higher-first; drawdown is stored negative, so higher means shallower.
- No database/schema change, backfill, lifecycle write, global shell-width change, or safety
  invariant change was made.

Adversarial review evidence:
- Three independent passes covered the diff, reachable edge cases, and acceptance/context rules.
- Patch findings fixed: stale-empty/refresh visibility, funnel preservation, long-name wrapping,
  accessible filter state/groups, payload validation, negative sample sizes, malformed recency,
  fixture parity, cache-reset races, realistic identity testing, and obsolete `RunList` removal.
- Rejected findings either contradicted the frozen `(created_at DESC, id DESC)` rule, requested
  new evidence-age UI outside this lean story, or challenged the already-documented non-temporal
  degradation profile. No intent gap, bad-spec loopback, deferred issue, or invariant change.

## Suggested Review Order

**Projection contract**

- Start with the identity-safe latest-gate projection and malformed-evidence handling.
  [`strategy_board.py:97`](../../../algua/registry/strategy_board.py#L97)

- Confirm the public CLI stays a thin JSON-emitting read seam.
  [`runs_cmd.py:56`](../../../algua/cli/runs_cmd.py#L56)

- Verify the monitor delegates through the existing cached CLI adapter.
  [`main.py:297`](../../../web/backend/main.py#L297)

**Human-facing board**

- Review fail-closed response validation before the component renders operator evidence.
  [`StrategyBoard.tsx:46`](../../../web/frontend/src/components/StrategyBoard.tsx#L46)

- Follow filtering, exploratory ordering, null treatment, and accessible controls.
  [`StrategyBoard.tsx:115`](../../../web/frontend/src/components/StrategyBoard.tsx#L115)

- Confirm Research preserves the funnel, scatter, and idea pool around the board.
  [`Research.tsx:133`](../../../web/frontend/src/screens/Research.tsx#L133)

- Check mobile wrapping and the dense two-line evidence-row treatment.
  [`theme.css:540`](../../../web/frontend/src/theme.css#L540)

**Supporting evidence**

- Inspect latest-not-best and identity-isolation projection tests.
  [`test_strategy_board.py:65`](../../../tests/registry/test_strategy_board.py#L65)

- Inspect stale-empty and malformed-contract UI regressions.
  [`StrategyBoard.test.tsx:175`](../../../web/frontend/src/components/StrategyBoard.test.tsx#L175)

- Confirm the versioned frontend contract mirrors the scalar-only response.
  [`types.ts:427`](../../../web/frontend/src/types.ts#L427)

- Review demo-mode parity with production latest-gate semantics.
  [`steady-state.ts:423`](../../../web/frontend/src/fixtures/steady-state.ts#L423)
