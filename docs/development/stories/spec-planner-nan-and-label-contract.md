---
title: 'Planner input contract: non-finite current weights and symbol labels'
type: 'correctness'
created: '2026-10-22'
status: 'in-review'
baseline_commit: 'a3c71fa'
context:
  - 'docs/development/stories/deferred-work.md'
  - 'docs/development/stories/1-1-extract-in-process-decision-planner.md'
  - 'docs/development/specs/spec-story-1-3c-frozen-paper-execution'
  - 'docs/agent/story-delivery.md'
  - 'AGENTS.md'
todoist:
  - 'https://app.todoist.com/app/task/6hfMw6W8mcVjQ2wp'  # NaN current-holding weight policy
  - 'https://app.todoist.com/app/task/6hfMw6Vvhv8rRgMp'  # symbol-label contract
---

<!--
Id/slug choice: these are two deferred correctness reviews from Story 1.1, not Epic 1/2 stories,
so this follows the repository's quick-spec convention for non-epic work
(`docs/development/stories/spec-<slug>.md`, as `spec-strategy-board.md` and
`spec-dashboard-decision-layer.md` do) and is not added to sprint-status.yaml. One combined spec
because both findings live in the same function and resolve to one shared input contract.
-->

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `build_intents` (`algua/live/planner_decision.py:34-45`) is the one function that
turns target weights and current holding weights into order intents. It trusts both inputs:

1. A **non-finite current weight** is silently mishandled. `abs(target - NaN) > WEIGHT_TOL` is
   `False`, so a NaN holding produces **no intent at all**, whatever the target (reproduced on
   `a3c71fa`: target 0.5 → `[]`; target 0.0, i.e. an exit → `[]`). A `+inf` holding produces a
   `SELL` intent toward a positive target. Either way the planner emits a decision that is not a
   function of the true book, without raising.
2. **Symbol labels have no type contract on the strategy side.** `check_finite_weights` rejects null
   and duplicate labels but not non-string ones, and `check_universe_membership` admits any
   zero-weight label. `build_intents` then calls `sorted(set(weights.index) | set(current_weights))`
   with no key, so a strategy returning `{1: 0.0, "AAPL": 0.5}` against string holdings raises a bare
   `TypeError` (reproduced). The same bare `sorted(...)` appears in the risk rails' *error messages*
   (`algua/risk/limits.py:121,135,178,192`): a mixed-label Series that trips the finite, cap or
   long-only rail raises `TypeError` instead of `RiskBreach`, so it escapes the
   breach → kill-switch routing (`paper_cmd` traps `RiskBreach` only) and is reported as a generic
   error instead.

Neither has a known production incident. Both pre-date the Story 1.1 extraction.

**Approach:** Define one **planner input contract**, enforced at the planner boundary and shared
with the risk rails rather than duplicated in the planner:

- every symbol label (strategy target-weight index **and** current-weight keys) is a non-empty
  `str`;
- every current weight is a finite `float`;
- a violation **fails closed** (raises before any intent is built). It never coerces to flat or
  skips the symbol.

All of this is already true for the frozen/operational path, which validates captured state
upstream. The change makes the in-process planner and the shared rails enforce the same contract
instead of relying on every caller to have done it.

## Boundaries & Constraints

**Always:** fail closed; keep `validate_decision_weights` the one weight-validation path for
backtest, paper and live (no planner-only drift); keep `DECISION_BREACH_KINDS` and the frozen
child's closed `RISK_BREACH_KINDS` vocabulary unchanged unless the owner picks option N2 below;
preserve the `t→t+1` rule, the live gate and the JSON command contract; test-first, with a
mutation check on every new guard.

**Ask first:** adding a new `RiskBreach` kind; changing the breach routing (kill-switch vs. account
halt) of any existing kind; any change to backtest results for a registered strategy.

**Never:** treat a NaN holding as 0.0 (that would buy the full target on top of an unknown real
position); silently drop or skip a symbol; widen what a frozen child may report.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Today (`a3c71fa`) | Expected after this spec |
|---|---|---|---|
| NaN holding, nonzero target | `current={"AAPL": nan}`, target 0.5 | `[]`, no error | Fails closed before intents |
| NaN holding, exit | `current={"AAPL": nan}`, target 0.0 | `[]`, the position is never exited | Fails closed before intents |
| +inf / -inf holding | `current={"AAPL": inf}` | `SELL` toward 0.5 | Fails closed before intents |
| Non-str holding key | `current={1: 0.1}` | TypeError in sort, or accepted | Fails closed (contract error) |
| Mixed target labels, zero weight | `{1: 0.0, "AAPL": 0.5}` | Rails pass; `build_intents` raises bare TypeError | `RiskBreach("non_finite_weight")`, "non-string symbol label" |
| Mixed labels that breach a rail | `{1: -0.1, "A": -0.1}`, long-only | Bare TypeError while rendering the message | `RiskBreach` of the right kind; message rendered with `key=str` |
| Empty-string label | `{"": 0.1}` | Accepted | `RiskBreach("non_finite_weight")` |
| Valid inputs | str labels, finite weights | Sorted delta intents | Byte-identical intents (parity) |

</frozen-after-approval>

## Evidence (baseline `a3c71fa`)

**Where the gap is**

- `algua/live/planner_decision.py:34-45`: `build_intents`. There is no finiteness or label-type check
  on `current_weights`, and an unkeyed `sorted(set(weights.index) | set(current_weights))` at
  line 39.
- `algua/live/planner_decision.py:48-54`: `canonical_weights` already sorts with `key=str`, so
  normalization tolerates mixed labels and the failure surfaces only later, in `build_intents`.
- `algua/live/planner_decision.py:57-69`: `plan()` validates target weights (via `canonical_weights`
  and `validate_decision_weights`) but passes `inputs.current_weights` through unchecked.
- `algua/risk/limits.py:107-139`: `check_finite_weights` rejects null labels (line 115), duplicates
  (line 120), bool/non-numeric dtype and non-finite values, but not non-`str` labels. Its own
  messages sort without `key=str` (lines 121, 135).
- `algua/risk/limits.py:142-167`: `check_universe_membership` admits zero-weight labels of any type
  and is the one rail that already renders with `key=str` (line 166).
- `algua/risk/limits.py:178,192`: the per-symbol cap and long-only messages sort without `key=str`.
- `algua/risk/limits.py:72-74`: `DECISION_BREACH_KINDS` already contains `non_finite_weight`, which
  already covers *label* defects (null, duplicate). That is the precedent for reusing it in label
  option L1.

**Callers and where `current_weights` comes from**

- `algua/live/paper_loop.py:96-100` (`run_paper`, the `paper run` sim replay): computes
  `qty * bar_closes.get(s, 0.0) / equity` with no finiteness check. A held symbol whose close is NaN
  at `t` would give a NaN weight. Today the equity guard at `paper_loop.py:88` fires first, because
  `SimBroker.equity` (`algua/execution/sim_broker.py:53-55`) also goes NaN, so the case is caught as
  `non_positive_equity` (fail-closed, but with a misleading kind). A held symbol missing from the
  `closes` columns is valued at `0.0` by both `.get` defaults: the holding becomes invisible rather
  than NaN.
- `algua/live/planner_late.py:84-109` (frozen/operational tick): already enforces the contract.
  `_numeric_mapping` (`:121-135`) requires non-empty `str` keys (NFC-normalized) and finite values
  through `_finite_number` (`:112-118`), then derives `current_weights` at `:96-98`.
- `algua/live/frozen_attempt.py:78-90`: the supervisor re-derives the child's intents with
  `build_intents` from the validated `verdict.current_weights`.
- `algua/live/frozen_wire_json.py:232-242`: wire `decode_pairs` requires `str` symbols, sorted and
  unique, with finite values.
- `algua/live/planner_binding.py:71,95`: bar symbols and planner mapping keys must be non-empty
  strings. `algua/data/schema.py:42-43`: bar symbols must be strings.
- `algua/live/live_loop.py:41`: re-exports `decide`. Live sizing goes through the late planner
  above.

**What follows from the callers:** the NaN-holding path is **latent**. Every operational caller
validates upstream, and the only unvalidated caller (`run_paper`) is shielded by an incidental
guard. The label path is **reachable** from any strategy whose `target_weights` returns a
non-string index label, through `plan()`/`decide()` in-process and through the shared rails in
backtests.

**Existing tests**

- `tests/test_planner.py:56-64` covers NaN *target* weights (`non_finite_weight`) only.
- No test covers NaN or inf *current* weights in `build_intents` or `plan`, mixed or non-string
  target labels, or non-string current-weight keys.
- `tests/test_live_loop.py` covers the late planner's captured-value validation.
- `tests/test_frozen_dispatch.py` pins `DECISION_BREACH_KINDS`. `tests/test_failure_vocabularies.py`
  pins `RISK_BREACH_KINDS`.

## Options

### Non-finite current weight (Todoist 6hfMw6W8mcVjQ2wp)

| Option | Behavior | For | Against |
|---|---|---|---|
| **N1. Fail closed: contract error (recommended)** | `plan()`/`build_intents` raise `ValueError("current_weights[...] must be a finite number")`, mirroring `planner_late._finite_number` | Same contract and error style the frozen path already uses. No vocabulary change. Unreachable from validated callers, so no operational behavior change. | A contract `ValueError` is not a `RiskBreach`, so it does not trip the strategy kill switch. It is reported as a generic error, which is correct for a caller bug. |
| N2. Fail closed: dark-feed breach | Raise `RiskBreach("unvaluable_marks")` (in `DARK_FEED_KINDS`): halt the account, preserve positions | Matches the real-world cause (an unvaluable mark) and routes to halt-and-preserve | Puts a strategy-free wall inside the pure decision step. A frozen child must not report it (`DECISION_BREACH_KINDS`), so the two paths would diverge. Needs routing review. |
| N3. Flatten (NaN → 0.0) | Treat the holding as flat and emit the target delta | Never blocks a tick | **Unsafe:** with an unknown real position it can buy the full target on top of it (over-exposure), or skip the exit of a real holding. Violates "never silently flatten" (`limits.py:108-112`). |
| N4. Skip with alert | Today's omission plus an audit/alert row | Minimal change | Keeps a stale position silently. The decision is still not a function of the book. Alert-only is fail-open. |

Also in scope with N1 or N2: in `run_paper`, check marks for held symbols *before* sizing, so a NaN
or missing close for a held symbol raises `RiskBreach("unvaluable_marks")` instead of the
misleading `non_positive_equity`, and a held symbol absent from `closes` is no longer valued at 0.0.
This uses an existing kind and is the same semantics the live lane applies (`assert_marks_usable`).

### Symbol-label contract (Todoist 6hfMw6Vvhv8rRgMp)

| Option | Behavior | For | Against |
|---|---|---|---|
| **L1. Validate at the shared rail (recommended)** | `check_finite_weights` also rejects any label that is not a non-empty `str`, raising `RiskBreach("non_finite_weight")`. `build_intents` checks `current_weights` keys the same way (`ValueError`). Every rail message sorts with `key=str`. | One rule for backtest, paper and live. Reuses a decision kind that already covers label defects. Matches wire, bars and late-planner rules. | The kind name says "weight", not "label". It can newly reject a backtest that today passes with a zero-weight non-`str` label (readiness must scan registered strategies). |
| L2. Planner-only validation | Check labels in `plan()` only | Smallest diff | The planner-only drift the Todoist card warns against. Backtests and the live rails disagree. |
| L3. New kind `invalid_symbol_label` | As L1 with a new `RiskBreach` kind | Precise name | Changes the frozen child's closed vocabulary and `DECISION_BREACH_KINDS`. A wider contract change and readiness rerun for Story 1.3c §7. |
| L4. Coerce with `key=str` everywhere | Sort with `key=str` and accept any hashable label | Never raises | Hides the bug. `1` and `"1"` collide in messages and broker symbols. A non-`str` symbol can still reach `OrderIntent`. |

Out of scope, noted: strategy-side NFC normalization of labels (the late planner normalizes
captured keys; strategy weights are not normalized). It needs a separate parity check.

## Recommendation

**N1 + L1, with the `run_paper` mark check**, all fail-closed:

- labels on both inputs must be non-empty `str`;
- current weights must be finite;
- target-label defects use the existing `non_finite_weight` decision breach;
- current-weight defects are contract `ValueError`s that mirror `planner_late`.

No new breach kind and no change to the frozen vocabulary. Valid inputs produce byte-identical
intents.

## Owner decision

Lior approved on 2026-10-22:

1. **N1.** Fail closed with a contract `ValueError` for NaN/inf current weights.
2. **L1.** Labels must be non-empty `str`, enforced in `check_finite_weights` as `non_finite_weight`.
3. The `run_paper` held-symbol mark check.

## Code Map (for the implementing story, after approval)

- `algua/risk/limits.py`: label-type rule in `check_finite_weights`; `key=str` in every rail
  message.
- `algua/live/planner_decision.py`: `current_weights` contract check (`str` keys, finite values)
  shared by `plan()` and `build_intents`. Prefer one helper reused from, or moved beside,
  `planner_late._numeric_mapping` over a third copy.
- `algua/live/paper_loop.py`: held-symbol mark check before sizing.
- `tests/test_planner.py`, `tests/test_risk_limits.py`, `tests/test_decision_parity.py`,
  `tests/test_frozen_dispatch.py`: new cases plus parity.

## Tasks & Acceptance

**Execution (after owner approval):**
- [x] Readiness: scan registered strategies and `algua/strategies/` for non-`str` weight labels;
  rerun readiness if any exist.
- [x] Tests first: one failing test per I/O-matrix row.
- [x] Implement N1/L1 (or the chosen options) and the `run_paper` mark check.
- [x] Mutation-check each new guard (break it, confirm a test fails, restore it).
- [x] Independent review round (story-delivery §4); record findings here.

**Acceptance Criteria:**
- Given a NaN, +inf or -inf current weight for any symbol, when `plan()` or `build_intents` runs,
  then it raises before returning any intent and no symbol is silently omitted.
- Given a target-weight Series with any non-`str` or empty-string label, when
  `validate_decision_weights` runs, then it raises `RiskBreach` of kind `non_finite_weight` naming a
  non-string symbol label. This holds identically in the backtest, paper and live paths.
- Given mixed-type labels that also breach the cap, long-only or finite rail, when the breach
  message is rendered, then a `RiskBreach` of that rail's kind is raised, never a bare `TypeError`.
- Given a `current_weights` mapping with a non-`str` key, when `plan()` or `build_intents` runs,
  then it raises a contract error before any intent is built.
- Given a held symbol with a NaN or missing close at decision bar `t` in `run_paper`, when the bar
  is processed, then the loop raises `RiskBreach("unvaluable_marks")` before sizing.
- Given valid inputs, when compared to baseline `a3c71fa`, then intents, the frozen-attempt
  comparison (`frozen_attempt.py:89`) and decision-parity tests are unchanged.
- `DECISION_BREACH_KINDS` and `RISK_BREACH_KINDS` are unchanged (under options N1 and L1).

## Test Plan

- `tests/test_planner.py`: parametrize `build_intents` and `plan` over NaN, +inf and -inf current
  weights, with targets of 0.0 and 0.5. Add non-`str` and empty current keys, and mixed target
  labels.
- `tests/test_risk_limits.py`: each rail (finite, cap, long-only, universe) with mixed labels raises
  `RiskBreach`, not `TypeError`. Non-`str` and empty labels raise `non_finite_weight`.
- `tests/test_paper_loop.py`: held symbol with a NaN close and held symbol
  missing from `closes` both raise `unvaluable_marks`.
- Parity: the existing `tests/test_decision_parity.py` and `tests/test_frozen_dispatch.py` stay
  green unmodified.
- Gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## Out of Scope

- New `RiskBreach` kinds or breach-routing changes (unless the owner picks N2 or L3).
- NFC normalization of strategy-returned labels.
- Broker or ledger valuation changes outside `run_paper`'s held-symbol mark check.
- Any live-activation, capital or authority-boundary change.

## Implementation record

Branch `fix/planner-nan-and-label-contract` from `db68600` (PR #711). Options N1 + L1 and the
`run_paper` held-symbol mark check. No new `RiskBreach` kind. `DECISION_BREACH_KINDS` and
`RISK_BREACH_KINDS` are unchanged.

**Readiness.** `uv run algua registry list` on an empty local registry returned `[]` (this checkout
has no `data/algua.db`). All 26 bundled strategies were executed on synthetic string-symbol bars.
The two model strategies have no manifest here, so their signals were called with the pinned
artifact bytes. Every non-empty weight index was `str`. Four mean-reversion strategies returned an
empty series on a monotonic path (no labels). No non-`str` weight label, so implementation
proceeded.

**What landed.**
- `check_finite_weights` rejects a label that is not a non-empty `str` as
  `RiskBreach("non_finite_weight")` naming a non-string symbol label. The check runs before the
  duplicate and value checks, so a non-finite weight on a bad label is still named as a label
  defect. Cap, long-only, finite-value, and duplicate messages sort with `key=str`.
- `planner_late._finite_number` and `_numeric_mapping` moved into `planner_decision` as the
  public helpers `finite_number` and `numeric_mapping` (the late planner imports them) and are
  imported back. `build_intents` (and therefore `plan`) validates current weights through that
  helper with `normalize=False`: non-empty string keys, finite values, `ValueError` before any
  intent, and the key text is left unchanged. The late planner keeps the default
  `normalize=True` for captured quantities and market values. Strategy target labels are not
  normalized.
- `run_paper` checks held-symbol closes after warm-up and before the equity guard. A missing,
  non-finite, or non-positive close raises `RiskBreach("unvaluable_marks")` and is not sized at
  `0.0`. A close must be strictly positive (`mark > 0.0`), the same rule as the live lane's
  `assert_marks_usable`. Held closes of `0.0` and `-1.0` are `unvaluable_marks` and submit no
  orders. A missing close is `unvaluable_marks` here; the live lane reports absence as
  `stale_marks`.

**Mutation check** (each break restored the original bytes; the named test failed):
- label-type check: `test_non_string_symbol_label_raises_non_finite_weight` and
  `test_empty_symbol_label_raises_non_finite_weight`
- label-list `key=str`: the same non-string test, on `{1: 0.2, b"ZZ": 0.2}`
- cap `key=str` and long-only `key=str`: the mixed-label rail test
- current-weight finite check: NaN and ±inf cases on `plan` and `build_intents`
- current-weight key check: non-`str` and empty keys
- held-mark call, the finite/positive branch, and the missing-close branch: the two `run_paper`
  tests

Removing `key=str` from the duplicate-symbol and non-finite-value messages did not fail a test.
Those sorts are only reached after every label is a non-empty `str`, and string labels sort the
same either way. The `key=str` calls stay so a later reorder cannot bring the `TypeError` back.
Universe membership already sorted with `key=str`.

**Gate, this sandbox.** `uv run ruff check .`, `uv run mypy algua`, and `uv run lint-imports` pass.
`uv run pytest -q` was 6273 passed, 7 skipped, 39 failed. Every failure is a process-pool worker
abort in `threadpoolctl` after `fork` (`parallel sweep failed`), except
`test_transactional_failed_import_preserves_prior_registry`, which passes alone. With
`ALGUA_SWEEP_CPU_BUDGET=1`, 36 of the 39 pass. `test_run_combos_pool_preserves_order` forces the
pool and still aborts. `test_unregistered_strategy_full_cycle_registers_produces_evidence_and_promotes`
fails the same JSON decode on unmodified `main` under that budget. Not caused by this diff.

**Review round (PR #712, head `292ba5f`).** Findings and resolutions:

1. Medium, parity. `build_intents` NFC-normalized current-weight keys and left strategy target
   labels raw. A non-NFC held symbol already at target (`E` + combining acute, weight 0.5 on
   both sides) emitted `[BUY raw, SELL normalised]` instead of `[]`. The spec requires
   byte-identical intents for valid inputs and lists NFC of strategy labels as out of scope.
   Resolution: `numeric_mapping(..., normalize=False)` in `build_intents` validates keys and
   values without renaming. The late planner still NFC-normalizes captured state (default
   `normalize=True`). Failing test added first:
   `build_intents(Series({'E\u0301': 0.5}), {'E\u0301': 0.5}) == []`. Dropping `normalize=False`
   fails that test.
2. Low. `run_paper` rejects `close <= 0` as `unvaluable_marks`. Kept: it matches live
   `assert_marks_usable`. Tests added for a held symbol with close `0.0` and `-1.0`: kind
   `unvaluable_marks`, no orders submitted. The positive-close rule is recorded above.
3. Nits. `finite_number` and `numeric_mapping` are public names, because `planner_late` imports
   them across modules. `from collections.abc import Mapping` sits at the top of
   `planner_decision.py`.

No merge, no deploy, no broker or trading-config change.

## Spec Change Log

- 2026-10-22: drafted from the Story 1.1 deferred-work entries; awaiting owner decision.
- 2026-10-22: owner approved N1 + L1; status ready-for-dev.
- 2026-10-22: implemented N1 + L1 and the `run_paper` mark check on
  `fix/planner-nan-and-label-contract`. See the implementation record. Review round still open.
- 2026-10-22: review round on PR #712. `build_intents` no longer NFC-renames current-weight
  keys (`normalize=False`). Held closes of `0.0` and `-1.0` stay `unvaluable_marks` (positive
  close required, matching live). Helpers renamed `finite_number` / `numeric_mapping`; `Mapping`
  import moved to module top.

## Verification

**Commands:**
- `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
- This sandbox: ruff, mypy, and lint-imports exit 0. pytest exits 1 on 39 process-pool worker
  aborts described in the implementation record.
