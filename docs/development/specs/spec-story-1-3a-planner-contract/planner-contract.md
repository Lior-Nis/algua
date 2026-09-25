# Normative two-phase planner contract

This companion defines the logical contract for Story 1.3a. It is independent of Python object
layout and of Story 1.3c's later Parquet/JSON wire representation.

## Contract vocabulary

| Name | Meaning |
|---|---|
| boundary version | `1`; identifies this logical schema and binding algorithm only |
| request ID | 32 lowercase hexadecimal characters generated once for one supervisor tick; opaque and non-authoritative |
| deployed identity | the active deployment and artifact identity already verified by the supervisor; all fields are null only for the fixed legacy cohort |
| resolved configuration | canonical UTF-8 JSON already bound by the verified deployment manifest, or recomputed by the legacy compatibility path using the same canonical form |
| logical binding | full lowercase SHA-256 over the canonical Phase A binding preimage defined below |
| wire evidence | later hashes over exact request/Parquet/response bytes; distinct from the logical binding |

All datetimes are timezone-aware UTC instants. Symbols are non-empty strings. Lists called ordered
retain their input order; maps are logically unordered and canonicalize by key. A bool never
satisfies an integer or float field.

## Pure execution context

The invocation resolves one pure strategy implementation against `config_hash` and
`resolved_config_json` before either phase. In-process this may be a `LoadedStrategy`; frozen
execution will load the equivalent implementation from the verified artifact. It is code context,
not an authority-bearing request field. It may expose only deterministic strategy, construction,
overlay and shared risk behavior.

## Common early input

`EarlyPlannerInput` is immutable at the API boundary and contains exactly:

| Field | Type | Rule |
|---|---|---|
| `boundary_version` | integer | exactly `1` |
| `request_id` | string | 32 lowercase hexadecimal characters; generated once and reused by both phases |
| `strategy_name` | string | equals the resolved strategy name |
| `deployment_id` | integer or null | positive; null only for the fixed legacy cohort |
| `artifact_id` | integer or null | positive; null iff `deployment_id` is null |
| `manifest_digest` | string or null | 64 lowercase hexadecimal characters; null iff `deployment_id` is null |
| `config_hash` | string | the verified 32-character gate identity |
| `resolved_config_json` | UTF-8 JSON string | canonical object; reproduces `config_hash` and the resolved strategy configuration |
| `now` | UTC datetime | explicit tick clock; never read inside the planner |
| `timeframe` | string | currently exactly `1d`; the supervisor fail-fast guard rejects other values before acquisition and Phase A revalidates the captured value |
| `calendar_code` | string | exact exchange-calendar selector captured from current settings; the planner constructs the config-free calendar leaf from this value |
| `raw_bars` | pandas DataFrame logically conforming to the canonical bar schema | exact fetched union frame before closed-bar filtering; Phase A owns sorting/filtering |
| `early_positions` | map of string to float | exact early ledger/broker quantities, including zero entries; used to identify held symbols and report early no-decisions |
| `gate_universe` | ordered tuple of unique strings | exact point-in-time decision universe used for this deployment tick |
| `max_drawdown` | float or null | explicit per-strategy breaker bound; null preserves the disabled sentinel |

`resolved_config_json` carries the complete `StrategyConfig`, including the execution contract,
construction parameters, overlays and supported sidecar/model identity. Constants frozen in planner
source, such as mark-freshness and reconciliation tolerances, are not duplicated as caller inputs.
The settings-backed calendar factory never crosses the boundary; `calendar_code` captures its
behavior-affecting choice explicitly and the planner uses the pure `MarketCalendar` leaf. Provider
window bounds are provenance of acquisition, not behavior inputs after `raw_bars` has been captured,
and therefore do not cross this boundary.

The supervisor validates the static timeframe before positions/provider access to preserve the
baseline no-effect failure order; Phase A revalidates it so immutable planner behavior cannot accept
a request the supervisor guard should have refused. The supervisor may also validate structural
identity before invocation. Phase A owns all remaining behavior-affecting transformations after
capture: stable index sort, closed-bar cutoff, held/universe views, latest marks, freshness,
decision timestamp and warm-up state.

## Phase A result union

Phase A returns exactly one variant:

| Variant | Fields | Meaning |
|---|---|---|
| `EarlyNoDecision` | `reason`, `state` | terminal flat-book result; `reason` is `no_bars` or `warming` |
| `PlannerRiskFailure` | `kind`, `detail`, `is_dark_feed` | typed baseline `RiskBreach` reached before late capture |
| `PlannerInputFailure` | `code`, `detail` | deterministic request/configuration failure such as `unsupported_timeframe` or invalid identity |
| `SnapshotRequired` | `decision_ts`, `warming`, `phase_a_binding` | late state is required; `decision_ts` may be null only when the held book must be valued despite no usable universe decision time |

`state` has the following exact fields and baseline values for an early no-decision:

| Field | Type | Early value |
|---|---|---|
| `decision_ts` | UTC datetime or null | null for `no_bars`; latest closed universe timestamp for `warming` |
| `target_weights` | ordered tuple of `(symbol, float)` | empty |
| `positions_before` | ordered tuple of `(symbol, float)` | `early_positions` sorted by symbol, retaining zero entries exactly as the baseline report does |
| `equity` | float | `0.0` |
| `peak_equity` | float or null | null |
| `reconcile_ok` | bool | true |
| `realized_gross` | float | `0.0` |

A held book never terminates as an unvalued early no-decision: unusable held marks return
`PlannerRiskFailure`; otherwise Phase A returns `SnapshotRequired`, including during warm-up.

## Captured late input

`LatePlannerInput` contains the original `EarlyPlannerInput`, the exact `phase_a_binding`, and one
`CapturedStrategyState`:

| Field | Type | Rule |
|---|---|---|
| `request_id` | string | equals the original early request ID; prevents captured state from another tick being mixed in |
| `sizing_equity` | float | exact snapshot sizing denominator |
| `drawdown_equity` | float | exact NAV/equity basis used by the drawdown wall |
| `quantities` | map of string to float | exact captured snapshot quantities, including zero entries |
| `market_values` | map of string to float | exact captured snapshot market values, including zero entries |
| `persisted_peak_equity` | float or null | state read once before Phase B |
| `venue_belief` | tagged union | `disabled` or `enabled` with a quantity map; enabled-empty is distinct from disabled |

Phase B derives `positions_before`, `current_weights`, ratcheted peak, reconciliation status and
realized gross from these values. Callers cannot supply those derived values separately. Account
buying power, book-wide exposure and order-reservation state remain supervisor concerns and do not
cross this boundary.

## Phase B result union

Phase B first recomputes Phase A from the original early input. It returns exactly one variant:

| Variant | Fields | Meaning |
|---|---|---|
| `PhaseBindingFailure` | `code`, `detail` | recomputed Phase A is not `SnapshotRequired`, or its binding/outcome differs; no late risk or strategy decision runs |
| `PlannerRiskFailure` | `kind`, `detail`, `is_dark_feed` | typed baseline breach from equity, drawdown, reconciliation, realized gross or decision-weight validation |
| `LateNoDecision` | `reason`, `state` | valued held book remains in warm-up; `reason` is `warming` |
| `Decision` | `state`, `ordered_intents` | complete pure portfolio intent ready for supervisor effect checks; target weights are in `state` |
| `PlannerInputFailure` | `code`, `detail` | deterministic invalid late input or resolved-identity mismatch |

`state` uses the exact fields listed under Phase A. For late outcomes, it contains filtered nonzero
snapshot quantities as `positions_before`, `drawdown_equity`, the ratcheted peak,
`reconcile_ok=true`, and derived realized gross. `state.target_weights` preserves the strategy
series order for result parity. `ordered_intents` is sorted by the existing planner rule and contains
`symbol`, `side`, `target_weight` and `decision_ts`. It contains no order ID, client order ID,
submission result or broker response.

Expected risk and validation failures are values at the phase boundary. The current supervisor
adapter re-raises the corresponding existing exception so CLI behavior remains unchanged. An
unexpected strategy/programming exception is not converted into a successful planner outcome; it
propagates in-process and becomes a child-process failure in Story 1.3c.

## Canonical logical binding

The binding prevents a Phase A result from being mixed with a different request. It is not a MAC,
signature, approval or authorization token.

### Scalar normalization

- Strings are UTF-8 after NFC normalization.
- UTC datetimes are signed decimal Unix nanoseconds encoded as strings.
- Integers are signed base-10 strings; bool is rejected where an integer is required.
- Floats are IEEE-754 binary64 tokens: finite values use `float.hex()` after normalizing `-0.0` to
  `0.0`; positive/negative infinity use `+inf`/`-inf`; all NaN payloads normalize to `nan`.
- Tuples remain ordered arrays. Maps become arrays of `[normalized_key, normalized_value]` sorted by
  normalized UTF-8 key bytes. Tagged unions always include their tag, so null/disabled and empty are
  distinct.

### Bar-frame digest

`bars_sha256` is SHA-256 over canonical UTF-8 JSON with keys `index_name`, `columns` and `rows`.
`index_name` is `timestamp`; `columns` is the exact canonical bar-schema order. Each row is an array
of UTC timestamp nanoseconds, normalized symbol and normalized float tokens. Rows remain in exact
captured order, so a different order changes the binding; Phase A then performs the baseline stable
index sort for computation. Production provider output must already satisfy canonical
`(timestamp, symbol)` order and uniqueness. A different column order, index name or duplicate key is
invalid rather than silently repaired.

Defensive parity fixtures may contain NaN or infinity to exercise the mark wall; the float-token
rules make those frames bindable even though production `DataProvider` output normally rejects
them. Story 1.3c may carry the frame in Parquet, but after decoding it must reproduce this logical
digest. A separate hash records the exact Parquet bytes.

### Configuration digest

`resolved_config_sha256` is SHA-256 over the exact UTF-8 bytes of `resolved_config_json` after
verifying it is canonical JSON using sorted keys, compact separators and `allow_nan=False`. The
preimage also includes the existing `config_hash`; neither replaces the other.

### Root preimage

Construct one canonical JSON object with sorted keys, compact separators and ASCII field names:

```json
{
  "domain": "algua.phase-a-binding",
  "version": "1",
  "early": {
    "request_id": "...",
    "strategy_name": "...",
    "deployment_id": "... or null",
    "artifact_id": "... or null",
    "manifest_digest": "... or null",
    "config_hash": "...",
    "resolved_config_sha256": "...",
    "now_ns": "...",
    "timeframe": "1d",
    "calendar_code": "XNYS",
    "bars_sha256": "...",
    "early_positions": [],
    "gate_universe": [],
    "max_drawdown": "float token or null"
  },
  "outcome": {
    "kind": "snapshot_required",
    "decision_ts_ns": "... or null",
    "warming": true
  }
}
```

Every placeholder is replaced by its normalized value. `early_positions` and `gate_universe` use
the normalization above. The UTF-8 bytes of this JSON object are the sole root preimage;
`phase_a_binding = sha256(preimage).hexdigest()`.

Each behavior-affecting early value appears once in the root, directly or through one named full
SHA-256 component digest. Phase B recomputes the complete root from its original early input and
recomputed `SnapshotRequired` outcome and compares the 64 hex characters with constant-time digest
comparison before inspecting late state or invoking strategy decision code.

## Phase state machine

1. The supervisor verifies deployment/config identity and the static timeframe. Only then does it
   capture early positions, fetch the union of gate universe and nonzero held symbols, and create
   `EarlyPlannerInput`.
2. Phase A evaluates only the early input. A terminal result ends planner work. It performs no late
   acquisition.
3. On `SnapshotRequired`, the supervisor captures one sizing snapshot, drawdown basis, peak and
   tagged venue belief, then creates `LatePlannerInput` without mutating the early input.
4. Phase B recomputes Phase A and verifies the binding before reading/evaluating late state.
5. The supervisor translates the pure outcome into the existing result/exception surface, applies
   halt/cancel/submit hooks in the existing order and persists operational evidence.

## Required parity matrix

| Branch | Required assertion |
|---|---|
| unsupported timeframe | fails before positions, provider or late capture |
| flat + empty bars | early no-decision; no late capture |
| flat + warming | early no-decision; no late capture |
| held + missing/stale/non-finite mark | early typed breach; no late capture |
| held + warming | late capture and risk checks; valued no-decision |
| invalid/non-positive sizing equity | late typed breach before division/decision |
| drawdown breach | late typed breach before reconcile/decision |
| venue belief disabled | reconciliation skipped |
| venue belief enabled-empty | reconciliation performed; mismatch can breach |
| fractional reconciliation residual | tolerated exactly as baseline |
| realized gross breach | late typed breach before decision |
| dropped universe holding | valued from union bars and emitted as ordered zero-target intent |
| target-weight rails | same finite, universe, short, concentration and gross breaches |
| construction/overlay/capacity/gross utilization | identical weights and ordered intents |
| changed request/config/calendar/bar/position/bound/outcome | different Phase A binding |
| altered supplied binding | Phase B binding failure before late risk/decision |
| normal decision | identical state, weights, intents and supervisor effect trace |
| repeated invocation | identical logical result and binding for identical request identity/input |

The test suite must additionally prove the planner dependency closure cannot reach operational
modules or carry callable authority in either input type.
