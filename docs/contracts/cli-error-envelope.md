# CLI error envelope — machine-readable `code`

Every `uv run algua ...` command emits JSON on stdout. Failures use a **stable, machine-readable
`code`** so an operator (human or agent) can branch on the error *kind* without pattern-matching
free-text English (issue #337).

## The standard failure envelope

```json
{ "ok": false, "error": "<human-readable message>", "code": "<stable code>", "retryable": false }
```

`code` and `retryable` are **guaranteed present** on exactly these three surfaces, all rendered by
the CLI seam (`algua/cli/errors.py`, `algua/cli/main.py`):

1. **`@json_errors` command-body failures.** The decorator wraps every command body as a *catch-all*:
   ANY exception (including one no author anticipated — a `KeyError`/pandas error deep in
   `walk_forward`) is rendered as this envelope with exit code 1. A raw traceback can never leak and
   break the JSON contract mid-run.
2. **Argument-parse failures** (`main()`): a bad option type / unknown option / missing argument →
   `code: "usage_error"`; an aborted confirmation → `code: "aborted"`.
3. **The last-resort catch-all in `main()`**: anything escaping an *undecorated* command
   (`version`, `doctor`), the Typer callback, or the framework → the same envelope, exit 1.

`error` carries `str(exc)` — the message, **never** a traceback. On this local, single-operator CLI a
message may include a local path or symbol; that is an accepted, documented tradeoff (there is no
remote consumer to leak to), matching the issue's own recommendation.

`retryable` is a boolean **derived from `code`** (single source of truth:
`algua/cli/errors.py::RETRYABLE_CODES` + `is_retryable`) so an operator — human or agent — can branch
*retry-with-backoff* vs *abort* without pattern-matching English. It is deliberately conservative:
`retryable` defaults to **`false`**, and only a *transient environmental* failure (one that could
succeed identically on replay) opts a code in. A deterministic input/logic error is never retryable —
re-running it just burns the same failure. Today exactly one code is retryable:

| `code` | `retryable` | why |
|---|---|---|
| `db_unavailable` (`sqlite3.OperationalError`, e.g. a busy/locked registry DB) | `true` | transient contention — a bounded backoff-and-retry can clear it |
| every other code (incl. `internal`, `usage_error`, `aborted`, all input/logic/domain errors) | `false` | deterministic — replay reproduces the same failure |

Like `code`, `retryable` is an **additive** contract: `false` is the safe default, and a future code
may opt into `true` by joining `RETRYABLE_CODES` (a single-line change, never per-command).

`typer.Exit` / `typer.Abort` are treated as control flow (a command that emitted its *own* envelope
and asked to exit) and pass through unchanged; `SystemExit` / `KeyboardInterrupt` are never swallowed.

## The code registry (source of truth: `algua/cli/errors.py::_registry`)

Codes are keyed by exception **type** (identity), resolved most-specific-first. Every exception
resolves — anything unmatched is `internal`.

| exception type | `code` |
|---|---|
| `AllocationError` | `allocation_error` |
| `TransitionError` | `wrong_stage` |
| `ProviderError` | `provider_error` |
| `ConstructionError` | `construction_error` |
| `LiveSizingError` | `sizing_error` |
| `RiskBreach` | `risk_breach` |
| `SnapshotNotFound` | `not_found` |
| `SignatureError` | `bad_signature` |
| `LiveAuthorizationError` | `live_unauthorized` |
| `BrokerError` | `broker_error` |
| `BacktestError` | `backtest_error` |
| `TickHalted` | `tick_halted` |
| `ManifestLockReplacedError` | `manifest_lock_replaced` |
| `FileNotFoundError` | `file_not_found` |
| `sqlite3.OperationalError` | `db_unavailable` |
| any other `ValueError` | `invalid_input` |
| any other `LookupError` (incl. `StrategyNotFound`, `IdeaNotFound`, `FactorNotFound`) | `not_found` |
| anything else | `internal` |

Codes are an **additive** contract: existing codes are stable and are not renamed; new exception
types may add new codes. `internal` is the catch-all for genuinely unexpected/bug-class exceptions.

### Known limitation (deferred)

The generic `invalid_input` bucket collapses several distinct operational states that are all raised
as a bare `ValueError` (bad CLI input, a missing broker credential, a human-only relaxation refusal).
Finer codes for those require stamping a code at the *raise site* in non-CLI modules — several of
which are CODEOWNERS-protected — so they are a separate, human-merged follow-up. Type-keyed codes
already deliver distinct identifiers for every dedicated exception type above.

## Other envelope families (NOT retrofitted by #337)

These pre-existing shapes are already machine-readable in their own way and are documented here so
the contract is honest rather than aspirational — #337 does **not** add `code` to them:

- **Kill-switch / breach** (`algua/cli/_common.py::breach_payload`):
  `{ "ok": false, "kill_switch": "tripped", "error": ..., "kind": "<breach kind>" }`. The `kind`
  field (e.g. `gross_exposure`, `per_symbol`, `drawdown`) is the machine discriminator for this
  family.
- **Partial-result / halt envelopes** (paper/live `run-all`, reconcile): a command may `emit` its own
  `{ "ok": false, ... }` payload and raise `typer.Exit` — these keep their bespoke shape.
- **`data verify`**: emits a per-snapshot report with `ok: false` and no `error`/`code`.
