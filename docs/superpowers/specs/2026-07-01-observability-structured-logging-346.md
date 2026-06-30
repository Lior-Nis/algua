# Design — Issue #346: structured logging for the always-on loop

## Problem
algua has zero leveled/structured logging (`grep import logging|getLogger|structlog` across `algua/` = 0 hits). The always-on systemd paper/live loop (`paper run-all` / `live run-all`) runs unattended with only `audit_log` (business events), the stdout JSON envelope, and `tick_snapshots` as telemetry. KB Golden Rule 12 + Observability Foundations north star ("diagnose a failure from telemetry alone, no debugger") are unmet for the one component that runs unattended.

## Goal
Add a stdlib-only structured JSON logging facility and wire it into the always-on loop with per-cycle correlation IDs, ERROR-level capture on the breach/flatten/reconcile paths, and golden-signal counters (ticks, breaches, flatten failures) emitted per cycle.

## Constraints discovered
- **stdout is a machine contract.** `cli/app.py:emit()` writes indented JSON to stdout (`typer.echo`); the whole CLI is "every data command emits JSON on stdout". Logs MUST go to **stderr** or they corrupt the JSON envelope. This is the load-bearing decision.
- contracts/features/provenance must stay pure (import-linter forbidden contracts). The new module must not break boundaries.
- Python 3.12, Typer, SQLite. No heavy deps wanted.

## Decisions (forks resolved)
1. **Library:** stdlib `logging` + a custom JSON formatter. No structlog / no new dependency.
2. **Module:** new pure-leaf package `algua/observability/` that imports **no** other `algua.*` module. Add an import-linter `forbidden` contract (mirrors `algua.provenance`) so it stays a leaf. cli/execution/live/risk may import it.
3. **Sink:** single `StreamHandler(sys.stderr)` with the JSON formatter, attached to the `algua` logger, `propagate=False`. Level from env `ALGUA_LOG_LEVEL` (default `INFO`).
4. **Correlation IDs:** a `contextvars.ContextVar` holding a per-cycle id, set at `run-all` entry via a `correlation_context()` context manager. The formatter reads it onto every record. Per-strategy logs additionally carry `strategy` + `tick_ts` fields. ("per-tick" in the issue = per loop iteration / cycle, the unit a systemd timer fires.)
5. **Golden-signal counters:** a small **pure** `CycleCounters` dataclass (ticks, breaches, flatten_failures, reconcile_deferred, reconcile_halted) accumulated during `run-all` and emitted as ONE structured `golden_signals` log line at cycle end. No Prometheus / metrics registry — KISS.
6. **Config entry:** `configure_logging()` is idempotent (guards against duplicate handlers across repeated CLI invocations / tests) and called once at the top of `cli/main.py:main()`.
7. **Extra fields:** call sites use `log.info(msg, extra={"fields": {...}})`; the formatter pulls the reserved `fields` dict so structured key/values land in the JSON without clobbering `LogRecord` attributes.

## Module surface (`algua/observability/`)
- `log.py` (named `log.py`, NOT `logging.py`, to avoid stdlib-shadow confusion — mirrors `algua/audit/log.py`):
  - `JsonFormatter(logging.Formatter)` → one JSON object per **physical line**. Keys: `ts` (ISO-8601 UTC), `level`, `logger`, `msg`, `correlation_id` (from contextvar; key omitted when unset), then merged `fields`. **Core keys always win** over caller `fields` — and the core set includes the exception keys `exc_type` (`exc_info[0].__name__`), `exc_message` (`str(value)`), `stacktrace` (`self.formatException(record.exc_info)` as one JSON string), so a caller `fields` dict can never obscure real exception metadata (merge `fields` first, then overwrite with all core keys). Serialize with `json.dumps(obj, default=str)`. **Final fallback:** the whole `format()` body is wrapped so that ANY serialization error (non-serializable, circular `fields`, etc.) degrades to a minimal hand-built JSON record `{"ts","level","msg","format_error": <err>}` (NOT reusing the unsafe user fields) — a record is never dropped and never raises into the logging machinery.
  - `get_logger(name: str = ...) -> Logger` — returns `logging.getLogger(name)`; callers pass `__name__` (already under `algua.*`). If a name does not start with `algua`, it is prefixed so every logger is a child of the `algua` root (which carries the handler).
  - `configure_logging() -> None` — idempotent: ALWAYS re-reads `ALGUA_LOG_LEVEL` and calls `logger.setLevel(...)` on the `algua` logger every call (so test monkeypatching of the env takes effect, matching the uncached-`get_settings()` discipline); validates/normalizes the level name and falls back to `INFO` on an unknown value; adds the stderr `StreamHandler` ONLY if our marked handler (`handler._algua_observability = True`) is not already attached — never removes foreign/test handlers. Sets `propagate=False` on the `algua` logger so records don't double-emit via the root `lastResort` handler.
  - `correlation_context(cid: str | None = None)` — context manager that sets/resets the contextvar via a token (generates a uuid4 hex if `cid is None`); resets in a `finally` so the id never leaks past the block even on exception.
- `metrics.py`:
  - `@dataclass CycleCounters` — plain int fields (`ticks`, `breaches`, `flatten_failures`, `reconcile_deferred`, `reconcile_halted`) + `as_fields() -> dict`. Pure, no I/O, no registry/decorators.

## Wiring (production)
- `cli/main.py:main()` — call `configure_logging()` first thing (real console-script entry).
- `cli/paper_cmd.py:run_all` & `cli/live_cmd.py:run_all`:
  - call `configure_logging()` idempotently at entry too (so the always-on loop is logged regardless of invocation path — tests/direct calls bypass `main()`).
  - open a `correlation_context()` for the whole cycle and initialize a `CycleCounters` IMMEDIATELY at entry (before snapshot load / venue ingest / reconcile), then wrap the **entire** run-all body in `try/finally`; INFO `cycle_start` (snapshot, n strategies).
  - ERROR `venue_ingest_failed` (paper), ERROR `reconcile_halt` (with `mismatches`), INFO `reconcile_deferred`.
  - increment `CycleCounters` per tick / breach; emit INFO `golden_signals` (counters) in the **`finally`** that wraps the whole body, so the rollup flushes even if the cycle fails BEFORE the strategy loop (snapshot/ingest/reconcile setup); on an unexpected exception also ERROR `cycle_failed` with `exc_info` before re-raising. (`typer.Exit` is the normal control-flow exit and is allowed to propagate through the `finally` after the rollup emits.)
- `_run_paper_strategy_tick` / `_run_strategy_tick`: ERROR `breach` (RiskBreach, with `exc_info`, carrying `strategy` + `tick_ts`), ERROR `flatten_failed` (the `except … as fexc` path, with `exc_info`), INFO `tick_halted`. Every tick-scoped log carries `strategy` + `tick_ts` fields (the per-strategy disambiguator within one per-cycle correlation id). These sit ALONGSIDE the existing `audit_append` + JSON emit — additive, no change to stdout.

## Testing (attach a handler to the `algua` logger directly; do NOT rely on `caplog` since `propagate=False`; use `capfd` for fd-level stderr assertions)
- `JsonFormatter`: emits a valid one-line JSON object with required keys; merges `fields` with core keys winning; renders `exc_type`/`exc_message`/`stacktrace` strings when `exc_info` is set (record NOT dropped); a non-serializable field is stringified, not dropped; correlation id present inside a context, key absent outside.
- `configure_logging`: idempotent (exactly one marked handler after repeated calls); a pre-attached foreign handler survives; re-reads `ALGUA_LOG_LEVEL` on each call; unknown level → INFO; writes to **stderr, not stdout** (assert stdout stays empty).
- `correlation_context`: sets within / resets after (nested + exception unwind).
- `CycleCounters`: accumulation + `as_fields`.
- Loop integration: a breach cycle emits an ERROR `breach` record AND a `golden_signals` line with `breaches>=1` even though the cycle exits non-zero (finally path); a clean cycle emits `cycle_start` + `golden_signals`; assert the logger writes nothing to **stdout**.

## CODEOWNERS / safety
Touches only: `algua/observability/*` (new), `cli/main.py`, `cli/paper_cmd.py`, `cli/live_cmd.py`, `pyproject.toml` (import-linter contract), tests. **None** are CODEOWNERS-protected → eligible for auto-merge on green CI.

## Explicitly out of scope (YAGNI) / deliberately declined review findings
- No metrics backend / Prometheus / OpenTelemetry.
- No log rotation / file sinks (systemd journal captures stderr).
- No threading a logger through pure execution/research core — wiring stays at the cli loop layer where the named breach/flatten/reconcile paths already live.
- No log statements sprinkled across non-loop CLI commands (the issue scopes this to the always-on loop).
- **Contextvar thread-propagation:** declined. `run-all` is single-threaded (sequential strategy loop; the only ProcessPool is research `sweep`, out of scope). Documented assumption; revisit only if run-all is ever parallelized.
- **Root-logger third-party non-JSON lines:** declined active mitigation. No third-party library logs at WARNING+ on these paths today; stderr is documented "JSON-best-effort: consumers skip non-JSON lines." `propagate=False` keeps algua's own records clean.
- **Per-tick (vs per-cycle) correlation id:** declined. One cycle = one invocation; the `strategy`+`tick_ts` fields disambiguate each strategy within the cycle's single correlation id.

## Import-linter
Add a `forbidden` contract: `source_modules = ["algua.observability"]`, forbidding every other `algua.*` package (mirrors the `algua.provenance` leaf contract) so the module stays a pure stdlib leaf. Also add `algua.observability` to the `forbidden_modules` of the `contracts`, `features`, `provenance`, and `portfolio` pure-leaf contracts so those layers can never import the (side-effectful) logger.
