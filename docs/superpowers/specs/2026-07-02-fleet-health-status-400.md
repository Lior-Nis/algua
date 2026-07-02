# Fleet-wide health/status command (#400)

## Problem
The only standing health rollup is `paper show <name>` — one strategy per invocation. With N
strategies an operator runs N `paper show` commands to find the one in `drift`/`halted`: an O(N)
manual sweep that does not scale and guarantees slow detection. The north star is observing and
halting unsafe behavior fast enough as strategy count scales; a single aggregate health view is the
minimum operability surface.

## Solution
A read-only, JSON-on-stdout `algua fleet status` command that emits one health rollup per strategy
across ALL stages in a single call, ranked worst-offender-first.

### New pure aggregation helper — `algua/execution/fleet_health.py`
- `strategy_health(conn, rec, calendar, *, halted_globally, now) -> dict`: the SINGLE per-strategy
  rollup, extracted from `paper show`'s body so both callers share it (DRY). Uses only existing
  persisted-state readers — `SqliteStrategyRepository`, `kill_switch.get`, `latest_tick_snapshot`,
  `get_peak_equity`/`get_nav_peak`, `believed_positions`/`paper_believed_positions`/`derive_positions`,
  `count_orders`/`count_venue_orders`. NO broker call, NO writes, NO locks — pure SELECTs.
- `fleet_status(conn, calendar, *, now) -> list[dict]`: `list_strategies()` over ALL stages, calls
  `strategy_health` per strategy, ranks worst-first.

Placed in `algua.execution` (not `paper_cmd`) because a `cli->cli` sibling import is forbidden by the
`independence` contract; `algua.execution -> algua.registry/risk/calendar` is not forbidden and there
is no load-time cycle (registry->execution is lazy/in-function).

### Liveness / staleness (folds in #399's hazard, fail-closed)
`paper show`'s health='ok' does NOT require liveness — a strategy whose loop ticked yesterday then
died reports `ok` forever. The fleet rollup fixes this:
- `staleness_sessions = calendar.sessions_between(last_tick_dt.date(), now.date())` (the exact
  convention `forward_promotion.py:307` uses).
- `tick_ts` is parsed fail-closed via a local `_parse_utc` (ISO-8601 -> aware UTC; tz-naive or
  unparseable -> None; a `None` or FUTURE tick on a non-idle strategy is treated as `stale`, never
  crashes, never silently `ok`).
- New `stale` health state: a non-idle strategy whose newest tick is older than
  `STALE_AFTER_SESSIONS` (default 5, matching `MAX_STALENESS_SESSIONS`) is `stale`, not `ok`. So
  `ok` now REQUIRES a parseable, non-future tick within threshold.
- Market-closed correctness: 0 sessions elapsed is never stale (`sessions_between` returns 0 the
  same session).

### Health severity (string + ranking)
`halted` > `drift` > `stale` > `idle` > `ok`. Global halt / kill-switch (`halted`) always wins.
A never-ticked strategy stays `idle` (not `stale`). `fleet status` ranks worst-first by this order.

### JSON shape (per strategy)
`{strategy, stage, health, staleness_sessions, stale_after_sessions, kill_switch{tripped,reason,
global_halt}, drawdown{peak_equity,last_equity,drawdown}, last_tick, positions, n_orders}`.
Bare JSON array (like `registry list`), worst-first. `recent_orders` is dropped from the fleet view
(kept in `paper show`) to keep the aggregate compact.

### CLI wiring
- New `algua/cli/fleet_cmd.py` with a `fleet` typer group + `status` command; imported in
  `cli/main.py`; added to the `independence` contract module list in `pyproject.toml`.
- `paper show` refactored to call `strategy_health` (closes the duplicated-rollup divergence risk).

## Non-goals / out of scope
No loop/write path change; no schema change; no broker calls; no per-strategy heartbeat alert
plumbing beyond exposing `staleness_sessions` (a follow-up can wire alerting off it).

## Review
GATE-1 (Codex, design): CHANGES-REQUESTED -> folded fail-closed tick parsing, CLI
registration+independence, exposed `staleness_sessions`+threshold -> APPROVE.
