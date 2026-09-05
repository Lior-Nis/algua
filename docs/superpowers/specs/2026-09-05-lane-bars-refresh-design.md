# Lane bars refresh — design (#556)

**Status:** design, revised after Gate-1 rounds 1–2 (Codex; the OpenCode second-lineage reviewer
hung silently both rounds and is noted as absent). **Closes:** #556.

## Problem

The paper (and live) daily tick computes signals against a **static** bars snapshot named by
`ALGUA_PAPER_SNAPSHOT`, expanded into the paper unit's `ExecStart`. Snapshots are immutable, so the
lane's data goes stale by construction: the reference snapshot found on 2026-08-14 ended
2024-01-01. Nothing owns refreshing it. Two layered failures were observed:

1. `ALGUA_PAPER_SNAPSHOT` unset → the placeholder expanded empty → `operator run` fail-closed with
   `command_mismatch` every night, silently, until a paper strategy existed and `fleet health`
   flagged the missing tick.
2. Even when set, the snapshot ages one session per session. The #452 mark-freshness wall inside
   `run_tick` checks every consumed symbol's newest mark against `now` with a 2-session tolerance
   — so a snapshot ≥3 sessions old trips the tick closed — but nothing *supplies* fresh bars, so
   the lane silently stops trading, and nothing surfaces DATA staleness as a health verdict.

## What the tick decides on (established, unchanged here)

`run-all` resolves a wall-clock window `[start, end)` = `[today-400d, today)`; the
`StoreBackedProvider` serves it half-open, so today's midnight-stamped bar never reaches the tick,
and `run_tick` additionally drops any bar dated ≥ today. On a tick after the close of session **S**
the decision bar is the newest bar of the latest completed session strictly before today — **S−1**.
Admitting S's own bar once `session_close(S)` has passed is a pre-existing, explicitly deferred
design (`live_loop.py`: "B2b's scheduler can use the exchange calendar to admit today's bar once
its session has closed") and is out of scope — see Follow-ups. This design makes the S−1 bar
reliably *present, fresh, and deep enough*; it does not move the cutoff.

Define **`expected_session`** = `calendar.previous_session(today)`.

## Design

### 1. `algua.data.refresh.refresh_bars` — resolve-or-ingest behind a coverage wall (data layer)

```python
def refresh_bars(
    store: DataStore, provider: BarProvider, *, symbols: Sequence[str], start: str, end: str,
    require_bar_on: str, min_rows: Mapping[str, int] | None = None,
    timeframe: str = "1d", adjustment: str = "none",
) -> tuple[SnapshotRecord, bool]:
```

Returns `(record, refreshed)`. Provider identity is `provider.name` (never a caller-supplied
string). `require_bar_on` is the ISO date of `expected_session`; `min_rows` is a per-symbol floor
on rows inside `[start, end)` (decision-universe symbols carry their strategies' history need;
held-only / broker-only symbols default to 0 — they need only the terminal mark). The caller owns
the calendar and the strategy contracts; this layer stays free of both.

The **coverage wall** is one predicate, applied identically to a candidate for reuse and to a
fresh fetch, over a canonical bar-schema frame: for every requested symbol, (a) rows exist
(`missing`), (b) the newest bar is dated exactly `require_bar_on` — older is `stale`, newer is
`misdated` (a weekend/holiday row from a bad vendor) — and (c) the row count ≥ `min_rows[symbol]`
(`short_history`). Any failure → `RefreshError(kind, symbols, require_bar_on)`.

Under a **request-keyed lease** (`flock.acquire(<data_dir>/refresh-locks/<sha256(key)>.lock,
verify_inode=True)`; the directory is a documented never-swept namespace) covering resolve → fetch
→ validate → ingest, the resolve re-run after acquiring:

- **Normalize** at entry: symbols via `normalize_symbols` (upper, dedup, sorted).
- **Resolve.** Take the **newest** record in manifest order whose `(provider.name, timeframe,
  adjustment, symbols, start, end)` equals the request. Read its bars back, recompute
  `logical_bars_hash` over the canonical frame and require it to equal `rec.content_hash` (a
  same-row-count tamper or bit-rot is never reused; `verify_snapshot`'s row-count read-back is
  strictly weaker), then run the **coverage wall** against the *current* `require_bar_on` /
  `min_rows`. Pass → return `(rec, False)`, no network. Any failure → treat as a miss and fetch;
  never fall back to an older match (a lax manual `ingest-bars` under the same key, or a
  previously accepted S−2 snapshot, can therefore never be reused for an S−1 requirement).
- **Fetch** `provider.get_bars(BarRequest(...))`, canonicalize with `to_bar_schema`, **clip** to
  half-open `[start, end)` (Alpaca's inclusive `end` normalized away), run the **coverage wall**,
  then ingest through the existing staged/fsync'd/manifest path. A refresh is a **new** snapshot
  id; snapshots stay immutable.

Why one lane-wide wall rather than per-strategy quarantine: a decision-universe symbol without
yesterday's bar (or with truncated history) is a data error, not a tradable state — the same
posture as `resolve_operational_universe` on an empty membership. One flat strategy's dead ticker
blocks the lane until a human retires or re-gates it — accepted, alerted, listed as a follow-up.

### 2. `algua data refresh-bars` — the manual/testable CLI seam

```
algua data refresh-bars --symbols A,B --start D --end D --require-bar-on D [--min-rows N]
                        [--provider P] [--timeframe 1d]
→ {"ok": true, "snapshot": {...}, "refreshed": bool}
```

`RefreshError` is registered in the CLI error registry as `code: "refresh_failed"`.

### 3. `paper run-all` / `live run-all --refresh` — the lane consumes it (lane parity)

`--snapshot` becomes optional; new flag `--refresh`. **Exactly one** is required. With `--refresh`
explicit `--start` / `--end` are rejected (`invalid_input`): the window is **derived** — the end
from today (so a caller cannot certify S−2 by passing yesterday, nor mint a partial
current-session snapshot by passing tomorrow), the start from the cycle plan's history need (so
the wall it introduces can always be satisfied by the operator's fixed argv). `--snapshot <id>`
is unchanged: the explicit replay / forensics path, with `--start`/`--end` as today.

The refresh runs at the same point in both lanes — **after** broker connect, venue-fill ingest,
and stranded-row recovery; **before** the account reconcile — so a refresh fault has its own
envelope (never confusable with drift). The broker positions are read **twice**: once before the
refresh, for symbol discovery only; and again immediately after it, adjacent to the reconcile
exactly as today, so the reconcile judges the book as it is *after* the network round-trip. A
resting order that fills during the refresh therefore surfaces as a reconcile mismatch (defer,
no tick; the next fire refreshes the widened set) rather than trading against a stale sample —
and the reconcile → tick window is unchanged from today.

1. `end` = today (`resolve_wall_clock_window(None, None)[1]`); `expected_session =
   get_calendar().previous_session(date.fromisoformat(end))`.
2. **Cycle plan** over the tickable strategies (paper: PAPER ∪ FORWARD_TESTED with an active
   allocation; live: verified ∧ allocated). For each: `load_tradable_strategy(name)` — the
   **tick's tradability admission** (it rejects fundamentals-/news-/model-dependent strategies
   whose runtime lanes do not exist, so a tenant cannot pass planning and fail tradability at
   tick time; stage/halt/kill-switch are still judged at tick time by `load_gated_strategy`, and
   a tenant tripped there is a per-tick `setup_error` — follow-up: consult the kill-switch at plan
   time) — then `resolve_operational_universe(conn, data_dir, name, strategy.universe)` (the
   #559 gate-bound universe). The plan records the universe, the ledger-believed held symbols,
   and a per-symbol history floor over **every bar-consuming contract**:
   `max(config.feature_lookback, execution.warmup_bars, execution.capacity.adv_window_bars
   if execution.capacity else 0) + 1`, maxed across the strategies sharing a symbol (a window
   too short for the capacity model's ADV lookback silently zeroes capacity and would force a
   held book flat — the floor must cover it). A strategy whose `feature_lookback` is **undeclared (`None`)**
   cannot state its history need and is skipped with reason `undeclared_feature_lookback` —
   `None` is never treated as zero (every agent-promoted strategy declares one; the gate
   requires it). Any other per-strategy failure is likewise a per-tenant setup fault: recorded in
   the envelope as `{"strategy", "traded": false, "skipped"}` (the existing `StrategySetupError`
   isolation) and excluded from the plan; siblings continue. `sqlite3.Error` / `OSError` are
   systemic and abort the cycle. **If every tickable strategy failed planning, the cycle fails**
   (`ok:false`, `code:"cycle_plan_failed"`, exit 1 — the operator alerts and retries); it must not
   collapse into the benign "no strategies" no-op.
3. **Cycle start** = the earlier of the default 400-calendar-day start and the date that
   provides `max(plan.min_rows) + 5` **actual exchange sessions** through `expected_session`,
   computed against the configured calendar: start from the heuristic `expected_session −
   (ceil(N × 1.6) + 10)` days and step back in 30-day increments until
   `len(calendar.sessions_in_range(start, expected_session)) ≥ N + 5`. Exact, not a ratio — a
   holiday-dense span or another exchange cannot leave the lane permanently wedged on the same
   insufficient start. The bundled `lagged_rank_persistence_momentum` declares
   `feature_lookback=336`, which the default window (≈275 sessions) can never satisfy; the derived
   start is what makes such a strategy both pass the wall **and actually decide** — this `start`
   is the one the ticks read bars over, not only the refresh.
4. **Symbol set** = ⋃ plan universes ∪ ⋃ ledger-held ∪ broker net positions (first read).
5. `refresh_bars(store, provider, symbols=…, start=start, end=end, require_bar_on=expected_session,
   min_rows=plan.min_rows)`. Provider is **per lane**: paper uses
   `settings.bars_refresh_provider` (default `"yfinance"`, env `ALGUA_BARS_REFRESH_PROVIDER`);
   live uses `settings.bars_refresh_provider_live` (**no default**, env
   `ALGUA_BARS_REFRESH_PROVIDER_LIVE`) — `live run-all --refresh` fails closed with
   `live_refresh_provider_unset` until a human names the real-money feed explicitly. A research
   convenience default is never silently the live decision data.
6. Broker positions are read again; the reconcile runs on that second read. The resolved
   `snapshot_id` builds the `StoreBackedProvider` exactly as `--snapshot` did. The cycle envelope
   gains `"snapshot": {"id", "refreshed", "symbols", "require_bar_on", "provider", "start",
   "end"}`; logs carry the id.
7. **Any refresh failure fails the cycle closed** (`ok:false`, `code:"refresh_failed"`, exit 1) —
   no strategy is ticked, the operator classifies + alerts, the session marker stays unwritten,
   the 20-minute timer retries. Never fall back to an older snapshot.

Orchestration lives in a new **`algua/cli/lane_refresh.py`** (`CyclePlan`, `build_cycle_plan`,
`lane_symbols`, `refresh_lane_snapshot`), shared by both run-alls. Layering: `cli` composes `data`,
`registry`, `execution`, `strategies`, `calendar` — all already-permitted edges; the walled lanes
never touch `algua.data`. The per-tenant tick helpers are unchanged (they still re-resolve the
gated universe at tick time — the structural lane-parity test depends on it).

### 4. Tick provenance: which data did this tick trade on?

`tick_snapshots` gains a nullable `snapshot_id TEXT` (schema **v45**, additive, no backfill —
legacy NULL is "unknown"). Both lanes' tick helpers pass it to `record_tick_snapshot`;
`latest_tick_snapshot` returns it, so `paper show` / `fleet status` `last_tick` carry it. The
operator's session marker and success envelope record the driver payload's `snapshot.id`, and the
paper job's completion predicate **requires, whenever the payload lists strategies, both a
non-empty `snapshot.id` and at least one strategy result with `ok: true`** — a cycle that ticked
without provenance, or whose every tenant failed at tick-time setup, cannot mark the session done;
the marker stays unwritten and the next fire retries.

### 5. `fleet_health.strategy_health` — decision-data staleness is a verdict (backstop)

Every tick carries `decision_ts` = the timestamp of the last **decided** bar. Two metrics:

```
decision_stale_sessions  = calendar.sessions_stale(decision_ts, now)      # the verdict
decision_stale_at_tick   = calendar.sessions_stale(decision_ts, tick_ts)  # forensics
```

The verdict is measured against **`now`**, not the tick, so it *ages*: when the refresh fails and
no new tick row lands, the last decision bar drifts further behind the calendar and trips the
watchdog on the second missed session — instead of staying green until the 5-session loop
heartbeat expires.

- New constant `DECISION_STALE_AFTER_SESSIONS = 2`, aligned with the tick's #452
  `MAX_STALE_SESSIONS`: a fresh after-close tick reads 1; the next session before its tick reads
  2 (tolerated); a second consecutive miss reads 3 and trips. Deliberately not the 5-session
  loop-heartbeat SLO — "the loop is alive" and "the loop decided on fresh bars" are different
  questions.
- `> DECISION_STALE_AFTER_SESSIONS` → `health = "stale"`, `stale_detail = "decision bars N
  sessions behind"`. Severity ordering unchanged.
- `decision_ts IS NULL` (a legitimate empty-bars / warm-up no-op tick) → `None`; verdict left to
  the tick-staleness path. An **unparseable** `decision_ts`, a calendar mapping failure, or a
  **negative** result (decision after now: clock skew / bad data) → fails closed to `stale`.
- `fleet health` (the watchdog gate) exits non-zero on stale decision data with no new code path.
- `SessionSpanCalendar` (the injected protocol) gains `sessions_stale`; `MarketCalendar` already
  implements it.

`decision_ts` is `max()` over the universe; a lagging single symbol hides behind it. Acceptable
**as a backstop**: the refresh's coverage wall is per-symbol at mint, and the tick's #452 wall is
per-symbol at decision.

### 6. Operator job + deployment — the placeholder goes away

- `OPERATOR_JOBS["paper"].argv_template` → `("algua", "paper", "run-all", "--refresh")`. No
  placeholder; the exact-arity structural match is preserved (an appended `--snapshot X` is a
  `command_mismatch`). `is_completed` additionally requires `snapshot.id` when strategies ticked.
- `deploy/systemd/algua-paper.service`: `ExecStart=… algua paper run-all --refresh`.
- `deploy/systemd/algua.env.example`, `README.md`, `install-user-units.sh`: `ALGUA_PAPER_SNAPSHOT`
  removed (documented as removed, with the reason); `ALGUA_BARS_REFRESH_PROVIDER` documented.

Failure 1 in the problem statement becomes structurally impossible.

## Declined review findings (with rationale) and follow-ups

Declined for this change, each filed as a follow-up issue on merge:

- **Session-addressed ticks** (bind refresh, cutoff and marker to one target session S; admit S's
  bar after `session_close(S)`; driver reports `processed_session`). Pre-existing lag, deferred in
  `run_tick` itself; touches the walled decision path.
- **A second reconcile after the refresh / a lane-wide lock over all mutating cycle commands /
  open-order symbols in discovery.** The refresh now precedes the reconcile, so the reconcile→tick
  window is unchanged from today; the lane-wide lock is the existing #316 operator-discipline
  deferral.
- **Proving zero exposure before isolating a setup-failed tenant.** Mirrors the existing
  `StrategySetupError` isolation semantics exactly; changing it is a lane-safety design of its
  own.
- **Per-strategy quarantine / security master** for a flat-only strategy whose symbol the
  provider stopped serving (today: lane-wide fail-closed + alert).
- **Persisting a tick-attempt record before the first submit** so a crash mid-tick keeps the
  snapshot→order link. All tick provenance (code/config/dependency hash) is recorded after
  `run_tick` today; this is the general fix, not a refresh concern.
- **`latest_tick_snapshot` by `strategy_id`+lane; persisting no-op ticks.** Pre-existing.
- **Adopting an orphan payload after a crash between publish and manifest append.** Pre-existing
  ingest property; bounded by the snapshot-GC follow-up.
- **Broker-clock-derived cycle instant for the tick cutoff.** `run_tick`'s `now` is pre-existing;
  rejecting explicit `--start`/`--end` under `--refresh` closes the caller-controlled half.
- **A versioned provider/producer fingerprint in the dedup key and snapshot metadata** (adapter
  version, locked vendor-library version, feed/endpoint class). `provider.name` is the identity
  the registry and every snapshot already use; the adapters are pinned by `uv.lock`, and the two
  bundled ones are not hot-swapped under one name. Real as a provenance refinement; belongs with
  the security-master follow-up (feed identity, finalized-bar metadata, data-quality checks).
- **Binding the live data provider into the signed live authorization.** Partially taken: the
  live provider is now a separate setting with no default (fail-closed), which makes the choice
  an explicit human act. Folding it into the signature payload is a live-gate change (CODEOWNERS)
  and is filed as a follow-up.
- **New-entrant seasoning for time-varying universes.** A constituent added today with fewer
  sessions of history than the deepest floor blocks the lane until it seasons. Today's operational
  universes are static snapshots; when membership changes become live, eligibility-until-seasoned
  is the follow-up (it needs listing dates, a security-master concern).
- **An immutable prepared-tenant bundle consumed by the tick** (no re-resolution at tick time).
  If a universe timeline changes between plan and tick, the tick requests symbols the snapshot
  lacks and the #452 wall halts the account fail-closed and loud. Accepted as the failure mode for
  this change; the lane-parity test depends on the tick's own resolution call and the bundle is a
  larger refactor of both tick helpers — follow-up.
- Snapshot GC; intraday timeframes; a live operator unit; alt-data refresh (#472); Alpaca provider
  pagination; `ALGUA_ALERT_CMD` as a deployment prerequisite; multi-PR staging (the tuition run is
  the soak; the paper unit is the only consumer).
- **Session marker records target session S while the decision bar is S−1 in the after-close
  window; two retries of the same session can decide on different bars across UTC midnight** —
  `snapshot_id` on the marker/tick row is what makes this auditable; part of the
  session-addressed-ticks follow-up.

## Test plan

- `tests/test_data_refresh.py`: resolve hit → no provider call; miss → ingest; key normalization
  (symbol order/case; `end` changes → miss); **a same-key candidate that fails the current
  coverage (older `require_bar_on`, short history, tampered content with equal row count) is not
  reused and a fresh snapshot is minted**; clip drops rows ≥ `end`; `missing` / `stale` /
  `misdated` / `short_history` each raise and mint nothing; provider exception propagates with no
  snapshot; provider identity comes from `provider.name`.
- `tests/test_cli_data.py`: `refresh-bars` envelope; a stale symbol is `code: "refresh_failed"`.
- `tests/test_lane_refresh.py`: plan resolves the gate-bound universe + held + history floors;
  a strategy without a gate row is isolated; `lane_symbols` union with broker net;
  `require_bar_on` = previous session (weekday and weekend cases).
- `tests/test_paper_run_all.py` / `test_cli_live.py`: `--refresh` xor `--snapshot`; `--refresh`
  with `--end` rejected; refresh runs after fill ingest and **before** reconcile; symbol set
  includes a broker-only orphan and a held-but-dropped symbol; one strategy's plan failure skips
  it and ticks the sibling; **all strategies failing planning is `cycle_plan_failed`, exit 1**;
  refresh failure → `refresh_failed`, exit 1, zero ticks; the tick row carries `snapshot_id`.
- `tests/test_fleet_health.py`: decision 1 behind → `ok`; 3+ behind **as of now** → `stale`
  even though the tick itself is fresh; `decision_ts=None` → falls through; unparseable /
  negative → `stale`; `fleet health` exits non-zero.
- `tests/test_operator_jobs.py` / `test_cli_operator.py` / `test_operator_schedule.py`: template
  is `--refresh`; trailing `--snapshot X` is `command_mismatch`; **a payload with ticked
  strategies but no `snapshot.id` is not completed**; unit file asserts `run-all --refresh` and no
  `ALGUA_PAPER_SNAPSHOT`; marker records `snapshot_id`.
- `tests/test_registry_db.py` / `test_family_registry.py`: `SCHEMA_VERSION == 45`.

## Files

New: `algua/data/refresh.py`, `algua/cli/lane_refresh.py`, `tests/test_data_refresh.py`,
`tests/test_lane_refresh.py`.
Modified: `algua/cli/data_cmd.py`, `algua/cli/errors.py`, `algua/cli/paper_cmd.py` (CODEOWNERS),
`algua/cli/live_cmd.py`, `algua/contracts/types.py`, `algua/execution/fleet_health.py`,
`algua/execution/order_state.py`, `algua/registry/db/` (CODEOWNERS: v45), `algua/operator/jobs.py`,
`algua/operator/session_runner.py`, `algua/operator/schedule.py`, `algua/config/settings.py`,
`deploy/systemd/{algua-paper.service,algua.env.example,README.md,install-user-units.sh}`,
`CLAUDE.md`.
