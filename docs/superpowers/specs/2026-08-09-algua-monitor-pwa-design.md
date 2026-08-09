# Algua Monitor — mobile-first read-only PWA (design)

## Context

Algua's autonomous research→paper loop runs unattended, but the only way to see fleet health,
strategy lifecycle position, and metrics is the CLI. This design adds a mobile-first monitoring
frontend. Prior specs deliberately deferred a web dashboard while building the system
"dashboard-ready" (SQL-first registry, JSON-on-stdout CLI, `tick_snapshots` annotated in
`algua/registry/db.py` as the future dashboard's equity read). There is no HTTP layer anywhere in
the repo today — this adds the first one, outside the `algua` package.

All decisions below were brainstormed with the human operator and gate-reviewed (3-round
adversarial design review on slice A).

## Decisions

- **Read-only monitor.** No ops actions from the phone; go-live stays ssh-signed off-phone.
- **V1 screens:** Home (fleet health rollup), Funnel (strategies by lifecycle stage), Strategy
  detail (equity curve, metrics, gate checks, transitions, recent orders), Activity feed, Idea
  pool. Deferred: funnel-governance panel (FDR throttle/cohort state), cross-strategy gate
  explorer.
- **Access:** Tailscale-only. uvicorn binds `127.0.0.1:8787`; `tailscale serve` fronts it with
  HTTPS (`https://<box>.<tailnet>.ts.net`) — required for service-worker/push. The tailnet is the
  auth; no auth code; never bind 0.0.0.0.
- **Stack:** new top-level `web/` dir: FastAPI backend + Vite/React/TypeScript SPA (PWA via
  vite-plugin-pwa, charts via uPlot); FastAPI serves the built SPA. One systemd unit
  (`algua-web.service`).
- **Data seam:** the web backend consumes ONLY the algua CLI via subprocess (the golden rule).
  The two missing read surfaces become new read-only CLI commands in algua itself (slice A below)
  — they serve agents as much as the frontend. No module imports, no direct SQLite from web.
- **Freshness/push:** data changes ~once per NYSE session, so no streaming. A background poller
  (~10 min) refreshes hot caches AND diffs fleet health to fire web-push alerts (pywebpush +
  VAPID) on worsening transitions. Pull-fresh on app open otherwise.
- **Design language:** extends the algua doc aesthetic (dark, Martian Mono / JetBrains Mono,
  dense) with tokens from `docs/algua-architecture.html`; fonts self-hosted (offline PWA).

## Critical constraints

1. **Root `uv.lock` is identity-bearing** — `algua/provenance/lockfile.py` derives
   `dependency_hash` (gate/live-authorization artifact identity) from it. Web deps live in a
   standalone uv project `web/pyproject.toml` + `web/uv.lock`; never `uv add` web deps at root.
2. **CODEOWNERS:** `algua/registry/store.py` and `algua/cli/_common.py` stay untouched; the gate
   read accessor lives in a new `algua/registry/gate_history.py`.
3. **CLI JSON quirks:** `registry list`/`fleet status`/`audit log` emit bare arrays; others emit
   `{ok:true,...}`; errors `{ok:false,error,code}`; `fleet health` exits 1 while emitting healthy
   JSON — the web runner parses stdout regardless of exit code.
4. **`holdout_returns.returns_blob` is SENSITIVE by design** — no new surface may reach it.
5. **CLI startup is 1.5–4 s** — exec `.venv/bin/algua` directly, poller pre-warms hot payloads,
   TTL memo + per-key lock + small subprocess semaphore, `asyncio.gather` for multi-command
   endpoints.
6. **Empty states are first-class** — `tick_snapshots` is nearly empty today; charts render an
   "awaiting tick history" placeholder (uPlot throws on degenerate scales).
7. **SQLite contention with the paper/research timers** — on CLI failure serve last-good cache
   with `stale:true`; a failed poll skips the alert diff (never treated as recovery).
8. **iOS push** requires iOS 16.4+, the PWA installed to the Home Screen, and a user-gesture
   permission prompt.

## Slice A — the two new algua CLI commands (this PR)

Both commands are strictly read-only: no writes, no transitions, no token minting, no ledger rows.

### `algua fleet series <name> [--since ISO] [--limit N=500] [--lane paper|live]`

- Command in `algua/cli/fleet_cmd.py`; accessor
  `tick_snapshot_series(conn, strategy_id, strategy_name, *, lane, since, limit)` in
  `algua/execution/order_state.py` beside `latest_tick_snapshot`.
- Data rows are selected by `strategy_id` (identity boundary, from `repo.get(name)`);
  `n_legacy_excluded` counts rows `WHERE strategy=<name> AND (strategy_id IS NULL OR lane IS
  NULL)` — legacy pre-v21 rows predate `strategy_id` and are inadmissible, as everywhere else.
- All reads inside an explicit `BEGIN DEFERRED` read transaction (plain `with conn:` does not pin
  a snapshot for SELECTs); never IMMEDIATE — this is a reader.
- **Segmented by lane** — one strategy can hold paper AND live rows; splicing them is a false
  curve. Stable payload keys `series: {"paper": <list|null>, "live": <list|null>}` +
  `lane_filter`: null = lane not requested, `[]` = requested and known empty. Row fields:
  `id, tick_ts, recorded_at, equity, peak_equity, reconcile_ok` (no `positions` — bloat).
- **Filter-then-limit on parsed time, end to end:** parse each `tick_ts` to aware UTC (naive →
  UTC), skip + count unparseable (`n_unparseable`) and invalid non-null lanes (`n_invalid_lane` —
  lane discipline is writer-enforced only), apply `--since`, sort `(parsed_ts, id)`, keep the
  newest N per lane, emit ASC with per-lane `truncated`. No SQL text ordering/comparison anywhere
  (tick_ts writers accept arbitrary strings). `--limit` bounded 1..5000.
- Unknown name → `{ok:false, code:"not_found"}`; registered with zero ticks → `ok:true` + empties.

### `algua registry gates <name> [--limit N=20]`

- Command in `algua/cli/registry_cmd.py`; accessors in NEW `algua/registry/gate_history.py`
  (pure reads). `--limit` bounded 1..200. Both SELECTs inside one explicit `BEGIN DEFERRED`
  transaction (consistent snapshot across the two ledgers).
- Two separate explicit column lists (verified against the real schema; never `SELECT *`):
  - `gate_evaluations`: id, passed, actor, created_at, consumed, pit_ok, pit_override,
    holdout_n_bars, min_holdout_observations, holdout_frac, n_funnel, own_lifetime_combos,
    windowed_total_combos, funnel_window_days, breadth_provenance, data_source, snapshot_id,
    code_hash, config_hash, dependency_hash, period_start, period_end, fdr_binding, fdr_p_value,
    fdr_alpha_level, fdr_rejected, fdr_test_index, fdr_cohort (fdr_* migration-added → nullable),
    decision_json→`decision`.
  - `forward_gate_evaluations`: id, passed, n_forward_observations, min_forward_observations,
    session_coverage, realized_sharpe, holdout_sharpe, degradation_factor, sharpe_floor,
    realized_vol, min_forward_vol, realized_max_drawdown, max_forward_drawdown, first_tick_id,
    last_tick_id, first_tick_ts, last_tick_ts, max_staleness_sessions, n_reconcile_failures,
    n_concurrent_forward, account_id, code_hash, config_hash, dependency_hash, actor, consumed,
    created_at, decision_json→`decision`.
- **`decision_json` is never emitted raw — two-layer allowlist projection** (the durable defense
  for the holdout-secrecy invariant): a key survives only if (1) it is on a MATERIALIZED frozen
  per-ledger allowlist constant (hand-written from the current `GateDecision` /
  `ForwardGateDecision` fields; never runtime-derived) AND (2) its value is a scalar, a dict of
  scalars, or — for `checks` — a list of dicts keeping only name/op/threshold/value/passed.
  Everything else is dropped and reported in `decision_dropped_keys`. A **golden drift test**
  fails when either decision dataclass gains a field outside allowlist ∪ explicit-excluded,
  forcing conscious projection review. Corrupt decision_json → `decision:null` +
  `decision_error`, sibling rows intact.

### Tests

Hermetic CLI tests (CliRunner + `ALGUA_DB_PATH`, per `tests/test_fleet_health.py`) pinning:
event-time ordering with id/tick_ts disagreement; lane segmentation + null-vs-empty shape;
legacy-row exclusion count; invalid-lane skip count; filter-then-limit semantics; unparseable
tick_ts skip; truncated flags; not_found; empty → `ok:true`; decision projection (vector key and
non-allowlisted scalar key dropped + reported); golden drift test; fdr_* nulls on legacy-shaped
rows; `returns_blob` absent from payloads; one bounds test through `algua.cli.main.main`
(usage-error → JSON envelope).

## Later slices (separate PRs)

B: `web/` uv project + CLI-runner seam + `/api/fleet` + Home screen (thin vertical proof).
C: remaining read endpoints + Funnel/Detail/Activity/Ideas screens.
D: PWA (manifest/icons/service worker) + poller cache-refresh + systemd unit + tailscale serve.
E: web-push pipeline (VAPID, subscribe endpoints, poller diff/dedup, on-device verification).
F: polish (stale badges, error states, README, frontend checks).

API surface (web backend, all normalized to `{ok, data, fetched_at, stale}`): `/api/fleet`,
`/api/strategies` (poller-cached); `/api/strategy/{name}`, `/api/strategy/{name}/series`,
`/api/activity`, `/api/ideas` (on-demand TTL); `/api/push/key`, `/api/push/subscribe`,
`/healthz`, SPA static fallback.

Push: poller severity diff (ok < idle < stale < drift < halted) on operational stages only;
global-halt trip; fleet-recovered; 24 h per-(strategy,health) dedup persisted in
`notify_state.json`; baseline poll after restart never notifies.
