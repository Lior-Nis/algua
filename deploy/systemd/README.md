# Algua operator — systemd packaging

A `oneshot` service fired by a `timer` drives the always-on **paper** operator (#486). Each firing
runs `algua operator run`, which decides — via the XNYS calendar gate, a per-session idempotency
marker, and a git-dir-anchored run lock — whether to actually run the wrapped driver command. A
weekend/holiday firing, a re-fire of a session already completed, or an overlap with a still-running
sibling all no-op cleanly.

- `algua-paper.{service,timer}` — the daily paper trading cycle (`--job paper`), ~30m after the US
  close (21:30 UTC). **Daily on purpose, NOT weekdays-only**: the operator wrapper's XNYS calendar
  gate no-ops non-session firings, so the timer never re-encodes the exchange calendar.
- `algua-research.{service,timer}` — the autonomous **research producer** cycle at **factory
  cadence: every 2 hours** (12/day; see
  `docs/superpowers/specs/2026-08-10-strategy-factory-design.md`): a Codex agent ideates → authors
  → backtests/walk-forwards/sweeps → gates (preview `research promote`) up to `candidate` in an
  explore-isolated worktree. See "Autonomous research loop" below.

The overnight **research merge-back** (consumer) timer is still intentionally **not shipped** — see
"Deferred merge-back timer" below.

## Install

Assumes the app is deployed at `/opt/algua` with its virtualenv at `/opt/algua/.venv`, and that
`algua` is on `PATH` for the operator's subprocess (a `.venv/bin` entry or a symlink).

1. Copy the environment file and fill it in (mode `0600`, it holds secrets):

   ```sh
   sudo install -D -m 0600 deploy/systemd/algua.env.example /etc/algua/algua.env
   sudo editor /etc/algua/algua.env
   ```

2. Set `ALGUA_PAPER_SNAPSHOT` in the env file — it is expanded into the paper unit's `ExecStart`
   (`… paper run-all --snapshot ${ALGUA_PAPER_SNAPSHOT}`) as the lone variable token of the `paper`
   job's canonical argv. The trailing command in `ExecStart` **must exactly match** that argv
   template (`algua paper run-all --snapshot {snapshot}`) — an exact-arity structural match — or the
   wrapper fail-closes with `command_mismatch`. If you launch via `uv run algua …`, adjust the
   entrypoint AND the job's `argv_template` to match; do NOT append ad-hoc flags to the always-on
   `ExecStart`.

3. Copy the units into place and enable the timer:

   ```sh
   sudo cp deploy/systemd/algua-paper.service deploy/systemd/algua-paper.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now algua-paper.timer
   ```

Inspect with `systemctl list-timers 'algua-*'` and `journalctl -u algua-paper.service`.

## Install as user units (single-box, no /opt deploy)

When the repo lives in a home directory (no `/opt/algua`, no root), install the same units as
**systemd user units** with the idempotent installer:

```sh
deploy/systemd/install-user-units.sh            # render + install + daemon-reload
deploy/systemd/install-user-units.sh --dry-run  # print the rendered units, write nothing
```

For each of `algua-research.{service,timer}`, `algua-paper.{service,timer}`, `algua-web.service`
it renders the `/opt/algua` template to `~/.config/systemd/user/<name>`, replacing `/opt/algua`
with the actual repo root (resolved from the script's own location), dropping
`EnvironmentFile=/etc/algua/algua.env` when that file doesn't exist (keeping the
credential-scrubbing `UnsetEnvironment=` lines), and rewriting `WantedBy=multi-user.target` →
`default.target` (timers keep `timers.target`). It then runs `systemctl --user daemon-reload` and
**prints** — never executes — the per-unit enable commands. Re-running is safe: installs are
atomic overwrites, so the installed units can never drift from the repo templates for longer than
one re-run. Use `loginctl enable-linger <user>` so user timers fire without an active login, and
inspect with `systemctl --user list-timers 'algua-*'` / `journalctl --user -u algua-research.service`.

**Factory cadence (slice 1).** Research fires **every 2h** (`OnCalendar=*-*-* 00/2:00:00 UTC`,
overlaps skip via the launcher's non-blocking flock); paper ticks **daily at 21:30 UTC**
(post-close; the calendar gate no-ops non-sessions — do NOT make it weekdays-only). Enable BOTH so
paper evidence starts accruing the moment the first strategy reaches the book.

**Mutual exclusion around the paper tick (#316).** `paper merge-back` and `paper trade-tick`/
`run-all` are mutually exclusive **by operator discipline**, not by a shared kernel lock. Until
slice 3 automates that discipline, do not run a manual `paper merge-back` around the 21:30 UTC
paper tick window — run merge-backs well clear of it (the research timer's schedule is already
offset from the paper window).

## Why the wrapper, not just the timer

**Calendar gate.** The timer fires every calendar day at a fixed wall-clock time, but the market is
open only on XNYS sessions. `algua operator run` resolves the most-recent *completed* XNYS session as
of now (respecting early/half-day closes) and no-ops when there is none (weekend, holiday, or before
that day's close). This makes the exact `OnCalendar=` time non-critical and immune to DST drift — the
gate, not the clock, decides whether a cycle acts.

`Persistent=true` covers a **missed window after downtime** — if the box was off across the scheduled
fire, systemd runs the unit once on next boot to cover the window it slept through. It is **NOT** a
failed-run retry mechanism: a unit that ran and exited non-zero is not re-executed. Retry of a
failed/crashed session comes solely from the marker being left unwritten — the **next** scheduled fire
re-attempts (the gate is `due` again).

**Session-idempotency marker.** `operator_sessions.json`, written beside the DB
(`$(dirname ALGUA_DB_PATH)`), records the last session each job ran (an enriched audit entry binding
the full canonical argv). A re-fire of a session already completed is suppressed (`already_ran`). The
marker is written ONLY after a run the job's completion predicate accepts — NOT bare `rc==0`: a
`deferred:true` cycle (the driver chose not to trade) exits 0 but is left unrecorded so the next fire
retries. A failed run likewise leaves it untouched, with the driver's own ingest-and-reconcile-before-
trade as the double-trade backstop. Marker reads **FAIL CLOSED**: an absent marker is benign (run), but
a present-but-corrupt marker (`marker_corrupt`) alerts and exits 1 — the operator must inspect/repair
the file before the loop resumes, so "must not run twice" holds unconditionally.

**Run lock (`operator.lock`).** Acquired at the repo's per-worktree git dir (not `db_path.parent`),
non-blocking, held across gate → run → record. It serializes overlapping fires (a slow cycle still
running when the next window opens) and — once the deferred research job lands — run-all vs merge-back.
An overlap within the job's expected duration is a benign no-op (`reason:"locked"`, exit 0, no alert);
a holder wedged **past** that grace is surfaced via an `operator_lock_stuck` alert (the fleet would
otherwise quietly stop trading). The kernel releases the flock on holder death, so a hard kill never
wedges the next fire.

**Direct-invocation residual risk.** `operator.lock` serializes runs that go through `operator run`;
it does **not** guard a *direct* `algua paper run-all` / `paper trade-tick` / (future) `merge-back`
that bypasses the wrapper. During the always-on window, **direct/manual invocation of these drivers is
prohibited by operator policy** — an operator-discipline contract (matching the `paper run-all`
docstring), not a kernel-level mutual-exclusion guarantee. The deeper backstop is run-all's
ingest-and-reconcile-before-trade: an accidental overlap reconciles/defers, it does not
blind-double-trade.

**Alert hook.** When a run fails, the wrapper classifies it (`global_halt` / `breach` / `job_failed`)
and calls the operator alert hook; anomalies (`marker_corrupt`, `calendar_out_of_bounds`,
`operator_lock_stuck`, `session_gap`, `completion_unconfirmed`, `unknown_job`, `command_mismatch`) fire
their own alert kinds. Every alert always lands as a structured `operator_alert` log record; if
`ALGUA_ALERT_CMD` is set, the alert JSON is also piped to that command — **split with `shlex` and run
with `shell=False` under a 10s timeout** (no shell interpolation of the payload; wrap any
pipe/redirect in a script). Delivery is best-effort and never crashes the run. A `calendar_out_of_bounds`
alert means the exchange calendar has run past its precomputed horizon — **refresh the calendar /
upgrade `exchange-calendars`**.

## Autonomous research loop

`algua-research.{service,timer}` runs the candidate **producer** — the piece that was "out of scope"
when only the paper operator shipped. It uses the **explore-isolated** topology. Each firing runs
`.codex/scripts/run-research-loop.sh`, which:

- creates a throwaway git worktree on `research-run/<stamp>` so the **code** the agent authors never
  touches your working tree or `main`;
- seeds a **per-run scratch funnel** from the real registry via sqlite's consistent online backup, and
  wires the agent to it — `ALGUA_DB_PATH`, `ALGUA_KNOWLEDGE_DIR`, `ALGUA_MLFLOW_TRACKING_URI` point at
  scratch; only `ALGUA_DATA_DIR` (the immutable snapshots) is the real one, shared read-only. So the
  agent's walk-forward/sweep/promote see the REAL accumulated breadth, families, and burned holdouts (a
  realistic pass/fail **preview**), but every write lands on the copy — **exploration can never mutate
  the authoritative FDR ledger / family graph / holdout rows**, and wasted search never taxes the real
  funnel;
- runs `codex exec` (bounded by `TIMEOUT`, default 45m) to drive ideate → author → walk-forward/sweep →
  `research promote` (preview) up to `candidate`, then writes `run-report.md` on the branch with the
  exact `paper merge-back` command for any candidate whose preview passed.

**Why isolated and not "direct authoritative".** `codex exec` runs unsandboxed and could edit
`promotion.py`/`gates.py`/`fdr_lord.py` in its mutable worktree; strategy code and gate code are the
same Python package, so if the agent ran against the real funnel it could execute altered gate logic
against the real registry. CODEOWNERS only gates *merges to main*, not a live agent's local execution.
Exploration therefore runs on a throwaway copy, and the **only** path to the authoritative funnel is
`paper merge-back` — whose diff-policy rejects gate-core edits *before* the merge and whose promote runs
trusted (main + allowlisted diff) code. That is the trusted reconciler.

**Prerequisites.** `codex` must be on `PATH` and **authenticated** for the user the unit runs as
(`codex exec` is invoked headless with `--dangerously-bypass-approvals-and-sandbox` inside the isolated
worktree). Enable with `sudo systemctl enable --now algua-research.timer`.

**The authoritative step (human-reviewed).** A passed *preview* is a candidate, not a promotion. To
commit it to the real funnel + main, review the branch, then run the `paper merge-back --branch
research-run/<stamp> --strategy <name> --universe <u> --start D --end D` line from the run report. That
is the trusted, authoritative promote (real breadth tax, real single-use holdout burn, family mint) +
gated code merge + paper intake. The agent **cannot go live** (human-signed cryptographic wall) and
cannot merge the CODEOWNERS-protected integrity files.

**Cadence and the FDR budget.** The timer runs at **factory cadence: every 2h, 12 runs/day**
(`OnCalendar=*-*-* 00/2:00:00 UTC`). That rate is safe for the funnel *because* exploration is
scratch: a cycle's `research promote` is a preview that records NO breadth and burns NO holdout on
the authoritative funnel, so it never consumes the FDR budget (**≤16 promotion-eligible binding
tests / rolling 365 days**). That budget is spent only by a human-run `paper merge-back` — which
records a metered breadth tax that raises the promotion Sharpe bar for ALL future strategies. So run
the producer as often as you like; pace the *merge-backs*. The holdout is single-use *per strategy*,
so the loop keeps authoring new strategies on the existing snapshot — the breadth/FDR tax at
merge-back, not the holdout, is what makes it progressively harder. The real per-cycle costs are
compute/API (tune `N_HYPOTHESES` / `TIMEOUT` in the env file; Codex plan rate-limit hits are
expected at 12/day and are recorded per run in the digest — see below) and disk (auto-pruned; see
below).

**Run digest (feedback contract, slice 1).** After every firing — success, codex failure, or
timeout — the launcher appends **one JSON line** to the durable, authority-side digest at
`data/research-runs.jsonl` (beside the authoritative DB; `ALGUA_RESEARCH_DIGEST_PATH` overrides).
Fields: `stamp`, `branch`, `thesis`, `exit_code`, `timed_out` (bool, `timeout` exit 124), `wall_s`,
`n_strategy_files` (strategy files in the driver's commit), `hypotheses` (sanitized titles from the
run-report's machine-readable trailer — charset-filtered, ≤120 chars, ≤40), `preview_gate`
(`{"passed": bool, "failed_checks": [...]}` or `null`), `trailer_parse_error` (bool),
`rate_limited` (bool, grepped from the in-worktree codex log), `report`
(`research-run/<stamp>:run-report.md`). This is the improve-algua backlog: `trailer_parse_error`/
`rate_limited`/`timed_out` clusters are machinery bugs to fix, `preview_gate.failed_checks`
clusters show where hypotheses die. The digest also feeds the next runs' anti-dup prompt context
(recent hypothesis titles, injected as sanitized untrusted data). A digest write failure warns but
never fails the run; the digest never stores raw report prose.

**Thesis rotation (slice 1).** When `THESIS` is not explicitly set, the launcher rotates
deterministically through `.codex/research-themes.txt` (one thesis per line;
`index = (days_since_epoch * 12 + hour_of_day / 2) % line_count`, i.e. one theme per 2h slot).
Edit that file to steer the factory's alpha-category mix; an explicit `THESIS`/`--thesis`
overrides rotation entirely.

**Contention.** The driver holds a non-blocking `data/research-loop.lock` for the whole cycle and skips
(no-op) rather than queue if another research cycle holds it — so the 2h cadence can never stack
overlapping cycles (with `TIMEOUT=45m` + `SYNC_TIMEOUT=5m` under `TimeoutStartSec=4200`, each slot has
~50m of headroom anyway). It does **not** contend with the paper operator (research writes only
scratch; the sole authoritative writer, `paper merge-back`, has its own `merge_back.lock` + operator
policy), and no research firing coincides with the 21:30-UTC paper slot.

**Failure propagation.** The driver captures the `codex exec` exit code and propagates a non-zero
(timeout=124, or an auth/runtime error) so the systemd unit **fails** rather than silently reporting a
no-op cycle as success. `TimeoutStartSec` (4200s) must stay above the driver's `TIMEOUT + SYNC_TIMEOUT`.

**Worktree cleanup (automatic).** Each cycle leaves its `research-run/<stamp>` worktree in place for
review, then a later cycle **auto-prunes** worktrees older than `RESEARCH_WORKTREE_RETENTION_DAYS`
(default 7) at startup — reclaiming only the disposable venv + scratch; the authored code persists on
its `research-run/<stamp>` branch after the worktree dir is removed. Lengthen the window (env file) if
you want more review time, or reap on demand with `git worktree remove ../algua-research-<stamp>`.

## Deferred merge-back timer

A `merge-back` (candidate **consumer**) always-on timer is still a filed follow-up, intentionally not
shipped here. A static `ExecStart` cannot know *which* candidate branch/strategy/universe/window to merge
back — that requires a candidate **selection** mechanism (a queue / pointer file the unit's `ExecStart`
resolves before invoking merge-back). The producer above now enqueues candidates (as `research-run/<stamp>`
branches + `run-report.md`), so what remains is the selection queue; the generic `operator run` wrapper
accepts the consumer later as a new `OPERATOR_JOBS` manifest entry + selection mechanism + unit pair, with
**no wrapper change**.
