# Algua operator — systemd packaging

A `oneshot` service fired by a `timer` drives the always-on **paper** operator (#486). Each firing
runs `algua operator run`, which decides — via the XNYS calendar gate, a per-session idempotency
marker, and a git-dir-anchored run lock — whether to actually run the wrapped driver command. A
weekend/holiday firing, a re-fire of a session already completed, or an overlap with a still-running
sibling all no-op cleanly.

- `algua-paper.{service,timer}` — the daily paper trading cycle (`--job paper`), ~30m after the US
  close (21:30 UTC).

The overnight **research merge-back** timer is intentionally **not shipped** — see "Deferred research
timer" below.

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

## Deferred research timer

A research-cycle / `merge-back` always-on timer is a filed follow-up, intentionally not shipped here.
A static `ExecStart` cannot know *which* candidate branch/strategy/universe/window to merge back — that
requires FIRST solving candidate **selection** (a queue / pointer file the research unit's `ExecStart`
resolves before invoking merge-back) AND that selection's *producer*, the research authoring loop that
enqueues candidate branches (out of scope). Until both exist, a research timer would no-op forever. The
generic `operator run` wrapper accepts it later as a new `OPERATOR_JOBS` manifest entry + selection
mechanism + unit pair, with **no wrapper change**.
