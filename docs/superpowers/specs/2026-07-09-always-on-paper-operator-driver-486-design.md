# Always-on paper-operator driver — systemd timer + session-idempotency + alerts (#486)

**Status:** Draft (planner) — 2026-07-09 (rev. 4, third-round GATE-1 blocking findings folded)
**Issue:** #486 (paper-operator **Slice 6**, epic #318 / the autonomous-paper-operator design §5 Group 4)
**Depends on:** #316 `paper run-all` (merged) and #485 `paper merge-back` (merged) — both drivers exist.

## 1. Scope (Group 4 of the operator design — the paper timer, nothing more)

This issue builds the **always-on clock** that fires the already-built `paper run-all` driver
unattended. It adds NO trading logic — the driver already exists and is the single-shot command
systemd repeats. Deliverables (a subset of the design doc §5 Group 4):

1. **One systemd timer/service pair** — for `paper run-all` (post-close). The research-cycle /
   `paper merge-back` timer is **explicitly deferred to a follow-up** (round-3 fix #1, §5): a static
   `ExecStart` cannot resolve *which* candidate branch/strategy/universe/window to merge back — that
   selection is the output of the research authoring loop (out of scope, §5), so shipping a research
   timer now would build a component that can never be exercised end-to-end. The generic wrapper (D1)
   is authored so the deferred research timer drops in later via a manifest entry + a selection
   mechanism + unit files, with **no wrapper change**.
2. **NYSE-calendar gate** — fire only on a *completed* XNYS trading session; reuse the existing
   session-clock (`algua.calendar.MarketCalendar`), never wall-clock.
3. **Session-idempotency guard** — a given session must not be run twice (crash/restart safe **and**
   concurrent-invocation safe).
4. **Structured logs + an alert hook** on job failure / global-halt / kill-switch trip / a corrupt
   idempotency marker / a wedged (stuck-lock) operator / a calendar-out-of-bounds anomaly / a
   `--job`↔command-identity mismatch.

Each `--job` key is **bound to a declared FULL canonical argv (not a bare head prefix) and a
completion predicate** (a small in-repo job manifest, §D7) so the wrapper never marks a session done
off an unrelated, altered, or non-completing command (round-3 fix #4, §D7).

## 2. Design decisions (forks resolved)

**D1 — One generic wrapper command; the paper timer ships, the research timer defers; each `--job`
bound to a declared full-argv identity.** Add a single `algua operator run --job <key> -- <command...>`
that (a) resolves `<key>` against the **job manifest** (§D7 — fail-closed on an unknown key),
(b) validates the trailing `<command...>` against that job's **full canonical argv template**
(exact-arity structural match, not a prefix — fail-closed on any deviation, round-3 fix #4),
(c) evaluates the session gate for `<key>`, (d) runs `<command...>` as a subprocess only when the
session is due, (e) records the session marker **only when the driver positively signals the session
completed** — not on bare `rc==0` (§D4, round-2 fix #1), (f) fires the alert hook on any non-clean /
anomalous outcome. The wrapper is job-parameterized and stays decoupled from pipeline internals — the
per-job *knowledge* (which exact command, what "completed" means, expected duration) lives
declaratively in the manifest, and the whole thing stays testable with an injected runner. **Today
the manifest ships exactly one job (`paper`)**; the deferred research job (round-3 fix #1) is a future
manifest entry + selection mechanism + unit pair, requiring no code change to the wrapper.

> **Numbering note.** rev-2 folded the FIRST GATE-1 round (its fixes were, confusingly, labelled
> "GATE-1 fix #1/#2/#3": run lock / marker lock / corrupt marker). rev-3 folded the SECOND round as
> **"round-2 fix #1…#5"** (completion-signal-not-rc0 §D4 / stuck-lock alert §D2 / command-identity
> §D7 / marker fsync §D3 / calendar-out-of-bounds §D5). This **rev-4** folds the THIRD round as
> **"round-3 fix #1…#5"**: **#1** defer the research timer (scope to paper only, §1/§3.5/§5) — a
> follow-up must first solve candidate branch/strategy selection; **#2** anchor `operator.lock` at
> the per-worktree git dir, not `db_path.parent` (§D2/§3.1/§3.3); **#3** the run lock covers *wrapped*
> runs only — direct invocation is prohibited-by-policy with a documented residual risk, not a full
> mutual-exclusion guarantee (§D2/§3.5); **#4** bind the FULL canonical argv (exact-arity match), not
> a head prefix (§D7/§D3/§3.3); **#5** the multi-session-gap alert fires at `skipped > 0`, not `> 1`
> (§D5/§3.3).

**D2 — Subprocess, not in-process import; a git-dir-anchored repo-global run lock serializes wrapped
operator runs.** The wrapper drives `paper run-all` via `subprocess`, not a Python import. This
(a) keeps `operator_cmd` free of any `cli->cli` sibling import (the import-linter independence
contract), (b) matches systemd "single-shot" semantics, and (c) makes the **driver's exit code** the
unified alert signal — `paper run-all` already exits non-zero on breach / kill-switch trip /
global-halt / reconcile-halt, so an exit-code-based alert captures "failure / global-halt /
kill-switch trip" without the wrapper needing a registry connection. The wrapper additionally parses
the driver's stdout JSON to *classify* the alert kind (`halted` / `ok:false` / crash).

**Run lock — process-level mutual exclusion, anchored at the per-worktree git dir (round-3 fix #2).**
Two fires of the paper timer could overlap (a slow run-all still executing when the next window
opens), and — once the deferred research timer lands — run-all and merge-back are mutually exclusive
(run-all mutates paper trading state; merge-back mutates the git checkout AND runs promote/intake
against the registry). To make "must not run twice" hold at the process level, `operator run` takes a
**non-blocking repo-global exclusive `flock` on `operator.lock`**, acquired **before** the session
gate and held across the whole gate → subprocess → marker-record sequence, released in a `finally`.

The lock file is anchored at the repo's **per-worktree git dir** (`git rev-parse --absolute-git-dir`,
resolved *before* the lock), **NOT** at `get_settings().db_path.parent`. This mirrors the #485
merge-back lock's HIGH-4 fix exactly (`algua/cli/paper_cmd.py` resolves the git dir for
`merge_back.lock`): the run lock protects the shared mutation surface (the paper account + the git
checkout), which is shared **per working tree**, not per db. Anchoring at `db_path.parent` would let
two invocations on the SAME working tree but different `ALGUA_DB_PATH` take DIFFERENT lock files and
mutate the one shared account/checkout concurrently — the exact defect HIGH-4 fixed for merge-back.
The git dir lives outside the working tree, so the lock file never dirties `git status` (which a
future merge-back's clean-checkout precondition depends on). This lock uses the same discipline as
`algua/operator/gitops.merge_back_lock`: non-blocking `LOCK_EX | LOCK_NB`, kernel-released on holder
death (a hard kill never wedges the next fire). Acquiring the lock *before* the gate makes the
gate→run→record window atomic, so two wrapped invocations cannot both pass the gate and both trade.
Non-blocking (not blocking) is deliberate: the paper timer must not queue for hours behind a long
job and then fire post-close-stale — the session gate + the next scheduled fire handle catch-up. This
lock is outer to (and distinct from) merge-back's own `merge_back.lock`; the deferred research job
nests them cleanly (operator.lock outer, merge_back.lock inner) since both anchor at the same git dir.

**The run lock covers WRAPPED runs only — direct invocation is prohibited-by-policy (round-3 fix
#3).** `operator.lock` is acquired by `operator run`; it does **not** make the drivers themselves
mutually exclusive against a *direct* `algua paper run-all` (or `paper trade-tick`, or a future
`merge-back`) that bypasses the wrapper. Rather than teach every driver to acquire the same lock —
which would self-deadlock when the wrapper (already holding it) shells out to the driver, absent an
extra held-lock signalling channel — this design keeps the lock at the wrapper boundary and is
**explicit about the residual risk**: the systemd README (§3.5) documents that during the always-on
window, direct/manual invocation of `paper run-all` / `trade-tick` / (deferred) `merge-back` is
**prohibited by operator policy**. This matches the pre-existing operator-discipline contract in the
`paper run-all` docstring ("mutually-exclusive BY OPERATOR DISCIPLINE only … an advisory paper-lane
lock … is a filed follow-up"). The deeper backstop remains: `paper run-all` ingests real venue fills
and **reconciles before trading**, so an accidental overlap reconciles/defers rather than
blind-double-trading. The design does NOT claim `operator.lock` is a full kernel-level mutual-exclusion
guarantee across all invocation paths — only that it serializes runs that go through `operator run`.

**Global (not per-job) is intentional; a wedged holder is surfaced, not masked (round-2 fix #2 +
MEDIUM stuck-lock).** The lock is repo-*global* on purpose — once the research job lands, `paper
run-all` and `paper merge-back` are mutually exclusive, so a single global lock is the correct
coupling. The hazard a global no-op lock could hide is a **wedged holder**: if some operator run hangs
(a stuck subprocess, a paused debugger), it holds `operator.lock` indefinitely and *every* subsequent
timer fire silently no-ops — the fleet quietly stops trading with no signal. To surface this: on
acquisition the holder **writes its identity into the lock-file body** (`{"pid", "job", "started_at":
<iso-utc>, "host"}`, flushed + fsync'd) and truncates it back on release. On contention
(`BlockingIOError`) the wrapper opens the lock file **read-only** (flock is advisory — reading the
body needs no lock) to recover the holder metadata, then computes `held = now - started_at`:
- `held <= grace` (an ordinary scheduling overlap) -> **benign no-op**: emit
  `{"ok": true, "job", "ran": false, "reason": "locked", "session": null, "holder": {…}}`, exit **0**,
  no alert.
- `held > grace` (a **wedged operator**) -> `emit_alert("operator_lock_stuck", {"job", "holder_pid",
  "holder_job", "started_at", "held_seconds", "grace_seconds"})`, then STILL emit the same
  `reason:"locked"` no-op envelope and exit **0** (the alert is the surfacing mechanism; exiting
  non-zero every fire would flood systemd with failures while the real problem is the *holder*, not
  this fire).

`grace` is the job manifest's `expected_duration_seconds` for `--job` (§D7) — "held longer than the
expected job duration" is exactly the reviewer's threshold. If the holder metadata is missing/garbled
(a holder that died mid-write), treat `held` as unknown and conservatively alert `operator_lock_stuck`
with `started_at: null`, so a genuinely wedged holder is never silently masked.

**D3 — File-based session marker, NO schema bump; marker RMW is itself lock-guarded and fail-closed
on corruption.** The idempotency marker is the driver's own operational state, not registry-domain
state — so it lives in a JSON file (`operator_sessions.json`) **beside the registry db**
(`get_settings().db_path.parent`), exactly mirroring how the #485 merge-back **journal**
(`JsonlJournal(settings.db_path.parent)`) avoided a `db.py` schema bump and the CODEOWNERS-protected
`store.py`. Note the deliberate split that mirrors #485 precisely: the *idempotency state* (marker,
like the #485 journal) is per-db at `db_path.parent`; the *run lock* (`operator.lock`, like the #485
`merge_back.lock`) is per-worktree at the git dir (§D2, round-3 fix #2). Session idempotency is
correctly per-deployment/per-db (each db tracks whether *its* session ran); the mutual-exclusion lock
is correctly per-working-tree (it guards the one shared account/checkout). This also keeps the marker
driver-agnostic (a deferred research cycle has no tick-snapshot rows to key on).

**Atomic write with fsync-before-commit (round-2 fix #4).** `record` mirrors the exact durability
discipline `algua/operator/journal.py` already uses for the #485 merge-back journal (and the #184
data-store `fsync_file` primitives): write the full map to a sibling temp file, `flush()` +
**`os.fsync(tmp_fd)`** so the bytes are on stable storage, `os.replace(tmp, path)` (atomic rename),
then open the parent directory and **`os.fsync(dir_fd)`** so the rename's directory entry is durable
too. Without both fsyncs a power loss between `os.replace` and the page-cache flush could resurrect a
stale (or absent) marker and re-run an already-completed session. The fsync helpers are re-implemented
inline with stdlib `os` (as `journal.py` does) rather than imported from `algua.data.files`, so
`algua.operator` takes no new cross-module import edge (import-linter contract).

**Enriched marker schema binds the FULL canonical argv (MEDIUM thin schema + round-3 fix #4).** Rather
than a bare `{job: session-iso}` map, each entry is a small object: `{"session": <iso date>,
"recorded_at": <iso utc>, "command": [<full argv...>], "rc": 0, "host": <str>, "pid": <int>}`.
`command` is the **full canonical argv actually run** (not just an invariant head) — so the marker
records exactly what marked the session done, closing the round-3 fix #4 audit gap (a head-prefix
alone could not distinguish `algua paper run-all --snapshot X` from a poisoned trailing variant).
`last_session` reads only the `session` field (tolerant of extra keys); the rest is an operational
audit trail (when the mark was written, by which exact command, on which host). A legacy bare-string
value (`{job: "2023-11-24"}`) is still accepted by `last_session` (parsed as the session) for forward
tolerance, but `record` always writes the object form.

**Marker lock (rev-2 fix — marker RMW serialization).** `SessionMarker.record` is a read-modify-write
of a SHARED file (read the whole map, merge THIS job's entry, atomically replace). Even though the D2
run lock already serializes the only production writers, `record` guards its own RMW with a
**blocking exclusive `flock` on a dedicated `operator_sessions.lock`** (beside the marker at
`db_path.parent`; distinct from `operator.lock` so the two never re-enter/deadlock — the run lock is
held on a different fd/file at the git dir and record's lock is inner and blocking). This is
belt-and-suspenders defense-in-depth: it guarantees two near-simultaneous `record` calls for
*different* jobs cannot read-then-clobber each other's entries regardless of how `record` is reached
(a direct API/test caller that bypasses the CLI run lock, or a future second writer). The read side
(`last_session`) does not need the lock — `os.replace` makes each write atomic, so a reader never sees
a torn file. A concurrency test (two jobs recording near-simultaneously, asserting BOTH entries
survive) accompanies the round-trip / multi-job-isolation / atomic-overwrite tests.

**Corrupt-marker handling — fail-closed-with-alert, NOT fail-open (rev-2 fix — corrupt marker).**
Distinguish two cases:
- **Absent file** (first-ever run for this deployment) → benign, treated as "no marker" → the gate is
  `due` and the job runs. This is normal bring-up, not an anomaly.
- **Present-but-corrupt file** (unparseable JSON, wrong shape, or a non-ISO session value) → an
  **anomaly that must never happen** (atomic `os.replace` writes preclude a torn file). When it does,
  the operator can no longer trust its idempotency state, so it **fails closed**: it does NOT run the
  command, emits `emit_alert("marker_corrupt", ...)`, prints `{"ok": false, ...}`, and exits 1. The
  operator must remediate (inspect/repair/delete the file) and re-fire. This makes the "must not run
  twice" deliverable hold **unconditionally**. (Run-all's pre-trade reconcile remains a deeper
  idempotency backstop, but we do not *rely* on it for basic marker correctness — a corrupt marker is
  surfaced to a human, not silently swallowed.) `last_session` therefore returns `date | None` for
  absent/valid and **raises `MarkerCorrupt`** for a present-invalid file; `session_gate` converts that
  into `SessionGateDecision(due=False, reason="marker_corrupt")`.

**D4 — Marker recorded only on a POSITIVELY-COMPLETED run (a per-job completion signal, NOT bare
`rc==0`); retry is the NEXT scheduled fire, not systemd `Persistent` (round-2 fix #1).** Ordering
(inside the run lock): gate → run → *then, iff the job's completion predicate holds,* record.

The subtle hole bare `rc==0` leaves open is real and grounded in the driver: `paper run-all` exits
**0** on a `reconcile_deferred` cycle while emitting `{"ok": true, "deferred": true, "reconcile": …}`
(`algua/cli/paper_cmd.py`) — it did NOT operate the session, it *deferred* (a transient reconcile
condition). If the wrapper recorded the marker on that `rc==0`, the session would be marked done
forever and the real trading for that session would **never** happen — a benign no-trade cycle
permanently suppressing retry. So the marker is gated on a **job-specific completion predicate over
the parsed stdout JSON**, sourced from the job manifest (§D7):
- **paper**: `completed ⇔ rc == 0 AND payload.get("deferred") is not True`. A `deferred:true` cycle
  (rc 0) is NOT completed → marker left unwritten → the next scheduled fire retries the session; the
  wrapper emits `{"ok": true, "ran": true, "recorded": false, "reason": "deferred", …}` and exits 0
  (a deferral is not a failure — no alert, but no false completion either).
- **fallback (any job)**: if `rc==0` but stdout does NOT parse as JSON (an anomaly — the drivers
  always emit JSON), the wrapper cannot confirm completion, so it **does not record** and emits
  `emit_alert("completion_unconfirmed", {job, rc, stdout_head})` + `{"ok": true, "ran": true,
  "recorded": false, "reason": "completion_unconfirmed"}` (exit 0 — the command itself succeeded; we
  simply refuse to assert a completion we can't verify, so the next fire retries).
- *(The deferred research job's completion predicate — `completed ⇔ rc == 0`, since every merge-back
  terminal `rc==0` status is a genuine terminal decision and its branch-tip-SHA attempt-token journal
  is the authoritative crash-idempotency — lands with the research job in the follow-up, round-3 #1.)*

A crash or breach (non-zero) likewise leaves the marker unwritten, so the **next scheduled
`OnCalendar` fire** re-attempts the session (its marker is still absent → the gate is `due` again).
This is safe because `paper run-all` ingests real venue fills and reconciles *before* trading — a
re-fire after a partial run reconciles and defers rather than blind-double-trading, and a re-fire
under an engaged global halt no-ops. The file marker is therefore the cheap "don't bother firing twice
per COMPLETED session" gate; run-all's reconcile is the authoritative crash-idempotency. This matches
the design's "consistent with the existing tick-idempotency."

**Note — what `Persistent=true` does and does NOT do (GATE-1 Important #2).** `Persistent=true` makes
systemd fire a *missed* window ASAP after downtime — i.e. if the machine was off/asleep across the
scheduled `OnCalendar` time, the timer runs once on next boot to cover the window it slept through. It
is **not** a crash/breach retry mechanism: a unit invocation that *ran and exited non-zero* is NOT
re-executed by `Persistent`. Retry of a failed session comes solely from D4 — the unwritten marker
means the **next** scheduled fire re-attempts. The `Persistent=true` directive is present only so a
weekend/holiday-adjacent downtime still catches the first post-boot completed session; the session
gate makes the exact fire time non-critical, so at most one catch-up fire per missed window is
harmless (it no-ops if the session already ran).

**D5 — Session-only gating for the paper timer.** "Completed session" = the most recent XNYS session
whose **actual close** ≤ `now` (in exchange time), where the close is the exchange-calendar's
per-session close (respecting early/half-day closes), NOT a fixed wall-clock time. Weekends/holidays
and a not-yet-closed session map to the prior completed session; if that session's marker is already
recorded, the run no-ops. (A non-session daily cadence is YAGNI; add a mode later if a non-XNYS
strategy appears — design §9. The deferred research timer, round-3 #1, will reuse this same
session-gate at an off-hours `OnCalendar`.)

**Calendar-out-of-bounds is an anomaly, not a routine no-op (round-2 fix #5).** `xcals` raises
`MinuteOutOfBounds` when `now` falls outside the calendar's precomputed date range (the calendar was
built with a bounded horizon that the running clock has now exceeded — a real operational condition:
the deployment needs a calendar refresh / `exchange-calendars` upgrade). rev-2 folded this into
`target_session → None → reason "no_session"`, i.e. it was silently swallowed into the same benign
no-op as an ordinary weekend gap — so a calendar that had *run out* would make the operator quietly
stop firing forever with no signal. This is split:
- `target_session` **raises `CalendarOutOfBounds`** (wrapping `MinuteOutOfBounds`) instead of returning
  `None` on OOB. Genuine "no completed session in range yet" (e.g. `now` before the calendar's first
  session — a bring-up curiosity) still returns `None`.
- `session_gate` catches `CalendarOutOfBounds` → `SessionGateDecision(due=False, session=None,
  reason="calendar_out_of_bounds")`; `None` → `reason="no_session"` (benign).
- The CLI treats `calendar_out_of_bounds` like `marker_corrupt`: `emit_alert("calendar_out_of_bounds",
  {job, now})`, `{"ok": false, …}`, **exit 1** — fail loud so a human refreshes the calendar. It is
  NOT folded into the exit-0 `no_session` path.

**Multi-session-gap visibility fires at `skipped > 0` (round-3 fix #5 + MEDIUM Persistent gap).** When
a `due` run fires after the machine was down across *one or more* sessions, `target_session` is only
the single most-recent completed session — the older skipped sessions are never operated (correct:
stale sessions must not be back-traded), but that silent skip should be *visible*. On a `due` decision,
`session_gate` computes `skipped_sessions` = the count of completed sessions strictly between the last
recorded session and the target (via the calendar). The CLI emits a non-fatal
`emit_alert("session_gap", {job, last_recorded, target, skipped_sessions})` **before** running
whenever `skipped_sessions > 0` (round-3 fix #5 — was `> 1`, which silently swallowed a **single**
missed session; a one-session outage is exactly the gap worth surfacing). The exit path is unchanged —
the run proceeds normally. The ordinary daily cadence (last recorded = the immediately-prior session)
is `skipped == 0` and fires no alert.

**D6 — Alert command is invoked hardened (GATE-1 Important #3/#4).** `ALGUA_ALERT_CMD` is a
first-class **`alert_cmd: str | None = None` field on `algua/config/settings.py`** (NOT an ad-hoc
`os.environ` read — `Settings.model_config` uses `extra="ignore"`, so an env var that is not a
declared field would never surface through `get_settings()`; declaring it is the consistent choice).
The CLI reads `get_settings().alert_cmd` and passes it to `emit_alert`. When set, the default runner
invokes it with **`shell=False`** (argv via `shlex.split(alert_cmd)` — no shell interpolation of the
payload), a **bounded `timeout`** (10 s), `capture_output=True`, and the JSON payload written on
**stdin**. Captured stdout/stderr are **truncated and redacted** before logging (cap to ~500 chars;
the alert detail dict, which may carry operational specifics, is never echoed back into the external
command's argv). A `TimeoutExpired`/non-zero/OSError from the command is swallowed (logged at
`warning`, `emit_alert` returns `False`) — an alert-delivery failure must never crash the operator.

**Classification is best-effort and never load-bearing (MEDIUM — brittle stdout-JSON classification).**
On an `rc != 0` outcome the wrapper *tries* to name the alert kind from the parsed stdout JSON
(`payload.get("halted") is True → "global_halt"`; `payload.get("ok") is False → "breach"`; else
`"job_failed"`), but classification is a **label only** — the alert *always* fires on `rc != 0`
regardless of parse success, and the alert detail *always* carries the truncated raw stdout
(`stdout_head`, ≤500 chars) and the `rc`. A malformed/absent/renamed field only downgrades the label
to `"job_failed"`; it can never *suppress* the alert or lose the underlying evidence. So the operator
never goes dark just because a driver's JSON shape drifted. (`json.loads` on stdout is wrapped in
`try/except`; a parse failure → `"job_failed"`.)

**D7 — Per-job manifest binds `--job` to a FULL canonical argv + completion predicate + expected
duration (round-3 fix #4, tightening round-2 fix #3).** A small pure module `algua/operator/jobs.py`
declares the allowlisted jobs. The prior rev accepted "any trailing args after a declared head" (a
prefix check) — round-3 fix #4 replaces that with an **exact-arity full-argv template match** so a
mistyped/rogue/extended trailing command can never mark a session done:

```
class CommandMismatch(Exception): ...

@dataclass(frozen=True)
class OperatorJob:
    key: str                            # "paper"
    argv_template: tuple[str, ...]      # the FULL canonical argv, with "{name}" placeholder tokens
    expected_duration_seconds: float    # stuck-lock grace threshold (§D2)
    is_completed: Callable[[int, dict | None], bool]  # §D4 completion predicate
    def bind(self, command: tuple[str, ...]) -> dict[str, str]:
        # EXACT structural match: len(command) == len(argv_template); each fixed token equal;
        # each "{name}" placeholder captures a non-empty value; ANY deviation (extra/missing/
        # altered token, wrong arity) raises CommandMismatch. Returns the captured {name: value}.

OPERATOR_JOBS: dict[str, OperatorJob] = {
    "paper": OperatorJob(
        "paper",
        ("algua", "paper", "run-all", "--snapshot", "{snapshot}"),
        expected_duration_seconds=900,
        is_completed=lambda rc, p: rc == 0 and not (p or {}).get("deferred"),
    ),
}
# The "research" job (("algua","paper","merge-back", …) + completed ⇔ rc0 + grace 3600) is
# DEFERRED to the follow-up (round-3 fix #1) — it needs a branch/strategy SELECTION mechanism first.
```

- **Unknown `--job`** (not in `OPERATOR_JOBS`) → fail-closed: `emit_alert("unknown_job", {job})`,
  `{"ok": false, …}`, exit 1. No marker, no subprocess.
- **Full-argv identity binding (round-3 fix #4).** The wrapper calls `job.bind(tuple(command))`. This
  is an **exact-arity structural match against the full canonical argv**, not a head prefix: the arity
  must equal the template's, every fixed token (`algua paper run-all --snapshot`) must match exactly,
  and each `{placeholder}` captures a non-empty variable value (the snapshot id). A mismatch — a wrong
  head, a trailing extra token (`… --snapshot X --evil`), a missing snapshot, a swapped flag — raises
  `CommandMismatch` → `emit_alert("command_mismatch", {job, expected: argv_template, got: command})`,
  `{"ok": false, …}`, exit 1, **before** the gate. This closes the marker-poisoning vector at its
  root: the old prefix check "accepts any trailing args after the declared head" (the reviewer's exact
  wording); the exact-arity match accepts *only* the one canonical shape and captures its lone variable
  (`snapshot`) for the audit record. The always-on window's command is fixed (the timer's `ExecStart`
  is static, §3.5), so a rigid full-argv contract fits the deployment precisely; ad-hoc drawdown/window
  overrides are a manual human-operator path *outside* the timer (consistent with the round-3 fix #3
  direct-invocation policy). Fail-closed on mismatch is the safe direction: a poisoned marker is worse
  than a refused run.
- `expected_duration_seconds` feeds the §D2 stuck-lock grace; `is_completed` feeds the §D4 record gate.

`jobs.py` imports only stdlib — it is pure declarative config, no I/O, no cross-module edge.

## 3. Components

### 3.1 `algua/operator/schedule.py` (new)
Pure logic over an injected `MarketCalendar` + a file marker + a lock helper; imports `algua.calendar`
+ stdlib (`fcntl`, `os`, `json`, `datetime`) only — **no** subprocess (the git dir is resolved by the
CLI and passed in, §3.3, so schedule.py stays free of a shell-out).

- `class CalendarOutOfBounds(Exception)` — raised by `target_session` when `now` is outside the
  calendar's precomputed range (wraps `xcals` `MinuteOutOfBounds`). A real operational anomaly
  (calendar needs a refresh), distinct from an ordinary weekend/pre-first-session gap.
- `class MarkerCorrupt(Exception)` — raised by `last_session` when the marker file is present but
  unparseable / wrong-shaped / carries a non-ISO session.
- `class OperatorLockHeld(Exception)` — carries `holder: dict | None` (the parsed lock-file body, or
  `None` if missing/garbled), raised by `operator_run_lock` on `BlockingIOError`.
- `target_session(now: datetime, calendar) -> date | None` — the most recent COMPLETED session as of
  `now`. `sess = calendar.session_of_instant(now)`; if `now < calendar.session_close(sess)` the
  session has not closed → `calendar.previous_session(sess)`; else `sess`. The `session_close` used
  here is the exchange-calendar per-session close, so a half-day (early close) session is judged closed
  at its ACTUAL early close, not a fixed 16:00 ET. A naive `now` is treated as UTC (matches
  `session_of_instant`). `MinuteOutOfBounds` → **raise `CalendarOutOfBounds`** (round-2 fix #5); a
  genuine before-first-session `now` still returns `None`.
- `@contextmanager operator_run_lock(lock_path: Path, *, job: str, host: str, pid: int)` — the D2 run
  lock. `open(lock_path, "a+")`; `flock(LOCK_EX | LOCK_NB)`; on success write
  `{"pid", "job", "started_at": <iso-utc>, "host"}` into the body (flush + `os.fsync`), `yield`, and in
  a `finally` truncate the body + `LOCK_UN` + close. On `BlockingIOError` open the file **read-only**,
  parse the holder body (→ `None` on missing/garbled), close, and **raise `OperatorLockHeld(holder)`**.
  The `lock_path` is the **git-dir-anchored** `operator.lock` (round-3 fix #2 — the CLI resolves the
  git dir and passes `git_dir / "operator.lock"`; schedule.py never assumes `db_path.parent`).
- `class SessionMarker` — file-backed per-job last-run store beside the registry db, one entry-object
  per job (`{"session", "recorded_at", "command", "rc", "host", "pid"}` — the enriched audit schema of
  §D3, `command` = the full canonical argv, round-3 fix #4):
  - `last_session(job: str) -> date | None` — absent file or absent-job → `None`; present-and-valid →
    the recorded `date` (reads the entry's `session`; a legacy bare-string entry is also accepted);
    present-but-corrupt → **raises `MarkerCorrupt`** (never silently `None`).
  - `record(job, session, *, command, rc, host, pid) -> None` — writes the enriched entry (`command` =
    the full argv), preserving other jobs' entries; **atomic + fsync-durable**: tmp file → `flush` +
    `os.fsync(tmp_fd)` → `os.replace` → `os.fsync(dir_fd)` (round-2 fix #4, mirroring
    `algua/operator/journal.py`); the whole read-modify-write runs **under a blocking `flock` on
    `operator_sessions.lock`** (the marker lock of §D3, beside the marker at `db_path.parent`).
- `@dataclass(frozen=True) SessionGateDecision(due, session, reason, skipped_sessions: int = 0)` and
  `session_gate(job, now, calendar, marker) -> SessionGateDecision` — `reason` ∈
  `{"due","already_ran","no_session","marker_corrupt","calendar_out_of_bounds"}`. Skip iff
  `target_session` is `None` (no_session) or the marker's recorded session ≥ target (already_ran; `>=`
  so clock skew never re-runs an older one). `CalendarOutOfBounds` from `target_session` → `due=False,
  reason="calendar_out_of_bounds"`. `MarkerCorrupt` from `last_session` → `due=False,
  reason="marker_corrupt"` (session = the target, for the alert detail). On a `due` decision it also
  sets `skipped_sessions` = the count of completed sessions strictly between the last recorded session
  and the target (round-3 fix #5 threshold applied by the CLI at `> 0`).

### 3.2 `algua/operator/alerts.py` (new)
Pure-ish leaf; imports `algua.observability` (logging) + stdlib only (`shlex`, `subprocess`, `json`).

- `emit_alert(kind: str, detail: dict, *, alert_cmd: str | None, runner=...) -> bool` — writes ONE
  structured `operator_alert` record via `get_logger` at `error` level (kind + detail in `fields`),
  and, iff `alert_cmd` is set, invokes it via the injected `runner` (default a thin wrapper over
  `subprocess.run(shlex.split(alert_cmd), shell=False, input=<json payload>, capture_output=True,
  text=True, timeout=10)`). The captured output is truncated/redacted before any logging. Never raises
  — an alert-delivery failure (`TimeoutExpired`, non-zero rc, `OSError`) is itself logged at `warning`
  and swallowed. Returns whether the external command was invoked **and** succeeded (rc 0).

### 3.3 `algua/cli/operator_cmd.py` (new) + mount in `algua/cli/main.py`
- `operator_app = typer.Typer(...)`; `app.add_typer(operator_app, name="operator")`; add
  `operator_cmd` to the `from algua.cli import (...)` block in `main.py`.
- `@operator_app.command("run")` `@json_errors`
  `run(job: str = --job, command: list[str] = typer.Argument(...), now: str | None = --now)`:
  - `configure_logging()`, `correlation_context()`.
  - Empty `command` → `ValueError("operator run requires a command after --")` (rendered by
    `@json_errors` as the standard `ok:false` envelope + exit 1).
  - **Resolve `job` against `OPERATOR_JOBS`** (§D7). Unknown key → `emit_alert("unknown_job", {job})`
    + `{"ok": false, …}` + exit 1. **Bind full-argv identity**: `job.bind(tuple(command))`; on
    `CommandMismatch` → `emit_alert("command_mismatch", {job, expected: job.argv_template, got:
    command})` + `{"ok": false, …}` + exit 1 (round-3 fix #4). Both happen **before** the lock/gate —
    no subprocess, no marker.
  - **Resolve the git dir** for the run-lock anchor (round-3 fix #2): `git_dir = Path(subprocess.run(
    ["git", "rev-parse", "--absolute-git-dir"], cwd=<this file's dir>, capture_output=True, text=True,
    check=True).stdout.strip())` — the same inline resolution `paper_cmd.py` uses for `merge_back.lock`.
  - Enter `operator_run_lock(git_dir / "operator.lock", job=job, host=…, pid=…)` (non-blocking, §D2),
    held across gate → run → record, released in the context manager's `finally`. On
    `OperatorLockHeld(holder)`: compute `held` from `holder["started_at"]` (unknown if `holder` is
    `None`/garbled); if `held is None or held > job.expected_duration_seconds` →
    `emit_alert("operator_lock_stuck", {job, holder_pid, holder_job, started_at, held_seconds,
    grace_seconds})` (round-2 fix #2); either way `emit({"ok": true, "job", "ran": false,
    "reason": "locked", "session": null, "holder": {…}})`; return (exit 0 — benign no-op).
  - Build `MarketCalendar()` + `SessionMarker(get_settings().db_path.parent)`; `now` defaults to
    `datetime.now(UTC)` (`--now` ISO override for tests/manual bring-up). *(The marker stays anchored
    at `db_path.parent`; only the run lock moved to the git dir — round-3 fix #2, §D3.)*
  - `session_gate(job, now, cal, marker)`:
    - `reason == "calendar_out_of_bounds"` → `emit_alert("calendar_out_of_bounds", {"job": job,
      "now": <iso>}, alert_cmd=…)`; `emit({"ok": false, "job": job, "ran": false,
      "reason": "calendar_out_of_bounds", "alerted": True})`; `raise typer.Exit(1)` (fail loud —
      round-2 fix #5; operator must refresh the calendar).
    - `reason == "marker_corrupt"` → `emit_alert("marker_corrupt", {"job": job, "session": <iso>},
      alert_cmd=…)`; `emit({"ok": false, "job": job, "ran": false, "reason": "marker_corrupt",
      "session": <iso>, "alerted": True})`; `raise typer.Exit(1)` (fail-closed — §D3).
    - not `due` (`no_session` / `already_ran`) → `emit(ok({"job": job, "ran": False,
      "reason": decision.reason, "session": <iso or None>}))`; return (exit 0 — a no-op is success).
    - `due` →
      - if `decision.skipped_sessions > 0` → `emit_alert("session_gap", {job, last_recorded, target,
        skipped_sessions})` first (non-fatal — round-3 fix #5, was `> 1`), then proceed.
      - run `command` via an injected subprocess runner (module-level `_run = subprocess.run` seam,
        monkeypatchable). Capture rc + stdout. Parse stdout JSON in a `try/except` → `payload | None`.
      - **completed** (`job.is_completed(rc, payload)` — §D4, NOT bare `rc==0`) → `marker.record(job,
        session, command=list(command), rc=rc, host=…, pid=…)`; `emit(ok({"job", "ran": True,
        "recorded": True, "session": <iso>, "rc": rc}))`.
      - **rc == 0 but NOT completed**:
        - `payload` parsed and `deferred` truthy → `emit(ok({"job", "ran": True, "recorded": False,
          "reason": "deferred", "session": <iso>, "rc": 0}))`; return (exit 0 — no alert; next fire
          retries).
        - `payload` unparseable → `emit_alert("completion_unconfirmed", {job, rc, stdout_head}, …)`;
          `emit(ok({"job", "ran": True, "recorded": False, "reason": "completion_unconfirmed",
          "rc": 0}))`; return (exit 0).
      - **rc != 0** → classify best-effort from `payload` (`halted`→`"global_halt"`, `ok is False`→
        `"breach"`, else/parse-fail→`"job_failed"`); `emit_alert(kind, {job, rc, stdout_head, session},
        alert_cmd=…)` (the alert ALWAYS fires and ALWAYS carries `rc` + truncated `stdout_head`
        regardless of parse — MEDIUM brittle-classification); marker **NOT** recorded (§D4);
        `emit({"ok": false, "job", "ran": True, "recorded": False, "session": <iso>, "rc": rc,
        "alerted": True, "alert_kind": kind})` **then `raise typer.Exit(1)`** so the systemd unit
        records a failure. Retry is the NEXT scheduled fire (marker left unwritten — §D4).

### 3.4 `MarketCalendar.session_close` (add to `algua/calendar/market_calendar.py`)
`def session_close(self, session: date) -> datetime:` → `self._cal.session_close(pd.Timestamp(session)).to_pydatetime()`
(UTC tz-aware). This is the exchange-calendar's **actual** per-session close, so it already respects
early/half-day closes (e.g. the day-after-Thanksgiving 13:00-ET close) — no fixed close time is ever
assumed. `market_calendar.py` is NOT CODEOWNERS-protected.

### 3.5 systemd units + docs (`deploy/systemd/`, new)
Static packaging (not exercised in CI beyond a shape assertion). **Paper timer only** (round-3 fix #1
— the research timer is deferred to a follow-up):
- `algua-paper.service` — `Type=oneshot`, `EnvironmentFile=`, `ExecStart=` running
  `algua operator run --job paper -- algua paper run-all --snapshot ${ALGUA_PAPER_SNAPSHOT}` (systemd
  expands `${…}` from the `EnvironmentFile`). The trailing command **must exactly match the `paper`
  job's `argv_template`** (`algua paper run-all --snapshot {snapshot}` — one variable token) or the
  wrapper fail-closes with `command_mismatch` — the README calls this out so a deployment that launches
  via `uv run algua …` adjusts the entrypoint/`argv_template` to match, and does NOT append ad-hoc
  flags to the always-on `ExecStart` (drawdown/window overrides are a manual human path, see below).
- `algua-paper.timer` — `OnCalendar` ~21:30 UTC (≈16:30 ET, ~30 min post-close), `Persistent=true`.
- `algua.env.example` — Alpaca **paper** creds + `ALGUA_DB_PATH` + `ALGUA_PAPER_SNAPSHOT` +
  `ALGUA_ALERT_CMD` (live wall is cryptographic, not creds-gated; see design §5 G5). `ALGUA_ALERT_CMD`
  maps to the declared `Settings.alert_cmd` field (D6).
- `deploy/systemd/README.md` — install + calendar-gate + idempotency + alert notes. It stresses:
  - the calendar gate makes the exact `OnCalendar` time non-critical (holiday firings no-op);
  - `Persistent=true` covers a **missed window after downtime**, NOT a failed-run retry (D4);
  - `calendar_out_of_bounds` alerts mean **refresh the exchange calendar** (D5);
  - **Direct-invocation residual risk (round-3 fix #3):** `operator.lock` serializes runs that go
    through `operator run`; it does **not** guard a *direct* `algua paper run-all` / `paper trade-tick`
    / (future) `merge-back` that bypasses the wrapper. During the always-on window, **direct/manual
    invocation of these drivers is prohibited by operator policy** — this is an operator-discipline
    contract (matching the `paper run-all` docstring), not a kernel-level mutual-exclusion guarantee.
    The deeper backstop is run-all's ingest-and-reconcile-before-trade (an accidental overlap
    reconciles/defers, it does not blind-double-trade).
  - **Deferred research timer (round-3 fix #1):** a research-cycle / `merge-back` always-on timer is a
    filed follow-up. It requires FIRST solving candidate **branch/strategy/universe/window selection**
    (a queue / pointer file the research unit's `ExecStart` resolves before invoking merge-back) AND
    that selection's *producer* — the research authoring loop that enqueues candidate branches (out of
    scope, §5). Until both exist, a research timer would no-op forever, so it is intentionally not
    shipped here; the generic wrapper (D1) accepts it as a later manifest entry + unit pair with no
    code change.

## 4. Testing

- `tests/test_operator_schedule.py` — `target_session`: after-close instant → that session;
  during-session instant → prior session; weekend/holiday instant → prior Friday session;
  **early-close (half-day) session**: an instant just after the ACTUAL early close (e.g. 2023-11-24
  13:30 ET) → that half-day session; an instant before it but after 13:00 (e.g. 12:30 ET) → prior
  session (proves the gate uses the real early close, not a fixed 16:00);
  **out-of-bounds → raises `CalendarOutOfBounds`** (round-2 #5); before-first-session → None.
  `SessionMarker`: record→last_session round-trip (asserts the **enriched entry object** — session +
  recorded_at + **`command` = full argv** + rc); legacy bare-string entry still parses; multi-job
  isolation; atomic overwrite; **concurrency (two jobs recording near-simultaneously via threads →
  BOTH entries survive)**; absent file → None; **present-but-corrupt file → raises `MarkerCorrupt`**;
  **fsync durability** — monkeypatch `os.fsync` to assert it is called on the temp fd AND the dir fd
  before/after `os.replace` (round-2 #4). `operator_run_lock`: acquire writes holder metadata
  (`{pid, job, started_at, host}`) into the **git-dir-anchored** `operator.lock` body; a second
  non-blocking acquire while held **raises `OperatorLockHeld`** carrying the parsed holder; release
  truncates the body; a garbled body → `OperatorLockHeld(holder=None)` (round-3 #2). `session_gate`:
  due / already_ran (`>=`) / no_session / **marker_corrupt** / **calendar_out_of_bounds** /
  **`skipped_sessions` set on a multi-session gap**. (Uses a real `MarketCalendar()` with fixed 2023
  dates — xcals is already a test dep.)
- `tests/test_operator_jobs.py` — `OPERATOR_JOBS`: `paper.is_completed` truth table (rc0+no-deferred →
  True, rc0+`deferred:true` → False, rc≠0 → False); **`bind` exact-arity full-argv match** — accepts
  `("algua","paper","run-all","--snapshot","SNAP")` and captures `{"snapshot": "SNAP"}`; **rejects a
  trailing extra token** (`… "--snapshot","SNAP","--evil"` → `CommandMismatch`); rejects a wrong head
  (`algua data inspect`), a missing snapshot value, an empty placeholder, a wrong arity (round-3 #4);
  unknown key lookup → `None`/`KeyError`.
- `tests/test_operator_alerts.py` — `emit_alert`: logs the record; invokes `alert_cmd` with the JSON
  payload on stdin via an injected runner; **the default runner uses `shell=False` + a bounded
  timeout** (assert argv is `shlex.split`, not a shell string; a runner raising `TimeoutExpired` is
  swallowed → returns False); a runner that raises is swallowed (returns False, no exception); a
  non-zero rc → returns False; no `alert_cmd` → no invocation.
- `tests/test_cli_operator.py` — invoke `algua operator run` via the CLI runner with a fake subprocess
  runner + `--now`, `ALGUA_DB_PATH` at a tmp dir (marker + `operator_sessions.lock` isolated), and a
  git dir (the tmp-dir worktree's `.git`) so `operator.lock` is isolated: (a) weekend `--now` →
  `ran:False, reason:no_session`, runner NOT called; (b) already-ran (pre-seeded marker) → `ran:False,
  reason:already_ran`; (c) due + runner rc 0 (completed) → runner called once, marker written
  (`recorded:True`, **entry `command` = the full argv**), `ran:True,rc:0`; (d) due + runner rc 1
  emitting `{"halted":true}` → alert fired (`alert_kind:"global_halt"`, detail carries `stdout_head`+
  `rc`), marker NOT written, **`ok:false`**, exit 1; (e) due + runner rc 1 non-JSON →
  `alert_kind:"job_failed"`, `ok:false`, exit 1; (f) empty command → error JSON;
  (g) **corrupt marker file → alert `marker_corrupt`, runner NOT called, `ok:false`, exit 1**;
  (h) **run-lock contention within grace (hold the git-dir `operator.lock` with fresh holder metadata)
  → `ran:False, reason:"locked"`, exit 0, runner NOT called, NO stuck alert**;
  (i) **run-lock contention past grace (hold the lock with a `started_at` older than the paper job's
  `expected_duration_seconds`) → `emit_alert("operator_lock_stuck", …)` fired, still `reason:"locked"`
  exit 0** (round-2 #2);
  (j) **round-2 #1 — due + runner rc 0 emitting `{"ok":true,"deferred":true}` → marker NOT written
  (`recorded:False, reason:"deferred"`), exit 0, NO alert** (the benign-no-trade non-suppression);
  (k) **due + runner rc 0 non-JSON → `completion_unconfirmed` alert, marker NOT written, exit 0**;
  (l) **calendar-out-of-bounds `--now` (a date past the calendar horizon) → alert
  `calendar_out_of_bounds`, runner NOT called, `ok:false`, exit 1** (round-2 #5);
  (m) **round-3 #4 — command-identity mismatch:** `--job paper -- algua data inspect` **and**
  `--job paper -- algua paper run-all --snapshot X --evil` both → alert `command_mismatch`, `ok:false`,
  exit 1, runner NOT called (a wrong head AND a trailing-junk variant are both rejected);
  (n) **unknown `--job frobnicate` → alert `unknown_job`, `ok:false`, exit 1**;
  (o) **round-3 #5 — due after a one-session gap (pre-seed the marker two sessions back) →
  `emit_alert("session_gap", {skipped_sessions: 1})` fired before the run, run proceeds, exit 0**
  (proves `> 0`, not `> 1`). Set `ALGUA_ALERT_CMD` via monkeypatched settings where an alert is
  asserted.
- A light shape assertion (in `test_cli_operator.py` or `test_operator_schedule.py`) that the two
  `deploy/systemd/algua-paper.{service,timer}` files exist and contain the required directives
  (`Type=oneshot`, `operator run --job paper`, `OnCalendar`, `Persistent=true`).

## 5. Non-goals
The research *authoring* loop (Codex ideate→author→sweep) and the merge-back internals are out of
scope (built in #485 / the research-loop scripts). This issue provides the clock + gate + idempotency
+ alerts that fire the existing single-shot **paper** driver.

**The research-cycle / `merge-back` always-on timer is explicitly deferred (round-3 fix #1).** A
follow-up must FIRST solve candidate **branch/strategy/universe/window selection** — a static
`ExecStart` cannot know which candidate to merge back; that requires an explicit selection mechanism
(a queue / pointer file the research systemd unit's `ExecStart` resolves before invoking merge-back)
AND its *producer*, the authoring loop that enqueues candidate branches. Shipping the timer before
either exists would build a component that no-ops forever, so it is a separate issue. The generic
wrapper (D1) is authored so that follow-up drops in as a manifest entry (`research`) + selection
mechanism + unit pair, with **no wrapper change**. NOVEL-family autonomy and the family-audit guard
(design §6.5) remain separate issues.

## 6. Task list (implementation decomposition)

Ordered, each independently buildable; each names its fast-check test target.

1. **`schedule.py` — session clock + calendar close + fsync-durable fail-closed marker + git-dir run
   lock.** Add `MarketCalendar.session_close` (§3.4, exchange-calendar actual close). Create
   `algua/operator/schedule.py` (§3.1): `CalendarOutOfBounds`, `MarkerCorrupt`, `OperatorLockHeld`,
   `target_session` (**raises `CalendarOutOfBounds` on `MinuteOutOfBounds`** — round-2 #5 — `None` only
   for a genuine before-first-session), `operator_run_lock` (**git-dir-anchored** `operator.lock`,
   non-blocking, writes/reads holder metadata, raises `OperatorLockHeld` on contention — round-3 #2),
   `SessionMarker` (enriched entry-object schema with **`command` = full argv** — round-3 #4;
   `last_session` raising `MarkerCorrupt` on a present-corrupt file and tolerating a legacy
   bare-string; `record` **fsync-durable** — tmp → `fsync(tmp_fd)` → `os.replace` → `fsync(dir_fd)`,
   round-2 #4, mirroring `algua/operator/journal.py` — AND guarded by a **blocking `flock` on
   `operator_sessions.lock`**), `SessionGateDecision(…, skipped_sessions)`, `session_gate` (reasons
   incl. `marker_corrupt`, `calendar_out_of_bounds`, and `skipped_sessions` on a `due` gap). Fast
   check: `... pytest -q tests/test_operator_schedule.py` — after-close/during/weekend/**early-close
   half-day**/**out-of-bounds→CalendarOutOfBounds** `target_session`; marker round-trip (enriched
   object w/ `command`) / legacy-string / multi-job / atomic-overwrite / **two-thread concurrency** /
   **fsync-called-on-tmp-and-dir** / absent→None / **corrupt→MarkerCorrupt**; **run-lock acquire/held→
   OperatorLockHeld/garbled-holder**; gate due/already_ran/no_session/**marker_corrupt**/
   **calendar_out_of_bounds**/**skipped_sessions**.

2. **`jobs.py` — the per-job manifest (FULL-argv identity + completion predicate + grace) [round-3
   #4, paper-only round-3 #1].** Create `algua/operator/jobs.py` (§D7): `CommandMismatch`,
   `OperatorJob` (`argv_template`, `expected_duration_seconds`, `is_completed(rc, payload)`,
   `bind(command)` = exact-arity full-argv match) and `OPERATOR_JOBS` with the single `paper` job
   (template `algua paper run-all --snapshot {snapshot}`; completed ⇔ rc0 ∧ ¬deferred; grace 900).
   The `research` job is DEFERRED (round-3 #1). Pure stdlib, no I/O. Fast check:
   `... pytest -q tests/test_operator_jobs.py` — `is_completed` truth table + **`bind` exact-arity
   accept/capture + reject-trailing-junk/wrong-head/missing-value/wrong-arity** + unknown-key.

3. **`alerts.py` — hardened alert hook + `Settings.alert_cmd`.** Add `alert_cmd: str | None = None` to
   `algua/config/settings.py` (§D6). Create `algua/operator/alerts.py` (§3.2): `emit_alert` logging one
   `operator_alert` record and invoking `alert_cmd` via the default runner with **`shell=False`
   (`shlex.split`), `timeout=10`, `capture_output=True`, payload on stdin, truncated/redacted output
   logging**; never raises. Fast check: `... pytest -q tests/test_operator_alerts.py` — log-record,
   stdin-payload invocation, `shell=False`/timeout assertions, swallow-on-raise /
   swallow-on-nonzero-rc, no-cmd→no-invoke.

4. **`operator_cmd.py` — the `operator run` wrapper (full-argv identity + git-dir run lock + gate +
   subprocess + alerts).** Create `algua/cli/operator_cmd.py` (§3.3) and mount `operator_app` in
   `algua/cli/main.py`. Resolve `--job` against `OPERATOR_JOBS` (**unknown_job** fail-closed, then
   **`bind` full-argv → command_mismatch** fail-closed — round-3 #4 — BEFORE the lock). Resolve the
   **git dir** (`git rev-parse --absolute-git-dir`) and acquire the git-dir-anchored `operator.lock`
   run lock via `operator_run_lock` around gate→run→record (round-3 #2); on `OperatorLockHeld` compute
   `held` and **alert `operator_lock_stuck` past grace / unknown holder** (round-2 #2), else benign
   `reason:"locked"` exit-0. Gate branches: **`calendar_out_of_bounds`** → alert + `ok:false` + exit 1
   (round-2 #5); `marker_corrupt` → alert + `ok:false` + exit 1; `no_session`/`already_ran` → `ok:true`
   no-op; `due` → **`session_gap` alert iff `skipped_sessions > 0`** (round-3 #5), run via the
   monkeypatchable `_run` seam, **record ONLY iff `job.is_completed(rc, payload)`** with `command` =
   the full argv (round-2 #1 / round-3 #4; `deferred`→`recorded:False` no-alert exit-0; unparseable→
   `completion_unconfirmed` exit-0), rc≠0 → best-effort classify + `emit_alert` (always carries
   rc+stdout_head) + `ok:false` + exit 1. Fast check: `... pytest -q tests/test_cli_operator.py` —
   cases (a)–(o) of §4 incl. **deferred non-suppression**, **stuck-lock alert**,
   **calendar_out_of_bounds**, **command_mismatch (wrong head AND trailing junk)**, **unknown_job**,
   **session_gap at skipped==1**.

5. **systemd paper units + env example + README (packaging) [paper-only, round-3 #1/#3].** Add
   `deploy/systemd/{algua-paper.service,algua-paper.timer}`, `algua.env.example` (incl.
   `ALGUA_PAPER_SNAPSHOT` + `ALGUA_ALERT_CMD`), and `deploy/systemd/README.md` (§3.5) — the `ExecStart`
   trailing command **exactly matches** the `paper` `argv_template`; with the corrected `Persistent=
   true` semantics (missed-window catch-up, not failed-run retry), the calendar-refresh note for
   `calendar_out_of_bounds`, the **direct-invocation-prohibited residual-risk note** (round-3 #3), and
   the **deferred-research-timer + selection-mechanism note** (round-3 #1). Fast check: the shape
   assertion in `tests/test_cli_operator.py` (or `tests/test_operator_schedule.py`) that the two paper
   unit files exist and carry `Type=oneshot` / `operator run --job paper` / `OnCalendar` /
   `Persistent=true`.
