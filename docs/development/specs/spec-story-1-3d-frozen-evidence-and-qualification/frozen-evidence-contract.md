# Story 1.3d frozen evidence contract

Normative companion to [SPEC.md](SPEC.md). Builds on the Story 1.3c contract; terms as defined there.

## 1. What an attempt is (CAP-1)

- An *attempt* is one phase dispatch the supervisor decided to run a child for: every entry into the
  frozen port's invocation step after its own verdict did not already settle the phase (Story 1.3c
  §3). It includes attempts refused before launch (request too large, content unsupported,
  unencodable input), which record no request bytes.
- Not attempts, and not recorded here: phases the supervisor settled without a child (a
  supervisor-found breach, a pending venue belief, a supervisor refusal of its own input). Those keep
  their existing records (kill-switch/breach audit, run-all `setup_error` entry).
- Exactly one row is written per attempt, once the supervisor has fully judged it: *success* means
  the result decoded and passed every Story 1.3c §7 cross-check for that phase; otherwise the row
  carries the `FrozenTenantFailure` code the supervisor raised. A Phase B decision later turned into
  a breach by the supervisor's weight-rule rerun is still a successful attempt (the breach is
  recorded by the existing kill-switch path and never writes a tick).
- A crash between attempt and row leaves no row and therefore no linkable tick. So does a systemic
  exception during an attempt (a launch `OSError` classified systemic, a SQLite error,
  `KeyboardInterrupt`): it propagates as today and records nothing.
- Phase A is recorded when `phase_a` returns. A later supervisor rejection in `closed_bars` (the
  decision-time cross-check) launches no Phase B and surfaces only as the tenant failure
  (`setup_error` entry and audit row); the Phase A row stays a success because that attempt was.

## 2. Schema (v48; this section is the protected schema review)

New context fragment `algua/registry/db/frozen_evidence.py`; `SCHEMA_VERSION` 47 → 48.

```sql
CREATE TABLE IF NOT EXISTS frozen_invocations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_id         INTEGER NOT NULL REFERENCES strategy_deployments(id),
    request_id            TEXT    NOT NULL CHECK (length(request_id) = 32),
    phase                 TEXT    NOT NULL CHECK (phase IN ('a', 'b')),
    phase_a_invocation_id INTEGER REFERENCES frozen_invocations(id),
    snapshot_id           TEXT    NOT NULL,
    bars_start            TEXT,
    bars_end              TEXT,
    request_json          TEXT CHECK (request_json IS NULL OR length(CAST(request_json AS BLOB)) <= 262144),
    request_sha256        TEXT CHECK (request_sha256 IS NULL OR length(request_sha256) = 64),
    bars_sha256           TEXT CHECK (bars_sha256 IS NULL OR length(bars_sha256) = 64),
    phase_a_binding       TEXT,
    result_kind           TEXT CHECK (result_kind IS NULL OR result_kind IN (
                              'early_no_decision', 'snapshot_required', 'risk_failure',
                              'late_no_decision', 'decision')),
    result_sha256         TEXT CHECK (result_sha256 IS NULL OR length(result_sha256) = 64),
    failure_code          TEXT CHECK (failure_code IS NULL OR failure_code IN (
                              'frozen_content_unavailable', 'frozen_content_unsupported',
                              'frozen_request_too_large', 'frozen_launch_failed', 'frozen_timeout',
                              'frozen_exit_abnormal', 'frozen_output_exceeded',
                              'frozen_result_invalid', 'frozen_planner_rejected',
                              'frozen_live_unsupported')),
    returncode            INTEGER,
    signal                INTEGER,
    timed_out             INTEGER NOT NULL DEFAULT 0 CHECK (timed_out IN (0, 1)),
    stdout_exceeded       INTEGER NOT NULL DEFAULT 0 CHECK (stdout_exceeded IN (0, 1)),
    stderr_truncated      INTEGER NOT NULL DEFAULT 0 CHECK (stderr_truncated IN (0, 1)),
    diagnostic            TEXT CHECK (diagnostic IS NULL OR length(diagnostic) <= 8192),
    started_at            TEXT NOT NULL,
    ended_at              TEXT NOT NULL,
    CHECK ((result_sha256 IS NULL) <> (failure_code IS NULL)),
    CHECK ((result_sha256 IS NULL) = (result_kind IS NULL)),
    CHECK ((request_json IS NULL) = (request_sha256 IS NULL)),
    CHECK ((phase = 'b') = (phase_a_invocation_id IS NOT NULL)),
    CHECK (diagnostic IS NULL OR failure_code IS NOT NULL),
    CHECK (result_kind IS NOT 'snapshot_required' OR phase_a_binding IS NOT NULL),
    UNIQUE (request_id, phase)
);

CREATE TRIGGER IF NOT EXISTS frozen_invocations_no_update
BEFORE UPDATE ON frozen_invocations
BEGIN SELECT RAISE(ABORT, 'frozen invocation evidence is append-only'); END;

CREATE TRIGGER IF NOT EXISTS frozen_invocations_no_delete
BEFORE DELETE ON frozen_invocations
BEGIN SELECT RAISE(ABORT, 'frozen invocation evidence is append-only'); END;

-- INSERT OR REPLACE would delete-and-reinsert without firing the delete trigger when recursive
-- triggers are off (a raw connection); refuse any insert that collides with an existing row.
CREATE TRIGGER IF NOT EXISTS frozen_invocations_no_replace
BEFORE INSERT ON frozen_invocations
WHEN EXISTS (SELECT 1 FROM frozen_invocations
             WHERE id = NEW.id OR (request_id = NEW.request_id AND phase = NEW.phase))
BEGIN SELECT RAISE(ABORT, 'frozen invocation evidence is append-only'); END;

-- A Phase B attempt must follow a successful Phase A attempt of the same tick and deployment,
-- and carry the binding that Phase A produced.
CREATE TRIGGER IF NOT EXISTS frozen_invocations_phase_b_follows_a
BEFORE INSERT ON frozen_invocations WHEN NEW.phase = 'b'
BEGIN
    SELECT RAISE(ABORT, 'phase b must follow a successful phase a of the same tick')
    WHERE NOT EXISTS (
        SELECT 1 FROM frozen_invocations a
        WHERE a.id = NEW.phase_a_invocation_id AND a.phase = 'a'
          AND a.deployment_id = NEW.deployment_id AND a.request_id = NEW.request_id
          AND a.result_kind = 'snapshot_required'
          AND a.phase_a_binding IS NEW.phase_a_binding
          AND a.snapshot_id IS NEW.snapshot_id);
END;
```

`tick_snapshots` gains one column, added by a guarded ALTER in `migrate()`; its index and triggers are
constants in `db/frozen_evidence.py`, executed by `migrate()` after the v47 block (after the ALTER):

```sql
ALTER TABLE tick_snapshots ADD COLUMN frozen_invocation_id INTEGER REFERENCES frozen_invocations(id);

CREATE UNIQUE INDEX IF NOT EXISTS tick_snapshots_one_tick_per_invocation
    ON tick_snapshots(frozen_invocation_id) WHERE frozen_invocation_id IS NOT NULL;

-- A tick of a frozen deployment must link a successful final (phase b) invocation of the same
-- deployment and snapshot; any other tick must not link one.
CREATE TRIGGER IF NOT EXISTS tick_snapshots_frozen_link
BEFORE INSERT ON tick_snapshots
BEGIN
    SELECT RAISE(ABORT, 'a frozen tick must link its successful final invocation')
    WHERE EXISTS (
        SELECT 1 FROM strategy_deployments d JOIN deployment_artifacts x ON x.id = d.artifact_id
        WHERE d.id = NEW.deployment_id AND x.source_kind = 'frozen')
      AND NOT EXISTS (
        SELECT 1 FROM frozen_invocations i
        WHERE i.id = NEW.frozen_invocation_id AND i.phase = 'b'
          AND i.deployment_id = NEW.deployment_id
          AND i.result_kind IN ('decision', 'late_no_decision')
          AND i.snapshot_id IS NEW.snapshot_id);
    SELECT RAISE(ABORT, 'only a frozen tick may link a frozen invocation')
    WHERE NEW.frozen_invocation_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM strategy_deployments d JOIN deployment_artifacts x ON x.id = d.artifact_id
        WHERE d.id = NEW.deployment_id AND x.source_kind = 'frozen');
END;

CREATE TRIGGER IF NOT EXISTS tick_snapshots_link_immutable
BEFORE UPDATE OF frozen_invocation_id, deployment_id, snapshot_id, strategy_id ON tick_snapshots
BEGIN SELECT RAISE(ABORT, 'a tick''s deployment and invocation link cannot change'); END;
```

- Identities are recorded by immutable reference: artifact, manifest, bundle, environment, protocol
  and strategy follow from `deployment_id` through trigger-immutable rows. No copies.
- `request_json` is the canonical `request.json` text (at most 256 KiB): identities, the recorded
  config, `now`, positions, gate universe and captured account values. It holds no bars,
  credentials, raw stderr or handles. `request_sha256` and `bars_sha256` are SHA-256 of the exact
  request and `bars.arrow` bytes; `result_sha256` is SHA-256 of the child's accepted stdout bytes
  (the canonical document plus its trailing newline), not of a re-encoding.
- `bars_start` and `bars_end` are ISO-8601 UTC renderings of the exact bounds passed to `get_bars`.
- `phase_a_binding` is the binding Phase A produced (on phase `a` rows) and the binding Phase B
  received (on phase `b` rows); the trigger requires them equal.
- `diagnostic` is Story 1.3c's `process_diagnostic` (printable ASCII, at most 8 KiB), on failure rows
  only.
- Rows are retained indefinitely (Phase 1): about two 10–20 KiB rows per frozen tenant per session.
  Ticks are not made immutable beyond the linked columns.
- v48 is forward-only. Running 1.3c code on a v48 database aborts `run-all` at the first frozen
  tenant (after that tenant's orders are sent). Before any rollback, retire frozen deployments or
  drop `tick_snapshots_frozen_link`; the runbook records this.

## 3. Recording seam (CAP-1, CAP-2)

- A pure value `FrozenAttempt` in `algua/contracts/frozen_evidence.py` carries every column above
  except `id`. The live port builds it; the registry persists it. Live still imports no registry
  module.
- `FrozenPlanner` takes an injected `record: Callable[[FrozenAttempt], int]` and the tick's
  `snapshot_id`, `bars_start`, `bars_end` at construction (the CLI builds one port per tick). It
  records each attempt as defined in §1 and exposes `final_invocation_id`: the id of the tick's
  successful Phase B attempt, or `None`.
- The CLI binds `record` to a registry store function that inserts the row and commits it in its
  own transaction. It first raises (an explicit check, not an `assert`) if the connection has an
  open transaction. A SQLite error while recording is systemic, as in Story 1.3c §8.
- `record_tick_snapshot` takes an optional `frozen_invocation_id`; `_run_paper_strategy_tick`
  passes the port's `final_invocation_id` for a frozen tenant. `run_tick` and `live_loop.py` are not
  changed.
- New modules: `algua/contracts/frozen_evidence.py`, `algua/registry/db/frozen_evidence.py`,
  `algua/registry/store/frozen_evidence.py` and `algua/live/frozen_attempt.py`. `frozen_dispatch.py`
  is at 296 lines, so existing code (the decision cross-check or the failure plumbing) moves into
  `frozen_attempt.py` together with the recording hooks; no pin is added. Update importers of moved
  names (for example `cli/errors.py` imports `FrozenTenantFailure`). All join CODEOWNERS and the hygiene set; the live module
  joins the frozen import-linter contract. `algua/execution/tick_snapshots.py` joins CODEOWNERS and
  the hygiene set.

## 4. Admissibility (CAP-3)

- `assemble_forward_evidence` gains one exclusion filter, `invocation_unlinked`, evaluated
  immediately after `deployment_mismatch` and only for frozen deployments: a tick is excluded unless
  it links a `frozen_invocations` row that is phase `b`, of the same deployment and the same
  `snapshot_id` as the tick, with `result_kind IN ('decision','late_no_decision')`. The existing
  filters then apply unchanged, so a linked `late_no_decision` tick counts exactly like the
  equivalent working-tree tick.
- For a frozen deployment, the identity the evidence is checked against is the descriptor's three
  hashes (so `identity_drift` only detects corruption). The supported protocol is the manifest's
  `frozen_wire` stamp, checked when the descriptor is parsed.
- 1.3c-era frozen ticks have no link and are excluded; `paper promote`'s `excluded_ticks` gains the
  `invocation_unlinked` key. No other JSON shape changes.

## 5. Promotion (CAP-4)

- One chokepoint, `promotion_identity(conn, record, *, data_dir) -> (deployment, identity)` in
  `algua/registry/forward_promotion.py`:
  - working-tree: exactly today's `compute_artifact_hashes` plus the deployment-hash match;
  - frozen: parse the recorded descriptor, verify its bundle and environment offline with a *fresh*
    `FrozenContentVerifier`, and return the descriptor's hashes. A verification failure refuses with
    `frozen_content_unavailable` / `frozen_content_unsupported` before any evaluation row, look
    count, token or stage change.
- Order: `paper promote` keeps today's ledger-only frozen check in its current slot (where
  `refuse_frozen_promotion` runs). Only for a frozen deployment does it call `promotion_identity`
  there, before actor authentication; working-tree and legacy strategies keep today's order and the
  places where identity is computed, so their refusal order is unchanged. Identity is resolved once
  and `(deployment, identity)` is passed into `run_forward_gate`, so the environment is verified
  once per promotion.
- `authenticate_actor` receives a lazy identity callable that only a human actor calls (so a human
  challenge binds the descriptor hashes for a frozen deployment); `promote_run.py` passes the
  checkout identity without growing past its size pin. No path of `paper promote` imports a frozen
  tenant's checkout module.
- `refuse_frozen_promotion` and the `frozen_qualification_pending` code are removed from
  `paper promote` and `run_forward_gate`. The raw `registry transition` edge to `forward_tested`
  keeps refusing a frozen deployment, now as a `TransitionError` (`wrong_stage`: "reach
  forward_tested only through paper promote"); the code is deleted from the CLI registry and the
  error-envelope doc. Go-live keeps `frozen_live_unsupported`.
- A frozen deployment promoted to `forward_tested` keeps ticking in paper and can refresh its
  certificate row with `paper promote`; it cannot go live until Epic 2.
- The holdout Sharpe lookup is unchanged.

## 6. Replay (CAP-5)

- A test helper, composed from production primitives only, replays a recorded attempt: load the row
  and its deployment, resolve and verify content with `FrozenContentVerifier`, re-read the bars from
  the row's `snapshot_id`, `bars_start`/`bars_end` and the symbols
  `sorted(set(gate_universe) | {s for s, q in early_positions if q != 0})` taken from
  `request_json`, through the same store-backed provider; require the decoded frame's logical
  `bars_digest` to equal the one in `request_json` (the child enforces it too); write the recorded
  `request_json` and the re-encoded bars into a sealed invocation directory; launch the child exactly
  as Story 1.3c §5 does; and require the SHA-256 of its stdout to equal `result_sha256`. It needs no
  Git and no uv. `bars_sha256` equality is asserted only within one supervisor environment and is
  pinned by an `encode_bars` golden-bytes test (Arrow bytes follow the supervisor's pyarrow).
- Tests replay one Phase A and one Phase B attempt after a simulated restart and with the checkout
  module edited or deleted.
- Residuals, recorded not solved: data snapshots have no deletion path but no formal retention
  guarantee; frozen environments bind the host interpreter (Story 1.3c §2), so a host patch upgrade
  stops a frozen tenant's clock until migration; replay is exact only for strategies whose output
  does not depend on string-hash order (the child runs with `-I`, which ignores `PYTHONHASHSEED`).

## 7. Failure classification and tests

- No new tenant failure codes. Recorder SQLite errors are systemic. `trade-tick`/`run-all` JSON is
  unchanged.
- Tests cover: every attempt kind recorded (success, each §8 failure, pre-launch refusal); no row
  for supervisor-settled phases; append-only (update/delete abort); the phase-b trigger (missing,
  failed or mismatched A); the tick trigger (frozen tick without, with a failed, a phase-a, a
  wrong-deployment or wrong-snapshot link; a working-tree tick with a link; duplicate link); link
  immutability; the admissibility filter and its count; promotion success from linked evidence
  without checkout access, promotion refusal on corrupt/replaced/permission-drifted content, the
  human-actor challenge binding descriptor hashes, the raw-edge refusal, go-live still refused;
  replay determinism; sibling isolation and systemic failures unchanged; working-tree and legacy
  evidence and promotion unchanged; migration from v47 (column, triggers, version).
- Known churn: `docs/architecture.md` (names `frozen_qualification_pending`); tests that stamp
  unlinked frozen ticks (test_deployments.py, test_frozen_runtime.py, the 1.3c e2e tests in
  test_frozen_paper_cli.py); the schema fingerprint in test_registry_db.py; the filter tuple in
  test_forward_promotion.py; test_frozen_promotion_refusal.py and the error-envelope doc.
