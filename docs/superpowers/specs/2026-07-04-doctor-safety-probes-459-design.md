# Doctor safety-state, credential & broker-reachability probes (#459)

## Problem
`uv run algua doctor` is advertised (AGENTS.md §1) as *the* environment readiness self-check — the
agent's only pre-flight before a paper/live cycle. It currently runs seven probes: four **required**
(`python`, `registry_db`, `calendar`, `knowledge_base`) and three **advisory** (`paper_credentials`,
`bars_snapshot`, `generated_provenance`), then emits `{ok: all-required-pass, checks:[...]}` and
exits 0/1 on the required set (`algua/cli/app.py:148-166`).

It probes **no safety state and no broker dependency**. `doctor` returns an all-green `{ok:true}`
while:
- **the global halt is engaged** (`algua/risk/global_halt.py:21` `is_engaged`) — every paper/live
  trading tick raises on the very next call;
- **a per-strategy kill-switch is tripped** (`algua/risk/kill_switch.py:17` `is_tripped`) — that
  strategy is silently benched;
- **a LIVE strategy's human go-live authorization is revoked or no longer re-verifies** against the
  `approvers/allowed_signers` trust anchor (`algua/registry/live_gate.py:154`
  `verify_live_authorization`, re-run before every live order) — that strategy is fail-closed
  skipped at submit;
- **Alpaca LIVE credentials are absent** (checked only inside the live command body,
  `algua/cli/live_cmd.py:73-74`, raising `ValueError` at call time);
- **the paper/live broker is unreachable** — the critical outbound dependency for both runners is
  never probed.

The Health-Checks playbook (KB software-engineering/15-observability): *"a health check that only
confirms the process is up is worse than none"* — a readiness check must probe critical dependencies
and safety state and report **not-ready** when any is unmet. Today an operator (human or autonomous
loop) who trusts a green `doctor` during a global halt, a broker outage, or after a live-auth
revocation believes the system is healthy and burns the ~1.5 CPU-s cold-start (#326) on a
guaranteed-to-fail cycle, discovering the gap only via a `ValueError` inside the command body.

Distinct from #400 (no per-strategy fleet status rollup): this is about the EXISTING advertised
health check returning misleadingly-healthy output.

## What `doctor` *means* (the definitional decision GATE-1 forced)
`doctor` is a **trading-readiness** pre-flight, exactly as AGENTS.md advertises it ("readiness
self-check … before a live cycle"). Its exit code answers **"is it safe and possible to start a
trading cycle right now?"** — NOT "can I run a backtest?". Two consequences, stated up front to keep
the rest of the spec consistent:
1. **Safety state (`global_halt`) gates every mode.** A global halt is an account-wide trading
   emergency; a trading-readiness check MUST go red during it. This is intentional and is the exact
   false-green #459 exists to kill — making it advisory would re-introduce the bug. (Research/backtest
   commands do not consult `doctor`'s exit code; they run regardless. An operator who has engaged a
   halt on purpose correctly sees red: "do not start a trading cycle now.")
2. **Lane-specific broker/credential dependencies gate only the lane you name** (below), because a
   generic `doctor` cannot know which lane you are about to run.

### Command-scoped readiness: `--paper` / `--live`
`doctor` grows two combinable mode flags that declare the lane being pre-flighted and promote **that
lane's** dependency probes to **required (exit-gating)**:
- **`doctor` (no flag)** — safety-state + local-invariant readiness. `global_halt` and the four
  existing local probes gate; every lane-specific credential/broker probe is **advisory**, and the
  **reachability probes do no network at all** (they report a `skipped` row — see below). So plain
  `doctor` stays local + fast and never makes an outbound call.
- **`doctor --paper`** = "can the paper loop actually run?" → promotes `paper_credentials` and
  `alpaca_paper_reachable` to required (and performs the paper broker call).
- **`doctor --live`** = "are the live lane's **dependencies** ready?" → promotes
  `alpaca_live_credentials`, `alpaca_live_reachable`, and `live_authorizations` to required (and
  performs the live broker call). "Ready" is deliberately scoped to **lane dependencies**: live
  creds present, the live broker reachable, and — among the LIVE strategies that exist — no
  *authorization* blocker (at least one re-verifies). It **does not promise that a strategy will
  actually trade**: whether any LIVE strategy has an active allocation, is not kill-switched, or is
  even present at all are runner-time / fleet-status concerns (`live_authorizations` checks
  authorization only; per-strategy allocation + kill-switch health is #400's rollup, deliberately
  out of scope here — folding them in would duplicate #400 and couple `doctor` to allocation state).
  Consequently **zero LIVE strategies is a ready state under `--live`** (the live infra is fine;
  there is simply nothing staged to authorize — an empty live book is not an infrastructure failure).
  This readiness definition is a property of `doctor --live`; it does not change `live run-all`'s own
  success semantics.
- **`doctor --paper --live`** (both) — required set = the **union** of both lanes' required probes;
  both broker calls are performed. Fully composable.

## Current state (already merged — this PR neither re-adds nor renames it)
`paper_credentials` (advisory today, Alpaca paper key/secret presence) and `bars_snapshot`
(advisory, ≥1 BARS snapshot ingested) **already exist** with `required=False`. The issue's
recommended `alpaca_paper_creds` and `data_snapshots` probes are therefore already present — this PR
does **not** duplicate them and does **not** rename them (a rename of merged, test-pinned probe
names — `tests/test_cli_core.py:79-82` asserts `paper_credentials`/`bars_snapshot`/
`generated_provenance` — is pure churn). The only change to `paper_credentials` is that `--paper`
promotes it to required at the call site; its default-mode advisory behaviour is unchanged.

## Scope of this change

### Probes and their per-mode required-set
Each probe reuses the existing `_check(name, fn, *, required=...)` shape (`app.py:41`): `fn` returns
a detail string on success or raises; `_check`'s broad `except Exception` renders any failure as a
red row, never a traceback. `doctor()` sets each lane-specific probe's `required=` from the mode
flags at the call site. No new envelope field, no schema change, no `SCHEMA_VERSION` bump.

| probe | row `ok` is false when | required in mode |
|---|---|---|
| `global_halt` | a halt row exists (`global_halt.get(conn)` non-None) | **all modes** |
| `kill_switches` | never — always `ok:true`; detail lists any tripped switches (advisory-informational) | never |
| `live_authorizations` | LIVE strategies exist AND **zero** of them re-verify (`ok` = *no LIVE strategies* OR *≥1 authorized*) | **`--live`** |
| `alpaca_live_credentials` | live key or secret unset | **`--live`** |
| `alpaca_paper_reachable` | in a mode that probes paper (`--paper`) and the broker call is not a valid account response; **skipped `ok:true` when not probing** | **`--paper`** |
| `alpaca_live_reachable` | in a mode that probes live (`--live`) and the broker call is not a valid account response; **skipped `ok:true` when not probing** | **`--live`** |

(Existing `paper_credentials` → required under `--paper`; `bars_snapshot`, `generated_provenance`
stay advisory in all modes.)

### Probe semantics (the details GATE-1 flagged)

**`global_halt` (required, all modes).** Reads `global_halt.get(conn)`; raises with
`reason/actor/created_at` when a halt row exists; else returns "not engaged". Narrowed rationale: a
global halt is the one state under which no paper/live **trading cycle** can start (every tick
raises), account-wide — so trading-readiness is false while it is engaged.

**`kill_switches` (advisory, all modes, never gates).** New additive helper
`kill_switch.list_tripped(conn) -> list[dict[str,str]]`
(`SELECT strategy,reason,actor,created_at FROM kill_switches ORDER BY strategy`). The probe
cross-references the tripped names against current stages
(`SqliteStrategyRepository.list_strategies()`) and its detail reports **two counts**: total tripped,
and how many affect a strategy currently in an **actively-trading** stage (`PAPER`, `LIVE`). It is
purely informational: `ok` stays `true` (a tripped switch benches one strategy; it is not an
account-wide readiness failure and a switch left on a retired/unknown name is noise). Detail carries
the affected names so the operator sees them.

**`live_authorizations` (advisory by default; gating under `--live`).** Enumerates
`list_strategies(Stage.LIVE)`. For each name, calls the **existing public**
`verify_live_authorization(conn, repo, name, ALLOWED_SIGNERS_PATH)` with **ordered catches**:
`except LiveAuthorizationError` → `unauthorized(reason)` (revoked / no matching row / anchor
re-verify failed), then `except Exception` → `unverifiable(reason)`. The broad second catch is
deliberate: `verify_live_authorization` → `compute_artifact_hashes(name)` → `load_strategy(name)`
(`algua/registry/approvals.py:62`) **imports the strategy module** and can raise loader/config/model
errors, not only a signature mismatch. Each LIVE strategy lands in exactly one bucket: `authorized`
| `unauthorized` | `unverifiable`. **The row's `ok` is a single, mode-independent rule:**
`ok = (no LIVE strategies) OR (≥1 authorized)`. Degraded/unverifiable strategies are surfaced in the
`detail` (counts per bucket + names) but do **not** flip `ok` as long as at least one LIVE strategy
can trade — so under `--live`, one revoked strategy alongside a healthy one does not fail the pre-flight
(matching `live run-all`, which per-order fail-closed **skips** the bad ones and trades the rest).
The probe checks **authorization only** — NOT allocation, kill-switch, or candidate presence (those
are runner-time / #400 concerns, out of scope; see the `--live` note above). It gates (`required`)
only under `--live`; zero LIVE strategies → vacuous `ok:true` in every mode (an empty live book is a
ready state, not an infra failure).

**`alpaca_live_credentials` (advisory by default; required under `--live`).** Presence of
`alpaca_live_api_key`/`alpaca_live_api_secret`; raises if either is unset. Mirrors the merged
`paper_credentials` probe.

**`alpaca_paper_reachable` / `alpaca_live_reachable`.** These make an outbound call **only in a mode
that probes their lane** (`--paper` / `--live` respectively). In any other mode they are a
**skip**: `ok:true`, `detail = "skipped: pass --paper/--live to probe reachability"`, and **no
network call** — this keeps plain `doctor` non-networked and avoids a skip ever coinciding with a
`required` row. When the lane IS being probed:
- **creds absent** → `ok:false`, `detail = "cannot probe: <lane> credentials not configured"` (a
  required-row red under the lane flag; it is honest that the lane is not ready, and pairs with the
  credentials row pointing at the same root cause — two reds, no false green).
- **creds present** → a **single-shot, bounded** GET, mirroring
  `live_cmd.py::_live_account_equity`: `requests.get(f"{url}/v2/account", timeout=5,
  allow_redirects=False, headers=APCA-*)` — no retry (unlike the broker's `account()`, which is
  `_TIMEOUT=30` × `_MAX_RETRIES=3` + backoff ≈ ~90 s, unacceptable for a pre-flight); `timeout=5`,
  `allow_redirects=False` (inherits the #394 credential-leak guard). Host is already pinned
  https-paper/https-live by the settings validators. **Green iff** status == 200 **and** the JSON
  body parses as a valid account — validating the **same minimum fields
  `_AlpacaBroker.account()` requires**: a **non-empty** `id` + numeric `equity`, `cash`,
  `buying_power` (an empty `id` is a failure, matching the broker parser). A
  200 with a malformed/incomplete body is a **failure** (else a broken account payload passes
  `doctor` and then fails the real runner).

### Exit-code / `ok:false` contract (mechanism unchanged; required-set is per-mode)
```
all_ok = all(c["ok"] for c in checks if c["required"])
emit({"ok": all_ok, "checks": checks})
raise typer.Exit(code=0 if all_ok else 1)
```
The only change is **which probes carry `required=True`**, computed from the mode flags:
- always required: `python`, `registry_db`, `calendar`, `knowledge_base`, `global_halt`
- additionally under `--paper`: `paper_credentials`, `alpaca_paper_reachable`
- additionally under `--live`: `alpaca_live_credentials`, `alpaca_live_reachable`,
  `live_authorizations`
- `doctor --paper --live`: union of the two.

All probes always appear in the `checks` array (reachability as a skip when its lane isn't probed).
A **clean env, no flag** still exits 0 (`global_halt` passes with no row; `live_authorizations`/
`kill_switches` pass; reachability rows are skips `ok:true`; credential rows fail **advisory**) —
preserving `test_doctor_passes_in_clean_env` and `test_doctor_advisory_rows_do_not_gate_exit`.

### Import hygiene
Every new probe imports its heavier dependencies (`requests`, the broker/URL settings, `live_gate`,
`SqliteStrategyRepository`, `global_halt`, `kill_switch`) **inside the probe function**, matching the
existing lazy-import style of `_registry_db_detail`/`_knowledge_base_detail`. Module-import of
`app.py` (hence `algua version`) pulls in nothing new.

## Design forks (resolved)

- **What does plain `doctor` mean — trading readiness or generic local health?** → **Trading
  readiness** (per AGENTS.md). Hence `global_halt` gates it; creds/broker gate only the named lane.
- **Command-scoped flags vs. auto-scoping from registry state?** → **Mode flags.** Auto-scoping
  ("live creds fatal iff a LIVE strategy exists") surprises a benched-to-research operator. Explicit
  `--paper`/`--live` names the cycle; `live_authorizations` still auto-scopes *within* `--live` (only
  gates when LIVE strategies exist).
- **`live_authorizations` row `ok` rule?** → **`ok = no-LIVE OR ≥1-authorized`**, single rule for all
  modes; degraded names ride in `detail`, never flipping `ok` while one strategy can still trade.
- **Broker probe: reuse `account()` or single-shot?** → **Single-shot `timeout=5`, no retry**; the
  retrying `account()` can take ~90 s.
- **Reachability network in default mode?** → **No.** Reachability calls out **only** under its lane
  flag; otherwise it is a `skipped` (`ok:true`) row.
- **Live-auth catch order?** → `except LiveAuthorizationError` (→ unauthorized) **first**, then
  `except Exception` (→ unverifiable).

## CODEOWNERS impact
This PR modifies `algua/cli/app.py` (probes + `--paper`/`--live` wiring) and `algua/risk/
kill_switch.py` (additive `list_tripped`); it **imports/calls but does not modify** `live_gate.py`,
`store.py`, or `approvals.py`. Neither modified path is in the repo's `.github/CODEOWNERS`
(`/.github/`, `/.gitleaks.toml`, `/.pip-audit-ignore.txt`) or the workflow's protected list, so the
PR is expected not to require code-owner review — **to be confirmed against the final diff** (any
edit that lands *inside* a protected file flips it to human-merge-only; the design avoids that by
reusing the existing public `verify_live_authorization`/`list_strategies` unchanged).

## Non-goals
- No new schema / `SCHEMA_VERSION` bump — all state read via existing tables + helpers.
- No rename of the merged `paper_credentials` / `bars_snapshot` / `generated_provenance` probes.
- No change to `live run-all`'s own success semantics — `--live` readiness ("live-lane dependencies
  ready: live creds present + broker reachable + no authorization blocker among existing LIVE
  strategies; zero LIVE strategies is ready") is a property of `doctor`, not a new runner gate.
- No per-strategy fleet-status rollup — that is #400. `doctor --live` deliberately does **not**
  check per-strategy allocation, kill-switch, or LIVE-candidate presence (so it does not promise
  submittability, only lane-dependency readiness); those belong to #400.
- No `status: pass|fail|skip` envelope field — the only skip case (reachability outside its lane) is
  a genuine N/A and rides as `ok:true` + a `skipped:` detail; a required reachability row that cannot
  probe (creds absent under its flag) is `ok:false`, never a green skip.
- No change to `global_halt.py`, `live_gate.py`, `store.py`, `approvals.py`, the broker classes, or
  any command body; `doctor` only reads existing state through existing public APIs.

## Test plan (FAST per-task: `-k doctor` + whole-tree `ruff`/`mypy`/`lint-imports`)
- `global_halt` engaged → required row red, `ok:false`, exit 1 in **all** modes; cleared → exit 0.
- `kill_switches`: one tripped active-stage strategy → `ok:true` row with two counts (total/active),
  exit 0; tripped on a retired name → `ok:true`, active-count 0.
- `live_authorizations`: revoked LIVE strategy → `unauthorized` bucket; a LIVE strategy whose module
  fails to import → `unverifiable` bucket (assert broad-except classification via ordered catches,
  not a traceback); an authorized one → `authorized`. Row `ok` = no-LIVE OR ≥1-authorized: assert a
  mixed fleet (one authorized + one revoked) is `ok:true`, and a fleet with **zero** authorized is
  `ok:false`. **Default mode:** advisory (exit 0 even when `ok:false`). **`--live`:** `ok:false`
  (zero authorized, LIVE exist) → exit 1; ≥1 authorized → exit 0; zero LIVE → exit 0.
- `alpaca_live_credentials`: absent → advisory red (default) / required red under `--live`; present
  → green.
- Reachability (broker/`requests` monkeypatched — no real network): **no lane flag** → skip row
  `ok:true` `detail=skipped:...` **and assert the network client is never called**; **`--paper`
  with creds absent** → `ok:false` `cannot probe: ...`, exit 1, no network call; **`--paper` + 200 +
  valid account** → green, exit 0; **`--paper` + 200 malformed body** → red, exit 1; **`--paper` +
  `RequestException`/non-200** → red, exit 1. Same for `--live` / `alpaca_live_reachable`.
- `doctor --paper --live`: required set is the union; both broker calls happen.
- Regression: `test_doctor_passes_in_clean_env`, `test_doctor_advisory_rows_do_not_gate_exit` pass
  unchanged.

## Quality gate (FULL, at integration/finish)
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
