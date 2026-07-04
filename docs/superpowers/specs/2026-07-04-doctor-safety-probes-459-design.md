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
  strategy is silently benched, and `doctor` today surfaces **nothing** about it. #459 makes this
  state **visible** (advisory); it does **not** make it exit-gating — per-strategy tradability is
  #400's rollup (see the scope note below);
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

**Scope note on kill-switches (GATE-1 reconciliation of Problem ↔ Scope).** The states listed above
split into two treatments, and #459 is deliberate about which is which. `global_halt` (account-wide)
and lane creds / broker reachability / LIVE authorization become **exit-gating** — a false-green
there means "start a doomed cycle." A tripped **kill-switch**, by contrast, benches exactly **one**
strategy; it is not an account-wide readiness failure, so gating a whole trading cycle on it would be
wrong. The concrete bug #459 fixes for kill-switches is the **silence**, not a missing gate: today
`doctor` reports nothing, so an operator cannot see a tripped switch at all. The `kill_switches`
probe therefore **surfaces** tripped switches as an **advisory (never-gating, `ok:true`)** row so
they are no longer invisible. **Gating** a trading cycle on per-strategy tradability (tripped-switch
*and* allocation *and* candidate presence) is the per-strategy fleet rollup owned **wholesale by
#400** and is out of scope here. So this Problem section claims to fix the kill-switch false-green
by **making it visible**, exactly matching the Scope section's decision **not to gate** it — no
contradiction.

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
  existing local probes gate; the credential probes are **advisory**, and the **reachability and
  `live_authorizations` probes are inert**: they neither call the network nor import strategy
  modules — each reports a `skipped` row (see below). This is the key GATE-1 hardening:
  `live_authorizations`'s only execution path (`verify_live_authorization → compute_artifact_hashes
  → load_strategy`) **imports strategy modules**, a side effect plain `doctor` must never trigger.
  So the *probe body itself* — not merely its `required` flag — is gated behind `--live`. Plain
  `doctor` (and `doctor --paper`) therefore stays **local, fast, side-effect-free, and never makes
  an outbound call or imports a strategy module**.
- **`doctor --paper`** = "can the paper loop actually run?" → promotes `paper_credentials` and
  `alpaca_paper_reachable` to required (and performs the paper broker call).
- **`doctor --live`** = "are the live lane's **dependencies** ready?" → **turns on** (executes) and
  promotes to required `alpaca_live_credentials`, `alpaca_live_reachable`, and `live_authorizations`
  (and performs the live broker call). "Ready" is scoped to **lane dependencies**: live creds
  present, the live broker reachable, and **every LIVE strategy that exists re-verifies its
  authorization** (no revoked/unverifiable strategy in the book). This is the **strict** rule
  (tightened after GATE-1): `doctor --live` goes red if **any** existing `Stage.LIVE` strategy
  fails authorization, because a green `--live` must mean *every* staged live strategy can trade —
  a single silently-skipped-at-submit strategy is exactly the false-green #459 exists to kill.
  `doctor` is deliberately **stricter** than `live run-all`'s per-order fail-closed-skip: the runner
  trades the healthy strategies and skips the bad ones, but the pre-flight's job is to make the
  operator *notice* the bad ones **before** starting a cycle, not to mirror the runner's degraded-mode
  tolerance. `--live` still **does not promise that a strategy will actually trade**: whether any LIVE
  strategy has an active allocation or is not kill-switched are runner-time / fleet-status concerns
  (`live_authorizations` checks *authorization* only; per-strategy allocation + kill-switch health is
  #400's `fleet status` rollup, deliberately out of scope here — folding them in would duplicate #400
  and couple `doctor` to allocation state).
  **Zero LIVE strategies is a ready state under `--live`** (the live infra is fine; there is simply
  nothing staged to authorize — an empty live book is not an infrastructure failure). To keep that
  from reading as a silent green, the `live_authorizations` detail **always surfaces `live_strategies=N`
  explicitly** (so an empty book shows `live_strategies=0`, not a bare `ok:true`).
  This readiness definition is a property of `doctor --live`; it does not change `live run-all`'s own
  success semantics.
- **`doctor --paper --live`** (both) — required set = the **union** of both lanes' required probes;
  both broker calls are performed. Fully composable.

### Command help text — the narrowed readiness claim (GATE-1 HIGH-3)
The claim each mode makes is pinned in the flag **help strings** (and mirrored in the exit-code
contract below), so the contract lives at the command surface an operator/agent actually reads —
not only in this design doc's non-goals:
- `--paper` help: *"gate exit on paper-lane dependencies (paper creds + paper broker /v2/account
  reachability)."*
- `--live` help: *"gate exit on live-lane DEPENDENCIES ONLY — live creds present, live broker
  /v2/account reachable, and every existing LIVE strategy re-verifies its go-live authorization.
  Does NOT assert any strategy will actually trade: allocation / kill-switch / candidate presence
  are fleet-status concerns (see `fleet status` / #400)."*

So `doctor --live` readiness is unambiguously **lane-dependency readiness, NOT tradability**. A green
`--live` with `live_strategies=0` means "the live infra is ready and there is simply nothing staged
to authorize" — NOT "a strategy is ready to trade." `live_strategies=N` rides in the
`live_authorizations` detail purely as an **informational** signal; it does **not** itself gate (the
gating condition is *every existing LIVE strategy authorized*, which is vacuously true at `N=0`). We
**deliberately reject** adding a gating `tradable_live_strategies` signal (finding-3 option a):
tradability (allocation + kill-switch) is #400's rollup, and coupling `doctor`'s exit code to
allocation state would duplicate #400 and would red-flag a perfectly-ready live infrastructure the
moment its book is empty or benched — surprising a benched-to-research operator. We take option (b):
**narrow and pin the claim.**

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
flags at the call site. No schema change, no `SCHEMA_VERSION` bump.

**Structured skip signal (GATE-1 HIGH-2 — replaces the `detail`-prefix convention).** Every check
row carries a new boolean field **`skipped`** (`_check` adds `"skipped": False` to its returned
dict; a real pass/fail is `skipped:false`). A **skip** row — a lane-gated probe whose lane flag is
absent — is emitted by a new tiny helper `_skip_row(name, detail) ->
{"check":name, "ok":True, "required":False, "detail":detail, "skipped":True}` at the **call site**
(the mode flags live there), so a skipped probe's `fn` is never invoked (this is what keeps plain
`doctor` from importing a strategy module or touching the network — the body itself is gated, not
just the `required` flag). A consumer that reads only `ok` can no longer mistake an un-probed
dependency for a passing one: the authoritative, machine-legible discriminator is `skipped:true`
(the human-readable `detail` still begins `"skipped: …"`, but the boolean is what code branches on).
This is a purely additive JSON field on the per-check row — **not** an envelope field, **not** a
`status` enum, and **not** a `SCHEMA_VERSION`-versioned surface (the `doctor` payload is an
uncounted diagnostic, not a persisted ledger row).

| probe | row `ok` is false when | required in mode |
|---|---|---|
| `global_halt` | a halt row exists (`global_halt.get(conn)` non-None) | **all modes** |
| `kill_switches` | never — always `ok:true`; detail lists any tripped switches + affected operational-stage names (advisory-informational) | never |
| `live_authorizations` | runs **only under `--live`**; `ok` false when **any** existing `Stage.LIVE` strategy fails to re-verify (`ok` = *no LIVE strategies* OR *ALL authorized*). Outside `--live`: **`skipped:true` row (`ok:true`), no strategy-module import** | **`--live`** |
| `alpaca_live_credentials` | live key or secret unset | **`--live`** |
| `alpaca_paper_reachable` | in a mode that probes paper (`--paper`) and the `/v2/account` call is not a valid account response; **`skipped:true` row (`ok:true`), no network when not probing** | **`--paper`** |
| `alpaca_live_reachable` | in a mode that probes live (`--live`) and the `/v2/account` call is not a valid account response; **`skipped:true` row (`ok:true`), no network when not probing** | **`--live`** |

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
and how many affect a strategy currently in an **operational** (loop-ticked) stage. To avoid a
second, drifting definition of "actively-trading" (the GATE-1 MEDIUM), that classification
**reuses `algua.execution.fleet_health.OPERATIONAL_STAGES`** — the single-sourced `{LIVE, PAPER,
FORWARD_TESTED}` set that #400/#399 already pin (via `test_operational_stages_match_gating`) to the
real tick surface — rather than inventing a private `{PAPER, LIVE}` list here (which would silently
omit `FORWARD_TESTED` and drift from the fleet-health model). `algua.cli → algua.execution` is an
allowed import (mirrors `live_cmd`), so this is a plain reuse, not a new dependency. The probe is a
**thin dependency probe**: it does not recompute per-strategy liveness/drift/staleness — its detail
**points the operator to `fleet health`** for the full #400 rollup. It is purely informational: `ok`
stays `true` (a tripped switch benches one strategy; it is not an account-wide readiness failure and
a switch left on a retired/unknown name is noise). Detail carries the affected operational-stage
names so the operator sees them.

**`live_authorizations` (runs and gates ONLY under `--live`; skipped — no strategy-module import —
otherwise).** This probe's body is gated behind `--live`, not just its `required` flag (the GATE-1
HIGH-2 fix). **Outside `--live`** (plain `doctor`, `doctor --paper`) it emits a **`_skip_row`**:
`skipped:true`, `ok:true`, `required:false`, `detail = "skipped: pass --live to probe live
authorizations"`, and it **does not call `verify_live_authorization`** — so it never imports a
strategy module. This is load-bearing: the
verification path `verify_live_authorization → compute_artifact_hashes(name) → load_strategy(name)`
(`algua/registry/approvals.py:62`) **imports the strategy module** (arbitrary import-time code +
config/model construction), a side effect that would violate plain `doctor`'s "local, fast,
side-effect-free" contract if it ran unconditionally.

**Under `--live`** it enumerates `list_strategies(Stage.LIVE)` and, for each name, calls the
**existing public** `verify_live_authorization(conn, repo, name, ALLOWED_SIGNERS_PATH)` with
**ordered catches**: `except LiveAuthorizationError` → `unauthorized(reason)` (revoked / no matching
row / anchor re-verify failed), then `except Exception` → `unverifiable(reason)` (the broad second
catch is deliberate — `load_strategy` can raise loader/config/model errors, not only a signature
mismatch). Each LIVE strategy lands in exactly one bucket: `authorized` | `unauthorized` |
`unverifiable`.

**The row's `ok` rule is the strict, all-must-verify rule (tightened after GATE-1):**
`ok = (no LIVE strategies) OR (EVERY LIVE strategy is authorized)`. **Any** `unauthorized` **or**
`unverifiable` strategy flips `ok:false`. The earlier "≥1 authorized" rule is **rejected**: it left
`doctor --live` green while a known revoked/unverifiable LIVE strategy was guaranteed to be
skipped-at-submit — the precise false-green #459 exists to kill. `doctor` is therefore intentionally
**stricter than `live run-all`** (which per-order fail-closed **skips** the bad ones and trades the
rest): the pre-flight's contract is "*every* staged live strategy can trade — investigate before you
start a cycle", not the runner's degraded-mode tolerance. Per-bucket counts + names ride in the
`detail`, and the detail **always surfaces `live_strategies=N`** (so an empty book reads as an
explicit `live_strategies=0`, ready — not a bare silent green; the GATE-1 MEDIUM). The probe checks
**authorization only** — NOT allocation, kill-switch, or candidate presence (those are runner-time /
#400 concerns, out of scope; see the `--live` note above). Zero LIVE strategies → vacuous `ok:true`
(an empty live book is a ready state, not an infra failure).

**`alpaca_live_credentials` (advisory by default; required under `--live`).** Presence of
`alpaca_live_api_key`/`alpaca_live_api_secret`; raises if either is unset. Mirrors the merged
`paper_credentials` probe.

**`alpaca_paper_reachable` / `alpaca_live_reachable`.** These make an outbound call **only in a mode
that probes their lane** (`--paper` / `--live` respectively). In any other mode they are a **skip**
row (emitted via `_skip_row`, so `skipped:true`, `ok:true`, `required:false`,
`detail = "skipped: pass --paper/--live to probe reachability"`) with **no network call** — this
keeps plain `doctor` non-networked and avoids a skip ever coinciding with a `required` row. When the
lane IS being probed:
- **creds absent** → `ok:false`, `detail = "cannot probe: <lane> credentials not configured"` (a
  required-row red under the lane flag — `skipped:false`; it is honest that the lane is not ready,
  and pairs with the credentials row pointing at the same root cause — two reds, no false green).
- **creds present** → a **single-shot, bounded** GET, mirroring
  `live_cmd.py::_live_account_equity`: `requests.get(f"{url}/v2/account", timeout=5,
  allow_redirects=False, headers=APCA-*)` — no retry (unlike the broker's `account()`, which is
  `_TIMEOUT=30` × `_MAX_RETRIES=3` + backoff ≈ ~90 s, unacceptable for a pre-flight); `timeout=5`,
  `allow_redirects=False` (inherits the #394 credential-leak guard). Host is already pinned
  https-paper/https-live by the settings validators. **Green iff** status == 200 **and** the JSON
  body parses as a valid account — validating the **same minimum fields
  `_AlpacaBroker.account()` requires** and then some: a **non-empty** `id`, plus `equity`, `cash`,
  `buying_power` that each parse as a **finite, non-negative real** number. A 200 with a
  malformed/incomplete body is a **failure** (else a broken account payload passes `doctor` and then
  fails the real runner).
- **Numeric validation is strict (GATE-1 MEDIUM (b)).** "Numeric" is not "parses as a float":
  `float("nan")`, `float("inf")`, and negative values are **rejected** (`ok:false`). The probe uses
  `math.isfinite(v) and v >= 0` on each of `equity`/`cash`/`buying_power` after coercion; a `NaN`
  `equity` (which would silently poison downstream sizing) is a broken payload, not a pass. This
  matches — and slightly hardens — the broker parser's own contract.
- **Failure detail is redaction-safe (GATE-1 MEDIUM (c)).** The reachability probe **never** lets a
  raw response body or a raw exception `repr` reach the row `detail` (either could carry the request
  URL, headers, or account metadata). It catches its own exceptions and raises a **fixed,
  pre-classified** message with **no interpolated body/exception text**: `requests.Timeout` →
  `"unreachable: timeout"`; other `requests.RequestException` (connection/TLS/DNS) →
  `"unreachable: connection error"`; a non-200 → `"unreachable: HTTP <status>"` (status code only,
  never the body); a 200-but-invalid body → `"invalid account payload"` (which field failed may be
  named — e.g. `"invalid account payload: equity not finite"` — but **no value is echoed**). The
  probe must therefore not rely on `_check`'s generic `str(exc)` for the network path (that would
  leak the `requests` exception repr, which includes the URL); it maps to the safe strings itself
  and raises those.

**Scope of the reachability probes (the GATE-1 MEDIUM).** These probe **only the `/v2/account`
endpoint** — i.e. credential validity + account-API liveness. They deliberately do **not** exercise
the other endpoints `live run-all` depends on (`/v2/account/activities`, `/v2/positions`,
`/v2/orders`, order submit/cancel). A green reachability row therefore attests "creds are valid and
the account API answered", NOT "every runner endpoint is up". To keep that honest and
machine-legible, the success `detail` is explicitly scoped —
`detail = "account API reachable (/v2/account); does not probe positions/orders/activities"` — so
neither a human nor the autonomous loop can read the green as full runner reachability. (A single
bounded account GET is the right cost/coverage tradeoff for a pre-flight; a full endpoint sweep
belongs to the runner's own startup, not `doctor`.)

### Exit-code / `ok:false` contract (mechanism unchanged; required-set is per-mode)
Each row is now `{"check", "ok", "required", "detail", "skipped"}` (the `skipped` boolean is the
only additive field). The exit-code mechanism is unchanged and never consults `skipped` (a skip row
is `ok:true, required:false`, so it cannot gate regardless):
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

All probes always appear in the `checks` array. **`live_authorizations` and the two reachability
probes are `_skip_row`s (`skipped:true`, `ok:true`, `required:false`, no side effect) whenever their
lane flag is absent** — so plain `doctor` and `doctor --paper` never import a strategy module and
never touch the network. A consumer distinguishes "passed" from "not probed" via the `skipped`
boolean, never by string-matching `detail`. A **clean env, no flag** still exits 0 (`global_halt`
passes with no row; `kill_switches` passes; `live_authorizations` and both reachability rows are
`skipped:true` skips; credential rows fail **advisory**) — preserving
`test_doctor_passes_in_clean_env` and `test_doctor_advisory_rows_do_not_gate_exit`.

### Import hygiene
Every new probe imports its heavier dependencies (`requests`, the broker/URL settings, `live_gate`,
`SqliteStrategyRepository`, `global_halt`, `kill_switch`, and
`algua.execution.fleet_health.OPERATIONAL_STAGES` for the `kill_switches` stage classification)
**inside the probe function**, matching the existing lazy-import style of
`_registry_db_detail`/`_knowledge_base_detail`. Module-import of `app.py` (hence `algua version`)
pulls in nothing new. Crucially, `live_authorizations`'s `verify_live_authorization` import **and its
call** live inside the `--live` branch of the probe body, so plain `doctor` never even imports the
authorization/strategy-loader path.

**`requests` is a declared dependency (GATE-1 MEDIUM (a)).** The reachability probes import
`requests`, so it becomes an **explicit** entry under `[project].dependencies` in `pyproject.toml`
rather than riding in transitively (today it is only pulled by `alpaca-py`/others). This PR adds that
line so the reachability path can never break on a transitive-dependency drop; `uv lock` already
resolves it, so no lockfile churn beyond recording the direct edge.

**`OPERATIONAL_STAGES` reuse vs. relocation (GATE-1 MINOR (d)).** The `kill_switches` probe runs in
all modes, so its lazy `from algua.execution.fleet_health import OPERATIONAL_STAGES` is the one new
edge that touches the (heavier) `algua.execution` package on the *default* `doctor` path. We keep the
reuse rather than fork a private stage set: `cli → execution` is an already-allowed import edge
(mirrors `live_cmd`/`fleet_cmd`), the import is **lazy** (inside the probe, so `algua version` and
module-load pay nothing), and single-sourcing the `{LIVE, PAPER, FORWARD_TESTED}` set — which #400's
`test_operational_stages_match_gating` pins to the real tick surface — is worth more than avoiding
one in-function import of a constant. Lifting `OPERATIONAL_STAGES` to a leaf module (e.g. next to
`Stage` in `algua/contracts/lifecycle.py`) so neither `cli` nor a future caller drags in
`live_ledger` for a frozenset is the cleaner long-term home, but is **deferred** (it edits #400's
module surface for no behavioural gain here); the lazy edge is the pragmatic choice for this PR.

## Design forks (resolved)

- **What does plain `doctor` mean — trading readiness or generic local health?** → **Trading
  readiness** (per AGENTS.md). Hence `global_halt` gates it; creds/broker gate only the named lane.
- **Command-scoped flags vs. auto-scoping from registry state?** → **Mode flags.** Auto-scoping
  ("live creds fatal iff a LIVE strategy exists") surprises a benched-to-research operator. Explicit
  `--paper`/`--live` names the cycle; `live_authorizations` still auto-scopes *within* `--live` (only
  gates when LIVE strategies exist).
- **`live_authorizations` row `ok` rule — `≥1-authorized` or `all-authorized`?** → **`ok = no-LIVE OR
  ALL-authorized`** (tightened after GATE-1). The `≥1-authorized` rule was rejected: it left
  `doctor --live` green with a known revoked/unverifiable LIVE strategy guaranteed to be
  skipped-at-submit — the exact false-green #459 targets. `doctor` is deliberately stricter than
  `live run-all`'s degraded-mode skip; per-bucket names + `live_strategies=N` ride in `detail`.
- **Does `live_authorizations` run in plain `doctor`?** → **No** (GATE-1 HIGH-2). The probe *body*,
  not just its `required` flag, is gated behind `--live`; outside `--live` it is an inert skip row.
  Otherwise `verify_live_authorization → load_strategy` would import strategy modules in the
  advertised "local, fast, side-effect-free" default mode.
- **`kill_switches` "actively-trading" stage set — new private list or reuse #400?** → **Reuse
  `fleet_health.OPERATIONAL_STAGES`** (`{LIVE, PAPER, FORWARD_TESTED}`), the single-sourced set #400
  already drift-guards, rather than a private `{PAPER, LIVE}` notion that would omit
  `FORWARD_TESTED` and diverge from the fleet-health model. The probe stays a thin dependency probe
  and points to `fleet health` for the full rollup.
- **Reachability coverage — account only or full endpoint sweep?** → **`/v2/account` only**, with the
  success `detail` explicitly scoping the claim ("does not probe positions/orders/activities") so a
  green row is never misread as full runner reachability. A full sweep belongs to runner startup.
- **Broker probe: reuse `account()` or single-shot?** → **Single-shot `timeout=5`, no retry**; the
  retrying `account()` can take ~90 s.
- **Reachability network in default mode?** → **No.** Reachability calls out **only** under its lane
  flag; otherwise it is a `skipped` (`ok:true`) row.
- **Live-auth catch order?** → `except LiveAuthorizationError` (→ unauthorized) **first**, then
  `except Exception` (→ unverifiable).
- **Kill-switch: gate it, or only surface it? (GATE-1 HIGH-1)** → **Surface only.** The Problem
  section flags kill-switch invisibility; the fix is making it **visible** (advisory `kill_switches`
  row), NOT exit-gating. A tripped switch benches one strategy, not the account; gating a cycle on
  per-strategy tradability (switch + allocation + candidate) is #400's rollup. This keeps Problem and
  Scope consistent — "fix the false-green" here means "kill the silence," not "add a gate."
- **Skip signal: `detail` string-prefix or a structured field? (GATE-1 HIGH-2)** → **Structured
  boolean `skipped`** on every row (`_skip_row` emits `skipped:true`; `_check` emits `skipped:false`).
  A consumer reading only `ok` can no longer confuse an un-probed dependency with a passing one; the
  human `"skipped: …"` `detail` stays for readability but is no longer the machine contract. Not a
  `status` enum, not an envelope field, no `SCHEMA_VERSION`.
- **`doctor --live` when zero LIVE strategies are tradable? (GATE-1 HIGH-3)** → **Narrow the claim,
  don't add a tradability gate.** `--live` readiness is pinned (in the flag **help text** + exit-code
  contract) as **lane-dependency readiness only** — live creds + broker reachable + every existing
  LIVE strategy authorized — explicitly **NOT** "a strategy will trade." `live_strategies=N` is
  informational, not gating; `N=0` is ready (infra fine, nothing staged). A gating
  `tradable_live_strategies` signal (allocation/kill-switch) was rejected: it is #400's job and would
  red-flag ready infra on an empty/benched book.
- **Reachability numeric validation — "parses as float" or strict finite/non-negative? (GATE-1 MED
  (b))** → **Strict**: `equity`/`cash`/`buying_power` must each be finite (`isfinite`) and `>= 0`;
  `NaN`/`inf`/negative fail the row, matching (and hardening) the broker parser.
- **Reachability failure detail — echo body/exception, or fixed classes? (GATE-1 MED (c))** →
  **Fixed, redaction-safe classes** (`unreachable: timeout` / `unreachable: connection error` /
  `unreachable: HTTP <status>` / `invalid account payload[: field]`) — never a response body or raw
  exception `repr`, which can carry URL/headers/account metadata.
- **`requests` — transitive or declared? (GATE-1 MED (a))** → **Declared** direct dependency in
  `pyproject.toml`; the reachability path must not rest on a transitive edge.
- **`OPERATIONAL_STAGES` — reuse `fleet_health` or relocate? (GATE-1 MINOR (d))** → **Reuse now**
  (lazy in-probe import; `cli → execution` is allowed; single-source over the drift-guarded set),
  **relocate later** to a leaf module — deferred as it edits #400's surface for no behavioural gain.

## CODEOWNERS impact
This PR modifies `algua/cli/app.py` (probes + `--paper`/`--live` wiring + additive per-row `skipped`
field + `_skip_row` helper), `algua/risk/kill_switch.py` (additive `list_tripped`), and
`pyproject.toml` (declares `requests` as a direct dependency); it **imports/calls but does not
modify** `live_gate.py`,
`store.py`, `approvals.py`, and `algua/execution/fleet_health.py` (reuses the exported
`OPERATIONAL_STAGES` constant only). None of these modified/imported paths is in the repo's
`.github/CODEOWNERS`
(`/.github/`, `/.gitleaks.toml`, `/.pip-audit-ignore.txt`) or the workflow's protected list, so the
PR is expected not to require code-owner review — **to be confirmed against the final diff** (any
edit that lands *inside* a protected file flips it to human-merge-only; the design avoids that by
reusing the existing public `verify_live_authorization`/`list_strategies` unchanged).

## Non-goals
- No new schema / `SCHEMA_VERSION` bump — all state read via existing tables + helpers. (The additive
  per-row `skipped` boolean is a field of the `doctor` diagnostic payload, not a persisted/versioned
  ledger row, so it carries no schema-version implication.)
- No rename of the merged `paper_credentials` / `bars_snapshot` / `generated_provenance` probes.
- No change to `live run-all`'s own success semantics — `--live` readiness ("live-lane dependencies
  ready: live creds present + broker reachable + **every** existing LIVE strategy re-verifies; zero
  LIVE strategies is ready") is a property of `doctor`, not a new runner gate. `doctor --live` is
  intentionally **stricter** than the runner (which skips a revoked strategy and trades the rest);
  the pre-flight surfaces the blocker instead of tolerating it.
- No per-strategy fleet-status rollup — that is #400. `doctor --live` deliberately does **not**
  check per-strategy allocation, kill-switch, or LIVE-candidate presence (so it does not promise
  submittability, only lane-dependency readiness); those belong to #400.
- No `status: pass|fail|skip` **enum** and no **envelope**-level field. The skip cases (reachability,
  and `live_authorizations`, outside their lane flag) are genuine N/A and are marked by an additive
  **per-row boolean `skipped:true`** (GATE-1 HIGH-2 — supersedes the earlier detail-prefix-only plan):
  a machine reader branches on the boolean, never on a `detail` substring, so an un-probed dependency
  can never be mistaken for a passed one. The human `"skipped: …"` `detail` remains for readability.
  A required reachability row that cannot probe (creds absent under its flag) is `ok:false`
  `skipped:false`, never a green skip.
- No change to `global_halt.py`, `live_gate.py`, `store.py`, `approvals.py`, the broker classes, or
  any command body; `doctor` only reads existing state through existing public APIs.

## Test plan (FAST per-task: `-k doctor` + whole-tree `ruff`/`mypy`/`lint-imports`)
- `global_halt` engaged → required row red, `ok:false`, exit 1 in **all** modes; cleared → exit 0.
- `kill_switches`: one tripped operational-stage strategy → `ok:true` row with two counts
  (total/operational), exit 0; tripped on a retired name → `ok:true`, operational-count 0; a tripped
  `FORWARD_TESTED` strategy counts as operational (asserts the probe uses
  `fleet_health.OPERATIONAL_STAGES`, not a private `{PAPER, LIVE}` set).
- **Structured `skipped` field (HIGH-2):** every check row carries `skipped`; a real pass/fail is
  `skipped:false`, and each lane-gated skip is `skipped:true`. Assert a consumer can distinguish
  "passed" from "not probed" **on the boolean alone** — e.g. a plain-`doctor` `live_authorizations`
  row has `ok:true, skipped:true` while `global_halt` has `ok:true, skipped:false`.
- `live_authorizations`:
  - **Outside `--live` (plain `doctor`, `doctor --paper`):** skip row `ok:true`, **`skipped:true`**,
    `detail="skipped: ..."`, exit 0, **and assert `verify_live_authorization` is never called / no
    strategy module is imported** (monkeypatch/spy — this is the HIGH-2 regression guard).
  - **Under `--live`:** revoked LIVE strategy → `unauthorized` bucket; a LIVE strategy whose module
    fails to import → `unverifiable` bucket (assert broad-except classification via ordered catches,
    not a traceback); an authorized one → `authorized`. Strict row rule `ok = no-LIVE OR
    ALL-authorized`: a **mixed** fleet (one authorized + one revoked) is now `ok:false` → **exit 1**
    (the tightened HIGH-1 behaviour — assert it is NOT green); an all-authorized fleet → `ok:true`
    exit 0; a fleet with an `unverifiable` strategy → `ok:false` exit 1; **zero LIVE → exit 0** and
    detail contains `live_strategies=0`. Detail carries per-bucket names + `live_strategies=N`.
- **`--live` help / narrowed-claim contract (HIGH-3):** assert the `--live` flag's help string states
  readiness is **lane-dependency only** (mentions creds + reachability + authorization, and disclaims
  tradability / points to `fleet status`/#400); assert `live_strategies` does **not** gate — a green
  `--live` at `live_strategies=0` exits 0, and no `tradable_live_strategies` field is emitted.
- `alpaca_live_credentials`: absent → advisory red (default) / required red under `--live`; present
  → green.
- Reachability (broker/`requests` monkeypatched — no real network): **no lane flag** → skip row
  `ok:true`, **`skipped:true`**, `detail=skipped:...` **and assert the network client is never
  called**; **`--paper` with creds absent** → `ok:false`, `skipped:false`, `cannot probe: ...`, exit
  1, no network call; **`--paper` + 200 + valid account** → green, exit 0, **and assert the success
  detail is account-scoped (mentions `/v2/account`, disclaims positions/orders/activities)**;
  **`--paper` + 200 malformed body** → red, exit 1; **`--paper` + 200 body with `NaN`/`inf`/negative
  `equity` (or `cash`/`buying_power`)** → red, exit 1 (MED (b) strict-numeric guard); **`--paper` +
  `RequestException`/timeout/non-200** → red, exit 1. Same for `--live` / `alpaca_live_reachable`.
- **Redaction (MED (c)):** on a `RequestException` (repr carrying the URL) and on a non-200, assert
  the row `detail` is the fixed class string (`unreachable: connection error` / `unreachable:
  timeout` / `unreachable: HTTP <status>`) and does **not** contain the URL, response body, or raw
  exception repr; on an invalid-payload failure assert no field **value** is echoed.
- `doctor --paper --live`: required set is the union; both broker calls happen.
- Regression: `test_doctor_passes_in_clean_env`, `test_doctor_advisory_rows_do_not_gate_exit` pass
  unchanged; existing rows tolerate the additive `skipped` field (assert `test_cli_core` probe-name
  assertions still hold — no probe renamed).

## Quality gate (FULL, at integration/finish)
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
