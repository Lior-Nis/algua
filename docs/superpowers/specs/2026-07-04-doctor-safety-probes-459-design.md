# Doctor safety-state & live-authorization probes (#459)

> **Scope of THIS PR (read first).** This PR ships the **safety-state + live-authorization** slice of
> #459: it makes `doctor` a *trading-readiness* pre-flight that goes red on an engaged global halt
> (all modes) and, under a new `--live` flag, on any un-re-verifiable LIVE strategy. It adds a
> `--paper` flag that promotes the existing paper-credential probe to required, and a per-row
> `skipped` boolean. It does **NOT** ship broker-reachability probes, the `requests` direct
> dependency, an `alpaca_live_credentials` probe, or the `kill_switches` operational-stage rollup —
> those are **deferred to an explicit follow-up** (see "Deferred / follow-up" below). The spec text
> describes only what this diff actually does.

## Problem
`uv run algua doctor` is advertised (AGENTS.md §1) as *the* environment readiness self-check — the
agent's only pre-flight before a paper/live cycle. Before this PR it ran seven probes: four
**required** (`python`, `registry_db`, `calendar`, `knowledge_base`) and three **advisory**
(`paper_credentials`, `bars_snapshot`, `generated_provenance`), then emitted
`{ok: all-required-pass, checks:[...]}` and exited 0/1 on the required set.

It probed **no safety state**. `doctor` returned an all-green `{ok:true}` while:
- **the global halt is engaged** (`algua/risk/global_halt.py` `is_engaged`) — every paper/live
  trading tick raises on the very next call, account-wide;
- **a per-strategy kill-switch is tripped** (`algua/risk/kill_switch.py`) — that strategy is silently
  benched, and `doctor` surfaced **nothing** about it;
- **a LIVE strategy's human go-live authorization is revoked or no longer re-verifies** against the
  `approvers/allowed_signers` trust anchor (`algua/registry/live_gate.py` `verify_live_authorization`,
  re-run before every live order) — that strategy is fail-closed skipped at submit, yet the operator
  had no pre-flight signal.

The Health-Checks playbook (KB software-engineering/15-observability): *"a health check that only
confirms the process is up is worse than none."* An operator (human or autonomous loop) who trusts a
green `doctor` during a global halt or after a live-auth revocation believes the system is healthy and
burns the cold-start on a guaranteed-to-fail cycle, discovering the gap only via a raise inside the
command body.

Distinct from #400 (no per-strategy fleet-status rollup): this is about the EXISTING advertised
health check returning misleadingly-healthy output on **safety state**.

## What `doctor` *means*
`doctor` is a **trading-readiness** pre-flight, exactly as AGENTS.md advertises it. Its exit code
answers **"is it safe and possible to start a trading cycle right now?"** — NOT "can I run a
backtest?" (research/backtest commands never consult `doctor`'s exit code; they run regardless). Two
consequences:
1. **Safety state (`global_halt`) gates every mode.** A global halt is an account-wide trading
   emergency; a trading-readiness check MUST go red during it. This is the exact false-green #459
   exists to kill — making it advisory would re-introduce the bug. An operator who engaged a halt on
   purpose correctly sees red: "do not start a trading cycle now."
2. **Lane-specific dependencies gate only the lane you name** (`--paper` / `--live`), because a
   generic `doctor` cannot know which lane you are about to run.

## Scope of this change

### Mode flags: `--paper` / `--live`
`doctor` grows two combinable mode flags that declare the lane being pre-flighted and promote **that
lane's** probes to **required (exit-gating)**. The claim each mode makes is pinned in the flag **help
strings** (the surface an operator/agent actually reads):
- **`doctor` (no flag)** — safety-state + local-invariant readiness. `global_halt` and the four
  existing local probes gate; `paper_credentials` stays advisory; `live_authorizations` is **inert**:
  it neither runs its body nor imports any strategy module — it reports a `skipped:true` row.
- **`doctor --paper`** — promotes `paper_credentials` to **required**.
  Help: *"Gate exit on the paper lane's dependencies: promotes paper_credentials to required."*
- **`doctor --live`** — **runs and requires** `live_authorizations`: every existing `Stage.LIVE`
  strategy must re-verify its go-live authorization.
  Help: *"Gate exit on the live lane's AUTHORIZATION ONLY: run and require the live_authorizations
  probe so every existing LIVE strategy must re-verify its go-live signature. This is the only mode
  that imports strategy modules. Does NOT assert any strategy will actually trade — allocation /
  kill-switch / candidate presence are fleet-status concerns (see `fleet health` / #400)."*
- **`doctor --paper --live`** — required set = the union. Fully composable.

So `doctor --live` readiness is unambiguously **live-lane AUTHORIZATION readiness, NOT tradability**.
A green `--live` with `live_strategies=0` means "there is simply nothing staged to authorize" — NOT
"a strategy is ready to trade." `live_strategies=N` rides in the `live_authorizations` detail as an
**informational** signal; it does not itself gate (the gating condition is *every existing LIVE
strategy authorized*, vacuously true at `N=0`). We **deliberately reject** a gating
`tradable_live_strategies` signal (allocation/kill-switch): that is #400's rollup, and coupling
`doctor`'s exit code to allocation state would red-flag a perfectly-ready live infra the moment its
book is empty or benched.

### The `skipped` boolean
Every check row carries an additive boolean **`skipped`**. A real pass/fail (from `_check`) is
`skipped:false`; a lane-gated probe whose flag is absent is emitted at the **call site** by a tiny
helper `_skip_row(name, detail) -> {"check", "ok":True, "required":False, "detail", "skipped":True}`,
so the skipped probe's `fn` is **never invoked**. This is what keeps plain `doctor` from importing a
strategy module: the *body* is gated, not merely the `required` flag. A consumer that reads only `ok`
can no longer mistake an un-probed dependency for a passing one — the authoritative discriminator is
`skipped:true` (the human `detail` still begins `"skipped: …"`, but the boolean is what code branches
on). This is a purely additive JSON field on the per-check row — not an envelope field, not a
`status` enum, not a `SCHEMA_VERSION`-versioned surface (the `doctor` payload is an uncounted
diagnostic, not a persisted ledger row).

### Probes shipped in this PR

| probe | required in mode | row `ok` is false when |
|---|---|---|
| `global_halt` | **all modes** | a halt row exists (`global_halt.get(conn)` non-None) |
| `kill_switches` | never (advisory) | any tripped switch (`ok:false`, `required:false` — reported but never gates exit); detail lists tripped switch names |
| `live_authorizations` | **`--live`** | runs **only under `--live`**; `ok` false when **any** existing `Stage.LIVE` strategy fails to re-verify (`ok = no-LIVE OR ALL-authorized`). Outside `--live`: `skipped:true` row (`ok:true`), **no strategy-module import** |
| `paper_credentials` (already merged) | **`--paper`** | Alpaca paper key/secret unset (advisory default; required under `--paper`) |

`bars_snapshot`, `generated_provenance` stay advisory in all modes.

**`global_halt` (required, all modes).** Reads `global_halt.get(conn)`; raises with
`reason/actor/created_at` when a halt row exists; else returns "no global halt engaged". A global halt
is the one state under which no paper/live **trading cycle** can start (every tick raises),
account-wide — so trading-readiness is false while it is engaged, in every mode.

**`kill_switches` (advisory, all modes, never gates).** Uses the additive helper
`kill_switch.list_tripped(conn) -> list[str]` (`SELECT strategy FROM kill_switches ORDER BY
strategy`). Detail lists any tripped strategy names. It stays **advisory** by deliberate design: a
tripped switch benches exactly **one** strategy; it is not an account-wide readiness failure, so
gating a whole trading cycle on it would be wrong. The bug #459 fixes for kill-switches is the
**silence**, not a missing gate — today `doctor` reports nothing, so an operator cannot see a tripped
switch at all. This probe makes it **visible** as a never-gating (`ok:true`) row. (Gating a cycle on
per-strategy tradability — switch *and* allocation *and* candidate presence — is #400's fleet rollup,
out of scope here.)

**`live_authorizations` (runs and gates ONLY under `--live`; inert skip otherwise).** The probe's
**body** is gated behind `--live` at the call site (`_skip_row` when the flag is absent), not merely
its `required` flag. This is load-bearing: the verification path `verify_live_authorization →
compute_artifact_hashes(name) → load_strategy(name)` **imports the strategy module** (arbitrary
import-time code), a side effect that would violate plain `doctor`'s "local, fast, side-effect-free"
contract if it ran unconditionally. Outside `--live` (plain `doctor`, `doctor --paper`) the row is
`skipped:true, ok:true, required:false, detail="skipped: pass --live to probe live authorizations"`
and `verify_live_authorization` is **never called**.

Under `--live` it enumerates `list_strategies(Stage.LIVE)` and, for each name, calls the **existing
public** `verify_live_authorization(conn, repo, name, ALLOWED_SIGNERS_PATH)`, catching any exception
per-strategy (`LiveAuthorizationError` for revoked/no-matching-row/anchor-fail, or a broader loader
error from `load_strategy`) so one failure never masks another. **Strict `ok` rule:**
`ok = (no LIVE strategies) OR (EVERY LIVE strategy re-verifies)`. **Any** failing strategy flips
`ok:false`. The `≥1-authorized` rule is **rejected**: it left `doctor --live` green while a known
revoked/unverifiable LIVE strategy was guaranteed to be skipped-at-submit — the precise false-green
#459 targets. `doctor` is therefore intentionally **stricter than `live run-all`** (which per-order
fail-closed **skips** the bad ones and trades the rest): the pre-flight's contract is "*every* staged
live strategy can trade — investigate before you start a cycle." The detail **always surfaces
`live_strategies=N`** (an empty book reads as an explicit `live_strategies=0`, ready — not a bare
silent green). Authorization ONLY — not allocation, kill-switch, or candidate presence.

### Exit-code / `ok:false` contract
Each row is now `{"check", "ok", "required", "detail", "skipped"}` (the `skipped` boolean is the only
additive field). The exit-code mechanism is unchanged and never consults `skipped` (a skip row is
`ok:true, required:false`, so it cannot gate regardless):
```
all_ok = all(c["ok"] for c in checks if c["required"])
emit({"ok": all_ok, "checks": checks})
raise typer.Exit(code=0 if all_ok else 1)
```
The only change is **which probes carry `required=True`**, computed from the mode flags:
- always required: `python`, `registry_db`, `calendar`, `knowledge_base`, `global_halt`
- additionally under `--paper`: `paper_credentials`
- additionally under `--live`: `live_authorizations` (whose body also only runs under `--live`)
- `doctor --paper --live`: union of the two.

A **clean env, no flag** still exits 0 (`global_halt` passes with no row; `kill_switches` passes;
`live_authorizations` is a `skipped:true` skip; `paper_credentials` fails **advisory**) — preserving
`test_doctor_passes_in_clean_env` and `test_doctor_advisory_rows_do_not_gate_exit`.

### Import hygiene
Every new probe imports its heavier dependencies (`global_halt`, `kill_switch`, `live_gate`,
`SqliteStrategyRepository`) **inside the probe function**, matching the existing lazy-import style.
Module-import of `app.py` (hence `algua version`) pulls in nothing new. Crucially,
`live_authorizations`'s `verify_live_authorization` import **and its call** are reached only inside the
`--live` branch at the call site, so plain `doctor` never even imports the authorization/strategy-loader
path.

## Design forks (resolved)
- **What does plain `doctor` mean — trading readiness or generic local health?** → **Trading
  readiness** (per AGENTS.md). Hence `global_halt` gates it in all modes; lane deps gate only the
  named lane.
- **Command-scoped flags vs. auto-scoping from registry state?** → **Mode flags.** Auto-scoping
  surprises a benched-to-research operator. `live_authorizations` still auto-scopes *within* `--live`
  (only fails when a LIVE strategy fails to verify).
- **`live_authorizations` row `ok` rule — `≥1-authorized` or `all-authorized`?** → **`ok = no-LIVE OR
  ALL-authorized`**. The `≥1-authorized` rule was rejected: it left `doctor --live` green with a known
  revoked/unverifiable LIVE strategy guaranteed to be skipped-at-submit — the exact false-green #459
  targets.
- **Does `live_authorizations` run in plain `doctor`?** → **No.** The probe *body*, not just its
  `required` flag, is gated behind `--live`; otherwise `verify_live_authorization → load_strategy`
  would import strategy modules in the advertised "local, fast, side-effect-free" default mode.
- **Kill-switch: gate it, or only surface it?** → **Surface only.** A tripped switch benches one
  strategy, not the account; gating a cycle on per-strategy tradability is #400's rollup.
- **Skip signal: `detail` string-prefix or a structured field?** → **Structured boolean `skipped`**
  on every row. A consumer reading only `ok` can no longer confuse an un-probed dependency with a
  passing one; the human `"skipped: …"` detail stays for readability but is not the machine contract.
- **`doctor --live` when zero LIVE strategies?** → **Ready (`ok:true`), not a tradability gate.**
  `--live` readiness is pinned (flag help + exit-code contract) as lane-authorization readiness only;
  `live_strategies=N` is informational; `N=0` is ready. A gating `tradable_live_strategies` signal was
  rejected as #400's job.

## Deferred / follow-up (NOT in this PR — tracked in #489)
The following were explored in earlier revisions of this design but are **out of scope** for this
slice and are **not** implemented by this diff. They are recorded here so the spec never claims a
capability the code lacks:
- **Broker-reachability probes** (`alpaca_paper_reachable` / `alpaca_live_reachable`): a bounded,
  redaction-safe single-shot `GET /v2/account` (timeout=5, `allow_redirects=False`, strict
  finite/non-negative numeric validation of `equity`/`cash`/`buying_power`, fixed error classes that
  never echo the body/URL), promoted to required under the matching lane flag; skip row otherwise.
- **`requests` as a declared direct dependency** in `pyproject.toml` (needed only by the reachability
  probes above).
- **`alpaca_live_credentials` probe** (presence of `alpaca_live_api_key`/`alpaca_live_api_secret`,
  required under `--live`). The settings fields already exist; only the probe is deferred.
- **`kill_switches` operational-stage rollup** — cross-referencing tripped names against
  `algua.execution.fleet_health.OPERATIONAL_STAGES` (`{LIVE, PAPER, FORWARD_TESTED}`) to report a
  "how many affect an operational-stage strategy" count and point at `fleet health`. Shipped
  `kill_switches` lists the tripped names only.

## CODEOWNERS impact
This PR modifies `algua/cli/app.py` (probes + `--paper`/`--live` wiring + additive per-row `skipped`
field + `_skip_row` helper) and `algua/risk/kill_switch.py` (additive `list_tripped`). It
**imports/calls but does not modify** `live_gate.py`, `store.py`, and `approvals.py` (reuses the
existing public `verify_live_authorization`/`list_strategies` unchanged). None of these modified paths
is in the repo's `.github/CODEOWNERS`, so the PR is expected not to require code-owner review — to be
confirmed against the final diff.

## Non-goals
- No new schema / `SCHEMA_VERSION` bump — all state read via existing tables + helpers. The additive
  per-row `skipped` boolean is a field of the `doctor` diagnostic payload, not a persisted ledger row.
- No rename of the merged `paper_credentials` / `bars_snapshot` / `generated_provenance` probes.
- No change to `live run-all`'s own success semantics — `--live` readiness is a property of `doctor`,
  not a new runner gate. `doctor --live` is intentionally **stricter** than the runner (which skips a
  revoked strategy and trades the rest).
- No per-strategy fleet-status rollup — that is #400. `doctor --live` deliberately does **not** check
  per-strategy allocation, kill-switch, or LIVE-candidate presence.
- No broker-reachability / `requests` dependency / live-credential probe / kill-switch operational
  rollup in THIS PR — deferred, see above.

## Test plan (FAST: `tests/test_cli_core.py tests/test_kill_switch.py` + whole-tree ruff/mypy/lint-imports)
- `global_halt` engaged → required row red, `ok:false`, **exit 1** (all modes); cleared → exit 0.
- `kill_switches`: one tripped strategy → `ok:true` advisory row naming it, exit 0.
- Structured `skipped`: plain-`doctor` `live_authorizations` row is `ok:true, skipped:true` while
  `global_halt` is `ok:true, skipped:false` — a consumer distinguishes "passed" from "not probed" on
  the boolean alone.
- `live_authorizations` outside `--live`: skip row, and **`verify_live_authorization` is never
  called** (monkeypatch spy — the HIGH-2 regression guard: no strategy-module import).
- `live_authorizations` under `--live`: zero LIVE → exit 0, detail contains `live_strategies=0`; an
  un-re-verifiable LIVE strategy → `ok:false`, required, **exit 1** (strict all-must-verify rule).
- `--live` help text disclaims tradability (mentions authorization + fleet).
- `--paper` promotes `paper_credentials` to required → absent creds gate, exit 1.
- Regression: `test_doctor_passes_in_clean_env`, `test_doctor_advisory_rows_do_not_gate_exit` pass
  unchanged; no merged probe renamed.

## Quality gate (FULL, at integration/finish)
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
