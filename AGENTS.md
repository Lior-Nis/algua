# AGENTS.md — Review & Fix Guide for `algua`

You are reviewing **algua**, the system being built toward a mostly autonomous quantitative
trading company. **Your mission: review the system for real problems — correctness, safety,
data-integrity, design, test gaps — and fix the ones that are in scope, while respecting the
invariants and boundaries below.** When a problem touches a safety invariant or
not-yet-built scope, *flag it for the human* rather than fixing it silently.

This file is your entry point. Read the referenced docs before changing code.

---

## 1. How to run things

Toolchain: **Python 3.12 + uv**. From the repo root:

```bash
uv sync                  # install deps
uv run pytest -q         # tests
uv run ruff check .      # lint  (must stay clean)
uv run mypy algua        # types (must stay clean)
uv run lint-imports      # architectural import boundaries (must stay "0 broken")
uv run algua doctor      # environment readiness self-check (JSON)
```

**The full gate must stay green after every change:** `pytest`, `ruff`, `mypy`, `lint-imports`.
Do not weaken a contract, delete a test, or `# type: ignore` your way to green — fix the root cause.

---

## 2. Architecture map — read these first

**Design intent (read before touching code):**
- `docs/PRD.md` — **product vision of record**: objectives, constraints, capital policy and development sequence. Read before prioritizing or proposing work.
- `docs/architecture.md` — current package responsibilities and extension seams. Read before changing implementation.
- `docs/vision-reconciliation.md` — current implementation gaps, historical-document status and issue migration. Read when translating the vision into work or interpreting an older plan.
- `docs/superpowers/specs/2026-05-29-algua-platform-architecture-design.md` — historical architecture rationale; its original lifecycle, authentication mechanism and roadmap are not current operating instructions.
- `docs/superpowers/plans/2026-05-29-foundation-command-surface.md` — historical plan for the completed foundation, not the present scope limit.
- `CLAUDE.md` — the agent operating contract (command surface, golden rules, live-gate summary).
- `docs/agent/operating.md` — the *why* behind the rules (live-gate rationale, module boundaries, JSON convention).
- `docs/contracts/bar-schema.md` — **FROZEN** data contract for `DataProvider.get_bars`. The data
  lane's `get_bars` output MUST conform to this exact shape; it is the integration seam with the
  research lane. Do not change it (or `contracts/types.py::DataProvider`) without cross-lane agreement.
- `README.md`, `.env.example` — quickstart and configuration interface.

**Operating the research loop (vs. reviewing).** This guide is for *reviewing/fixing* algua. If you
were instead launched to *operate* the research loop autonomously (ideate → author → backtest →
gate → candidate), your playbooks are the skills under `.opencode/skills/` — start with
`operating-algua`, then `run-the-research-loop`, and delegate to the `.opencode/agents/` subagents
(`author`, `interpret`). The same golden rules apply: drive everything through `uv run algua ...`,
never go past `candidate`, and never edit the CODEOWNERS-protected integrity files.

That ceiling applies to the isolated research worker. Operational commands may advance through
`forward_tested` under the existing gates, as described in `CLAUDE.md`; live activation requires
the authenticated human ceremony. The vision's future merge/deployment autonomy does not expand
today's allowlists or approvals. Document conflicts as implementation gaps, never permission
to bypass a control.

**Agent runtime.** The autonomous loops (research, leap, forage) run on **OpenCode**, invoked
through the single seam `.opencode/scripts/run_agent.sh`. That script is the only place in the repo
that names a runtime, a model or a sandbox flag — model choices live in `.opencode/opencode.json`, agent
definitions in `.opencode/agents/`. If you are changing how an agent is launched, change the seam,
not a driver. Work on a branch, not directly on `main`.

**Foundation modules (part of the implemented system):**
- `algua/contracts/lifecycle.py` — `Stage`/`Actor` enums + `ALLOWED_TRANSITIONS` state machine + `validate_transition`. **Pure** (stdlib only).
- `algua/contracts/types.py` — `ExecutionContract` (encodes the `t→t+1` anti-look-ahead rule), `OrderIntent`, and `Strategy`/`DataProvider`/`Broker` protocols. **Pure** (pandas only under `TYPE_CHECKING`).
- `algua/calendar/market_calendar.py` — NYSE (`XNYS`) session calendar wrapper; `next_session`/`previous_session` are **strictly** after/before the given day.
- `algua/config/settings.py` — pydantic-settings (`ALGUA_` env prefix). `get_settings()` is intentionally uncached (test isolation).
- `algua/registry/db/` — SQLite connection (WAL, `foreign_keys=ON`) + idempotent `migrate()` (schema versioned via `PRAGMA user_version`). Tables: `strategies`, `stage_transitions`, `approvals`.
- `algua/registry/store/` — typed registry API: `add_strategy`, `get_strategy`, `list_strategies`, `list_transitions`, `transition` (**contains the live gate**).
- `algua/registry/approvals.py` — `record_approval` (mints a human approval) + `has_valid_approval` (verifier).
- `algua/cli/app.py` — Typer app + `emit()` (JSON), `version`, `doctor`.
- `algua/cli/registry_cmd.py` — `registry` subcommands (`add`/`list`/`show`/`transition`/`approve`) + `_json_errors` decorator.
- `algua/cli/main.py` — entry point (`algua = "algua.cli.main:app"`).

Data/snapshot storage, research/backtesting, tracking/knowledge, portfolio/risk, execution,
paper/live ticks, audit/observability and autonomous operator machinery also exist. Use
`docs/architecture.md` for their package map and `tests/` for current coverage. A package's
existence does not establish that a future capability or unattended live acceptance target
has been delivered.

---

## 3. Invariants you MUST NOT weaken

Treat these as hard constraints. If a "fix" requires violating one, **stop and flag it** instead.

1. **The live gate.** Entering `Stage.LIVE` requires ALL of: `actor == Actor.HUMAN`, both
   `code_hash` and `config_hash` provided, and a matching unrevoked row in `approvals`.
   `transition` coerces inputs to enums first so a raw string `"live"` cannot skip the gate.
   The live runner must trust the *approval*, never the bare `stage` flag. Never make
   this easier to bypass. (`algua/registry/store/`, `algua/registry/approvals.py`)
2. **Module purity / boundaries.** `algua/contracts` and `algua/calendar` import no other
   `algua` modules (enforced by `lint-imports`). `contracts`/`features` stay side-effect-free.
   Don't introduce cross-layer imports to make something convenient.
3. **The `t→t+1` rule.** `ExecutionContract.decision_lag_bars >= 1` — features computed on a
   closed bar `t` may not fill on the same bar. Don't relax this; it prevents look-ahead bias.
4. **Reproducibility & point-in-time correctness:**
   backtests must record exactly which data snapshot/config/seed they used.
5. **JSON command contract.** Every CLI command (success *and* error) emits parseable JSON on
   stdout with a meaningful exit code. Don't reintroduce raw tracebacks.

---

## 4. Intentionally deferred — do NOT flag these as defects

Review the implemented system, not only its original foundation. Distinguish defects in shipped
behavior from missing target capabilities in `docs/vision-reconciliation.md`. Do not implement
future capabilities during a review without explicit scope, especially when they affect safety.

- **Target capabilities remain partial:** immutable deployed planners, complete hourly operation,
  comprehensive experiment memory, the complete autonomous repair/release loop and compatible
  external-capital integrations. Their absence is roadmap work, not a foundation defect.
- **`--actor human` below the live wall is now AUTHENTICATED (#329).** Asserting `--actor human`
  on a gated command (`research promote`, `paper promote`) requires an SSH signature (namespace
  `algua-human-actor`, enrolled via `registry enroll-approver --namespace human-actor`) over a
  fresh single-use challenge that binds the command + strategy + recomputed artifact identity + the
  full run context (every gate-relevant input incl. the exact relaxation set). A bare `--actor
  human` no longer unlocks any human-only path — it just prints a challenge and runs nothing. The
  gate reuses the go-live trust anchor (`approvers/allowed_signers`) under a distinct namespace and
  fails closed on a forged/replayed/expired/cross-run signature. So the human-only relaxations
  (declared breadth, holdout reuse, non-PIT, assume-terminal-last-close, NOVEL-family mint, the
  paper thresholds) are a cryptographic boundary against a same-CLI agent that cannot write the
  tree/DB — no longer "discipline." The go-live `registry transition --to live` signed ceremony was
  already authenticated. The RESIDUAL deferred half is deploy-time anchor immutability (below).
- **Live-hardening remains an acceptance obligation:** verify economic-state reconciliation,
  crash/restart recovery, monitoring/alerting, production secrets and runtime isolation against
  the intended deployment. Reconciliation, monitoring, kill switches, Docker research runs and
  systemd operators already have implementations; do not describe them all as absent. Their
  existence does not prove safe unattended real-money operation.

**Accepted residual exposures (threat-model).** These are understood, accepted trade-offs with a
named mitigation — not defects to fix:
- **The audit log is operational, not tamper-evident.** `algua/audit/log.py` records what happened;
  it is not a cryptographic chain and a writer with DB access could rewrite it. Gate enforcement does
  NOT trust audit rows — it lives in *recomputed* identities (code/config/dependency hashes,
  re-verified signatures), so a forged audit row cannot let a strategy past a gate.
- **The strategy runtime sandbox is deferred.** Authored strategy modules run in-process, so a
  malicious module could do more than compute weights. The mitigation is that go-live approval hashes
  the strategy's transitive **first-party** (`algua.*`) import closure into `code_hash`
  (`algua/registry/approvals.py::compute_artifact_hashes`), so a prior approval can satisfy the live
  gate only against the same statically-reachable first-party source a human reviewed (dynamic
  `importlib` string imports and non-source data files are outside the closure). A true execution
  strategy sandbox is future live-hardening work; the OpenCode agent's write sandbox is a
  separate boundary. Artifact-freeze implementation is partial; see the reconciliation record.

**Deployment hardening (enforce when deployment lands).** The trust anchor
`approvers/allowed_signers` is the root of BOTH go-live authority AND (since #329) authenticated
`--actor human`. In any real deployment it MUST NOT be writable by the runtime (agent/operator)
user — only by the human who controls CODEOWNERS — else the runtime could enroll its own key and
self-authorize (this is the explicit RESIDUAL of #329: the gate code reads the anchor from the
running tree, so a tree/DB writer defeats it exactly as it defeats go-live). Docker research and
systemd operator deployment files exist. Verify an immutable installed anchor distinct from the
mutable worktree before real deployment; these files alone do not prove that requirement is met.
Changes to the anchor or its enforcement require human review.

If you believe something deferred is mis-scoped or risky, **flag it with reasoning** — don't build it.

---

## 5. Foundation-era triage — now resolved (do NOT re-report)

The three Minor items flagged in the foundation's final review have all been fixed. They are kept
here only so they are not re-reported as open:
- **CLI DB connections aren't closed.** RESOLVED — `registry_conn()` (`algua/cli/_common.py`) is the
  single connect→migrate→auto-close idiom, and every CLI command opens the registry through it; the
  bare `_conn()` is gone.
- **`transition` signature vs. coercion.** RESOLVED — the transition edge now lives in
  `algua/registry/transitions.py::transition_strategy`, annotated `to: Stage | str, actor: Actor | str`,
  so the signature no longer lies about its coercion.
- **No CI yet.** RESOLVED — `.github/workflows/ci.yml` runs the full gate
  (`pytest + ruff + mypy + lint-imports`) on every push and pull request.

Finding *new* real issues beyond this list is exactly your job.

---

## 6. How to review and fix

1. **Read** the spec + `CLAUDE.md` + `operating.md`, then the module(s) in question.
2. **Classify** each finding: Critical / Important / Minor, and whether it's in-scope (implemented behavior)
   or deferred/safety-invariant (flag-only).
3. **Fix in scope, test-first.** Write or update a failing test that captures the bug, then fix it.
   Match existing style (small focused modules, typed, JSON-emitting CLI, parameterized SQL).
4. **Keep the gate green** (`pytest && ruff check . && mypy algua && lint-imports`) before committing.
5. **Commit granularly** with conventional messages (`fix:`, `chore:`, `test:`, `docs:`), one logical
   change per commit. Work on a branch, not directly on `main`, if opening a PR.
6. **For anything touching §3 invariants or §4 deferred scope: do not change code — write up the
   risk and recommendation for the human to decide.**

---

## 7. Report format

When done, summarize:
- Findings by severity, each with `file:line` and a one-line recommendation.
- What you fixed (with the commit), what you flagged (and why you didn't fix it).
- Confirmation that `pytest`, `ruff`, `mypy`, and `lint-imports` are all green.
