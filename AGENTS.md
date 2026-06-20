# AGENTS.md — Codex Review & Fix Guide for `algua`

You (Codex) are reviewing **algua**, an agent-first algorithmic-trading research & lifecycle
platform. **Your mission: review the system for real problems — correctness, safety,
data-integrity, design, test gaps — and fix the ones that are in scope, while respecting the
invariants and boundaries below.** When a problem touches a safety invariant or
not-yet-built scope, *flag it for the human* rather than fixing it silently.

This file is your entry point. Read the referenced docs before changing code.

---

## 1. How to run things

Toolchain: **Python 3.12 + uv**. From the repo root:

```bash
uv sync                  # install deps
uv run pytest -q         # tests (currently 40 passing)
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
- `docs/superpowers/specs/2026-05-29-algua-platform-architecture-design.md` — the **architecture spec**: thesis, constraints, engine model, lifecycle, the live gate, correctness essentials, the 6-sub-project roadmap, and what is intentionally deferred. **This is the source of truth for design intent.**
- `docs/superpowers/plans/2026-05-29-foundation-command-surface.md` — the implementation plan that built the current foundation (sub-project 1), task by task.
- `CLAUDE.md` — the agent operating contract (command surface, golden rules, live-gate summary).
- `docs/agent/operating.md` — the *why* behind the rules (live-gate rationale, module boundaries, JSON convention).
- `docs/contracts/bar-schema.md` — **FROZEN** data contract for `DataProvider.get_bars`. The data
  lane's `get_bars` output MUST conform to this exact shape; it is the integration seam with the
  research lane. Do not change it (or `contracts/types.py::DataProvider`) without cross-lane agreement.
- `README.md`, `.env.example` — quickstart and configuration interface.

**Operating the research loop (vs. reviewing).** This guide is for *reviewing/fixing* algua. If you
were instead launched to *operate* the research loop autonomously (ideate → author → backtest →
gate → candidate), your playbooks are the skills under `.codex/skills/` — start with
`operating-algua`, then `run-the-research-loop`, and delegate to the `.codex/agents/` subagents
(`author`, `interpret`). The same golden rules apply: drive everything through `uv run algua ...`,
never go past `candidate`, and never edit the CODEOWNERS-protected integrity files.

**Parallel-lane note:** work is currently split across two agents. **Codex owns the data lane**
(`algua/data/*`, `algua/cli/data_cmd.py`): finish the `DataProvider` adapters (Alpaca, yfinance)
and the `get_bars` read API conforming to `docs/contracts/bar-schema.md`. **Claude owns the
research lane** (`algua/strategies|features|backtest|tracking/*`). Neither edits the other's
modules; both meet only at the bar-schema contract. Work on a branch, not directly on `main`.

**Source modules (what exists today — foundation only):**
- `algua/contracts/lifecycle.py` — `Stage`/`Actor` enums + `ALLOWED_TRANSITIONS` state machine + `validate_transition`. **Pure** (stdlib only).
- `algua/contracts/types.py` — `ExecutionContract` (encodes the `t→t+1` anti-look-ahead rule), `OrderIntent`, and `Strategy`/`DataProvider`/`Broker` protocols. **Pure** (pandas only under `TYPE_CHECKING`).
- `algua/calendar/market_calendar.py` — NYSE (`XNYS`) session calendar wrapper; `next_session`/`previous_session` are **strictly** after/before the given day.
- `algua/config/settings.py` — pydantic-settings (`ALGUA_` env prefix). `get_settings()` is intentionally uncached (test isolation).
- `algua/registry/db.py` — SQLite connection (WAL, `foreign_keys=ON`) + idempotent `migrate()` (schema versioned via `PRAGMA user_version`). Tables: `strategies`, `stage_transitions`, `approvals`.
- `algua/registry/store.py` — typed registry API: `add_strategy`, `get_strategy`, `list_strategies`, `list_transitions`, `transition` (**contains the live gate**).
- `algua/registry/approvals.py` — `record_approval` (mints a human approval) + `has_valid_approval` (verifier).
- `algua/cli/app.py` — Typer app + `emit()` (JSON), `version`, `doctor`.
- `algua/cli/registry_cmd.py` — `registry` subcommands (`add`/`list`/`show`/`transition`/`approve`) + `_json_errors` decorator.
- `algua/cli/main.py` — entry point (`algua = "algua.cli.main:app"`).

**Tests** mirror the modules under `tests/` (`test_lifecycle.py`, `test_contracts.py`,
`test_calendar.py`, `test_config.py`, `test_registry_db.py`, `test_registry_store.py`,
`test_registry_approvals.py`, `test_cli_core.py`, `test_cli_registry.py`).

---

## 3. Invariants you MUST NOT weaken

Treat these as hard constraints. If a "fix" requires violating one, **stop and flag it** instead.

1. **The live gate.** Entering `Stage.LIVE` requires ALL of: `actor == Actor.HUMAN`, both
   `code_hash` and `config_hash` provided, and a matching unrevoked row in `approvals`.
   `transition` coerces inputs to enums first so a raw string `"live"` cannot skip the gate.
   The live runner (future) must trust the *approval*, never the bare `stage` flag. Never make
   this easier to bypass. (`algua/registry/store.py`, `algua/registry/approvals.py`)
2. **Module purity / boundaries.** `algua/contracts` and `algua/calendar` import no other
   `algua` modules (enforced by `lint-imports`). `contracts`/`features` stay side-effect-free.
   Don't introduce cross-layer imports to make something convenient.
3. **The `t→t+1` rule.** `ExecutionContract.decision_lag_bars >= 1` — features computed on a
   closed bar `t` may not fill on the same bar. Don't relax this; it prevents look-ahead bias.
4. **Reproducibility & point-in-time correctness** (design-level, becomes code in sub-project 2):
   backtests must record exactly which data snapshot/config/seed they used.
5. **JSON command contract.** Every CLI command (success *and* error) emits parseable JSON on
   stdout with a meaningful exit code. Don't reintroduce raw tracebacks.

---

## 4. Intentionally deferred — do NOT flag these as defects

The current code is **only sub-project 1 (foundation)**. The following are *known, planned
absences*, scoped to later sub-projects in the spec — do not "fix" them or report them as bugs:

- **No data layer, backtest engine, features, execution, or paper/live runner yet** (sub-projects 2–5).
- **Live-gate authenticity is mechanism-only.** Genuine enforcement that `--actor human` is real
  and that `record_approval` is human-only is **deferred to sub-project 6 (live hardening)**.
  Today an operator/agent could pass `--actor human`; that's expected at this stage. You may
  *note* it, but it is not a foundation defect.
- **Deferred to live-hardening:** full economic-state reconciliation, crash/restart playbook,
  real monitoring/alerting, production secrets (keyring/mounted vs `.env`), Docker/cloud deploy,
  kill-switch hardening.

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
  gate only against the exact source a human reviewed; a true execution sandbox is future
  live-hardening work.

**Deployment hardening (enforce when deployment lands).** The signed-live-gate trust anchor
`approvers/allowed_signers` is the root of go-live authority. In any real deployment it MUST NOT be
writable by the runtime (agent/operator) user — only by the human who controls CODEOWNERS — else the
runtime could enroll its own key and self-authorize. No deployment exists yet (see Docker/cloud deploy
above), so this is a requirement to enforce at deploy time, not a code check today.

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
2. **Classify** each finding: Critical / Important / Minor, and whether it's in-scope (foundation)
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
