# Stage 6d — operator_cmd Extractions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close §6 item 4 — the last CLI→domain extraction. Move the driver-payload parsing helpers into `algua/operator/driver_payload.py` and the session decision tree into `algua/operator/session_runner.py`, with the emit surface injected so `algua/operator/` stays free of `algua.cli`.

**Architecture:** Two new modules in the existing `algua/operator/` package. The payload helpers are pure text/JSON functions and move unchanged. `_run_session` is the harder one: it is not a pure decision tree — it performs **10 `emit()` calls, 8 `emit_alert()` calls and 5 `raise typer.Exit`**. The spec's parenthetical *"(emit callback injected)"* is therefore the whole design, not a detail: the decision logic moves, the output mechanism is passed in, and the CLI keeps owning `typer`.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §6 item 4, final bullet (line 209).

**Ground truth:** a research pass against `main`@`1a7df18`. Every count below was measured.

### Decisions (recorded for a reviewer to check, not re-derive)

1. **`algua/operator/` is cli-free TODAY, and Task 1 locks that in BEFORE any code moves.** Probed
   with a real `forbidden: algua.operator -> algua.cli` contract on `main`@`1a7df18`: **KEPT**
   (27 contracts, 0 broken with the probe added). That property is exactly what stage 6a's
   `human_actor.py` lost — it now holds the program's only `registry -> cli` edge (#592), created by
   a move that seemed harmless. **Adding the contract first turns "don't do that" into "cannot do
   that"**, so a lazy `from algua.cli.app import emit` inside the moved body fails the gate instead
   of shipping.
2. **The emit surface is injected as parameters, not imported.** `_run_session` calls `emit` 10×,
   `emit_alert` 8× and `raise typer.Exit` 5×. `emit_alert` already lives in `algua/operator/alerts.py`
   (no problem). `emit` is `algua.cli.app.emit` and **must not** be imported by the moved code. Inject
   it as a callable parameter.
3. **`typer.Exit` does NOT move.** Raising a CLI framework's control-flow exception from a domain
   module is the same mistake as importing its printer — and it is what makes `human_actor.py`
   un-contractable today. The moved function **returns** its outcome; the CLI wrapper decides whether
   that means `raise typer.Exit()`. This is a deliberate deviation from a naive "pure move" and
   **must be called out in the PR**: it is the one place in this stage where behaviour-preserving
   requires a signature change rather than a copy.
4. **Only the decision tree moves — the lock, the marker and the driver invocation stay wired in the
   CLI.** `_run_session` runs *inside a held run lock* (`lock_run`), and the lock lifecycle is a CLI
   concern. Moving the tree must not move lock acquisition, or a future caller could run the tree
   unlocked.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. Baseline is **26 contracts, 0 broken** (27 after Task 1).
- **The full suite takes ~7–10 minutes on this machine.** Pass `timeout: 900000` explicitly to the Bash tool. If the harness backgrounds the run anyway, **do NOT end your turn waiting** — read the output file the harness names, or re-run in the foreground. **Nine agents on this program have stalled exactly this way.** To check whether a real pytest is live, match the **executable** (`readlink /proc/PID/exe`), not `pgrep -f pytest` — the cmdline match is polluted by leaked test processes whose data-dir path contains "pytest" (#588).
- **A "pure move" NEVER licenses duplication to satisfy it.** Six instances across 6a/6b, all undone. If a moved body needs a helper, **the helper moves to its owning layer or is imported** — never copied, never inlined to dodge an import. **The plan's helper list has been incomplete on every stage of this program so far** (6c alone: a third travelling helper in Task 1, a wrong module guess in Task 2, a cross-half type alias in Task 3 — each found by an implementer measuring instead of trusting). Regenerate every helper's callers untruncated and trust your measurement over this plan.
- **Every scripted edit must assert it changed something.**
- **Do not over-claim.** Report what you measured, not what the task hoped for. Reporting "this grep still returns one hit" is the right answer when it does.
- Moved bodies keep names (underscore dropped only where a task says), signatures, `# noqa` codes, docstrings and comments **verbatim** — except the deliberate `typer.Exit` change in Decision 3.
- **CODEOWNERS:** if a moved body was protected at its old location, its new home must be protected too. Stage 6c found this program had silently un-protected three integrity-critical modules — one for four stages — because nothing fails when a rule stops matching. `tests/test_repo_hygiene.py::test_integrity_critical_modules_are_codeowner_protected` now guards a named set; **if you move something into or out of that set, update it.** `operator_cmd.py` is NOT currently CODEOWNERS-protected, so this stage is expected to need no change — verify rather than assume.
- `git add` scoped to named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known hazard: some test writes demo strategy files into `algua/strategies/momentum/`. Check `git status --porcelain` before staging and delete any stray untracked file there.

---

### Task 1: Contract the boundary BEFORE moving anything

**Files:** Modify `pyproject.toml`

This task exists because of what stage 6a did to `human_actor.py`: a move put a domain module one lazy import away from the CLI, nobody noticed, and the repo now carries a permanent `registry -> cli` edge (#592) that cannot be contracted without an exemption. `algua/operator/` is clean today. Lock it first, then move code into a boundary that is already enforced.

- [ ] **Step 1: Verify the property still holds**

```bash
grep -rn "algua\.cli" --include='*.py' algua/operator/
```
Expect **nothing**. If anything appears, this plan's premise is wrong — stop and report.

- [ ] **Step 2: Add the contract**

```toml
[[tool.importlinter.contracts]]
name = "operator lane stays off the cli layer"
type = "forbidden"
source_modules = ["algua.operator"]
forbidden_modules = ["algua.cli"]
```
Give it a comment explaining WHY it exists now: the operator lane is about to receive the session decision tree, whose original called `algua.cli.app.emit` 10 times; the contract is what forces that surface to be injected instead of imported.

- [ ] **Step 3: Prove it is load-bearing, not decorative**

Temporarily add `from algua.cli.app import emit` to any module in `algua/operator/`, run `lint-imports`, confirm it **FAILS** naming that edge, then remove it. A contract that passes whether or not the edge exists is worthless — this program has shipped one such contract before (stage 6a found `lint-imports` reporting green while an `importlib` escape carried the edge at runtime).

- [ ] **Step 4: Gate and commit** (a contract-only change; the full suite must still be green).

---

### Task 2: Move the driver-payload helpers to `algua/operator/driver_payload.py`

**Files:**
- Create: `algua/operator/driver_payload.py`
- Modify: `algua/cli/operator_cmd.py`

**Interfaces:**
- Produces: `last_top_level_object(text: str) -> str | None`, `parse_driver_payload(stdout: str) -> dict | None`, `classify_failure(payload: dict | None) -> str` — today's exact signatures, underscore dropped.

- [ ] **Step 1: Read and measure**

Measured on `main`@`1a7df18`: `_last_top_level_object` (`:111`, ~36 lines), `_parse_driver_payload` (`:147`, ~24 lines), `_classify_failure` (`:171`, ~11 lines). **Re-verify these line numbers and check each helper's full caller list untruncated** — including whether `_resolve_driver_argv` / `_run_driver` / `_resolve_git_dir` belong with them (this plan says they do NOT: they are process/git concerns, not payload parsing. Disagree with evidence if you find otherwise, and say so).

- [ ] **Step 2: Move the three bodies verbatim.** These parse an untrusted subprocess's stdout — the brace-matching in `_last_top_level_object` is deliberately defensive. Comments verbatim.

- [ ] **Step 3: Re-point callers; keep any monkeypatch pin alive**

Enumerate pins untruncated: `grep -rn "monkeypatch.setattr" tests/ | grep -E "_parse_driver_payload|_classify_failure|_last_top_level_object"`. If any exists, keep the name bound as a module attribute in `operator_cmd` (the alias-preserving trick) and **prove the pin still fires in both directions** with `PYTHONDONTWRITEBYTECODE=1` and `__pycache__` cleared. Report counts.

- [ ] **Step 4: Gate and commit.**

---

### Task 3: Move the session decision tree to `algua/operator/session_runner.py`

**Files:**
- Create: `algua/operator/session_runner.py`
- Modify: `algua/cli/operator_cmd.py`

**Interfaces:**
- Produces: `run_session(job, op_job, command, now_dt, alert_cmd, host, pid, *, emit, run_driver) -> None` — the decision tree with its output surface injected. Exact parameter names are yours to choose; **document them in your report** so the reviewer can check them against the call site.

- [ ] **Step 1: Map every side effect before moving a line**

```bash
grep -c "emit(" algua/cli/operator_cmd.py   # then scope to the function body
```
Measured inside `_run_session` (`:403`, ~209 lines): **10 `emit(`, 8 `emit_alert(`, 5 `raise typer.`**. Re-verify. List every one in your report with its line, because each is a decision the injection has to preserve.

- [ ] **Step 2: Inject `emit`; leave `emit_alert` alone**

`emit_alert` is already `algua/operator/alerts.py` — a domain module — so it travels as an ordinary import. `emit` is `algua.cli.app.emit` and Task 1's contract now makes importing it a **gate failure**. Pass it in as a callable parameter.

- [ ] **Step 3: `typer.Exit` stays in the CLI (Decision 3)**

The five `raise typer.Exit` calls are CLI control flow. The moved function must not raise them. Give `run_session` a return value that tells the wrapper what to do, and let the thin `operator_cmd` wrapper translate that into `raise typer.Exit()`.

**Preserve exit codes exactly.** Measured on `main`@`1a7df18`: all five are `raise typer.Exit(1)` —
at `:430`, `:444`, `:490` (`from None`), `:515` (`from None`), `:611` — so a single "failed" outcome
suffices and no per-branch code needs distinguishing. **Re-verify that yourself**; if any call has
gained a different code, the return value must distinguish it. An operator's systemd unit and the
`OnFailure=` watchdog key off these, so a flattened exit code is a real behaviour change, not a
refactor.

Note two of the five use `from None` — that suppresses exception chaining on a path that already
emitted its own JSON envelope. Whatever the wrapper does must keep the traceback suppressed; a bare
`raise typer.Exit(1)` inside an `except` block would start leaking the original exception's context
into operator output.

- [ ] **Step 4: The lock stays put (Decision 4)**

`_run_session` runs inside a held run lock. Move the tree, NOT the lock acquisition. Verify by reading `lock_run` that the lock still wraps the call after your change.

- [ ] **Step 5: Prove the exit-code behaviour is unchanged**

Find the tests covering `operator run` failure paths (`grep -rn "operator" tests/ | grep -i "exit\|fail"` — regenerate untruncated). Run them before and after. If no test asserts a non-zero exit code from this path, **say so in your report** — that is a coverage gap worth knowing about on a watchdog-facing command, and worth adding a test for.

- [ ] **Step 6: Gate and commit.**

---

### Task 4: Close-out verification

**Files:** none expected (verification only; fix anything found).

- [ ] **Step 1: Counts.** Report `operator_cmd.py` before/after (**611** on `main`@`1a7df18`) and the two new modules' sizes.

- [ ] **Step 2: The contract holds and the operator lane is still cli-free**

```bash
grep -rn "algua\.cli" --include='*.py' algua/operator/
uv run lint-imports
```
First returns nothing; second reports **27 kept, 0 broken**.

- [ ] **Step 3: No duplication**

```bash
grep -rn "local equivalent\|private copy\|byte-identical duplicate\|reproduced here" --include='*.py' algua/
```
Plus: confirm every moved name has exactly ONE definition tree-wide.

- [ ] **Step 4: CODEOWNERS + the hygiene guard**

`uv run pytest tests/test_repo_hygiene.py -q` must pass. Confirm nothing moved out from under a protection rule (stage 6c's finding — this is now a standing check, not a one-off).

- [ ] **Step 5: Full gate + CLI smoke** (`timeout: 900000`): `uv run pytest -q`, ruff, mypy, lint-imports, then `uv run algua doctor` and `uv run algua operator run --help` (the command whose body moved).

- [ ] **Step 6: Commit any fixes.** If nothing needed fixing, make no commit.
