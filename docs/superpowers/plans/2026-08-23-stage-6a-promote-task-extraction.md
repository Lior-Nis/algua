# Stage 6a — promote_task Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the two shared CLI task bodies (`research_cmd.promote_task`, `backtest_cmd.sweep_task`) and the input resolvers they depend on out of `algua/cli/` into domain modules, so the **two `importlib` escapes in `paper_cmd.py` can become ordinary static imports**. Those escapes exist solely to dodge the `cli command modules are independent of one another` import-linter contract at runtime — a contract the code obeys statically and violates dynamically.

**Architecture:** Two new modules plus one per the spec. `algua/evaluation/inputs.py` takes the four input resolvers (provider / eval / universe / delistings) — they import `data`, `strategies`, `config`, `contracts` and nothing from `cli` or `registry`, so they were always domain code that happened to live in the CLI package. `algua/evaluation/sweep_run.py` takes the sweep body. `algua/registry/promote_run.py` takes the promote body (spec-named). The CLI commands become thin wrappers that keep their local aliases, which is what makes this cheap.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §6 item 4, first bullet.

**Ground truth:** a research pass against `main`@`354644b` that regenerated every count untruncated and read every call site. It found one fact that changes the risk profile of the whole stage, recorded as Decision 1.

### What this stage actually fixes

`paper merge-back` must run the real sweep body and the real promote body. Both live in sibling CLI
modules, and the `independence` contract (`pyproject.toml:298`) forbids `algua.cli.paper_cmd` →
`algua.cli.backtest_cmd` / `algua.cli.research_cmd`. The code's answer today is two runtime escapes:

- `paper_cmd.py:732` — `bt = importlib.import_module("algua.cli.backtest_cmd")`
- `paper_cmd.py:764` — `promote_task = importlib.import_module("algua.cli.research_cmd").promote_task`

Their own comments admit it: *"the cli-independence contract forbids a static paper_cmd->backtest_cmd
sibling edge"* and *"a dynamic import keeps the two command modules structurally independent"*. The
dependency is real; only its visibility is suppressed. `lint-imports` reports 23/23 green while the
edge exists at runtime — so the contract is not enforcing what it claims, and any future reader
trusting it is misled.

After this stage both are static imports of domain modules, the escapes and the `importlib` import
are deleted, and the contract tells the truth.

### Decisions (recorded for a reviewer to check, not re-derive)

1. **The 57 `_select_provider` monkeypatch pins cost nothing — because of how the alias is formed.**
   Six test files patch `algua.cli.<module>._select_provider` (17 in `test_paper_run_all.py`, 16+2 in
   `test_cli_live.py`, 12+2 in `test_cli_paper.py`, 5 in `test_paper_venue_reconcile.py`, 3 in
   `test_observability_wiring.py`). Each CLI module binds the name with
   `from algua.cli._common import select_provider as _select_provider` (`paper_cmd.py:27`). Changing
   only the **import source** leaves `<module>._select_provider` a module attribute, so every patch
   keeps resolving and keeps covering. **Zero test edits for the resolver move.** This is a
   prediction; Task 1 either confirms it or reports the deviation loudly.
2. **`algua/evaluation/` is a new package, not a dumping ground in `registry`.** The four resolvers
   import `data`, `strategies`, `backtest._sample`, `config`, `contracts` — and **no registry**.
   Putting them in `registry/` would manufacture a registry→data edge that does not exist today.
   Legal: no contract forbids a new package; `evaluation` will import `data`/`strategies`/`backtest`
   and (in `sweep_run.py`) `registry` + `tracking`.
3. **`promote_run.py` goes in `registry/` as the spec names it**, because promote *is* a registry
   operation (it drives the `backtested -> candidate` transition and mints a single-use gate token).
   The sweep body does NOT: it is an evaluation that records breadth as a side effect, so it lives in
   `evaluation/sweep_run.py`. **This is a deviation from a literal reading of the spec bullet**, which
   names only `registry/promote_run.py` and says the sweep-task extraction rides along without naming
   its home. Recorded here so a reviewer checks the reasoning rather than the letter.
4. **`cli/_common.py` keeps `registry_conn`, `authenticate_actor`, `sync_kb_doc`, and the emit/ok/
   project helpers.** They are genuinely CLI-shaped (connection lifecycle for a command, actor
   authentication for a command invocation, JSON envelope helpers). Only the input resolvers move.
5. **`select_provider` does NOT become a registry.** Spec §5 item 4 asks for it; it should be
   dropped. `data/providers/_REGISTRY` is the *ingest* seam — name-keyed, third-party pluggable.
   `select_provider` picks a *serving* provider from two mutually-exclusive CLI flags
   (`--demo`/`--snapshot`); there is no name to key on and a table would have two rows and no
   extension point. That is indirection without extensibility — the same judgment the Stage 5c
   calendar review endorsed. Approved by the operator 2026-08-23.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **When running the full suite, pass `timeout: 600000` explicitly to the Bash tool.** Without that parameter the 120s default fires and the harness **auto-backgrounds** the command — this has stalled six agents on this program, every time for the same reason. If a run looks hung, check `pgrep -f pytest` instead of waiting.
- **Regenerate every count untruncated at the moment you assert it** — no `| head`/`| tail` on an enumeration you are counting. Across this program, nine separate figures in plans were wrong; one would have caused a net regression. If a number here disagrees with what you measure, trust your measurement and say so.
- **This is a PURE MOVE. No behaviour changes, no signature changes, no "while I'm here" improvements.** Every moved body keeps its comments verbatim — several encode hard-won invariants (the holdout burn-on-peek / release-on-failure saga, the strict-agent pinning, the `attempt_token` attribution). Losing a comment here loses the reason a wall exists.
- **A moved function keeps its name.** Do not rename `promote_task`/`sweep_task` on the way out; a rename plus a move is two changes and only one of them is reviewable in the diff.
- **CODEOWNERS-protected files this stage touches: `algua/cli/paper_cmd.py`.** On the paper→live wall. Expected; note it in the PR.
- `git add` scoped to named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known hazard: some test writes demo strategy files into `algua/strategies/momentum/`. If `git status` shows an untracked file there, delete it before staging.

---

### Task 1: Move the input resolvers to `algua/evaluation/inputs.py`

**Files:**
- Create: `algua/evaluation/__init__.py`, `algua/evaluation/inputs.py`
- Modify: `algua/cli/_common.py`, plus every module importing the four moved names

**Interfaces:**
- Produces: `select_provider(demo: bool, snapshot: str | None) -> DataProvider`,
  `resolve_eval_inputs(name, demo, snapshot, start, end, *, reload=False) -> tuple[LoadedStrategy, DataProvider, datetime, datetime]`,
  `resolve_delisting_inputs(...)`, `resolve_universe_inputs(...)` — all with **signatures byte-identical to today's**.

- [ ] **Step 1: Read before moving**

Read `algua/cli/_common.py` in full (349 lines, 14 public functions). Note which imports at the top serve ONLY the four moving functions — those travel; the rest stay. Read `pyproject.toml`'s import-linter contracts (from `[[tool.importlinter.contracts]]` onward) so you know what edges are legal for a new `algua.evaluation` package.

- [ ] **Step 2: Enumerate the callers untruncated**

```bash
grep -rn "select_provider\|resolve_eval_inputs\|resolve_universe_inputs\|resolve_delisting_inputs" --include='*.py' algua/ tests/
```

Write the full list into your report. Research measured 8 non-definition references in `algua/` for `select_provider` and 6–7 each for the resolvers, but **regenerate it yourself** — this list is the task.

- [ ] **Step 3: Create the package and move the four functions verbatim**

`algua/evaluation/__init__.py` gets a docstring naming the package's job (shared evaluation task bodies and their input resolution, importable by both `cli` and `registry` without a sibling edge) and nothing else — no re-exports, so import sites stay explicit.

Move the four functions into `algua/evaluation/inputs.py` **unchanged**, with their docstrings and comments intact, plus a module docstring explaining why they left `cli/_common.py`: they import `data`, `strategies`, `backtest._sample`, `config`, `contracts` and nothing from `cli` or `registry`, so they were domain code living in the CLI package, and `registry`/`evaluation` code cannot reach them where they were.

- [ ] **Step 4: Re-point every import, preserving local aliases**

This is the step Decision 1 rests on. In each CLI module, change only the SOURCE:

```python
# before
from algua.cli._common import select_provider as _select_provider
# after
from algua.evaluation.inputs import select_provider as _select_provider
```

**Keep the `as _select_provider` alias exactly.** `<module>._select_provider` must remain a module attribute or 57 monkeypatch pins silently stop covering their call sites.

- [ ] **Step 5: Prove the pins still fire**

Do not reason about this — measure it. Pick one pinned test, break the patch target, and confirm it fails:

```bash
cd /path/to/worktree && PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_paper_run_all.py -q
```
Then temporarily change one `monkeypatch.setattr("algua.cli.paper_cmd._select_provider", ...)` to a
nonexistent attribute and confirm it raises `AttributeError` (proving the target is real), and
separately make the patched lambda raise and confirm the test **fails** (proving it is reached).
Restore both. Clear `__pycache__` or keep `PYTHONDONTWRITEBYTECODE=1` set — a same-second restore can
otherwise reuse mutated bytecode and make the result lie in either direction.

Report the observed counts. If any pin does NOT fire, stop and report it — that is a pre-existing
dead patch and a finding worth more than this task.

- [ ] **Step 6: Full gate, then commit**

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (Bash `timeout: 600000`). Expect **zero test-file edits** in this task's diff. If you had to edit a test, say why in the report — it means Decision 1 was wrong and the reviewer needs to know.

---

### Task 2: Extract the sweep body to `algua/evaluation/sweep_run.py` and kill escape #1

**Files:**
- Create: `algua/evaluation/sweep_run.py`
- Modify: `algua/cli/backtest_cmd.py`, `algua/registry/mergeback_intake.py`, `algua/cli/paper_cmd.py`

**Interfaces:**
- Consumes: `algua.evaluation.inputs` (Task 1).
- Produces: `sweep_task(name, *, start, end, demo, snapshot, universe, windows, holdout_frac, param, rank_by, top, fundamentals_snapshot, news_snapshot, delistings, assume_terminal_last_close, track, reload) -> dict` — **the exact signature it has today** at `backtest_cmd.py:371`.

- [ ] **Step 1: Move the body verbatim**

Move `sweep_task` (research measured ~60 lines at `backtest_cmd.py:371`; re-verify) into `algua/evaluation/sweep_run.py`, keeping its full docstring — it documents that the function opens its own `registry_conn()`, that `--summary` projection stays in the typer wrapper while `top` truncation lives in the body, and what `reload` is for. Those are contracts, not commentary.

`backtest_cmd.py` imports it and its typer command calls it. Do not leave a re-export shim in `backtest_cmd` beyond the import it actually uses.

- [ ] **Step 2: Kill the escape in `mergeback_intake` / `paper_cmd`**

At `paper_cmd.py:732` the escape reads:

```python
intake_mod = importlib.import_module("algua.registry.mergeback_intake")
bt = importlib.import_module("algua.cli.backtest_cmd")
```

The `bt` half exists only to reach the sweep body. Replace it with a static
`from algua.evaluation.sweep_run import sweep_task` and pass it through as before. **Do not change
what is passed or how `produce_evidence` is called** — only where the callable comes from.

Re-read the comment above that escape and rewrite it to describe the new reality: the evidence is
still the REAL sweep body, but it now arrives by a legal static import instead of an `importlib`
dodge. Keep the strict-agent pinning sentence verbatim.

- [ ] **Step 3: Gate and commit**

Full gate with `timeout: 600000`. `lint-imports` must still report all contracts kept — if the new edge trips a contract, STOP and report rather than editing `pyproject.toml`; a contract change is a design decision, not a fix.

---

### Task 3: Extract the promote body to `algua/registry/promote_run.py` and kill escape #2

**Files:**
- Create: `algua/registry/promote_run.py`
- Modify: `algua/cli/research_cmd.py`, `algua/cli/paper_cmd.py`, `tests/test_cli_merge_back.py`

**Interfaces:**
- Consumes: `algua.evaluation.inputs` (Task 1).
- Produces: `promote_task(...) -> int` with **today's exact signature** (`research_cmd.py:208`, ~248 lines, `# noqa: PLR0913, PLR0915`).

- [ ] **Step 1: Move the body verbatim — comments are load-bearing here**

This body carries the holdout **burn-on-peek / release-on-failure** saga, the family-classification
deferral, and the `attempt_token` attribution. Move every comment with the code. If a comment
references a line number or a neighbour that no longer applies after the move, fix the reference —
do not delete the comment.

Keep the `# noqa` codes; the function is long by design and this task is not the place to split it.

- [ ] **Step 2: Kill escape #2**

Replace `paper_cmd.py:764`'s
`promote_task = importlib.import_module("algua.cli.research_cmd").promote_task`
with a static `from algua.registry.promote_run import promote_task`.

Rewrite the surrounding comment: the paragraph explaining that "a dynamic import keeps the two
command modules structurally independent" is now false and must go, but the sentences about
strict-agent inputs, no relaxation flags reaching the seam, and `promote_task` owning its own
`registry_conn` are still true and must stay.

- [ ] **Step 3: Check the `importlib` import — it almost certainly STAYS**

`paper_cmd.py:3` imports `importlib`. Verified against `main`@`354644b`: after Tasks 2 and 3 remove
the two cli→cli escapes (`:732`, `:764`), **four uses remain** — `:614` (`algua.operator.gate_runner`)
and `:679`, `:720`, `:731` (`algua.registry.mergeback_intake`). Those are lazy imports of *non-cli*
modules, not independence-contract escapes, and they stay.

So the expected outcome of this step is **no change**. Re-check untruncated anyway with
`grep -n "importlib" algua/cli/paper_cmd.py` and report the surviving line numbers. Remove the
module import ONLY if your own grep shows zero remaining uses — do not remove it on the strength of
this plan's expectation.

- [ ] **Step 4: Re-point the one test pin — and prove it still fires**

`tests/test_cli_merge_back.py` does `monkeypatch.setattr(research_cmd, "promote_task", _fake_promote)`.
Once `paper_cmd` imports from `algua.registry.promote_run`, patching `research_cmd` **no longer
affects the merge-back path**. If you only re-point it and the suite goes green, that proves nothing.

Verify in both directions with `PYTHONDONTWRITEBYTECODE=1`:
- make `_fake_promote` raise → the merge-back tests using it must **FAIL**
- restore → they must **PASS**

Report the observed pass/fail counts. **This is the failure mode that silently disarmed a go-live
guard in Stage 5b and was caught only by mutation in 5c: a patch that resolves, binds a real
attribute, and covers nothing.** Do not report this step as "confirmed" without the numbers.

- [ ] **Step 5: Gate and commit** — full gate, `timeout: 600000`.

---

### Task 4: Close-out verification

**Files:** none expected (verification only; fix anything found).

- [ ] **Step 1: Both escapes are gone**

```bash
grep -rn "importlib" algua/cli/paper_cmd.py
grep -rn "import_module" --include='*.py' algua/cli/
```
Expect no `import_module("algua.cli.…")` anywhere — a dynamic import of a *sibling CLI module* is the
thing this stage removes. Remaining dynamic imports of non-cli modules are fine and should be listed
in the report so the reviewer sees they were considered, not missed.

- [ ] **Step 2: The contract now tells the truth**

`uv run lint-imports` — all contracts kept. Then confirm the `independence` contract is actually
load-bearing by TEMPORARILY adding `import algua.cli.research_cmd` to `algua/cli/paper_cmd.py` and
checking `lint-imports` **fails**. Remove it. A contract that passes whether or not the edge exists
would be worthless, and this stage's whole claim is that the contract now means something.

- [ ] **Step 3: Full gate + CLI smoke** (`timeout: 600000`)

`uv run pytest -q`, ruff, mypy, lint-imports, then:
```bash
uv run algua doctor
uv run algua backtest sweep cross_sectional_momentum --demo --param fast=5,10 --summary
```
(`cross_sectional_momentum` is a real strategy; `momentum` is a category directory and will fail.)

- [ ] **Step 4: Commit any fixes.** If nothing needed fixing, make no commit — expected, and consistent with every prior close-out in this program.
