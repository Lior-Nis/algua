# Stage 7 — engine.py + gates.py Cuts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carve the two remaining god-modules. `backtest/engine.py` (837 lines) sheds its PIT-masking and dual-path/parity machinery so `simulate`+`run` stand alone as an orchestrator. `research/gates.py` (668 lines) sheds `GateDecision.to_dict` and — the bigger win — **deletes a 34-name re-export shim** left behind by #335.

**Architecture:** Four new modules, no new packages. `backtest/pit_view.py` (as-of masking + the static operating view), `backtest/decision_path.py` (canonical/fast dual path + the parity guard), `research/gate_serialization.py` (`to_dict`). `holdout_window` moves to `backtest/walkforward.py`, the windowing home the spec names. Then the `gates.py` re-export surface is removed and every call site re-pointed at the real owner.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §6 items 6 and 7.

**Ground truth:** a research pass against `main`@`cf2a20c`. Every size and count below was measured.

### Decisions (recorded for a reviewer to check, not re-derive)

1. **`backtest/execution_model.py` is DROPPED from the spec — there is nothing to put in it.** The
   spec names it for "costs/fill-price". Measured: there is no costs body. `fees` and `slippage` are
   two kwargs on the `vbt.Portfolio.from_orders(...)` call inside `simulate` (`engine.py:748-749`),
   and `fill_price` is a one-line grid selection (`:716`). The substantial thing at that site is a
   **comment** explaining why charging costs on the already-`t->t+1`-shifted execution weights
   introduces no look-ahead (#325) — and that comment has to stay attached to the call it explains.
   Extracting a module here would produce a file containing two parameter names plus a comment
   severed from its subject. **Say this in the PR and annotate the spec**, the way stage 6b's
   deviations were annotated.
2. **The `gates.py` re-export shim is the highest-value cut in this stage.** Measured: `__all__` has
   **43 names, of which 34 are pure re-exports** of things that now live in `research/dsr.py`,
   `research/regime.py`, `research/haircut.py`, `research/_constants.py` and `backtest/walkforward.py`.
   That is a compat shim, and the spec's own words are *"the post-#335 re-export shim in `gates.py`
   is removed once call sites are updated (no compat cruft)"*. Removing it is what makes `gates.py`
   mean "the gate", not "the place everything about gates used to live".
3. **`holdout_window` goes to `walkforward.py`, and this one has real blast radius.** It is imported
   by `registry/promote_run.py:20` (the `backtested -> candidate` gate body) and pinned across
   `tests/test_holdout_window.py`. It is also the **#192 single-use-holdout identity source** — the
   comments at `engine.py:502/517/550` explain that `adj` (the adj_close grid) stays the date-index
   truth for holdout identity even when the fill grid changes. Those comments live in
   `adj_open_grid`/`_static_operating_view`, which are NOT moving, so **the references must be fixed
   rather than left pointing at a function that has left the module**.
4. **`simulate`+`run` stay put and stay together.** Measured ~237 lines (spec predicted ~200). Do not
   chase the number by splitting the orchestrator further — the spec's point is that what remains is
   *an orchestrator*, not that it hits a line count.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. Baseline is **27 contracts, 0 broken**; the suite is **3554 passing**.
- **The full suite takes ~7–10 minutes on this machine.** Pass `timeout: 900000` explicitly to the Bash tool. If the harness backgrounds the run anyway, **do NOT end your turn waiting** — read the output file the harness names, or re-run in the foreground. **Ten agents on this program have stalled exactly this way.** To check whether a real pytest is live, match the **executable** (`readlink /proc/PID/exe`), not `pgrep -f pytest` (#588).
- **`algua/backtest/engine.py` and `algua/research/gates.py` are BOTH CODEOWNERS-protected** — engine.py for anti-look-ahead `t->t+1` enforcement, gates.py for promotion-gate criteria. **If a moved body was protected at its old location, its new home must be protected too.** Stage 6c found this program had silently un-protected three integrity-critical modules, one for four stages, because nothing fails when a rule stops matching. `tests/test_repo_hygiene.py::test_integrity_critical_modules_are_codeowner_protected` guards a named set — **update it when you add a module**, and it will fail loudly if you forget.
- **A "pure move" NEVER licenses duplication to satisfy it.** Six instances across 6a/6b, all undone. Never copy a helper, never inline a one-liner to dodge an import.
- **This plan's dependency lists have been incomplete on EVERY stage of this program** (6c: a third travelling helper, a wrong module guess, a cross-half type alias; 6d: a third injected callable). **Regenerate every caller list untruncated and trust your measurement over this plan.** Say so when they differ.
- **Every scripted edit must assert it changed something.**
- **Do not over-claim.** Report what you measured.
- Moved bodies keep names, signatures, `# noqa` codes, docstrings and comments **verbatim**.
- `git add` scoped to named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known hazard: some test writes demo strategy files into `algua/strategies/momentum/`. Check `git status --porcelain` before staging and delete any stray untracked file there.

---

### Task 1: Carve the PIT-masking view into `backtest/pit_view.py`

**Files:**
- Create: `algua/backtest/pit_view.py`
- Modify: `algua/backtest/engine.py` (CODEOWNERS-protected), `CODEOWNERS`, `tests/test_repo_hygiene.py`

**Interfaces:**
- Produces: `members_as_of`, `_assert_fundamentals_shape`, `_fundamentals_as_of`, `_assert_news_shape`, `_news_as_of`, `_static_operating_view` — today's exact signatures. Keep the underscores on the private ones.

- [ ] **Step 1: Measure before moving**

Measured on `main`@`cf2a20c`: `members_as_of` (`:47`, 18), `_assert_fundamentals_shape` (`:65`, 21), `_fundamentals_as_of` (`:86`, 16), `_assert_news_shape` (`:102`, 22), `_news_as_of` (`:124`, 16), `_static_operating_view` (`:541`, 25) — **~118 lines**. Re-verify, and regenerate every caller list untruncated.

**Known cross-task hazard:** `members_as_of` is imported by `backtest/walkforward.py:18` and called
at `:154` — and Task 3 also edits `walkforward.py`. Re-point that import in THIS task; Task 3 must
not assume it still reads from `engine`.

- [ ] **Step 2: Move verbatim**

These functions implement the anti-look-ahead PIT contract — an as-of mask is the difference between a legitimate backtest and one that peeks. Comments verbatim.

- [ ] **Step 3: CODEOWNERS follows the code**

`engine.py` is protected for anti-look-ahead enforcement, and the as-of masking IS that enforcement. Add `/algua/backtest/pit_view.py` to CODEOWNERS **and** to `INTEGRITY_CRITICAL_MODULES` in `tests/test_repo_hygiene.py`. Verify with `uv run pytest tests/test_repo_hygiene.py -q`.

- [ ] **Step 4: Gate and commit.**

---

### Task 2: Carve the dual path + parity guard into `backtest/decision_path.py`

**Files:**
- Create: `algua/backtest/decision_path.py`
- Modify: `algua/backtest/engine.py`, `CODEOWNERS`, `tests/test_repo_hygiene.py`

**Interfaces:**
- Produces: `_decision_weights`, `_canonical_row`, `_fast_weights`, `_decision_weights_fast`, `_parity_sample_positions`, `_assert_parity`, `_decision_weights_fast_or_loop`, `verify_signal_panel_parity` — today's exact signatures.

- [ ] **Step 1: Measure**

Measured: `_decision_weights` (`:140`, 90), `_canonical_row` (`:230`, 24), `_fast_weights` (`:254`, 55), `_decision_weights_fast` (`:309`, 14), `_parity_sample_positions` (`:323`, 15), `_assert_parity` (`:338`, 40), `_decision_weights_fast_or_loop` (`:378`, 34), `verify_signal_panel_parity` (`:412`, 67) — **~339 lines**, the largest single cut in this stage. Re-verify and regenerate caller lists untruncated.

**`verify_signal_panel_parity` carries a PATCH TARGET — the shape that silently disarms coverage.**
Measured consumers: `registry/promotion.py:9` (import) + `:162` (the promotion gate's exhaustive
parity check), `tests/test_fast_path.py` (6 call sites), `tests/test_static_observation_parity.py`
(2), and critically **`tests/registry/test_family_creation_guard.py:94`**, which does
`patch("algua.registry.promotion.verify_signal_panel_parity")`.

That patch targets the name as bound in `promotion.py`, so it survives IF `promotion.py` still binds
the name and only the import SOURCE changes. **Prove it, do not assume it:** point the patch at a
nonexistent attribute, confirm the tests error, restore, confirm they pass — with
`PYTHONDONTWRITEBYTECODE=1` and `__pycache__` cleared. Report counts. This is exactly how a go-live
guard was silently disarmed in stage 5b.

- [ ] **Step 2: Move verbatim**

This is the fast-vs-canonical equivalence guard: the fast vectorized path is only sound because the parity check proves it agrees with the loop. **The sampling bound and the assertion message are the safety property** — a weakened parity guard silently licenses a wrong fast path. Comments verbatim.

- [ ] **Step 3: CODEOWNERS follows the code** — same as Task 1 Step 3, for `decision_path.py`.

- [ ] **Step 4: Gate and commit.**

---

### Task 3: Move `holdout_window` to `backtest/walkforward.py`

**Files:**
- Modify: `algua/backtest/walkforward.py`, `algua/backtest/engine.py`, `algua/registry/promote_run.py`, `tests/test_holdout_window.py`

- [ ] **Step 1: Read the #192 identity comments FIRST**

`engine.py:502`, `:517`, `:550` reference `holdout_window` from inside `adj_grid` / `adj_open_grid` / `_static_operating_view` — functions that are **not** moving (`_static_operating_view` moved in Task 1; the grids stay). They explain that `adj` remains the date-index source of truth for the #192 single-use holdout identity even when the fill grid changes. After the move those references must name the new location. **Fix the references; do not delete the comments** — they are why the holdout identity is stable.

- [ ] **Step 2: Move `holdout_window` verbatim** (`:566`, ~35 lines) into `walkforward.py`.

- [ ] **Step 3: Re-point callers**

Measured: `registry/promote_run.py:20` (import) + `:234` (call), and `tests/test_holdout_window.py` (6 call sites). Regenerate untruncated.

- [ ] **Step 4: Check for a cycle**

`walkforward.py` and `engine.py` already reference each other's concepts. Verify `uv run python -c "import algua.backtest.engine, algua.backtest.walkforward"` succeeds and that no `TYPE_CHECKING` guard was needed to make it work — **a guard hiding a real cycle is a finding, not a fix** (stage 6c shipped one and had to undo it).

- [ ] **Step 5: Gate and commit.**

---

### Task 4: Extract `to_dict` and DELETE the re-export shim

**Files:**
- Create: `algua/research/gate_serialization.py`
- Modify: `algua/research/gates.py` (CODEOWNERS-protected) + every module importing a re-exported name

**This is the task that removes compat cruft, so it is the one to do thoroughly.**

- [ ] **Step 1: Extract `GateDecision.to_dict`**

Measured at `gates.py:275`, ~76 lines. It is a method on a dataclass, so decide deliberately: a module-level `gate_decision_to_dict(decision)` in `gate_serialization.py` with the method delegating to it, OR the method moved wholesale. **State which you chose and why in your report** — the call sites and any `.to_dict()` usage in tests must keep working either way. Regenerate the `.to_dict(` caller list untruncated.

- [ ] **Step 2: Enumerate the shim, untruncated**

```bash
python3 - <<'PY'
import ast, pathlib
t = ast.parse(pathlib.Path('algua/research/gates.py').read_text())
alls = [n for n in t.body if isinstance(n, ast.Assign) and any(getattr(x,'id','')=='__all__' for x in n.targets)]
names = [e.value for e in alls[0].value.elts]
defined = {n.name for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
defined |= {x.id for n in t.body if isinstance(n,ast.Assign) for x in n.targets if isinstance(x,ast.Name)}
print(sorted(n for n in names if n not in defined))
PY
```
Measured: **43 names in `__all__`, 34 pure re-exports.** Re-verify. For each re-exported name, find its real owner and every importer.

- [ ] **Step 3: Re-point every importer at the real owner, then delete the shim**

Every `from algua.research.gates import X` where X is a re-export becomes an import from X's actual module. Then remove those names from `__all__` and remove the now-unused imports from `gates.py`.

**`__all__` should end up listing only what `gates.py` actually defines.** Report the before/after counts.

If a name turns out to have NO importer outside `gates.py`, say so — that is dead surface and should simply go.

- [ ] **Step 4: Gate and commit.** Expect a wide diff of import lines; that is the point.

---

### Task 5: Close-out verification

**Files:** none expected (verification only; fix anything found).

- [ ] **Step 1: Counts.** Report `engine.py` (**837** at base) and `gates.py` (**668**) before/after, plus the new modules' sizes.

- [ ] **Step 2: No duplication; one definition per moved name**

```bash
grep -rn "local equivalent\|private copy\|byte-identical duplicate\|reproduced here" --include='*.py' algua/
```

- [ ] **Step 3: CODEOWNERS + hygiene guard.** `uv run pytest tests/test_repo_hygiene.py -q` must pass, and both new backtest modules must be in the protected set.

- [ ] **Step 4: The shim is gone**

```bash
grep -c '"' <<< "$(python3 -c "import ast,pathlib;t=ast.parse(pathlib.Path('algua/research/gates.py').read_text());print([e.value for n in t.body if isinstance(n,ast.Assign) and any(getattr(x,'id','')=='__all__' for x in n.targets) for e in n.value.elts])")"
```
Report the final `__all__` and confirm every name in it is defined in `gates.py`.

- [ ] **Step 5: Probe for layering inversions** — do not reason about this. Temporarily add each, run `lint-imports`, record, remove:
- `forbidden: algua.backtest -> algua.registry` and `forbidden: algua.research -> algua.backtest`.
- If either PASSES, **add it permanently**; a guarantee is worth only the contract enforcing it. If it fails, report the violating chain verbatim rather than adding an exemption.

- [ ] **Step 6: Full gate + CLI smoke** (`timeout: 900000`): `uv run pytest -q`, ruff, mypy, lint-imports, then `uv run algua doctor` and `uv run algua backtest run cross_sectional_momentum --demo` (exercises the carved orchestrator end to end).

- [ ] **Step 7: Commit any fixes.** If nothing needed fixing, make no commit.
