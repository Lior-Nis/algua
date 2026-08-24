# Stage 6b — live_cmd Risk Extractions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the account-level book machinery and the drawdown-peak rebase policy out of the CLI into their owning domain layers, and stop `live_cmd` hand-rolling an authenticated HTTP call. Continues §6 item 4 after Stage 6a.

**Architecture:** Three destinations, chosen by what each body actually depends on rather than by what it is named. `algua/risk/book_cycle.py` takes the book loss breaker (risk-only dependencies). `algua/live/book_exposure.py` takes the book exposure builder — it needs three things from `live/live_loop.py`, and `risk` sits **below** `live` today, so filing it under `risk/` would invert the layering. `algua/risk/peaks.py` takes the peak-rebase policy currently duplicated inline across two `paper_cmd` commands. `_live_account_equity` joins the broker class that already owns that endpoint.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §6 item 4, bullets 3–5.

**Ground truth:** a research pass against `main`@`9ca8cb4` that read every body and its dependency closure. Unlike Stage 6a, **no target depends on `cli/_common` at all** — the cascade that dominated 6a does not recur here.

### Decisions (recorded for a reviewer to check, not re-derive)

1. **The spec's single `risk/book_cycle.py` destination splits into two homes, on evidence.**
   - `_evaluate_book_loss_breaker` (`live_cmd.py:343`, ~36 lines) depends only on `broker.account()`,
     `BrokerError`, `risk.book_breaker`, `risk.book_equity`, `config`, `math`. **No live-lane
     dependency.** It goes to `algua/risk/book_cycle.py` exactly as the spec says.
   - `_build_book_exposure` (`live_cmd.py:379`, ~58 lines) needs **three** names from
     `algua/live/live_loop.py`: `assert_marks_usable` (the shared #452 mark-freshness wall),
     `_latest_bar_ts`, `_latest_marks`. Verified today: `algua/live/*` imports `algua.risk.limits`,
     and **`algua/risk/*` imports nothing from `algua.live`** — risk is below live. Filing this body
     under `risk/` would create `risk -> live` and a package-level cycle. It goes to
     `algua/live/book_exposure.py`.
   - A reviewer should check the dependency claim, not the naming: if `_build_book_exposure` truly
     had no live dependency it would belong in `risk/` and this decision is wrong.
2. **`_live_account_equity` is a CONSISTENCY fix, not a security fix — do not oversell it.** It
   hand-rolls a `requests.get` with LIVE credentials instead of going through the broker class that
   declares `_ALLOWED_HOSTS`. But the `alpaca_live_url` settings validator
   (`config/settings.py:107-117`) already pins **both scheme and hostname**
   (`parsed.hostname != _ALPACA_LIVE_HOST` raises), so the credential-exfiltration risk that
   `require_https_allowlisted_host` exists to stop is already closed for this URL. The value here is
   one HTTP idiom instead of two, and the CLI no longer importing `requests`. Say exactly that in
   the PR; a false security claim is worse than no claim.
3. **The peak-rebase extraction is a genuine de-duplication, not a move.** `paper resume`
   (`paper_cmd.py:362`) and `paper resume-all` (`paper_cmd.py:1516`) each encode the same policy
   inline: *which* peak tables to clear, and the ordering invariant that peaks are rebased FIRST and
   the un-halt written LAST so any failure leaves the strategy safely halted. `resume` picks the
   table by stage (`clear_nav_peak` for LIVE, `clear_peak_equity` otherwise); `resume-all` clears all
   three (`clear_all_peaks`, `clear_all_nav_peaks`, `clear_book_peak`). The comments carry #27, #109
   and a codex C1 review finding. Naming the policy once is the point.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass. **24 contracts, 0 broken** is the current baseline.
- **The full suite takes ~7–10 minutes on this machine.** Pass `timeout: 600000` explicitly to the Bash tool. If the harness backgrounds the run anyway, **do NOT end your turn waiting** — read the output file the harness names, or re-run in the foreground. Seven agents on this program have stalled exactly this way. To check whether a real pytest is live, match the **executable** (`readlink /proc/PID/exe`), not `pgrep -f pytest` — the cmdline match is polluted by leaked test processes whose data-dir path contains "pytest".
- **A "pure move" NEVER licenses duplication to satisfy it.** Five times in Stage 6a an implementer hit a helper it needed, correctly reasoned that importing it would create a bad edge, and duplicated it instead; every one was undone. If a moved body needs a shared helper, **the helper moves to its owning layer**. If you catch yourself writing "local equivalent of X" / "private copy of X" / "byte-identical duplicate of X" — stop.
- **Every scripted edit must assert it changed something.** An unguarded `.replace()` that silently misses is indistinguishable from one that succeeded; this produced a duplicated helper in Stage 6a that survived a commit.
- **Regenerate every count untruncated at the moment you assert it** — no `| head`/`| tail` on an enumeration you are counting. If a number here disagrees with what you measure, trust your measurement and say so.
- Moved bodies keep their names, signatures, `# noqa` codes, docstrings and comments **verbatim**. The comments on these particular bodies encode fail-closed reasoning (GATE-1 peak-corruption, GATE-2 unvaluable-book, the #452 shared staleness wall, the dark-feed halt-without-flatten routing). Losing one loses the reason a wall exists.
- **CODEOWNERS-protected files this stage touches: `algua/cli/paper_cmd.py`** (Task 2). On the paper→live wall. Expected; note it in the PR.
- `git add` scoped to named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known hazard: some test writes demo strategy files into `algua/strategies/momentum/`. Check `git status --porcelain` before staging and delete any stray untracked file there.

---

### Task 1: Move the two book bodies to their owning layers

**Files:**
- Create: `algua/risk/book_cycle.py`, `algua/live/book_exposure.py`
- Modify: `algua/cli/live_cmd.py`

**Interfaces:**
- Produces: `evaluate_book_loss_breaker(conn, broker) -> BookBreach | None` and
  `build_book_exposure(broker, provider, net_positions, start, end, now=None) -> tuple[BookExposure | None, str | None]`
  — **today's exact signatures**. Drop only the leading underscore, since they stop being module-private.

- [ ] **Step 1: Verify the dependency claim yourself before moving anything**

Decision 1 rests on a measurement. Confirm it:

```bash
grep -rn "from algua.risk\|import algua.risk" --include='*.py' algua/live/
grep -rn "from algua.live\|import algua.live" --include='*.py' algua/risk/
```
Expect: `live -> risk` edges exist, `risk -> live` returns nothing. If `risk` already imports `live`, this plan's layering argument is wrong — stop and report rather than proceeding on it.

Then confirm `_build_book_exposure` really needs `assert_marks_usable`, `_latest_bar_ts` and `_latest_marks` from `algua/live/live_loop.py`, and that `_evaluate_book_loss_breaker` really needs none of them.

- [ ] **Step 2: Move both bodies verbatim**

`_evaluate_book_loss_breaker` → `algua/risk/book_cycle.py` as `evaluate_book_loss_breaker`.
`_build_book_exposure` → `algua/live/book_exposure.py` as `build_book_exposure`.

Give each module a docstring saying which layer it belongs to **and why**, including the constraint that decided it: `risk` is below `live`, so a body needing the live-lane marks wall cannot live in `risk`.

`live_cmd` imports both. Keep its local names bound as module attributes — see Step 3.

- [ ] **Step 3: Protect the test pins**

`tests/test_cli_live.py` patches `algua.cli.live_cmd._evaluate_book_loss_breaker` (1 pin, verified).
Keep a module-level binding in `live_cmd` so the pin still resolves:

```python
from algua.risk.book_cycle import evaluate_book_loss_breaker as _evaluate_book_loss_breaker
```

This is the same alias-preserving move that made Stage 6a's 57 `_select_provider` pins free. **Prove it fires** rather than assuming: make the patched replacement raise, confirm the test FAILS, restore, confirm it PASSES. Run with `PYTHONDONTWRITEBYTECODE=1` and clear `__pycache__` — a same-second restore can otherwise reuse mutated bytecode and make the result lie in either direction.

- [ ] **Step 4: Full gate, then commit.**

---

### Task 2: Extract the peak-rebase policy to `algua/risk/peaks.py`

**Files:**
- Create: `algua/risk/peaks.py`
- Modify: `algua/cli/paper_cmd.py` (CODEOWNERS-protected)

**Interfaces:**
- Produces: `rebase_strategy_peak(conn, name: str, stage: Stage) -> None` and `rebase_all_peaks(conn) -> None`.

- [ ] **Step 1: Read both call sites in full before writing anything**

`paper_cmd.py:362` (`resume`) and `paper_cmd.py:1516` (`resume-all`). Read every comment. The policy has three parts and all three must survive into the new module's docstring, because they are the reasons it exists:
- **Ordering:** rebase the peak FIRST, clear the kill-switch / global halt LAST, so any earlier failure leaves the strategy safely halted and the operation stays retryable (#109).
- **Per-stage table choice:** a LIVE strategy's drawdown breaker reads the NAV peak (`live_nav_peaks`), not the paper peak. Clearing the wrong one leaves a resumed live strategy re-tripping on a stale pre-breach NAV peak (codex C1 review).
- **Why rebase at all:** without it, a drawdown trip → flatten-to-cash re-trips every tick against the stale pre-loss peak (#27).

- [ ] **Step 2: Write the two policy functions**

`rebase_strategy_peak(conn, name, stage)` — `clear_nav_peak` when `stage is Stage.LIVE`, else `clear_peak_equity`.
`rebase_all_peaks(conn)` — `clear_all_peaks` + `clear_all_nav_peaks` + `clear_book_peak`.

**The ordering invariant stays at the call sites**, because it is about the relationship between the rebase and the un-halt write — a function cannot enforce the order of a write it does not perform. Say so in the module docstring so the next reader does not assume the function covers it.

`algua/risk/` must not import `algua.cli` (existing contract). These functions take an open `conn`, so they need no connection helper.

- [ ] **Step 3: Replace both inline blocks with calls**

Keep the ordering comments at the call sites. Replace only the table-selection lines.

- [ ] **Step 4: Prove the per-stage branch still works**

This is the branch a codex review already caught once. Confirm both directions are covered:
```bash
grep -rn "resume" tests/test_cli_paper.py | grep -i "live\|nav_peak" 
```
Measured on `main`@`9ca8cb4`: **8 matching lines**, so coverage very likely already exists — confirm it actually asserts the TABLE choice, not merely that resume succeeded. If nothing pins the branch, **add a test** — a policy whose whole point is picking the right table per stage needs a test that fails when the branch is inverted. Verify it does fail when inverted.

- [ ] **Step 5: Full gate, then commit.**

---

### Task 3: Move `_live_account_equity` onto the broker

**Files:**
- Modify: `algua/execution/alpaca_broker.py`, `algua/cli/live_cmd.py`

- [ ] **Step 1: Read the existing broker classes first**

`algua/execution/alpaca_broker.py` — note how `_ALLOWED_HOSTS` is declared per subclass and validated in the base via `require_https_allowlisted_host` (`:104`), and how existing methods issue requests. The new method must follow that idiom, not import `requests` independently.

`AlpacaLiveReadOnlyBroker` (`:483`) is the natural host: reading account equity is a read-only live operation, and that class exists for exactly this shape of call — its docstring says *"constructed WITHOUT a LiveAuthorization because it never places an order — both endpoints are GETs"* — with `_ALLOWED_HOSTS = {"api.alpaca.markets"}` at `:490`.

- [ ] **Step 2: Move the body**

Preserve behaviour exactly, including the `allow_redirects=False` line and its comment — that comment records #394 (refusing to chase a redirect so APCA credential headers can never reach a foreign target). Preserve the error translation: a `RequestException` and a non-200 both become `ValueError`, and the missing-credentials check raises `ValueError` before any request.

- [ ] **Step 3: Keep the CLI pin alive**

Three tests patch `algua.cli.live_cmd._live_account_equity` (`tests/test_cli_live.py` ×2, `tests/test_forward_certificate.py` ×1 — regenerate this list untruncated). Keep `_live_account_equity` as a thin module-level function in `live_cmd` that delegates to the broker, so all three pins keep resolving and keep covering.

**`tests/test_forward_certificate.py` is on the go-live path.** Stage 5b silently disarmed a guard in exactly that file by moving a binding out from under a patch. Prove this pin still fires in both directions before you call the task done.

- [ ] **Step 4: Confirm `requests` is gone from the CLI**

```bash
grep -n "requests" algua/cli/live_cmd.py
```
Expect nothing. Report what you find.

- [ ] **Step 5: Full gate, then commit.**

---

### Task 4: Close-out verification

**Files:** none expected (verification only; fix anything found).

- [ ] **Step 1: The bodies are gone from the CLI**

```bash
grep -n "def _evaluate_book_loss_breaker\|def _build_book_exposure\|requests" algua/cli/live_cmd.py
```
Only the thin `_live_account_equity` delegate and the two import aliases should remain. Report `live_cmd.py`'s line count before and after (it was **837** on `main`@`9ca8cb4`).

- [ ] **Step 2: The layering claim holds**

```bash
grep -rn "from algua.live\|import algua.live" --include='*.py' algua/risk/
```
Must return nothing — if the new `risk/book_cycle.py` or `risk/peaks.py` reaches into `live`, Decision 1 was implemented wrong.

Then confirm `uv run lint-imports` reports **24 kept, 0 broken**.

- [ ] **Step 3: Consider a `risk -> live` contract**

`risk` importing `live` is currently prevented only by convention. Probe whether a
`forbidden: algua.risk -> algua.live` contract passes on this tree. **If it passes, add it** — Stage
6a's review showed that the guarantee a stage claims is worth only as much as the contract enforcing
it. If it fails, do NOT add it with an exemption: report the violating edge instead.

- [ ] **Step 4: Full gate + CLI smoke** (`timeout: 600000`): `uv run pytest -q`, ruff, mypy, lint-imports, then `uv run algua doctor` and `uv run algua fleet status`.

- [ ] **Step 5: Commit any fixes.** If nothing needed fixing, make no commit — expected, and consistent with every prior close-out in this program.
