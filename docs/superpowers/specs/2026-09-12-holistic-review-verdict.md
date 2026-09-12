# Algua holistic review — the verdict

**Status:** agreed 2026-09-12 between the operator (Lior) and the agent.
**Scope:** a whole-system review against `docs/PRD.md`, aligned to one goal — *the factory reliably
produces strategies that pass the **unrelaxed** forward gate* — and to the principles the operator
asked to be judged by: architecture, clean code, DRY, KISS, SOLID, YAGNI.
**Authority:** this document does not amend the PRD. Where it disagrees with `docs/PRD.md`, the PRD
wins and the disagreement is a bug here. It *does* supersede the roadmap ordering in PRD §10 for the
next two quarters, with the operator's agreement recorded in §2.

---

## 1. The finding that reframes everything

**The forward gate cannot be passed in sixty days by any strategy, and could not have been.**

`paper → forward_tested` has a statistical-significance wall: the one-sided lower confidence bound
on the realized annualized Sharpe must clear the performance bar, not merely clear zero
(`algua/research/forward_gates.py:198-235`). Every check on that gate binds — unlike the research
gate, the forward decision is `all(c["passed"] for c in checks)` with no advisory exemption
(`forward_gates.py:392`).

Solving the shipped formula for the Sharpe an observed window must show to clear even the 0.3 floor:

| Observations | Required observed annual Sharpe | Calendar at 1 bar/session | Calendar at 6.5 bars/session |
|---|---|---|---|
| 63 | 3.66 | 3 months | 2 weeks |
| 126 | 2.65 | 6 months | 1 month |
| 252 | 1.95 | 12 months | 2 months |
| 504 | 1.47 | 24 months | 4 months |

The code says so itself, at `forward_gates.py:53-56`: *"at MIN_FORWARD_OBSERVATIONS=63 clearing even
the 0.3 floor bar at the LCB demands an observed ANNUAL Sharpe of ~3.8; the remedy for a marginal
strategy is a LONGER forward window, NOT a weaker bar."*

Two consequences follow, and they are the spine of this document.

**First, 63 is a floor, not a target.** It is the minimum before the gate is *evaluated*. Treating it
as the pass condition has quietly set an expectation the machine was never built to meet. The first
realistic pass is somewhere around 250 to 500 observations.

**Second, the stated rationale for the intraday contract is wrong, but the contract is still right.**
PRD §10 step 7 says intraday unblocks *"faster forward validation (63 hourly observations is two
weeks)"*. Sixty-three hourly observations require the same unreachable 3.66 as sixty-three daily
ones — worse, in fact, since intraday returns are autocorrelated and carry less independent
information per observation. What intraday actually buys is **6.5 observations per calendar day
instead of one**, which reaches a *passable* observation count in a quarter rather than in two
years. That is a strong case. It is simply a different one, and the observation floor must be
re-derived to match (see §5).

The operator's decision, recorded: **keep the wall, buy time with intraday.** The calibration is not
touched. The horizon moves instead.

---

## 2. What the system is actually doing right now

The factory has produced zero candidates since the 2026-09-03 reset. It is not leaking strategies at
a gate. It is **starved at the top and stopped at the bottom**, and every stage in between is
correctly reporting that it has nothing to do.

| Stage | State before this review | Evidence |
|---|---|---|
| forage | **never executed**; first fire was scheduled for 2026-09-13 06:00 | `journalctl --user -u algua-forage`: no entries |
| `kb/inspirations/` | empty — only `_sources.yaml` | directory listing |
| leap | *"no fresh material (0 non-exhausted inspirations); nothing to do"*, every two hours | journal, five consecutive fires |
| idea pool | empty: 0 unclaimed, 0 claimed, 0 parked, against `refill_at: 72` | `algua research idea depth` |
| research loop | outcome `pool_empty`, exit 0, every two hours | `data/research-runs.jsonl` |
| paper lane | timer **disabled**; global halt **engaged since 2026-09-05** | `systemctl --user status`, `algua doctor` |

The one strategy at `paper` had accrued **zero of its sixty-three observations**, because the clock
had never started.

Meanwhile the research gate itself has been asked exactly **once**, and it **passed** —
`gate_evaluations` held a single row with 536 holdout observations against a floor of 63. Issue #517
("bundled bars too shallow") is resolved: the snapshot carries 20 symbols over 2016-01-01 to
2026-09-05, roughly 2,690 sessions. **Data depth is not the bottleneck, and has not been for some
time.**

### 2.1 What was cleared during this review

Four blocks were costing one hundred per cent of throughput. All four were agent-executable under
PRD §7; none needed a human. With the operator's approval they were cleared:

1. **The global halt** — engaged on 2026-09-05 to flatten orphan positions left by the wiped fleet.
   The account was verified flat first (equity equal to cash, zero positions, zero open orders), so
   the halt was stale state rather than a live guard. `algua doctor` now passes every check.
2. **The paper timer** — enabled. The first cycle refreshed its own bars, minted snapshot
   `4dae3256cd7b6796`, completed the 2026-09-11 session, submitted three orders and reconciled
   clean. The lane reports `health: ok`. **The observation clock has started.**
3. **A stale systemd drop-in** pinning `ALGUA_PAPER_SNAPSHOT` to a snapshot that no longer exists.
   The variable is dead code; the drop-in was removed.
4. **#636, the self-dirtying deadlock** — see §3.

### 2.2 The cadence defect that will re-starve the pool

Forage runs **daily** with `FORAGE_MAX_NOTES=10` over `FORAGE_SLICES=2` categories
(`.codex/scripts/forage.sh:29,34`). The research loop consumes `hypotheses_per_run` 3 ×
`runs_per_day` 12 = **36 ideas per day**, against a refill trigger of 72. Ten notes per day cannot
feed thirty-six claims per day. The pool will oscillate around empty even once forage starts.

This is a sizing bug, not a design flaw: either forage runs several times a day, or `FORAGE_MAX_NOTES`
rises, or `runs_per_day` falls to what the top of the funnel can actually supply. **Work item F1.**

---

## 3. The bug that would have fired on the first real candidate

The four gate commands sync an Obsidian document per strategy as a best-effort out-of-transaction
side effect; `algua/knowledge/sync.py:85` writes into `knowledge_dir/"strategies"`. That directory
was neither tracked nor ignored, so **every successful gate run left the working tree dirty**.

`algua/operator/mergeback.py:200` refuses to merge a candidate branch unless `git status --porcelain`
is empty.

Those two facts compose into a self-inflicted deadlock: **the factory blocked its own merge-back the
moment a gate run succeeded.** `mergeback_evidence` holds zero rows, so this had never been survived
in production — it would have fired on the first candidate the ideation engine produced.

Fixed in PR #645 by ignoring `kb/strategies/` alongside the sibling runtime vault directories
(`kb/experience/`, `kb/inspirations/`) it belongs with, and by untracking
`kb/.obsidian/workspace.json`, per-user pane state that churns on every editor focus change.

**One residual the operator must resolve:** `approvers/allowed_signers` carries an uncommitted line
enrolling a key for both the go-live and human-actor namespaces. Until it is committed, the checkout
is dirty and merge-back still refuses. That file is the signature trust anchor, so committing it is
the operator's act, not the agent's. **Work item F2.**

---

## 4. Where the walls are weaker than the documents claim

Two walls named in `docs/architecture.md` do not hold the way the prose says.

### 4.1 The executable CODEOWNERS wall is one-legged

There are two CODEOWNERS files. The repository root one lists 34 integrity-critical paths and is what
`algua/operator/diff_policy.py:110` parses at runtime for the merge-back denylist — **that leg works**.
But `.github/CODEOWNERS` also exists and lists three paths, and GitHub resolves `.github/` first, so
**GitHub enforces code-owner review on three paths and ignores the other thirty-four.**

Verified empirically: PR #644 touched `algua/portfolio/overlays.py`, a root-CODEOWNERS path, and
merged with no review decision, no review requests and no reviews.

Stated fairly: for a sole owner, GitHub code-owner review is self-defeating — you cannot approve your
own pull request, and `required_approving_review_count` is 0. The minimal `.github/CODEOWNERS` may
well be deliberate. **The defect is the documentation claiming a human-review wall that is not
enforced, and the two-file split that hides which file is live.** Either delete `.github/CODEOWNERS`
so the root file governs both legs, or amend `docs/architecture.md` and `AGENTS.md` to say plainly
that CODEOWNERS is a runtime merge denylist and not a review gate. **Work item W1.**

### 4.2 Lane parity is a test standing in for a design, and it already missed something

`tests/test_lane_parity.py` names the two lane functions **by string** and asserts four call sites.
Its own docstring concedes the lanes are *"only ~38% structurally similar, so a fix to one has no
mechanical reason to reach the other."* Measured similarity on normalized lines: `run_all` 0.43,
the tick 0.42, flatten and halt-all 0.27.

The test enumerates four invariants rather than deriving them, so it cannot see the drift that has
accumulated:

- **The paper lane has no whole-account loss breaker.** `book_` appears 24 times in `live_cmd.py`
  (the book loss breaker, `book_stale_marks_halt`, `book_circuit_breaker`) and five times in
  `paper_cmd.py` — where all five are `paper_book_capacity` configuration strings. **The rehearsal
  lane is missing a production safety wall**, and the parity test is silent about it.
- Error precedence is inverted between the lanes: `live_cmd.py:335-341` validates `--max-drawdown`
  before the snapshot/refresh exclusivity check, `paper_cmd.py:969-975` after. The same bad
  invocation produces a different error per lane.
- The tick-halt envelope differs: paper returns `{"ok": False, "strategy": name, ...}`, live returns
  a bare payload with no `strategy` key, while `operator/jobs.py:82` aggregates on both.
- The audit vocabulary has forked (`trade_tick_halted` versus `live_trade_tick_halted`), so any
  cross-lane audit query must know both spellings.

**The design that makes the test unnecessary is one lane body and two lane specifications** — see
§6, change 1. Until then, the missing paper book breaker is a live gap and should be closed on its
own. **Work item W2.**

---

## 5. The intraday execution contract

Pulled forward from PRD §10 step 7 to become the next substantial build, on the corrected rationale
of §1: more observations per calendar day, not a shorter observation count.

### 5.1 What is already ready

More than expected. The entire `algua/features/` package is bar-counted and interval-agnostic — no
`252`, no "days", no baked-in window defaults. So are `algua/portfolio/`, the reconcile stack
(cycle-ordinal and symbol-grouped, never date-grouped), `execution/order_state.py` (second-resolution
client order ids, id-ordered reads, an append-only `tick_snapshots` with no per-session unique key),
Alpaca activity ingestion (instant cursor), and `execution/sim_broker.py::fill_pending`, which
**already is** the "decide on a closed bar, fill on the next bar" contract. `algua/data/timeframes.py`
already defines `DAILY`, `INTRADAY` and `is_intraday`, and both providers already branch on it.

`feature_lookback` is documented and used as **bars** everywhere (`strategies/base.py:69`), which is
the right unit and needs no change.

### 5.2 The five clusters that must change

1. **Annualization.** A single `ANN = 252` (`backtest/_constants.py:9`) fans into `backtest/metrics.py`
   and `research/{gates,forward_gates,dsr,haircut,regime}.py`. There is no `periods_per_year`
   anywhere, and `runs.ann_vol_is`/`ann_vol_oos` are persisted with no unit tag.
2. **Session-counted freshness.** `risk/limits.py:31` `MAX_STALE_SESSIONS` (documented as
   "un-relaxable by design, no settings field"), `execution/fleet_health.py:44,49`,
   `forward_gates.py:70,89,98`, `forward_evidence.py:54`, `registry/live_certificate.py:74` — all
   route through `calendar/market_calendar.py:91-102` `sessions_stale`, which maps a bar by its
   UTC-midnight **date**. The intraday-correct siblings `session_of_instant` and
   `sessions_between_instants` already exist but are not on the `SessionCalendar` protocol
   (`forward_evidence.py:37-43`).
3. **Window derivation.** `cli/_common.py:141` `LIVE_WINDOW_LOOKBACK_DAYS = 400` and
   `cli/lane_refresh.py:110-145`, where `cycle_start` counts sessions one-for-one against a **bar**
   requirement — a 6.5× over-fetch at hourly, 26× at fifteen minutes. `lane_refresh.py:170`
   `require_bar_on = previous_session(today)` is the single line that decides which session the tick
   decides on.
4. **The forward-gate observation key.** `forward_evidence.py:223` keys observations by
   `session_on_or_before(decision_ts.date())`, collapsing every tick in a session to one, and
   coverage is decided-sessions over exchange-sessions (`:229-234`).
   `forward_gate_evaluations` carries **no timeframe or clock column** (`db/forward_gate.py:20-50`),
   so a certificate cannot record which contract earned it.
5. **Operator scheduling.** `operator/schedule.py` keys idempotency on a session date and returns
   `already_ran` for any second fire in a session. Note that `operator/loop_health.py` already runs
   **two** clocks — wall-clock for research and merge-back, sessions for paper — which is proof the
   design accommodates a third.

### 5.3 Two traps and one honest caveat

- `live/live_loop.py:216-218` filters bars with `ts.date() < today`. Under hourly bars this discards
  **every bar of the current day**, so an intraday loop would decide on yesterday's last bar forever.
  A silent no-trade, not an error.
- `risk/book_cycle.py:55` ratchets the account high-water mark **once per cycle**. Going from one
  cycle a day to twenty-six would tighten the drawdown breaker by a factor of twenty-six with nobody
  changing a setting.
- **The fill model stops matching.** Orders are submitted `type=market, time_in_force=day` with no
  `extended_hours` (`execution/alpaca_broker.py:313`). Today's post-close submission fills at the next
  open *by accident of scheduling*, which happens to match the simulator. A market order submitted
  mid-session fills at the next print, not at the next bar's open, so the backtest's fill reference
  would no longer describe live behaviour. **This must be designed, not inherited.**

### 5.4 The seam

One `Cadence` value object carried on `ExecutionContract`: bar timeframe, periods per year, bars per
session, and the staleness unit with its bound. Daily becomes `Cadence("1d", 252, 1, sessions=2)` and
nothing about the daily path changes. It threads into metrics, the observation key, the freshness
walls, window derivation, the engine's `get_bars`, the closed-bar rule, the schedule gate, and a new
timeframe column on `forward_gate_evaluations` and `runs`.

**The observation floor must be re-derived at the same time.** Sixty-three is calibrated for daily
independence. An intraday floor needs to be materially larger, and the standard error needs an
autocorrelation correction — Newey-West, or a block bootstrap — or the gate will look harsh while
being statistically weaker than the daily one it replaces. Shipping the clock without this would
quietly convert a wall into a rubber stamp.

### 5.5 A sequencing constraint worth knowing up front

Nearly every file this build touches is on the root CODEOWNERS list: `backtest/engine.py`,
`walkforward.py`, `pit_view.py`, `decision_path.py`, `grid.py`, `research/gates.py`,
`research/forward_gates.py`, the registry promotion and forward-evidence modules, `registry/db/`,
`registry/store/`, `cli/_common.py`, `cli/paper_cmd.py`, `portfolio/construction.py`. Whatever §4.1
is resolved to, these are the integrity-critical paths, so the work should land in **few, large,
reviewable pull requests** rather than many small ones.

---

## 6. Principles: what the code is carrying

42,122 source lines and 65,680 test lines, a ratio of 1.56 to 1. The tests are the larger artifact,
so every deletion returns more than its own weight.

### 6.1 The structural fact behind most of the excess

`algua/research/gates.py:501` composes the research verdict as
`passed=all(c["passed"] for c in checks if not c.get("advisory"))`. **Exactly three checks bind**:
`min_holdout_observations`, `holdout_sharpe_floor`, `pit_required`. Everything else — deflated
Sharpe, window stability, DSR, bootstrap, regime robustness, idiosyncratic alpha, the false-discovery
ledger — is `advisory=True` and vetoes nothing.

That is the PRD §4 soft gate working as designed. But it means the whole breadth → deflation → FDR →
family chain is **computed, recorded, and consumed by no decision.** Keeping the number is cheap;
keeping the machinery that produces it is not.

### 6.2 The five structural changes

1. **One lane body, two lane specifications.** A `LaneSpec` carrying ledger kind, stage set, broker
   factory, authorization hook, ingest function, order-record hooks and audit prefix, with the cycle
   and the tick moving to `algua/live/lane_cycle.py`. The two command modules become roughly
   200-line flag shells. **Then delete `tests/test_lane_parity.py`** — a structural test that names
   two functions by string is strictly weaker than having one function, and it already failed to
   catch the missing book breaker it exists to prevent. Removes about 250 duplicated lines, moves
   about 900, and collapses roughly 5,100 lines of lane tests to about 3,200. **Risk: high** — this
   is the real-money path. Extract to the specification with both behavioural suites green at every
   step, and do the live lane last.
2. **Delete the ten unused repository protocols.** `registry/repository.py` is 965 lines, about 616
   of them protocol declarations, with exactly one implementer and only three of thirteen ever used
   to narrow a parameter. Two of the declarations carry 27 and 28 parameters maintained in two
   places. Drops to about 350 lines. Risk low; mypy proves it.
3. **Retire the frozen false-discovery surface.** `promotion.py:517` states that
   `final_passed == provisional_passed` unconditionally, yet `fdr_` appears roughly 200 times across
   fifteen files and `GateDecision` is 115 lines with about twenty permanently-null fields. Keep the
   columns for audit history, delete the write plumbing. The #529 spec is a cheaper record of how to
   re-tighten than 200 live references.
4. **Lift the merge-back saga out of the CLI.** `cli/paper_cmd.py:468-670` builds the saga inside a
   Typer command — shelling out to `git rev-parse` twice, taking the flock, constructing the journal,
   reading CODEOWNERS, and defining five policy closures — and reaches its collaborators through
   `importlib.import_module` four times, **to route around the import-linter layering rather than
   satisfy it**. Move it to `operator/mergeback_driver.py`. CODEOWNERS coverage must extend to the
   new module *in the same commit*, or this weakens the denylist.
5. **Replace the frozen size ratchet with a per-package budget.** Thirty-four modules are pinned and
   **thirty-two sit at exactly their pin**. That is not a ratchet, it is a freeze: every pinned
   module is one line from failing, so the cheapest edit anywhere in the core is "put it in a new
   file" regardless of where the change belongs. A package-level budget makes carving within a
   package free while total growth still ratchets. Separately, rename the simulation path — `paper
   run` to `backtest replay`, `paper_orders`/`paper_fills` to `sim_orders`/`sim_fills` — so that
   "paper" means the lane and nothing else. Today it means two unrelated things and the schema has
   forked along the ambiguity.

### 6.3 Smaller items worth doing anyway

Two protocols with zero references (`contracts/types.py:304`, `:325`); two empty package directories
(`algua/shadow/`, `algua/monitoring/`, both containing only `__pycache__`); and **six copies of
`_now()`** across the registry plus two more timestamp parsers, which should consolidate into
`primitives/timeparse.py`. About forty lines, no risk, and it removes a class of UTC-format drift
between ledgers that a future reconcile bug would otherwise be blamed on.

Also note **820 issue-number citations across 172 source files**, densest in `repository.py` (45),
`paper_cmd.py` (41) and `live_cmd.py` (29). Much of it is load-bearing rationale. But in the lane
modules the pattern is substitution — a comment plus a test standing in for the shared function that
would have made the drift impossible. When a reader needs `gh issue view` to understand a branch,
the design is living in the issue tracker.

---

## 7. The cut list

Roughly **6,000 source lines and 12,500 test lines** — 23% of source, 19% of tests — without touching
a single binding gate check, the point-in-time wall, the single-use holdout, the human live wall,
lane parity, or the runtime merge denylist.

One fact makes all of it cheap: **there is no migration ladder.** `registry/db/migrate.py:32` is a
224-line idempotent bootstrap, not a numbered sequence, so removing a subsystem never requires a
migration — dropping a table is one line, exactly as versions 40 and 41 already did for
`shadow_evaluations` and `factor_evaluations`.

| Rank | Subsystem | Verdict | Source | Tests | Risk |
|---|---|---|---|---|---|
| 1 | Family governance: clustering, mint cap, breadth inheritance, search breadth | delete | ~1,200 | ~2,670 | med-low |
| 2 | Advisory statistics: `regime`, `bootstrap`, `neff`, `haircut`; shrink `dsr` to one scalar | delete / shrink | ~1,290 | ~2,400 | low-med |
| 3 | Monitor PWA under `web/` | **operator's call** | 7,595 | included | low for `algua/` |
| 4 | Fundamentals and news point-in-time seams, two strategies, four commands, the three-argument dispatch | delete | ~800 | 937 | medium |
| 5 | Merge-back saga: seven-state taxonomy to three, drop the durable journal and resurrection path | shrink to ~600 | 745 | ~1,900 | med-high |
| 6 | LORD++ false-discovery ledger and `search_trials` | delete | ~450 | ~500 | low |
| 7 | `eval gate` harness | delete | 472 | 190 | low |
| 8 | Knowledge governance (SR 11-7) and negative-results ledger | delete | ~640 | ~200 | low |
| 9 | MLflow tracker, leaving the SQLite backend | shrink | 360 | ~150 | low |
| 10 | `knowledge/sync.py`: drop the wikilink graph and dependent-doc propagation | shrink to ~120 | 355 | ~316 | medium |
| 11 | `research run-all` batch worker | delete | 183 | — | low |
| 12 | Dead commands: governance, `audit log` reader, `registry set`, `backfill-from-kb`, `paper allocate` | delete | ~250 | ~100 | low |
| 13 | Human-actor signing: fold `approvals.py` in, remove `--allow-non-pit` | **shrink only** | 297 | ~293 | **medium — a wall** |
| 14 | Empty directories and dead migration comments | delete | 0 | 18 | none |
| 15 | `algua/models`: drop the MLflow registration arm | shrink | 254 | — | low |

### 7.1 Three cuts that need their reasoning stated

**The family mint cap is a live throughput brake, and it is the first thing to remove.**
`AGENT_NOVEL_MINT_CAP = 8` per rolling ninety days (`registry/store/family.py:22`, raising at `:379`)
blocks the ninth novel-family promotion. So the only live effect of about 1,200 lines of clustering,
code-ancestry analysis and compare-and-swap fingerprinting is **a hard cap on how many genuinely
uncorrelated strategies the factory may promote per quarter** — enforcing a breadth tax that is no
longer levied, since its only consumer is an advisory check. Against PRD §4, which wants "as many
uncorrelated hypotheses as it can", this is an active anti-goal. It is a one-line change and it
should ship even if nothing else on this list does.

**The fundamentals and news seams can never reach the gate that matters.**
`algua/strategies/tradable.py:15,25` already raises for any `needs_fundamentals` or `needs_news`
strategy entering paper or live — *"paper/live news wiring is not built"*. No provider serves either
feed. So this is a backtest-only lane whose two example strategies are structurally incapable of
reaching `forward_tested`. Deleting it also collapses the two-argument versus three-argument signal
dispatch in the engine and the loader, which pays out well beyond its line count.

**MLflow should go, and the system is already telling us so.** During this review a `research
promote` run emitted a deprecation warning and a `FileStore` traceback for a malformed experiment.
PRD §11 names the deprecated file store as an open question blocking both a security bump and the
model-artifact seam. Nothing downstream of a gate, promote, tick or health check reads an MLflow run;
the only consumer is a number in an Obsidian document. Making the existing SQLite backend the only
one resolves the open question by deletion.

### 7.2 What stays, and why, so it is not re-litigated

- **The `dormant` stage.** Four lines of enum; removing it touches eighteen source files and ten test
  files to reclaim about ninety lines. Negative return. (There is a real defect here — `dormant →
  paper` restores the stage but not the book slice — which argues for fixing the edge or banning it,
  not for deleting the stage.)
- **Both bulk importers.** PRD §5 names FirstRate *and* Databento as pre-built optionality for
  roadmap step 2, the data purchase. Deleting one means rebuilding it inside a month.
- **The live lane, all twelve empty tables.** It is empty because the roadmap has not reached it, not
  because it is dead. PRD §10 step 1 *is* the tuition trade to live. **Emptiness is evidence of
  deadness only when the roadmap has already passed the code by** — which is exactly the distinction
  between the live lane and `web/`.
- **The human-actor signature.** PRD §8 requires rung-0 relaxation to be signed: a recorded act of
  the human taking responsibility, never an agent path. Deleting it would turn `--actor human` back
  into a forgeable string. Shrink the surface it guards; never remove the guard.
- **`research/eval_harness.py` and the unexercised human-only relaxation flags.** Both are cheap, and
  both are escape hatches whose absence gets discovered at the worst possible moment.

---

## 8. The operator runtime moves to OpenCode

The operator's decision: **one runtime, provider-agnostic**, fed from OpenRouter and Chinese-model
subscriptions. Codex and Claude launchers go away.

This is smaller than it sounds. The four drivers — research, leap, forage and the merge-back drainer
— are over ninety per cent runtime-agnostic bash over `uv run algua ...`, and the drainer uses no
language model at all. Only the invocation block is Codex-specific. OpenCode 1.18.30 is already
installed and credentialed, and it already discovers the repository's seven skills through the
existing `.claude/skills` symlinks, and already auto-loads `AGENTS.md`.

**Shape:** one `opencode.json` at the repository root as the single model and permission
configuration; five agent definitions under `.opencode/agents/`; and one
`.opencode/scripts/run_agent.sh` seam taking mode, working directory, prompt file, model, variant,
timeout and a dry-run flag. The drivers move from `.codex/scripts/` to `.opencode/scripts/` with
their `CODEX_CMD` blocks replaced by a call to that seam. Roughly 400 new lines, 150 edited, fifteen
renames. Model selection becomes explicit and per-loop, where today every loop silently inherits
whatever is in the user's global Codex configuration.

**Three things have no equivalent and must be handled deliberately:**

1. **The kernel write wall.** Codex's `-s workspace-write` is real filesystem containment. OpenCode's
   permissions are tool-level only, so a cheap model's shell can write the authority database. The
   seam must reintroduce a kernel wall — `bwrap` or `systemd-run` — which also closes the `/tmp` hole
   the ideation spike found in Codex.
2. **The network kill switch.** Leap and forage run with `network_access=false` today. OpenCode's
   agent traffic and its shell share a namespace, so the closest equivalent is denying network
   binaries by bash pattern.
3. **Configuration bleed.** OpenCode always merges the user's global configuration, which currently
   injects a plugin, about ninety global skills and a global `AGENTS.md` into every run — enough to
   confuse a smaller model. The seam should override `XDG_CONFIG_HOME` to a repository-owned
   directory and pass `--pure`.

Two further risks to hold in mind: headless `ask` permissions default to blocking, so every rule must
resolve to allow or deny or a run will hang until its timeout; and OpenCode's exit-code semantics on
early model termination are unverified. The drivers already fail closed on a missing trailer, which
bounds the damage. Smoke-test both before cut-over.

The multi-model review panel loses its second lineage. It can be preserved by running two OpenCode
invocations on different model families, which keeps model independence but makes a single command
line a single point of failure for the whole panel. Worth stating explicitly rather than discovering.

---

## 9. The program

Ordered by what unblocks what, not by size.

**Phase 0 — the machine runs.** Done during this review, except where noted.
Global halt cleared; paper timer enabled and ticking; stale drop-in removed; #636 fixed in PR #645.
Remaining: **F1** rebalance the forage-to-research cadence so the pool stops draining; **F2** the
operator commits `approvers/allowed_signers` so merge-back's clean-checkout precondition can hold.

**Phase 1 — width, immediately.** Re-gate the strategy modules orphaned by the 2026-09-03 wipe. This
exercises the whole `backtested → candidate → paper` path today without waiting on ideation, and
puts many strategies in the book so their observation clocks run in parallel. Width is the only lever
that requires no new code, and with the gate's LCB posture unchanged it is also the only way to get
enough shots at a high realized Sharpe. Expect the family mint cap to bite at the ninth novel family
— which is the empirical case for cut §7.1.

**Phase 2 — cut weight.** The list in §7, largest and least risky first: family governance, the
advisory statistics stack, the frozen false-discovery surface, the fundamentals and news seams.
Batched into few pull requests because the paths are integrity-critical.

**Phase 3 — the intraday contract.** The `Cadence` seam of §5.4, **with the observation floor
re-derived and an autocorrelation-corrected standard error**, plus an explicit decision on the
intraday fill model. This is the build that converts an effectively impassable gate into a passable
one within a quarter.

**Phase 4 — the runtime move.** §8. Independent of the others; can run in parallel with any of them.

**Phase 5 — the structural five.** §6.2. The lane unification is the largest and the riskiest; it
wants a quiet period and both behavioural suites green at every step.

### 9.1 The forecast, stated plainly

With every block cleared and many strategies ticking in parallel from tomorrow, the first
`forward_tested` **on the daily contract** is not before roughly 2026-12, and only for a strategy
running a realized Sharpe near 3.7 at the sixty-three-observation floor — or later, at a saner
observation count. A `forward_tested` winner inside sixty days is out of reach on the daily contract
regardless of any code change.

The intraday contract is what makes the horizon a quarter instead of two years. That is the honest
case for building it, and it is the reason it now leads the roadmap.

---

## 10. Changes this document proposes to the PRD

None are made here; all require the operator's edit.

1. **§10 roadmap order.** Step 7 (intraday) moves ahead of steps 2, 4, 5 and 6. Step 2 (the data
   purchase) is no longer blocking: the current snapshot yields 536 holdout observations against a
   floor of 63, so depth is adequate and the purchase now buys *breadth and regime variety* rather
   than statistical power.
2. **§10 step 7 rationale.** "63 hourly observations is two weeks" should be replaced. The correct
   rationale is 6.5 observations per calendar day, reaching a passable observation count in a
   quarter rather than in two years.
3. **§4, the forward gate description.** "63 broker-clocked observations" reads as a pass condition
   and is a floor before evaluation. Worth one sentence saying the binding wall is the confidence
   bound, and that a marginal strategy's remedy is a longer window.
4. **§7, division of labour.** The four human steps found inside the funnel were all ops omissions
   rather than sanctioned duties, and three are now cleared. The list did not grow; it is worth
   recording that it was tested and held.
