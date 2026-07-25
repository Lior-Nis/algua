# #529 — promotion-gate recalibration: task list (round 4, post-GATE-1)

Design: `docs/superpowers/specs/2026-07-25-promotion-gate-recalibration-529-design.md`.
Approach (round 4, after GATE-1 BLOCKed round 3 for quantifying-but-not-ENFORCING the cumulative-
exposure budget): keep every round-3 gain — (i) the γ-normalization bug fix (γ over the restart
horizon N, not 10 000), (ii) `FDR_W0` LEFT at the standard `FDR_ALPHA/2 = 0.025` (round-2's doubling
REVERTED), (iii) `FDR_COHORT_SIZE = 8` DERIVED from an explicit near-term cumulative-exposure budget
(`S(16 attempts) ≤ 5%`) — AND ADD the missing enforcement: (iv) a **hard, schema-free windowed
PROMOTION-ELIGIBILITY throttle** (Codex path (a) + option C) that caps PROMOTION-ELIGIBLE binding FDR
tests at `FDR_NEAR_TERM_BINDING_BUDGET = 16` per `FDR_THROTTLE_WINDOW_DAYS = 365` — the 17th+ still
COMMIT a fail-closed binding row and still advance the LORD++ stream but CANNOT promote — making the
windowed all-null FALSE-PROMOTION probability an AGENT-ENFORCED cap: rigorous worst-case `≤3·p_cohort
≈7.2%` over ANY rolling window (the 16 eligible tests touch ≤3 always-fresh cohorts; each windowed
slice dominated by its full fresh cohort — §3.5), equivalently a ≤16-eligible/window RATE cap, instead
of a human-signed hope. (NOT the overclaimed "exact 4.9% per rolling window" — a rolling window does
not align to cohort boundaries; 4.9% is only the aligned typical case.) The throttle runs inside the EXISTING `BEGIN IMMEDIATE` (no
schema, no new lock, no TOCTOU), keeps `fdr_rejected` pure LORD++ (throttle is a SEPARATE named
`final_passed` term), and has a HUMAN-ONLY #329-signed `--fdr-throttle-override` bypass (rejected on
the agent path). Keep (a) breadth-deflated Sharpe, (b) DSR, and economic/stability floors
HARD. The lifetime posture is honestly named as an intentional CODEOWNERS-signed risk acceptance
(unbounded-in-count but now rate-capped by the throttle), NOT presented as closed. New AUDIT-ONLY
partial-cohort + throttle-state fields are added; the completed-only exposure formula is unchanged.
CODEOWNERS-protected (`gates.py`, `fdr_lord.py`, `promotion.py`, `store.py`) → PR stays OPEN for human
merge.

## Task 0 — GATE-1 (Codex, read-only) on the recalibration design — DONE, PASS (round 4b)
Round-1 (α-floor), round-2 (unbudgeted cumulative exposure), round-3 (budget quantified-not-enforced),
round-4a (throttle claimed to cap committed rows — false — + dry-spell approximation), and round-4b
(claimed S(16)=4.9% EXACT per rolling window — false, windows don't align to cohort boundaries) were
all BLOCKed. The current design adopts the §3.5 windowed PROMOTION-ELIGIBILITY throttle (Codex path
(a) + option C), replaces the exact claim with a RIGOROUS worst-case valid for ANY rolling window
(≤3·p_cohort≈7.2% via dominance/union bound over ≤3 always-fresh cohorts — Codex option 2), and names
the residual lifetime posture as an explicit CODEOWNERS-signed risk acceptance. **Codex GATE-1
APPROVED** it (dominance step sound, ≤3-cohort touch correct, off-by-one correct, throttled-rows-in-
stream coherent, no hole in bypass/mint/atomicity). Two non-blocking GATE-2 watchpoints folded into
the design/tests: (1) state the standard LORD++ null-p-value/predictability assumption behind
`p_cohort`; (2) word the throttle as a per-decision cap on PRIOR committed in-window rows, not a
retrospective-window property. Impl (Tasks 1-4) is unblocked.

## Task 1 — recalibrate LORD++ constants + γ normalization + throttle constants (fdr_lord.py)
- `algua/research/fdr_lord.py`: `FDR_COHORT_SIZE = 8` (was 64); `FDR_W0` UNCHANGED at `FDR_ALPHA/2`;
  **ADD** `FDR_NEAR_TERM_BINDING_BUDGET = 16` and `FDR_THROTTLE_WINDOW_DAYS = 365` (with the
  §3.1/§3.5 derivation in the docstring — the cap MUST equal `H_near`); normalize `_compute_lord_
  gamma` / `_LORD_GAMMA` over `FDR_COHORT_SIZE` terms; **REMOVE `FDR_GAMMA_TRUNCATION`** (vestigial)
  and drop its re-export in `gates.py` + its test in the SAME PR. Rewrite the `FDR_COHORT_SIZE`
  rationale block to the round-4 story (γ-normalization bug fix; budget-derived N=8 under the near-term
  `S(16)≤5%` budget with the §3.1 table; the ENFORCING throttle §3.5; the honest per-cohort rescope +
  rate-cap §3.4; LORD++-assumptions/operating-target caveat). `lord_plus_plus_level` math unchanged.
- `gates.py`: re-export the two new throttle constants; drop the `FDR_GAMMA_TRUNCATION` re-export.
- FAST check: `ruff + mypy + lint-imports` whole tree; `pytest -k "fdr or lord or promotion"`.

## Task 2 — the hard windowed throttle + audit fields (store.py + promotion.py)
- `store.py`: ADD `_windowed_binding_test_count(window_days)` — COUNT PRIOR `fdr_binding=1` rows with
  `created_at >= now − FDR_THROTTLE_WINDOW_DAYS`; call it INSIDE the existing `BEGIN IMMEDIATE` in
  `record_gate_with_fdr_and_maybe_promote`, compute `promotion_eligible = count <
  FDR_NEAR_TERM_BINDING_BUDGET or fdr_throttle_override`, and AND it into `final_passed`
  (`final_passed = provisional_passed and fdr_rejected and promotion_eligible`). The throttled row
  STILL commits `fdr_binding=1` and STILL advances the LORD++ stream (it is a PROMOTION throttle, not
  a row/stream cap) — only `final_passed` goes False. `fdr_rejected` STAYS pure LORD++ (unchanged).
  Surface a `fdr_throttle` `checks[]` entry (value=count, threshold=budget, op `<`,
  passed=promotion_eligible) + `fdr_throttle_tripped`/`fdr_throttle_window_binding`
  /`fdr_throttle_override` audit fields.
- `store.py`: accept new `fdr_throttle_override: bool = False` param threaded from the promote path.
- `store.py` audit (§4): `_read_fdr_stream` also returns the in-progress cohort's applied-α sum +
  within-cohort position; write `fdr_active_cohort_position`, `fdr_active_cohort_applied_alpha`,
  `fdr_expected_false_discoveries_incl_active = FDR_ALPHA·cohorts_completed + applied_alpha`. LEAVE
  the completed-only `fdr_expected_false_discoveries` and the `fdr_rejected` formula UNCHANGED. NO
  schema change (all in `decision_json`).
- `promotion.py` + `research promote` CLI: thread the HUMAN-ONLY `--fdr-throttle-override` flag
  through to `record_gate_with_fdr_and_maybe_promote`; REJECT it on the agent path and bind it to the
  #329 signed run (mirror `--allow-non-pit`/`--allow-holdout-reuse` handling).
- FAST check: `pytest tests/test_registry_store.py tests/test_promotion.py -k "fdr or throttle or
  exposure or cohort or override"`.

## Task 3 — tests (update, do not delete; add throttle + budget + adversarial + regression + audit)
Per design §7: retune `test_fdr_constants` (W0 UNCHANGED, COHORT_SIZE==8, plus the two throttle
constants), the γ-property tests (Σ=1 over N=8), the accept/reject binding cases (α_1≈0.00764), and
cohort-boundary tests referencing 64. Keep skip cases. ADD:
- `test_lord_cohort_spends_full_budget` (Σα_t==FDR_W0 + pin α_1≈0.00764/α_2≈0.00382),
- `test_lord_all_null_first_discovery_probability` (per-cohort ≤ FDR_ALPHA/2+ε — ≈2.5% not 72%),
- `test_lord_retry_surface_within_near_term_budget` (aligned S(16)≤5% at N=8, RIGOROUS worst-case
  `1−(1−p_cohort)^3 ≤ 0.08` with `max_touched_cohorts==ceil(H_near/N)+1==3`, AND `H_near ==
  FDR_NEAR_TERM_BINDING_BUDGET` consistency — locks the §3.1/§3.5 cap + worst-case ceiling to the
  arithmetic),
- **`test_fdr_throttle_blocks_promotion_beyond_budget`** (after `FDR_NEAR_TERM_BINDING_BUDGET` binding
  rows in-window, the next otherwise-passing binding eval → `final_passed=False`,
  `fdr_throttle_tripped=True`, NO stage advance, NO mint; BUT it STILL commits `fdr_binding=1` with
  true `fdr_rejected` and STILL advances the stream — a PROMOTION throttle; 16th promotes, 17th does
  not; an out-of-window row does NOT count),
- **`test_fdr_throttle_human_override_promotes`** (over-budget + `fdr_throttle_override=True` promotes;
  override REJECTED on the agent path),
- `test_run_gate_fdr_recalibration_promotes_previously_unpassable` (dsr≈0.995/p≈0.005 promotes now,
  within budget; was rejected under α≈0.00165),
- `test_fdr_active_cohort_exposure_surfaced` (§4 audit fields; completed-only field unchanged),
- a #524-composition assertion (pending-NOVEL failing (c) OR tripping the throttle is NOT minted;
  clearing all hard checks within budget IS).
Confirm `test_registry_store.py` completed-only exposure asserts STAY green.
- FAST check: `pytest tests/test_research_gates.py tests/test_promotion.py tests/test_registry_store.py`.

## Task 4 — docs + integration + PR
- `CLAUDE.md`: update the `research promote` LORD++ paragraph (cohort recalibrated to 8 under an
  explicit near-term cumulative-exposure budget; γ over the restart horizon; W0 unchanged at
  FDR_ALPHA/2; the budget is ENFORCED by a hard windowed throttle ≤16 binding tests / 365d with a
  human-only signed `--fdr-throttle-override`; per-cohort FDR still an operating target at FDR_ALPHA;
  lifetime exposure now RATE-CAPPED, not unbounded-at-burst; per-test α now O(0.5–0.8%), ~4.6× the old
  first-test level).
- Note #524 ordering (disjoint region in the same `BEGIN IMMEDIATE`; second-to-merge rebases).
- PR body MUST record explicit human sign-off on (1) the chosen `FDR_COHORT_SIZE ∈ {6,8,12}`, (2) the
  throttle window/cap (`FDR_THROTTLE_WINDOW_DAYS=365` / `FDR_NEAR_TERM_BINDING_BUDGET=16`) and the
  agent-ENFORCED per-window worst-case false-promotion surface `≤3·p_cohort≈7.2%` (typical aligned
  4.9%), and (3) the intentional CODEOWNERS-signed risk acceptance
  of the residual unbounded-in-count-but-rate-capped LIFETIME exposure (§3.4/§9) — all with the
  §3.1/§3.2 table in view.
- FULL gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Open PR (stays OPEN — CODEOWNERS). GATE-2 (Codex) on the whole diff before merge.
