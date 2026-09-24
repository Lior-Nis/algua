# Vision reconciliation — 2026-09-24

**Purpose:** translate the [Vision of Record](PRD.md) into consistent repository guidance and
backlog direction. This is an implementation-gap record, not another product vision or a grant
of operating authority. Evidence below was inspected at base commit
`d9ed65455a626d3120559eca30fc0ed0818f9084`; it is not an inspection of funded accounts or deployed services.

## Authority and document roles

| Document | Authority / use |
|---|---|
| [PRD](PRD.md) | Canonical product objectives, constraints and eight-phase development sequence; the owner's supplied 30 sections are preserved verbatim. |
| [Architecture](architecture.md) | Current packages, extension seams and implementation boundaries. |
| [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md), [operating guide](agent/operating.md) | Current review and operating rules; narrower worker permissions remain binding. |
| [Contracts](contracts/bar-schema.md), code, tests and enforced policies | Current interface and safety behavior. A conflict with the vision requires an explicit implementation change through existing review. |
| [Dated specs and plans](superpowers/README.md) | Design history and scoped technical rationale; old roadmap and product assumptions do not override the current PRD. |
| [Decision log](.decision-log.md) | Owner decisions and reconciliation audit, including superseded policy. |

The vision's priority order is a product tradeoff order within its capital, integrity and human
authority constraints. It does not permit trading away an approval wall for expected profit.
Its conceptual journey through shadow/paper/experimental/scaled live is not a change to the
current `Stage` enum. Research workers stop at `candidate`; gated operational commands can
reach `forward_tested`; real-money activation still requires the signed human ceremony.

## Current capability versus target

| Area | Verified current state | Target / next work |
|---|---|---|
| Operating kernel | Data snapshots, backtests, gates, execution, paper/live tick engines, risk and operator packages exist. | Demonstrate one strategy's safe end-to-end lifecycle; package presence is not acceptance evidence. Phase 1. |
| Artifact reproducibility | The pure planner seam and explicit append-only deployment epochs are shipped. New paper intake records a canonical working-tree/environment descriptor; ticks, forward evidence and certificates bind to its deployment ID. Existing migration-time tenants remain an explicit unmigrated cohort. | Execution still imports the mutable working tree and halts on drift. Materialize a recoverable immutable planner artifact next, then bind signed live authorization to the exact deployment under [#661](https://github.com/Lior-Nis/algua/issues/661). Phase 1, PRD §§5, 7, 10. |
| Experiment memory | `tracking/`, `knowledge/`, idea outcomes and strategy/family notes exist. `Settings.tracking_backend` defaults to `mlflow-sqlite`. | Complete the durable experiment graph, failed-result coverage, prior-search requirement and knowledge synthesis. Phase 2, PRD §9. |
| Data and hourly operation | Bars, PIT universes and fundamentals/news storage exist. `data/capabilities.py` admits US equities and daily-family research horizons; supported storage is not proof of usable live feeds. | Deep PIT OHLCV, fundamentals, timestamped news/filings and complete hourly operation. Phase 3; [#625](https://github.com/Lior-Nis/algua/issues/625), [#630](https://github.com/Lior-Nis/algua/issues/630), and live alternative-data gap [#472](https://github.com/Lior-Nis/algua/issues/472). |
| Autonomous engineering | Telemetry, operator machinery and research merge-back exist. `operator/diff_policy.py` permits strategy Python files and `kb/` under its deny rules. | Full incident → reproduction → test → fix → independent review → permitted release → verification loop. General core-repair merges/deployments are not enabled by PRD adoption. Phase 4. |
| Portfolio | Construction, allocations, exposures, overlays and account risk controls exist. | Validate weak correlation and incremental contribution, then allocate within approved limits. Phase 5. |
| ML / LLM strategies | Model-related contracts and research paths exist; provider/inference support must be evaluated per feature. | Broader evaluated model capabilities, simpler baselines and reproducible prompt/input/output provenance. Phase 6; [#627](https://github.com/Lior-Nis/algua/issues/627). |
| Capital and market breadth | Initial vision scope is personal capital in long-only stocks and unleveraged ETFs. | Evidence-led compatible external capital in Phase 7; economically justified additional markets in Phase 8. No automatic purchases, funding or broker activation. |
| Unattended operation | Docker research runs, systemd workers, health commands and safety controls exist. | Demonstrate the PRD §20 72-hour target with predefined safe states and auditable recovery. This review did not run that acceptance exercise. |

Hourly research and execution are part of the first complete architecture. A daily strategy may
prove the first lifecycle slice before hourly completion. More bars alone do not establish more
independent evidence, justify a lower gate, or promise a fixed forward-validation duration.

## Important gaps requiring separate implementation review

These findings are **flag-only** in this documentation change. Current controls remain binding.

1. **Experimental-account loss policy — Important.**
   `algua/config/settings.py:79` defaults `book_max_drawdown` to **0.15**; this is not the new
   **10%** intervention rule. `algua/risk/book_breaker.py:79` uses a strict below-threshold
   comparison. `algua/cli/live_cmd.py:506` halts the book and attempts account-wide closing.
   Before treating PRD §21 as enforced, specify the equity/high-water reference, cash-flow
   treatment, exact threshold boundary, outstanding-order/position handling and human-reviewed
   resumption; implement and test through safety review. Owner: Lior with the operating-kernel
   implementer, before experimental live acceptance. Do not silently change runtime settings.
2. **Immutable execution — Important.**
   The planner seam and working-tree deployment/epoch slice are implemented, but a deployment
   descriptor is evidence, not recoverable executable content. Preserve the frozen decision/current
   supervisor split. Artifact materialization, signed deployment identity and any narrowing of the
   approved code closure require the existing protected review. Owner: #661 implementer;
   prerequisite for immutable-artifact acceptance.
3. **Legacy signed relaxations — Important.**
   The old PRD tied signed research/forward relaxations to dollar capital rungs. Those rungs
   are superseded. Existing authenticated commands remain as implemented; the new vision
   neither authorizes a shortcut nor silently removes a mechanism. Owner: Lior, with #624,
   must settle their experimental-capital policy before using one for live qualification.
4. **Authority and deployment protection — Important.**
   The autonomous diff gate reads root `CODEOWNERS`; both root `CODEOWNERS` and
   `.github/CODEOWNERS` exist with different coverage. This reconciliation does not establish
   effective GitHub branch protection or deployed filesystem ownership. Verify owner-review
   coverage and an immutable installed signing anchor before relying on them for autonomous
   releases. Owner: Lior/deployment implementer, before Phase 1 live acceptance or Phase 4
   permission expansion. Preserve existing controls; do not edit protection files here.
5. **Initial capital and instrument constraints — Important.**
   `algua/config/settings.py:65` defaults `book_max_gross` to **2.0** (`book_max_net` is 1.0).
   Existing generic exposure limits do not establish PRD §21's long-only, unleveraged ETF,
   no-borrowing, no-derivatives policy or the approved ₪2,000 experimental capital ceiling.
   Verify eligible instruments, account funding/buying-power semantics and enforced sizing
   limits before experimental live acceptance. Owner: Lior and the #624 implementer; separate
   safety review, no runtime-default changes in this documentation pass.
6. **Inherited calendar-purity wording — Important.**
   `AGENTS.md` §3 retains the supplied rule that calendar imports no other Algua modules.
   The shipped import-linter rule instead keeps calendar independent of CLI and registry;
   `calendar.get_calendar()` uses settings, as the current architecture describes. This is an
   existing specification/enforcement mismatch, not a new permission. Owner: Lior, before
   work changes calendar boundaries, must reconcile the intended invariant with the shipped
   contract. This pass neither weakens the supplied invariant nor changes code/import policy.

The 90-day target measures an end-to-end operating machine. It does not waive research evidence,
forward qualification or human live approval if a strategy cannot qualify within that time.

## Backlog migration

Issue numbers are retained so links, discussions and design history survive. The phase mapping
below changes product priority and scope, not issue completion state or deployed configuration.

| Existing issue | Reconciled scope | Admission / dependency |
|---|---|---|
| [#624](https://github.com/Lior-Nis/algua/issues/624) | Phase 1 operating kernel and experimental personal live acceptance | Replace dollar rungs and stale account assertions; require actual readiness, #661, the 10% policy review and authenticated human activation. |
| [#625](https://github.com/Lior-Nis/algua/issues/625) | Phase 3 deep PIT US-equity/ETF data, daily/hourly | OHLCV, fundamentals, timestamped news/filings; no vendor purchase commitment. Shared expense ceiling is up to ₪2,000. |
| [#627](https://github.com/Lior-Nis/algua/issues/627) | Phase 6 evaluated ML/DL/LLM strategy capabilities | Phase 1 artifact/environment provenance remains foundational. SQLite tracking already exists; avoid reviving its resolved file-store question. |
| [#628](https://github.com/Lior-Nis/algua/issues/628) | Phase 8 optional additional market data | A specific economic hypothesis and justified data cost precede adapters/calendars/instrument work. Four mandatory lanes are superseded. |
| [#629](https://github.com/Lior-Nis/algua/issues/629) | Phase 8 conditional crypto execution | Deferred beyond initial stocks/ETF scope; requires compatible evidence, risk/capital policy and human live authorization. |
| [#630](https://github.com/Lior-Nis/algua/issues/630) | Phase 3 hourly operation, anticipated by Phase 1 interfaces | Closed-bar timing, costs, reconciliation, staleness and evidence must be cadence-correct. No bar-count-to-calendar validation promise. |
| [#661](https://github.com/Lior-Nis/algua/issues/661) | Phase 1 immutable deployed decision artifacts | Planner extraction and explicit working-tree epochs are implemented. Materialize and execute recoverable immutable planner content next; retain separate protected review for signed deployment binding. |

Phases 2, 4, 5 and 7 also require scoped acceptance work; this mapping does not claim that seven
legacy issues exhaust the new roadmap. Each future issue must state its contribution under
PRD §28 and distinguish implemented behavior from proposed work.

## Historical reference map

The [September 6 PRD](https://github.com/Lior-Nis/algua/blob/d9ed65455a626d3120559eca30fc0ed0818f9084/docs/PRD.md)
remains in Git history. Its old section numbers must not be interpreted as current anchors.

| Historical source | How to read it now |
|---|---|
| May 29 architecture and foundation plan | Original rationale and build history; old TOTP, lifecycle, absent-module and six-project statements are historical. Use current architecture and operating guides. |
| September 8 ideation design/plan | Technical history; new product anchors are PRD §§3, 8–10, 25. Volume is supporting evidence; retained learning and validated profit are outcomes. |
| September 10 overlays design | Tighten-only contract remains; anchors move to PRD §§7, 10, 13–15. |
| September 12 holistic verdict | Dated diagnosis. Its roadmap override and quarter-versus-years validation projections are superseded; observations are not current account state. |
| September 22 artifact-freeze design | Aligned Phase 1 direction, PRD §§5, 7, 10, 24–25; retain explicit partial status. |

## Completion evidence

- All 30 canonical sections preserved; an independent manual comparison found no textual
  divergence from the supplied vision. Conversational introduction and citation placeholders
  are excluded. The file also matches the source text used for the replacement exactly.
- Current entry points, research guidance, categories and inline source-documentation citations
  reconciled. Python changes are comments/docstrings only, verified by comparing syntax trees
  with docstrings removed; no executable behavior changed.
- Local Markdown destinations checked. Historical source notices and the design-history index
  point to the current vision; old PRD numbers remain explicitly historical.
- Issue bodies updated and read back successfully: #624, #625, #627, #628, #629, #630, #661.
  Historical reference wording corrected in #633–#639, preserving their technical findings.
  All 14 issues remain open. Roadmap issues retain their prior bodies as historical context;
  #661 retains its technical slices. Updates identify this branch as pending merge.
- `uv run pytest -q`: **3,985 passed**, 171 warnings, 492.72 seconds.
- `uv run ruff check .`: passed. `uv run mypy algua`: passed, 292 source files.
  `uv run lint-imports`: **28 kept, 0 broken**. These checks were run after pytest completed;
  running lint during pytest briefly encountered a test-created temporary strategy module.
- Whitespace checked with Markdown hard-break trailing spaces allowed, preserving the owner's
  exact two-space line breaks in the PRD metadata.
- [Review](review-vision-reconciliation.md): the capital-policy omission was added above;
  completion bookkeeping is recorded here. Current safety and authority controls remain binding.

This pass does not certify live readiness or execute trading operations.
