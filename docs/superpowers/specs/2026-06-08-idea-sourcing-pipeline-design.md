# Structured agentic idea-sourcing pipeline (#126)

**Status:** design — GATE-1 reviewed (Codex + Gemini), triaged; pending user sign-off
**Issue:** #126 — widen top-of-funnel with diversity + discipline, not raw volume
**Date:** 2026-06-08

## Problem

The top of the research funnel is endogenous. The only entry points are manual
(`registry add` / `strategy new`) and the research loop's "ideate" step — a pure
prompt with no code, where the agent reads `kb/strategies/_index.md` + `_families.md`
and invents a hypothesis as a variation on a family that already exists. Nothing
sources ideas externally (papers, filings, forums), and the `deep-research` skill is
not wired into the funnel. Ideation explores only the neighborhood of what's already
there.

## Goal — diversity + discipline, not raw volume

"Make the funnel bigger to raise conversion" is a fallacy: more raw ideas through the
same gates mechanically *lowers* pass rate and yields more *false* survivors, because
the promotion gate deflates the holdout-Sharpe bar by search breadth (multiple-testing
defense). The real lever is **idea diversity** (external priors widening the hypothesis
space) under **breadth discipline** (the gate stays honest as the pool grows). This
slice builds the diversity machinery and the discipline scaffolding, and is deliberately
sequenced so the diversity is not switched to autopilot before the discipline exists
(see "Breadth-sequencing resolution").

## Scope

### In scope
1. A separate `ideas` table (registry sqlite DB, schema v18 → v19) — the structured pool,
   with a real FK to `strategies`.
2. Pure `Idea` contract + `IdeaStatus`/`SourceType`/`DataCapability` enums and an
   `IdeaStatus` transition state machine.
3. `needs_data` parking, derived from a controlled `DataCapability` vocabulary whose
   *supported* subset is owned by `algua/data`.
4. Two-layer dedup: a deterministic fail-closed CLI gate (enriched signature, null-family
   safe, refuted-aware via FK join) + agent semantic judgment.
5. `uv run algua research idea ...` CLI surface (add / list / show / dedup-check / set-status).
6. A funnel-breadth signal (idea counts **by status**) exposed but **not** wired into the gate.
7. A standalone `source-ideas` skill (deep-research → dedup → `research idea add`), available
   for deliberate use but **not** wired in as the research loop's default ideation step yet.

### Out of scope (deferred, with rationale)
- **The gate-formula change.** Folding the funnel-breadth signal into `gates.py`'s
  `sharpe_haircut` / `effective_funnel_breadth` is a statistical-design decision on a
  CODEOWNERS-protected file → a separate human-designed PR. This slice only *exposes*
  the signal.
- **Auto-wiring `source-ideas` into the research loop** as the default top-of-funnel step.
  Deferred until the gate consumes the breadth signal (see resolution below) — closing
  the "widen before counting" hazard.
- **Re-checking parked ideas when new data lands** (auto-flip `needs_data → open`). `set-status`
  allows manual flips meanwhile.
- **Propagating a strategy's refute into the idea's *display* status.** Dedup resolves
  refuted-ness via the live FK join (authoritative); refreshing the stored idea status is
  cosmetic and would touch the protected transition path — deferred.
- **No auto-promotion.** Sourced ideas enter the pool and climb the normal gated ladder.
- **Embedding-based dedup** and a **full structured-hypothesis taxonomy** (mechanism/
  universe/horizon/cadence/direction as required fields) — coarse lexical+structural dedup
  plus the agent layer is enough now; no new model dependency.

## Design decisions (brainstorming + GATE-1)

| # | Fork | Decision |
|---|------|----------|
| 1 | Cross-funnel breadth accounting in `gates.py` | **Defer** the gate change; expose the signal only. |
| 2 | Where idea records live | **Separate `ideas` table**; registry `strategies` untouched, linked by FK. |
| 3 | How `deep-research` connects | **Deterministic CLI store + agent-driven `source-ideas` skill**; no LLM in the CLI. |
| 4 | Dedup mechanism | **Deterministic CLI gate (fail-closed) + agent semantic judgment.** |
| A | "satisfiable data" source | **Derive from platform-supported data capabilities** (owned by `algua/data`), not loaded snapshots. |
| B | Sourcing playbook location | **Standalone `source-ideas` skill.** |
| C (GATE-1) | Breadth-sequencing hazard | **Collection-only this slice** — build + expose, do NOT auto-wire into the loop until the gate counts ideas. |
| D (GATE-1) | "Never refill a refuted idea" depth | **Enforce via FK + live join** in the deterministic dedup (no protected-file change). |

## Breadth-sequencing resolution (GATE-1 CRITICAL, both models)

Shipping a funnel-widener while `gates.py` is blind to idea volume is the exact failure
the issue forbids: an agent could source N ideas, author the best-looking one, and face a
gate calibrated only to parameter-sweep breadth — letting lucky false winners through.
Since the gate change is deferred (CODEOWNERS), the guard is **sequencing, not gate math**:

- Build the full pipeline (pool, dedup, backlog, provenance, `source-ideas` skill) and
  **expose** idea counts by status.
- Do **not** rewire `run-the-research-loop` to use `source-ideas` as its default ideation
  step. The skill is usable deliberately; the funnel is not put on autopilot.
- The "flip to default" switch lands in the later human PR that teaches the gate to count
  ideas. That PR consumes the signal this slice exposes.

Net: all infrastructure ships now; the unsafe autopilot does not precede the safety net.

## Data model — `ideas` table (schema v18 → v19)

New table in the existing registry sqlite DB. The `strategies` table and its repository
(`store.py`, protected) are **unchanged**.

```
ideas:
  id                  INTEGER PRIMARY KEY AUTOINCREMENT
  title               TEXT NOT NULL
  hypothesis          TEXT NOT NULL            -- the claimed edge, in prose
  family              TEXT                     -- thesis-family slug; NULL/"unknown" => compared against ALL (see dedup)
  tags                TEXT NOT NULL            -- json list (default "[]")
  source_type         TEXT NOT NULL            -- paper|url|forum|filing|thesis|manual
  source_ref          TEXT                     -- url / citation / doi
  source_date         TEXT                     -- ISO date of the source (recency signal); nullable
  source_note         TEXT                     -- optional one-line why-it's-credible
  required_data       TEXT NOT NULL            -- json list of DataCapability values (validated)
  status              TEXT NOT NULL            -- open | needs_data | authored | refuted | discarded
  signature           TEXT NOT NULL            -- normalized dedup signature (stored)
  authored_strategy_id INTEGER                 -- FK -> strategies(id); set when authored
  duplicate_of_idea_id INTEGER                 -- FK -> ideas(id); set when added via --allow-duplicate
  override_reason     TEXT                     -- dedup-override audit (paired with duplicate_of_idea_id)
  created_at          TEXT NOT NULL            -- ISO8601
  updated_at          TEXT NOT NULL            -- ISO8601
  FOREIGN KEY (authored_strategy_id) REFERENCES strategies(id)
  FOREIGN KEY (duplicate_of_idea_id) REFERENCES ideas(id)
```

Migration: a `v18 → v19` step in `db.py` creating the `ideas` table (no backfill).

## Contracts (`algua/contracts/idea.py`, pure)

```python
class IdeaStatus(StrEnum):
    OPEN = "open"            # implementable now (needs only supported capabilities)
    NEEDS_DATA = "needs_data"  # parked: needs a capability the platform can't provide yet
    AUTHORED = "authored"    # promoted into a registered strategy
    REFUTED = "refuted"      # rejected; a dedup sentinel
    DISCARDED = "discarded"  # dropped without testing

class SourceType(StrEnum):
    PAPER = "paper"; URL = "url"; FORUM = "forum"
    FILING = "filing"; THESIS = "thesis"; MANUAL = "manual"

class DataCapability(StrEnum):
    OHLCV = "ohlcv"                 # price/volume bars — SUPPORTED today
    FUNDAMENTALS = "fundamentals"
    FORM_13F = "form_13f"
    OPTIONS_FLOW = "options_flow"
    DARK_POOL = "dark_pool"
    FORM_4 = "form_4"               # insider transactions

# Allowed IdeaStatus transitions (enforced by set-status). open<->needs_data;
# open/needs_data -> authored|discarded; authored -> refuted; * -> discarded.
ALLOWED_IDEA_TRANSITIONS: dict[IdeaStatus, set[IdeaStatus]] = { ... }

@dataclass
class Idea:
    id: int; title: str; hypothesis: str
    family: str | None; tags: list[str]
    source_type: SourceType; source_ref: str | None
    source_date: str | None; source_note: str | None
    required_data: list[DataCapability]
    status: IdeaStatus; signature: str
    authored_strategy_id: int | None
    duplicate_of_idea_id: int | None; override_reason: str | None
    created_at: str; updated_at: str
```

The `DataCapability` vocabulary is the controlled set that makes `needs_data` meaningful
(no `13f`/`form_13f`/`filings_13f` fragmentation). `required_data` is validated against it
in `add` (unknown tag → error).

## Platform capabilities → needs-data parking

`algua/data/capabilities.py` (data layer owns the *supported subset*):

```python
def supported_capabilities() -> frozenset[DataCapability]:
    """DataCapability values the platform can actually provide to a backtest, derived from
    the data layer's dataset support. "Supported" = an ingestion/serving path EXISTS, not
    "a snapshot is loaded" (demo mode provides ohlcv with no snapshot). Today: {OHLCV}."""
```

Status classification is a **pure** function in `algua/research/ideas.py`:

```python
def classify_status(required: list[DataCapability], supported: frozenset[DataCapability]) -> IdeaStatus:
    return IdeaStatus.OPEN if set(required) <= supported else IdeaStatus.NEEDS_DATA
```

The CLI gathers `supported_capabilities()`, calls `classify_status`, stores the result — so
`registry` never imports `data`.

**Semantics note (GATE-1 HIGH):** `open` means *implementable by the platform* (the data
kind has an ingestion path), NOT *"a covering snapshot exists / is empirically testable now."*
Real-data readiness — snapshot coverage of the requested universe/dates and point-in-time
soundness — is already enforced downstream by the **existing promotion PIT gate** (shipped
in c26dd26: PIT-by-default, fails closed). The idea pool deliberately does not duplicate
that wall; a second status axis was considered and declined as redundant.

## Dedup / novelty — two layers

**Layer 1 — deterministic CLI gate** (`algua/research/idea_dedup.py`, pure):
- `signature(title, hypothesis) -> str`: lowercased, stopword-stripped, sorted token set.
- A **blocking key** combines `family` + sorted `required_data` to scope comparisons.
- `find_collisions(candidate, existing, *, threshold=0.6) -> list[Idea]`: collision when the
  blocking keys are comparable AND `jaccard(signature) >= threshold`.
- **Null/unknown family is fail-safe (GATE-1 HIGH):** a NULL/`unknown` family on either side
  does NOT suppress comparison — it compares against ALL families (stricter), so a missing
  family can never silently hide a collision.
- **Refuted-aware (GATE-1 CRITICAL):** the oracle includes (a) `status=refuted` ideas and
  (b) ideas whose `authored_strategy_id` points to a strategy currently
  `hypothesis_status=refuted` — resolved by a **live join** in `IdeaRepository`, so a refuted
  *strategy* blocks its idea's near-duplicates without mutating idea rows or touching the
  protected transition path.
- `research idea add` **fails closed** on collision (non-zero exit + `collisions` payload).
  A **refuted** collision (a `status=refuted` idea, or one whose linked strategy is now refuted)
  is NOT overridable — it fails closed unconditionally (GATE-2 HIGH). Only non-refuted
  collisions may be overridden with `--allow-duplicate --reason "..."`, recording
  `duplicate_of_idea_id` + `override_reason`.

**Layer 2 — agent semantic judgment** (the `source-ideas` skill): reads kb `_index`/`_families`
+ the near-misses from `dedup-check`; only overrides for a genuinely new angle, with a reason.

**Guarantee level (documented):** the deterministic wall is exact for anything that flowed
through the pool — duplicate ideas, `status=refuted` ideas, and ideas whose FK-linked strategy
is now refuted (all blocked, the refuted ones unconditionally). It does NOT string-match against
*legacy* strategies that never had an idea row (authored before the pool, or via `registry add`/
`strategy new` directly): their hypothesis text lives in kb docs, not the registry DB, and a
`name + description` match on sparse slug data would be noise, not a wall. Kb-wide novelty for
those is **layer 2's job** (the agent reads `_index`/`_families`, which list refuted strategies).
This residual is explicit, not silent — see GATE-2 triage finding #2.

## CLI surface (`algua/cli/idea_cmd.py`, registered under `research`)

```
uv run algua research idea add \
    --title T --hypothesis H [--family F] \
    --source-type paper --source-ref URL [--source-date D] [--source-note N] \
    [--tag t1 --tag t2] [--required-data ohlcv,form_13f] \
    [--allow-duplicate --reason R]
uv run algua research idea list [--status STATUS] [--family F]      # includes counts-by-status summary
uv run algua research idea show <id>
uv run algua research idea dedup-check --title T --hypothesis H [--family F]   # preflight, no write
uv run algua research idea set-status <id> --to authored --strategy NAME       # state-machine + actor checked
```

- All emit JSON via the existing `emit`/`ok` envelope.
- `add`: validates `required_data` against `DataCapability`, derives `status`, runs the dedup
  gate. Collision → `{"ok": false, "error": ..., "collisions": [...]}` non-zero, unless `--allow-duplicate`.
- `set-status`: enforces `ALLOWED_IDEA_TRANSITIONS`; `--to authored --strategy NAME` resolves
  NAME → `strategies.id`, validates it exists, sets `authored_strategy_id`.

## Breadth signal (exposed, NOT wired into the gate)

`IdeaRepository.windowed_idea_counts(window_days) -> dict[IdeaStatus, int]` — counts by status
in the rolling window (created/open/needs_data/authored/refuted/discarded), not one conflated
number, so the later gate change can choose the right denominator (likely authored/tested, not
raw scraped). Surfaced in `research idea list`. `gates.py` is **untouched**.

## Module layout & import boundaries

| Module | Responsibility | Imports | Protected? |
|--------|----------------|---------|-----------|
| `algua/contracts/idea.py` | `Idea`, enums, transition map (pure) | stdlib | no |
| `algua/data/capabilities.py` | `supported_capabilities()` | `algua/data` | no |
| `algua/research/idea_dedup.py` | signature / jaccard / blocking / find_collisions (pure) | contracts | no |
| `algua/research/ideas.py` | `classify_status` (pure) | contracts | no |
| `algua/registry/ideas.py` | `IdeaRepository` (sqlite): add/list/get/set_status/counts + refuted live-join | contracts; shares db.py | no (new file) |
| `algua/registry/db.py` | schema v19 migration (adds `ideas` table) | (existing) | no |
| `algua/cli/idea_cmd.py` | `research idea ...`; orchestrates data+research+registry | data, research, registry, contracts | no |

**No CODEOWNERS-protected file is modified.** Protected files are `store.py`, `lifecycle.py`,
`backtest/engine.py`, `gates.py`, `approvers/`, `live_gate.py`, `transitions.py`, `promotion.py`
— none appear above. The refuted wall uses a read-only join, not a write into the transition
path. `registry` does not import `data` (capability derivation happens at the CLI seam).

## Sourcing skill (`source-ideas`)

A standalone skill that drives the agent: run `deep-research` on a topic/family → extract
structured hypotheses (title, hypothesis, family, provenance, required-data) → `research idea
dedup-check` each, consulting kb for novelty → `research idea add` survivors (needs-data ones
auto-park) → pick `open` ideas to author via the existing loop. **Not** wired as the research
loop's default ideation step (see breadth-sequencing resolution); `run-the-research-loop` is
left unchanged this slice.

## Testing

- **Pure:** `idea_dedup` (signature normalization, jaccard, blocking, null-family stricter
  compare, refuted/duplicate collisions); `classify_status` (subset → open/needs_data);
  `supported_capabilities` (== {OHLCV}); `ALLOWED_IDEA_TRANSITIONS` legality.
- **Repository:** ideas CRUD; v18→v19 migration; `windowed_idea_counts`; `set_status` legality +
  authored linkage; the **refuted live-join** (idea whose linked strategy is refuted blocks a
  near-duplicate); `--allow-duplicate` override fields.
- **CLI:** add/list/show/dedup-check/set-status; fail-closed collision (non-zero + payload);
  `required_data` vocabulary validation (unknown tag → error); needs-data parking (`form_13f`
  → `needs_data`); JSON envelope shape.

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## GATE-1 review triage (Codex deep-lens + Gemini broad-lens)

| # | Finding (sev; who) | Disposition |
|---|--------------------|-------------|
| 1 | Funnel widens before breadth accounting (CRITICAL; both) | **Accept** → collection-only; don't auto-wire the loop (decision C). |
| 2 | "Never refill a refuted idea" not enforced (CRITICAL; both) | **Accept** → FK + refuted live-join in dedup (pool-linked); legacy strategies (no idea row) = agent/kb layer only (decision D). |
| 3 | Dedup key gameable/misfires (HIGH; both) | **Partial** → enrich signature/blocking (family + required_data); agent layer for paraphrase. Decline full taxonomy (gold-plating). |
| 4 | Nullable family fails open (HIGH; both) | **Accept** → null/unknown compares against ALL, never suppresses. |
| 5 | `required_data` needs controlled vocab (HIGH; both) | **Accept** → `DataCapability` enum, validated. |
| 6 | "supported" ≠ testable (HIGH; Codex) | **Accept concern, resolve by semantics** → `open`=implementable; PIT gate owns real-data readiness. Decline 2nd status axis (redundant). |
| 7 | Demo contaminates readiness (HIGH; Codex) | **Accept** → same as #6; `open` never asserts a loaded snapshot; promotion PIT gate enforces real data. |
| 8 | Soft string link / lifecycle drift (MED; both) | **Accept** → real FK `authored_strategy_id`. |
| 9 | Status transitions underspecified (MED; Codex) | **Accept** → `ALLOWED_IDEA_TRANSITIONS` + actor checks. |
| 10 | `--allow-duplicate` audit too weak (MED; Codex) | **Accept** → dedicated `duplicate_of_idea_id` + `override_reason`. |
| 11 | `windowed_idea_count` not decision-grade (MED; Codex) | **Accept** → counts by status. |
| 12 | Provenance shallow (LOW; Codex) | **Partial** → add `source_date`; defer credibility-tier taxonomy. |

## GATE-2 review triage (Codex deep-lens, on the branch diff)

| # | Finding (sev) | Disposition |
|---|---------------|-------------|
| 1 | Refuted wall overridable by `--allow-duplicate` (HIGH) | **Accept** → a refuted collision (idea or FK-linked refuted strategy) now fails closed unconditionally; only non-refuted collisions are overridable. |
| 2 | Dedup doesn't string-match legacy refuted strategies (MED) | **Resolve by correcting the spec** → the deterministic wall covers pool-linked refuted strategies (FK join); legacy strategies (no idea row) are layer-2/kb's job. A `name+description` match on sparse slug data is noise, not a wall — declined as gold-plating. |
| 3 | `set_status` persists `authored_strategy_id` on non-authored transitions (LOW) | **Accept** → reject a strategy link unless the target status is `authored`. |

## Interactions

- **#122** (registry organizational metadata) — idea records mirror `family`/`tags`; the FK
  links an authored idea to its strategy; the refuted join reads `hypothesis_status`.
- **#125** (`dormant`) — dormant re-eval and idea-sourcing both feed the same gated ladder.
- **kb `_index`/`_families`** — the agent's semantic-novelty oracle (layer 2).
- **`gates.py`** — consumes the breadth signal in a later human-designed PR (the autopilot switch).

## Deferred follow-ups (record in project memory on merge)

1. Wire `windowed_idea_counts` into the gate's breadth deflation **and** flip `source-ideas`
   to the loop's default ideation step (human, CODEOWNERS `gates.py`).
2. Re-check `needs_data` ideas when a new ingestion capability lands (auto-flip to `open`).
3. Optional: refresh an idea's stored status to `refuted` when its strategy is refuted (cosmetic;
   dedup already correct via join).
4. Deeper structured-hypothesis dedup (mechanism/universe/horizon/direction) + credibility tiers.
