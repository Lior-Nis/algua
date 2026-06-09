---
name: source-ideas
description: Source external trading-idea priors (papers/filings/forums) into the structured idea pool — run deep-research, extract hypotheses, dedup them, and `research idea add` the survivors. Use to widen the top of the funnel with diversity under discipline. NOT yet the research loop's default ideation step.
---

# Source ideas (structured external priors → idea pool)

Widen the top of the funnel with DIVERSITY under DISCIPLINE — not raw volume. You source
external priors into structured, deduped, provenance-stamped records; the CLI is the
deterministic store, you are the semantic judge.

> Collection-only: this skill populates the pool for deliberate use. It is NOT yet wired in as
> the research loop's automatic ideation step — that waits until the promotion gate counts idea
> breadth (a human, CODEOWNERS change). Do not mass-produce ideas straight into authoring.

## Steps

1. **Pick a topic / thesis family.** A family slug from `kb/strategies/_families.md`, or a new
   external angle (a factor, an anomaly, an alt-data thesis).

2. **Run `deep-research`** on it (the deep-research skill). Capture, for each candidate edge:
   - a short `title` and a one-paragraph `hypothesis` (the claimed edge + why),
   - `source_type` (paper|url|forum|filing|thesis) + `source_ref` (url/doi) + `source_date`,
   - the `required_data` capabilities it needs (`ohlcv` today; alt-data like `form_13f`,
     `options_flow`, `dark_pool`, `form_4` will park as `needs_data`),
   - a candidate `family` slug.

3. **Dedup each candidate (deterministic + semantic):**
   - Run `uv run algua research idea dedup-check --title T --hypothesis H --family F`.
   - If `is_novel` is false, READ the returned collisions. If any collision's
     `effective_status` is `refuted`, DO NOT re-add — that idea was already rejected.
   - Cross-check the kb (`_index.md`, `_families.md`) for a refuted/duplicate you recognize that
     a token match would miss (paraphrase, synonym, ticker swap). You are the semantic backstop.

4. **Add the survivors:**
   `uv run algua research idea add --title T --hypothesis H --family F --source-type paper
   --source-ref URL --source-date D --required-data ohlcv [--tag t1]`
   - A genuine, distinct angle that nonetheless trips the token dedup may be added with
     `--allow-duplicate --reason "<why it's genuinely new>"` — use sparingly, with a real reason.

5. **Triage the pool:** `uv run algua research idea list --status open` are testable now;
   `--status needs_data` are parked until their data lands. Pick `open` ideas to author via the
   normal research loop (`strategy new` → backtest → walk-forward → sweep → `research promote`).
   On authoring, link the idea: `uv run algua research idea set-status <id> --to authored
   --strategy <name>`.

6. **Check funnel breadth:** `uv run algua research idea stats` shows idea counts by status — the
   discipline signal. A wide pool is healthy ONLY because the gate will (soon) count it.
