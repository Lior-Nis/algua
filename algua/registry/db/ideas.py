"""Idea-pool context: ``ideas``.

The structured, deduped, provenance-stamped top-of-funnel hypothesis pool (#126), operated by
``algua/registry/ideas.py``.
"""
from __future__ import annotations

SCHEMA = """
-- ideas is the structured top-of-funnel pool (#126): externally-sourced, deduped,
-- provenance-stamped hypothesis records that climb the normal gated ladder. authored_strategy_id
-- is the relational link to the strategy an idea became (NULL until authored); the dedup gate
-- resolves a refuted strategy through this FK (a live join), so a refuted strategy blocks its
-- idea's near-duplicates without mutating idea rows. duplicate_of_idea_id records a deliberate
-- --allow-duplicate override (paired with override_reason).
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    family TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    source_type TEXT NOT NULL,
    source_ref TEXT,
    source_date TEXT,
    source_note TEXT,
    required_data TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    signature TEXT NOT NULL,
    authored_strategy_id INTEGER REFERENCES strategies(id),
    duplicate_of_idea_id INTEGER REFERENCES ideas(id),
    override_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ideas_status ON ideas(status);
CREATE INDEX IF NOT EXISTS ix_ideas_family ON ideas(family);
-- v46 (ideation engine, spec §7). idea_attempts is APPEND-ONLY: one row per claim, its
-- outcome written once under the claim's fencing token by a trusted driver. idea_inspirations
-- links an idea to every kb/inspirations note it leaped from (full credit each).
CREATE TABLE IF NOT EXISTS idea_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id INTEGER NOT NULL REFERENCES ideas(id),
    run_stamp TEXT NOT NULL,
    claim_token TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    outcome TEXT,
    reason TEXT,
    evidence_ref TEXT,
    strategy_name TEXT,
    outcome_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_attempts_idea ON idea_attempts(idea_id);
CREATE TABLE IF NOT EXISTS idea_inspirations (
    idea_id INTEGER NOT NULL REFERENCES ideas(id),
    inspiration_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    obscurity TEXT NOT NULL,
    created_by_run TEXT NOT NULL,
    PRIMARY KEY (idea_id, inspiration_id)
);
"""
