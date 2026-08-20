"""Authorization context: ``live_challenges``, ``actor_challenges``, ``live_authorizations``,
``strategy_allocations``.

The go-live and human-actor challenge/signature ledgers (``algua/registry/challenges.py``,
``live_gate.py``, ``human_actor.py``) and the capital allocations the live wall gates
(``algua/registry/allocations.py``).
"""
from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_challenges (
    nonce           TEXT PRIMARY KEY,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    code_hash       TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    dependency_hash TEXT,
    issued_at       TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    consumed_at     TEXT
);
-- Human-actor authentication challenges (#329). Mirrors live_challenges: a bare `--actor human`
-- string is forgeable, so asserting a human actor on a gated command (research/paper promote) now
-- requires an SSH signature (namespace algua-human-actor) over a fresh single-use challenge. The
-- signed payload binds the command + strategy + RECOMPUTED artifact identity + the FULL canonical
-- run_context (every gate-relevant input, incl. the exact relaxation set) + nonce + expiry, so a
-- captured signature cannot be replayed onto a different artifact/run/relaxation/command/strategy
-- or a second run. Like live_challenges we persist only the non-identity parts (verify REBUILDS
-- the identity + run_context), and consume the nonce single-use.
CREATE TABLE IF NOT EXISTS actor_challenges (
    nonce           TEXT PRIMARY KEY,
    command         TEXT NOT NULL,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    stage_from      TEXT NOT NULL,
    stage_to        TEXT NOT NULL,
    code_hash       TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    dependency_hash TEXT,
    run_context     TEXT NOT NULL,
    issued_at       TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    consumed_at     TEXT
);
-- The signed payload is NEVER stored verbatim and re-verified — an agent with DB write could then
-- pair vetted-identity columns with a foreign signature (codex CRITICAL). We store only the
-- non-identity payload parts (nonce, expires_at); trade-time verification REBUILDS the canonical
-- challenge from the RECOMPUTED identity + strategy + these, so a signature is valid only over the
-- current artifact.
CREATE TABLE IF NOT EXISTS live_authorizations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    code_hash       TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    dependency_hash TEXT,
    nonce           TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    signature       TEXT NOT NULL,
    principal       TEXT NOT NULL,
    authorized_at   TEXT NOT NULL,
    revoked_at      TEXT
);
CREATE TABLE IF NOT EXISTS strategy_allocations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id   INTEGER NOT NULL REFERENCES strategies(id),
    capital       REAL NOT NULL,
    effective_ts  TEXT NOT NULL,
    actor         TEXT NOT NULL,
    revoked_ts    TEXT
);
"""
