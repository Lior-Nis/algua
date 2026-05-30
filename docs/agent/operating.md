# Operating Playbook (detail)

This expands `CLAUDE.md` with the why behind the rules.

## The live gate
Lifecycle stage lives in the SQLite registry (`algua/registry`). State is a *record*,
not a wall: because agents can write the registry, a bare `stage='live'` flag is not a
security boundary. Going live therefore requires:
1. A human actor (`--actor human`), and
2. A matching, unrevoked approval row created by `registry approve`, binding the exact
   `code_hash` + `config_hash` being promoted.

The future live runner (sub-project 6) will additionally verify the approval against the
hash of the artifact it is about to run. Trust the approval, never the flag.

## Module boundaries
- `contracts/` — pure types/protocols. No I/O, no other algua imports.
- `calendar/` — market sessions; depended on by both backtest and live.
- `registry/` — lifecycle source of truth (`db`, `store`, `approvals`).
- `config/` — pydantic settings (env prefix `ALGUA_`).
- `cli/` — the shared command surface; `main.py` is the entry point.

## JSON everywhere
Commands print indented JSON so an agent can parse results and a human can read them.
Parse stdout; use exit codes for pass/fail (e.g. `doctor`, illegal `transition`).
