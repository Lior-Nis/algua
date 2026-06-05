# Operating Playbook (detail)

This expands `CLAUDE.md` with the why behind the rules.

## The live gate
Lifecycle stage lives in the SQLite registry (`algua/registry`). State is a *record*,
not a wall: because agents can write the registry, a bare `stage='live'` flag is not a
security boundary. Going live is a two-step signed ceremony requiring a human actor:

1. Run `registry transition <name> --to live --actor human` (no `--signature`) — the CLI issues a
   cryptographic challenge bound to the strategy's current code+config hash and returns it as JSON.
2. Sign the `challenge` value with your enrolled SSH key:
   `ssh-keygen -Y sign -n algua-go-live -f <key> <file>`
3. Re-run: `registry transition <name> --to live --actor human --signature <file>.sig` — the CLI
   verifies the signature, consumes the single-use challenge, and advances the stage to `live`.

The challenge is single-use and expires after 10 minutes. Trust the signature, never the flag.

## Module boundaries
- `contracts/` — pure types/protocols. No I/O, no other algua imports.
- `calendar/` — market sessions; depended on by both backtest and live.
- `registry/` — lifecycle source of truth (`db`, `store`, `approvals`).
- `config/` — pydantic settings (env prefix `ALGUA_`).
- `cli/` — the shared command surface; `main.py` is the entry point.

## JSON everywhere
Commands print indented JSON so an agent can parse results and a human can read them.
Parse stdout; use exit codes for pass/fail (e.g. `doctor`, illegal `transition`).
