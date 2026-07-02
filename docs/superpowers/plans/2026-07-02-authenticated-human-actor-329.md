# Plan: Authenticated `--actor human` (#329)

Spec: `docs/superpowers/specs/2026-07-02-authenticated-human-actor-329.md`. Quality gate between
tasks: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## Task 1 — `verify_signature` namespace param (live_gate.py) + unscoped-line defense
- Add `namespace: str = "algua-go-live"` param to `verify_signature`; pass it to BOTH ssh-keygen
  `-n` invocations (find-principals via `-Overify-time`? no — `find-principals` takes `-n`? it does
  NOT; keep find-principals as-is which is namespace-agnostic, and pass `-n namespace` to the
  `verify` step — that is where the namespace is actually enforced). Default preserves callers.
- FAIL CLOSED (raise SignatureError) if any non-comment, non-blank anchor line lacks an explicit
  `namespaces=` token (unscoped-legacy-signer defense). Pre-parse before ssh-keygen runs.
- Test: unscoped line -> SignatureError; scoped go-live key does NOT verify under algua-human-actor.

## Task 2 — schema: `actor_challenges` table (db.py, 31 -> 32)
Add `CREATE TABLE IF NOT EXISTS actor_challenges (...)` (nonce PK, command, strategy_id, stage_from,
stage_to, code_hash, config_hash, dependency_hash, run_context, issued_at, expires_at, consumed_at)
mirroring live_challenges; bump SCHEMA_VERSION 31 -> 32. Test: migrate() creates it.

## Task 3 — new module `algua/registry/human_actor.py`
Reuse `live_gate.verify_signature`/`SignatureError`/`ALLOWED_SIGNERS_PATH` (import, don't duplicate).
- `_NAMESPACE = "algua-human-actor"`, `_TTL = timedelta(minutes=10)`.
- `canonical_run_context(opts: dict) -> str`: injective canonical JSON — `json.dumps(clean,
  sort_keys=True, separators=(",",":"))` where `clean` drops None values. Booleans stay true/false.
- `build_actor_challenge(command, strategy, strategy_id, stage_from, stage_to, code_hash,
  config_hash, dependency_hash, run_context, nonce, expires_at) -> str` — the ONE payload.
- `issue_actor_challenge` -> {nonce, expires_at, challenge}; INSERT a row.
- `find_pending_actor_challenge` — newest unconsumed/unexpired matching ALL bound fields.
- `consume_actor_challenge(conn, nonce)` — single-use UPDATE, rowcount>0.
- `verify_actor_assertion(conn, ..., signature, anchor=None) -> str | None`: rebuild payload from
  RECOMPUTED identity + re-canonicalized run_context, find pending row, `verify_signature(anchor,
  payload, signature, namespace=_NAMESPACE)`, consume nonce on success. None/fail-closed otherwise.
- `HumanActorChallengeRequired(RuntimeError)` carrying the issued challenge dict.
- `resolve_effective_actor(conn, command, strategy, strategy_id, stage_from, stage_to, code_hash,
  config_hash, dependency_hash, declared_actor, run_context, signature, anchor=None) -> Actor`:
  declared agent/system -> unchanged; declared human + no sig -> issue challenge + raise
  HumanActorChallengeRequired; declared human + sig -> verify_actor_assertion, return HUMAN on
  success else raise ValueError (fail closed).

## Task 4 — wire research promote (research_cmd.py)
Add `--actor-signature PATH`. After `actor_enum` resolves and inputs are resolved (identity needs
the strategy to exist), when `actor_enum is Actor.HUMAN`: compute `identity =
compute_artifact_hashes(name)`; `run_context = canonical_run_context({start,end,demo,snapshot,
fundamentals_snapshot,news_snapshot,universe,windows,holdout_frac,min_holdout_sharpe,
min_holdout_return,min_pct_positive,min_window_sharpe,delistings,n_combos,allow_holdout_reuse,
allow_non_pit,assume_terminal_last_close,new_family})`; open a registry_conn; call
`resolve_effective_actor(command="research promote", stage_from=rec.stage.value, stage_to="candidate",
...)`. Catch HumanActorChallengeRequired -> emit challenge JSON (mirror go-live: action, strategy,
nonce, expires_at, challenge, sign instructions with `-n algua-human-actor`) then return. On
ValueError it surfaces via @json_errors. Then proceed with resolved `Actor.HUMAN`. The `--actor
human` no-sig path must NOT run the walk-forward — challenge-and-return happens first.

## Task 5 — wire paper promote (paper_cmd.py)
Same pattern. `--actor-signature`; `run_context = canonical_run_context({min_observations,
min_coverage,degradation_factor,sharpe_floor,min_vol,max_drawdown,max_staleness})`;
command="paper promote", stage_from=rec.stage.value, stage_to="forward_tested". Challenge-and-return
before constructing the broker. Resolved Actor.HUMAN passed to forward_promotion_preflight/run.

## Task 6 — enrollment: dual-namespace `enroll-approver` (registry_cmd.py)
Add `--namespace` (default "algua-go-live"; accept "algua-go-live" | "algua-human-actor" | "both").
Write `namespaces="<ns>"` (both -> `algua-go-live,algua-human-actor`). Keep the strict-principal +
dup-key checks. Update committed `approvers/allowed_signers` header to document both namespaces and
the "every line MUST be namespace-scoped" rule.

## Task 7 — self-protection: CODEOWNERS (root /CODEOWNERS)
Add `/algua/registry/human_actor.py`, `/algua/cli/research_cmd.py`, `/algua/cli/paper_cmd.py`, and
`/CODEOWNERS` (self). (`/algua/registry/live_gate.py`, `/approvers/` already protected.)

## Task 8 — tests + docs
Tests (tmp anchor + ephemeral ed25519 test key, `ssh-keygen -Y sign -n algua-human-actor`):
- forge: `--actor human` no sig -> challenge issued, walk-forward NOT run, relaxation NOT unlocked.
- happy: sign the challenge -> HUMAN resolved, relaxation unlocked.
- replay across artifact: sign for identity A, edit strategy so code_hash changes -> verify fails.
- replay across run_context: sign for start=X, resubmit with start=Y (same relaxations) -> fails.
- replay across relaxation: sign without --allow-non-pit, resubmit with it -> fails.
- replay across command/stage: reject.
- nonce single-use: second submit of same sig -> fails (consumed).
- expiry: expired challenge -> fails.
- namespace confusion: go-live-namespace key cannot authenticate actor; unscoped line -> SignatureError.
- agent path unchanged: `--actor agent` never issues a challenge; relaxation still refused.
Docs: CLAUDE.md command-surface note on `--actor-signature`; AGENTS.md deferred-scope note that the
forgeable-string half is CLOSED and only deploy-time anchor immutability remains.
