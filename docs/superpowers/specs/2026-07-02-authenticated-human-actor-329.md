# Design: Authenticate the human actor (#329)

## Problem
Every human-only guardrail below the live wall keys off a self-declared `--actor human` string
(`Actor(actor)`), forgeable by any agent driving the same CLI. Unlocks: research `--n-combos`
(declared breadth), `--allow-holdout-reuse`, `--allow-non-pit`, `--assume-terminal-last-close`,
NOVEL/PARENTAGE `--new-family`; paper 7 threshold relaxations. FAIL CLOSED required.

## Approach (reuse the go-live signature trust anchor)
Authenticate `--actor human` on gated commands via an SSH signed challenge over the SAME trust
anchor `approvers/allowed_signers`, under a DISTINCT namespace `algua-human-actor` (a go-live key
does NOT auto-authenticate an actor unless also enrolled for the actor namespace, and vice versa —
namespace separation is enforced by `ssh-keygen -Y verify -n <ns>`).

New module `algua/registry/human_actor.py` mirroring `live_gate.py`:
- `build_actor_challenge(command, strategy, strategy_id, stage_from, stage_to, code_hash,
  config_hash, dependency_hash, run_context, nonce, expires_at)` — ONE canonical payload. The
  `run_context` string (see below) binds the full concrete invocation, not just the relaxation set.
- `canonical_run_context(opts: dict) -> str` — INJECTIVE canonical JSON (`json.dumps` with
  `sort_keys=True`, `separators=(",",":")`, None-valued keys dropped, bool as true/false) of the
  FULL gate-relevant invocation input set, NOT only the human-only relaxations. JSON (not a
  `key=value;` join) so a value containing the delimiter cannot forge a different invocation into
  the same canonical string (Codex GATE-1 HIGH: delimiter injection). For research this includes
  start, end, demo,
  snapshot, fundamentals_snapshot, news_snapshot, universe, windows, holdout_frac, the 4 criteria
  thresholds, delistings, AND the human-only relaxations (n_combos, allow_holdout_reuse,
  allow_non_pit, assume_terminal_last_close, new_family). For paper it is the full ForwardGateCriteria
  (all 7 thresholds). Signed, so a signature authorizes EXACTLY this concrete run — a captured sig
  cannot be re-spent on a different start/end/snapshot/universe/threshold set even with an identical
  relaxation subset (Codex GATE-1 HIGH). One string, so no per-command drift in what is bound.
- `issue_actor_challenge` INSERTs a row into new table `actor_challenges` (nonce PK, command,
  strategy_id, stage_from, stage_to, code/config/dependency_hash, run_context, issued_at,
  expires_at 10min TTL, consumed_at). Returns {nonce, expires_at, challenge}.
- `find_pending_actor_challenge` — newest unconsumed/unexpired matching EVERY bound field.
- `consume_actor_challenge` — single-use UPDATE ... WHERE consumed_at IS NULL, rowcount>0.
- `verify_actor_assertion(conn, ..., signature, anchor)` — REBUILD payload from RECOMPUTED
  identity (compute_artifact_hashes) + re-canonicalized run_context (never stored bytes), find
  pending row, verify_signature over that payload with `-n algua-human-actor`, THEN consume nonce.
  None/fail-closed on: no enrolled signer, bad sig, no/expired/consumed challenge, lost consume race.
- `resolve_effective_actor(conn, command, ..., declared_actor, run_context, signature)` — the ONE
  chokepoint: declared agent/system -> unchanged; declared human + no sig -> issue challenge, raise
  HumanActorChallengeRequired (CLI prints it, mirrors go-live); declared human + sig -> verify,
  return HUMAN iff authenticates else raise ValueError.

`live_gate.verify_signature` gains a `namespace: str = "algua-go-live"` param (default preserves
existing callers) so the actor path can verify under its own namespace with the SAME primitive.

## Wiring
The persisted + matched + signed field is named `run_context` end-to-end (NOT `relaxations`); the CLI
callers MUST pass the FULL canonical invocation context so an implementer cannot wire only the
human-only flags.
- research_cmd.promote: add `--actor-signature PATH`. When actor==human, BEFORE promotion_preflight
  compute identity, build `run_context = canonical_run_context({...ALL gate-relevant inputs...})` =
  {start, end, demo, snapshot, fundamentals_snapshot, news_snapshot, universe, windows, holdout_frac,
  min_holdout_sharpe, min_holdout_return, min_pct_positive, min_window_sharpe, delistings, n_combos,
  allow_holdout_reuse, allow_non_pit, assume_terminal_last_close, new_family}. Call
  resolve_effective_actor(command="research promote", stage_from=rec.stage.value,
  stage_to="candidate", run_context=run_context, ...). Catch HumanActorChallengeRequired -> emit
  challenge JSON, return. Pass resolved Actor onward. Agent path: actor stays AGENT, downstream
  guards refuse relaxations exactly as today (unchanged).
- paper_cmd.promote: same, `run_context = canonical_run_context({...})` = the full ForwardGateCriteria
  (all 7 thresholds: min_observations, min_coverage, degradation_factor, sharpe_floor, min_vol,
  max_drawdown, max_staleness); stage_to="forward_tested".

## Enrollment (GATE-1 hole #2)
`registry enroll-approver` gains `--namespace` (default `algua-go-live`, choices incl.
`algua-human-actor`, and both). ALWAYS writes an explicit `namespaces="<ns>"` (never an unscoped
line). A human-actor key MUST be explicitly enrolled for the actor namespace. Committed
`approvers/allowed_signers` header documents both namespaces and states every line MUST be
namespace-scoped. `/approvers/` is already CODEOWNERS-protected, so enrollment only edits a
worktree copy; it cannot affect the INSTALLED trust anchor until a human merges the anchor diff
through code-owner review (same guarantee as go-live enrollment).

## Namespace-confusion / legacy-signer defense (Codex rev2 HIGH)
OpenSSH treats an `allowed_signers` line with NO `namespaces="..."` restriction as valid for ANY
namespace — so a single unscoped line would let a go-live-only key authenticate `algua-human-actor`.
Existing go-live keys are scoped `namespaces="algua-go-live"` and thus already do NOT match the
actor namespace, but we defend against a hand-edited unscoped line: `verify_signature` FAILS CLOSED
(SignatureError) if ANY non-comment, non-blank line in the anchor lacks an explicit `namespaces=`
field. This is a cheap pre-parse of the anchor before ssh-keygen runs; it protects BOTH namespaces.
The shared verifier `live_gate.verify_signature` (which the actor path reuses with `-n
algua-human-actor`) is ALREADY CODEOWNERS-protected (`/algua/registry/live_gate.py`), so an agent
PR cannot quietly make it ignore the namespace arg and reach main.

## Self-protection (GATE-1 hole #1 + Codex CRITICAL)
The enforcement (resolve_effective_actor call) lives in the CLI command files, so protecting only
the downstream promotion modules is insufficient: an agent PR could delete/skip the CLI resolve
call and hand `Actor.HUMAN` straight to the (already-protected) downstream gate without touching a
protected file. So add to the active CODEOWNERS: `/algua/registry/human_actor.py`,
`/algua/cli/research_cmd.py`, `/algua/cli/paper_cmd.py`, AND the CODEOWNERS file itself (self).
The repo's active CODEOWNERS is at repo ROOT `/CODEOWNERS` (confirmed: no `.github/CODEOWNERS`);
protect `/CODEOWNERS` as self. CLI callers pass NO anchor path — resolve_effective_actor always uses the
installed ALLOWED_SIGNERS_PATH (anchor override is a test-only kwarg).

## Replay defense (GATE-1 hole #3)
The signed payload binds: namespace + command + strategy + strategy_id + stage_from + stage_to +
code_hash + config_hash + dependency_hash + FULL run_context (every gate-relevant input incl. the
exact relaxation set) + single-use nonce + expiry. Verification REBUILDS from recomputed identity +
re-canonicalized run_context, so a captured sig cannot replay onto: a new artifact (identity
differs), any different concrete invocation — different start/end/snapshot/universe/threshold OR a
different relaxation set (run_context differs), a different command/stage edge, another strategy, or
a second run (nonce consumed).

## Schema
SCHEMA_VERSION 31 -> 32; add `actor_challenges` (IF NOT EXISTS, idempotent, mirrors live_challenges).

## Trust boundary (honest residual — the deferred half of #329)
Same as go-live, IDENTICAL boundary: `verify_signature` reads `ALLOWED_SIGNERS_PATH`
(`parents[2]/approvers/allowed_signers`) in the running tree — the same file `enroll-approver`
appends to. So a process that can WRITE the local tree (or the registry DB) can enroll a key and
authenticate, exactly as it can already defeat the go-live gate. CODEOWNERS protects MERGE to main,
not local-runtime immutability. This is NOT a new hole introduced here: it is the pre-existing,
issue-#329-documented "enforce actor authenticity out-of-band at deploy + write-protect the anchor
at runtime" half, explicitly DEFERRED (a deploy-time-immutable installed anchor distinct from the
mutable worktree copy). This PR closes the in-scope hole — the forgeable UNAUTHENTICATED `--actor
human` string — so the human-only gates are now a cryptographic boundary against a same-CLI agent
that cannot write the tree/DB, not mere discipline. The deploy-immutability follow-up is filed
separately.
