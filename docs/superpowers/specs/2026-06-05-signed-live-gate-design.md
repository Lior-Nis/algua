# Signed `paper → live` Gate (Sub-project 6, first slice)

**Date:** 2026-06-05
**Status:** Accepted (pending implementation)
**Scope:** Replace the honor-system live gate with a cryptographic human signature. Going live
requires an Ed25519 signature over an artifact-bound, single-use challenge, made with a private key
that never touches the computer. Completes the wall the foundation only mechanism-stubbed.

---

## 1. Context & non-goals

Lifecycle stage lives in the agent-writable SQLite registry. The `paper → live` transition already
requires `actor=human` + a matching unrevoked `approvals` row whose `code_hash`/`config_hash` are
recomputed from the live source (`algua/registry/approvals.py::compute_artifact_hashes`). But
nothing proves the `--actor human` claim, and `registry approve` is callable by anyone — so an
autonomous agent could in principle approve its own code and promote it. This slice closes that
hole with public-key cryptography.

**Decision (supersedes the earlier TOTP note in the live-gate memory):** use an **asymmetric
signature**, not a shared-secret TOTP. TOTP would require the seed on the computer too; a signature
keeps the private key solely on the operator's device and stores only the **public key** on the
computer — nothing forgeable ever lives where the agent could reach it.

**Non-goals (later sub-6 slices):** the actual live execution path / live broker (none exists yet);
WebAuthn / hardware-key support; key-rotation tooling beyond editing the approvers file; recovery
beyond "enroll a backup key + revoke a lost one". **Out of scope:** changing any `paper`-and-below
behaviour.

---

## 2. Design decisions (settled in brainstorming)

- **Proof = an Ed25519 signature** over an artifact-bound, single-use challenge; private key only on
  the operator's device, computer holds only public keys.
- **The chokepoint is the `transition --to live` itself** (not a separate `approve`): the human is
  cryptographically present at the exact go-live moment; an agent can never flip the switch even for
  already-reviewed code.
- **Format = SSH signatures** (`ssh-keygen -Y sign` / `-Y verify`, namespace `algua-go-live`),
  reusing an existing Ed25519 SSH key and standard tooling. No new pip dependency — `ssh-keygen` is a
  system binary we shell out to.

---

## 3. Trust anchor — `approvers/allowed_signers`

Enrolled approver **public** keys live in a version-controlled SSH `allowed_signers` file at
`approvers/allowed_signers`, **protected by CODEOWNERS + branch protection** (the same integrity
mechanism already guarding store/lifecycle/engine/gates).

```
# approvers/allowed_signers   (one line per enrolled approver public key)
lior namespaces="algua-go-live" ssh-ed25519 AAAAC3Nz...  lior@device
```

**Why this is the trust anchor, not the DB:** a public key is safe for the agent to *read*, but the
agent must not be able to *enroll its own key*. The registry DB is agent-writable, so keys there
would be forgeable. A reviewed, CODEOWNERS-gated file on `main` is not: an agent editing it in its
sandbox/worktree never reaches `main`, and **go-live runs vetted `main` code in a context the agent
is not in** (the existing operator-harness model). The same protection covers the gate code itself.

**Enrollment:** `algua registry enroll-approver --pubkey "<ssh-ed25519 …>" --name <id>` appends a
correctly-formatted line to `approvers/allowed_signers`. It is a *convenience writer*; the trust
comes from committing that change through code-owner review. Multiple keys may be enrolled (a backup
key on a second device is the recovery story); revoking a lost device = removing its line via a
reviewed commit. `--name` must be non-empty; a duplicate pubkey is rejected.

---

## 4. Persistence — `live_challenges` table

```sql
CREATE TABLE IF NOT EXISTS live_challenges (
    nonce        TEXT PRIMARY KEY,         -- random 256-bit hex
    strategy_id  INTEGER NOT NULL REFERENCES strategies(id),
    code_hash    TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    issued_at    TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    consumed_at  TEXT                       -- NULL until the signature is accepted (single-use)
);
```
`SCHEMA_VERSION` bumps by one (bootstrap `migrate()` adds the table). The nonce table being
agent-writable is fine: forging a row doesn't help, because the agent still cannot sign the
challenge.

---

## 5. The challenge & the two-step transition

**Challenge payload** (the exact bytes signed) — a single deterministic line:
```
algua-go-live\nstrategy=<name>\nstrategy_id=<id>\ncode_hash=<h>\nconfig_hash=<h>\nnonce=<nonce>\nexpires_at=<iso>
```
(If PR #114's `dependency_hash` lands first, add `dependency_hash=<h>` — coordinate at implementation
time.)

**Step 1 — issue:** `algua registry transition <name> --to live` (no `--signature`):
1. Resolve the strategy; require current stage `paper` (the existing edge check).
2. `code_hash, config_hash = compute_artifact_hashes(name)` (pins the real artifact).
3. Generate a random `nonce`; persist a `live_challenges` row (`expires_at = now + TTL`, e.g. 10 min).
4. Emit `{ok:true, action:"go_live_challenge", challenge:"<payload>", nonce, expires_at,
   instructions:"sign with: ssh-keygen -Y sign -n algua-go-live -f <key> <file>; then re-run with
   --signature <sigfile>"}`. **No transition happens.**

**Step 2 — verify & transition:** `algua registry transition <name> --to live --signature <path>`:
1. Read the signature file; recompute `code_hash, config_hash` from the *current* source.
2. Look up the pending `live_challenges` row whose `(strategy_id, code_hash, config_hash)` match and
   that is **unconsumed and unexpired**. If none → `{ok:false}` (`"no matching pending go-live
   challenge — run without --signature first, and don't change the code in between"`).
3. Reconstruct the exact challenge payload from the row. Discover which enrolled key signed via
   `ssh-keygen -Y find-principals -f approvers/allowed_signers -s <sig>` (reading the payload on
   stdin) — no match → the signer is not enrolled → reject. Then verify with that principal:
   `ssh-keygen -Y verify -f approvers/allowed_signers -I <principal> -n algua-go-live -s <sig>`
   over the same payload. (find-principals → verify is the canonical multi-approver flow; the
   operator never has to name themselves.)
4. On success: mark the row `consumed_at = now` (single-use), `record_approval(...)` (audit + the
   existing approval record), and perform the `paper → live` transition with `actor=human`.
5. On any failure (bad signature, unenrolled key, expired/consumed/missing challenge, artifact
   changed): emit `{ok:false}`, transition does NOT happen, challenge stays unconsumed (so a
   transient error is retryable).

**Single CLI, two modes:** `--signature` absent → issue; present → verify. Keeps one command.

---

## 6. Removing the honor-system `approve`

`registry approve` (no proof — anyone can write an approval row) is **removed**: its purpose is now
served by the signed transition, which records the approval as part of going live. `record_approval`
/ `has_valid_approval` / `compute_artifact_hashes` stay (reused by the transition path and by
whatever live-execution slice consumes the approval record). Any test/doc referencing `registry
approve` is updated to the signed flow. (Coordinate with PR #114 if it also touches this surface.)

---

## 7. Verification helper

A small `algua/registry/signing.py` (or `live_gate.py`) module:
- `build_challenge(strategy, strategy_id, code_hash, config_hash, nonce, expires_at) -> str` — the
  canonical payload (one definition, used to both issue and verify, so they can't drift).
- `verify_signature(allowed_signers_path, namespace, payload, signature_bytes) -> bool` — shells to
  `ssh-keygen -Y find-principals` (signer enrolled?) then `ssh-keygen -Y verify -I <principal>`,
  returns True only on a clean verify against an enrolled key; wraps process/`FileNotFoundError`
  errors into a typed `SignatureError` (rendered `{ok:false}`), never a raw traceback.

Keep this module pure of CLI concerns; the command wires it to the DB + emit.

---

## 8. Testing

- **Challenge lifecycle** — issue persists a row + prints the payload, no transition; lookup matches
  on `(strategy_id, code_hash, config_hash)`; expired row rejected; consumed row rejected (replay);
  no-matching-row rejected.
- **Signature verify (hermetic)** — in-test, `ssh-keygen -t ed25519` a throwaway key, enroll its
  public key into a temp `allowed_signers`, `ssh-keygen -Y sign` the canonical payload, assert
  `verify_signature` True; assert False for: a different (unenrolled) key, a signature over a
  *different* payload (artifact changed), a garbage signature.
- **End-to-end transition** — step 1 leaves stage `paper`; step 2 with a valid enrolled signature
  advances to `live`, records an approval, and consumes the nonce; re-running step 2 with the same
  signature fails (consumed).
- **Negatives** — `--signature` with no pending challenge → `{ok:false}`, no transition; unenrolled
  signer → `{ok:false}`; tampered payload → `{ok:false}`.
- **Enrollment** — `enroll-approver` appends a well-formed line; empty `--name` and duplicate pubkey
  rejected.
- **Honor-system removal** — `registry approve` no longer exists (invoking it errors).
- **Gate** — `pytest · ruff · mypy · lint-imports` (contracts kept, 0 broken). `ssh-keygen` is
  assumed present in CI (skip-marker the signature tests if it is genuinely unavailable, but it ships
  with the OpenSSH client that's standard on the dev/CI image).
- **Live acceptance (manual, documented)** — enroll a real key, issue a challenge, sign on a phone/
  device, complete the transition; confirm an unsigned attempt and a wrong-key attempt are refused.

---

## 9. Consequences

- The `paper → live` wall is now real: only a holder of an enrolled private key can move a strategy
  to live, and only for the exact reviewed artifact, once, within a short window. An autonomous
  agent — even with full code access — cannot promote to live.
- The trust anchor (approver public keys) and the gate code are both CODEOWNERS-protected on `main`,
  so the wall can't be moved by agent-authored changes that never run where go-live happens.
- This is the foundation for the rest of sub-project 6 (the actual live execution path), which can
  now rely on a genuinely human-gated `live` stage. Recovery is "enroll a backup key"; richer
  rotation/WebAuthn are later slices.
