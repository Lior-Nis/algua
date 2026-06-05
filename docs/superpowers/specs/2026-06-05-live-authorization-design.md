# Trade-Time Live Authorization (Sub-project 6, live-execution slice 1)

**Date:** 2026-06-05
**Status:** Accepted (pending implementation)
**Scope:** Persist the cryptographic evidence of a signed `paper → live` approval, and build the
trade-time `verify_live_authorization` primitive that re-verifies it against the trust anchor before
any live order. **No live venue and no live orders in this slice** — this is the wall, built before
the door (slices 2–3 add the live broker + loop and call this primitive).

---

## 1. Context & non-goals

PR #115 made the `paper → live` *transition ceremony* require an Ed25519 SSH signature over an
artifact-bound, single-use challenge. But it explicitly left the **money wall** for trade time: an
agent with raw SQLite write can forge `strategies.stage='live'` or an `approvals` row, so the actual
defense must **re-verify cryptographic evidence before placing a live order**, never trust DB state
(see `live-execution-reverify-mandate` / spec §11 of the signed-gate design). This slice discharges
that mandate by (a) persisting the signature evidence at go-live and (b) building the re-verification
primitive the live loop will call.

**Non-goals (later slices):** the live Alpaca broker (`api.alpaca.markets`), live keys, the
`live trade-tick` command/loop, capital caps, position reconciliation, an explicit
authorization-revocation command. **Out of scope:** changing the signed-transition ceremony itself
(#115) beyond writing the new evidence row.

---

## 2. Design decisions (settled in brainstorming)

- **Evidence lives in a dedicated `live_authorizations` table** — the `live_challenges` table stays
  ephemeral (issue→consume); the authorization is the durable, re-verifiable proof.
- **The primitive raises** `LiveAuthorizationError` on any failure (rendered `{ok:false}` by the CLI
  layer later), rather than returning a bare bool — the failure reason matters at a money gate.
- **One shared trust anchor:** `ALLOWED_SIGNERS_PATH` moves into `algua/registry/live_gate.py` so
  both the CLI and the primitive resolve the same `__file__`-anchored file.
- **Revocation is implicit this slice:** a code/config/dependency change (hash mismatch), a
  transition off `live` (stage check), or removing the approver's key from `allowed_signers`
  (verify fails) all invalidate a live authorization. An explicit `revoked_at`-setting command is a
  later slice; the column exists now so the data model is ready.

---

## 3. Persistence — `live_authorizations` (schema v12)

```sql
CREATE TABLE IF NOT EXISTS live_authorizations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id     INTEGER NOT NULL REFERENCES strategies(id),
    code_hash       TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    dependency_hash TEXT,
    challenge       TEXT NOT NULL,   -- the EXACT signed payload, so re-verify can't drift
    signature       TEXT NOT NULL,   -- base64 of the SSH signature bytes
    principal       TEXT NOT NULL,   -- the enrolled approver who signed
    authorized_at   TEXT NOT NULL,
    revoked_at      TEXT
);
```
`SCHEMA_VERSION` 11 → 12 (bootstrap `migrate()` adds the table).

---

## 4. Writing the evidence at go-live

`live_gate.verify_and_consume(...)` already, on a successful signed transition: finds the pending
challenge, rebuilds the exact payload, `ssh-keygen -Y verify`s it, and consumes the nonce. It has
everything needed for the evidence row (strategy_id, the identity hashes, the challenge payload, the
signature bytes, the verified principal). Extend it so that — **in the same step, before returning
the principal** — it also inserts a `live_authorizations` row:
`(strategy_id, code_hash, config_hash, dependency_hash, challenge=payload,
signature=base64(signature_bytes), principal, authorized_at=now)`.

So every successful signed `transition --to live` now leaves a self-proving authorization record.
The existing `record_approval(...)` (the #114 hash-pinned approvals row, used for audit /
`has_valid_approval`) is unchanged and kept — `live_authorizations` is the *crypto-evidence* record,
distinct from the approvals audit row.

---

## 5. The primitive — `verify_live_authorization`

`algua/registry/live_gate.py`:
```python
class LiveAuthorizationError(RuntimeError): ...

def verify_live_authorization(conn, repo, name, allowed_signers_path) -> sqlite3.Row:
    # 1) stage must be LIVE
    # 2) identity = compute_artifact_hashes(name)  (code/config/dependency)
    # 3) row = newest UNREVOKED live_authorizations matching (strategy_id, identity)
    # 4) verify_signature(allowed_signers_path, row["challenge"], b64decode(row["signature"]))
    #    must return a principal == row["principal"]  (still enrolled in the CURRENT anchor)
    # else raise LiveAuthorizationError(<specific reason>)
    return row
```
- Step 2 binds to the **current** source/config/lockfile, so any change since approval fails.
- Step 4 re-runs `ssh-keygen` against the **current** `allowed_signers`, so a revoked/rotated
  approver key fails; and a forged authorization row fails because the agent can't produce a
  signature that verifies for the stored challenge without the private key.
- It trusts the `stage` column only as a fast precheck — the *binding* security comes from steps 3–4
  (the signature), never from `stage` or `approvals` alone.

This is a pure registry-layer function; the future `live trade-tick` (slice 3) calls it before every
order and refuses to trade if it raises.

---

## 6. Testing

- **Schema** — `migrate` adds `live_authorizations`; `user_version == SCHEMA_VERSION`.
- **Evidence written** — a full signed go-live (hermetic `ssh-keygen` key, enroll, issue, sign,
  transition) leaves exactly one `live_authorizations` row with the expected hashes, principal, and a
  `signature`/`challenge` that re-verify.
- **`verify_live_authorization` happy path** — after a real signed go-live, the primitive returns the
  row.
- **Rejections (each raises `LiveAuthorizationError`)** — strategy not at `live`; no matching
  authorization (e.g. a forged `stage='live'` with no row); code changed since approval (hash
  mismatch); approver key removed from `allowed_signers` (re-verify fails); a tampered
  `challenge`/`signature` row; a `revoked_at`-set row ignored.
- **Shared anchor** — `ALLOWED_SIGNERS_PATH` importable from `live_gate` and used by both the CLI and
  the primitive.
- **Gate** — `pytest · ruff · mypy · lint-imports` (contracts kept, 0 broken).

---

## 7. Consequences

- The trade-time wall now **exists before any live-order code does**: slices 2–3 physically cannot
  place a live order without calling a primitive that re-verifies a human signature against the
  CODEOWNERS trust anchor.
- The platform stops trusting the agent-writable `stage`/`approvals` rows as the live authority; the
  authority is a signature re-verified at trade time.
- `live_authorizations` is the seam the eventual live loop, a kill-on-revoke flow, and a live audit
  view all read. Explicit revocation, the live broker, and the live loop are the next slices.
