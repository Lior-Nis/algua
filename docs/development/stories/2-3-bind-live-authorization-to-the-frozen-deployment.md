---
baseline_commit: e875b6d
---

# Story 2.3: Bind the signed live authorization to the exact frozen deployment

Status: backlog

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR8, FR1 (live activation of the evidence-bearing artifact), NFR1, NFR4–NFR8.
Upstream: [#661](https://github.com/Lior-Nis/algua/issues/661) slice 6,
[#614](https://github.com/Lior-Nis/algua/issues/614).
Depends on: Story 2.1 (qualification predicate) and Story 2.2 (paper exit drain on the go-live
edge).
Gated by: both predecessors merged, then its contract and readiness review. No owner decision is
needed to build it.

## Story

As Algua's owner,
I want my go-live signature to name the exact frozen deployment, manifest and certificate that
earned the evidence, and to be re-verified against that deployment before every live cycle,
so that a signature can never authorize a different artifact, epoch, data provider or account, and
can be ended by revocation or by leaving live.

## Context

The ceremony today signs the checkout, not the deployment:

- The signed payload binds strategy, strategy id and the three identity hashes
  (`algua/registry/live_gate.py:17-22`, `:34-41`). Step 1 (`algua/cli/registry_cmd.py:203-228`) and
  step 2 (`registry_cmd.py:230-255`, `transitions.py:116-157`) both compute that identity by
  importing the checkout module (`compute_artifact_hashes`, `transitions.py:141`,
  `registry_cmd.py:215`).
- Trade-time re-verification also recomputes the checkout identity (`live_gate.py:178`) and selects
  the authorization by hashes (`live_gate.py:181-185`). Two deployments with equal hashes could
  share one signature; Story 1.2 closed the certificate side of that hole and left the payload to
  this story.
- Frozen deployments are refused at the wall: `refuse_frozen_deployment` raises
  `frozen_live_unsupported` (`transitions.py:198-218`), and the certificate verifier calls
  `require_tick_deployment` (`live_certificate.py:71`), which refuses any frozen row
  (`algua/registry/store/deployment.py:100-101`). Since Story 1.3c every new admission is frozen, so
  no strategy can reach live at all today.
- Nothing ever sets `live_authorizations.revoked_at` (no writer exists; only reads at
  `live_gate.py:189`, `:229`). An authorization survives `live -> dormant -> ... -> live` with the
  same hashes. `authorization_active` (`live_gate.py:217-233`), the per-order check, matches
  identity columns only and is documented as advisory.
- `transitions._default_approval_verifier` (`transitions.py:227-230`) still accepts an `approvals`
  row for programmatic callers; only the CLI injects the signature verifier
  (`registry_cmd.py:232-246`).
- #614: the live bars provider is a required, human-chosen setting
  (`algua/config/settings.py:51-55`, `algua/cli/lane_refresh.py:158-162`), but changing it does not
  invalidate a standing signature.
- The frozen descriptor and content chokepoint already exist for promotion: `promotion_slot`
  (`algua/registry/forward_promotion.py:111-130`) parses the recorded descriptor and verifies the
  bundle and environment offline with a fresh `FrozenContentVerifier`, never the checkout.
- Go-live sheds the paper allocation and requires paper-lane flatness (`transitions.py:30-33`,
  `algua/registry/store/crud.py:333-357`); Story 2.2 adds the open-order drain to that edge.

## Scope and authority

In scope: a deployment-bound signed challenge and authorization for frozen deployments; frozen
certificate verification from the recorded descriptor; trade-time re-verification from the
deployment record; binding of the live bars provider (#614) and the live account; revocation and
episode binding; a frozen-only go-live.

Must not: rebuild, re-freeze, re-materialize or re-hash the checkout during activation; add any
agent path to live; add a waiver; change the signing namespace or key enrollment; change capital
(go-live still lands unallocated, #497); activate any strategy; delete the working-tree or legacy
tick paths (owner deferral of 2026-10-01). The live lane keeps refusing to tick a frozen deployment
until Story 2.4, which this story records as a temporary limitation.

## Acceptance criteria

1. **Verified before a challenge exists.** For a `forward_tested` strategy whose active deployment
   is frozen, step 1 verifies, before writing any challenge: the recorded descriptor parses and its
   bundle and environment verify through a fresh offline verifier (the `promotion_slot` chokepoint);
   the frozen branch of the certificate verifier passes for that deployment; Story 2.1's
   qualification predicate is empty; the live bars provider is configured; the live account identity
   is readable with the live credentials. Each failure has a stable code.
2. **What the human signs.** The single-use challenge (namespace `algua-go-live`, existing 10-minute
   TTL) binds: strategy name and id, deployment id, manifest digest, bundle digest, environment
   digest, the three identity hashes, research gate id, forward-certificate id, the live bars
   provider policy (the provider name plus every provider-selecting setting the contract
   enumerates), the live account id, nonce and expiry. The printed challenge also shows the
   certificate summary, both relaxation sets, the universe binding and, when recorded, the provider
   of the evidence snapshots.
3. **Completion rebuilds, never trusts stored bytes.** Step 2 rebuilds the payload from the
   deployment record, the certificate row, the current provider policy and the current live account
   id, re-runs every step-1 check, and verifies the signature before the write lock. One transaction
   then consumes the challenge with its full predicate re-asserted (same active deployment,
   unexpired, unconsumed), inserts the authorization row, performs the `forward_tested -> live` CAS,
   revokes the paper allocation and runs the paper flatness and drain checks. Any failure rolls
   everything back and leaves the nonce unburned.
4. **No rebuild.** The frozen go-live path calls no artifact preparation, environment provisioning
   or checkout hashing (`compute_artifact_hashes`); a test fails if it does.
5. **Trade-time re-verification.** For a frozen deployment, `verify_live_authorization` requires
   stage `live`, that the authorization's deployment is still the active one, that the newest
   authorization for that strategy and deployment is unrevoked, and that the signature verifies over
   the payload rebuilt from the deployment record, the stored nonce and expiry, the current provider
   policy and the current live account id. A provider or account change fails it with a stable code;
   `live run-all` then skips that strategy before any effect.
6. **Per-order check bound to the row.** `LiveAuthorization` carries the authorization id and
   deployment id; `authorization_active` checks that exact row is unrevoked and its deployment still
   active.
7. **No crossing.** An authorization for deployment A never authorizes deployment B, even with
   identical hashes; a retired deployment's authorization never verifies.
8. **Episode and revocation.** Every exit from `live` (to `paper`, `dormant` or `retired`) revokes
   the active authorization in the same transaction, with a reason. `live revoke NAME --reason`
   revokes immediately and is agent-allowed (risk-reducing); the next mid-tick check stops further
   orders. Rows are never deleted; revocation is the only permitted update.
9. **Frozen-only go-live.** Go-live requires an active frozen deployment. A working-tree deployment
   or a strategy without a deployment is refused with a stable code (none exists in production, and
   none can be admitted since Story 1.3c). Only a verified signature can approve a frozen go-live;
   the `approvals`-row path cannot.
10. **Schema.** The protected migration adds the deployment binding to `live_challenges` and
    `live_authorizations` (`algua/registry/db/authz.py:11-20`, `:48-60`), with append-preserving
    triggers that allow only one-way revocation. No backfill: production has no authorization rows.
11. **Authority preserved and documented.** Actor must be human; no agent path; `CLAUDE.md`,
    `docs/architecture.md` and the error envelope describe the new ceremony and the temporary lane
    limitation. Full gate passes; protected review.

## Tasks / subtasks

- [ ] Contract and readiness: payload fields and order, provider-policy fields, DDL and triggers,
      entry-point inventory (CLI steps 1 and 2, programmatic `transition_strategy`, every
      `verify_live_authorization` caller: `live run-all`, `live flatten`, the live exit guard).
- [ ] Frozen certificate branch from the recorded descriptor (AC1).
- [ ] Deployment-bound challenge spec, issue and completion; carve the ceremony out of
      `registry_cmd.py` (AC1–AC4).
- [ ] Trade-time verification and per-order check (AC5–AC7).
- [ ] Episode revocation in the exit transaction; `live revoke` (AC8).
- [ ] Frozen-only go-live and removal of the approvals-row path for frozen (AC9).
- [ ] Schema migration (AC10); docs (AC11).
- [ ] End-to-end ceremony test with a real `ssh-keygen` key; full gate; independent review.

## Dev notes

### Seams

| Seam | Change |
|---|---|
| `algua/registry/live_gate.py:17-41`, `:132-159` | New deployment-bound spec and payload; pending verification from the deployment record |
| `algua/registry/live_gate.py:166-233` | Frozen trade-time verification; row-bound active check |
| `algua/registry/live_certificate.py:65-99` | Frozen branch: recorded descriptor plus fresh content verification instead of `require_tick_deployment` |
| `algua/registry/transitions.py:116-157`, `:198-218` | Frozen go-live path; `refuse_frozen_deployment` no longer refuses `-> live` for frozen; refuses non-frozen |
| `algua/registry/store/base.py:48-81` | Consume with deployment predicate; insert bound row |
| `algua/registry/store/crud.py:240-318` | Revoke the authorization on live exits |
| `algua/cli/registry_cmd.py:181-260` | Thin shell; the ceremony moves to a registry module (446 pin, 2 lines headroom) |
| `algua/contracts/types.py:439-468` | `LiveAuthorization` gains authorization id and deployment id |

### Traps

- The certificate's `account_id` is the PAPER account (hygiene continuity); the authorization's
  account id is the LIVE account. Keep them distinct in names and code.
- Read the live account id with the read-only live broker kind; it needs no authorization. If live
  credentials are missing, refuse with an actionable message.
- Verify the signature (a slow `ssh-keygen` subprocess) before `BEGIN IMMEDIATE`, as today
  (`live_gate.py:139-143`).
- `--snapshot` replays: if snapshot metadata does not record its provider, refuse a frozen
  authorization's tick under `--snapshot` rather than guess.
- `live -> dormant` keeps the deployment (Story 1.2 matrix) but must end the authorization; a later
  `dormant -> paper -> forward_tested -> live` needs a fresh certificate and a fresh signature.
- Do not silently keep the `has_valid_approval` fallback reachable for frozen strategies.

### Test matrix

Each step-1 refusal writes no challenge; payload rebuilt from records equals the issued bytes;
tampered deployment id, manifest, provider, account, certificate or gate id fails; expired, reused
or raced nonce; deployment retired between steps; two deployments with equal hashes; provider
changed after go-live; account changed after go-live; revocation by exit and by command; mid-tick
revocation stops orders; non-frozen go-live refused; no checkout hashing on the frozen path;
flatness and Story 2.2 drain enforced on the edge.

## Owner decisions

None to build. Activating requires the open human-access task (live credentials, provider choice,
key custody, runtime-unwritable `approvers/allowed_signers`). Design calls the owner may revisit:
binding the live account id, the shared `algua-go-live` namespace, and `live revoke` being
agent-allowed.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- `docs/superpowers/specs/2026-09-22-artifact-freeze-design.md` ("Live", decomposition slice 6)
- [#614](https://github.com/Lior-Nis/algua/issues/614),
  [#661](https://github.com/Lior-Nis/algua/issues/661), #254 (atomic authorization write), #329
  (namespaces), #497 (unallocated go-live)
- [Story 1.2](1-2-record-working-tree-deployments.md) AC10,
  [Story 1.3d](1-3d-bind-operational-evidence-and-qualification.md)
- Todoist:
  [Bind signed live authorization to exact deployment](https://app.todoist.com/app/task/bind-signed-live-authorization-to-exact-deployment-6hfCrg4FrJjHwgPG)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
