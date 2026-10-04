---
baseline_commit: e875b6d
---

# Story 2.4: Hand a frozen deployment off from paper and execute it in the live lane

Status: backlog

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR1 (live portion), FR6 and FR10 (retained for live), FR12 (live deployment,
invocation and tick linkage), NFR1–NFR6.
Depends on: Story 2.3 (a frozen deployment can be authorized and re-verified for live).
Gated by: Story 2.3 merged, then contract and readiness review. No owner decision is needed.

## Story

As Algua's operator,
I want an authorized frozen deployment to leave the paper book cleanly and then run in the live lane
from the same verified bundle and environment that earned its evidence,
so that live trades come from exactly the artifact the human approved, under the same supervisor
walls as paper.

## Context

The live lane runs the checkout today:

- `_run_strategy_tick` loads the strategy module from the checkout (`algua/cli/live_cmd.py:151`) and
  resolves the deployment through `deploy.resolve_tick` (`live_cmd.py:158`,
  `algua/registry/deployment_runtime.py:12-29`), whose `require_tick_deployment` refuses a frozen
  row with `frozen_live_unsupported`.
- The live cycle plan passes no frozen views (`live_cmd.py:434-436`), so a frozen tenant would be
  planned from the checkout. The paper lane passes them (`algua/cli/lane_refresh.py:72-73`).
- The frozen planner port is built in `algua/cli/paper_cmd.py:613-634` (`_frozen_planner`); command
  modules may not import each other (`pyproject.toml` independence contract), so the live lane
  cannot reuse it as it stands.
- Paper tenant resolution routes by `source_kind` (`algua/registry/frozen_runtime.py:176-237`) and
  isolates tenant faults before effects (`frozen_runtime.py:252-279`). The tick-to-invocation link
  trigger is lane-agnostic (`algua/registry/db/frozen_evidence.py:108-128`), and the tick guard
  already accepts `lane='live'` with stage `live` (`algua/execution/tick_snapshots.py:63-78`).
- `paper resume` of a LIVE strategy loads the checkout module to find its universe
  (`paper_cmd.py:258-261`).
- `live_cmd.py` sits at exactly its 749-line pin (`tests/test_module_size_ratchet.py:56`).

The handoff has no clean path today. Go-live requires the paper slice to be flat and drained
(`algua/registry/store/crud.py:333-357`, Story 2.2). The certificate requires the kill switch to be
clear at go-live (`algua/registry/live_certificate.py:147-149`), yet `paper flatten` trips it
(`paper_cmd.py:1295`). Clearing it with `paper resume` lets the next paper cycle re-buy the
positions, because a `forward_tested` tenant keeps ticking while it waits for the signature
(`algua/registry/gating.py:20-22`). A strategy that holds positions can therefore only go live by
racing the paper timer. Paper `run-all` skips a paper-lane tenant without an active allocation
(`paper_cmd.py:944-946`), which is the hook a handoff needs.

## Scope and authority

In scope: a paper-to-live handoff command; live tenant resolution and frozen dispatch for frozen
deployments; invocation evidence and tick linkage for live frozen ticks; frozen live cycle planning;
`paper resume` of a frozen live strategy; the `live_cmd.py` carve.

Must not: let the planner child see live credentials, the registry or a broker; move any risk wall,
reservation, cancel, submission, reconcile or ledger write into the child; change paper behavior;
change the signing ceremony (Story 2.3) or capital (Stories 2.5, 2.6); delete the working-tree or
legacy live paths (owner deferral of 2026-10-01; none can reach live after Story 2.3). The handoff
is risk-reducing and agent-allowed; going live stays the human's signed act.

## Acceptance criteria

1. **Handoff.** `paper handoff NAME` (stage `forward_tested` only, agent-allowed, under
   `operator.lock`) revokes the paper allocation so paper `run-all` stops ticking the tenant,
   cancels its own resting paper orders and offsets its believed paper positions through the
   existing `flatten_strategy` path, without tripping the kill switch and without writing a tick
   snapshot. It is audited (`paper_handoff`), idempotent, and reports unsold or pending offsets like
   `paper flatten`. The tenant can rejoin paper through `paper allocate` if the go-live is
   abandoned.
2. **Certificate survives the handoff.** After the handoff fills land, the certificate still
   verifies: no kill-switch trip, no new evidence tick, every fill attributable. An end-to-end test
   runs certificate, handoff, fills, signed go-live (Story 2.3) and the first live tick.
3. **Live tenant resolution.** A live strategy whose active deployment is frozen resolves from its
   recorded descriptor, freshly verified bundle and environment, and the deployment's gate-bound
   universe. The checkout module is never imported and the identity is never recomputed.
4. **Frozen dispatch.** Phase A and Phase B run in fresh children through the same frozen port as
   paper, moved to one shared helper both lanes import. Every attempt is recorded in
   `frozen_invocations`. The live tick snapshot links its successful Phase B invocation and carries
   the deployment id and bars snapshot id; a frozen live tick without a snapshot id is refused
   before dispatch.
5. **Supervisor keeps every authority.** The live supervisor alone performs fill ingest, reconcile,
   book breakers, book exposure, buying-power reservation, scoped cancel, submission, ledger writes
   and authorization checks, in today's order. A test proves the live credentials
   (`ALGUA_ALPACA_LIVE_*`) and the registry path never reach the child (the Story 1.3c scrub).
6. **Tenant isolation.** A content, protocol, timeout or invalid-result failure is that tenant's
   stable-coded setup error with zero cancel, submit or downstream effect for it; siblings continue.
   Shared-infrastructure faults abort the cycle, exactly as in paper.
7. **Planning and resume.** The live cycle plan reads frozen views, never the checkout.
   `paper resume` of a frozen live strategy takes its universe from the deployment view.
8. **Parity and restart.** For representative normal, early-return and breach fixtures, live frozen
   effect traces equal the paper frozen and in-process traces. After a restart without a Git
   checkout the same deployment resolves and ticks identically.
9. **Structure.** `live_cmd.py` shrinks: the per-strategy tick moves to a CLI helper module and the
   pin is lowered, never raised. New protected modules join CODEOWNERS and the integrity-critical
   set. Lane parity tests cover the frozen live route. Full gate passes; protected review.

## Tasks / subtasks

- [ ] Contract and readiness: handoff sequence, live tenant contract, effect-order trace,
      entry-point inventory (`live run-all`, `live flatten`, the live exit guard, `paper resume`).
- [ ] Carve `_run_strategy_tick` out of `live_cmd.py`; move the frozen port to a shared helper
      (AC9).
- [ ] Live tenant resolution and routing (AC3, AC6).
- [ ] Frozen dispatch, invocation recording and tick linkage in live (AC4–AC5).
- [ ] Frozen cycle planning and `paper resume` universe (AC7).
- [ ] `paper handoff` command (AC1–AC2).
- [ ] Parity, restart and isolation tests (AC5–AC8); full gate; independent review.

## Dev notes

- Reuse, do not copy. `frozen_runtime._resolve_frozen` couples resolution to `require_paper_gates`;
  generalize the gate check (stage set plus kill switch plus global halt) rather than duplicating
  resolution for live.
- The live sizing snapshot hook (`live_cmd.py:177-178`) and `build_live_sizing_snapshot` take the
  universe; pass the frozen view's universe.
- The handoff must hold `operator.lock` so it cannot interleave with a paper cycle, and must revoke
  the allocation before cancelling, so a cycle that starts after it never re-buys.
- `flatten_strategy` does not trip the kill switch itself; the callers do. Do not route the handoff
  through `paper flatten`.
- A fleet-health `stale` verdict for an unallocated `forward_tested` tenant after several sessions
  is expected during a long handoff; go live promptly or re-allocate.

### Test matrix

Frozen live tick normal, early-return and breach parity; missing, corrupt and unsupported content;
child timeout and invalid result; sibling continues; systemic abort; snapshot id required; restart
without checkout; handoff with positions, with resting orders, idempotent re-run, abandoned go-live
and re-allocation; certificate valid after handoff; child environment scrub.

## Owner decisions

None. Design calls the owner may revisit: the handoff as a separate agent-allowed command (rather
than flattening inside the signed ceremony), and leaving the paper book at go-live (PRD §16 suggests
paper could keep shadowing a live deployment; the current one-stage model does not allow it).

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- [Story 1.3c](1-3c-execute-frozen-planners-in-paper.md) and its contract (dispatch, limits, failure
  classes); [Story 1.3d](1-3d-bind-operational-evidence-and-qualification.md) (invocation evidence)
- `docs/superpowers/specs/2026-09-22-artifact-freeze-design.md` ("Live")
- #497 (book-exit wind-down), [#685](https://github.com/Lior-Nis/algua/issues/685) (Story 2.2)
- Todoist:
  [Bind signed live authorization to exact deployment](https://app.todoist.com/app/task/bind-signed-live-authorization-to-exact-deployment-6hfCrg4FrJjHwgPG)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
