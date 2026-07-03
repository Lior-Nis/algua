# Live emergency flatten + halt-all (#449)

## Problem
The live command surface has only `live allocate` and `live run-all`. There is **no CLI path to
emergency-exit a LIVE (real-money) strategy or halt the LIVE account**. The only emergency controls
live on the PAPER app:
- `paper flatten` explicitly REJECTS live strategies (`stage not in (PAPER, FORWARD_TESTED)`).
- `paper halt-all` engages the shared global-halt flag but builds the PAPER broker and closes only
  PAPER positions — live positions at the real-money broker are untouched.
- `paper kill` trips the per-strategy kill-switch (stops FUTURE ticks) but never closes an open
  live position; in the live loop a tripped kill-switch raises `TickHalted`, whose handler does not
  flatten.

Net: an operator who detects a misbehaving LIVE strategy can stop new orders but cannot close the
existing real-money positions through any algua command. A kill switch that halts new orders but
cannot exit positions is only half a kill switch.

## Scope of this change
Add two commands to the **live** app (`algua/cli/live_cmd.py`):
1. `algua live flatten <name>` — trip the strategy's kill-switch (unconditional once the strategy is
   confirmed LIVE, fail-safe, BEFORE any authorization check) + submit **provably risk-reducing**
   offsetting close orders for its believed positions (per-strategy scoped, each offset capped to the
   actually broker-held signed quantity).
2. `algua live halt-all` — engage the **global halt** (stop all future ticks in both lanes). Nothing
   more. It NEVER submits close orders in this PR.

Both mirror the risk-reducing structure of the existing paper siblings but operate over the LIVE
lane. `live flatten` DELEGATES the offset-liquidation loop to the single-sourced `flatten_strategy`
helper (`algua/execution/flatten.py`, #336) — the same helper the live-breach handler in
`_run_strategy_tick` and `paper flatten` already call — so this change adds no fourth copy of the
safety-critical control flow. This PR ALSO hardens that shared helper so a live offset can never
increase exposure (see Fork B / finding 2).

**The account-wide market close is DEFERRED and NOT shipped here.** A single-call whole-account
liquidation (`close_all_positions` / `DELETE /v2/positions?cancel_orders=true`) is irreversible,
spans positions AND resting orders belonging to no algua strategy, and cannot be authorized soundly
by any control that exists today (see Fork D). It is routed to the deferred, human-signed
**account-level emergency-liquidation authority** tracked in **#478**. Until that lands, the operator
exits live risk with per-strategy `live flatten <name>` (one call per live strategy) plus
`live halt-all` to stop all future ticks; the raw-broker escalation is the interim break-glass and is
surfaced (with **#478**) in every fail-closed `note`.

Note on wording: these commands **submit** close orders; Alpaca offset fills land asynchronously
(possibly the next open), so a success payload reports *what was submitted*, never that the account is
confirmed flat. The payload says `liquidation_submitted` / `offsets_submitted`, not `flat`.

(`live_cmd.py` and `algua/execution/flatten.py` are not CODEOWNERS-protected, which is relevant only
to merge mechanics — it is NOT a justification for any safety decision in this design. The safety
design stands on its own; see Fork C for why `paper kill` is deliberately left untouched on
architectural, not merge-convenience, grounds.)

## Design decisions (resolved forks)

### Fork A — decouple the risk-REDUCING kill from trade-time artifact authorization (GATE-1 r1 finding 1)
The emergency control has TWO distinct effects, and they need DIFFERENT authority:
- **Stopping future ticks** (tripping the per-strategy kill-switch) is a local DB write that only ever
  REDUCES risk and requires no broker and no signature — it is the operator's inherent control over
  their own registry. It must ALWAYS succeed for a LIVE strategy, in every authorization state.
- **Placing offsetting close orders** requires an `AlpacaLiveBroker`, which requires a verified
  `LiveAuthorization` (a read-only broker refuses POST/DELETE). We obtain it via
  `verify_live_authorization(conn, repo, name, ALLOWED_SIGNERS_PATH)` — the SAME trade-time wall
  `run-all` uses before every order; it re-verifies the human signature over the current artifact
  identity and raises `LiveAuthorizationError` if the strategy is not live / has no matching / a
  revoked authorization.

**The v1 ordering was the bug GATE-1 flagged.** `verify_live_authorization`
(`algua/registry/live_gate.py:181`) fails closed exactly when the strategy is most likely to need an
emergency exit — its authorization was just revoked, or its code/config drifted off the signed
artifact. If we verify FIRST and only then trip the switch, a revoked/drifted LIVE strategy fails the
verify, the command aborts, and the switch is **never tripped** — the live loop keeps ticking it. That
is the "cannot even kill it" hole.

**Fix (adopted): confirm LIVE stage, then TRIP THE KILL-SWITCH BEFORE `verify_live_authorization`.**
`live flatten` resolves the strategy record (LookupError → envelope if it does not exist), asserts it
is `Stage.LIVE` (a plain DB read — see Fork E / finding 4), trips the per-strategy kill-switch + writes
the audit row, and ONLY THEN verifies authorization to build the broker. A revoked/drifted live
strategy is STILL `Stage.LIVE` (revoking a `LiveAuthorization` does not change the lifecycle stage), so
this stage guard does not reintroduce the hole — the strategy is confirmed LIVE and its switch is
tripped even when the broker cannot be built. Tripping can never increase risk (it is a pure halt), so
once the strategy is confirmed LIVE it is unconditional — it does not additionally require the
authorization to verify. On a verify failure the command fails closed but reports
`kill_switch: "tripped"` so the operator KNOWS future ticks are halted, and points at the break-glass
path (raw broker + **#478**) for closing the still-open position.

**Break-glass for closing the open position of a revoked/drifted strategy.** Tripping the switch stops
NEW ticks but does not close the existing position. Two paths cover that:
1. **Long-term (deferred, tracked as #478): a signed account-level emergency-liquidation authority
   DISTINCT from per-strategy trade authorization** — a human-signed grant scoped to the live ACCOUNT,
   independent of any single strategy's artifact identity or revocation state, with its own signing
   ceremony, trust anchor, and audit trail (and a `--break-glass` signed-challenge flatten flow). That
   authority is the correct home for CLOSING (not just halting) a revoked/drifted strategy. Per GATE-1's
   own guidance ("land at least the trip-first fix here … file the signed break-glass path as the
   tracked follow-up") it is a **named, filed follow-up (#478)**, not a silent deferral.
2. **Interim (until #478 exists): escalation to the raw broker** (Alpaca dashboard / API with the live
   keys, `DELETE /v2/positions?cancel_orders=true`) — the account owner's inherent authority, outside
   algua's per-strategy wall. `live flatten` on a revoked/drifted strategy fails closed with a payload
   whose `note` names this path AND **#478**, so the escalation is LOUD and discoverable rather than a
   silent dead end.

### Fork B — human-only, or agent-permitted? And is `live flatten` REALLY risk-reducing? (GATE-1 r1 finding 2 + r2 finding 2)
- **`live flatten` (ONE strategy) — agent-permitted**, matching the paper siblings (`--actor`
  defaults to `agent`, validated via `Actor(actor)`). The agent-permission rationale rests on the
  claim that it **only ever REDUCES risk** — so this PR must MAKE that claim provably true, not merely
  assert it (r2 finding 2).

  **The gap r2 flagged.** The shared helper offsets each *believed* position: it submits
  `submit_offset(symbol, believed_qty)`. If algua's belief has drifted ABOVE the actually-held
  quantity — believed `+10` but the account holds `+3` — the raw offset SELLS 10, driving the real
  position to `−7`: a NEW short. If belief and reality disagree in SIGN — believed `+10`, held `−3` —
  the raw offset sells 10, driving `−3 → −13`, MORE short. Both are exposure-INCREASING, which
  invalidates the agent-permitted rationale.

  **Fix (adopted, in-scope): cap every LIVE offset to the actual broker-held signed quantity.** The
  shared `flatten_strategy` helper gains an optional injected `held: Callable[[], dict[str, float]]`.
  When provided (the LIVE call sites inject it; paper call sites do not), the helper — AFTER `ingest()`
  reconciles landed fills — reads the signed held quantities once and, per believed position:
  - `h = held.get(symbol, 0.0)`; if `abs(h) <= DEFAULT_TOLERANCE` → **skip** (nothing actually held →
    never open a fresh position off a stale belief);
  - `close = min(abs(believed_qty), abs(h))`; if `close <= DEFAULT_TOLERANCE` → skip;
  - `eff = math.copysign(close, h)` — the offset's sign follows the ACTUAL held sign, so
    `submit_offset(symbol, eff, coid)` can only move `|held|` toward zero.

  This is provably risk-reducing: `|eff| ≤ |h|` (never exceeds held quantity) and `sign(eff) = sign(h)`
  (offsetting the held direction), so the post-offset position is `h − eff` with `|h − eff| = |h| −
  close ≥ 0` and the SAME sign as `h` (or exactly zero) — it can NEVER flip sign or increase exposure.
  Bounding by `|believed_qty|` also keeps the per-strategy scope honest: on a shared ticker where the
  account holds more than this strategy is attributed (`believed +3`, `held +10`, a foreign/other-
  strategy `+7`), it closes only `3`, never liquidating the unattributed `7`. Paper call sites pass no
  `held` → the helper's behaviour is byte-identical to today (no paper test churn), because in the
  paper lane the venue ledger IS the position of record.

  `live flatten` AND the live-breach handler (`_run_strategy_tick`, `live_cmd.py` ~194-200) both inject
  `held=lambda: _broker_net_positions(broker)` (the existing helper at `live_cmd.py:258`, which returns
  `{sym: float(q)}` from `broker.get_positions()`), so BOTH live offset paths are provably
  risk-reducing — the finding's rationale applies to any live offset, not just the new command.

- **`live halt-all` — agent-permitted, and pure risk reduction.** It engages the global halt (stop all
  future ticks in both lanes) and does NOTHING else — no broker is built, no order is submitted. Engaging
  the halt is reversible, account-independent, and only ever reduces risk, so it needs no signature and
  no live authorization and is permitted for both actors. The **account-wide market close is not part of
  this command in this PR** — it is deferred to #478 (Fork D). An agent (or human) using `live halt-all`
  gets the halt; whole-account liquidation is reached only through the deferred signed account-level
  authority, or per-strategy via `live flatten`.

### Fork C — the third issue bullet ("paper kill should flatten live positions")
**Deferred, not shipped here.** The clean fix for the "half a kill switch" confusion is that
`live flatten` now exists as the correct command. Overloading `paper kill` (which lives on the paper
app and would need a live authorization + live broker) crosses lanes — a paper-app command reaching
into the live broker/ledger is exactly the lane bleed this codebase's import-linter walls exist to
prevent. That architectural reason alone defers it. (Incidentally `paper_cmd.py` is also
CODEOWNERS-protected, but merge mechanics are NOT why this is deferred — the lane-crossing is.) A
follow-up can add an informational nudge to `paper kill` / `paper show` output when the strategy is
LIVE (pointing to `live flatten`).

### Fork D — the account-wide close is DEFERRED to a distinct signed authority (GATE-1 r1 finding 2 + r2 findings 1 & 3)
`close_all_positions()` is account-wide (`DELETE /v2/positions?cancel_orders=true` — see
`algua/execution/alpaca_broker.py:229`): it market-closes EVERY position AND cancels EVERY resting
order in the account, including positions and orders belonging to other live strategies and to no
algua strategy at all.

**v1 borrowed the first-verified per-strategy signature to authorize this; GATE-1 r1 rejected that.**
The r1 revision then tried to make it sound with an ENFORCED single-tenant-by-quantity invariant plus
`--close-account --actor human`. **GATE-2 r2 rejected THAT on two further axes:**
- **r2 finding 1 — a `--actor human` CLI string is not authentication.** Agent and human drive the
  same CLI, so a flag string is not proof a human acted. The only authentication in the codebase below
  the live wall is the #329 authenticated-human-actor flow — but that flow is bound to a specific
  strategy's artifact identity (`strategy_id` + code/config/dependency hashes + a stage transition; see
  `_common.authenticate_actor`). An account-wide, strategy-less market close has NO artifact to bind, so
  the #329 flow cannot authorize it. Building an account-SCOPED signed challenge is, itself, the deferred
  account-level authority.
- **r2 finding 3 — the single-tenant invariant covered POSITIONS but not ORDERS.** `cancel_orders=true`
  cancels every resting order, including foreign ones, which a positions-only invariant never checks.

**Resolution (adopted — GATE-1 r1 option (b) / r2 findings 1 & 3): defer the whole-account close, in
its entirety, to #478.** Rather than half-authenticate an irreversible whole-account blast radius with
a stretched signature and a positions-only invariant, this PR does NOT ship `close_all_positions()`, does
NOT ship a `--close-account` flag, and does NOT build the live broker inside `halt-all`. `live halt-all`
engages the global halt and stops. The account-wide close — with its own account-scoped signing ceremony,
its orders-inclusive single-tenant assertion (or a close primitive that does not cancel unrelated
orders), and its own audit trail — is #478. This resolves r2 finding 1 (no unauthenticated whole-account
close exists to authorize) and r2 finding 3 (no `cancel_orders=true` is issued, so no foreign resting
order is ever cancelled) by construction. The operator still has a complete, sound live-exit surface
today: per-strategy `live flatten <name>` (provably risk-reducing) for each live strategy, plus
`live halt-all` to stop all future ticks; the raw broker is the interim account-wide break-glass, named
alongside **#478** in the fail-closed `note`.

### Fork E — `live flatten` requires `Stage.LIVE` before it touches the kill-switch (GATE-1 r2 finding 4)
The kill-switch is a shared, cross-lane control. Without a stage guard, `live flatten <some_paper_name>`
would trip a PAPER/research strategy's kill-switch through the LIVE command surface — halting that
strategy's paper ticks: a cross-lane denial-of-service reachable from the live app. **Fix:** immediately
after `rec = repo.get(name)`, assert `rec.stage is Stage.LIVE`; a non-LIVE strategy fails closed
(envelope, exit 1) with NO kill-switch write. Stage is a plain DB read that needs no live-artifact
authorization, so this guard is orthogonal to Fork A: a revoked/drifted strategy is still `Stage.LIVE`
(revocation touches the `LiveAuthorization`, not the stage), so it still trips-then-fails-closed exactly
as Fork A requires. Only genuinely non-LIVE strategies (paper/backtested/idea/…) are refused before any
switch write.

## Command specs

### `live flatten <name> [--actor human|agent]`
`@json_errors(ValueError, LookupError, BrokerError)`
1. `actor_enum = Actor(actor)` — fail fast on a bad actor before touching a switch.
2. `with registry_conn() as conn:`
3. `repo = SqliteStrategyRepository(conn)`; `rec = repo.get(name)` — LookupError (→ envelope, exit 1)
   if the strategy does not exist; nothing has been tripped yet.
4. **Require `Stage.LIVE` (Fork E — before any switch write):** if `rec.stage is not Stage.LIVE`,
   `emit({"ok": False, "strategy": name, "error": f"live flatten requires a LIVE strategy; "
   f"{name} is {rec.stage.value}", "kill_switch": "not_tripped"})` + `raise typer.Exit(1)`. The
   kill-switch is NOT written for a non-LIVE strategy, so the live surface cannot DoS a paper/research
   strategy.
5. **Trip the kill-switch (Fork A — fail-safe, BEFORE any authorization check):**
   `kill_switch.trip(conn, name, reason="flatten", actor=actor_enum.value)` + `audit_append(...,
   action="flatten", ...)`. This halts future ticks even for a revoked/drifted LIVE strategy whose
   broker cannot be built, and can never increase risk.
6. **Now verify authorization to build the broker**, catching the failure explicitly so the fail-closed
   payload can report the switch state and point at break-glass:
   ```python
   try:
       authorization = verify_live_authorization(conn, repo, name, ALLOWED_SIGNERS_PATH)
   except LiveAuthorizationError as exc:
       emit({"ok": False, "strategy": name, "kill_switch": "tripped",
             "liquidation_submitted": False, "error": str(exc),
             "note": "future ticks are halted (kill-switch tripped); the open position was NOT "
                     "closed — the live authorization is revoked/absent or the artifact drifted. "
                     "Break-glass: close via the raw broker (DELETE /v2/positions?cancel_orders=true "
                     "with the live keys). A signed account-level break-glass path is tracked in #478."})
       raise typer.Exit(1) from exc
   ```
   This is the documented emergency-authority gap failing closed VISIBLY: the strategy is provably
   halted and the operator is handed the exact escalation + the follow-up issue, not a silent dead end.
7. `broker = _alpaca_live_broker(authorization)`.
8. **Delegate the offset-liquidation loop to the single-sourced helper, injecting `held` so every
   offset is capped to the actually-held signed quantity (Fork B / finding 2):**
   ```python
   res = flatten_strategy(
       conn, broker, name, LedgerKind.LIVE, lane="live",
       cancel=lambda: _scoped_cancel(conn, broker, name),
       ingest=lambda: ingest_activities(
           conn, _broker_account_activities(broker, fill_cursor(conn, LedgerKind.LIVE)),
           LedgerKind.LIVE),
       held=lambda: _broker_net_positions(broker),
   )
   ```
   The helper cancels only this strategy's resting orders (scoped `cancel`), reconciles fills up to the
   live broker clock (`ingest`), reads `held()` once, then for every believed position records → caps to
   held (`min(|believed|,|held|)`, sign of held, skip if nothing held) → `submit_offset` → backfills the
   broker order id, counting `n_offsets`. It fails SAFE: ANY exception is captured into
   `res.flatten_error` (+ an audited `flatten_failed` row), never a raw traceback. There is NO
   hand-rolled loop in `live flatten` and NO `_RECONCILE_TOL` — the dust skip and the held-cap both use
   `DEFAULT_TOLERANCE` INSIDE the helper.
   - **Payload contract — mirror `paper flatten`** (`paper_cmd.py:685-693`): if
     `res.flatten_error is not None`, `emit(breach_payload(res.flatten_error, strategy=name,
     liquidation_submitted=False, offsets_submitted=res.n_offsets))` + `raise typer.Exit(1)`. The
     kill-switch (tripped in step 5) STAYS tripped — a partial/failed liquidation must not silently
     re-enable the strategy. `res.n_offsets` gives correct partial-liquidation reporting even when the
     loop errored part-way.
9. Success: `emit(ok({"strategy": name, "kill_switch": "tripped",
   "liquidation_submitted": res.n_offsets > 0, "offsets_submitted": res.n_offsets}))`.
   `liquidation_submitted` reflects whether any offset order ACTUALLY went out (an already-flat strategy,
   or one whose belief exceeds a now-zero held quantity, submits none → `False`, not a phantom-
   liquidation `True`); accepted offset fills land ASYNC — "submitted", never "confirmed flat".

### `live halt-all --reason R [--actor human|agent]`
`@json_errors(ValueError, LookupError, BrokerError)`
1. `actor_enum = Actor(actor)`.
2. `with registry_conn() as conn:`
3. **Engage the global halt (fail-safe, agent OR human — the always-reachable action):**
   `global_halt.engage(conn, reason=reason, actor=actor_enum.value)` + `audit_append(...,
   action="halt_all", strategy=None)`. This alone stops all future ticks in both lanes.
4. Success: `emit(ok({"global_halt": "set", "liquidation_submitted": False,
   "note": "global halt engaged — all future ticks (paper AND live) are stopped. This command does NOT "
   "close open positions. To exit LIVE positions: run 'live flatten <name>' per live strategy (each is "
   "provably risk-reducing). A single-call account-wide close requires the signed account-level "
   "emergency-liquidation authority tracked in #478; the interim account-wide break-glass is the raw "
   "broker (DELETE /v2/positions?cancel_orders=true with the live keys)."}))` (exit 0).

   `live halt-all` builds NO broker, submits NO order, and takes NO `--close-account` flag in this PR.
   The account-wide close is entirely #478.

## New imports in `live_cmd.py`
- `Actor` added to the existing `from algua.contracts.lifecycle import Stage` →
  `from algua.contracts.lifecycle import Actor, Stage`.

That is the only new import in `live_cmd.py`. `live flatten` delegates to `flatten_strategy` (already
imported at `live_cmd.py:17`) and reuses `_broker_net_positions` (already defined at `live_cmd.py:258`)
for the `held` callable, so **no `_RECONCILE_TOL` and no `DEFAULT_TOLERANCE` import is added** to
`live_cmd.py` — the held-cap tolerance lives entirely inside the helper. Everything else the two
commands need (`flatten_strategy`, `global_halt`, `kill_switch`, `fill_cursor`, `ingest_activities`,
`believed_positions`, `audit_append`, `_scoped_cancel`, `_alpaca_live_broker`,
`verify_live_authorization`, `ALLOWED_SIGNERS_PATH`, `SqliteStrategyRepository`,
`LiveAuthorizationError`, `BrokerError`, `LedgerKind`, `_broker_account_activities`,
`_broker_net_positions`, `breach_payload`, `ok`, `emit`, `registry_conn`) is already imported/defined.

## Changes in `algua/execution/flatten.py` (finding 2 — shared helper hardening)
Add an optional keyword `held: Callable[[], dict[str, float]] | None = None` to `flatten_strategy`
and, when it is not `None`, cap each offset (see Fork B for the exact rule). Requires `import math`
for `math.copysign`. Behaviour is UNCHANGED when `held is None` (the paper call sites), so
`tests/test_flatten.py` and the paper-lane tests are not perturbed; new helper tests cover the capped
path (below). The `held()` call happens INSIDE the existing `try` (after `ingest()`), so a
`get_positions()` failure still fails SAFE into `flatten_error`.

## Tests

### `tests/test_flatten.py` (helper — the cap is single-sourced, so test it here)
- `test_flatten_caps_offset_to_held_quantity`: believed `{"AAA": 10.0}`, `held()` returns
  `{"AAA": 3.0}` → `submit_offset("AAA", 3.0, ...)` (capped to held, NOT 10), `n_offsets == 1`.
- `test_flatten_skips_symbol_with_no_held_position`: believed `{"AAA": 10.0}`, `held()` returns `{}`
  (nothing actually held) → `submit_offset` NEVER called, `n_offsets == 0` (no fresh position opened
  off a stale belief).
- `test_flatten_sign_disagreement_reduces_exposure`: believed `{"AAA": 10.0}`, `held()` returns
  `{"AAA": -3.0}` → `submit_offset("AAA", -3.0, ...)` (sign follows HELD → buys to cover the short,
  never sells into a deeper short); the post-offset position moves toward zero.
- `test_flatten_shared_symbol_closes_only_attributed`: believed `{"AAA": 3.0}`, `held()` returns
  `{"AAA": 10.0}` → `submit_offset("AAA", 3.0, ...)` (closes only the attributed 3, leaves the foreign
  7 — per-strategy scope preserved).
- `test_flatten_held_none_unchanged`: `held=None` → the loop offsets the full believed qty exactly as
  before (pins paper-lane behaviour is byte-identical).
- `test_flatten_held_getter_failure_fails_safe`: `held()` raises → captured into `flatten_error`, an
  audited `flatten_failed` row is written, no traceback propagates.

### `tests/test_cli_live.py` — `live flatten` (reuse `_to_live`, `_auth`, `_isolated`)
Fake live broker per test (like `_LiqBroker` at line 268): `account_activities`, `get_positions`
(configurable pandas Series), `list_open_orders`→[], `cancel_order`, `submit_offset` (records calls),
plus a `fail` flag to raise `BrokerError`.

**Assert DELEGATION to `flatten_strategy`, not a re-implementation.** The load-bearing assertion is
that the command calls `flatten_strategy` with the LIVE arguments (including a `held` callable) and then
honours the returned `FlattenResult`; the loop's internal behaviour (scoped cancel, ingest, held-cap,
record→submit→backfill, fail-safe capture) is covered by the helper's own tests above and is NOT
re-litigated here.

- `test_live_flatten_delegates_to_flatten_strategy`: monkeypatch `live_cmd.flatten_strategy` with a
  spy; run `live flatten`. Assert it was called EXACTLY once with `kind=LedgerKind.LIVE`, `lane="live"`,
  and `cancel`/`ingest`/`held` callables — the LIVE args. Assert the DB kill-switch is tripped BEFORE
  the helper is invoked (spy records `kill_switch.is_tripped(...)` was already true at call time). Spy
  returns `FlattenResult(n_offsets=1, flatten_error=None)`; assert the success payload is
  `{"strategy": name, "kill_switch": "tripped", "liquidation_submitted": True, "offsets_submitted": 1}`.
- `test_live_flatten_non_live_stage_refused_no_trip` **(Fork E / finding 4 — load-bearing new test)**:
  a strategy in a non-LIVE stage (e.g. PAPER) → `ok False`, `kill_switch: "not_tripped"`, exit 1;
  `flatten_strategy` NEVER called; the DB kill-switch is NOT written (assert `kill_switch.is_tripped`
  is False after). Pins that the live surface cannot DoS a paper/research strategy.
- `test_live_flatten_already_flat_reports_not_submitted`: spy returns
  `FlattenResult(n_offsets=0, flatten_error=None)` → payload `liquidation_submitted False`,
  `offsets_submitted 0`, `kill_switch tripped`.
- `test_live_flatten_error_stays_tripped`: spy returns `FlattenResult(n_offsets=2,
  flatten_error="boom")` → command emits `breach_payload("boom", strategy=name,
  liquidation_submitted=False, offsets_submitted=2)`, exits 1, DB kill-switch STILL tripped.
- `test_live_flatten_revoked_live_trips_then_fails_closed` **(Fork A finding-1 fix)**: a LIVE strategy
  whose authorization does NOT verify (REAL `verify_live_authorization` raises `LiveAuthorizationError`
  — revoked / artifact-drift fixture). Assert: (1) the DB kill-switch IS tripped (trip precedes verify);
  (2) `flatten_strategy` is NEVER called and no live broker is built; (3) the payload is `{"ok": False,
  "strategy": name, "kill_switch": "tripped", "liquidation_submitted": False, ...}` with a `note` naming
  the raw-broker break-glass path AND `#478`, exit 1.
- `test_live_flatten_missing_strategy`: unknown name → LookupError envelope, exit 1, kill-switch never
  written (existence check precedes the trip).
- `test_live_flatten_offsets_and_trips` (END-TO-END, `_LiqBroker`, NO spy): believed `{"AAA": 5.0}`,
  `get_positions()` returns `{"AAA": 5.0}` → `submit_offset("AAA", 5.0)` called, `live_orders` row
  `side='sell'` with `broker_order_id` backfilled, payload `liquidation_submitted True`. Exercises the
  real wiring through the shared helper (with the `held` callable), including the cap.
- `test_live_flatten_belief_exceeds_held_caps_offset` (END-TO-END, `_LiqBroker`, NO spy): believed
  `{"AAA": 10.0}` but `get_positions()` returns `{"AAA": 3.0}` → `submit_offset("AAA", 3.0)` (capped),
  proving the command's injected `held` reaches the helper and bounds the live offset.

### `tests/test_cli_live.py` — `live halt-all`
- `test_live_halt_all_engages_halt_only`: `--actor agent`. DB `global_halt` engaged; payload
  `ok True`, `liquidation_submitted False`, `note` present (names `#478` + the raw-broker break-glass);
  exit 0; NO broker built and NO order submitted (assert `_alpaca_live_broker` is NEVER invoked via a
  monkeypatched sentinel).
- `test_live_halt_all_human_same_as_agent`: `--actor human` → identical halt-only behaviour (there is
  no account-wide close in this command for either actor); DB `global_halt` engaged, exit 0, no broker.
- `test_live_halt_all_no_close_account_flag`: invoking `live halt-all --close-account` errors as an
  UNKNOWN option (the flag does not exist in this PR — the account-wide close is #478), pinning that
  the whole-account blast radius is not reachable from this command.

## Quality gate
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

## What this PR fixes vs. what is a tracked follow-up (#478)
- **r1 finding 1(a) — trip-before-verify:** `live flatten` trips the kill-switch before any
  authorization check (Fork A), so a revoked/drifted live strategy is provably halted.
- **r2 finding 2 — provably risk-reducing:** every LIVE offset is capped to the actually-held signed
  quantity in the shared helper (Fork B) — it can never exceed held quantity or flip sign, so the
  agent-permission of `live flatten` rests on a true claim.
- **r2 finding 4 — no cross-lane DoS:** `live flatten` requires `Stage.LIVE` before touching the
  kill-switch (Fork E).
- **r1 finding 2 + r2 findings 1 & 3 — the unsound whole-account close is not shipped:** the
  account-wide `close_all_positions()`, its `--close-account` flag, and the `halt-all` broker build are
  ALL deferred to the signed account-level authority (Fork D / #478). No unauthenticated whole-account
  close exists (finding 1) and no `cancel_orders=true` is issued, so no foreign resting order is
  cancelled (finding 3).
- **r2 finding 5 — real filed follow-up:** the account-level emergency-liquidation authority is filed
  as **#478**, and #478 appears in every fail-closed `note` (both commands), so the emergency-close gap
  cannot silently become permanent.

## Out of scope / tracked follow-ups
- **Signed account-level emergency-liquidation authority + in-algua break-glass flatten — filed as
  #478.** A human-signed grant scoped to the live ACCOUNT (its own signing ceremony / trust anchor /
  audit trail), independent of any per-strategy `LiveAuthorization` and of a strategy's
  stage/artifact/revocation state, plus a `--break-glass` account-wide flatten flow with an
  ORDERS-inclusive single-tenant assertion (or a close primitive that does not cancel unrelated
  orders). It is the correct authority for both (a) CLOSING (not just halting) a strategy whose
  per-strategy authorization is revoked/drifted, and (b) the account-wide close. Until it exists, the
  interim break-glass is escalation to the raw broker (Alpaca dashboard /
  `DELETE /v2/positions?cancel_orders=true` with the live keys), surfaced with #478 in the fail-closed
  `note` text of both commands.
- A convenience `live halt-all` mode that ALSO runs the per-strategy provably-risk-reducing
  `flatten_strategy` over every `Stage.LIVE` strategy (halt + flatten-all-attributed, touching NO
  foreign position or order) is a possible future primitive; it belongs with or after #478 and is not
  built here.
- `paper kill` / `paper show` informational nudge for LIVE strategies (pointing at `live flatten`).
  Deferred because a paper-app command must not reach into the live lane (import-linter walls), not for
  merge-mechanics reasons. Follow-up issue.
