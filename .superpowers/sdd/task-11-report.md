### Task 11 Report: Fail-closed evidence + clock-resilience refactor

**Status:** DONE  
**Commit:** f78d207  
**Tests:** 2087 passed, 0 failed (full suite)

---

#### What was done

**algua/cli/paper_cmd.py**

1. `_ingest_paper_venue` signature changed from `(conn, broker)` to `(conn, broker, until: str)`.
   Body no longer calls `broker.clock()` — the caller passes `until` as the window upper bound.

2. In `trade_tick`, the broker clock is now resolved ONCE immediately after `broker.account()`:
   ```python
   tick_ts, clock_source = tick_clock(broker.clock)
   ```
   `tick_clock` already catches `BrokerError`/`ValueError`/`TypeError` and falls back to local
   clock — so a clock-down venue does NOT abort the tick (resilient).

3. The normal-path ingest is now fail-closed:
   ```python
   try:
       _ingest_paper_venue(conn, broker, tick_ts)
   except BrokerError as exc:
       audit_append(conn, actor="system", action="venue_ingest_failed", reason=str(exc), strategy=name)
       emit(breach_payload(str(exc), strategy=name, kind="venue_ingest_failed"))
       raise typer.Exit(1) from exc
   ```
   A fetch failure exits before `run_tick` is called → no `record_tick_snapshot` → no
   `reconcile_ok=True` fabricated.

4. The duplicate `tick_ts, clock_source = tick_clock(broker.clock)` call at the snapshot site
   (~line 380) was removed; the values from step 2 are reused.

5. The breach-handler ingest (inside `except RiskBreach`) now passes `tick_ts`:
   `_ingest_paper_venue(conn, broker, tick_ts)`.

6. The `flatten` command calls `broker.clock()` inline:
   `_ingest_paper_venue(conn, broker, broker.clock())` — already inside a `try/except BrokerError`
   block; no resilience fallback needed there (no snapshot to record).

**tests/test_cli_paper.py**

- `test_trade_tick_unusable_broker_clock_falls_back_to_local`: removed the `_ingest_paper_venue`
  monkeypatch (which was hiding the clock→ingest coupling). Added
  `account_activities_window(self, after, until): return []` to `_ClockFailBroker` so the
  now-real ingest path runs harmlessly. Test still asserts `clock_source="local"`.

**tests/test_paper_venue_reconcile.py**

- Task-3 direct call sites updated: `_ingest_paper_venue(c, broker)` →
  `_ingest_paper_venue(c, broker, broker.clock())`.

- NEW `test_trade_tick_fails_closed_on_ingest_fetch_failure`: broker whose
  `account_activities_window` raises `BrokerError`; asserts exit_code==1,
  `payload["kind"]=="venue_ingest_failed"`, and `latest_tick_snapshot` is `None`.

- NEW `test_trade_tick_survives_broker_clock_failure`: broker whose `clock()` raises `BrokerError`
  but `account_activities_window` returns `[]`; asserts exit_code==0 and
  `snap["clock_source"]=="local"`.

---

#### Answers to key questions

- **Did `breach_payload` accept `kind`?** Yes — `breach_payload(error, **extra)` passes all kwargs
  through; `kind="venue_ingest_failed"` works. Existing code already uses `kind=exc.kind`.

- **Duplicate `tick_clock` call removed?** Yes — the second call at the snapshot site is gone.

- **Monkeypatch removed?** Yes — the `_ingest_paper_venue` lambda monkeypatch is removed from
  `test_trade_tick_unusable_broker_clock_falls_back_to_local`. `_ClockFailBroker` now has
  `account_activities_window`.

- **Unused type: ignore?** Fixed — the `# type: ignore[attr-defined]` added to `broker.clock()`
  in `flatten` was unnecessary (mypy knows the type is `AlpacaPaperBroker`); removed.

- **Concerns:** None. All 2087 tests pass, ruff clean, mypy clean, lint-imports clean.

---

### Review-round fixes (commit 8f4bec1)

**Status:** DONE  
**Commit:** 8f4bec1  
**Covering tests:** `uv run pytest tests/test_paper_venue_reconcile.py tests/test_cli_paper.py -q` → **60 passed**  
**Full gate:** 2089 passed, ruff clean, mypy clean, lint-imports clean.

#### FIX #1 — flatten uses resilient clock bound (`algua/cli/paper_cmd.py`)

Changed `_ingest_paper_venue(conn, broker, broker.clock())` to:
```python
flat_ts, _ = tick_clock(broker.clock)
_ingest_paper_venue(conn, broker, flat_ts)
```
A `BrokerError` from `broker.clock()` now falls back to local time via `tick_clock`, so the
offset loop runs even when the clock endpoint is down.

**Test added:** `test_flatten_still_offsets_when_broker_clock_raises` in
`tests/test_paper_venue_reconcile.py` — seeds a believed AAA position, patches a broker whose
`clock()` raises `BrokerError` but `account_activities_window` returns `[]` and `submit_offset`
records calls; asserts `exit_code==0`, `liquidation_submitted=True`, and `submit_offset` was
called for AAA.

#### FIX #2 — breach handler uses a fresh clock bound (`algua/cli/paper_cmd.py`)

Changed the breach-handler re-ingest from `_ingest_paper_venue(conn, broker, tick_ts)` to:
```python
# Re-ingest up to NOW (not the stale top-of-tick ts) so fills that landed during this
# tick are reflected in the belief the offset loop liquidates. tick_clock keeps this
# resilient to a broker-clock outage (local fallback).
breach_ts, _ = tick_clock(broker.clock)
_ingest_paper_venue(conn, broker, breach_ts)
```
The top-of-tick ingest already advanced the cursor to `tick_ts`; reusing it gave an empty
`(tick_ts, tick_ts]` window — a regression vs the prior `(T1, now]` behavior.

**Verification:** existing breach tests (`test_trade_tick_breach_flattens_dropped_symbol_and_clears_belief`
etc.) still pass; no new test added (the window math is an observable of the fill ledger state,
not the exit code path covered by FIX #1/#3 tests).

#### FIX #3 — widen ingest catch to `Exception` (`algua/cli/paper_cmd.py`)

Changed `except BrokerError as exc:` to `except Exception as exc:` on the normal-path ingest
wrap. A `RuntimeError` / `OSError` / `JSONDecodeError` from the broker transport now:
- emits `{"ok": false, "kind": "venue_ingest_failed", ...}` (structured payload)
- calls `audit_append` before exiting
- exits non-zero before `run_tick` (no `record_tick_snapshot`, no fabricated `reconcile_ok=True`)

**Test added:** `test_trade_tick_fails_closed_on_non_brokererror_ingest_failure` in
`tests/test_paper_venue_reconcile.py` — broker whose `account_activities_window` raises
`RuntimeError`; asserts `exit_code==1`, `payload["kind"]=="venue_ingest_failed"`, and
`latest_tick_snapshot` is `None`.
