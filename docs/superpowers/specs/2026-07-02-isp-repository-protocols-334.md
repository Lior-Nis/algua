# Spec: Split the StrategyRepository god-Protocol into narrow role protocols (#334)

## Intent
`algua/registry/repository.py` declares one 45-method `StrategyRepository` Protocol spanning ~8
bounded contexts, and `algua/registry/store.py`'s `SqliteStrategyRepository` implements it. Any
alternative backing store must implement all 45 methods, and every consumer depends on the whole
interface regardless of the slice it uses — an Interface-Segregation-Principle violation. Split the
Protocol into narrow, cohesive role protocols so callers depend only on the slice they use.

This is a **behavior-preserving refactor**: no SQL changes, no method signature changes, no changes
to `store.py`'s concrete implementation or its atomic `BEGIN IMMEDIATE` critical sections
(#161/#192/#193/#246/#247/#339). Docstrings move verbatim with their methods.

## Design (GATE-1 APPROVED, Codex, 2 rounds)

### Narrow read slices
- `StrategyReader(Protocol)`: `get`
- `StrategyLister(Protocol)`: `list_strategies`

### Bounded-context protocols
- `StrategyStore(StrategyReader, StrategyLister, Protocol)`: `add`, `update_metadata`,
  `backfill_metadata`, `default_fill_metadata_nulls`, `delete`, `list_transitions`,
  `apply_transition` (composes the two readers + write/lifecycle methods)
- `ApprovalLedger(Protocol)`: `record_approval`, `has_valid_approval`
- `SearchBreadthLedger(Protocol)`: `record_search_trial`, `pooled_trial_sharpe_var`,
  `funnel_trial_sharpe_var`, `total_search_combos`, `windowed_search_combos`
- `HoldoutLedger(Protocol)`: `reserve_holdout`, `finalize_holdout_reservation`,
  `release_holdout_reservation`, `record_holdout_returns`, `overlapping_holdout_return_streams`
- `GateLedger(Protocol)`: `record_gate_evaluation`, `find_consumable_gate_evaluation`,
  `fdr_stream_state`, `record_gate_with_fdr_and_maybe_promote`
- `ForwardGateLedger(Protocol)`: `record_forward_gate_evaluation`, `record_forward_pass_and_promote`,
  `find_consumable_forward_gate_evaluation`, `latest_forward_gate_row`
- `FactorLedger(Protocol)`: `record_factor_evaluation`, `factor_hypothesis_breadth`,
  `windowed_factor_irs`, `finalize_factor_evaluation`
- `FamilyGraph(Protocol)`: `create_family`, `assign_strategy_to_family`, `strategy_family`,
  `family_ancestry`, `add_parent_edge`, `all_families_with_member_profiles`,
  `windowed_family_combos`, `lifetime_combos_for_families`, `family_lifetime_combos`, `family_names`
- `BacktestReturnsLedger(Protocol)`: `persist_backtest_returns`, `load_backtest_returns`

### Composed superset (unchanged 45-method structural contract)
```python
class StrategyRepository(
    StrategyStore, ApprovalLedger, SearchBreadthLedger, HoldoutLedger, GateLedger,
    ForwardGateLedger, FactorLedger, FamilyGraph, BacktestReturnsLedger, Protocol
): ...
```
Structurally identical to the old 45-method Protocol, so every untouched broad consumer
(`transitions.py`, `promotion.py`, `forward_promotion.py`) keeps working byte-for-byte.

### Composed consumer protocol (inheritance, NOT union)
```python
class ApprovalRepository(StrategyReader, ApprovalLedger, Protocol): ...
```
A union (`A | B`) means "either", so mypy would forbid calling methods of both — inheritance is
required for "needs both".

### Call-site narrowing (only clean single/paired-context consumers; no unions)
- `approvals.py`: `repo` typed `ApprovalRepository` (needs `get` + approval methods)
- `live_gate.py` `verify_live_authorization`: `repo` typed `StrategyReader` (only `get`)
- `lineage.py` `dependents_of`: `repo` typed `StrategyLister` (only `list_strategies`)
- `transitions.py`, `forward_promotion.py`, `promotion.py`: LEFT on the full composed
  `StrategyRepository`. They genuinely span contexts and carry the atomic promotion methods;
  `transitions.py`'s callback aliases (`ApprovalVerifier`/`ForwardCertificateVerifier`) keep taking
  `StrategyRepository` so nothing narrows underneath them.

`store.py` is untouched: `SqliteStrategyRepository` structurally satisfies every narrow protocol.

## Safety / invariants
- No SQL moved; all `BEGIN IMMEDIATE` sections byte-for-byte identical.
- CLI files import the concrete `SqliteStrategyRepository` directly — unaffected.
- Backed by the existing test suite; `uv run mypy algua` catches any protocol variance issue.
