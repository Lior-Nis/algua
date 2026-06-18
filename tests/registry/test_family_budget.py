from algua.registry.family_budget import FamilyBudgetLedger, InMemoryFamilyBudgetLedger


def test_reserve_within_cap():
    ledger = InMemoryFamilyBudgetLedger(global_cap=1.0)
    key = ledger.reserve(1, p_value=0.3, actor="agent")
    assert key is not None
    assert abs(ledger.family_wealth(1) - 0.7) < 1e-9


def test_reserve_exhausts_budget():
    ledger = InMemoryFamilyBudgetLedger(global_cap=0.2)
    ledger.reserve(1, p_value=0.15, actor="agent")
    assert ledger.reserve(1, p_value=0.10, actor="agent") is None  # 0.25 > 0.20


def test_settle_releases_wealth():
    ledger = InMemoryFamilyBudgetLedger(global_cap=1.0)
    key = ledger.reserve(1, p_value=0.5, actor="agent")
    assert key is not None
    ledger.settle(1, reservation_key=key)
    assert abs(ledger.family_wealth(1) - 1.0) < 1e-9


def test_sum_alpha_never_exceeds_cap():
    ledger = InMemoryFamilyBudgetLedger(global_cap=1.0)
    for _ in range(5):
        assert ledger.reserve(1, p_value=0.2, actor="agent") is not None
    assert ledger.reserve(1, p_value=0.01, actor="agent") is None


def test_protocol_structural_conformance():
    # InMemoryFamilyBudgetLedger satisfies FamilyBudgetLedger Protocol at runtime
    ledger: FamilyBudgetLedger = InMemoryFamilyBudgetLedger()  # type: ignore[assignment]
    assert ledger.global_cap() == 1.0


def test_settle_unknown_key_is_noop():
    ledger = InMemoryFamilyBudgetLedger(global_cap=1.0)
    ledger.settle(1, reservation_key="nonexistent")  # must not raise
    assert abs(ledger.family_wealth(1) - 1.0) < 1e-9


def test_global_cap_accessible():
    ledger = InMemoryFamilyBudgetLedger(global_cap=0.5)
    assert ledger.global_cap() == 0.5
