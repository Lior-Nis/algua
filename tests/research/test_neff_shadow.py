from algua.research.gates import (
    MIN_CORR_OVERLAP_BARS,
    MIN_N_EFF_SIBLINGS,
    RHO_BAR_SHRINKAGE_K,
    GateDecision,
)


def test_constants():
    assert MIN_N_EFF_SIBLINGS == 5
    assert MIN_CORR_OVERLAP_BARS == 21
    assert RHO_BAR_SHRINKAGE_K == 1.0


def test_gatedecision_neff_fields_default_none_and_serialize():
    d = GateDecision(passed=True, checks=[])
    assert d.dsr_n_eff is None and d.dsr_rho_bar is None and d.dsr_n_siblings is None
    dd = d.to_dict()
    assert dd["dsr_n_eff"] is None and dd["dsr_rho_bar"] is None and dd["dsr_n_siblings"] is None
    d2 = GateDecision(passed=True, checks=[], dsr_n_eff=12, dsr_rho_bar=0.4, dsr_n_siblings=7)
    dd2 = d2.to_dict()
    assert dd2["dsr_n_eff"] == 12 and dd2["dsr_rho_bar"] == 0.4 and dd2["dsr_n_siblings"] == 7
