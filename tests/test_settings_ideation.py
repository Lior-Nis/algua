# tests/test_settings_ideation.py
from algua.config.settings import Settings
from algua.contracts.idea import Horizon, Market
from algua.data.capabilities import supported_horizons, supported_markets


def test_ideation_settings_defaults():
    s = Settings(_env_file=None)
    assert (s.research_runs_per_day, s.research_hypotheses_per_run) == (12, 3)
    assert (s.idea_pool_floor_days, s.idea_pool_ceiling_days) == (2, 7)
    assert s.idea_claim_ttl_minutes == 180


def test_supported_markets_and_horizons_today():
    assert supported_markets() == frozenset({Market.US_EQUITIES, Market.ANY})
    assert supported_horizons() == frozenset(
        {Horizon.DAILY, Horizon.WEEKLY, Horizon.MONTHLY, Horizon.EVENT})
