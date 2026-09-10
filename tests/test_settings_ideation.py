# tests/test_settings_ideation.py
from algua.config.settings import Settings
from algua.contracts.idea import Horizon, Market
from algua.data.capabilities import supported_horizons, supported_markets
from algua.registry.idea_attempts import DEFAULT_PREVIEW_HOLD_HOURS


def test_ideation_settings_defaults():
    s = Settings(_env_file=None)
    assert (s.research_runs_per_day, s.research_hypotheses_per_run) == (12, 3)
    assert (s.idea_pool_floor_days, s.idea_pool_ceiling_days) == (2, 7)
    assert s.idea_claim_ttl_minutes == 180
    assert s.idea_preview_hold_hours == 72


def test_preview_hold_default_matches_the_repositorys_own_fallback():
    """`Settings` is what the CLI passes; `DEFAULT_PREVIEW_HOLD_HOURS` is what a direct
    repository caller gets. Two constants, one meaning — this is the drift guard."""
    assert Settings(_env_file=None).idea_preview_hold_hours == DEFAULT_PREVIEW_HOLD_HOURS


def test_the_preview_hold_comfortably_exceeds_the_ordinary_claim_ttl():
    """The hold exists precisely because a backlogged merge-back outlives the claim TTL; a hold
    at or below the TTL would silently restore the stranding bug."""
    s = Settings(_env_file=None)
    assert s.idea_preview_hold_hours * 60 > s.idea_claim_ttl_minutes


def test_supported_markets_and_horizons_today():
    assert supported_markets() == frozenset({Market.US_EQUITIES, Market.ANY})
    assert supported_horizons() == frozenset(
        {Horizon.DAILY, Horizon.WEEKLY, Horizon.MONTHLY, Horizon.EVENT})
