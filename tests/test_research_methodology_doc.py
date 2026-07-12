"""Doc-presence regression guard for the #521 price-provenance correction.

These are substring assertions (lowercased, short stable fragments) so the guard survives
minor wording changes but fails loudly if the raw/PIT-adjusted/restated trichotomy — or the
actionable `adj_close` steer — is silently dropped from the methodology note or the skill.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_methodology_documents_pit_adjusted_trichotomy() -> None:
    text = (REPO / "kb" / "principles" / "research-methodology.md").read_text().lower()

    # Arm 3: raw is corporate-action CONTAMINATED, not leak-avoidance.
    assert "raw" in text, "methodology must discuss the RAW price arm"
    assert "contaminat" in text, "methodology must call raw close/volume corporate-action tainted"

    # Arm 1: PIT-correct adjusted.
    assert "pit" in text or "point-in-time" in text, "methodology must name the PIT arm"
    assert "adjust" in text, "methodology must discuss adjusted prices"

    # Arm 2: future / vendor-restated adjusted (the provenance leak).
    assert "restated" in text, "methodology must name the restated (leak) arm"

    # The actionable accessor.
    assert "adj_close" in text, "methodology must point at the adj_close accessor"


def test_author_skill_steers_to_adjusted() -> None:
    text = (REPO / ".codex" / "skills" / "author-a-strategy" / "SKILL.md").read_text().lower()

    assert "adj_close" in text, "author skill must steer signals to adj_close"
    assert "contaminat" in text or "raw close" in text, (
        "author skill must warn that raw close/volume is not leak-safe"
    )
