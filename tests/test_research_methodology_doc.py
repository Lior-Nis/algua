"""Doc-presence regression guard for the #521 price-provenance correction.

These pin short, COUPLED phrases (not bare words that could appear anywhere) so the guard
survives incidental rewording but fails loudly if the raw/PIT-adjusted/restated trichotomy, the
actionable `adj_close` steer, or the honestly-documented residual gaps are silently watered down
in the methodology note or the author skill.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_methodology_documents_pit_adjusted_trichotomy() -> None:
    text = (REPO / "kb" / "principles" / "research-methodology.md").read_text().lower()

    # Arm 3: raw is corporate-action CONTAMINATED, not leak-avoidance — pin the coupled claim,
    # not just the two words appearing anywhere in the file.
    assert "raw close/volume is a correctness defect, not a leak-avoidance virtue" in text, (
        "methodology must plainly state raw close/volume is a contamination defect"
    )

    # Arm 1: PIT-correct adjusted is the correct arm.
    assert "pit-correct adjusted" in text, "methodology must name the PIT-correct-adjusted arm"
    assert "point-in-time-adjusted series" in text, (
        "methodology must tell authors to read the point-in-time-adjusted series"
    )

    # Arm 2: future / vendor-restated adjusted (the provenance leak).
    assert "vendor-restated" in text, "methodology must name the restated (leak) arm"

    # The actionable accessor, plus its two honestly-documented residual gaps (round-6 wording) —
    # coupled facts that would silently drift away if the doc were watered down.
    assert "adj_close" in text, "methodology must point at the adj_close accessor"
    assert "no `as_of` parameter yet" in text, (
        "methodology must document the as-of gap in get_bars, not overclaim a true PIT read"
    )
    assert "no adjusted-volume column" in text, (
        "methodology must document that no adjusted-volume column exists"
    )


def test_author_skill_steers_to_adjusted() -> None:
    text = (REPO / ".codex" / "skills" / "author-a-strategy" / "SKILL.md").read_text().lower()

    assert "derive returns/momentum from `adj_close`, never raw `close`".lower() in text, (
        "author skill must steer signals to adj_close and never raw close"
    )
    assert "corporate-action **contaminated**".lower() in text, (
        "author skill must warn that raw close/volume is not leak-safe"
    )
    assert ".sort_index()" in text, (
        "author skill must document the sort-before-positional-indexing hygiene rule"
    )


def test_author_skill_example_code_sorts_before_positional_indexing() -> None:
    """Regression: the copyable `signal()` example must itself follow the sort-index rule it
    teaches -- a template that violates its own adjacent rule is exactly what caused #521's
    secondary bug (missing sort_index() before positional .iloc)."""
    text = (REPO / ".codex" / "skills" / "author-a-strategy" / "SKILL.md").read_text()

    pivot_idx = text.index('values="adj_close")')
    next_iloc_idx = text.index(".iloc[-1]", pivot_idx)
    between = text[pivot_idx:next_iloc_idx]
    assert ".sort_index()" in between, (
        "the signal() example must call .sort_index() on the pivoted frame before any "
        "positional .iloc indexing"
    )
