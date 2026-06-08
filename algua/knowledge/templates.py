from __future__ import annotations

from datetime import UTC, datetime

from algua.knowledge.frontmatter import render_doc


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def scaffold_strategy_doc(
    name: str,
    *,
    family: str | None = None,
    derived_from: str | None = None,
    created: str | None = None,
) -> str:
    """Initial strategy doc: authored sections empty, synced RESULTS block empty.

    Frontmatter keys family/tags/author/hypothesis_status/derived_from/description are
    registry-owned — edit via ``registry set``; they are overwritten on every sync.
    """
    fm: dict[str, object] = {
        "name": name,
        "stage": "idea",
        "hypothesis_status": "untested",
        "created": created or _today(),
    }
    if family:
        fm["family"] = f"[[{family}]]"
    if derived_from:
        fm["derived_from"] = f"[[{derived_from}]]"
    body = (
        "## Hypothesis\n\n_What edge is claimed, and why._\n\n"
        "## Derivation\n\n_Forked from what; what changed._\n\n"
        "## Results\n\n"
        "<!-- ALGUA:RESULTS -->\n_No tracked runs yet._\n<!-- /ALGUA:RESULTS -->\n\n"
        "## Verdict & next\n\n"
        "_What was learned; the next idea as a [[link]]._\n"
    )
    return render_doc(fm, body)


def scaffold_family_doc(name: str, *, created: str | None = None) -> str:
    """Initial thesis-family hub doc."""
    fm: dict[str, object] = {
        "type": "family",
        "name": name,
        "status": "exploring",
        "created": created or _today(),
    }
    body = (
        "## Thesis\n\n_The hypothesis for the whole family._\n\n"
        "## Members\n\n"
        "<!-- ALGUA:MEMBERS -->\n_No members yet._\n<!-- /ALGUA:MEMBERS -->\n\n"
        "## State of exploration\n\n_Members + one-line outcomes._\n\n"
        "## Open questions / next\n\n"
        "_Which axes are exhausted; what's left; when to park._\n"
    )
    return render_doc(fm, body)
