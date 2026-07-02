from algua.knowledge.frontmatter import parse_doc
from algua.knowledge.templates import scaffold_family_doc, scaffold_strategy_doc


def test_scaffold_strategy_doc_has_frontmatter_and_sections():
    text = scaffold_strategy_doc("alpha_v1", family="momentum", derived_from="alpha_v0",
                                 created="2026-06-03")
    fm, body = parse_doc(text)
    assert fm["name"] == "alpha_v1"
    assert fm["stage"] == "idea"
    assert fm["hypothesis_status"] == "untested"
    assert fm["family"] == "[[momentum]]"
    assert fm["derived_from"] == "[[alpha_v0]]"
    assert fm["created"] == "2026-06-03"
    assert "## Hypothesis" in body
    assert "## Derivation" in body
    assert "## Verdict & next" in body
    assert "<!-- ALGUA:RESULTS -->" in body and "<!-- /ALGUA:RESULTS -->" in body
    # Principles backlink footer: a revised principle can find the strategies authored under it.
    assert "[[research-methodology]]" in body and "[[risk-conventions]]" in body


def test_scaffold_strategy_doc_omits_absent_lineage():
    fm, _ = parse_doc(scaffold_strategy_doc("root", created="2026-06-03"))
    assert "family" not in fm
    assert "derived_from" not in fm


def test_scaffold_family_doc_has_thesis_and_members_block():
    fm, body = parse_doc(scaffold_family_doc("momentum", created="2026-06-03"))
    assert fm["type"] == "family"
    assert fm["name"] == "momentum"
    assert fm["status"] == "exploring"
    assert "## Thesis" in body
    assert "## Open questions / next" in body
    assert "<!-- ALGUA:MEMBERS -->" in body and "<!-- /ALGUA:MEMBERS -->" in body
