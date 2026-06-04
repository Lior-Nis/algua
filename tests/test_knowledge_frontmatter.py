from algua.knowledge.frontmatter import parse_doc, render_doc, replace_block


def test_parse_round_trips_frontmatter_and_body():
    text = "---\nname: x\nstage: idea\n---\n## Hypothesis\n\nbody here\n"
    fm, body = parse_doc(text)
    assert fm == {"name": "x", "stage": "idea"}
    assert body == "## Hypothesis\n\nbody here\n"


def test_parse_no_frontmatter_returns_empty_dict():
    fm, body = parse_doc("just text\n")
    assert fm == {}
    assert body == "just text\n"


def test_render_then_parse_is_stable():
    fm = {"name": "x", "family": "[[fam]]", "stage": "backtested"}
    body = "## Hypothesis\n\nclaim\n"
    text = render_doc(fm, body)
    again_fm, again_body = parse_doc(text)
    assert again_fm == fm
    assert again_body == body


def test_replace_block_replaces_between_markers():
    text = "before\n<!-- ALGUA:RESULTS -->\nold\n<!-- /ALGUA:RESULTS -->\nafter\n"
    out = replace_block(text, "RESULTS", "new content")
    assert "new content" in out
    assert "old" not in out
    assert out.startswith("before\n")
    assert out.rstrip().endswith("after")


def test_replace_block_inserts_when_markers_absent():
    out = replace_block("body only\n", "RESULTS", "fresh")
    assert "<!-- ALGUA:RESULTS -->\nfresh\n<!-- /ALGUA:RESULTS -->" in out
