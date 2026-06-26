import pytest

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


def test_parse_ignores_delimiter_substring_inside_value():
    # #252: a bare `---` inside a YAML value is NOT the closing delimiter. The old split('---', 2)
    # truncated the frontmatter here and spilled the rest into the body.
    text = "---\nname: x\ndescription: a---b\ntags:\n- kept\n---\nthe body\n"
    fm, body = parse_doc(text)
    assert fm == {"name": "x", "description": "a---b", "tags": ["kept"]}
    assert body == "the body\n"


def test_render_then_parse_round_trips_value_with_delimiter():
    # #252: the self-compounding case — a `--description "a---b"` must survive a render→sync→parse
    # cycle with tags kept, description intact, and no YAML spilled into the prose.
    fm = {"name": "x", "tags": ["kept"], "description": "a---b"}
    body = "claim\n"
    again_fm, again_body = parse_doc(render_doc(fm, body))
    assert again_fm == fm
    assert again_body == body


def test_round_trips_value_and_body_each_containing_a_delimiter_line():
    # An interior `---` LINE in a value, and a `---` horizontal-rule line in the body, both survive.
    fm = {"name": "x", "description": "before\n---\nafter", "tags": ["k"]}
    body = "intro\n\n---\n\nmore\n"
    again_fm, again_body = parse_doc(render_doc(fm, body))
    assert again_fm == fm
    assert again_body == body


def test_parse_no_closing_delimiter_is_not_frontmatter():
    # A leading `---` with no closing delimiter line (e.g. a doc opening with a rule) is body, not
    # a half-parsed frontmatter block.
    text = "---\njust a dangling opener and prose\nno closing line\n"
    fm, body = parse_doc(text)
    assert fm == {}
    assert body == text


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


def test_replace_block_raises_on_orphan_open_marker():
    # A start marker with no close would otherwise get a second block appended, and the
    # next sync would swallow the prose between the orphan and the appended close marker.
    text = "intro\n<!-- ALGUA:RESULTS -->\nstuff\nno close marker\n"
    with pytest.raises(ValueError):
        replace_block(text, "RESULTS", "new")


def test_replace_block_raises_on_orphan_close_marker():
    with pytest.raises(ValueError):
        replace_block("intro\n<!-- /ALGUA:RESULTS -->\nrest\n", "RESULTS", "new")


def test_replace_block_raises_on_duplicate_markers():
    text = (
        "<!-- ALGUA:RESULTS -->\na\n<!-- /ALGUA:RESULTS -->\n"
        "prose\n"
        "<!-- ALGUA:RESULTS -->\nb\n<!-- /ALGUA:RESULTS -->\n"
    )
    with pytest.raises(ValueError):
        replace_block(text, "RESULTS", "new")
