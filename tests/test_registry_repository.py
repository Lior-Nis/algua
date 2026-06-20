from algua.contracts.lifecycle import Stage
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.registry.repository import StrategyRecord, kb_metadata


def test_kb_metadata_projects_frontmatter_fields():
    rec = StrategyRecord(
        id=1,
        name="test_strategy",
        stage=Stage.IDEA,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        family="momentum",
        tags=["equity", "daily"],
        author=Author.HUMAN,
        hypothesis_status=HypothesisStatus.UNTESTED,
        derived_from=None,
        description="A test strategy",
    )
    meta = kb_metadata(rec)
    assert set(meta) == {"family", "tags", "author", "hypothesis_status",
                         "derived_from", "description"}
    assert "id" not in meta and "stage" not in meta
    assert meta["family"] == "momentum"
    assert meta["tags"] == ["equity", "daily"]
    assert meta["author"] == Author.HUMAN.value
    assert meta["hypothesis_status"] == HypothesisStatus.UNTESTED.value
    assert meta["derived_from"] is None
    assert meta["description"] == "A test strategy"
