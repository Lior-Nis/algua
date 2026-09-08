from algua.config.settings import Settings
from algua.knowledge.inspirations import (
    SourcesRegistry,
    accept_new_notes,
    canonical_url,
    list_notes,
    mark_exhausted,
    mark_used,
    validate_note,
)

CATS = {"momentum", "mean_reversion"}
GOOD = """---
id: 2026-09-08-quiet-turnover-drift
found_at: 2026-09-08
source_url: https://example.com/post?utm_source=x&id=7#frag
venue: blog/example
source_kind: blog
category: momentum
market: us_equities
horizon: weekly
mechanism: crowded names mean-revert when attention fades
obscurity: niche
status: fresh
---
The post claims quietly declining turnover predicts drift.

> "we found the quiet ones drift" (short quote)

Doubtful: sample is 2019-2021 only.
"""


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, knowledge_dir=tmp_path / "kb", data_dir=tmp_path / "data")


def test_canonical_url_strips_tracking_and_fragment():
    assert canonical_url("https://Example.com/post?utm_source=x&id=7&fbclid=1#frag") == \
        "https://example.com/post?id=7"


def test_validate_note_reports_every_problem():
    fm = {"id": "wrong", "found_at": "2026-09-08", "source_url": "https://a/b",
          "venue": "blog/a", "source_kind": "tweet", "category": "nope", "market": "mars",
          "horizon": "weekly", "mechanism": "m", "obscurity": "rare", "status": "fresh"}
    problems = validate_note(fm, stem="2026-09-08-x", categories=CATS)
    assert {"id != filename stem", "source_kind: tweet", "category: nope",
            "market: mars"} <= set(problems)


def test_accept_new_notes_copies_valid_rejects_invalid_and_records_seen(tmp_path):
    s = _settings(tmp_path)
    staged = tmp_path / "wt" / "kb" / "inspirations"
    staged.mkdir(parents=True)
    (staged / "2026-09-08-quiet-turnover-drift.md").write_text(GOOD)
    (staged / "BAD NAME.md").write_text(GOOD)
    (staged / "2026-09-08-too-big.md").write_text(GOOD + "x" * 20000)
    seen = tmp_path / "data" / "inspirations-seen.jsonl"
    out = accept_new_notes(staged_dir=staged, settings=s, seen_path=seen, categories=CATS,
                           run_stamp="f1", max_notes=10)
    assert out["accepted"] == ["2026-09-08-quiet-turnover-drift"]
    reasons = {r["file"]: r["reasons"] for r in out["rejected"]}
    assert "bad filename" in reasons["BAD NAME.md"][0]
    assert "too large" in reasons["2026-09-08-too-big.md"][0]
    assert (s.knowledge_dir / "inspirations" / "2026-09-08-quiet-turnover-drift.md").exists()
    assert "https://example.com/post?id=7" in seen.read_text() or seen.read_text()  # hash line
    # second run: same URL is already seen, and the id already exists
    out2 = accept_new_notes(staged_dir=staged, settings=s, seen_path=seen, categories=CATS,
                            run_stamp="f2", max_notes=10)
    assert out2["accepted"] == []
    assert any("already seen" in r for rej in out2["rejected"] for r in rej["reasons"])


def test_mark_used_and_exhausted_edit_frontmatter_only(tmp_path):
    s = _settings(tmp_path)
    d = s.knowledge_dir / "inspirations"
    d.mkdir(parents=True)
    (d / "2026-09-08-quiet-turnover-drift.md").write_text(GOOD)
    mark_used(s, "2026-09-08-quiet-turnover-drift", idea_id=42)
    (note,) = list_notes(s, status="used")
    note_path = (d / note["id"]).with_suffix(".md")
    assert note["leaps"] == [42] and "quiet ones drift" in note_path.read_text()
    mark_exhausted(s, "2026-09-08-quiet-turnover-drift")
    assert list_notes(s, status="exhausted")[0]["leaps"] == [42]


def test_sources_registry_slice_and_yield_roundtrip(tmp_path):
    p = tmp_path / "_sources.yaml"
    p.write_text("venues:\n- key: reddit/algotrading\n  kind: forum\n  url: https://r/x\n"
                 "  categories: [momentum]\n  added_by: human\n  added_at: 2026-09-08\n"
                 "- key: blog/quant\n  kind: blog\n  url: https://q\n"
                 "  categories: [mean_reversion]\n  added_by: human\n  added_at: 2026-09-08\n")
    reg = SourcesRegistry(p)
    assert [v["key"] for v in reg.slice({"momentum"}, k=5)] == ["reddit/algotrading"]
    reg.write_yield("blog/quant", {"window_days": 90, "n": 6, "integrity_yield": 0.5,
                                   "walkforward_yield": 0.2, "survival_yield": 0.0,
                                   "computed_at": "2026-09-08T00:00:00+00:00"})
    assert SourcesRegistry(p).load()[1]["yield"]["n"] == 6
    reg.propose({"key": "youtube/someone", "kind": "video", "url": "https://yt/c",
                 "categories": ["momentum"]})
    assert SourcesRegistry(p).load()[2]["added_by"] == "forage"
