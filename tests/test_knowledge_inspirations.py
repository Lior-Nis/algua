from algua.config.settings import Settings
from algua.knowledge.inspirations import (
    SourcesRegistry,
    accept_new_notes,
    canonical_url,
    list_notes,
    load_note,
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


def test_accept_new_notes_rejects_dangling_symlink_without_aborting_batch(tmp_path):
    s = _settings(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "2026-09-08-quiet-turnover-drift.md").write_text(GOOD)
    dangling = staged / "2026-09-08-dangling-note.md"
    dangling.symlink_to(staged / "does-not-exist.md")
    seen = tmp_path / "data" / "inspirations-seen.jsonl"

    out = accept_new_notes(staged_dir=staged, settings=s, seen_path=seen, categories=CATS,
                           run_stamp="f1", max_notes=10)

    assert out["accepted"] == ["2026-09-08-quiet-turnover-drift"]
    reasons = {r["file"]: r["reasons"] for r in out["rejected"]}
    assert reasons["2026-09-08-dangling-note.md"] == ["not a regular file"]


def test_accept_new_notes_copies_staged_bytes_exactly(tmp_path):
    s = _settings(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    staged_file = staged / "2026-09-08-quiet-turnover-drift.md"
    staged_file.write_text(GOOD)
    seen = tmp_path / "data" / "inspirations-seen.jsonl"

    accept_new_notes(staged_dir=staged, settings=s, seen_path=seen, categories=CATS,
                     run_stamp="f1", max_notes=10)

    vault_file = s.knowledge_dir / "inspirations" / "2026-09-08-quiet-turnover-drift.md"
    assert vault_file.read_bytes() == staged_file.read_bytes()


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


def test_sources_registry_yield_and_propose_roundtrip(tmp_path):
    s = _settings(tmp_path)
    path = s.knowledge_dir / "inspirations" / "_sources.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("venues:\n- key: reddit/algotrading\n  kind: forum\n  url: https://r/x\n"
                    "  categories: [momentum]\n  added_by: human\n  added_at: 2026-09-08\n"
                    "- key: blog/quant\n  kind: blog\n  url: https://q\n"
                    "  categories: [mean_reversion]\n  added_by: human\n  added_at: 2026-09-08\n")
    reg = SourcesRegistry(s)
    reg.write_yields({"blog/quant": {"window_days": 90, "n": 6, "integrity_yield": 0.5,
                                     "walkforward_yield": 0.2, "survival_yield": 0.0,
                                     "computed_at": "2026-09-08T00:00:00+00:00"}})
    assert SourcesRegistry(s).load()[1]["yield"]["n"] == 6
    reg.propose({"key": "youtube/someone", "kind": "video", "url": "https://yt/c",
                 "categories": ["momentum"]})
    assert SourcesRegistry(s).load()[2]["added_by"] == "forage"


# --- Final fix wave (whole-branch review 2026-09-09) ---------------------------------------------


BAD_YAML = "---\nid: [unclosed\n  broken: :\n---\nbody\n"
LIST_FRONTMATTER = "---\n- a\n- b\n---\nbody\n"


def test_accept_new_notes_rejects_unparseable_frontmatter_without_aborting_the_batch(tmp_path):
    # A single malformed staged note used to raise out of accept_new_notes (yaml error, or an
    # AttributeError on a non-dict frontmatter) and abort the WHOLE batch — every good note the
    # forage agent wrote alongside it was lost. Each note now fails on its own.
    s = _settings(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "2026-09-08-quiet-turnover-drift.md").write_text(GOOD)
    (staged / "2026-09-08-broken-yaml-note.md").write_text(BAD_YAML)
    (staged / "2026-09-08-list-frontmatter.md").write_text(LIST_FRONTMATTER)
    seen = tmp_path / "data" / "inspirations-seen.jsonl"

    out = accept_new_notes(staged_dir=staged, settings=s, seen_path=seen, categories=CATS,
                           run_stamp="f1", max_notes=10)

    assert out["accepted"] == ["2026-09-08-quiet-turnover-drift"]
    reasons = {r["file"]: r["reasons"] for r in out["rejected"]}
    assert any("unparseable frontmatter" in r
               for r in reasons["2026-09-08-broken-yaml-note.md"])
    assert any("unparseable frontmatter" in r
               for r in reasons["2026-09-08-list-frontmatter.md"])


def test_list_notes_exclude_status(tmp_path):
    s = _settings(tmp_path)
    d = s.knowledge_dir / "inspirations"
    d.mkdir(parents=True)
    (d / "2026-09-08-quiet-turnover-drift.md").write_text(GOOD)
    (d / "2026-09-07-spent-note.md").write_text(
        GOOD.replace("2026-09-08-quiet-turnover-drift", "2026-09-07-spent-note")
            .replace("status: fresh", "status: exhausted"))

    ids = [n["id"] for n in list_notes(s, exclude_status="exhausted")]
    assert ids == ["2026-09-08-quiet-turnover-drift"]
    assert len(list_notes(s)) == 2
    assert list_notes(s, exclude_status="exhausted", limit=1)[0]["id"] == \
        "2026-09-08-quiet-turnover-drift"


def test_load_note_returns_frontmatter_or_none(tmp_path):
    s = _settings(tmp_path)
    d = s.knowledge_dir / "inspirations"
    d.mkdir(parents=True)
    (d / "2026-09-08-quiet-turnover-drift.md").write_text(GOOD)

    fm = load_note(s, "2026-09-08-quiet-turnover-drift")
    assert fm is not None and fm["venue"] == "blog/example" and fm["obscurity"] == "niche"
    assert load_note(s, "2026-09-08-does-not-exist") is None
    assert load_note(s, "../escape") is None           # bad id: never raises, never escapes
    (d / "2026-09-08-broken-yaml-note.md").write_text(BAD_YAML)
    assert load_note(s, "2026-09-08-broken-yaml-note") is None


def test_sources_registry_writes_are_atomic_and_locked(tmp_path):
    s = _settings(tmp_path)
    reg = SourcesRegistry(s)
    reg.propose({"key": "youtube/someone", "kind": "video", "url": "https://yt/c",
                 "categories": ["momentum"]})
    assert reg.path == s.knowledge_dir / "inspirations" / "_sources.yaml"
    assert SourcesRegistry(s).load()[0]["added_by"] == "forage"
    # No temp file is left behind by the atomic write.
    assert [p.name for p in reg.path.parent.iterdir()] == ["_sources.yaml"]


def test_sources_registry_write_yields_does_one_load_and_one_save(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    reg = SourcesRegistry(s)
    for key in ("a/one", "b/two"):
        reg.propose({"key": key, "kind": "blog", "url": "https://x", "categories": ["momentum"]})

    calls = {"load": 0, "save": 0}
    real_load, real_save = SourcesRegistry.load, SourcesRegistry._save
    monkeypatch.setattr(SourcesRegistry, "load",
                        lambda self: (calls.__setitem__("load", calls["load"] + 1),
                                      real_load(self))[1])
    monkeypatch.setattr(SourcesRegistry, "_save",
                        lambda self, v: (calls.__setitem__("save", calls["save"] + 1),
                                         real_save(self, v))[1])
    reg.write_yields({"a/one": {"n": 6}, "b/two": {"n": 7}})
    assert calls == {"load": 1, "save": 1}
    loaded = {v["key"]: v.get("yield") for v in SourcesRegistry(s).load()}
    assert loaded == {"a/one": {"n": 6}, "b/two": {"n": 7}}
