import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()

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

RARE = """---
id: 2026-09-07-rare-one
found_at: 2026-09-07
source_url: https://example.com/rare
venue: blog/example
source_kind: blog
category: momentum
market: us_equities
horizon: weekly
mechanism: something obscure
obscurity: rare
status: fresh
---
Body.
"""

COMMON = """---
id: 2026-09-06-common-one
found_at: 2026-09-06
source_url: https://example.com/common
venue: blog/example
source_kind: blog
category: momentum
market: us_equities
horizon: weekly
mechanism: something well known
obscurity: common
status: fresh
---
Body.
"""

BAD_NAME = "BAD NAME.md"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "kb"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))
    yield tmp_path


@pytest.fixture
def categories_file(tmp_path):
    p = tmp_path / "categories.txt"
    p.write_text("momentum\nmean_reversion  # some comment token counts too\n# a full comment\n")
    return p


def _run(*args):
    return runner.invoke(app, ["research", "inspirations", *args])


def _json(r):
    return json.loads(r.output)


def test_accept_copies_valid_and_rejects_invalid(tmp_path, categories_file):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "2026-09-08-quiet-turnover-drift.md").write_text(GOOD)
    (staged / BAD_NAME).write_text(GOOD)

    r = _run("accept", "--from", str(staged), "--run", "f1",
             "--categories-file", str(categories_file))
    assert r.exit_code == 0, r.output
    out = _json(r)
    assert out["ok"] is True
    assert out["accepted"] == ["2026-09-08-quiet-turnover-drift"]
    reasons = {row["file"]: row["reasons"] for row in out["rejected"]}
    assert "bad filename" in reasons[BAD_NAME][0]

    kb_dir = tmp_path / "kb" / "inspirations"
    assert (kb_dir / "2026-09-08-quiet-turnover-drift.md").exists()


def test_list_rare_first_orders_by_obscurity_then_newest(tmp_path, categories_file):
    staged = tmp_path / "staged"
    staged.mkdir()
    for text in (GOOD, RARE, COMMON):
        (staged / f"{text.splitlines()[1].split(': ')[1]}.md").write_text(text)
    r = _run("accept", "--from", str(staged), "--run", "f1",
             "--categories-file", str(categories_file))
    assert r.exit_code == 0, r.output
    assert len(_json(r)["accepted"]) == 3

    r = _run("list", "--rare-first")
    assert r.exit_code == 0, r.output
    ids = [n["id"] for n in _json(r)]
    assert ids == ["2026-09-07-rare-one", "2026-09-08-quiet-turnover-drift",
                   "2026-09-06-common-one"]


def test_mark_used_then_list_status_used(tmp_path, categories_file):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "2026-09-08-quiet-turnover-drift.md").write_text(GOOD)
    r = _run("accept", "--from", str(staged), "--run", "f1",
             "--categories-file", str(categories_file))
    assert r.exit_code == 0, r.output

    r = _run("mark-used", "2026-09-08-quiet-turnover-drift", "--idea", "42")
    assert r.exit_code == 0, r.output
    assert _json(r)["leaps"] == [42]

    r = _run("list", "--status", "used")
    assert r.exit_code == 0, r.output
    (note,) = _json(r)
    assert note["id"] == "2026-09-08-quiet-turnover-drift" and note["leaps"] == [42]


def test_mark_exhausted(tmp_path, categories_file):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "2026-09-08-quiet-turnover-drift.md").write_text(GOOD)
    r = _run("accept", "--from", str(staged), "--run", "f1",
             "--categories-file", str(categories_file))
    assert r.exit_code == 0, r.output

    r = _run("mark-exhausted", "2026-09-08-quiet-turnover-drift")
    assert r.exit_code == 0, r.output
    assert _json(r)["status"] == "exhausted"

    r = _run("list", "--status", "exhausted")
    assert r.exit_code == 0, r.output
    assert len(_json(r)) == 1


def test_propose_validates_and_writes_sources_registry(tmp_path, categories_file):
    r = _run("propose", "--key", "youtube/someone", "--kind", "video",
             "--url", "https://yt/c", "--categories", "momentum",
             "--categories-file", str(categories_file))
    assert r.exit_code == 0, r.output
    out = _json(r)
    assert out["key"] == "youtube/someone"

    sources = tmp_path / "kb" / "inspirations" / "_sources.yaml"
    assert sources.exists()
    text = sources.read_text()
    assert "youtube/someone" in text and "added_by: forage" in text


def test_propose_rejects_unknown_category(categories_file):
    r = _run("propose", "--key", "youtube/someone", "--kind", "video",
             "--url", "https://yt/c", "--categories", "not_a_category",
             "--categories-file", str(categories_file))
    assert r.exit_code == 1
    assert _json(r)["ok"] is False


def test_propose_rejects_bad_kind(categories_file):
    r = _run("propose", "--key", "youtube/someone", "--kind", "tweet",
             "--url", "https://yt/c", "--categories", "momentum",
             "--categories-file", str(categories_file))
    assert r.exit_code == 1
    assert _json(r)["ok"] is False


def test_propose_rejects_non_https_url(categories_file):
    r = _run("propose", "--key", "youtube/someone", "--kind", "video",
             "--url", "http://yt/c", "--categories", "momentum",
             "--categories-file", str(categories_file))
    assert r.exit_code == 1
    assert _json(r)["ok"] is False


def test_write_yield_from_scorecard_file(tmp_path):
    # Seed the sources registry directly (write-yield only updates an existing venue row).
    from algua.knowledge.inspirations import SourcesRegistry
    sources = tmp_path / "kb" / "inspirations" / "_sources.yaml"
    sources.parent.mkdir(parents=True)
    SourcesRegistry(sources).propose(
        {"key": "blog/quant", "kind": "blog", "url": "https://q", "categories": ["momentum"]})

    scorecard = tmp_path / "scorecard.json"
    scorecard.write_text(json.dumps({
        "ok": True, "days": 90, "min_n_for_rates": 5,
        "by_venue": {
            "blog/quant": {"n": 6, "integrity_yield": 0.5, "walkforward_yield": 0.2,
                           "survival_yield": 0.0},
            "too/thin": {"n": 2, "integrity_yield": None, "walkforward_yield": None,
                        "survival_yield": None},
        },
    }))

    r = _run("write-yield", "--from-scorecard", str(scorecard))
    assert r.exit_code == 0, r.output
    out = _json(r)
    assert out["updated"] == ["blog/quant"]

    venues = SourcesRegistry(sources).load()
    (quant,) = [v for v in venues if v["key"] == "blog/quant"]
    assert quant["yield"]["n"] == 6 and quant["yield"]["window_days"] == 90
