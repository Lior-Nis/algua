from pathlib import Path

import pytest

from algua.config.settings import Settings
from algua.knowledge.frontmatter import parse_doc
from algua.knowledge.sync import (
    family_doc_path,
    generate_indexes,
    kb_check,
    render_results_block,
    strategies_dir,
    strategy_doc_path,
    strategy_family,
    sync_all,
    sync_family_doc,
    sync_strategy_doc,
)
from algua.knowledge.templates import scaffold_family_doc, scaffold_strategy_doc


def _settings(tmp_path) -> Settings:
    return Settings(
        db_path=tmp_path / "r.db",
        knowledge_dir=tmp_path / "vault",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _scaffold(s: Settings, name: str, **kw) -> None:
    _write(strategy_doc_path(s, name), scaffold_strategy_doc(name, created="2026-06-03", **kw))


def _scaffold_family(s: Settings, name: str) -> None:
    _write(family_doc_path(s, name), scaffold_family_doc(name, created="2026-06-03"))


def test_render_results_block_handles_no_metrics():
    assert "No tracked runs" in render_results_block(None)


def test_render_results_block_renders_metrics():
    # Pins the holdout seal at the knowledge surface: even if a caller somehow passes a
    # metrics dict containing a holdout key, render_results_block renders whatever it receives.
    # The real seal is at latest_run_metrics (the read boundary); here we verify that a
    # normal metric IS rendered and that an already-filtered dict produces correct output.
    out = render_results_block(
        {"run_id": "abcd1234ef", "kind": "walk_forward",
         "snapshot_id": "ds_2", "seed": "7", "metrics": {"sharpe": 0.35}}
    )
    assert "walk_forward" in out
    assert "sharpe" in out
    assert "holdout.sharpe" not in out
    assert "ds_2" in out


def test_sync_strategy_doc_updates_stage_and_preserves_prose(tmp_path):
    s = _settings(tmp_path)
    _scaffold(s, "alpha")
    assert sync_strategy_doc(s, "alpha", stage="backtested") is True
    fm, body = parse_doc(strategy_doc_path(s, "alpha").read_text())
    assert fm["stage"] == "backtested"          # synced stage (passed in from the seam)
    assert "## Hypothesis" in body              # prose preserved
    assert "No tracked runs yet" in body        # synced RESULTS block present


def test_sync_strategy_doc_leaves_stage_when_unregistered(tmp_path):
    s = _settings(tmp_path)
    _scaffold(s, "alpha")
    assert sync_strategy_doc(s, "alpha", stage=None) is True
    fm, _ = parse_doc(strategy_doc_path(s, "alpha").read_text())
    assert fm["stage"] == "idea"                # scaffold default, untouched


def test_sync_strategy_doc_false_when_doc_missing(tmp_path):
    s = _settings(tmp_path)
    s.knowledge_dir.mkdir(parents=True)
    assert sync_strategy_doc(s, "ghost", stage=None) is False


def test_doc_paths_live_under_strategies_subdir(tmp_path):
    s = _settings(tmp_path)
    assert strategy_doc_path(s, "alpha") == s.knowledge_dir / "strategies" / "alpha.md"
    assert family_doc_path(s, "mom") == s.knowledge_dir / "strategies" / "families" / "mom.md"


def test_doc_paths_reject_traversal(tmp_path):
    s = _settings(tmp_path)
    with pytest.raises(ValueError):
        strategy_doc_path(s, "../escape")
    with pytest.raises(ValueError):
        sync_strategy_doc(s, "../escape", stage="idea")
    # family paths are contained to families/ itself, not just the vault root
    with pytest.raises(ValueError):
        family_doc_path(s, "../escape")


def test_strategy_family_unwraps_wikilink(tmp_path):
    s = _settings(tmp_path)
    _scaffold(s, "alpha", family="momentum")
    assert strategy_family(s, "alpha") == "momentum"
    assert strategy_family(s, "ghost") is None


def test_generate_indexes_lists_strategies_and_families(tmp_path):
    s = _settings(tmp_path)
    _scaffold(s, "alpha", family="momentum")
    _scaffold_family(s, "momentum")
    generate_indexes(s)
    index = (strategies_dir(s) / "_index.md").read_text()
    families = (strategies_dir(s) / "_families.md").read_text()
    assert "[[alpha]]" in index
    assert "[[momentum]]" in families


def test_sync_family_doc_counts_members_by_stage(tmp_path):
    s = _settings(tmp_path)
    _scaffold(s, "alpha", family="momentum")
    _scaffold_family(s, "momentum")
    sync_strategy_doc(s, "alpha", stage="backtested")
    assert sync_family_doc(s, "momentum") is True
    members = family_doc_path(s, "momentum").read_text()
    assert "backtested 1" in members


def test_kb_check_flags_missing_doc(tmp_path):
    s = _settings(tmp_path)
    s.knowledge_dir.mkdir(parents=True)
    ok, detail = kb_check(s, {"alpha": "idea"})
    assert ok is False
    assert "alpha" in detail


def test_kb_check_passes_when_in_sync(tmp_path):
    s = _settings(tmp_path)
    _scaffold(s, "alpha")
    sync_strategy_doc(s, "alpha", stage="idea")
    ok, _ = kb_check(s, {"alpha": "idea"})
    assert ok is True


def test_sync_all_returns_summary(tmp_path):
    s = _settings(tmp_path)
    _scaffold(s, "alpha", family="momentum")
    _scaffold_family(s, "momentum")
    summary = sync_all(s, {"alpha": "idea"})
    assert "alpha" in summary["strategies"]
    assert "momentum" in summary["families"]
    members = family_doc_path(s, "momentum").read_text()
    assert "1 members" in members or "members: idea 1" in members


def test_sync_all_applies_metadata(tmp_path):
    from algua.knowledge.frontmatter import parse_doc
    from algua.knowledge.templates import scaffold_strategy_doc

    s = _settings(tmp_path)
    p = strategy_doc_path(s, "a")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(scaffold_strategy_doc("a"))
    a_meta = {
        "family": "mean-reversion", "tags": ["slow"], "author": "agent",
        "hypothesis_status": "untested", "derived_from": None, "description": None,
    }
    sync_all(s, {"a": "idea"}, metadata={"a": a_meta})
    fm, _ = parse_doc(p.read_text())
    assert fm["family"] == "[[mean-reversion]]"
    assert fm["tags"] == ["slow"]


def test_sync_writes_owned_metadata_and_preserves_foreign_keys(tmp_path):
    from algua.knowledge.frontmatter import parse_doc, render_doc

    s = _settings(tmp_path)
    path = strategy_doc_path(s, "a")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Seed a doc with a foreign frontmatter key that must survive the sync.
    from algua.knowledge.templates import scaffold_strategy_doc
    fm, body = parse_doc(scaffold_strategy_doc("a"))
    fm["my_note"] = "keep me"
    path.write_text(render_doc(fm, body))

    meta = {
        "family": "mean-reversion", "tags": ["carry", "slow"], "author": "human",
        "hypothesis_status": "supported", "derived_from": "parent", "description": "d",
    }
    sync_strategy_doc(s, "a", stage="backtested", metadata=meta)

    fm2, _ = parse_doc(path.read_text())
    assert fm2["family"] == "[[mean-reversion]]"
    assert fm2["derived_from"] == "[[parent]]"
    assert fm2["tags"] == ["carry", "slow"]
    assert fm2["author"] == "human"
    assert fm2["hypothesis_status"] == "supported"
    assert fm2["stage"] == "backtested"
    assert fm2["my_note"] == "keep me"  # foreign key preserved
