from contextlib import closing
from pathlib import Path

from algua.config.settings import Settings
from algua.knowledge.frontmatter import parse_doc
from algua.knowledge.sync import (
    generate_indexes,
    kb_check,
    render_results_block,
    sync_all,
    sync_strategy_doc,
)
from algua.knowledge.templates import scaffold_family_doc, scaffold_strategy_doc
from algua.registry import store
from algua.registry.db import connect, migrate


def _settings(tmp_path) -> Settings:
    return Settings(
        db_path=tmp_path / "r.db",
        knowledge_dir=tmp_path / "vault",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_render_results_block_handles_no_metrics():
    assert "No tracked runs" in render_results_block(None)


def test_render_results_block_renders_metrics():
    out = render_results_block(
        {"run_id": "abcd1234ef", "kind": "walk_forward",
         "snapshot_id": "ds_2", "seed": "7", "metrics": {"holdout.sharpe": 0.28}}
    )
    assert "walk_forward" in out
    assert "holdout.sharpe" in out
    assert "ds_2" in out


def test_sync_strategy_doc_updates_stage_and_preserves_prose(tmp_path):
    s = _settings(tmp_path)
    _write(s.knowledge_dir / "alpha.md",
           scaffold_strategy_doc("alpha", created="2026-06-03"))
    with closing(connect(s.db_path)) as conn:
        migrate(conn)
        store.add_strategy(conn, "alpha")
        store.transition(conn, "alpha", "backtested", "agent", "test")
        assert sync_strategy_doc(s, conn, "alpha") is True
    fm, body = parse_doc((s.knowledge_dir / "alpha.md").read_text())
    assert fm["stage"] == "backtested"          # synced from registry
    assert "## Hypothesis" in body              # prose preserved
    assert "No tracked runs yet" in body        # synced RESULTS block present


def test_sync_strategy_doc_false_when_doc_missing(tmp_path):
    s = _settings(tmp_path)
    s.knowledge_dir.mkdir(parents=True)
    with closing(connect(s.db_path)) as conn:
        migrate(conn)
        assert sync_strategy_doc(s, conn, "ghost") is False


def test_generate_indexes_lists_strategies_and_families(tmp_path):
    s = _settings(tmp_path)
    _write(s.knowledge_dir / "alpha.md",
           scaffold_strategy_doc("alpha", family="momentum", created="2026-06-03"))
    _write(s.knowledge_dir / "families" / "momentum.md",
           scaffold_family_doc("momentum", created="2026-06-03"))
    generate_indexes(s)
    index = (s.knowledge_dir / "_index.md").read_text()
    families = (s.knowledge_dir / "_families.md").read_text()
    assert "[[alpha]]" in index
    assert "[[momentum]]" in families


def test_kb_check_flags_missing_doc(tmp_path):
    s = _settings(tmp_path)
    s.knowledge_dir.mkdir(parents=True)
    with closing(connect(s.db_path)) as conn:
        migrate(conn)
        store.add_strategy(conn, "alpha")
    ok, detail = kb_check(s)
    assert ok is False
    assert "alpha" in detail


def test_sync_all_returns_summary(tmp_path):
    s = _settings(tmp_path)
    _write(s.knowledge_dir / "alpha.md",
           scaffold_strategy_doc("alpha", family="momentum", created="2026-06-03"))
    _write(s.knowledge_dir / "families" / "momentum.md",
           scaffold_family_doc("momentum", created="2026-06-03"))
    with closing(connect(s.db_path)) as conn:
        migrate(conn)
        store.add_strategy(conn, "alpha")
        summary = sync_all(s, conn)
    assert "alpha" in summary["strategies"]
    assert "momentum" in summary["families"]
    members = (s.knowledge_dir / "families" / "momentum.md").read_text()
    assert "1 members" in members or "members: idea 1" in members
