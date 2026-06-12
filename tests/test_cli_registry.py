import json
import sqlite3

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_add_and_list():
    added = _json(runner.invoke(app, ["registry", "add", "alpha"]))
    assert added["ok"] is True  # success envelope discriminator
    listed = _json(runner.invoke(app, ["registry", "list"]))
    # `registry list` is the documented exception: a bare JSON array (collection), not an envelope.
    assert isinstance(listed, list)
    assert [s["name"] for s in listed] == ["alpha"]
    assert listed[0]["stage"] == "idea"


def test_transition_legal():
    runner.invoke(app, ["registry", "add", "alpha"])
    out = _json(runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "backtested",
              "--actor", "agent", "--reason", "ran"]))
    assert out["stage"] == "backtested"


def test_transition_illegal_exits_nonzero():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "live", "--actor", "agent"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def _advance_to_paper(strategy: str) -> None:
    runner.invoke(app, ["registry", "add", strategy])
    # CANDIDATE via human: scaffolding to paper, not exercising the agent shortlist gate.
    for stage, actor in (("backtested", "agent"), ("candidate", "human"), ("paper", "agent")):
        runner.invoke(app, ["registry", "transition", strategy,
                            "--to", stage, "--actor", actor])


def _advance_to_forward_tested(strategy: str) -> None:
    _advance_to_paper(strategy)
    runner.invoke(app, ["registry", "transition", strategy,
                        "--to", "forward_tested", "--actor", "human"])


def _make_key(tmp_path, name="id"):
    import subprocess
    key = tmp_path / name
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)],
                   check=True, capture_output=True)
    return key, (tmp_path / f"{name}.pub").read_text().strip()


def _allowed_signers_file(tmp_path, principal, pub):
    keytype, keyblob = pub.split()[0], pub.split()[1]
    path = tmp_path / "allowed_signers"
    path.write_text(f'{principal} namespaces="algua-go-live" {keytype} {keyblob}\n')
    return path


def _sign_file(key, content_path):
    import subprocess
    subprocess.run(["ssh-keygen", "-Y", "sign", "-n", "algua-go-live", "-f", str(key),
                    str(content_path)],
                   check=True, capture_output=True)
    return content_path.parent / (content_path.name + ".sig")


def test_full_path_to_live_signed_ceremony(tmp_path, monkeypatch):
    """Two-step signed go-live: challenge then signature."""
    strategy = "cross_sectional_momentum"
    _advance_to_forward_tested(strategy)

    key, pub = _make_key(tmp_path)
    signers = _allowed_signers_file(tmp_path, "lior", pub)
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)

    # Allocate capital so the go-live guard is satisfied (guard: requires allocation)
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 100_000.0)
    assert runner.invoke(app, ["live", "allocate", strategy, "--capital", "10000"]).exit_code == 0

    # Step 1: no --signature -> challenge issued, stage stays forward_tested
    result1 = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                  "--actor", "human"])
    assert result1.exit_code == 0, result1.stdout
    out1 = json.loads(result1.stdout)
    assert out1["ok"] is True
    assert out1["action"] == "go_live_challenge"
    assert "challenge" in out1

    show_after_challenge = json.loads(
        runner.invoke(app, ["registry", "show", strategy]).stdout)
    assert show_after_challenge["stage"] == "forward_tested", \
        "stage must still be forward_tested after step-1"

    # Sign the challenge value
    challenge_file = tmp_path / "challenge.txt"
    challenge_file.write_text(out1["challenge"])
    sig_file = _sign_file(key, challenge_file)

    # Step 2: transition --to live --signature <file>.sig -> accepted, stage becomes live
    result2 = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                  "--actor", "human", "--signature", str(sig_file)])
    assert result2.exit_code == 0, result2.stdout
    out2 = json.loads(result2.stdout)
    assert out2["ok"] is True
    assert out2["stage"] == "live"

    # Replay: same .sig again must NOT succeed (challenge consumed / strategy already live)
    result3 = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                  "--actor", "human", "--signature", str(sig_file)])
    assert result3.exit_code != 0 or json.loads(result3.stdout).get("ok") is False, \
        "replayed signature must not produce a fresh go-live"


def test_go_live_without_signature_stays_paper(tmp_path, monkeypatch):
    """Step-1 alone (no --signature) returns challenge and leaves stage forward_tested."""
    strategy = "cross_sectional_momentum"
    _advance_to_forward_tested(strategy)

    key, pub = _make_key(tmp_path)
    signers = _allowed_signers_file(tmp_path, "lior", pub)
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)

    # Allocate capital so the go-live guard is satisfied (guard: requires allocation)
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 100_000.0)
    assert runner.invoke(app, ["live", "allocate", strategy, "--capital", "10000"]).exit_code == 0

    result = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                 "--actor", "human"])
    assert result.exit_code == 0, result.stdout
    out = json.loads(result.stdout)
    assert out["action"] == "go_live_challenge"
    assert "challenge" in out

    show = json.loads(runner.invoke(app, ["registry", "show", strategy]).stdout)
    assert show["stage"] == "forward_tested"


def test_go_live_with_signature_but_no_pending_challenge(tmp_path, monkeypatch):
    """Presenting a signature without a prior challenge must fail."""

    strategy = "cross_sectional_momentum"
    _advance_to_forward_tested(strategy)

    key, pub = _make_key(tmp_path)
    signers = _allowed_signers_file(tmp_path, "lior", pub)
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)

    # Sign an arbitrary payload (no challenge was issued)
    arb = tmp_path / "arbitrary.txt"
    arb.write_text("not a real challenge")
    sig_file = _sign_file(key, arb)

    result = runner.invoke(app, ["registry", "transition", strategy, "--to", "live",
                                 "--actor", "human", "--signature", str(sig_file)])
    assert result.exit_code != 0 or json.loads(result.stdout).get("ok") is False
    show = json.loads(runner.invoke(app, ["registry", "show", strategy]).stdout)
    assert show["stage"] == "forward_tested"


def test_unknown_strategy_emits_json_error():
    result = runner.invoke(app, ["registry", "show", "ghost"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "ghost" in payload["error"]


def test_invalid_stage_value_emits_json_error():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "bogus", "--actor", "agent"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_duplicate_add_emits_json_error():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(app, ["registry", "add", "alpha"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_registry_command_closes_connection(monkeypatch, tmp_path):
    # The connect+migrate+close lifecycle now lives in cli._common.registry_conn (the single
    # shared idiom), so the connection factory is patched there.
    from algua.cli import _common

    closed = []

    class TrackingConnection(sqlite3.Connection):
        def close(self) -> None:
            closed.append(True)
            super().close()

    def connect_tracking(_db_path):
        conn = sqlite3.connect(tmp_path / "tracked.db", factory=TrackingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(_common, "connect", connect_tracking)

    result = runner.invoke(app, ["registry", "add", "alpha"])

    assert result.exit_code == 0, result.stdout
    assert closed == [True]


def test_enroll_approver_appends_line(tmp_path, monkeypatch):
    import json as _json

    signers = tmp_path / "allowed_signers"
    signers.write_text("# header\n")
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)
    r = runner.invoke(app, ["registry", "enroll-approver", "--name", "lior",
                            "--pubkey", "ssh-ed25519 AAAAC3NzaC1lZDI1 lior@dev"])
    assert r.exit_code == 0, r.stdout
    assert _json.loads(r.stdout)["ok"] is True
    body = signers.read_text()
    assert 'lior namespaces="algua-go-live" ssh-ed25519 AAAAC3NzaC1lZDI1' in body
    r2 = runner.invoke(app, ["registry", "enroll-approver", "--name", "x",
                             "--pubkey", "ssh-ed25519 AAAAC3NzaC1lZDI1 other"])
    assert r2.exit_code == 1  # duplicate pubkey rejected


def test_enroll_approver_rejects_bad_principal(tmp_path, monkeypatch):
    import json as _json
    signers = tmp_path / "allowed_signers"
    signers.write_text("# header\n")
    monkeypatch.setattr("algua.cli.registry_cmd.ALLOWED_SIGNERS_PATH", signers)
    # a --name with whitespace/newline must be rejected (no allowed_signers line injection)
    r = runner.invoke(app, ["registry", "enroll-approver", "--name", "bad name",
                            "--pubkey", "ssh-ed25519 AAAAaaa x"])
    assert r.exit_code == 1 and _json.loads(r.stdout)["ok"] is False
    r2 = runner.invoke(app, ["registry", "enroll-approver", "--name", "evil\nattacker",
                             "--pubkey", "ssh-ed25519 BBBBbbb x"])
    assert r2.exit_code == 1
    assert "namespaces" not in signers.read_text().replace("# header", "")  # nothing enrolled


# ---------------------------------------------------------------------------
# Go-live guard helpers (Task 5: requires allocation + ≤1 live strategy)
# ---------------------------------------------------------------------------

def _seed_forward_tested(monkeypatch, tmp_path, name: str) -> str:
    """Register a strategy and advance it to the forward_tested stage; return its name."""
    from algua.registry.repository import ArtifactIdentity

    # paper -> forward_tested now recomputes identity for audit pinning (#124); fake strategies
    # here have no module on disk, so stub the recompute for the seeding transition.
    monkeypatch.setattr(
        "algua.registry.approvals.compute_artifact_hashes",
        lambda n: ArtifactIdentity(code_hash="aabb", config_hash="ccdd", dependency_hash="eeff"),
    )
    _advance_to_forward_tested(name)
    return name


def _force_live(monkeypatch, tmp_path, name: str) -> None:
    """Register a strategy and force-set its stage to 'live' directly in the DB."""
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    runner.invoke(app, ["registry", "add", name])
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name=?", (name,))
        conn.commit()


def _allocate(monkeypatch, tmp_path, name: str, capital: float) -> None:
    """Set a live allocation for the named strategy via the CLI (monkeypatching equity read)."""
    monkeypatch.setattr("algua.cli.live_cmd._live_account_equity", lambda: 1_000_000.0)
    r = runner.invoke(app, ["live", "allocate", name, "--capital", str(capital)])
    assert r.exit_code == 0, f"_allocate failed: {r.stdout}"


def test_go_live_requires_allocation(monkeypatch, tmp_path):
    # a strategy at forward_tested with NO allocation cannot be issued a go-live challenge
    name = _seed_forward_tested(monkeypatch, tmp_path, "s1")
    r = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    assert r.exit_code == 1 and "allocation" in r.stdout.lower()


def test_list_emits_metadata_fields():
    runner.invoke(app, ["registry", "add", "alpha"])
    out = _json(runner.invoke(app, ["registry", "list"]))
    assert out[0]["author"] == "agent"
    assert out[0]["hypothesis_status"] == "untested"
    assert out[0]["tags"] == []
    assert out[0]["family"] is None


def test_show_emits_metadata_fields():
    runner.invoke(app, ["registry", "add", "alpha"])
    out = _json(runner.invoke(app, ["registry", "show", "alpha"]))
    assert out["author"] == "agent"
    assert out["hypothesis_status"] == "untested"
    assert out["tags"] == []


def test_add_with_metadata_flags():
    out = _json(runner.invoke(app, [
        "registry", "add", "mr1", "--family", "mean-reversion",
        "--tag", "slow", "--tag", "carry", "--author", "human",
        "--hypothesis-status", "supported", "--description", "desc",
    ]))
    assert out["ok"] is True
    rec = _json(runner.invoke(app, ["registry", "show", "mr1"]))
    assert rec["family"] == "mean-reversion"
    assert rec["tags"] == ["carry", "slow"]
    assert rec["author"] == "human"
    assert rec["hypothesis_status"] == "supported"


def test_invalid_family_slug_emits_json_error():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(app, ["registry", "add", "beta", "--family", "Bad_Family"])
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False

    result2 = runner.invoke(app, ["registry", "set", "alpha", "--family", "-bad"])
    assert result2.exit_code != 0
    payload2 = json.loads(result2.stdout)
    assert payload2["ok"] is False


def test_set_changes_metadata_and_reports_before_after(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    runner.invoke(app, ["registry", "add", "a", "--hypothesis-status", "untested"])
    out = _json(runner.invoke(app, [
        "registry", "set", "a", "--hypothesis-status", "supported", "--add-tag", "carry",
    ]))
    assert out["changed"]["hypothesis_status"] == {"before": "untested", "after": "supported"}
    rec = _json(runner.invoke(app, ["registry", "show", "a"]))
    assert rec["hypothesis_status"] == "supported"
    assert rec["tags"] == ["carry"]


def test_list_filters_compose():
    runner.invoke(app, ["registry", "add", "a", "--family", "mean-reversion", "--tag", "slow"])
    runner.invoke(app, ["registry", "add", "b", "--family", "momentum", "--tag", "fast"])
    out = _json(runner.invoke(app, ["registry", "list", "--family", "mean-reversion"]))
    assert [r["name"] for r in out] == ["a"]
    out2 = _json(runner.invoke(app, ["registry", "list", "--tag", "fast"]))
    assert [r["name"] for r in out2] == ["b"]


# ---------------------------------------------------------------------------
# Task 13: registry backfill-from-kb
# ---------------------------------------------------------------------------

def test_backfill_from_kb_reports_and_fills(monkeypatch, tmp_path):
    """A legacy NULL-column registry row is filled from a kb doc's frontmatter."""
    from algua.config.settings import get_settings
    from algua.knowledge.frontmatter import render_doc
    from algua.registry.db import connect, migrate

    # Isolate the kb vault.
    vault = tmp_path / "vault"
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(vault))

    # Insert a legacy NULL-metadata row directly (bypassing add() which writes defaults).
    db_path = get_settings().db_path
    conn = connect(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('alpha','idea','2026-01-01','2026-01-01')"
    )
    conn.commit()
    conn.close()

    # Write a kb doc for "alpha" with family and hypothesis_status in frontmatter.
    strat_dir = vault / "strategies"
    strat_dir.mkdir(parents=True)
    fm = {
        "name": "alpha",
        "stage": "idea",
        "family": "[[mean-reversion]]",
        "hypothesis_status": "supported",
        "tags": ["slow", "carry"],
    }
    (strat_dir / "alpha.md").write_text(render_doc(fm, "## Hypothesis\n"))

    out = _json(runner.invoke(app, ["registry", "backfill-from-kb"]))
    assert "alpha" in out["processed"]
    assert "unmappable" in out
    assert "kb_docs_without_registry_row" in out
    assert "registry_rows_without_kb_doc" in out

    rec = _json(runner.invoke(app, ["registry", "show", "alpha"]))
    assert rec["family"] == "mean-reversion"
    assert rec["hypothesis_status"] == "supported"
    assert rec["tags"] == ["carry", "slow"]


def test_backfill_from_kb_does_not_overwrite_existing_values(monkeypatch, tmp_path):
    """A second backfill run is a no-op once columns are non-NULL."""
    from algua.knowledge.frontmatter import render_doc

    vault = tmp_path / "vault"
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(vault))

    # Register with a concrete family value.
    runner.invoke(app, ["registry", "add", "alpha", "--family", "momentum"])

    # Write a kb doc claiming a different family.
    strat_dir = vault / "strategies"
    strat_dir.mkdir(parents=True)
    fm = {
        "name": "alpha",
        "stage": "idea",
        "family": "[[mean-reversion]]",
        "hypothesis_status": "supported",
    }
    (strat_dir / "alpha.md").write_text(render_doc(fm, "## Hypothesis\n"))

    _json(runner.invoke(app, ["registry", "backfill-from-kb"]))
    rec = _json(runner.invoke(app, ["registry", "show", "alpha"]))
    # family was already 'momentum'; backfill must not overwrite it
    assert rec["family"] == "momentum"


def test_backfill_from_kb_reports_unmappable_status(monkeypatch, tmp_path):
    """A kb doc with an unknown hypothesis_status value is reported as unmappable."""
    from algua.config.settings import get_settings
    from algua.knowledge.frontmatter import render_doc
    from algua.registry.db import connect, migrate

    vault = tmp_path / "vault"
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(vault))

    db_path = get_settings().db_path
    conn = connect(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('beta','idea','2026-01-01','2026-01-01')"
    )
    conn.commit()
    conn.close()

    strat_dir = vault / "strategies"
    strat_dir.mkdir(parents=True)
    fm = {"name": "beta", "stage": "idea", "hypothesis_status": "unknown_value"}
    (strat_dir / "beta.md").write_text(render_doc(fm, "## Hypothesis\n"))

    out = _json(runner.invoke(app, ["registry", "backfill-from-kb"]))
    assert any(u["name"] == "beta" and u["field"] == "hypothesis_status"
               for u in out["unmappable"])


def test_backfill_from_kb_reports_orphan_lists(monkeypatch, tmp_path):
    """Orphan reporting: kb doc without registry row and registry row without kb doc."""
    from algua.config.settings import get_settings
    from algua.knowledge.frontmatter import render_doc
    from algua.registry.db import connect, migrate

    vault = tmp_path / "vault"
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(vault))

    db_path = get_settings().db_path
    conn = connect(db_path)
    migrate(conn)
    # Row with no matching kb doc
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('no_doc','idea','2026-01-01','2026-01-01')"
    )
    conn.commit()
    conn.close()

    # kb doc with no matching registry row
    strat_dir = vault / "strategies"
    strat_dir.mkdir(parents=True)
    fm = {"name": "ghost_doc", "stage": "idea"}
    (strat_dir / "ghost_doc.md").write_text(render_doc(fm, "## Hypothesis\n"))

    out = _json(runner.invoke(app, ["registry", "backfill-from-kb"]))
    assert "no_doc" in out["registry_rows_without_kb_doc"]
    assert "ghost_doc" in out["kb_docs_without_registry_row"]


# ---------------------------------------------------------------------------
# Fix 1: backfill-from-kb keys on filename (doc.stem), not frontmatter name
# ---------------------------------------------------------------------------

def test_backfill_keys_on_filename_not_frontmatter_name(monkeypatch, tmp_path):
    """When frontmatter name: differs from filename, doc.stem is used as the registry key;
    the mismatch is reported in frontmatter_name_mismatches."""
    from algua.config.settings import get_settings
    from algua.knowledge.frontmatter import render_doc
    from algua.registry.db import connect, migrate

    vault = tmp_path / "vault"
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(vault))

    db_path = get_settings().db_path
    conn = connect(db_path)
    migrate(conn)
    # Row registered under the filename stem, NOT under the frontmatter name.
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('file_stem','idea','2026-01-01','2026-01-01')"
    )
    conn.commit()
    conn.close()

    strat_dir = vault / "strategies"
    strat_dir.mkdir(parents=True)
    # The file is named file_stem.md but frontmatter says name: wrong_name
    fm = {"name": "wrong_name", "stage": "idea", "family": "[[momentum]]"}
    (strat_dir / "file_stem.md").write_text(render_doc(fm, "## Hypothesis\n"))

    out = _json(runner.invoke(app, ["registry", "backfill-from-kb"]))
    # The strategy should be processed under file_stem (the filename), not wrong_name.
    assert "file_stem" in out["processed"]
    # The mismatch should be reported.
    assert any(m["file"] == "file_stem.md" and m["frontmatter_name"] == "wrong_name"
               for m in out["frontmatter_name_mismatches"])
    # The family fill should have landed on file_stem (not on wrong_name).
    rec = _json(runner.invoke(app, ["registry", "show", "file_stem"]))
    assert rec["family"] == "momentum"


# ---------------------------------------------------------------------------
# Fix 4: backfill validates derived_from (self-reference and ghost name)
# ---------------------------------------------------------------------------

def test_backfill_derived_from_self_reference_goes_to_unmappable(monkeypatch, tmp_path):
    """A kb doc with derived_from pointing to itself must land in unmappable, not stored."""
    from algua.config.settings import get_settings
    from algua.knowledge.frontmatter import render_doc
    from algua.registry.db import connect, migrate

    vault = tmp_path / "vault"
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(vault))

    db_path = get_settings().db_path
    conn = connect(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('self_ref','idea','2026-01-01','2026-01-01')"
    )
    conn.commit()
    conn.close()

    strat_dir = vault / "strategies"
    strat_dir.mkdir(parents=True)
    # derived_from is a wikilink to the strategy itself (self-reference)
    fm = {"name": "self_ref", "stage": "idea", "derived_from": "[[self_ref]]"}
    (strat_dir / "self_ref.md").write_text(render_doc(fm, "## Hypothesis\n"))

    out = _json(runner.invoke(app, ["registry", "backfill-from-kb"]))
    assert any(u["name"] == "self_ref" and u["field"] == "derived_from"
               for u in out["unmappable"])
    rec = _json(runner.invoke(app, ["registry", "show", "self_ref"]))
    assert rec["derived_from"] is None


def test_backfill_derived_from_ghost_goes_to_unmappable(monkeypatch, tmp_path):
    """A kb doc with derived_from pointing to an unregistered strategy must go to unmappable."""
    from algua.config.settings import get_settings
    from algua.knowledge.frontmatter import render_doc
    from algua.registry.db import connect, migrate

    vault = tmp_path / "vault"
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(vault))

    db_path = get_settings().db_path
    conn = connect(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('child_strat','idea','2026-01-01','2026-01-01')"
    )
    conn.commit()
    conn.close()

    strat_dir = vault / "strategies"
    strat_dir.mkdir(parents=True)
    # derived_from references a strategy that is not in the registry
    fm = {"name": "child_strat", "stage": "idea", "derived_from": "[[ghost_parent]]"}
    (strat_dir / "child_strat.md").write_text(render_doc(fm, "## Hypothesis\n"))

    out = _json(runner.invoke(app, ["registry", "backfill-from-kb"]))
    assert any(u["name"] == "child_strat" and u["field"] == "derived_from"
               and u["value"] == "ghost_parent"
               for u in out["unmappable"])
    rec = _json(runner.invoke(app, ["registry", "show", "child_strat"]))
    assert rec["derived_from"] is None


def test_backfill_derived_from_valid_parent_is_stored(monkeypatch, tmp_path):
    """A kb doc with a valid derived_from wikilink stores the bare parent name in the registry."""
    from algua.knowledge.frontmatter import render_doc

    vault = tmp_path / "vault"
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(vault))

    # Register both parent and child via the CLI (plain add leaves derived_from NULL).
    runner.invoke(app, ["registry", "add", "parent"])
    runner.invoke(app, ["registry", "add", "child"])

    # Seed the child kb doc with a valid wikilinked derived_from pointing to parent.
    strat_dir = vault / "strategies"
    strat_dir.mkdir(parents=True)
    fm = {"name": "child", "stage": "idea", "derived_from": "[[parent]]"}
    (strat_dir / "child.md").write_text(render_doc(fm, "## Hypothesis\n"))

    out = _json(runner.invoke(app, ["registry", "backfill-from-kb"]))
    # child must not appear in unmappable
    assert not any(u["name"] == "child" and u["field"] == "derived_from"
                   for u in out["unmappable"])
    # derived_from must be stored as the bare parent name (unwikilinked)
    rec = _json(runner.invoke(app, ["registry", "show", "child"]))
    assert rec["derived_from"] == "parent"


def test_go_live_allows_second_live_strategy_with_allocation(monkeypatch, tmp_path):
    # one strategy already live; a SECOND with an allocation now reaches the go-live challenge
    from algua.registry.repository import ArtifactIdentity
    _force_live(monkeypatch, tmp_path, "already")
    name = _seed_forward_tested(monkeypatch, tmp_path, "s2")
    _allocate(monkeypatch, tmp_path, name, 1000.0)
    # patch compute_artifact_hashes so s2 (no real module) doesn't raise StrategyNotFound
    monkeypatch.setattr(
        "algua.cli.registry_cmd.compute_artifact_hashes",
        lambda n: ArtifactIdentity(code_hash="aabb", config_hash="ccdd", dependency_hash="eeff"),
    )
    r = runner.invoke(app, ["registry", "transition", name, "--to", "live", "--actor", "human"])
    assert r.exit_code == 0  # a challenge is issued (no ≤1-live refusal)
    assert json.loads(r.stdout)["action"] == "go_live_challenge"
