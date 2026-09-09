"""Verification for the forage launcher (ideation engine spec 2026-09-08 §5) and the leap
launcher (§6): both are TRUSTED DRIVERS wrapping a sandboxed Codex agent, and what these tests
pin is the privilege story (what the agent may reach) plus the driver's own authority-side step
list — the two things a refactor can silently weaken. Also carries the systemd units and the
sources-registry seed check for both stages.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FORAGE = REPO / ".codex" / "scripts" / "forage.sh"
LEAP = REPO / ".codex" / "scripts" / "leap.sh"


def _dry(script, *args):
    return subprocess.run(["bash", str(script), "--dry-run", *args], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


def test_forage_dry_run_is_sandboxed_web_search_only_no_registry():
    out = _dry(FORAGE, "--categories", "momentum,seasonality", "--max-notes", "4",
               "--timeout", "10m")
    assert "DRY RUN" in out and "codex exec" in out
    assert "-s workspace-write" in out
    assert "sandbox_workspace_write.network_access=false" in out
    assert "web_search=live" in out
    assert "--dangerously-bypass-approvals-and-sandbox" not in out
    assert "mcp_servers" not in out                       # MCP is opt-in (FORAGE_MCP=1)
    # No registry path VALUE reaches the agent (an `ALGUA_DB_PATH=...` assignment would be a leak;
    # the `-u ALGUA_DB_PATH` UNSET directive asserted below is the opposite of that).
    assert "ALGUA_DB_PATH=" not in out.split("would run:")[1]
    # The codex CHILD must have these four stripped even though the driver's own environment (and
    # its later accept/propose/digest calls) legitimately needs them.
    assert ("env -u ALGUA_DB_PATH -u ALGUA_KNOWLEDGE_DIR -u ALGUA_DATA_DIR "
            "-u ALGUA_MLFLOW_TRACKING_URI") in out
    assert "research inspirations accept" in out          # trusted driver lands the notes
    assert "categories: momentum,seasonality" in out and "max notes: 4" in out
    assert "forage/" in out and "timeout 10m" in out


def test_forage_mcp_opt_in_drops_the_sandbox_and_says_so():
    out = subprocess.run(["bash", str(FORAGE), "--dry-run"], cwd=REPO, capture_output=True,
                         text=True, check=True, env={**os.environ, "FORAGE_MCP": "1"}).stdout
    assert "--dangerously-bypass-approvals-and-sandbox" in out
    assert "NO OS WALL" in out and "paper-search-mcp==" in out  # pinned spec


def test_forage_rejects_unknown_argument():
    proc = subprocess.run(["bash", str(FORAGE), "--bogus"], cwd=REPO, capture_output=True,
                          text=True)
    assert proc.returncode == 2


def test_forage_units_shaped_and_daily():
    svc = (REPO / "deploy/systemd/algua-forage.service").read_text()
    assert "Type=oneshot" in svc and "forage.sh" in svc and "TimeoutStartSec=" in svc
    tmr = (REPO / "deploy/systemd/algua-forage.timer").read_text()
    assert "OnCalendar=*-*-* 03:00:00 UTC" in tmr and "Persistent=true" in tmr


def test_sources_registry_seed_has_venues_for_every_category():
    # The TRACKED artifact is the SEED under deploy/ — kb/inspirations/ is runtime state and
    # git-ignored (the vault's own registry is curated by a human and stamped by write-yield).
    import yaml
    reg = yaml.safe_load((REPO / "deploy/kb/inspirations/_sources.yaml").read_text())
    cats = {line.split()[0] for line in (REPO / ".codex/categories.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")}
    covered = set()
    for v in reg["venues"]:
        covered |= set(v["categories"])
    assert cats <= covered


def test_forage_dry_run_leaves_the_rotation_cursor_untouched(tmp_path):
    cursor = tmp_path / "forage-cursor"
    cursor.write_text("3")
    subprocess.run(["bash", str(FORAGE), "--dry-run"], cwd=REPO, capture_output=True, text=True,
                   check=True, env={**os.environ, "ALGUA_DATA_DIR": str(tmp_path)})
    assert cursor.read_text() == "3"


def test_forage_lock_skip_leaves_the_rotation_cursor_untouched(tmp_path):
    # The cursor write-back happens only after the flock is held AND the worktree + uv sync
    # succeed — a lock-skip must exit 0 before ever touching CURSOR_FILE, so the next firing
    # retries the SAME rotation slice rather than silently skipping past it.
    cursor = tmp_path / "forage-cursor"
    cursor.write_text("3")
    lock = tmp_path / "forage.lock"
    holder = subprocess.Popen(["flock", "-n", str(lock), "sleep", "30"])
    try:
        deadline = time.monotonic() + 5
        while not lock.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)  # give flock a moment to actually acquire, not just create the file
        proc = subprocess.run(
            ["bash", str(FORAGE)], cwd=REPO, capture_output=True, text=True, timeout=30,
            env={**os.environ, "ALGUA_DATA_DIR": str(tmp_path)})
        assert proc.returncode == 0, proc.stderr
        assert cursor.read_text() == "3"
    finally:
        holder.terminate()
        holder.wait(timeout=5)


# --- Leap (spec §6) ------------------------------------------------------------------------------


def test_leap_dry_run_is_sandboxed_no_web_scratch_db_then_import():
    out = _dry(LEAP, "--max-ideas", "5", "--timeout", "10m", "--force")
    assert "-s workspace-write" in out
    assert "sandbox_workspace_write.network_access=false" in out
    assert "web_search=disabled" in out
    assert "mcp_servers" not in out and "--dangerously-bypass" not in out
    assert ".leap-scratch/data/algua.db" in out             # ALGUA_DB_PATH -> scratch copy
    assert "research idea import --from" in out            # trusted driver imports
    assert "research inspirations write-yield" in out       # scorecard -> _sources.yaml
    assert "max ideas: 5" in out and "leap/" in out


def test_leap_dry_run_prints_every_planned_driver_step():
    # The dry run is the ONLY cheap check that the driver still does its whole authority-side
    # step list (seed -> codex -> import -> yield -> digest) in the right order; each line here
    # is one step that would otherwise be droppable without any test noticing.
    out = _dry(LEAP, "--max-ideas", "5", "--timeout", "10m", "--force")
    assert "DRY RUN" in out
    assert "depth gate: skipped (--force)" in out
    assert "would create worktree: " in out and "on branch leap/" in out
    assert "max ideas: 5" in out
    assert "would seed scratch from: " in out
    assert "would pre-warm env: " in out
    assert "would run: " in out and "codex exec" in out and "timeout 10m" in out
    assert "would import via: uv run algua research idea import --from" in out
    assert "--seeded-max-id" in out and "--critic-file" in out
    assert ("would write yield via: uv run algua research inspirations write-yield "
            "--from-scorecard -") in out
    assert "would append digest to: " in out and "leap-runs.jsonl" in out
    # Order matters: nothing authoritative may be planned before the agent's run.
    assert out.index("would run: ") < out.index("would import via: ")
    assert out.index("would import via: ") < out.index("would write yield via: ")


def test_leap_dry_run_without_force_still_gates_on_pool_depth():
    out = _dry(LEAP, "--max-ideas", "5")
    assert "would check depth: " in out and "research idea depth" in out


def test_leap_rejects_unknown_argument():
    proc = subprocess.run(["bash", str(LEAP), "--bogus"], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 2


def test_leap_units_every_two_hours_at_half_past_and_disjoint_from_paper():
    from tests.test_operator_layer import _fire_minutes, _oncalendar
    leap = _fire_minutes(_oncalendar("algua-leap.timer"))
    paper = _fire_minutes(_oncalendar("algua-paper.timer"))
    assert leap == {30} and not (leap & paper)
    tmr = (REPO / "deploy/systemd/algua-leap.timer").read_text()
    assert "OnCalendar=*-*-* 00/2:30:00 UTC" in tmr


def test_leap_units_shaped_and_installed():
    svc = (REPO / "deploy/systemd/algua-leap.service").read_text()
    assert "Type=oneshot" in svc and "leap.sh" in svc
    assert "TimeoutStartSec=2100" in svc          # > LEAP_TIMEOUT (25m) + prewarm
    tmr = (REPO / "deploy/systemd/algua-leap.timer").read_text()
    assert "Persistent=true" in tmr and "WantedBy=timers.target" in tmr
    installer = (REPO / "deploy/systemd/install-user-units.sh").read_text()
    assert "algua-leap.service" in installer and "algua-leap.timer" in installer


# --- Leap fix round 1 (review findings) -----------------------------------------------------------


def test_leap_dry_run_scratch_mlflow_tracking_uri_not_authority():
    # Without an explicit ALGUA_MLFLOW_TRACKING_URI in the codex child's `env` prefix, the agent
    # would inherit the unit's EnvironmentFile= (authority) tracking URI — a needless authority-path
    # leak into the sandbox. The `would run:` line is the codex invocation itself, so this pins the
    # var lives there, scoped under .leap-scratch, never the bare/authority value.
    out = _dry(LEAP, "--max-ideas", "5", "--timeout", "10m", "--force")
    run_line = out.split("would run: ", 1)[1].splitlines()[0]
    assert "ALGUA_MLFLOW_TRACKING_URI=" in run_line
    mlflow_val = next(
        tok for tok in run_line.split() if tok.startswith("ALGUA_MLFLOW_TRACKING_URI="))
    val = mlflow_val.split("=", 1)[1]
    assert val.endswith("/.leap-scratch/mlruns")


def _extract_import_capture_block() -> str:
    # The IMPORT_OUT capture (stdout-only) plus the `_import_field` fail-loud parsing, extracted
    # verbatim from leap.sh via literal start/end markers — the same "run the ACTUAL launcher code,
    # not a reimplementation" technique test_research_run_digest.py uses for run-research-loop.sh's
    # heredocs, applied here to a plain shell block instead of a python heredoc.
    src = LEAP.read_text(encoding="utf-8")
    start = 'echo "Importing survivors via: ${IMPORT_CMD[*]}"'
    end = "CRITIC_ROWS=\"$(_import_field critic_rows '0')\""
    start_idx = src.index(start)
    end_idx = src.index(end, start_idx) + len(end)
    return src[start_idx:end_idx]


_IMPORT_CAPTURE_SRC = _extract_import_capture_block()


def _fake_uv_bin(tmp_path: Path, *, stdout: str, stderr: str = "") -> Path:
    bindir = tmp_path / "fakebin-uv"
    bindir.mkdir(exist_ok=True)
    fake_uv = bindir / "uv"
    script = "#!/usr/bin/env bash\n"
    if stderr:
        script += f"cat >&2 <<'ERR'\n{stderr}\nERR\n"
    script += f"cat <<'OUT'\n{stdout}\nOUT\n"
    script += "exit 0\n"
    fake_uv.write_text(script, encoding="utf-8")
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


def _run_import_capture(tmp_path: Path, fake_uv_bindir: Path) -> dict:
    run_log = tmp_path / "run.log"
    script = tmp_path / "run_import_capture.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'REPO_ROOT={tmp_path!s}\n'
        f'RUN_LOG={run_log!s}\n'
        "IMPORT_CMD=(uv run algua research idea import --from x --run y --max 1"
        " --seeded-max-id 0 --critic-file z)\n"
        f"{_IMPORT_CAPTURE_SRC}\n"
        'echo "RESULT_IMPORTED=${IMPORTED_JSON}"\n'
        'echo "RESULT_SKIPPED=${SKIPPED_JSON}"\n'
        'echo "RESULT_CRITIC=${CRITIC_ROWS}"\n'
        'echo "RESULT_PARSE_ERROR=${IMPORT_PARSE_ERROR}"\n',
        encoding="utf-8",
    )
    env = {**os.environ, "PATH": f"{fake_uv_bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env,
                          timeout=30, check=True)
    result: dict = {"stdout": proc.stdout, "stderr": proc.stderr, "run_log": run_log}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_"):
            key, _, val = line.partition("=")
            result[key] = val
    return result


def test_leap_import_stderr_noise_never_corrupts_the_parsed_imported_list(tmp_path):
    # The bug this closes: folding stderr into the captured JSON (`2>&1`) meant ANY stderr line (a
    # `uv` resolve notice, a Python warning) silently turned IMPORTED_JSON into "[]" with rc 0 —
    # mark-used would then never run, and the run looked healthy. A stderr line ahead of clean JSON
    # on stdout must not affect parsing at all.
    payload = '{"ok": true, "imported": [123, 456], "skipped": [], "critic_rows": 1, "ceiling": 5}'
    fake_uv = _fake_uv_bin(tmp_path, stdout=payload,
                           stderr="warning: resolved uv.lock in 4ms (fake noise)")
    result = _run_import_capture(tmp_path, fake_uv)
    assert json.loads(result["RESULT_IMPORTED"]) == [123, 456]
    assert json.loads(result["RESULT_CRITIC"]) == 1
    assert result["RESULT_PARSE_ERROR"] == "0"
    # The noise must still be visible somewhere (the run log or the terminal), not swallowed.
    assert "fake noise" in result["run_log"].read_text(encoding="utf-8")


def test_leap_import_output_not_json_fails_loudly_and_flags_the_digest(tmp_path):
    fake_uv = _fake_uv_bin(tmp_path, stdout="not json at all")
    result = _run_import_capture(tmp_path, fake_uv)
    assert json.loads(result["RESULT_IMPORTED"]) == []
    assert result["RESULT_PARSE_ERROR"] == "1"
    assert "WARNING: import output was not JSON" in result["stderr"]
    # Printed exactly once, not once per field (imported/skipped/critic_rows).
    assert result["stderr"].count("WARNING: import output was not JSON") == 1


def test_leap_seeded_max_id_read_and_backup_share_one_python_invocation():
    # The seed watermark (authority's max idea id BEFORE the agent runs) must be read in the SAME
    # python invocation that performs the sqlite backup, so the race window between "read the
    # watermark" and "seed the scratch copy" is as tight as possible; and the comment must name the
    # REAL safety net (research idea import's own collision re-check), not just the read timing.
    src = LEAP.read_text(encoding="utf-8")
    m = re.search(
        r"seeding scratch registry.*?<<'PY'\n(.*?)\nPY\n", src, re.DOTALL)
    assert m, "could not find the scratch-seeding python heredoc in leap.sh"
    body = m.group(1)
    assert "SELECT COALESCE(MAX(id),0) FROM ideas" in body
    assert "src.backup(dst)" in body
    assert "collision" in src  # the corrected comment documents the actual safety net


def test_leap_dry_run_seeded_max_id_is_a_placeholder_not_a_premature_read():
    # A dry run must not need (or claim) a live watermark value — the value is only meaningful once
    # the scratch registry is actually seeded, which a dry run never does.
    out = _dry(LEAP, "--max-ideas", "5", "--timeout", "10m", "--force")
    assert "would import via: uv run algua research idea import --from" in out
    assert "--seeded-max-id <read at seed time>" in out


_HEREDOC_RE = re.compile(r"<<'PY'[^\n]*\n(.*?)\n^PY$", re.DOTALL | re.MULTILINE)


def _leap_heredocs() -> list[str]:
    return _HEREDOC_RE.findall(LEAP.read_text(encoding="utf-8"))


# Heredoc 0 = the combined seed-watermark-read + sqlite-backup step; heredoc 1 = the mark-used /
# mark-exhausted inspiration bookkeeping (what the exhausted-section tests below exercise); heredoc
# 2 = the digest append. Keep these indices in sync when adding a heredoc to leap.sh.
_EXHAUSTED_SRC = _leap_heredocs()[1]


def _fake_algua_bin(tmp_path: Path) -> Path:
    # A fake `uv` intercepting exactly the two `uv run algua research inspirations ...` calls the
    # extracted heredoc makes (mark-used, mark-exhausted), so this test never shells out to the
    # real CLI / touches the real vault. `research idea show` is not reached (imported_json is
    # always "[]" below, so the mark-used loop never runs).
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    fake_uv = bindir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *mark-exhausted* || \"$*\" == *mark-used* ]]; then\n"
        "  echo '{\"ok\": true}'\n"
        "  exit 0\n"
        "fi\n"
        "echo \"unexpected fake-uv invocation: $*\" >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


def _run_exhausted_block(tmp_path: Path, report_text: str | None) -> subprocess.CompletedProcess:
    report_path = tmp_path / "leap-report.md"
    if report_text is not None:
        report_path.write_text(report_text, encoding="utf-8")
    bindir = _fake_algua_bin(tmp_path)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    return subprocess.run(
        [sys.executable, "-", "[]", str(report_path), str(tmp_path)],
        input=_EXHAUSTED_SRC, capture_output=True, text=True, env=env, timeout=30)


def test_leap_exhausted_section_accepts_any_heading_depth(tmp_path):
    for hashes in ("##", "###", "####"):
        proc = _run_exhausted_block(
            tmp_path, f"{hashes} Exhausted inspirations\n- 2026-09-07-index-add-crowding\n")
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout) == ["2026-09-07-index-add-crowding"]
        assert 'note: no "Exhausted inspirations" section' not in proc.stderr


def test_leap_exhausted_section_absent_vs_present_but_empty(tmp_path):
    # Absent (no heading at all) gets a note distinguishing it from a deliberate empty list.
    absent = _run_exhausted_block(tmp_path, "some other report content, no heading at all\n")
    assert absent.returncode == 0, absent.stderr
    assert json.loads(absent.stdout) == []
    assert 'note: no "Exhausted inspirations" section in leap-report.md' in absent.stderr

    # Present but empty: no such note (the agent deliberately judged nothing spent).
    empty = _run_exhausted_block(tmp_path, "## Exhausted inspirations\n\n## Something else\n")
    assert empty.returncode == 0, empty.stderr
    assert json.loads(empty.stdout) == []
    assert 'note: no "Exhausted inspirations" section' not in empty.stderr


# --- Vault placement (final fix wave): the seed lives in deploy/, the vault is runtime state ------


def test_vault_inspirations_dir_is_git_ignored_and_the_seed_is_tracked():
    assert "kb/inspirations/" in (REPO / ".gitignore").read_text()
    assert (REPO / "deploy/kb/inspirations/_sources.yaml").exists()
    tracked = subprocess.run(["git", "ls-files", "kb/inspirations"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    assert tracked.strip() == ""


def test_installer_dry_run_would_seed_the_sources_registry_when_absent(tmp_path):
    installer = REPO / "deploy/systemd/install-user-units.sh"
    env = {**os.environ, "ALGUA_KNOWLEDGE_DIR": str(tmp_path / "kb")}
    out = subprocess.run(["bash", str(installer), "--dry-run"], cwd=REPO, capture_output=True,
                         text=True, check=True, env=env).stdout
    assert "would seed sources registry: " in out
    assert str(tmp_path / "kb" / "inspirations" / "_sources.yaml") in out
    assert not (tmp_path / "kb").exists()          # a dry run writes nothing


def test_installer_never_overwrites_an_existing_sources_registry(tmp_path):
    installer = REPO / "deploy/systemd/install-user-units.sh"
    dest = tmp_path / "kb" / "inspirations" / "_sources.yaml"
    dest.parent.mkdir(parents=True)
    dest.write_text("venues: []  # human-curated\n")
    env = {**os.environ, "ALGUA_KNOWLEDGE_DIR": str(tmp_path / "kb")}
    out = subprocess.run(["bash", str(installer), "--dry-run"], cwd=REPO, capture_output=True,
                         text=True, check=True, env=env).stdout
    assert "already present" in out and "would seed sources registry" not in out
    assert dest.read_text() == "venues: []  # human-curated\n"


def test_forage_seeds_the_sources_registry_when_the_vault_has_none(tmp_path):
    # A vault relocated via ALGUA_KNOWLEDGE_DIR self-heals at forage startup — copy-if-absent,
    # never over an existing registry.
    kb = tmp_path / "kb"
    env = {**os.environ, "ALGUA_KNOWLEDGE_DIR": str(kb)}
    out = subprocess.run(["bash", str(FORAGE), "--dry-run"], cwd=REPO, capture_output=True,
                         text=True, check=True, env=env).stdout
    assert "would seed sources registry: " in out
    assert not kb.exists()

    dest = kb / "inspirations" / "_sources.yaml"
    dest.parent.mkdir(parents=True)
    dest.write_text("venues: []\n")
    out = subprocess.run(["bash", str(FORAGE), "--dry-run"], cwd=REPO, capture_output=True,
                         text=True, check=True, env=env).stdout
    assert "seed sources registry" not in out
    assert dest.read_text() == "venues: []\n"


# --- Forage fix round 2 (final fix wave) ---------------------------------------------------------


def _extract_block(script: Path, start: str, end: str) -> str:
    src = script.read_text(encoding="utf-8")
    start_idx = src.index(start)
    end_idx = src.index(end, start_idx) + len(end)
    return src[start_idx:end_idx]


_ACCEPT_CAPTURE_SRC = _extract_block(
    FORAGE,
    'echo "Landing foraged notes via: ${ACCEPT_CMD[*]}"',
    "REJECTED_JSON=\"$(_accept_field rejected '[]')\"")


def _run_accept_capture(tmp_path: Path, fake_uv_bindir: Path) -> dict:
    run_log = tmp_path / "run.log"
    script = tmp_path / "run_accept_capture.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"REPO_ROOT={tmp_path!s}\n"
        f"RUN_LOG={run_log!s}\n"
        "ACCEPT_CMD=(uv run algua research inspirations accept --from x --run y --max 1)\n"
        f"{_ACCEPT_CAPTURE_SRC}\n"
        'echo "RESULT_ACCEPTED=${ACCEPTED_JSON}"\n'
        'echo "RESULT_REJECTED=${REJECTED_JSON}"\n'
        'echo "RESULT_PARSE_ERROR=${ACCEPT_PARSE_ERROR}"\n',
        encoding="utf-8",
    )
    env = {**os.environ, "PATH": f"{fake_uv_bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env,
                          timeout=30, check=True)
    result: dict = {"stdout": proc.stdout, "stderr": proc.stderr, "run_log": run_log}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_"):
            key, _, val = line.partition("=")
            result[key] = val
    return result


def test_forage_accept_stderr_noise_never_corrupts_the_accepted_list(tmp_path):
    # Folding stderr into the captured JSON (`2>&1`) meant ANY stderr line (a `uv` resolve notice,
    # a Python warning) silently turned ACCEPTED_JSON into "[]" with rc 0 — the digest then said
    # the run foraged nothing while the vault had the notes. Same bug leap.sh already fixed.
    payload = ('{"ok": true, "accepted": ["2026-09-08-a-note"], '
               '"rejected": [{"file": "x.md", "reasons": ["bad filename: x.md"]}]}')
    fake_uv = _fake_uv_bin(tmp_path, stdout=payload,
                           stderr="warning: resolved uv.lock in 4ms (fake noise)")
    result = _run_accept_capture(tmp_path, fake_uv)
    assert json.loads(result["RESULT_ACCEPTED"]) == ["2026-09-08-a-note"]
    assert json.loads(result["RESULT_REJECTED"])[0]["file"] == "x.md"
    assert result["RESULT_PARSE_ERROR"] == "0"
    assert "fake noise" in result["run_log"].read_text(encoding="utf-8")


def test_forage_accept_output_not_json_fails_loudly_and_flags_the_digest(tmp_path):
    fake_uv = _fake_uv_bin(tmp_path, stdout="not json at all")
    result = _run_accept_capture(tmp_path, fake_uv)
    assert json.loads(result["RESULT_ACCEPTED"]) == []
    assert result["RESULT_PARSE_ERROR"] == "1"
    assert "WARNING: accept output was not JSON" in result["stderr"]
    assert result["stderr"].count("WARNING: accept output was not JSON") == 1


def test_forage_digest_records_the_accept_parse_error_flag():
    src = FORAGE.read_text(encoding="utf-8")
    assert '"accept_parse_error": accept_parse_error == "1"' in src


# forage.sh heredoc 0 = the seen-hash reader; 1 = the sources-registry slice; 2 = the
# proposed-venue validator/proposer (extracted here); 3 = the digest append. Keep these indices in
# sync when adding a heredoc to forage.sh.
_PROPOSED_SRC = _HEREDOC_RE.findall(FORAGE.read_text(encoding="utf-8"))[2]


def test_forage_proposed_venues_block_emits_only_json_on_stdout(tmp_path):
    # PROPOSED_JSON is a command substitution of this block's stdout. A progress line printed on
    # stdout ("proposed venue: X") made the whole capture unparseable, so the digest row's
    # `proposed` list came back empty even on a run that really did add a venue.
    report = tmp_path / "forage-report.md"
    report.write_text("## Proposed venues\n"
                      "- key: blog/goodone kind: blog url: https://g/x categories: [momentum]\n"
                      "- key: BADKEY kind: blog url: https://g/y categories: [momentum]\n")
    cats = tmp_path / "categories.txt"
    cats.write_text("momentum\n")
    bindir = tmp_path / "fakebin-propose"
    bindir.mkdir()
    fake_uv = bindir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\necho '{\"ok\": true}'\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}

    proc = subprocess.run(
        [sys.executable, "-", str(report), str(cats), str(tmp_path)],
        input=_PROPOSED_SRC, capture_output=True, text=True, env=env, timeout=30)

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == ["blog/goodone"]      # stdout is PURE JSON
    assert "proposed venue: blog/goodone" in proc.stderr    # the progress line still shows
    assert "BADKEY" in proc.stderr


def test_forage_seen_hash_list_is_trimmed_to_the_newest_300():
    assert "hashes[-300:]" in FORAGE.read_text(encoding="utf-8")


def test_forage_help_covers_the_env_block():
    out = subprocess.run(["bash", str(FORAGE), "-h"], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout
    assert "Env: FORAGE_MAX_NOTES" in out and "FORAGE_TIMEOUT" in out


# --- Leap fix round 2 (final fix wave) -----------------------------------------------------------


def test_leap_reads_every_non_exhausted_note_not_only_fresh():
    # `mark-used` flips a note to `used` the first time it feeds a hypothesis, but a note usually
    # holds more than one leap. Reading `--status fresh` starved leap of material after one pass.
    src = LEAP.read_text(encoding="utf-8")
    assert "--exclude-status exhausted --limit 20 --rare-first" in src
    assert "--status fresh" not in src


def test_leap_dry_run_prints_the_material_gate():
    out = _dry(LEAP, "--max-ideas", "5", "--force")
    assert ("would check material: uv run algua research inspirations list "
            "--exclude-status exhausted --limit 1") in out
    # The material gate is planned AFTER the (cheaper) depth gate and BEFORE any worktree.
    assert out.index("depth gate: skipped") < out.index("would check material: ")
    assert out.index("would check material: ") < out.index("would create worktree: ")


def test_leap_with_an_empty_vault_exits_zero_and_creates_no_worktree(tmp_path):
    # --force clears the depth gate; the vault is a fresh empty dir, so the material gate must
    # stop the run before the flock, the worktree, the uv sync and the codex call.
    runs_before = set((REPO / ".runs").glob("leap-*")) if (REPO / ".runs").exists() else set()
    env = {**os.environ, "ALGUA_KNOWLEDGE_DIR": str(tmp_path / "empty-kb"),
           "ALGUA_DATA_DIR": str(tmp_path / "data")}
    proc = subprocess.run(["bash", str(LEAP), "--force"], cwd=REPO, capture_output=True,
                          text=True, timeout=180, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "no fresh material (0 non-exhausted inspirations); nothing to do" in proc.stdout
    runs_after = set((REPO / ".runs").glob("leap-*")) if (REPO / ".runs").exists() else set()
    assert runs_after == runs_before


# --- Unit shape + operator docs (final fix wave) --------------------------------------------------


def test_forage_service_is_installable_like_the_other_oneshot_units():
    # Without [Install], `systemctl enable algua-forage.service` fails and the installer's
    # multi-user.target -> default.target rewrite has nothing to rewrite. Mirrors
    # algua-mergeback-drain.service / algua-leap.service.
    svc = (REPO / "deploy/systemd/algua-forage.service").read_text()
    assert "[Install]\nWantedBy=multi-user.target" in svc


def test_leap_timer_and_readme_tell_the_truth_about_the_shared_minute():
    # Leap fires at :30 — the merge-back drainer's own minute. The old comments claimed the grid
    # was disjoint from the drainer, which it is not; what is true is that leap takes no
    # operator.lock, so sharing the minute is harmless.
    tmr = (REPO / "deploy/systemd/algua-leap.timer").read_text()
    assert "SHARES the merge-back drainer's :30 minute" in tmr
    assert "leap does NOT" in tmr and "operator.lock" in tmr
    readme = (REPO / "deploy/systemd/README.md").read_text()
    assert "shares the merge-back drainer's `:30` minute" in readme
    assert "leap\ntakes no `operator.lock`" in readme


def test_env_example_documents_the_driver_timeouts():
    env = (REPO / "deploy/systemd/algua.env.example").read_text()
    for line in ("# LEAP_TIMEOUT=25m", "# FORAGE_TIMEOUT=20m", "# SYNC_TIMEOUT=5m"):
        assert line in env
