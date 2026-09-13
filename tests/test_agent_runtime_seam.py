"""The agent-runtime seam (`.opencode/scripts/run_agent.sh`).

These tests exist because the seam is the ONLY place the repo names a runtime, a model or a
sandbox flag. Before it, each of the three loop drivers embedded its own `codex exec -s ... -c ...`
invocation and the model came from a user-global config file no driver passed — so the runtime was
smeared across three scripts plus an untracked file, and the test suite pinned a vendor's flag
spelling as if it were an invariant.

What is actually invariant is asserted here: one entry point, a hard timeout, a kernel write wall,
no prompt that can block an unattended run, and a prompt delivered as a FILE rather than as argv.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEAM = REPO / ".opencode" / "scripts" / "run_agent.sh"
CONFIG = REPO / "opencode.json"
AGENTS_DIR = REPO / ".opencode" / "agents"

MODES = ("research", "leap", "forage")


def _dry(mode: str, **env: str) -> str:
    proc = subprocess.run(
        [str(SEAM), "--mode", mode, "--workdir", str(REPO), "--prompt-file", str(CONFIG),
         "--dry-run"],
        cwd=REPO, capture_output=True, text=True, check=True,
        env={**__import__("os").environ, **env},
    )
    return proc.stdout


def test_seam_exists_and_is_executable():
    assert SEAM.is_file()
    assert SEAM.stat().st_mode & 0o111, "the drivers invoke the seam directly; it must be +x"


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_dry_runs_with_its_bounds(mode: str):
    out = _dry(mode)
    assert out.startswith("would run:")
    assert f"--agent {mode}" in out
    assert "timeout 45m" in out, "an unattended run must always carry a hard wall-clock bound"
    # The prompt rides a FILE. Linux caps a single argv entry at 128 KiB and the leap prompt
    # (inspirations + pool state + refuted list) is the one that gets close to it.
    assert " -f " in out


def test_only_forage_gets_the_web_search_backend():
    """Web access is a per-loop capability, granted by the SEAM -- not by config alone.

    This asserts the half that is verifiable without a live model: the search backend's env var is
    exported for forage and for nothing else. The per-agent `permission` blocks in
    `.opencode/agents/*.md` are defence in depth ON TOP of this, and are NOT yet verified to take
    effect -- `opencode debug agent` does not reflect them, and distinguishing "not applied" from
    "not rendered" needs a live run against a working provider. Do not treat those blocks as a
    wall until that run has happened; the wall that IS verified is the bwrap mount namespace below.
    """
    assert "OPENCODE_ENABLE_EXA=1" in _dry("forage")
    for mode in ("research", "leap"):
        assert "OPENCODE_ENABLE_EXA" not in _dry(mode)


def test_the_baseline_config_never_asks():
    """An unattended run that prompts is a run that hangs until its timeout.

    Measured on opencode 1.18.30: `doom_loop` and `external_directory` default to `ask`, and a
    probe run against an exhausted provider produced zero bytes and had to be killed. Every
    baseline value must therefore resolve to allow or deny.
    """
    config = json.loads(CONFIG.read_text())
    for name, value in config["permission"].items():
        assert value != "ask", f"permission {name!r} is 'ask'; an unattended run would block on it"


def test_the_baseline_config_does_not_decide_web_access_globally():
    """Web access must not be set at the top level, in either direction.

    A global `deny` resolved onto the forage agent and would have silently broken the only loop
    that needs the web -- caught by reading the resolved permissions rather than the config. A
    global `allow` would be worse: it would hand untrusted web text to the loop that writes
    strategy code.
    """
    config = json.loads(CONFIG.read_text())
    assert "websearch" not in config["permission"]
    assert "webfetch" not in config["permission"]


def test_the_kernel_write_wall_is_requested_when_available():
    """Tool-level permissions are not a filesystem wall.

    OpenCode's `edit` permission does not stop `bash` from opening an absolute path, and the
    research and leap loops legitimately need bash for `uv run algua`. Codex gave us a real wall
    via `-s workspace-write`; the seam re-imposes one with bwrap so the migration does not quietly
    trade a kernel guarantee for a prompt-level one.
    """
    if not __import__("shutil").which("bwrap"):
        pytest.skip("bwrap not installed; the seam falls back to ALGUA_AGENT_SANDBOX=none")
    out = _dry("research")
    assert "bwrap" in out
    assert "--ro-bind / /" in out, "everything outside the worktree must be read-only"
    assert f"--bind {REPO} {REPO}" in out, "the worktree itself must stay writable"
    assert "--tmpfs /tmp" in out, "a private /tmp closes the hole the Codex sandbox left open"


def test_sandbox_can_be_disabled_only_explicitly():
    out = _dry("research", ALGUA_AGENT_SANDBOX="none")
    assert "bwrap" not in out
    proc = subprocess.run(
        [str(SEAM), "--mode", "research", "--workdir", str(REPO), "--prompt-file", str(CONFIG),
         "--dry-run"],
        cwd=REPO, capture_output=True, text=True,
        env={**__import__("os").environ, "ALGUA_AGENT_SANDBOX": "bogus"},
    )
    assert proc.returncode == 2, "an unrecognised sandbox mode must fail closed, not fall back"


def test_model_precedence_flag_then_per_mode_then_global_then_config():
    assert "-m x/flag-model" in subprocess.run(
        [str(SEAM), "--mode", "leap", "--workdir", str(REPO), "--prompt-file", str(CONFIG),
         "--model", "x/flag-model", "--dry-run"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert "-m x/per-mode" in _dry("leap", ALGUA_AGENT_MODEL_LEAP="x/per-mode")
    assert "-m x/global" in _dry("leap", ALGUA_AGENT_MODEL="x/global")
    # Per-mode beats global.
    out = _dry("leap", ALGUA_AGENT_MODEL="x/global", ALGUA_AGENT_MODEL_LEAP="x/per-mode")
    assert "-m x/per-mode" in out and "x/global" not in out
    # With nothing set, no -m is passed at all: opencode.json decides, which is where model
    # choices belong. A driver should never have to know one.
    assert " -m " not in _dry("leap")


def test_bad_usage_fails_closed():
    for args in (["--mode", "bogus"], ["--workdir", str(REPO)], ["--mode", "leap"]):
        proc = subprocess.run([str(SEAM), *args, "--dry-run"], cwd=REPO,
                              capture_output=True, text=True)
        assert proc.returncode == 2, f"{args} should be a usage error"


def test_the_drivers_reach_the_runtime_only_through_the_seam():
    """No driver may name the runtime, a model or a sandbox flag itself.

    This is the invariant the whole seam exists for. A driver that shells out to a runtime
    directly re-creates the smear the migration removed, and the next runtime change would once
    again have to touch every script.
    """
    drivers = sorted((REPO / ".opencode" / "scripts").glob("*.sh"))
    assert drivers, "no drivers found"
    for driver in drivers:
        if driver.name == "run_agent.sh":
            continue
        text = driver.read_text()
        assert "opencode run" not in text, f"{driver.name} invokes the runtime directly"
        assert "codex exec" not in text, f"{driver.name} still invokes codex"
        assert "-s workspace-write" not in text, f"{driver.name} names a vendor sandbox flag"


def test_every_referenced_agent_definition_exists():
    for mode in MODES:
        assert (AGENTS_DIR / f"{mode}.md").is_file(), f"seam mode {mode!r} has no agent definition"
    # The research loop delegates; its subagents must exist or the delegation silently degrades.
    for sub in ("author", "interpret"):
        assert (AGENTS_DIR / f"{sub}.md").is_file()
    research = (AGENTS_DIR / "research.md").read_text()
    assert "author: allow" in research and "interpret: allow" in research


def test_interpret_declares_itself_read_only():
    """The read-only judge is what keeps a promote recommendation honest.

    If `interpret` could edit or run commands it could fix the strategy it is judging, and the
    split between authoring and judging would stop meaning anything.

    CAVEAT, and it is the reason this test is named "declares" rather than "is": the declaration
    is NOT yet verified to be enforced. `opencode debug agent interpret` does not show `edit` or
    `bash` denied -- they fall through to the catch-all -- and whether that means "not applied" or
    "not rendered by the debug command" cannot be settled without a live run against a working
    provider. Until that run happens, treat the read-only property as INTENDED, not guaranteed,
    and rely on the bwrap wall (which IS verified) for anything that matters.
    """
    text = (AGENTS_DIR / "interpret.md").read_text()
    assert "edit: deny" in text
    assert "bash: deny" in text
