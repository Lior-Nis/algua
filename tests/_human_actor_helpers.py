"""Shared test helpers for the authenticated `--actor human` gate (#329).

CLI tests that need a human-only relaxation must now authenticate: a bare `--actor human` only
issues a challenge. These helpers enroll an ephemeral test key into a tmp trust anchor and run the
full challenge -> sign -> resubmit dance, so a test can obtain a genuinely-authenticated human
promote with one call. The security property is preserved (a real signature is produced); the
helper just automates the two-step ceremony the operator would do by hand.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner


def install_human_actor_anchor(monkeypatch, tmp_path: Path, *, principal: str = "tester") -> Path:
    """Generate an ed25519 test key, enroll it for the algua-human-actor namespace into a tmp
    allowed_signers anchor, and point BOTH gate modules' ALLOWED_SIGNERS_PATH at it. Returns the
    private key path (used to sign challenges)."""
    key = tmp_path / "human_actor_key"
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "t", "-f", str(key)],
                   check=True, capture_output=True)
    pub = (tmp_path / "human_actor_key.pub").read_text().strip()
    keytype, keyblob = pub.split()[0], pub.split()[1]
    anchor = tmp_path / "allowed_signers"
    anchor.write_text(f'{principal} namespaces="algua-human-actor" {keytype} {keyblob}\n')
    monkeypatch.setattr("algua.registry.live_gate.ALLOWED_SIGNERS_PATH", anchor)
    # human_actor.py binds `from ... import ALLOWED_SIGNERS_PATH` at import, so its module-level
    # name is a SEPARATE reference — patch it too, otherwise verification reads the real anchor.
    monkeypatch.setattr("algua.registry.human_actor.ALLOWED_SIGNERS_PATH", anchor)
    return key


_sign_counter = 0


def _sign(key: Path, challenge: str, tmp_path: Path) -> Path:
    # Unique per-call filename: ssh-keygen writes `<file>.sig` and a test may sign several
    # challenges under one tmp_path — a fixed name could leave a prior invocation's signature in
    # place and mis-verify the next challenge.
    global _sign_counter
    _sign_counter += 1
    data = tmp_path / f"actor_challenge_{_sign_counter}.txt"
    data.write_text(challenge)
    sig = tmp_path / f"actor_challenge_{_sign_counter}.txt.sig"
    if sig.exists():
        sig.unlink()
    subprocess.run(
        ["ssh-keygen", "-Y", "sign", "-n", "algua-human-actor", "-f", str(key), str(data)],
        check=True, capture_output=True)
    return sig


def promote_signed(runner: CliRunner, app, base_args: list[str], key: Path, tmp_path: Path):
    """Run a gated promote as an AUTHENTICATED human: invoke once to get the challenge, sign it,
    then re-invoke with --actor-signature. `base_args` must already include `--actor human` and the
    command's flags. Returns the final CliRunner result."""
    first = runner.invoke(app, base_args)
    try:
        payload = json.loads(first.stdout)
    except (json.JSONDecodeError, ValueError):
        # stdout isn't JSON (e.g. an unstructured error/traceback path) — hand it back as-is.
        return first
    if not isinstance(payload, dict) or payload.get("action") != "human_actor_challenge":
        # No challenge was issued (e.g. the command fails-closed earlier for a different reason);
        # return the first result so the test can assert on it directly.
        return first
    sig = _sign(key, payload["challenge"], tmp_path)
    return runner.invoke(app, [*base_args, "--actor-signature", str(sig)])
