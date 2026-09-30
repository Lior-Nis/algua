"""Deterministic replay of one recorded frozen planner attempt (Story 1.3d contract §6, AC7).

Composed from production primitives only; nothing here re-implements the wire, the bars codec,
content verification or the launch:

- the row: ``registry.store.frozen_evidence.frozen_invocation``;
- its deployment's recorded descriptor: ``SqliteStrategyRepository.deployment_artifact_by_digest``
  and ``ArtifactRecord.frozen_manifest`` (the Story 1.3b parser);
- content: a FRESH ``FrozenContentVerifier`` over the data dir, as a restarted supervisor
  verifies it (a failure raises ``FrozenContentUnavailable``);
- the request: the recorded ``request_json`` bytes, re-parsed with the wire's strict
  ``parse_canonical`` / ``decode_pairs`` only to name the bars they bind;
- the bars: ``StoreBackedProvider`` over the row's ``snapshot_id`` (what the paper lane's
  ``select_provider`` builds for ``--snapshot``), for ``sorted(set(gate_universe) | held)`` over
  ``[bars_start, bars_end)``, exactly the fetch ``run_tick`` makes; re-encoded with ``encode_bars``
  and decoded with ``decode_bars``, whose logical ``bars_digest`` must be the request's;
- the launch: ``frozen_invocation.unsupported_content`` then ``launch_child`` (the sealed
  invocation directory, the exact §5 argv and replacement environment, the §6 bounds) with
  ``run_contained``, and ``process_failure`` / ``process_diagnostic`` for how the child ended.

Seams, stated rather than hidden: production has no lookup of a deployment by id (only
``active_deployment(strategy_id)``, which misses a retired epoch), so one SELECT reads the
deployment's artifact id and manifest digest; the interpreter is ``<environment>/bin/python``,
the path ``FrozenTenant.interpreter`` names. Nothing here needs Git or uv.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from algua.contracts.frozen_evidence import FrozenAttempt
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.live.frozen_invocation import (
    Runner,
    launch_child,
    process_diagnostic,
    process_failure,
    unsupported_content,
)
from algua.live.frozen_wire_arrow import decode_bars, encode_bars
from algua.live.frozen_wire_json import decode_pairs, parse_canonical
from algua.live.planner_binding import bars_digest
from algua.primitives.contained_process import run_contained
from algua.registry.frozen_runtime import FrozenContentVerifier
from algua.registry.store import SqliteStrategyRepository
from algua.registry.store.frozen_evidence import frozen_invocation


class ReplayMismatch(Exception):
    """The recorded attempt cannot be replayed as recorded; ``reason`` names the check."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class ReplayInputs:
    """Everything one replayed child is launched over."""

    attempt: FrozenAttempt
    bundle_root: Path
    environment_root: Path
    request_json: bytes  # the recorded request, byte for byte
    bars_arrow: bytes  # the re-read bars, re-encoded by this supervisor


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _deployment_artifact(conn: sqlite3.Connection, deployment_id: int) -> tuple[int, str]:
    row = conn.execute(
        "SELECT d.artifact_id, a.manifest_digest FROM strategy_deployments d"
        " JOIN deployment_artifacts a ON a.id = d.artifact_id WHERE d.id = ?",
        (deployment_id,),
    ).fetchone()
    if row is None:
        raise ReplayMismatch("deployment", f"no deployment {deployment_id}")
    return int(row[0]), str(row[1])


def _replay_symbols(early: dict[str, Any]) -> list[str]:
    """The symbols ``run_tick`` fetched: the gate universe and every nonzero early position."""
    positions = decode_pairs(early["early_positions"], "early.early_positions")
    return sorted(set(early["gate_universe"]) | {s for s, q in positions.items() if q != 0})


def prepare_replay(
    conn: sqlite3.Connection, data_dir: Path, attempt: FrozenAttempt,
) -> ReplayInputs:
    """Verify the recorded content, re-read the recorded bars and require they are the ones the
    recorded request binds. Raises ``FrozenContentUnavailable`` or :class:`ReplayMismatch`."""
    if attempt.request_json is None or attempt.bars_start is None or attempt.bars_end is None:
        raise ReplayMismatch("no_request", "a pre-launch refusal sent no request to replay")
    request_json = attempt.request_json.encode("utf-8")
    if _sha256(request_json) != attempt.request_sha256:
        raise ReplayMismatch("request_sha256", "the stored request is not the bytes recorded")
    request = parse_canonical(request_json)

    artifact_id, manifest_digest = _deployment_artifact(conn, attempt.deployment_id)
    manifest = SqliteStrategyRepository(conn).deployment_artifact_by_digest(
        manifest_digest).frozen_manifest()
    recorded = (attempt.phase, attempt.request_id, attempt.deployment_id, artifact_id,
                manifest_digest, manifest.bundle.digest, manifest.environment.digest)
    sent = tuple(request[key] for key in (
        "phase", "request_id", "deployment_id", "artifact_id", "manifest_digest",
        "bundle_digest", "environment_digest"))
    if sent != recorded:
        raise ReplayMismatch("identity", "the request does not name the recorded deployment")

    verifier = FrozenContentVerifier(data_dir)  # fresh: nothing cached from before a restart
    bundle_root = verifier.bundle(manifest.bundle, deployment_id=attempt.deployment_id)
    environment_root = verifier.environment(
        manifest.environment, deployment_id=attempt.deployment_id)

    early = request["early"]
    frame = StoreBackedProvider(DataStore(data_dir), attempt.snapshot_id).get_bars(
        _replay_symbols(early), datetime.fromisoformat(attempt.bars_start),
        datetime.fromisoformat(attempt.bars_end), early["timeframe"])
    bars_arrow = encode_bars(frame)
    if bars_digest(decode_bars(bars_arrow)) != early["bars"]["bars_digest"]:
        raise ReplayMismatch("bars_digest", "the re-read bars are not the bars the request binds")
    return ReplayInputs(attempt, bundle_root, environment_root, request_json, bars_arrow)


def launch_replay(
    inputs: ReplayInputs, *, invocations_root: Path, run: Runner = run_contained,
) -> bytes:
    """Launch one child over ``inputs`` exactly as the dispatcher does; return its stdout."""
    problem = unsupported_content(inputs.bundle_root)
    if problem is not None:
        raise ReplayMismatch("frozen_content_unsupported", problem)
    ended = launch_child(
        interpreter=inputs.environment_root / "bin" / "python",
        bundle_root=inputs.bundle_root,
        environment_root=inputs.environment_root,
        request_json=inputs.request_json,
        bars_arrow=inputs.bars_arrow,
        invocations_root=invocations_root,
        run=run,
    )
    code = process_failure(ended)
    if code is not None:
        raise ReplayMismatch(code, process_diagnostic(ended))
    return ended.stdout


def replay_attempt(
    conn: sqlite3.Connection, data_dir: Path, invocation_id: int, *,
    run: Runner = run_contained,
) -> bytes:
    """Replay the recorded attempt ``invocation_id``; return the child's stdout bytes, whose
    SHA-256 a deterministic replay makes equal to the row's ``result_sha256``."""
    attempt = frozen_invocation(conn, invocation_id)
    if attempt is None:
        raise ReplayMismatch("no_attempt", f"no frozen invocation {invocation_id}")
    inputs = prepare_replay(conn, data_dir, attempt)
    return launch_replay(
        inputs, invocations_root=Path(data_dir) / "frozen" / "invocations", run=run)
