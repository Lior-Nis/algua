"""Pure value types for the model-artifact seam (issue #376).

These are the ONLY model types that cross into strategies/base (which must stay off the model
I/O layer). No I/O lives here — the filesystem-backed registry is `algua.models.registry`, which
produces `ModelVersion`s and hands the strategy loader a `ModelHandle`.

The seam is deserialization-agnostic: a `ModelHandle` carries the raw artifact `bytes` plus the
resolved `ModelVersion`; the strategy's `signal(view, params, model)` deserializes those bytes
in-memory (pure — no file I/O in the strategy module). No torch/sklearn/onnx dependency is added.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def compute_provenance_digest(
    *,
    digest: str,
    training_snapshot_id: str | None,
    training_as_of: str,
    code_hash: str | None,
    hyperparameters: dict[str, Any],
    seed: int | None,
    eval_report: dict[str, Any],
) -> str:
    """Stable digest committing to the FULL training provenance of a model version, not just its
    artifact bytes. Pinned in `ModelRef` and recomputed on every registry read, so rewriting ANY
    provenance field (snapshot id, code_hash, hyperparameters, seed, eval report) — not only the
    artifact — is detected as a history rewrite (issue #376). Serialized canonically
    (sorted keys, allow_nan=False) so a non-finite value cannot yield a non-canonical digest."""
    payload = json.dumps(
        {
            "digest": digest,
            "training_snapshot_id": training_snapshot_id,
            "training_as_of": training_as_of,
            "code_hash": code_hash,
            "hyperparameters": hyperparameters,
            "seed": seed,
            "eval_report": eval_report,
        },
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def normalize_as_of(value: str) -> pd.Timestamp:
    """Normalize a training-as-of / bar timestamp to a comparable UTC instant. A bare date means
    start-of-day (00:00:00 UTC). Fails closed (raises) on an unparseable value so a malformed
    training_as_of can never silently pass the PIT guard (issue #376)."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class ModelVersion:
    """One immutable registered model version — the registry's system-of-record row, resolved
    together with the on-disk artifact path. `provenance_digest` commits to the full training
    provenance (see `compute_provenance_digest`)."""

    name: str
    version: int
    digest: str
    created_at: str
    training_snapshot_id: str | None
    training_as_of: str
    code_hash: str | None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    eval_report: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    provenance_digest: str = ""

    def as_ref(self) -> ModelRef:
        """The PINNED config reference for this version (name/version/digest/training_as_of/
        provenance_digest) — what a strategy's CONFIG carries to bind exactly this model."""
        return ModelRef(
            name=self.name,
            version=self.version,
            digest=self.digest,
            training_as_of=self.training_as_of,
            provenance_digest=self.provenance_digest,
        )


@dataclass(frozen=True)
class ModelRef:
    """A strategy CONFIG's PINNED reference to a model version. NEVER 'latest' — an explicit
    integer version plus the artifact `digest`, the `training_as_of` PIT cutoff, and the
    `provenance_digest` the config was VALIDATED against. The loader fails closed unless the
    resolved registry version matches all three, so a strategy can never silently bind a different
    model (or a model with rewritten training provenance) than the one it was validated with."""

    name: str
    version: int
    digest: str
    training_as_of: str
    provenance_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "training_as_of": self.training_as_of,
            "provenance_digest": self.provenance_digest,
        }


@dataclass(frozen=True)
class ModelHandle:
    """What the strategy loader injects alongside `params` into a needs_model `signal`. Carries the
    resolved immutable `ModelVersion` and the raw artifact `bytes`; the strategy deserializes the
    bytes in-memory (pure). The model is fixed for the whole run, so binding it once at load time
    is point-in-time safe (no per-bar model refetch, no future-model leakage)."""

    version: ModelVersion
    artifact_bytes: bytes
