from __future__ import annotations

import hashlib
import inspect

from algua.registry.repository import StrategyRepository
from algua.strategies.base import config_hash
from algua.strategies.loader import load_strategy


def compute_artifact_hashes(name: str) -> tuple[str, str]:
    """Recompute ``(code_hash, config_hash)`` from the strategy's *actual* source + resolved
    config. This is the single function both the ``approve`` and the ``transition --to live``
    paths call, so the live gate pins the real artifact: a constant or caller-supplied hash can
    no longer satisfy it, because both sides derive the hash from the loaded module itself.
    """
    loaded = load_strategy(name)
    module = inspect.getmodule(loaded.fn)
    source = inspect.getsource(module) if module is not None else ""
    code_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
    return code_hash, config_hash(loaded)


def record_approval(repo: StrategyRepository, name: str, approved_by: str) -> int:
    """Record a human approval. The approved hashes are computed from the live strategy source
    and config, never supplied by the caller, so the approval binds to the code it approves."""
    _require_non_empty("approved_by", approved_by)
    rec = repo.get(name)
    code_hash, config_hash_ = compute_artifact_hashes(name)
    return repo.record_approval(rec.id, code_hash, config_hash_, approved_by)


def has_valid_approval(
    repo: StrategyRepository, strategy_id: int, code_hash: str, config_hash: str
) -> bool:
    return repo.has_valid_approval(strategy_id, code_hash, config_hash)


def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
