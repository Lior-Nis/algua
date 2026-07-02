"""Model-artifact registry: the system-of-record for trained model versions (issue #376).

Distinct from the strategy/code registry. A leaf filesystem-I/O layer — it may import
`algua.contracts` but nothing higher (cli/registry/backtest/live/execution/research/data), which
import-linter enforces. The strategy loader and the MLflow tracker are the only resolve/ingress
callers.
"""

from algua.models.registry import (
    ModelRegistryError,
    get_version,
    get_version_with_bytes,
    list_versions,
    load_artifact_bytes,
    register,
)

__all__ = [
    "ModelRegistryError",
    "get_version",
    "get_version_with_bytes",
    "list_versions",
    "load_artifact_bytes",
    "register",
]
