"""Local-file bar importers and their construction registry.

These are the `BarImporter` seam (`algua.data.contracts`): `import_bars(ImportRequest) ->
Iterator[ProviderBars]`, used by `algua data import-bars` to normalize vendor files into one
consolidated bar-schema snapshot. Distinct from the network-fetch `BarProvider` seam
(`algua/data/providers/`) — the source here is local files, not an API.

Construction lives here (not the CLI) so adding a vendor is open-for-extension: register a
name->factory and `get_importer` picks it up — no if/elif ladder to edit.
"""
from __future__ import annotations

from collections.abc import Callable

from algua.data.contracts import BarImporter

ImporterFactory = Callable[[], BarImporter]


_REGISTRY: dict[str, ImporterFactory] = {}


def register_importer(name: str, factory: ImporterFactory) -> None:
    """Register a name->factory so `get_importer(name)` can build it."""
    _REGISTRY[name] = factory


def get_importer(name: str) -> BarImporter:
    """Construct a `BarImporter` by name. Raises ValueError for an unknown name."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unsupported bar importer: {name}") from None
    return factory()
