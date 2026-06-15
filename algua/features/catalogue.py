# algua/features/catalogue.py
from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import algua.features as _features_pkg
from algua.contracts.idea import DataCapability


class FactorKind(StrEnum):
    """Controlled, deliberately minimal factor categories (OTHER is the escape hatch). Extend as
    real factors demand, not speculatively."""

    MOMENTUM = "momentum"
    REVERSION = "reversion"
    VALUE = "value"
    SENTIMENT = "sentiment"
    VOLATILITY = "volatility"
    QUALITY = "quality"
    OTHER = "other"


class FactorNotFound(LookupError):
    pass


@dataclass(frozen=True)
class FactorSpec:
    """Catalogued metadata for one pure factor. Collection fields are tuples so a registered spec
    cannot be mutated in place (a frozen dataclass does not stop list mutation)."""

    name: str
    summary: str
    kind: FactorKind
    tags: tuple[str, ...]
    data_needs: tuple[DataCapability, ...]
    import_path: str
    module: str
    signature: str
    doc: str | None
    standalone: bool = False


_SPEC_ATTR = "__factor_spec__"


def _assert_signal_shaped(fn: Callable[..., Any], name: str) -> None:
    """A standalone-evaluable factor must be signal-shaped: exactly two POSITIONAL_OR_KEYWORD
    params (view, params) and no *args/**kwargs. Structural arity check only — it cannot verify
    semantics (it cannot tell (view, params) from (prices, lookback)); marking a factor standalone
    is a deliberate author assertion. Fails closed on the obvious mistakes (transforms, varargs)."""
    params = list(inspect.signature(fn).parameters.values())
    ok = len(params) == 2 and all(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params
    )
    if not ok:
        raise ValueError(
            f"factor {name!r} declares standalone=True but is not signal-shaped "
            f"(view, params); got signature {inspect.signature(fn)}"
        )


def _first_nonempty_line(doc: str | None) -> str | None:
    if not doc:
        return None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def factor(
    *,
    name: str | None = None,
    summary: str | None = None,
    tags: Iterable[str] = (),
    kind: FactorKind = FactorKind.OTHER,
    data_needs: Iterable[DataCapability] | None = None,
    standalone: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Annotate a pure factor function with discoverability metadata. PURE: it stamps a FactorSpec
    on the function as ``__factor_spec__`` and returns the function UNCHANGED (no wrapper) so
    call semantics, ``inspect.getsource`` and the live-gate ``code_hash`` see the real function.
    It mutates no module global at import time — discovery (``load_all_factors``) scans for the
    stamp. ``data_needs`` states the factor's INPUT requirement, not current platform availability.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = name or fn.__name__
        if standalone:
            _assert_signal_shaped(fn, resolved_name)
        doc = inspect.getdoc(fn)
        resolved_summary = summary or _first_nonempty_line(doc)
        if not resolved_summary:
            raise ValueError(
                f"factor {resolved_name!r} needs a summary (pass summary= or add a docstring)"
            )
        spec = FactorSpec(
            name=resolved_name,
            summary=resolved_summary,
            kind=kind,
            tags=tuple(tags),
            data_needs=tuple(data_needs) if data_needs is not None else (DataCapability.OHLCV,),
            import_path=f"{fn.__module__}:{fn.__qualname__}",
            module=fn.__module__,
            signature=str(inspect.signature(fn)),
            doc=doc,
            standalone=standalone,
        )
        setattr(fn, _SPEC_ATTR, spec)
        return fn

    return decorate


_REGISTRY: dict[str, FactorSpec] = {}
_loaded = False


def load_all_factors() -> dict[str, FactorSpec]:
    """Discover every catalogued factor by scanning ``algua.features`` modules for the
    ``__factor_spec__`` stamp. Transactional: builds a fresh dict and binds it to ``_REGISTRY`` in
    ONE assignment only after every module imports cleanly (a failing import raises before the
    bind, leaving the prior registry intact — no half-populated global is ever observable). Skips
    ``_``-prefixed modules. Accepts a stamped function ONLY at its defining module
    (``fn.__module__ == module.__name__``) so a re-export does not double-register. Idempotent:
    re-scans the import-cached modules each call. Fails closed on a duplicate ``name``."""
    global _REGISTRY, _loaded
    fresh: dict[str, FactorSpec] = {}
    for mod_info in pkgutil.iter_modules(_features_pkg.__path__):
        if mod_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{_features_pkg.__name__}.{mod_info.name}")
        for value in vars(module).values():
            spec = getattr(value, _SPEC_ATTR, None)
            if spec is None or getattr(value, "__module__", None) != module.__name__:
                continue  # not a factor, or a re-export not defined here
            if spec.name in fresh:
                raise ValueError(
                    f"duplicate factor name {spec.name!r}: {fresh[spec.name].import_path} "
                    f"and {spec.import_path} (pass name= to disambiguate)"
                )
            fresh[spec.name] = spec
    _REGISTRY = fresh
    _loaded = True
    return _REGISTRY


def _reset_registry() -> None:
    """Test hook: clear discovered state so the next read re-discovers."""
    global _REGISTRY, _loaded
    _REGISTRY = {}
    _loaded = False


def _ensure_loaded() -> None:
    if not _loaded:
        load_all_factors()


def get_factor(name: str) -> FactorSpec:
    _ensure_loaded()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise FactorNotFound(name) from None


def all_factors() -> list[FactorSpec]:
    _ensure_loaded()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def load_factor_callable(spec: FactorSpec) -> Callable[..., Any]:
    """Resolve a FactorSpec back to its function object via ``import_path`` ("module:qualname").
    The catalogue scan already imported the module, so this is import-safe. Fails closed
    (``FactorNotFound``) if the resolved object is not the matching stamped factor — guarding
    against a spec whose import_path drifted off its function."""
    module_name, _, qualname = spec.import_path.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise FactorNotFound(spec.name) from None
    resolved = getattr(obj, _SPEC_ATTR, None)
    if resolved is None or resolved.name != spec.name:
        raise FactorNotFound(spec.name)
    return obj


def filter_factors(
    *, tag: str | None = None, kind: FactorKind | None = None
) -> list[FactorSpec]:
    """Catalogue factors filtered by tag and/or kind (AND-combined)."""
    out = all_factors()
    if tag is not None:
        out = [f for f in out if tag in f.tags]
    if kind is not None:
        out = [f for f in out if f.kind is kind]
    return out
