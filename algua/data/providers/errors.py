from __future__ import annotations


class ProviderError(ValueError):
    """A data provider failed to deliver bars.

    Subclasses ``ValueError`` so the CLI's ``@json_errors`` decorator renders it as
    ``{"ok": false, "error": ...}`` on stdout instead of letting a raw transport
    traceback (e.g. ``requests.HTTPError``, ``ImportError``) escape the JSON contract.
    """
