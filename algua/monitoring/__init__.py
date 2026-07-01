"""Leading-indicator drift monitoring (issue #343).

An ADVISORY layer that flags alpha decay from SIGNAL-side indicators. It gates nothing,
persists nothing, and never touches the live/paper order path or the promotion/forward gates
(those judge the LAGGING realized-return layer). See :mod:`algua.monitoring.drift`.
"""
