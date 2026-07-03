"""Shadow-mode / champion-challenger evaluation seam (issue #392).

A challenger strategy is replayed on the SAME point-in-time data a live champion sees, its would-be
decisions and hypothetical (paper-accounted) P&L are recorded, and an operator can compare
champion-vs-challenger BEFORE any promotion. This lane is strictly ADVISORY: it uses only an
in-process SimBroker, never submits a real order, never consumes a live/paper allocation or the
single-use holdout, and never mints a gate token or drives a lifecycle transition.

The package is structurally walled by import-linter: it imports NONE of the live lane, no real
broker, no ledger/reservation/reconcile, no kill-switch/global-halt, and no allocation/live-gate
module — so no code path can wire a shadow decision into a real order or capital allocation.
"""
