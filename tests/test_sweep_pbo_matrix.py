"""The trials x windows OOS-Sharpe matrix returned by sweep_with_matrix (#467).

Covers: shape, GENERATED-COMBO-ORDER alignment (NOT ranked order), bit-identical with/without
compute_holdout, and that the public sweep() exposes no way to reach the matrix.
"""

from algua.backtest._sample import SyntheticProvider
from algua.backtest.sweep import _combos, _override, sweep, sweep_with_matrix
from algua.backtest.walkforward import walk_forward
from tests.test_sweep import END, START, _momentum


def test_matrix_shape_excludes_holdout():
    windows = 4
    grid = {"lookback": [20, 40]}
    res, matrix = sweep_with_matrix(
        _momentum(), SyntheticProvider(seed=3), START, END,
        grid=grid, windows=windows, holdout_frac=0.2,
    )
    assert isinstance(matrix, list)
    assert len(matrix) == res.n_combos == 2
    for row in matrix:
        # Exactly `windows` OOS columns, NOT windows+1: the holdout segment is excluded by
        # construction (window_metrics never contains the holdout).
        assert len(row) == windows
        assert all(isinstance(x, float) for x in row)


def test_matrix_rows_are_generated_combo_order_not_ranked():
    # A grid whose RANKED order provably differs from generation order (seed 3 reorders it): the
    # matrix must stay in GENERATION order (row i == combo i), never rank order.
    windows = 4
    grid_values = [10, 20, 40, 60, 80]
    seed = 0
    res, matrix = sweep_with_matrix(
        _momentum(), SyntheticProvider(seed=seed), START, END,
        grid={"lookback": grid_values}, windows=windows, holdout_frac=0.2,
    )
    gen_order = [c["lookback"] for c in _combos({"lookback": grid_values})]
    ranked_order = [r["params"]["lookback"] for r in res.ranked]
    # Guard the test's own premise: ranking DID reorder, so generation order is a real distinction.
    assert ranked_order != gen_order, "fixture no longer reorders — pick a reordering seed/grid"
    # Row i is combo i's per-window Sharpes (generation order), reproduced by an INDEPENDENT
    # walk_forward on that combo — never the ranked-best combo's row.
    for lookback, row in zip(gen_order, matrix, strict=True):
        ov = _override(_momentum(), {"lookback": lookback})
        wf = walk_forward(ov, SyntheticProvider(seed=seed), START, END,
                          windows=windows, holdout_frac=0.2, compute_holdout=False)
        assert row == [w["sharpe"] for w in wf.window_metrics]


def test_matrix_bit_identical_with_and_without_holdout():
    windows = 4
    grid = {"lookback": [20, 40, 60]}
    args = dict(grid=grid, windows=windows, holdout_frac=0.2)
    _, m_with = sweep_with_matrix(
        _momentum(), SyntheticProvider(seed=3), START, END, compute_holdout=True, **args)
    _, m_without = sweep_with_matrix(
        _momentum(), SyntheticProvider(seed=3), START, END, compute_holdout=False, **args)
    assert m_with == m_without


def test_public_sweep_returns_result_with_no_matrix_attribute():
    res = sweep(
        _momentum(), SyntheticProvider(seed=3), START, END,
        grid={"lookback": [20, 40]}, windows=4, holdout_frac=0.2,
    )
    # No matrix rides on the SweepResult — not under any plausible attribute name.
    for attr in ("matrix", "trial_window_sharpes", "window_sharpes", "trial_matrix"):
        assert not hasattr(res, attr), f"SweepResult unexpectedly exposes {attr!r}"
