import numpy as np

from examples import aak_siso_schmidt_pair_demo as demo
from examples.finite_nehari_rational_bridge import synthetic_anticausal_tail


def test_schmidt_pair_residuals_are_small():
    rows = cols = 20
    tail = synthetic_anticausal_tail(rows + cols - 1)
    result = demo.schmidt_pair_diagnostics(tail, rows=rows, cols=cols, rank=2)

    assert result["critical_sigma"] > 0.0
    assert result["left_residual"] < 1e-10
    assert result["right_residual"] < 1e-10


def test_rank_svd_error_matches_first_neglected_singular_value():
    rows = cols = 18
    rank = 3
    tail = synthetic_anticausal_tail(rows + cols - 1)
    result = demo.schmidt_pair_diagnostics(tail, rows=rows, cols=cols, rank=rank)

    singular_values = np.asarray(result["singular_values"], dtype=float)
    np.testing.assert_allclose(
        result["rank_svd_error"],
        singular_values[rank],
        rtol=1e-10,
        atol=1e-10,
    )


def test_hankel_from_tail_has_expected_anti_diagonals():
    tail = np.arange(1.0, 10.0)
    h = demo.hankel_from_tail(tail, rows=3, cols=4)
    expected = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]
    )
    np.testing.assert_allclose(h, expected)
