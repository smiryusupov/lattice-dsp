import numpy as np
import pytest

import lattice_dsp as ld
from examples import nehari_aak_siso_toy as toy


def test_finite_nehari_unconstrained_error_matches_next_singular_value():
    rows = cols = 14
    gamma = toy.anticausal_tail(rows + cols - 1)

    result = ld.finite_nehari_approximate_tail(gamma.tolist(), rank=3, rows=rows, cols=cols)

    np.testing.assert_allclose(
        result["unconstrained_hankel_error"],
        result["sigma_next"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert len(result["approximated_tail"]) == rows + cols - 1
    assert result["method"] == "finite_nehari_hankelized_svd"


def test_finite_nehari_reports_svd_bound_and_hankelized_error():
    rows = cols = 12
    gamma = toy.anticausal_tail(rows + cols - 1)
    result = ld.finite_nehari_approximate_tail(gamma.tolist(), rank=2, rows=rows, cols=cols)

    # sigma_next is the Eckart--Young error for the unconstrained rank-r
    # matrix approximation. After anti-diagonal averaging, the approximation is
    # Hankel-structured again but is no longer guaranteed to have rank r, so
    # its error is diagnostic rather than lower-bounded by sigma_next.
    assert result["sigma_next"] > 0.0
    assert result["unconstrained_hankel_error"] >= result["sigma_next"] - 1e-10
    assert result["unconstrained_hankel_error"] <= result["sigma_next"] + 1e-8
    assert result["hankelized_hankel_error"] >= 0.0
    assert result["hankelized_hankel_error"] < result["unconstrained_hankel_error"] * 2.0


def test_finite_nehari_rejects_invalid_rank_and_short_tail():
    gamma = [1.0, 0.5, 0.25]

    with pytest.raises(ValueError):
        ld.finite_nehari_approximate_tail(gamma, rank=1, rows=3, cols=3)

    with pytest.raises(ValueError):
        ld.finite_nehari_approximate_tail([1.0] * 8, rank=5, rows=4, cols=4)
