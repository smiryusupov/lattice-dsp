import numpy as np
import pytest

import lattice_dsp as ld


def exact_tail(n_terms=95):
    poles = np.array([-0.42, 0.18, 0.76], dtype=float)
    weights = np.array([1.25, -0.7, 0.4], dtype=float)
    n = np.arange(n_terms, dtype=float)
    return sum(w * p**n for w, p in zip(weights, poles, strict=True)), poles


def test_finite_hankel_matrix_from_tail_places_antidiagonals():
    tail = np.arange(7, dtype=float)
    h = ld.finite_hankel_matrix_from_tail(tail, rows=3, cols=4)
    expected = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
        ]
    )
    np.testing.assert_allclose(h, expected)


def test_finite_aak_certificate_reports_schmidt_pair_identities():
    tail, _ = exact_tail()
    cert = ld.finite_aak_siso_certificate(tail, rank=3, rows=48, cols=48)

    assert cert["method"] == "finite_section_siso_aak_nehari_certificate"
    assert cert["rank"] == 3
    assert cert["sigma_next"] < 1e-6
    assert cert["rank_svd_error"] <= cert["sigma_next"] + 1e-8
    assert cert["schmidt_left_residual"] < 1e-10
    assert cert["schmidt_right_residual"] < 1e-10


def test_finite_aak_certificate_recovers_exact_rank_three_candidate():
    tail, true_poles = exact_tail()
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1e-8,
        max_rational_error=1e-8,
        max_pole_radius=0.99,
    )
    cert = ld.finite_aak_siso_certificate(tail, rank=3, rows=48, cols=48, criteria=criteria)
    candidate = cert["candidate"]

    assert cert["accepted"] is True
    assert candidate["hankelized_tail_error"] < 1e-10
    assert candidate["rational_error"] < 1e-10
    np.testing.assert_allclose(
        np.sort(np.real(candidate["poles"])),
        np.sort(true_poles),
        atol=1e-8,
    )


def test_finite_aak_certificate_rejects_rank_without_sigma_next():
    tail = np.exp(-0.1 * np.arange(15))
    with pytest.raises(ValueError, match="sigma_next"):
        ld.finite_aak_siso_certificate(tail, rank=4, rows=4, cols=4)
