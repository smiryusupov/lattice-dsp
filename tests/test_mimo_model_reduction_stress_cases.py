from __future__ import annotations

import numpy as np

from examples import mimo_model_reduction_stress_cases as stress


def test_one_over_f_matrix_tail_shape_and_decay():
    markov = stress.one_over_f_matrix_tail(12)
    assert markov.shape == (12, 3, 3)
    np.testing.assert_allclose(markov[0], np.ones((3, 3)))
    assert np.all(markov[-1] < markov[1])


def test_random_rational_markov_shape_and_finiteness():
    markov = stress.random_rational_markov(n_terms=40, channels=4, n_basis=7, seed=10)
    assert markov.shape == (40, 4, 4)
    assert np.all(np.isfinite(markov))
    assert np.linalg.norm(markov[-1]) < np.linalg.norm(markov[0])


def test_tangential_schur_diagnostic_returns_psd_scaled_pick():
    markov = 0.05 * stress.random_rational_markov(n_terms=35, channels=3, n_basis=5, seed=11)
    diag = stress.finite_tangential_schur_diagnostic(markov, n_points=5, seed=12)
    assert diag["scale"] >= 1.0
    assert diag["min_pick_eigenvalue"] >= -1e-10
    assert np.asarray(diag["pick_eigenvalues"]).ndim == 1


def test_small_mimo_reduction_beats_short_fir_on_rational_tail():
    markov = stress.random_rational_markov(
        n_terms=80, channels=2, n_basis=5, seed=13, pole_max=0.86
    )
    approx, reduction, timings = stress.reduce_mimo_markov(
        markov, order=4, block_rows=8, block_cols=8
    )
    assert approx is not None
    assert reduction["A"].shape == (4, 4)
    assert timings["total_seconds"] >= 0.0
    finite_error = stress.relative_h2_error(markov, approx)
    fir_error = stress.relative_h2_error(markov, stress.truncated_fir(markov, order=4))
    assert finite_error < fir_error


def test_tangential_sample_residual_is_zero_for_identical_markov():
    markov = stress.random_rational_markov(n_terms=30, channels=2, n_basis=4, seed=14)
    diag = stress.finite_tangential_schur_diagnostic(markov, n_points=4, seed=15)
    assert stress.tangential_sample_residual(markov, markov.copy(), diag) < 1e-14


def test_relative_hankel_norm_error_is_zero_for_identical_markov():
    markov = stress.random_rational_markov(n_terms=30, channels=2, n_basis=4, seed=16)
    assert (
        stress.relative_hankel_norm_error(markov, markov.copy(), block_rows=6, block_cols=6) < 1e-14
    )


def test_stress_case_orders_include_large_targets():
    cases = {case.slug: case for case in stress.build_cases()}
    assert 70 in cases["one_over_f_3x3"].reduction_orders
    assert 70 in cases["random_rational_10x10"].reduction_orders
    assert 400 in cases["ill_conditioned_2x2"].reduction_orders
    assert cases["ill_conditioned_2x2"].block_rows * 2 >= 400


def test_finite_hankel_tail_error_uses_next_singular_value():
    singular = np.array([10.0, 5.0, 1.0, 0.1])
    assert stress.finite_hankel_tail_error(singular, 0) == 1.0
    assert stress.finite_hankel_tail_error(singular, 2) == 0.1
    assert stress.finite_hankel_tail_error(singular, 4) == 0.0


def test_large_ill_conditioned_case_uses_shared_lapack_and_reports_hankel_tail():
    case = [c for c in stress.build_cases() if c.slug == "ill_conditioned_2x2"][0]
    # Keep the smoke test light by using the first two orders only.
    small_case = stress.StressCase(
        name=case.name,
        slug=case.slug,
        markov=case.markov[:500],
        reduction_orders=(40, 100),
        block_rows=80,
        block_cols=80,
        description=case.description,
        condition_hint=case.condition_hint,
    )
    rows, hsv, _ = stress.evaluate_case(small_case, seed=17)
    finite = [row for row in rows if row["method"] == "finite_block_hankel_mimo"]
    assert len(finite) == 2
    assert all(np.isfinite(row["relative_hankel_norm_error"]) for row in finite)
    assert finite[-1]["relative_hankel_norm_error"] <= finite[0]["relative_hankel_norm_error"]
    assert hsv.size >= 100
