from __future__ import annotations

import numpy as np
import pytest

import lattice_dsp as ld
from examples.mimo_coupled_model_reduction import coupled_state_space


def test_mimo_state_space_frequency_response_direct_term_and_shape() -> None:
    A = np.array([[0.25]])
    B = np.array([[1.0, -0.5]])
    C = np.array([[0.7], [-0.2]])
    D = np.array([[0.1, 0.2], [-0.3, 0.4]])
    omega = np.array([0.0, np.pi / 3])

    response = ld.mimo_state_space_frequency_response(A, B, C, D, omega)
    assert response.shape == (2, 2, 2)
    zinv = np.exp(-1j * omega[0])
    expected0 = D + C @ (zinv * np.linalg.solve(np.eye(1) - zinv * A, B))
    np.testing.assert_allclose(response[0], expected0)


def test_polar_factor_response_is_unitary() -> None:
    rng = np.random.default_rng(101)
    response = rng.normal(size=(8, 3, 3)) + 1j * rng.normal(size=(8, 3, 3))
    polar = ld.polar_factor_response(response)
    eye = np.eye(3)
    assert polar.shape == response.shape
    assert max(np.linalg.norm(u.conj().T @ u - eye) for u in polar) < 1e-12


def test_experimental_state_space_to_matrix_lattice_returns_stable_fit() -> None:
    A, B, C, D = coupled_state_space(order=8, outputs=3, inputs=3, seed=102)
    markov = ld.mimo_state_space_markov_response(A, B, C, D, 96)
    reduced = ld.finite_hankel_reduce_mimo(markov, reduced_order=4, block_rows=12, block_cols=12)

    result = ld.experimental_mimo_state_space_to_matrix_lattice(
        reduced["A"],
        reduced["B"],
        reduced["C"],
        reduced["D"],
        order=4,
        n_markov=96,
        n_freq=64,
        candidate_gains=[0.2, 0.45, 0.7],
        fit_static_gains=True,
        static_gain_iterations=3,
        n_threads=1,
    )

    lattice = result["lattice"]
    assert isinstance(lattice, ld.MatrixLatticeAllPass)
    assert lattice.order == 4
    assert lattice.dimension == 3
    assert result["selected_gain"] in {0.2, 0.45, 0.7}
    assert np.isfinite(result["polar_factor_relative_error"])
    assert result["unitarity_error"] < 1e-9
    assert result["max_reflection_singular_value"] < 1.0
    assert np.asarray(result["candidate_errors"]).shape == (3,)
    assert np.isfinite(result["state_response_relative_error"])
    assert np.isfinite(result["static_gain_relative_error"])
    assert result["diagnostic_classification"] in {
        "good_allpass_polar_fit",
        "mostly_static_gain_or_nonunitary_mismatch",
        "poor_lattice_scaffold_fit",
    }


def test_experimental_fit_rejects_rectangular_transfer_matrix() -> None:
    A, B, C, D = coupled_state_space(order=4, outputs=2, inputs=3, seed=103)
    with pytest.raises(ValueError, match="square MIMO"):
        ld.experimental_mimo_state_space_to_matrix_lattice(
            A, B, C, D, order=2, n_markov=32, n_freq=32
        )
