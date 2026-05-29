from __future__ import annotations

import numpy as np

import lattice_dsp as ld
from examples.mimo_coupled_model_reduction import coupled_state_space
from examples.mimo_hankel_to_matrix_lattice_bridge import (
    matrix_lattice_scaffold_from_markov,
    polar_factor_per_frequency,
    response_relative_error,
    state_space_frequency_response,
)


def test_polar_factor_per_frequency_is_unitary() -> None:
    rng = np.random.default_rng(40)
    response = rng.normal(size=(12, 3, 3)) + 1j * rng.normal(size=(12, 3, 3))
    polar = polar_factor_per_frequency(response)
    eye = np.eye(3)
    assert polar.shape == response.shape
    assert max(np.linalg.norm(u.conj().T @ u - eye) for u in polar) < 1e-12


def test_markov_scaffold_is_stable_and_unitary() -> None:
    A, B, C, D = coupled_state_space(order=8, outputs=3, inputs=3, seed=41)
    markov = ld.mimo_state_space_markov_response(A, B, C, D, 80)
    reduced = ld.finite_hankel_reduce_mimo(markov, reduced_order=4, block_rows=12, block_cols=12)
    reduced_markov = ld.mimo_state_space_markov_response(
        reduced["A"], reduced["B"], reduced["C"], reduced["D"], 80
    )

    scaffold = matrix_lattice_scaffold_from_markov(reduced_markov, order=4)
    omega = np.linspace(0.0, np.pi, 64)
    h_reduced = state_space_frequency_response(
        reduced["A"], reduced["B"], reduced["C"], reduced["D"], omega
    )
    polar = polar_factor_per_frequency(h_reduced)
    h_scaffold = scaffold.frequency_response(omega, n_threads=1)

    assert scaffold.dimension == 3
    assert scaffold.order == 4
    assert scaffold.max_reflection_singular_value() < 1.0
    assert scaffold.unitarity_error(omega) < 1e-9
    assert np.isfinite(response_relative_error(polar, h_scaffold))
