from __future__ import annotations

import numpy as np

from examples.coupled_mimo_lattice_filter import (
    apply_matrix_lattice_finite_adjoint_time_domain,
    apply_matrix_lattice_streaming,
    coupled_complex_signal,
    make_coupled_lattice,
    normalized_covariance_magnitude,
)


def test_coupled_mimo_lattice_streaming_is_energy_preserving_with_tail() -> None:
    filt = make_coupled_lattice(channels=3, order=3, seed=22)
    x = coupled_complex_signal(samples=256, channels=3, seed=23)
    tail = 512

    y = apply_matrix_lattice_streaming(x, filt, tail=tail)
    x_hat = apply_matrix_lattice_finite_adjoint_time_domain(
        y, filt, tail=tail, output_length=x.shape[0]
    )

    energy_error = abs(np.vdot(y, y).real - np.vdot(x, x).real) / np.vdot(x, x).real
    reconstruction_error = np.linalg.norm(x_hat - x) / np.linalg.norm(x)

    assert y.shape == (x.shape[0] + tail, x.shape[1])
    assert energy_error < 1e-10
    assert reconstruction_error < 1e-8


def test_normalized_covariance_magnitude_has_unit_diagonal() -> None:
    x = coupled_complex_signal(samples=128, channels=3, seed=24)
    cov = normalized_covariance_magnitude(x)
    assert cov.shape == (3, 3)
    assert np.allclose(np.diag(cov), 1.0)
    assert np.all(cov >= 0.0)
    assert np.all(cov <= 1.0 + 1e-12)
