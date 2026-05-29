from __future__ import annotations

import numpy as np

import lattice_dsp as ld


def _calibration_lattice() -> ld.MatrixLatticeAllPass:
    rng = np.random.default_rng(909)
    reflections = []
    for scale in (0.25, 0.18, 0.10):
        raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        reflections.append(ld.contractive_matrix_from_raw(scale * raw))
    residue = ld.unitary_polar_factor(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    return ld.MatrixLatticeAllPass(reflections, residue=residue)


def test_fit_static_matrix_gains_recovers_gain_wrapped_lattice_response() -> None:
    lattice = _calibration_lattice()
    omega = np.linspace(0.0, np.pi, 96)
    response = lattice.frequency_response(omega, n_threads=1)
    left = np.diag([1.7, 0.8, 0.4]).astype(np.complex128)
    right = ld.unitary_polar_factor(
        np.array([[1.0, 0.3, -0.2], [0.1, -0.7, 0.5], [0.4, 0.2, 0.9]], dtype=np.complex128)
    )
    target = np.asarray([left @ h @ right for h in response], dtype=np.complex128)

    fit = ld.fit_static_matrix_gains(target, response, mode="both", n_iter=40)

    assert float(fit["raw_relative_error"]) > 1e-2
    assert float(fit["compensated_relative_error"]) < 5e-5
    assert float(fit["raw_relative_error"]) / float(fit["compensated_relative_error"]) > 1e4
    assert float(fit["left_gain_condition"]) > 1.0
    assert float(fit["right_gain_condition"]) >= 1.0
    assert np.asarray(fit["compensated_response"]).shape == target.shape


def test_fit_static_matrix_gains_left_mode_handles_exact_left_gain() -> None:
    lattice = _calibration_lattice()
    omega = np.linspace(0.0, np.pi, 48)
    response = lattice.frequency_response(omega, n_threads=1)
    left = np.diag([1.4, 0.9, 0.6]).astype(np.complex128)
    target = np.asarray([left @ h for h in response], dtype=np.complex128)

    fit = ld.fit_static_matrix_gains(target, response, mode="left")

    assert float(fit["compensated_relative_error"]) < 1e-5
    np.testing.assert_allclose(np.asarray(fit["right_gain"]), np.eye(3), atol=1e-14)


def test_experimental_solver_reports_static_gain_diagnostics() -> None:
    rng = np.random.default_rng(910)
    a = np.diag([0.2, 0.45, 0.7, -0.35])
    b = rng.normal(size=(4, 2))
    c = rng.normal(size=(2, 4))
    d = np.array([[1.0, 0.2], [-0.1, 0.8]])

    result = ld.experimental_mimo_state_space_to_matrix_lattice(
        a,
        b,
        c,
        d,
        order=2,
        n_markov=64,
        n_freq=48,
        candidate_gains=[0.1, 0.4],
        fit_static_gains=True,
        static_gain_iterations=3,
        n_threads=1,
    )

    assert result["fit_static_gains"] is True
    assert np.isfinite(result["state_response_relative_error"])
    assert np.isfinite(result["static_gain_relative_error"])
    assert (
        float(result["static_gain_relative_error"])
        <= float(result["state_response_relative_error"]) + 1e-12
    )
    assert result["diagnostic_classification"] in {
        "good_allpass_polar_fit",
        "mostly_static_gain_or_nonunitary_mismatch",
        "poor_lattice_scaffold_fit",
    }
