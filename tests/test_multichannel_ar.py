from __future__ import annotations

import numpy as np

import lattice_dsp as ld


def _simulate_var(
    coefficients: list[np.ndarray], samples: int = 50000, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = len(coefficients)
    channels = coefficients[0].shape[0]
    x = np.zeros((samples + 512, channels))
    noise = rng.normal(size=x.shape)
    for n in range(order, x.shape[0]):
        y = noise[n].copy()
        for lag, a_lag in enumerate(coefficients, start=1):
            y -= a_lag @ x[n - lag]
        x[n] = y
    return x[512:]


def test_multichannel_autocorrelation_shape_and_hermitian_zero_lag() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1024, 3))
    r = ld.multichannel_autocorrelation(x, order=5)
    assert r.shape == (6, 3, 3)
    assert np.allclose(r[0], r[0].conj().T)


def test_block_toeplitz_shape() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2048, 2))
    r = ld.multichannel_autocorrelation(x, order=4)
    t = ld.block_toeplitz_from_autocorrelation(r, order=4)
    assert t.shape == (8, 8)
    assert np.allclose(t, t.conj().T)


def test_block_levinson_matches_direct_solve() -> None:
    coeffs = [
        np.array([[0.40, 0.08], [-0.03, 0.31]]),
        np.array([[-0.15, 0.02], [0.01, -0.10]]),
    ]
    x = _simulate_var(coeffs, samples=30000, seed=3)
    r = ld.multichannel_autocorrelation(x, order=2)
    direct = ld.solve_block_yule_walker_direct(r, order=2)
    levinson = ld.block_levinson_durbin(r, order=2)
    assert np.linalg.norm(direct.coefficients - levinson.coefficients) < 1e-10
    assert np.linalg.norm(direct.prediction_error - levinson.prediction_error) < 1e-10
    assert np.max(levinson.reflection_spectral_norms) < 1.0


def test_multichannel_ar_estimates_known_var() -> None:
    coeffs = [
        np.array([[0.45, -0.06], [0.09, 0.34]]),
        np.array([[-0.18, 0.03], [-0.02, -0.13]]),
    ]
    x = _simulate_var(coeffs, samples=60000, seed=4)
    r = ld.multichannel_autocorrelation(x, order=2)
    result = ld.block_levinson_durbin(r, order=2)
    rel_err = np.linalg.norm(result.coefficients.real - np.asarray(coeffs)) / np.linalg.norm(coeffs)
    assert rel_err < 0.08
    assert ld.companion_spectral_radius(result.coefficients) < 1.0


def test_prediction_error_and_frequency_response_shapes() -> None:
    coeffs = np.asarray(
        [
            [[0.30, 0.04], [-0.02, 0.25]],
            [[-0.08, 0.01], [0.02, -0.06]],
        ]
    )
    x = _simulate_var([coeffs[0], coeffs[1]], samples=1024, seed=5)
    e = ld.multichannel_prediction_error(x, coeffs)
    h = ld.matrix_ar_frequency_response(coeffs, np.linspace(0, np.pi, 32))
    assert e.shape == (1022, 2)
    assert h.shape == (32, 2, 2)
    assert np.isfinite(h).all()
