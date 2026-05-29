from __future__ import annotations

import numpy as np
import pytest

import lattice_dsp as ld


def _simulate_var(coefficients: np.ndarray, samples: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order, channels, _ = coefficients.shape
    x = np.zeros((samples + 256, channels), dtype=np.float64)
    noise = rng.normal(scale=0.35, size=x.shape)
    for n in range(order, x.shape[0]):
        y = noise[n].copy()
        for lag in range(1, order + 1):
            y -= coefficients[lag - 1] @ x[n - lag]
        x[n] = y
    return x[256:]


def test_online_mimo_lattice_matches_block_levinson_ar_residual() -> None:
    coeffs = np.asarray(
        [
            [[0.33, 0.07], [-0.04, 0.29]],
            [[-0.11, 0.03], [0.02, -0.09]],
        ],
        dtype=np.float64,
    )
    x = _simulate_var(coeffs, samples=4096, seed=1)
    r = ld.multichannel_autocorrelation(x, order=2)
    result = ld.block_levinson_durbin(r, order=2)

    predictor = ld.MIMOLatticePredictor.from_levinson(result)
    prediction, error = predictor.process(x)

    direct_error = ld.multichannel_prediction_error(x, result.coefficients)
    assert prediction.shape == x.shape
    assert error.shape == x.shape
    assert np.linalg.norm(error[2:] - direct_error) / np.linalg.norm(direct_error) < 1e-12
    assert np.allclose(error, x - prediction, atol=1e-12)


def test_predict_is_available_before_update_and_does_not_see_current_sample() -> None:
    kf = np.asarray(
        [
            [[0.25, -0.05], [0.03, 0.18]],
            [[-0.08, 0.02], [0.01, -0.06]],
        ]
    )
    kb = np.asarray(
        [
            [[0.24, 0.04], [-0.02, 0.19]],
            [[-0.07, -0.01], [0.03, -0.05]],
        ]
    )
    predictor = ld.MIMOLatticePredictor(kf, kb)

    for sample in (np.array([1.0, -0.5]), np.array([0.2, 0.7]), np.array([-0.1, 0.4])):
        prediction = predictor.predict()
        error = predictor.update(sample)
        assert np.allclose(error, sample - prediction)

    before_a = predictor.predict()
    before_b = predictor.predict()
    assert np.allclose(before_a, before_b)

    error_a = predictor.update(np.array([3.0, -2.0]))
    assert np.allclose(error_a, np.array([3.0, -2.0]) - before_a)


def test_causal_mimo_lattice_predict_wrapper_and_validation() -> None:
    x = np.arange(12, dtype=float).reshape(6, 2) / 10.0
    kf = np.zeros((1, 2, 2), dtype=float)
    kf[0] = np.array([[0.1, 0.02], [-0.03, 0.2]])
    kb = kf.copy()

    prediction, error = ld.causal_mimo_lattice_predict(x, kf, kb)
    assert prediction.shape == x.shape
    assert error.shape == x.shape
    assert np.allclose(error, x - prediction)

    with pytest.raises(ValueError, match="same shape"):
        ld.MIMOLatticePredictor(kf, np.zeros((2, 2, 2)))


def test_from_levinson_rejects_direct_solver_result() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(256, 2))
    r = ld.multichannel_autocorrelation(x, order=2)
    direct = ld.solve_block_yule_walker_direct(r, order=2)
    with pytest.raises(ValueError, match="block_levinson"):
        ld.MIMOLatticePredictor.from_levinson(direct)


def test_diagonal_online_mimo_predictor_equals_independent_siso_predictors() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(128, 3))
    forward_scalars = np.asarray(
        [[0.18, -0.12, 0.07], [-0.05, 0.03, -0.02], [0.015, -0.01, 0.012]], dtype=np.float64
    )
    backward_scalars = np.asarray(
        [[0.16, -0.10, 0.06], [-0.04, 0.02, -0.01], [0.012, -0.008, 0.010]], dtype=np.float64
    )
    kf = np.asarray([np.diag(row) for row in forward_scalars])
    kb = np.asarray([np.diag(row) for row in backward_scalars])

    mimo = ld.MIMOLatticePredictor(kf, kb)
    siso = [
        ld.MIMOLatticePredictor(
            forward_scalars[:, ch].reshape(-1, 1, 1),
            backward_scalars[:, ch].reshape(-1, 1, 1),
        )
        for ch in range(x.shape[1])
    ]
    prediction_mimo = np.empty_like(x, dtype=np.float64)
    prediction_siso = np.empty_like(x, dtype=np.float64)
    error_mimo = np.empty_like(x, dtype=np.float64)
    error_siso = np.empty_like(x, dtype=np.float64)

    for n, sample in enumerate(x):
        # Explicit online contract: predict first using only previous samples,
        # then update with the current vector.
        prediction_mimo[n] = mimo.predict().real
        error_mimo[n] = mimo.update(sample).real
        for ch, predictor in enumerate(siso):
            prediction_siso[n, ch] = predictor.predict()[0].real
            error_siso[n, ch] = predictor.update(np.array([sample[ch]]))[0].real

    assert np.allclose(prediction_mimo, prediction_siso, atol=1e-13)
    assert np.allclose(error_mimo, error_siso, atol=1e-13)

    # The batch convenience path should produce the same online sequence.
    prediction_batch, error_batch = ld.MIMOLatticePredictor(kf, kb).process(x)
    assert np.allclose(prediction_batch.real, prediction_mimo, atol=1e-13)
    assert np.allclose(error_batch.real, error_mimo, atol=1e-13)


def test_coupled_online_mimo_predictor_beats_independent_siso_on_coupled_var() -> None:
    coeffs = np.asarray(
        [
            [[0.55, 0.30, 0.00], [-0.25, 0.45, 0.22], [0.18, -0.12, 0.40]],
            [[-0.18, 0.08, 0.02], [0.05, -0.14, -0.05], [-0.03, 0.07, -0.10]],
        ],
        dtype=np.float64,
    )
    order = coeffs.shape[0]
    x = _simulate_var(coeffs, samples=4096, seed=12)
    train = x[:3000]
    test = x[3000:]

    full_result = ld.block_levinson_durbin(
        ld.multichannel_autocorrelation(train, order=order), order=order
    )
    _, full_error = ld.MIMOLatticePredictor.from_levinson(full_result).process(test)

    independent_error = np.empty_like(test)
    independent_prediction = np.empty_like(test)
    predictors = []
    for ch in range(test.shape[1]):
        channel_result = ld.block_levinson_durbin(
            ld.multichannel_autocorrelation(train[:, [ch]], order=order),
            order=order,
        )
        predictors.append(ld.MIMOLatticePredictor.from_levinson(channel_result))

    for n, sample in enumerate(test):
        for ch, predictor in enumerate(predictors):
            # Same online contract as the public example: prediction first,
            # update after the current sample is revealed.
            independent_prediction[n, ch] = predictor.predict()[0].real
            independent_error[n, ch] = predictor.update(np.asarray([sample[ch]]))[0].real

    warmup = order
    full_rms = np.sqrt(np.mean(full_error[warmup:].real ** 2))
    independent_rms = np.sqrt(np.mean(independent_error[warmup:] ** 2))
    assert full_rms < 0.98 * independent_rms

    def mean_abs_offdiag_corr(error: np.ndarray) -> float:
        centered = error[warmup:] - np.mean(error[warmup:], axis=0, keepdims=True)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        corr = cov / (np.sqrt(np.outer(np.diag(cov), np.diag(cov))) + 1e-30)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        return float(np.mean(np.abs(corr[mask])))

    assert mean_abs_offdiag_corr(full_error.real) < mean_abs_offdiag_corr(independent_error)
