import numpy as np

import lattice_dsp as ld


def _diagonal_markov(responses):
    responses = [np.asarray(r, dtype=float) for r in responses]
    n = min(len(r) for r in responses)
    channels = len(responses)
    markov = np.zeros((n, channels, channels), dtype=float)
    for ch, h in enumerate(responses):
        markov[:, ch, ch] = h[:n]
    return markov


def test_mimo_state_space_markov_response_shapes_and_values():
    A = np.array([[0.5]])
    B = np.array([[2.0]])
    C = np.array([[3.0]])
    D = np.array([[1.0]])
    markov = ld.mimo_state_space_markov_response(A, B, C, D, 5)
    expected = np.array([1.0, 6.0, 3.0, 1.5, 0.75])
    assert markov.shape == (5, 1, 1)
    np.testing.assert_allclose(markov[:, 0, 0], expected, atol=1e-12, rtol=1e-12)


def test_finite_hankel_reduce_mimo_recovers_diagonal_first_order_systems():
    poles = [0.25, -0.35, 0.55]
    gains = [1.0, 0.7, -1.2]
    responses = [g * (p ** np.arange(80)) for p, g in zip(poles, gains, strict=True)]
    markov = _diagonal_markov(responses)

    result = ld.finite_hankel_reduce_mimo(markov, reduced_order=3, block_rows=12, block_cols=12)
    reduced = ld.mimo_state_space_markov_response(
        result["A"], result["B"], result["C"], result["D"], markov.shape[0]
    )

    assert result["stable"] is True
    assert result["state_order"] == 3
    assert result["retained_hankel_energy"] > 1.0 - 1e-12
    assert result["relative_markov_error"] < 1e-12
    np.testing.assert_allclose(reduced, markov, atol=1e-10, rtol=1e-10)


def test_finite_hankel_reduce_mimo_lower_order_has_smaller_state():
    responses = [
        ld.iir_impulse_response([1.0, -0.6, 0.08], [1.0], 120),
        ld.iir_impulse_response([1.0, 0.45, 0.05], [0.8], 120),
    ]
    markov = _diagonal_markov(responses)

    result = ld.finite_hankel_reduce_mimo(markov, reduced_order=2, block_rows=16, block_cols=16)
    reduced = ld.mimo_state_space_markov_response(
        result["A"], result["B"], result["C"], result["D"], markov.shape[0]
    )

    assert result["A"].shape == (2, 2)
    assert result["B"].shape == (2, 2)
    assert result["C"].shape == (2, 2)
    assert result["D"].shape == (2, 2)
    assert result["stable"] is True
    assert np.mean((markov - reduced) ** 2) / np.mean(markov**2) < 0.25


def test_finite_hankel_reduce_mimo_rejects_too_short_markov_sequence():
    markov = np.zeros((5, 2, 2))
    try:
        ld.finite_hankel_reduce_mimo(markov, reduced_order=1, block_rows=3, block_cols=3)
    except ValueError as exc:
        assert "block_rows + block_cols + 1" in str(exc) or "too short" in str(exc)
    else:
        raise AssertionError("expected ValueError")
