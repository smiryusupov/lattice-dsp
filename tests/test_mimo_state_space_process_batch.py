import numpy as np
import pytest

import lattice_dsp as ld


def python_state_space_process(A, B, C, D, x):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    x = np.asarray(x, dtype=float)
    batch, samples, _ = x.shape
    state = np.zeros((batch, A.shape[0]), dtype=float)
    y = np.zeros((batch, samples, D.shape[0]), dtype=float)
    for n in range(samples):
        xn = x[:, n, :]
        y[:, n, :] = state @ C.T + xn @ D.T
        if A.shape[0]:
            state = state @ A.T + xn @ B.T
    return y


def test_mimo_state_space_process_batch_matches_python_reference():
    rng = np.random.default_rng(123)
    state_order = 5
    inputs = 3
    outputs = 2
    A = 0.35 * rng.normal(size=(state_order, state_order))
    A *= 0.8 / max(abs(np.linalg.eigvals(A)))
    B = 0.2 * rng.normal(size=(state_order, inputs))
    C = 0.2 * rng.normal(size=(outputs, state_order))
    D = 0.05 * rng.normal(size=(outputs, inputs))
    x = rng.normal(size=(4, 25, inputs))

    expected = python_state_space_process(A, B, C, D, x)
    got = ld.mimo_state_space_process_batch(A, B, C, D, x, n_threads=1)

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_mimo_state_space_process_batch_handles_direct_feedthrough_only():
    D = np.array([[1.0, -0.5], [0.25, 2.0]])
    A = np.zeros((0, 0))
    B = np.zeros((0, 2))
    C = np.zeros((2, 0))
    x = np.arange(12, dtype=float).reshape(2, 3, 2)

    got = ld.mimo_state_space_process_batch(A, B, C, D, x)
    expected = np.einsum("bni,oi->bno", x, D)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_mimo_state_space_process_batch_validates_shapes():
    A = np.eye(2)
    B = np.ones((2, 2))
    C = np.ones((1, 2))
    D = np.ones((1, 2))
    x = np.ones((1, 4, 3))
    with pytest.raises(ValueError, match="x must have shape"):
        ld.mimo_state_space_process_batch(A, B, C, D, x)
