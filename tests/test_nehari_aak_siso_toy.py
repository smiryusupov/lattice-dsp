import numpy as np

from examples import nehari_aak_siso_toy as toy


def test_finite_eckart_young_error_matches_next_singular_value():
    rows = cols = 12
    gamma = toy.anticausal_tail(rows + cols - 1)
    hankel = toy.hankel_from_tail(gamma, rows, cols)
    u, s, vt = np.linalg.svd(hankel, full_matrices=False)

    rank = 3
    truncated = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    error = np.linalg.norm(hankel - truncated, 2)

    np.testing.assert_allclose(error, s[rank], rtol=1e-11, atol=1e-11)


def test_hankelize_returns_constant_anti_diagonals():
    rng = np.random.default_rng(123)
    matrix = rng.normal(size=(5, 4))
    tail, projected = toy.hankelize(matrix)

    assert tail.shape == (8,)
    for i in range(projected.shape[0]):
        for j in range(projected.shape[1]):
            assert projected[i, j] == tail[i + j]
