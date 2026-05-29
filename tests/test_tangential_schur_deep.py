from __future__ import annotations

import numpy as np
import pytest

import lattice_dsp as ld


def _constant_contraction(
    rng: np.random.Generator,
    output_dim: int,
    input_dim: int,
    *,
    scale: float = 0.55,
) -> np.ndarray:
    raw = rng.normal(size=(output_dim, input_dim)) + 1j * rng.normal(size=(output_dim, input_dim))
    sigma = np.linalg.svd(raw, compute_uv=False)[0]
    return scale * raw / sigma


def _unitary(rng: np.random.Generator, size: int) -> np.ndarray:
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    q, r = np.linalg.qr(raw)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1.0)
    return q * phases.conj()


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    out = np.zeros((rows, cols), dtype=np.complex128)
    r0 = 0
    c0 = 0
    for block in blocks:
        r1 = r0 + block.shape[0]
        c1 = c0 + block.shape[1]
        out[r0:r1, c0:c1] = block
        r0 = r1
        c0 = c1
    return out


def test_random_constant_mimo_data_are_recovered_across_shapes() -> None:
    rng = np.random.default_rng(101)
    cases = [
        (1, 1, 3),
        (2, 1, 4),
        (1, 3, 4),
        (3, 2, 5),
        (4, 3, 6),
    ]
    for output_dim, input_dim, n_points in cases:
        s0 = _constant_contraction(rng, output_dim, input_dim, scale=0.62)
        points = 0.72 * np.exp(1j * np.linspace(0.1, 2.4, n_points))
        # Same multiplicity at every point.  The first point contains a full
        # basis so the constant matrix is uniquely recoverable.
        directions = rng.normal(size=(n_points, input_dim, input_dim)) + 1j * rng.normal(
            size=(n_points, input_dim, input_dim)
        )
        directions[0] = np.eye(input_dim)
        values = np.einsum("oi,nir->nor", s0, directions)
        data = ld.RightTangentialSchurData(points, directions, values)

        pick = ld.right_tangential_pick_matrix(data)
        eig = ld.pick_matrix_eigenvalues(pick)
        assert eig[0] > -5e-10
        assert ld.is_tangential_schur_solvable(data)
        recovered = ld.constant_schur_solution(data)
        np.testing.assert_allclose(recovered, s0, atol=2e-10)
        assert ld.max_tangential_residual(data, recovered) < 2e-10
        assert np.linalg.svd(recovered, compute_uv=False)[0] <= 1.0 + 1e-12


def test_pick_matrix_is_hermitian_and_unitary_mixing_is_congruent() -> None:
    rng = np.random.default_rng(102)
    output_dim = 3
    input_dim = 2
    n_points = 4
    multiplicity = 2
    s0 = _constant_contraction(rng, output_dim, input_dim, scale=0.45)
    points = np.array([0.0, 0.25 + 0.1j, -0.22 + 0.2j, 0.15 - 0.35j])
    directions = rng.normal(size=(n_points, input_dim, multiplicity)) + 1j * rng.normal(
        size=(n_points, input_dim, multiplicity)
    )
    values = np.einsum("oi,nir->nor", s0, directions)
    data = ld.RightTangentialSchurData(points, directions, values)
    pick = ld.right_tangential_pick_matrix(data)
    np.testing.assert_allclose(pick, pick.conj().T, atol=1e-14)

    mixers = [_unitary(rng, multiplicity) for _ in range(n_points)]
    mixed_directions = np.stack([directions[i] @ mixers[i] for i in range(n_points)])
    mixed_values = np.stack([values[i] @ mixers[i] for i in range(n_points)])
    mixed = ld.RightTangentialSchurData(points, mixed_directions, mixed_values)
    mixed_pick = ld.right_tangential_pick_matrix(mixed)
    block_q = _block_diag(mixers)

    np.testing.assert_allclose(mixed_pick, block_q.conj().T @ pick @ block_q, atol=2e-13)
    np.testing.assert_allclose(
        ld.pick_matrix_eigenvalues(mixed_pick), ld.pick_matrix_eigenvalues(pick), atol=2e-12
    )


def test_scalar_blaschke_data_are_feasible_but_not_constant() -> None:
    alpha = 0.25 - 0.2j
    gain = 0.7 * np.exp(0.3j)
    points = np.array([-0.3, 0.0, 0.2 + 0.25j, -0.1 + 0.35j])
    values = gain * ld.disk_blaschke(points, alpha)
    data = ld.RightTangentialSchurData(points, np.ones((points.size, 1)), values[:, None])
    pick = ld.right_tangential_pick_matrix(data)
    assert ld.pick_matrix_eigenvalues(pick)[0] > -1e-10
    assert ld.is_tangential_schur_solvable(data)
    assert (
        ld.max_tangential_residual(data, lambda z: np.array([[gain * ld.disk_blaschke(z, alpha)]]))
        < 1e-13
    )
    with pytest.raises(ValueError, match="constant"):
        ld.constant_schur_solution(data)


def test_single_node_norm_condition_is_exact_for_rank_one_data() -> None:
    feasible = ld.RightTangentialSchurData([0.2], [[2.0, 0.0]], [[0.3, 0.4]])
    infeasible = ld.RightTangentialSchurData([0.2], [[1.0, 0.0]], [[1.05, 0.0]])
    assert ld.is_tangential_schur_solvable(feasible)
    assert not ld.is_tangential_schur_solvable(infeasible)
    feasible_margin = (
        np.vdot(feasible.directions[0][:, 0], feasible.directions[0][:, 0]).real
        - np.vdot(feasible.values[0][:, 0], feasible.values[0][:, 0]).real
    )
    assert feasible_margin > 0.0
    np.testing.assert_allclose(
        ld.right_tangential_pick_matrix(feasible)[0, 0].real,
        feasible_margin / (1.0 - abs(feasible.points[0]) ** 2),
        atol=1e-14,
    )


def test_diagonal_mimo_tangential_data_decompose_into_scalar_pick_blocks() -> None:
    channels = 3
    points_per_channel = 4
    alpha = np.array([0.1, -0.2 + 0.1j, 0.25j])
    gain = np.array([0.45, -0.35 + 0.1j, 0.25j])
    all_points: list[complex] = []
    directions: list[np.ndarray] = []
    values: list[np.ndarray] = []
    scalar_blocks: list[np.ndarray] = []
    for ch in range(channels):
        z_ch = 0.65 * np.exp(1j * (np.linspace(0.2, 2.3, points_per_channel) + 0.17 * ch))
        w_ch = gain[ch] * ld.disk_blaschke(z_ch, alpha[ch])
        scalar = ld.RightTangentialSchurData(z_ch, np.ones((points_per_channel, 1)), w_ch[:, None])
        scalar_blocks.append(ld.right_tangential_pick_matrix(scalar))
        for z, w in zip(z_ch, w_ch, strict=True):
            e = np.zeros(channels, dtype=np.complex128)
            e[ch] = 1.0
            all_points.append(complex(z))
            directions.append(e)
            values.append(w * e)
    data = ld.RightTangentialSchurData(np.array(all_points), np.stack(directions), np.stack(values))
    pick = ld.right_tangential_pick_matrix(data)
    expected = _block_diag(scalar_blocks)
    np.testing.assert_allclose(pick, expected, atol=2e-13)
    assert ld.is_tangential_schur_solvable(data)


def test_near_boundary_data_remain_finite_and_report_conditioning() -> None:
    rng = np.random.default_rng(103)
    s0 = _constant_contraction(rng, 2, 2, scale=0.5)
    radii = np.array([0.9, 0.97, 0.992, 0.997])
    points = radii * np.exp(1j * np.array([0.0, 0.7, 1.4, 2.1]))
    directions = np.tile(np.eye(2, dtype=np.complex128)[None, :, :], (points.size, 1, 1))
    values = np.einsum("oi,nir->nor", s0, directions)
    data = ld.RightTangentialSchurData(points, directions, values)
    pick = ld.right_tangential_pick_matrix(data)
    eig = ld.pick_matrix_eigenvalues(pick)
    assert np.all(np.isfinite(eig))
    assert eig[0] > -1e-9
    # Near-boundary nodes amplify the kernel.  The test is not a performance
    # target; it verifies that the diagnostic remains finite and visible.
    assert np.linalg.cond(pick) > 10.0


def test_j_inner_factor_preserves_j_form_on_many_boundary_points() -> None:
    rng = np.random.default_rng(104)
    factors = []
    for point in [0.0, 0.2 + 0.1j, -0.25j, 0.15 - 0.2j]:
        u = rng.normal(size=3) + 1j * rng.normal(size=3)
        raw_v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v = 0.35 * raw_v / np.linalg.norm(raw_v) * np.linalg.norm(u)
        factor = ld.elementary_potapov_factor(point, u, v)
        idem, jself = factor.projection_residuals()
        assert idem < 1e-12
        assert jself < 1e-12
        assert factor.annihilation_residual() < 1e-12
        factors.append(factor)
    product = ld.PotapovProduct(factors)
    omega = np.linspace(0.0, 2 * np.pi, 1024, endpoint=False)
    residuals = ld.j_unitarity_residual(product.evaluate(np.exp(1j * omega)), product.j)
    assert float(np.max(residuals)) < 2e-11


def test_potapov_product_rejects_mixed_j_signatures() -> None:
    f1 = ld.elementary_potapov_factor(0.0, [1.0, 0.0], [0.2])
    f2 = ld.elementary_potapov_factor(0.1, [1.0], [0.2, 0.1])
    with pytest.raises(ValueError, match="same J"):
        ld.PotapovProduct([f1, f2])


def test_vectorized_potapov_evaluation_matches_pointwise_loop() -> None:
    factor = ld.elementary_potapov_factor(0.2 - 0.1j, [1.0, 0.3j], [0.15 - 0.05j])
    points = np.array([0.0, 0.1 + 0.2j, -0.4j, np.exp(0.7j)])
    vectorized = factor.evaluate(points)
    pointwise = np.stack([factor.evaluate(complex(z)) for z in points])
    np.testing.assert_allclose(vectorized, pointwise, atol=1e-14)


def test_constant_solution_contractivity_check_rejects_compatible_expansion() -> None:
    data = ld.RightTangentialSchurData([0.0], [[1.0, 0.0]], [[1.2, 0.0]])
    with pytest.raises(ValueError, match="contractive"):
        ld.constant_schur_solution(data)
    unconstrained = ld.constant_schur_solution(data, require_contractivity=False)
    assert np.linalg.svd(unconstrained, compute_uv=False)[0] > 1.0


def test_residual_rejects_bad_candidate_shapes_for_mimo_data() -> None:
    data = ld.RightTangentialSchurData([0.0, 0.2], np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="incompatible"):
        ld.tangential_interpolation_residual(data, np.eye(2))
    with pytest.raises(ValueError, match="shape"):
        ld.tangential_interpolation_residual(data, np.zeros((3, 3, 2)))
    with pytest.raises(ValueError, match="incompatible"):
        ld.tangential_interpolation_residual(data, lambda _z: np.eye(2))
