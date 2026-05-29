from __future__ import annotations

import numpy as np
import pytest

import lattice_dsp as ld


def _constant_contraction(
    rng: np.random.Generator, output_dim: int, input_dim: int, scale: float = 0.55
) -> np.ndarray:
    raw = rng.normal(size=(output_dim, input_dim)) + 1j * rng.normal(size=(output_dim, input_dim))
    smax = np.linalg.svd(raw, compute_uv=False)[0]
    return scale * raw / smax


def test_right_tangential_pick_matrix_scalar_reduces_to_classical_pick() -> None:
    z = np.array([0.0, 0.3 + 0.1j, -0.25j])
    w = 0.5 * z  # feasible scalar Schur data from f(z)=0.5 z
    data = ld.RightTangentialSchurData(z, np.ones((z.size, 1)), w[:, None])
    pick = ld.right_tangential_pick_matrix(data)
    expected = np.empty_like(pick)
    for i, zi in enumerate(z):
        for j, zj in enumerate(z):
            expected[i, j] = (1.0 - np.conj(w[i]) * w[j]) / (1.0 - np.conj(zi) * zj)
    np.testing.assert_allclose(pick, expected, atol=1e-14)
    assert ld.is_tangential_schur_solvable(data)


def test_pick_matrix_detects_infeasible_scalar_data() -> None:
    data = ld.RightTangentialSchurData([0.0], [[1.0]], [[1.2]])
    pick = ld.right_tangential_pick_matrix(data)
    assert ld.pick_matrix_eigenvalues(pick)[0] < 0.0
    assert not ld.is_tangential_schur_solvable(data)


def test_right_tangential_pick_matrix_for_constant_mimo_solution_is_psd() -> None:
    rng = np.random.default_rng(1)
    s0 = _constant_contraction(rng, output_dim=3, input_dim=2)
    z = np.array([0.0, 0.15 - 0.2j, -0.35 + 0.1j, 0.25j])
    directions = rng.normal(size=(z.size, 2)) + 1j * rng.normal(size=(z.size, 2))
    values = np.einsum("oi,ni->no", s0, directions)
    data = ld.RightTangentialSchurData(z, directions, values)
    pick = ld.right_tangential_pick_matrix(data)
    assert np.min(ld.pick_matrix_eigenvalues(pick)) > -1e-10
    recovered = ld.constant_schur_solution(data)
    np.testing.assert_allclose(recovered, s0, atol=1e-10)
    assert ld.max_tangential_residual(data, recovered) < 1e-10


def test_constant_schur_solution_rejects_nonconstant_compatible_pick_data() -> None:
    z = np.array([0.0, 0.4])
    u = np.ones((2, 1), dtype=np.complex128)
    v = np.array([[0.0], [0.4]], dtype=np.complex128)  # S(z)=z is feasible but not constant.
    data = ld.RightTangentialSchurData(z, u, v)
    assert ld.is_tangential_schur_solvable(data)
    with pytest.raises(ValueError, match="constant"):
        ld.constant_schur_solution(data)


def test_tangential_residual_accepts_callable_and_pointwise_arrays() -> None:
    s0 = np.array([[0.2, -0.1j], [0.05, 0.3]], dtype=np.complex128)
    z = np.array([0.1, -0.2j])
    u = np.array([[1.0, 0.0], [0.5, 1.0]], dtype=np.complex128)
    v = np.einsum("oi,ni->no", s0, u)
    data = ld.RightTangentialSchurData(z, u, v)
    assert ld.max_tangential_residual(data, lambda _z: s0) < 1e-14
    pointwise = np.stack([s0, s0], axis=0)
    assert ld.max_tangential_residual(data, pointwise) < 1e-14


def test_multiple_tangential_directions_per_point() -> None:
    rng = np.random.default_rng(3)
    s0 = _constant_contraction(rng, 2, 3)
    z = np.array([0.1 + 0.2j, -0.25])
    u = rng.normal(size=(2, 3, 2)) + 1j * rng.normal(size=(2, 3, 2))
    v = np.einsum("oi,nir->nor", s0, u)
    data = ld.RightTangentialSchurData(z, u, v)
    assert data.multiplicities == (2, 2)
    assert data.total_conditions == 4
    pick = ld.right_tangential_pick_matrix(data)
    assert pick.shape == (4, 4)
    assert ld.is_tangential_schur_solvable(data)
    np.testing.assert_allclose(ld.constant_schur_solution(data), s0, atol=1e-10)


def test_j_signature_and_graph_contractivity_sign() -> None:
    j = ld.j_signature(2, 3)
    assert j.shape == (5, 5)
    u = np.array([1.0, 2.0, -1.0j])
    v = np.array([0.2, -0.1j])
    xi = np.r_[v, u]
    value = np.real(xi.conj().T @ j @ xi)
    assert value == pytest.approx(np.vdot(v, v).real - np.vdot(u, u).real)
    assert value < 0.0


def test_blaschke_factor_zero_and_unit_magnitude_on_boundary() -> None:
    alpha = 0.2 - 0.3j
    assert abs(ld.disk_blaschke(alpha, alpha)) < 1e-14
    omega = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    values = ld.disk_blaschke(np.exp(1j * omega), alpha)
    np.testing.assert_allclose(np.abs(values), 1.0, atol=1e-13)


def test_elementary_potapov_factor_is_j_inner_and_annihilates_data_vector() -> None:
    u = np.array([1.0, 0.5 - 0.25j])
    v = np.array([0.3 + 0.1j, -0.2])
    factor = ld.elementary_potapov_factor(0.25 + 0.1j, u, v)
    assert factor.annihilation_residual() < 1e-13
    idem, jself = factor.projection_residuals()
    assert idem < 1e-13
    assert jself < 1e-13
    assert factor.boundary_j_residual() < 1e-12


def test_elementary_potapov_factor_rejects_non_strict_data() -> None:
    with pytest.raises(ValueError, match="strict"):
        ld.elementary_potapov_factor(0.0, [1.0], [1.0])
    with pytest.raises(ValueError, match="strict"):
        ld.elementary_potapov_factor(0.0, [1.0], [1.2])


def test_potapov_product_is_j_inner_on_boundary() -> None:
    rng = np.random.default_rng(5)
    s0 = _constant_contraction(rng, 2, 2, scale=0.4)
    z = np.array([0.0, 0.2 + 0.1j, -0.15j])
    u = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    v = np.einsum("oi,ni->no", s0, u)
    data = ld.RightTangentialSchurData(z, u, v)
    product = ld.potapov_product_from_rank_one_data(data)
    assert product.dimension == 4
    assert product.boundary_j_residual() < 2e-12


def test_diagonal_mimo_pick_matrix_equals_independent_scalar_blocks_when_nodes_are_separated() -> (
    None
):
    # One tangential condition per channel at distinct points gives scalar Pick
    # entries when input/output directions are coordinate vectors.
    z = np.array([0.1, -0.2 + 0.1j, 0.3j])
    gains = np.array([0.2, -0.35 + 0.1j, 0.45j])
    directions = np.eye(3, dtype=np.complex128)
    values = gains[:, None] * directions
    data = ld.RightTangentialSchurData(z, directions, values)
    pick = ld.right_tangential_pick_matrix(data)
    expected = np.empty_like(pick)
    for i, zi in enumerate(z):
        for j, zj in enumerate(z):
            expected[i, j] = (
                directions[i].conj() @ directions[j] - values[i].conj() @ values[j]
            ) / (1.0 - np.conj(zi) * zj)
    np.testing.assert_allclose(pick, expected, atol=1e-14)
    assert ld.is_tangential_schur_solvable(data)


def test_shape_validation_errors_are_clear() -> None:
    with pytest.raises(ValueError, match="unit disk"):
        ld.RightTangentialSchurData([1.1], [[1.0]], [[0.0]])
    with pytest.raises(ValueError, match="first dimension"):
        ld.RightTangentialSchurData([0.0, 0.1], np.ones((1, 2)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="multiplicities"):
        ld.RightTangentialSchurData([0.0], np.ones((1, 2, 2)), np.ones((1, 1, 1)))


def test_j_unitarity_residual_rejects_bad_shapes() -> None:
    j = ld.j_signature(1, 1)
    with pytest.raises(ValueError, match="square"):
        ld.j_unitarity_residual(np.eye(2), np.ones((2, 3)))
    with pytest.raises(ValueError, match="incompatible"):
        ld.j_unitarity_residual(np.eye(3), j)
