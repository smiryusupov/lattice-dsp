"""Tangential Schur/Pick and elementary J-inner utilities.

This module intentionally implements a conservative finite-dimensional subset
of tangential Schur interpolation.  The supported public objects are:

* right tangential Schur data ``S(z_i) U_i = V_i``;
* the definite tangential Pick matrix certificate;
* constant contractive solutions when the finite data happen to be compatible
  with a constant Schur function;
* elementary Potapov/J-inner factors associated with strict rank-one data
  vectors ``[v; u]``.

It is not a full generalized indefinite Schur solver.  The routines are useful
for checking interpolation data, building J-inner diagnostics, and connecting
matrix-lattice/all-pass examples to their tangential-Schur background.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[complex] | Sequence[Sequence[complex]]


def _as_points(points: ArrayLike) -> np.ndarray:
    pts = np.asarray(points, dtype=np.complex128)
    if pts.ndim != 1:
        raise ValueError("points must be a one-dimensional array")
    if pts.size == 0:
        raise ValueError("at least one interpolation point is required")
    if not np.all(np.isfinite(pts.real)) or not np.all(np.isfinite(pts.imag)):
        raise ValueError("points must contain only finite values")
    if np.any(np.abs(pts) >= 1.0):
        raise ValueError("all interpolation points must lie strictly inside the unit disk")
    return np.ascontiguousarray(pts)


def _as_point_matrices(value: ArrayLike, n_points: int, name: str) -> tuple[np.ndarray, ...]:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 1:
        if n_points != 1:
            raise ValueError(f"{name} has one vector but there are {n_points} interpolation points")
        return (np.ascontiguousarray(arr[:, None]),)
    if arr.ndim == 2:
        if arr.shape[0] != n_points:
            raise ValueError(f"{name} must have first dimension equal to the number of points")
        return tuple(np.ascontiguousarray(arr[i, :, None]) for i in range(n_points))
    if arr.ndim == 3:
        if arr.shape[0] != n_points:
            raise ValueError(f"{name} must have first dimension equal to the number of points")
        return tuple(np.ascontiguousarray(arr[i]) for i in range(n_points))
    raise ValueError(
        f"{name} must have shape (dim,), (points, dim), or (points, dim, multiplicity)"
    )


def _as_vector(value: ArrayLike, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.complex128)
    if vec.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if vec.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(vec.real)) or not np.all(np.isfinite(vec.imag)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(vec)


def _hermitian_part(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


@dataclass(frozen=True, init=False)
class RightTangentialSchurData:
    """Right tangential Schur interpolation data.

    The data describe matrix-valued Schur functions

    ``S : D -> C^{output_dim x input_dim}``

    satisfying

    ``S(z_i) U_i = V_i``.

    ``U_i`` is an ``input_dim x r_i`` direction matrix and ``V_i`` is an
    ``output_dim x r_i`` target matrix.  A common rank-one input accepts arrays
    with shape ``(points, input_dim)`` and ``(points, output_dim)``.
    """

    points: np.ndarray
    directions: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]
    input_dim: int
    output_dim: int

    def __init__(self, points: ArrayLike, directions: ArrayLike, values: ArrayLike) -> None:
        pts = _as_points(points)
        dirs = _as_point_matrices(directions, pts.size, "directions")
        vals = _as_point_matrices(values, pts.size, "values")
        input_dim = int(dirs[0].shape[0])
        output_dim = int(vals[0].shape[0])
        if input_dim == 0 or output_dim == 0:
            raise ValueError("input and output dimensions must be non-zero")
        for i, (u, v) in enumerate(zip(dirs, vals, strict=True)):
            if u.ndim != 2 or v.ndim != 2:
                raise ValueError("directions and values must be matrices after normalization")
            if u.shape[0] != input_dim:
                raise ValueError("all direction matrices must have the same input dimension")
            if v.shape[0] != output_dim:
                raise ValueError("all value matrices must have the same output dimension")
            if u.shape[1] != v.shape[1]:
                raise ValueError(
                    f"directions and values at point {i} have different multiplicities"
                )
            if u.shape[1] == 0:
                raise ValueError("multiplicity must be positive at every point")
            if not np.all(np.isfinite(u.real)) or not np.all(np.isfinite(u.imag)):
                raise ValueError("directions must contain only finite values")
            if not np.all(np.isfinite(v.real)) or not np.all(np.isfinite(v.imag)):
                raise ValueError("values must contain only finite values")
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "directions", tuple(np.ascontiguousarray(u) for u in dirs))
        object.__setattr__(self, "values", tuple(np.ascontiguousarray(v) for v in vals))
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "output_dim", output_dim)

    @property
    def n_points(self) -> int:
        """Number of interpolation nodes."""

        return int(self.points.size)

    @property
    def multiplicities(self) -> tuple[int, ...]:
        """Tangential multiplicity at each node."""

        return tuple(int(u.shape[1]) for u in self.directions)

    @property
    def total_conditions(self) -> int:
        """Total number of tangential columns across all nodes."""

        return int(sum(self.multiplicities))

    def stacked_directions(self) -> np.ndarray:
        """Return ``[U_1 ... U_n]`` with shape ``(input_dim, total_conditions)``."""

        return np.ascontiguousarray(np.hstack(self.directions))

    def stacked_values(self) -> np.ndarray:
        """Return ``[V_1 ... V_n]`` with shape ``(output_dim, total_conditions)``."""

        return np.ascontiguousarray(np.hstack(self.values))


def right_tangential_pick_matrix(data: RightTangentialSchurData) -> np.ndarray:
    """Return the definite right-tangential Pick matrix.

    The block ``(i, j)`` is

    ``(U_i^* U_j - V_i^* V_j) / (1 - conj(z_i) z_j)``.
    """

    if not isinstance(data, RightTangentialSchurData):
        data = RightTangentialSchurData(data.points, data.directions, data.values)  # type: ignore[attr-defined]
    sizes = data.multiplicities
    offsets = np.cumsum((0, *sizes))
    pick = np.empty((data.total_conditions, data.total_conditions), dtype=np.complex128)
    for i, (zi, ui, vi) in enumerate(zip(data.points, data.directions, data.values, strict=True)):
        row = slice(offsets[i], offsets[i + 1])
        for j, (zj, uj, vj) in enumerate(
            zip(data.points, data.directions, data.values, strict=True)
        ):
            col = slice(offsets[j], offsets[j + 1])
            numerator = ui.conj().T @ uj - vi.conj().T @ vj
            denominator = 1.0 - np.conj(zi) * zj
            pick[row, col] = numerator / denominator
    return np.ascontiguousarray(_hermitian_part(pick))


def pick_matrix_eigenvalues(pick: np.ndarray) -> np.ndarray:
    """Return sorted Hermitian eigenvalues of a Pick matrix."""

    mat = np.asarray(pick, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("pick must be a square matrix")
    return np.linalg.eigvalsh(_hermitian_part(mat))


def is_pick_positive_semidefinite(pick: np.ndarray, *, tol: float = 1e-10) -> bool:
    """Return ``True`` when a Pick matrix is numerically positive semidefinite."""

    if tol < 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and non-negative")
    eig = pick_matrix_eigenvalues(pick)
    return bool(eig[0] >= -tol)


def is_tangential_schur_solvable(data: RightTangentialSchurData, *, tol: float = 1e-10) -> bool:
    """Return the definite Schur-class feasibility test for the data."""

    return is_pick_positive_semidefinite(right_tangential_pick_matrix(data), tol=tol)


def constant_schur_solution(
    data: RightTangentialSchurData,
    *,
    tol: float = 1e-10,
    require_contractivity: bool = True,
    contractive_tol: float = 1e-10,
) -> np.ndarray:
    """Return a constant contractive matrix satisfying compatible data.

    This solves ``S @ [U_i] = [V_i]`` by a minimum-norm pseudoinverse.  It is a
    complete solution only for data compatible with a constant Schur function;
    otherwise a ``ValueError`` is raised.  General finite tangential Schur
    synthesis is intentionally outside this helper.
    """

    if tol < 0.0 or contractive_tol < 0.0:
        raise ValueError("tolerances must be non-negative")
    u = data.stacked_directions()
    v = data.stacked_values()
    solution = v @ np.linalg.pinv(u)
    residual = np.linalg.norm(solution @ u - v)
    scale = max(1.0, np.linalg.norm(v))
    if residual > tol * scale:
        raise ValueError("tangential data are not compatible with a constant Schur function")
    if require_contractivity:
        sigma_max = float(np.linalg.svd(solution, compute_uv=False)[0]) if solution.size else 0.0
        if sigma_max > 1.0 + contractive_tol:
            raise ValueError("the compatible constant solution is not contractive")
    return np.ascontiguousarray(solution)


def tangential_interpolation_residual(
    data: RightTangentialSchurData,
    transfer: np.ndarray | Callable[[complex], np.ndarray],
) -> np.ndarray:
    """Return residual blocks ``S(z_i) U_i - V_i`` for each interpolation node."""

    residuals: list[np.ndarray] = []
    if callable(transfer):
        for z, u, v in zip(data.points, data.directions, data.values, strict=True):
            s = np.asarray(transfer(complex(z)), dtype=np.complex128)
            if s.shape != (data.output_dim, data.input_dim):
                raise ValueError("transfer callable returned a matrix with incompatible shape")
            residuals.append(np.ascontiguousarray(s @ u - v))
        return np.ascontiguousarray(np.hstack(residuals))

    arr = np.asarray(transfer, dtype=np.complex128)
    if arr.ndim == 2:
        if arr.shape != (data.output_dim, data.input_dim):
            raise ValueError("constant transfer matrix has incompatible shape")
        return np.ascontiguousarray(
            np.hstack([arr @ u - v for u, v in zip(data.directions, data.values, strict=True)])
        )
    if arr.ndim == 3:
        if arr.shape[0] != data.n_points or arr.shape[1:] != (data.output_dim, data.input_dim):
            raise ValueError("transfer array must have shape (points, output_dim, input_dim)")
        return np.ascontiguousarray(
            np.hstack(
                [
                    arr[i] @ u - v
                    for i, (u, v) in enumerate(zip(data.directions, data.values, strict=True))
                ]
            )
        )
    raise ValueError("transfer must be a matrix, an array over points, or a callable")


def max_tangential_residual(
    data: RightTangentialSchurData, transfer: np.ndarray | Callable[[complex], np.ndarray]
) -> float:
    """Return the Frobenius norm of all interpolation residuals."""

    return float(np.linalg.norm(tangential_interpolation_residual(data, transfer)))


def j_signature(output_dim: int, input_dim: int) -> np.ndarray:
    """Return ``diag(I_output, -I_input)`` for Schur graph vectors ``[v; u]``."""

    p = int(output_dim)
    q = int(input_dim)
    if p <= 0 or q <= 0:
        raise ValueError("output_dim and input_dim must be positive")
    return np.ascontiguousarray(np.diag(np.r_[np.ones(p), -np.ones(q)]).astype(np.complex128))


def disk_blaschke(z: complex | np.ndarray, alpha: complex) -> complex | np.ndarray:
    """Scalar Blaschke factor ``(z-alpha)/(1-conj(alpha) z)``."""

    a = complex(alpha)
    if not np.isfinite(a.real) or not np.isfinite(a.imag) or abs(a) >= 1.0:
        raise ValueError("alpha must lie strictly inside the unit disk")
    zz = np.asarray(z, dtype=np.complex128)
    return (zz - a) / (1.0 - np.conj(a) * zz)


def j_unitarity_residual(theta: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Return ``||Theta^* J Theta - J||_2`` for one or many matrices."""

    th = np.asarray(theta, dtype=np.complex128)
    jj = np.asarray(j, dtype=np.complex128)
    if jj.ndim != 2 or jj.shape[0] != jj.shape[1]:
        raise ValueError("j must be a square matrix")
    if th.ndim == 2:
        if th.shape != jj.shape:
            raise ValueError("theta and j shapes are incompatible")
        return np.asarray(np.linalg.norm(th.conj().T @ jj @ th - jj, ord=2))
    if th.ndim == 3:
        if th.shape[1:] != jj.shape:
            raise ValueError("theta and j shapes are incompatible")
        return np.asarray([np.linalg.norm(t.conj().T @ jj @ t - jj, ord=2) for t in th])
    raise ValueError("theta must have shape (dim, dim) or (points, dim, dim)")


@dataclass(frozen=True)
class TangentialPotapovFactor:
    """Elementary J-inner factor associated with one strict tangential vector.

    For ``xi = [v; u]`` and ``J = diag(I_output, -I_input)``, strict Schur data
    satisfy ``xi^* J xi < 0``.  The factor

    ``Theta(z) = I + (b_alpha(z) - 1) P``

    uses the J-orthogonal projection ``P`` onto ``span(xi)`` and is J-inner on
    the unit circle.  At ``z = alpha`` it annihilates the data vector ``xi``.
    """

    point: complex
    input_direction: np.ndarray
    output_value: np.ndarray
    projection: np.ndarray
    j: np.ndarray
    j_norm: complex

    def __init__(
        self,
        point: complex,
        input_direction: ArrayLike,
        output_value: ArrayLike,
        *,
        strict_tol: float = 1e-12,
    ) -> None:
        alpha = complex(point)
        if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag) or abs(alpha) >= 1.0:
            raise ValueError("point must lie strictly inside the unit disk")
        if strict_tol <= 0.0 or not np.isfinite(strict_tol):
            raise ValueError("strict_tol must be positive and finite")
        u = _as_vector(input_direction, "input_direction")
        v = _as_vector(output_value, "output_value")
        jj = j_signature(v.size, u.size)
        xi = np.r_[v, u].astype(np.complex128)
        jnorm = complex(xi.conj().T @ jj @ xi)
        if abs(jnorm.imag) > 100.0 * np.finfo(float).eps * max(1.0, abs(jnorm.real)):
            raise ValueError("J-norm should be real for a Schur graph vector")
        if jnorm.real >= -strict_tol:
            raise ValueError("strict rank-one data require ||output_value|| < ||input_direction||")
        proj = np.outer(xi, xi.conj().T @ jj) / jnorm
        object.__setattr__(self, "point", alpha)
        object.__setattr__(self, "input_direction", u)
        object.__setattr__(self, "output_value", v)
        object.__setattr__(self, "projection", np.ascontiguousarray(proj))
        object.__setattr__(self, "j", jj)
        object.__setattr__(self, "j_norm", jnorm)

    @property
    def dimension(self) -> int:
        """Total J-space dimension ``output_dim + input_dim``."""

        return int(self.j.shape[0])

    @property
    def output_dim(self) -> int:
        """Schur output dimension."""

        return int(self.output_value.size)

    @property
    def input_dim(self) -> int:
        """Schur input dimension."""

        return int(self.input_direction.size)

    @property
    def data_vector(self) -> np.ndarray:
        """Return the graph vector ``[v; u]`` associated with the datum."""

        return np.ascontiguousarray(np.r_[self.output_value, self.input_direction])

    def evaluate(self, z: complex | Iterable[complex] | np.ndarray) -> np.ndarray:
        """Evaluate the elementary factor at one point or a vector of points."""

        zz = np.asarray(z, dtype=np.complex128)
        ident = np.eye(self.dimension, dtype=np.complex128)
        b = disk_blaschke(zz, self.point)
        if zz.ndim == 0:
            return np.ascontiguousarray(ident + (complex(b) - 1.0) * self.projection)
        out = np.empty(zz.shape + ident.shape, dtype=np.complex128)
        for index in np.ndindex(zz.shape):
            out[index] = ident + (b[index] - 1.0) * self.projection
        return (
            np.ascontiguousarray(out.reshape((-1, self.dimension, self.dimension)))
            if zz.ndim == 1
            else out
        )

    def annihilation_residual(self) -> float:
        """Return ``||Theta(point) [v; u]||_2``."""

        return float(np.linalg.norm(self.evaluate(self.point) @ self.data_vector))

    def projection_residuals(self) -> tuple[float, float]:
        """Return idempotence and J-self-adjoint projection residuals."""

        p = self.projection
        idem = np.linalg.norm(p @ p - p)
        jself = np.linalg.norm(p.conj().T @ self.j - self.j @ p)
        return float(idem), float(jself)

    def boundary_j_residual(self, omega: np.ndarray | None = None) -> float:
        """Return the maximum J-unitarity residual on a unit-circle grid."""

        if omega is None:
            omega = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        theta = self.evaluate(np.exp(1j * np.asarray(omega)))
        return float(np.max(j_unitarity_residual(theta, self.j)))


@dataclass(frozen=True)
class PotapovProduct:
    """Product of elementary J-inner factors with the same signature."""

    factors: tuple[TangentialPotapovFactor, ...]
    j: np.ndarray

    def __init__(self, factors: Sequence[TangentialPotapovFactor]) -> None:
        if not factors:
            raise ValueError("at least one factor is required")
        first = factors[0]
        for factor in factors:
            if factor.j.shape != first.j.shape or not np.allclose(factor.j, first.j):
                raise ValueError("all factors must share the same J signature")
        object.__setattr__(self, "factors", tuple(factors))
        object.__setattr__(self, "j", first.j.copy())

    @property
    def dimension(self) -> int:
        return int(self.j.shape[0])

    def evaluate(self, z: complex | Iterable[complex] | np.ndarray) -> np.ndarray:
        """Evaluate the product at one point or a vector of points."""

        zz = np.asarray(z, dtype=np.complex128)
        scalar_input = zz.ndim == 0
        flat = zz.reshape(-1)
        out = np.empty((flat.size, self.dimension, self.dimension), dtype=np.complex128)
        for i, point in enumerate(flat):
            theta = np.eye(self.dimension, dtype=np.complex128)
            for factor in self.factors:
                theta = theta @ factor.evaluate(complex(point))
            out[i] = theta
        if scalar_input:
            return np.ascontiguousarray(out[0])
        return np.ascontiguousarray(out)

    def boundary_j_residual(self, omega: np.ndarray | None = None) -> float:
        """Return the maximum J-unitarity residual on a unit-circle grid."""

        if omega is None:
            omega = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        theta = self.evaluate(np.exp(1j * np.asarray(omega)))
        return float(np.max(j_unitarity_residual(theta, self.j)))


def elementary_potapov_factor(
    point: complex,
    input_direction: ArrayLike,
    output_value: ArrayLike,
    *,
    strict_tol: float = 1e-12,
) -> TangentialPotapovFactor:
    """Construct an elementary rank-one J-inner Potapov factor."""

    return TangentialPotapovFactor(point, input_direction, output_value, strict_tol=strict_tol)


def potapov_product_from_rank_one_data(
    data: RightTangentialSchurData, *, strict_tol: float = 1e-12
) -> PotapovProduct:
    """Build a J-inner product from the rank-one columns of tangential data.

    This product is a diagnostic chain of elementary factors attached to the
    data vectors.  It is J-inner on the unit circle, but it is not advertised as
    a full recursive Schur-synthesis solver for arbitrary data.
    """

    factors: list[TangentialPotapovFactor] = []
    for z, u_mat, v_mat in zip(data.points, data.directions, data.values, strict=True):
        for col in range(u_mat.shape[1]):
            factors.append(
                TangentialPotapovFactor(
                    complex(z), u_mat[:, col], v_mat[:, col], strict_tol=strict_tol
                )
            )
    return PotapovProduct(factors)
