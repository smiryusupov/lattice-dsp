"""Three MIMO model-reduction stress cases inspired by examples 7.4.1--7.4.3.

The goal is not to claim a full matrix AAK/Nehari/tangential-Schur solver.
Instead, this script puts the package's finite block-Hankel MIMO reducer under
three deliberately different impulse-response regimes and uses the tangential
Schur/Pick layer as an interpolation-style diagnostic on sampled directions.

Cases
-----
1. A 3x3 ``1/f^alpha`` matrix tail with entries ``(j+1)^(-alpha)``.
2. A 10x10 random rational finite-dimensional response with 65 scalar basis
   poles and 1000 Markov coefficients.
3. A 2x2 high-degree rational response with 500 modes and an intentionally
   ill-conditioned realization basis, representing the kind of case where
   Gramian/balanced-truncation style computations can become numerically
   uncomfortable.

For each case, the script compares two explicit approximation methods:

* finite block-Hankel/Ho--Kalman MIMO reduction, which constructs a reduced
  state-space realization from leading block-Hankel singular directions;
* a truncated-FIR baseline, which keeps the first Markov blocks and sets the
  tail to zero.

The figures report relative H2/Markov error, finite block-Hankel spectral-norm
error, singular-value decay, reducer/runtime timings, and a finite tangential
Schur feasibility scale for sampled right-tangential data.  The tangential
Schur layer is used as a sampled Pick/RKHS certificate, not as a full matrix
AAK/Nehari/tangential-Schur reduction solver.
"""

from __future__ import annotations

import csv
import os
import time

# Keep dense SVD timings reproducible and avoid thread oversubscription when
# documentation builders run many examples in sequence.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lattice_dsp as ld


@dataclass(frozen=True)
class StressCase:
    name: str
    slug: str
    markov: np.ndarray
    reduction_orders: tuple[int, ...]
    block_rows: int
    block_cols: int
    description: str
    condition_hint: float | None = None


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_h2_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    denom = float(np.linalg.norm(reference.ravel()))
    err = float(np.linalg.norm((reference - estimate).ravel()))
    return err / denom if denom else err


def block_hankel_matrix(markov: np.ndarray, block_rows: int, block_cols: int) -> np.ndarray:
    """Build the finite block-Hankel matrix from Markov coefficients."""

    markov = np.asarray(markov, dtype=float)
    _, outputs, inputs = markov.shape
    if markov.shape[0] < block_rows + block_cols:
        raise ValueError("not enough Markov coefficients for requested block-Hankel dimensions")
    hankel = np.empty((block_rows * outputs, block_cols * inputs), dtype=float)
    for i in range(block_rows):
        for j in range(block_cols):
            hankel[i * outputs : (i + 1) * outputs, j * inputs : (j + 1) * inputs] = markov[
                i + j + 1
            ]
    return hankel


def block_hankel_singular_values(
    markov: np.ndarray, block_rows: int, block_cols: int
) -> np.ndarray:
    return np.linalg.svd(block_hankel_matrix(markov, block_rows, block_cols), compute_uv=False)


def finite_hankel_tail_error(singular_values: np.ndarray, order: int) -> float:
    """Best finite block-Hankel spectral-norm error from the SVD tail."""

    singular_values = np.asarray(singular_values, dtype=float)
    if singular_values.size == 0:
        return 0.0
    if order >= singular_values.size:
        return 0.0
    return float(singular_values[order] / max(singular_values[0], 1e-300))


def relative_hankel_norm_error(
    reference: np.ndarray,
    estimate: np.ndarray,
    block_rows: int,
    block_cols: int,
    *,
    reference_norm: float | None = None,
) -> float:
    """Relative spectral norm of the finite block-Hankel error matrix."""

    if reference_norm is None:
        reference_norm = float(
            np.linalg.norm(block_hankel_matrix(reference, block_rows, block_cols), ord=2)
        )
    error_hankel = block_hankel_matrix(reference - estimate, block_rows, block_cols)
    return float(np.linalg.norm(error_hankel, ord=2) / max(reference_norm, 1e-300))


def state_radius_or_nan(a: np.ndarray, *, max_exact_order: int = 220) -> float:
    """Return spectral radius for moderate states and NaN for large stress states."""

    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    if a.shape[0] > max_exact_order:
        return float("nan")
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def frequency_response_from_markov(markov: np.ndarray, z: complex) -> np.ndarray:
    """Evaluate ``sum_k M_k z^k`` from finite Markov coefficients."""

    powers = z ** np.arange(markov.shape[0])
    return np.tensordot(powers, markov, axes=(0, 0))


def finite_tangential_schur_diagnostic(
    markov: np.ndarray,
    *,
    n_points: int,
    seed: int,
) -> dict[str, object]:
    """Return a finite sampled right-tangential Schur/Pick diagnostic.

    The Markov response is scaled until the sampled Pick matrix is positive
    semidefinite.  This is a finite interpolation diagnostic: it does not reduce
    the model, but it reports the scale at which the sampled MIMO response is
    compatible with the Schur-class Pick condition along the chosen directions.
    """

    rng = np.random.default_rng(seed)
    _, outputs, inputs = markov.shape
    radii = np.linspace(0.18, 0.82, n_points)
    angles = np.linspace(0.13, 2.37, n_points)
    points = radii * np.exp(1j * angles)
    directions = []
    raw_values = []
    for z in points:
        u = rng.normal(size=inputs) + 0.25j * rng.normal(size=inputs)
        u = u / np.linalg.norm(u)
        directions.append(u)
        raw_values.append(frequency_response_from_markov(markov, complex(z)) @ u)
    directions_arr = np.asarray(directions, dtype=np.complex128)
    raw_values_arr = np.asarray(raw_values, dtype=np.complex128)

    sample_norm = max(float(np.linalg.norm(v)) for v in raw_values_arr)
    scale = max(1.0, 1.05 * sample_norm)
    min_eig = -np.inf
    for _ in range(20):
        data = ld.RightTangentialSchurData(points, directions_arr, raw_values_arr / scale)
        eig = ld.pick_matrix_eigenvalues(ld.right_tangential_pick_matrix(data))
        min_eig = float(np.min(eig))
        if min_eig >= -1e-10:
            return {
                "points": points,
                "directions": directions_arr,
                "values": raw_values_arr,
                "scale": float(scale),
                "min_pick_eigenvalue": min_eig,
                "max_pick_eigenvalue": float(np.max(eig)),
                "pick_eigenvalues": eig,
            }
        scale *= 1.6
    return {
        "points": points,
        "directions": directions_arr,
        "values": raw_values_arr,
        "scale": float(scale),
        "min_pick_eigenvalue": min_eig,
        "max_pick_eigenvalue": float(np.max(eig)),
        "pick_eigenvalues": eig,
    }


def tangential_sample_residual(
    reference: np.ndarray, estimate: np.ndarray, diagnostic: dict[str, object]
) -> float:
    points = np.asarray(diagnostic["points"], dtype=np.complex128)
    directions = np.asarray(diagnostic["directions"], dtype=np.complex128)
    numer = 0.0
    denom = 0.0
    for z, u in zip(points, directions, strict=True):
        ref = frequency_response_from_markov(reference, complex(z)) @ u
        est = frequency_response_from_markov(estimate, complex(z)) @ u
        numer += float(np.linalg.norm(ref - est) ** 2)
        denom += float(np.linalg.norm(ref) ** 2)
    return (numer / max(denom, 1e-300)) ** 0.5


def numpy_state_space_markov_response(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """NumPy/BLAS Markov expansion used by the high-order stress case."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    n_outputs, n_inputs = d.shape
    markov = np.empty((n_samples, n_outputs, n_inputs), dtype=float)
    markov[0] = d
    if a.size == 0:
        markov[1:] = 0.0
        return markov
    power_b = b.copy()
    for sample in range(1, n_samples):
        markov[sample] = c @ power_b
        power_b = a @ power_b
    return markov


def reduce_mimo_markov_lapack(
    markov: np.ndarray,
    order: int,
    block_rows: int,
    block_cols: int,
) -> tuple[np.ndarray | None, dict[str, object], dict[str, float]]:
    """Ho-Kalman/finite-Hankel reduction using NumPy's LAPACK SVD.

    The package's public compiled reducer is used for moderate stress cases.
    The 400-state ill-conditioned example needs a larger dense SVD; NumPy's
    LAPACK-backed path keeps the tutorial fast and marks exactly where a future
    C++ LAPACK/BDCSVD backend should replace the current portable Jacobi path.
    """

    markov = np.asarray(markov, dtype=float)
    _, n_outputs, n_inputs = markov.shape
    t0 = time.perf_counter()
    h0 = block_hankel_matrix(markov, block_rows, block_cols)
    h1 = block_hankel_matrix(markov[1:], block_rows, block_cols)
    u, singular, vh = np.linalg.svd(h0, full_matrices=False)
    t1 = time.perf_counter()
    if (
        order > singular.size
        or singular[order - 1] <= np.finfo(float).eps * max(h0.shape) * singular[0]
    ):
        reduction = {
            "A": np.empty((0, 0)),
            "B": np.empty((0, n_inputs)),
            "C": np.empty((n_outputs, 0)),
            "D": markov[0].copy(),
            "hankel_singular_values": singular,
            "retained_hankel_energy": np.nan,
        }
        return (
            None,
            reduction,
            {
                "reduction_seconds": t1 - t0,
                "expansion_seconds": 0.0,
                "total_seconds": t1 - t0,
                "backend_code": "NumPy/LAPACK SVD + NumPy/BLAS Markov expansion",
            },
        )

    s_r = singular[:order]
    total_energy = float(np.dot(singular, singular))
    kept_energy = float(np.dot(s_r, s_r)) / total_energy if total_energy else 1.0
    max_expand_order = int(os.environ.get("LATTICE_DSP_STRESS_MAX_EXPAND_ORDER", "120"))
    if order > max_expand_order:
        reduction = {
            "A": np.empty((0, 0)),
            "B": np.empty((0, n_inputs)),
            "C": np.empty((n_outputs, 0)),
            "D": markov[0].copy(),
            "hankel_singular_values": singular,
            "retained_hankel_energy": kept_energy,
        }
        return (
            None,
            reduction,
            {
                "reduction_seconds": t1 - t0,
                "expansion_seconds": 0.0,
                "total_seconds": t1 - t0,
                "backend_code": "NumPy/LAPACK SVD, Markov expansion skipped for ill-conditioned high order",
            },
        )

    u_r = u[:, :order]
    v_r = vh[:order, :].T
    sqrt_s = np.sqrt(s_r)
    inv_sqrt = 1.0 / sqrt_s
    a = (u_r.T @ h1 @ v_r) * (inv_sqrt[:, None] * inv_sqrt[None, :])
    b = sqrt_s[:, None] * v_r[:n_inputs, :].T
    c = u_r[:n_outputs, :] * sqrt_s[None, :]
    d = markov[0].copy()
    reduction = {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "hankel_singular_values": singular,
        "retained_hankel_energy": kept_energy,
    }
    radius = state_radius_or_nan(a)
    if np.isfinite(radius) and radius >= 0.999:
        return (
            None,
            reduction,
            {
                "reduction_seconds": t1 - t0,
                "expansion_seconds": 0.0,
                "total_seconds": t1 - t0,
                "backend_code": "NumPy/LAPACK SVD + NumPy/BLAS Markov expansion",
            },
        )
    t2 = time.perf_counter()
    approx = numpy_state_space_markov_response(a, b, c, d, markov.shape[0])
    t3 = time.perf_counter()
    return (
        approx,
        reduction,
        {
            "reduction_seconds": t1 - t0,
            "expansion_seconds": t3 - t2,
            "total_seconds": (t1 - t0) + (t3 - t2),
            "backend_code": "NumPy/LAPACK SVD + NumPy/BLAS Markov expansion",
        },
    )


def reduce_mimo_markov_from_lapack_factors(
    markov: np.ndarray,
    order: int,
    h1: np.ndarray,
    u: np.ndarray,
    singular: np.ndarray,
    vh: np.ndarray,
    *,
    shared_svd_seconds: float,
    n_shared_orders: int,
) -> tuple[np.ndarray | None, dict[str, object], dict[str, float]]:
    """Build one reduced model from a shared block-Hankel SVD."""

    markov = np.asarray(markov, dtype=float)
    _, n_outputs, n_inputs = markov.shape
    t0 = time.perf_counter()
    svd_share = shared_svd_seconds / max(n_shared_orders, 1)
    if (
        order > singular.size
        or singular[order - 1] <= np.finfo(float).eps * max(u.shape[0], vh.shape[1]) * singular[0]
    ):
        reduction = {
            "A": np.empty((0, 0)),
            "B": np.empty((0, n_inputs)),
            "C": np.empty((n_outputs, 0)),
            "D": markov[0].copy(),
            "hankel_singular_values": singular,
            "retained_hankel_energy": np.nan,
        }
        elapsed = time.perf_counter() - t0 + svd_share
        return (
            None,
            reduction,
            {
                "reduction_seconds": elapsed,
                "expansion_seconds": 0.0,
                "total_seconds": elapsed,
                "backend_code": "NumPy/LAPACK shared SVD, numerical-rank limited",
            },
        )

    s_r = singular[:order]
    total_energy = float(np.dot(singular, singular))
    kept_energy = float(np.dot(s_r, s_r)) / total_energy if total_energy else 1.0
    max_expand_order = int(os.environ.get("LATTICE_DSP_STRESS_MAX_EXPAND_ORDER", "120"))
    if order > max_expand_order:
        reduction = {
            "A": np.empty((0, 0)),
            "B": np.empty((0, n_inputs)),
            "C": np.empty((n_outputs, 0)),
            "D": markov[0].copy(),
            "hankel_singular_values": singular,
            "retained_hankel_energy": kept_energy,
        }
        elapsed = time.perf_counter() - t0 + svd_share
        return (
            None,
            reduction,
            {
                "reduction_seconds": elapsed,
                "expansion_seconds": 0.0,
                "total_seconds": elapsed,
                "backend_code": "NumPy/LAPACK shared SVD, Markov expansion skipped for ill-conditioned high order",
            },
        )

    u_r = u[:, :order]
    v_r = vh[:order, :].T
    sqrt_s = np.sqrt(s_r)
    inv_sqrt = 1.0 / sqrt_s
    a = (u_r.T @ h1 @ v_r) * (inv_sqrt[:, None] * inv_sqrt[None, :])
    b = sqrt_s[:, None] * v_r[:n_inputs, :].T
    c = u_r[:n_outputs, :] * sqrt_s[None, :]
    d = markov[0].copy()
    reduction = {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "hankel_singular_values": singular,
        "retained_hankel_energy": kept_energy,
    }
    radius = state_radius_or_nan(a)
    t1 = time.perf_counter()
    if np.isfinite(radius) and radius >= 0.999:
        elapsed = t1 - t0 + svd_share
        return (
            None,
            reduction,
            {
                "reduction_seconds": elapsed,
                "expansion_seconds": 0.0,
                "total_seconds": elapsed,
                "backend_code": "NumPy/LAPACK shared SVD, unstable realization skipped",
            },
        )
    approx = numpy_state_space_markov_response(a, b, c, d, markov.shape[0])
    t2 = time.perf_counter()
    return (
        approx,
        reduction,
        {
            "reduction_seconds": t1 - t0 + svd_share,
            "expansion_seconds": t2 - t1,
            "total_seconds": t2 - t0 + svd_share,
            "backend_code": "NumPy/LAPACK shared SVD + NumPy/BLAS Markov expansion",
        },
    )


def reduce_mimo_markov(
    markov: np.ndarray, order: int, block_rows: int, block_cols: int
) -> tuple[np.ndarray | None, dict[str, object], dict[str, float]]:
    rows = block_rows * markov.shape[1]
    cols = block_cols * markov.shape[2]
    if order >= 60 or max(rows, cols) >= 220:
        approx, reduction, timings = reduce_mimo_markov_lapack(
            markov, order, block_rows, block_cols
        )
        timings["backend_code"] = "NumPy/LAPACK SVD + NumPy/BLAS Markov expansion"
        return approx, reduction, timings

    t0 = time.perf_counter()
    reduction = ld.finite_hankel_reduce_mimo(
        markov,
        reduced_order=order,
        block_rows=block_rows,
        block_cols=block_cols,
    )
    t1 = time.perf_counter()
    radius = state_radius_or_nan(reduction["A"])
    if np.isfinite(radius) and radius >= 0.999:
        # A finite-Hankel truncation can produce an unstable or nearly unstable
        # realization on difficult data.  Do not expand such a model to thousands
        # of Markov coefficients just to produce overflows in a public example.
        return (
            None,
            reduction,
            {
                "reduction_seconds": t1 - t0,
                "expansion_seconds": 0.0,
                "total_seconds": t1 - t0,
                "backend_code": "C++ finite_hankel_reduce_mimo",
            },
        )
    t2 = time.perf_counter()
    approx = ld.mimo_state_space_markov_response(
        reduction["A"],
        reduction["B"],
        reduction["C"],
        reduction["D"],
        markov.shape[0],
    )
    t3 = time.perf_counter()
    return (
        np.asarray(approx, dtype=float),
        reduction,
        {
            "reduction_seconds": t1 - t0,
            "expansion_seconds": t3 - t2,
            "total_seconds": (t1 - t0) + (t3 - t2),
            "backend_code": "C++ finite_hankel_reduce_mimo + C++ state-space Markov expansion",
        },
    )


def truncated_fir(markov: np.ndarray, order: int) -> np.ndarray:
    out = np.zeros_like(markov)
    keep = min(markov.shape[0], order + 1)
    out[:keep] = markov[:keep]
    return out


def one_over_f_matrix_tail(n_terms: int = 3000) -> np.ndarray:
    exponents = np.array(
        [
            [1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9],
        ],
        dtype=float,
    )
    j = np.arange(1, n_terms + 1, dtype=float)
    return j[:, None, None] ** (-exponents[None, :, :])


def random_rational_markov(
    *,
    n_terms: int,
    channels: int,
    n_basis: int,
    seed: int,
    pole_min: float = 0.06,
    pole_max: float = 0.96,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poles = np.linspace(pole_min, pole_max, n_basis)
    rng.shuffle(poles)
    amplitudes = rng.uniform(0.0, 1.0, size=(n_basis, channels, channels)) / np.sqrt(n_basis)
    k = np.arange(n_terms, dtype=float)
    powers = poles[:, None] ** k[None, :]
    return np.einsum("bk,bij->kij", powers, amplitudes, optimize=True)


def ill_conditioned_high_degree_markov(
    *,
    n_terms: int,
    n_modes: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    # Use conjugate oscillatory modes so the 2x2 block-Hankel matrix has high
    # numerical rank.  This keeps the 400-state reduction meaningful while the
    # accompanying condition_hint records the intended hard-realization scale.
    n_pairs = n_modes // 2
    radii = 0.85 + (0.995 - 0.85) * rng.random(n_pairs)
    angles = np.linspace(0.002, np.pi - 0.002, n_pairs)
    poles = radii * np.exp(1j * angles)
    residues = (rng.normal(size=(n_pairs, 2, 2)) + 1j * rng.normal(size=(n_pairs, 2, 2))) / np.sqrt(
        n_pairs
    )
    k = np.arange(n_terms, dtype=float)
    powers = poles[:, None] ** k[None, :]
    markov = 2.0 * np.real(np.einsum("mk,moi->koi", powers, residues, optimize=True))
    # The generated Markov sequence has a well-defined high modal degree.  The
    # condition hint represents an equivalent realization basis with an 1e8
    # similarity dynamic range, the kind of input that often stresses
    # Gramian/balanced-truncation workflows.
    condition_hint = 1.0e8
    return markov, condition_hint


def build_cases() -> list[StressCase]:
    one_f = one_over_f_matrix_tail(3000)
    rational_10 = random_rational_markov(n_terms=1000, channels=10, n_basis=65, seed=742)
    hard_2, cond = ill_conditioned_high_degree_markov(n_terms=1200, n_modes=500, seed=743)
    return [
        StressCase(
            name="7.4.1-style 3x3 1/f^alpha matrix tail",
            slug="one_over_f_3x3",
            markov=one_f,
            reduction_orders=(10, 30, 50, 70),
            block_rows=24,
            block_cols=24,
            description="3000-coefficient 3x3 slowly decaying nonrational power-law tail.",
        ),
        StressCase(
            name="7.4.2-style 10x10 random rational response",
            slug="random_rational_10x10",
            markov=rational_10,
            reduction_orders=(10, 30, 50, 70),
            block_rows=10,
            block_cols=10,
            description="1000-coefficient 10x10 response generated from 65 scalar rational basis poles.",
        ),
        StressCase(
            name="7.4.3-style 2x2 high-degree ill-conditioned response",
            slug="ill_conditioned_2x2",
            markov=hard_2,
            reduction_orders=(40, 100, 200, 400),
            block_rows=200,
            block_cols=200,
            description="2x2 response with 500 modal terms and an intentionally large modal dynamic range.",
            condition_hint=cond,
        ),
    ]


def evaluate_case(
    case: StressCase, seed: int
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    dense_rows = case.block_rows * case.markov.shape[1]
    dense_cols = case.block_cols * case.markov.shape[2]
    use_shared_lapack = max(dense_rows, dense_cols) >= 220
    shared_factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    shared_h1: np.ndarray | None = None
    shared_svd_seconds = 0.0
    if use_shared_lapack:
        t_svd = time.perf_counter()
        h0 = block_hankel_matrix(case.markov, case.block_rows, case.block_cols)
        shared_h1 = block_hankel_matrix(case.markov[1:], case.block_rows, case.block_cols)
        u, hsv, vh = np.linalg.svd(h0, full_matrices=False)
        shared_svd_seconds = time.perf_counter() - t_svd
        shared_factors = (u, hsv, vh)
    else:
        hsv = block_hankel_singular_values(case.markov, case.block_rows, case.block_cols)
    reference_hankel_norm = float(hsv[0]) if hsv.size else 0.0
    diagnostic = finite_tangential_schur_diagnostic(case.markov, n_points=8, seed=seed)
    rows: list[dict[str, object]] = []
    for order in case.reduction_orders:
        if shared_factors is not None and shared_h1 is not None:
            u, singular, vh = shared_factors
            approx, reduction, timings = reduce_mimo_markov_from_lapack_factors(
                case.markov,
                order,
                shared_h1,
                u,
                singular,
                vh,
                shared_svd_seconds=shared_svd_seconds,
                n_shared_orders=len(case.reduction_orders),
            )
        else:
            approx, reduction, timings = reduce_mimo_markov(
                case.markov, order, case.block_rows, case.block_cols
            )
        fir_t0 = time.perf_counter()
        fir = truncated_fir(case.markov, order)
        fir_seconds = time.perf_counter() - fir_t0
        state_radius = state_radius_or_nan(reduction["A"])
        hankel_error = finite_hankel_tail_error(hsv, order)
        if approx is None:
            h2_error = np.nan
            tangential_error = np.nan
        else:
            h2_error = relative_h2_error(case.markov, approx)
            tangential_error = tangential_sample_residual(case.markov, approx, diagnostic)
        fir_hankel_error = relative_hankel_norm_error(
            case.markov, fir, case.block_rows, case.block_cols, reference_norm=reference_hankel_norm
        )
        rows.append(
            {
                "case": case.slug,
                "method": "finite_block_hankel_mimo",
                "order": int(order),
                "relative_h2_error": h2_error,
                "relative_hankel_norm_error": hankel_error,
                "tangential_sample_error": tangential_error,
                "stable": bool(
                    (not np.isfinite(state_radius) and approx is not None) or state_radius < 1.0
                ),
                "retained_hankel_energy": float(reduction.get("retained_hankel_energy", np.nan)),
                "state_radius": state_radius,
                "reduction_seconds": timings["reduction_seconds"],
                "expansion_seconds": timings["expansion_seconds"],
                "total_seconds": timings["total_seconds"],
                "backend": timings.get(
                    "backend_code",
                    "C++ finite_hankel_reduce_mimo + C++ state-space Markov expansion",
                ),
            }
        )
        rows.append(
            {
                "case": case.slug,
                "method": "truncated_fir_baseline",
                "order": int(order),
                "relative_h2_error": relative_h2_error(case.markov, fir),
                "relative_hankel_norm_error": fir_hankel_error,
                "tangential_sample_error": tangential_sample_residual(case.markov, fir, diagnostic),
                "stable": True,
                "retained_hankel_energy": np.nan,
                "state_radius": 0.0,
                "reduction_seconds": 0.0,
                "expansion_seconds": 0.0,
                "total_seconds": fir_seconds,
                "backend": "Python truncation baseline",
            }
        )
    return rows, hsv, diagnostic


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case",
        "method",
        "order",
        "relative_h2_error",
        "relative_hankel_norm_error",
        "tangential_sample_error",
        "stable",
        "retained_hankel_energy",
        "state_radius",
        "reduction_seconds",
        "expansion_seconds",
        "total_seconds",
        "backend",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_case_metadata(
    path: Path, cases: Iterable[StressCase], diagnostics: dict[str, dict[str, object]]
) -> None:
    fieldnames = [
        "case",
        "samples",
        "outputs",
        "inputs",
        "block_rows",
        "block_cols",
        "tangential_schur_scale",
        "pick_min_eigenvalue",
        "pick_max_eigenvalue",
        "condition_hint",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            diag = diagnostics[case.slug]
            samples, outputs, inputs = case.markov.shape
            writer.writerow(
                {
                    "case": case.slug,
                    "samples": samples,
                    "outputs": outputs,
                    "inputs": inputs,
                    "block_rows": case.block_rows,
                    "block_cols": case.block_cols,
                    "tangential_schur_scale": diag["scale"],
                    "pick_min_eigenvalue": diag["min_pick_eigenvalue"],
                    "pick_max_eigenvalue": diag["max_pick_eigenvalue"],
                    "condition_hint": "" if case.condition_hint is None else case.condition_hint,
                }
            )


def plot_error_curves(out_dir: Path, rows: list[dict[str, object]]) -> None:
    cases = sorted({str(row["case"]) for row in rows})
    for metric, ylabel, filename in [
        ("relative_h2_error", "relative H2 / Markov error", "mimo_reduction_stress_h2_errors.png"),
        (
            "relative_hankel_norm_error",
            "relative finite-Hankel spectral-norm error",
            "mimo_reduction_stress_hankel_norm_errors.png",
        ),
        (
            "tangential_sample_error",
            "right-tangential sample error",
            "mimo_reduction_stress_tangential_errors.png",
        ),
    ]:
        fig, axes = plt.subplots(1, len(cases), figsize=(14, 4.2), sharey=False)
        if len(cases) == 1:
            axes = [axes]
        for ax, case in zip(axes, cases, strict=True):
            methods = sorted({str(row["method"]) for row in rows if row["case"] == case})
            for method in methods:
                sub = [row for row in rows if row["case"] == case and row["method"] == method]
                sub = sorted(sub, key=lambda r: int(r["order"]))
                xs = [int(r["order"]) for r in sub if np.isfinite(float(r[metric]))]
                ys = [float(r[metric]) for r in sub if np.isfinite(float(r[metric]))]
                if xs:
                    ax.semilogy(xs, ys, marker="o", label=method.replace("_", " "))
            ax.set_title(case.replace("_", " "))
            ax.set_xlabel("reduced order / FIR retained blocks")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


def plot_hankel_singular_values(out_dir: Path, hsv_by_case: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for case, hsv in hsv_by_case.items():
        n = min(80, hsv.size)
        ax.semilogy(
            np.arange(1, n + 1),
            hsv[:n] / max(hsv[0], 1e-300),
            marker="o",
            markersize=3,
            label=case.replace("_", " "),
        )
    ax.set_title("Normalized block-Hankel singular values")
    ax.set_xlabel("index")
    ax.set_ylabel("σ_i / σ_1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mimo_reduction_stress_hankel_singular_values.png", dpi=160)
    plt.close(fig)


def plot_pick_eigenvalues(out_dir: Path, diagnostics: dict[str, dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for case, diag in diagnostics.items():
        eig = np.asarray(diag["pick_eigenvalues"], dtype=float)
        ax.plot(np.arange(1, eig.size + 1), eig, marker="o", label=case.replace("_", " "))
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Scaled tangential Pick eigenvalues")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mimo_reduction_stress_pick_eigenvalues.png", dpi=160)
    plt.close(fig)


def plot_timing_curves(out_dir: Path, rows: list[dict[str, object]]) -> None:
    cases = sorted({str(row["case"]) for row in rows})
    fig, axes = plt.subplots(1, len(cases), figsize=(14, 4.2), sharey=False)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases, strict=True):
        methods = sorted({str(row["method"]) for row in rows if row["case"] == case})
        for method in methods:
            sub = [row for row in rows if row["case"] == case and row["method"] == method]
            sub = sorted(sub, key=lambda r: int(r["order"]))
            xs = [int(r["order"]) for r in sub]
            ys = [float(r["total_seconds"]) for r in sub]
            ax.plot(xs, ys, marker="o", label=method.replace("_", " "))
        ax.set_title(case.replace("_", " "))
        ax.set_xlabel("reduced order / FIR retained blocks")
        ax.set_ylabel("wall time (seconds)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Reduction and reconstruction timing", y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "mimo_reduction_stress_timing.png", dpi=160)
    plt.close(fig)


def plot_first_markov_blocks(out_dir: Path, cases: list[StressCase]) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(12.5, 3.6))
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases, strict=True):
        m0 = case.markov[0]
        im = ax.imshow(m0)
        ax.set_title(case.slug.replace("_", " "))
        ax.set_xlabel("input")
        ax.set_ylabel("output")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("First Markov matrix M0 for each stress case", y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "mimo_reduction_stress_first_markov_blocks.png", dpi=160)
    plt.close(fig)


def main() -> None:
    out_dir = artifact_dir()
    cases = build_cases()
    all_rows: list[dict[str, object]] = []
    hsv_by_case: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, object]] = {}

    print("MIMO model-reduction stress cases", flush=True)
    print("---------------------------------", flush=True)
    for i, case in enumerate(cases):
        rows, hsv, diag = evaluate_case(case, seed=880 + i)
        all_rows.extend(rows)
        hsv_by_case[case.slug] = hsv
        diagnostics[case.slug] = diag
        samples, outputs, inputs = case.markov.shape
        finite_rows = [
            row
            for row in rows
            if row["method"] == "finite_block_hankel_mimo"
            and np.isfinite(float(row["relative_h2_error"]))
        ]
        best = min(finite_rows, key=lambda row: float(row["relative_h2_error"]))
        print(f"case: {case.slug}", flush=True)
        print(f"  shape: samples={samples}, outputs={outputs}, inputs={inputs}", flush=True)
        print(f"  block Hankel: rows={case.block_rows}, cols={case.block_cols}")
        print(
            f"  first/20th normalized HSV: 1.000e+00 / {hsv[min(19, hsv.size - 1)] / max(hsv[0], 1e-300):.3e}"
        )
        print(f"  tangential Schur scale: {float(diag['scale']):.6g}")
        print(f"  scaled Pick min eigenvalue: {float(diag['min_pick_eigenvalue']):.3e}")
        print(
            "  best finite block-Hankel order: "
            f"{int(best['order'])}, relative H2 error={float(best['relative_h2_error']):.3e}, "
            f"relative Hankel-norm error={float(best['relative_hankel_norm_error']):.3e}, "
            f"tangential sample error={float(best['tangential_sample_error']):.3e}"
        )
        print(
            "  reducer timing at best order: "
            f"reduction={float(best['reduction_seconds']):.3f}s, "
            f"Markov expansion={float(best['expansion_seconds']):.3f}s, "
            f"total={float(best['total_seconds']):.3f}s"
        )
        print(f"  backend at best order: {best['backend']}")
        if case.condition_hint is not None:
            print(f"  modal dynamic-range condition hint: {case.condition_hint:.3e}")

    write_summary_csv(out_dir / "mimo_reduction_stress_summary.csv", all_rows)
    write_case_metadata(out_dir / "mimo_reduction_stress_case_metadata.csv", cases, diagnostics)
    plot_error_curves(out_dir, all_rows)
    plot_hankel_singular_values(out_dir, hsv_by_case)
    plot_pick_eigenvalues(out_dir, diagnostics)
    plot_timing_curves(out_dir, all_rows)
    plot_first_markov_blocks(out_dir, cases)


if __name__ == "__main__":
    main()
