"""Tangential Schur Pick, RKHS, and Potapov--Blaschke diagnostics.

This example is intentionally small but mathematically explicit.  It checks
finite right-tangential Schur data, the Pick/RKHS Gram matrix, a known constant
Schur solution, and elementary Potapov--Blaschke J-inner factors.

The factors here are interpolation-side J-inner objects.  They are related to
lossless/all-pass matrix-lattice systems, but this example is not the full
recursive tangential-Schur manifold algorithm used in the
Hanzon--Olivi--Peeters/Marmorat literature.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def _artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figures(pick_eigs: np.ndarray, omega: np.ndarray, residuals: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(np.arange(1, pick_eigs.size + 1), pick_eigs, marker="o")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("Pick eigenvalue")
    ax.set_title("Definite tangential Pick matrix is positive semidefinite")
    fig.tight_layout()
    fig.savefig(out_dir / "tangential_pick_eigenvalues.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.semilogy(omega, np.maximum(residuals, 1e-18))
    ax.set_xlabel("rad/sample")
    ax.set_ylabel(r"$\|\Theta^*J\Theta-J\|_2$")
    ax.set_title("Potapov product is J-inner on the unit circle")
    fig.tight_layout()
    fig.savefig(out_dir / "potapov_j_inner_residual.png", dpi=160)
    plt.close(fig)


rng = np.random.default_rng(52)
output_dim = 2
input_dim = 2
points = np.array([0.0, 0.22 - 0.10j, -0.30 + 0.08j, 0.15j])
raw = rng.normal(size=(output_dim, input_dim)) + 1j * rng.normal(size=(output_dim, input_dim))
known_schur = 0.45 * raw / np.linalg.svd(raw, compute_uv=False)[0]
directions = rng.normal(size=(points.size, input_dim)) + 1j * rng.normal(
    size=(points.size, input_dim)
)
values = np.einsum("oi,ni->no", known_schur, directions)

data = ld.RightTangentialSchurData(points, directions, values)
pick = ld.right_tangential_pick_matrix(data)
pick_eigs = ld.pick_matrix_eigenvalues(pick)
constant_solution = ld.constant_schur_solution(data)
residual = ld.max_tangential_residual(data, constant_solution)
product = ld.potapov_product_from_rank_one_data(data)
omega = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
theta = product.evaluate(np.exp(1j * omega))
j_residual = ld.j_unitarity_residual(theta, product.j)

print("points:", data.n_points)
print("input dimension:", data.input_dim)
print("output dimension:", data.output_dim)
print("total tangential conditions:", data.total_conditions)
print("known Schur spectral norm:", f"{np.linalg.svd(known_schur, compute_uv=False)[0]:.6f}")
print("Pick min eigenvalue:", f"{pick_eigs[0]:.6e}")
print("Pick condition number:", f"{np.linalg.cond(pick):.6e}")
print("definite Pick test says feasible:", ld.is_tangential_schur_solvable(data))
print("constant-solution interpolation residual:", f"{residual:.3e}")
print("Potapov product J-inner max residual:", f"{np.max(j_residual):.3e}")
print("first factor annihilation residual:", f"{product.factors[0].annihilation_residual():.3e}")
print("scope: finite definite Pick/RKHS/J-inner diagnostics")
print("not implemented here: full recursive tangential-Schur manifold parametrization")

_save_figures(pick_eigs, omega, j_residual)
