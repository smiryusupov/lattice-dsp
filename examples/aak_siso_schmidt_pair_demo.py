"""Tutorial: Schmidt-pair diagnostics for finite SISO AAK/Nehari intuition.

The finite Nehari tutorials introduced the Hankel matrix of an anticausal tail
and a rational bridge from a Hankelized approximation to a recursive model.
This tutorial focuses on the object that makes AAK theory more than ordinary
matrix approximation: the singular vector pair, or finite Schmidt pair, at the
first neglected Hankel singular value.

For a finite Hankel matrix ``H`` and a target rank ``r``, the singular pair
``(u_{r+1}, v_{r+1})`` satisfies approximately

    H v_{r+1} = sigma_{r+1} u_{r+1},
    H.T u_{r+1} = sigma_{r+1} v_{r+1}.

The singular value ``sigma_{r+1}`` is the finite Eckart--Young barrier for
unconstrained rank-r approximation.  In the infinite-dimensional scalar AAK
story, analogous Schmidt vectors have rational/inner-outer structure and are
used to construct an optimal rational approximant.  This example is deliberately
finite-dimensional: it visualizes the critical singular direction and compares
it with the finite Nehari and rational-bridge approximations.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np

try:
    from examples.finite_nehari_rational_bridge import (
        fit_rational_tail,
        rational_tail_response,
        relative_error,
        synthetic_anticausal_tail,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution from examples/
    from finite_nehari_rational_bridge import (
        fit_rational_tail,
        rational_tail_response,
        relative_error,
        synthetic_anticausal_tail,
    )


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def hankel_from_tail(tail: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Build ``H[i, j] = tail[i + j]`` from a one-sided anticausal tail."""

    tail = np.asarray(tail, dtype=float)
    if tail.ndim != 1:
        raise ValueError("tail must be one-dimensional")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if tail.size < rows + cols - 1:
        raise ValueError("tail is too short for the requested Hankel matrix")
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]
    return tail[i + j]


def svd_rank_approximation(
    hankel: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return SVD pieces and the unconstrained rank-r SVD approximation."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    u, s, vh = np.linalg.svd(hankel, full_matrices=False)
    if rank > s.size:
        raise ValueError("rank cannot exceed the numerical SVD dimension")
    if rank == 0:
        approx = np.zeros_like(hankel)
    else:
        approx = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    return u, s, vh, approx


def schmidt_pair_diagnostics(
    tail: np.ndarray, *, rows: int, cols: int, rank: int
) -> dict[str, object]:
    """Compute finite Schmidt-pair diagnostics for the first neglected mode."""

    hankel = hankel_from_tail(tail, rows, cols)
    u, s, vh, approx = svd_rank_approximation(hankel, rank)
    if rank >= s.size:
        raise ValueError("rank must leave at least one neglected singular value")

    sigma = float(s[rank])
    left = u[:, rank].copy()
    right = vh[rank, :].copy()

    # Stabilize the sign for tutorial plots.
    pivot = np.argmax(np.abs(right))
    if right[pivot] < 0:
        left *= -1.0
        right *= -1.0

    residual_left = float(np.linalg.norm(hankel @ right - sigma * left))
    residual_right = float(np.linalg.norm(hankel.T @ left - sigma * right))
    rank_error = float(np.linalg.norm(hankel - approx, ord=2))

    return {
        "hankel": hankel,
        "singular_values": s,
        "critical_sigma": sigma,
        "left_schmidt_vector": left,
        "right_schmidt_vector": right,
        "left_residual": residual_left,
        "right_residual": residual_right,
        "rank_svd_error": rank_error,
        "rank_approximation": approx,
    }


def write_summary(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import lattice_dsp as ld

    out_dir = artifact_dir()
    rows = cols = 48
    target_rank = 3
    tail = synthetic_anticausal_tail(rows + cols - 1)

    diag = schmidt_pair_diagnostics(tail, rows=rows, cols=cols, rank=target_rank)
    singular_values = np.asarray(diag["singular_values"], dtype=float)
    left = np.asarray(diag["left_schmidt_vector"], dtype=float)
    right = np.asarray(diag["right_schmidt_vector"], dtype=float)

    ranks = [2, 3, 4]
    summary: list[dict[str, float | int]] = []
    tail_approximations: dict[int, np.ndarray] = {}
    rational_responses: dict[int, np.ndarray] = {}
    fitted_poles: dict[int, np.ndarray] = {}

    for rank in ranks:
        nehari = ld.finite_nehari_approximate_tail(tail.tolist(), rank=rank, rows=rows, cols=cols)
        approx_tail = np.asarray(nehari["approximated_tail"], dtype=float)
        denominator, numerator, poles = fit_rational_tail(approx_tail, rank)
        rational = rational_tail_response(denominator, numerator, tail.size)

        tail_approximations[rank] = approx_tail
        rational_responses[rank] = rational
        fitted_poles[rank] = poles

        summary.append(
            {
                "rank": rank,
                "sigma_next": float(nehari["sigma_next"]),
                "finite_nehari_tail_relative_error": float(nehari["relative_tail_error"]),
                "rational_vs_original_relative_error": relative_error(tail, rational),
                "rational_vs_hankelized_relative_error": relative_error(approx_tail, rational),
                "max_pole_radius": float(np.max(np.abs(poles))) if poles.size else 0.0,
            }
        )

    summary_path = out_dir / "aak_siso_schmidt_pair_summary.csv"
    write_summary(summary_path, summary)

    print("finite Hankel matrix:", f"{rows} x {cols}")
    print("target rank:", target_rank)
    print("leading singular values:", [round(float(v), 6) for v in singular_values[:8]])
    print("critical sigma:", f"{float(diag['critical_sigma']):.6e}")
    print("rank-r SVD error:", f"{float(diag['rank_svd_error']):.6e}")
    print("left Schmidt residual:", f"{float(diag['left_residual']):.3e}")
    print("right Schmidt residual:", f"{float(diag['right_residual']):.3e}")
    for row in summary:
        print(
            "rank={rank}: sigma_next={sigma_next:.3e}, tail_error={finite_nehari_tail_relative_error:.3e}, "
            "rational_error={rational_vs_original_relative_error:.3e}, pole_radius={max_pole_radius:.4f}".format(
                **row
            )
        )
    print(f"wrote {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    idx = np.arange(1, 21)
    ax.semilogy(idx, singular_values[:20], marker="o", label="Hankel singular values")
    ax.scatter(
        [target_rank + 1], [diag["critical_sigma"]], marker="s", s=80, label="first neglected sigma"
    )
    ax.set_title("Finite Schmidt-pair diagnostic: first neglected Hankel mode")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "aak_schmidt_singular_values.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

    fig2, ax2 = plt.subplots(figsize=(9.0, 4.8))
    ax2.plot(np.arange(left.size), left, marker="o", markersize=3, label="left vector u")
    ax2.plot(np.arange(right.size), right, marker="s", markersize=3, label="right vector v")
    ax2.set_title("Critical finite Schmidt pair at sigma_{r+1}")
    ax2.set_xlabel("coefficient index")
    ax2.set_ylabel("singular-vector component")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    path2 = out_dir / "aak_schmidt_pair_vectors.png"
    fig2.savefig(path2, dpi=160)
    print(f"wrote {path2}")

    fig3, ax3 = plt.subplots(figsize=(9.0, 4.8))
    n = np.arange(1, 41)
    ax3.plot(n, tail[:40], linewidth=2.0, label="target tail")
    for rank in ranks:
        ax3.plot(
            n, tail_approximations[rank][:40], "--", linewidth=1.2, label=f"rank {rank} Hankelized"
        )
        ax3.plot(
            n, rational_responses[rank][:40], ":", linewidth=1.4, label=f"rank {rank} rational"
        )
    ax3.set_title("Schmidt-pair context: tail, Hankelized approximations, rational fits")
    ax3.set_xlabel("tail coefficient index")
    ax3.set_ylabel("coefficient value")
    ax3.grid(True, alpha=0.3)
    ax3.legend(ncol=2, fontsize=8)
    fig3.tight_layout()
    path3 = out_dir / "aak_schmidt_tail_and_rational_fit.png"
    fig3.savefig(path3, dpi=160)
    print(f"wrote {path3}")

    fig4, ax4 = plt.subplots(figsize=(5.6, 5.6))
    theta = np.linspace(0.0, 2.0 * math.pi, 512)
    ax4.plot(np.cos(theta), np.sin(theta), "--", linewidth=1.0, label="unit circle")
    for rank, poles in fitted_poles.items():
        ax4.scatter(np.real(poles), np.imag(poles), label=f"rank {rank} poles")
    ax4.axhline(0.0, linewidth=0.8)
    ax4.axvline(0.0, linewidth=0.8)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_title("Poles of rational fits from Hankelized tails")
    ax4.set_xlabel("real")
    ax4.set_ylabel("imaginary")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    fig4.tight_layout()
    path4 = out_dir / "aak_schmidt_rational_poles.png"
    fig4.savefig(path4, dpi=160)
    print(f"wrote {path4}")


if __name__ == "__main__":
    main()
