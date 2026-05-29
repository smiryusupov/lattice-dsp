"""Tutorial: finite Nehari/AAK intuition from a SISO Hankel matrix.

This is not a production Nehari or AAK solver.  It is a deliberately small
teaching example that shows the object those theories act on: a Hankel
operator built from an anticausal or future impulse-response tail.

For a finite Hankel matrix, the best unconstrained rank-r approximation error
in spectral norm is the next singular value.  AAK/Nehari theory is the deeper
infinite-dimensional rational/Hankel-structured version of this story.  This
example makes that bridge visible before implementing exact Nehari/AAK routines.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def anticausal_tail(n_terms: int) -> np.ndarray:
    """Return a synthetic anticausal tail gamma_1, gamma_2, ... .

    A sum of damped exponentials gives a low effective Hankel rank with a few
    weaker modes.  This keeps the tutorial readable: the singular values decay,
    so users can see why a low-order rational approximation may be plausible.
    """

    n = np.arange(n_terms, dtype=float)
    return 1.00 * 0.92**n + 0.30 * 0.63**n - 0.16 * (-0.42) ** n + 0.045 * 0.20**n


def hankel_from_tail(gamma: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if gamma.size < rows + cols - 1:
        raise ValueError("gamma is too short for the requested Hankel matrix")
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]
    return gamma[i + j]


def hankelize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a matrix onto the finite Hankel subspace by anti-diagonal averaging."""

    rows, cols = matrix.shape
    values = np.zeros(rows + cols - 1, dtype=float)
    counts = np.zeros(rows + cols - 1, dtype=float)
    for i in range(rows):
        for j in range(cols):
            values[i + j] += matrix[i, j]
            counts[i + j] += 1.0
    values /= counts
    return values, hankel_from_tail(values, rows, cols)


def write_summary(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = artifact_dir()

    rows = cols = 40
    gamma = anticausal_tail(rows + cols - 1)
    ranks = [1, 2, 3, 4, 6]
    summary: list[dict[str, float | int]] = []
    tail_approximations: dict[int, np.ndarray] = {}
    singular_values: np.ndarray | None = None

    for rank in ranks:
        result = ld.finite_nehari_approximate_tail(
            gamma.tolist(),
            rank=rank,
            rows=rows,
            cols=cols,
        )
        if singular_values is None:
            singular_values = np.asarray(result["hankel_singular_values"], dtype=float)
        tail_approximations[rank] = np.asarray(result["approximated_tail"], dtype=float)

        summary.append(
            {
                "rank": rank,
                "sigma_next": float(result["sigma_next"]),
                "unconstrained_svd_error": float(result["unconstrained_hankel_error"]),
                "hankelized_error": float(result["hankelized_hankel_error"]),
                "tail_relative_error": float(result["relative_tail_error"]),
            }
        )

    assert singular_values is not None

    csv_path = out_dir / "nehari_aak_siso_toy_summary.csv"
    write_summary(csv_path, summary)

    print("finite Hankel matrix:", f"{rows} x {cols}")
    print("leading singular values:", [round(float(v), 6) for v in singular_values[:8]])
    print("finite Eckart-Young check: ||H-H_r||_2 should equal sigma_{r+1}")
    for row in summary:
        print(
            "rank={rank}: sigma_next={sigma_next:.6e}, svd_error={unconstrained_svd_error:.6e}, "
            "hankelized_error={hankelized_error:.6e}, tail_rel_error={tail_relative_error:.3e}".format(
                **row
            )
        )
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    idx = np.arange(1, 21)
    ax.semilogy(idx, singular_values[:20], marker="o")
    ax.set_title("Finite Hankel singular values for the Nehari/AAK toy problem")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "nehari_aak_toy_singular_values.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {fig_path}")

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.5))
    rank_axis = [row["rank"] for row in summary]
    ax2.semilogy(rank_axis, [row["sigma_next"] for row in summary], marker="o", label="sigma_{r+1}")
    ax2.semilogy(
        rank_axis,
        [row["unconstrained_svd_error"] for row in summary],
        marker="s",
        label="||H-H_r||_2",
    )
    ax2.semilogy(
        rank_axis,
        [row["hankelized_error"] for row in summary],
        marker="^",
        label="Hankelized error",
    )
    ax2.set_title("Finite low-rank error versus Hankelized approximation error")
    ax2.set_xlabel("rank r")
    ax2.set_ylabel("spectral-norm error")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2_path = out_dir / "nehari_aak_toy_error_curve.png"
    fig2.savefig(fig2_path, dpi=160)
    print(f"wrote {fig2_path}")

    fig3, ax3 = plt.subplots(figsize=(8.8, 4.8))
    n = np.arange(1, gamma.size + 1)
    ax3.plot(n, gamma, linewidth=2.0, label="target tail")
    for rank in [1, 2, 4, 6]:
        ax3.plot(n, tail_approximations[rank], "--", linewidth=1.2, label=f"rank {rank} Hankelized")
    ax3.set_title("Anticausal tail and finite Hankelized approximations")
    ax3.set_xlabel("tail coefficient index")
    ax3.set_ylabel("coefficient value")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3_path = out_dir / "nehari_aak_toy_tail_approximations.png"
    fig3.savefig(fig3_path, dpi=160)
    print(f"wrote {fig3_path}")


if __name__ == "__main__":
    main()
