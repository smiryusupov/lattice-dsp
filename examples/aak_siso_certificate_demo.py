"""Tutorial: a finite-section SISO AAK/Nehari certificate.

This example is the first deliberately AAK/Nehari-shaped diagnostic in the
package.  It does not merely fit a recurrence.  It builds a finite Hankel
matrix, computes the Schmidt pair associated with the first neglected singular
value, checks the finite AAK identities, and then attaches the finite
Nehari/rational candidate for the same rank.

The example uses an exact rank-3 rational tail, so the rank-3 candidate should
be selected, recover the true poles, and have tiny tail/rational error.  This is
still a finite-section prototype, not a full infinite-dimensional Hardy-space
AAK/Nehari solver.
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


def exact_rational_tail(n_terms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_poles = np.array([-0.42, 0.18, 0.76], dtype=float)
    true_weights = np.array([1.25, -0.7, 0.4], dtype=float)
    n = np.arange(n_terms, dtype=float)
    tail = sum(weight * pole**n for weight, pole in zip(true_weights, true_poles, strict=True))
    return np.asarray(tail, dtype=float), true_poles, true_weights


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = artifact_dir()
    rows = cols = 48
    rank = 3
    tail, true_poles, true_weights = exact_rational_tail(rows + cols - 1)
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1e-8,
        max_rational_error=1e-8,
        max_pole_radius=0.99,
    )

    certificate = ld.finite_aak_siso_certificate(
        tail, rank=rank, rows=rows, cols=cols, criteria=criteria
    )
    candidate = certificate["candidate"]
    selected_poles = np.asarray(candidate["poles"])

    summary = [
        {
            "rank": certificate["rank"],
            "sigma_next": certificate["sigma_next"],
            "rank_svd_error": certificate["rank_svd_error"],
            "schmidt_left_residual": certificate["schmidt_left_residual"],
            "schmidt_right_residual": certificate["schmidt_right_residual"],
            "hankelized_tail_error": candidate["hankelized_tail_error"],
            "rational_error": candidate["rational_error"],
            "max_pole_radius": candidate["max_pole_radius"],
            "accepted": candidate["accepted"],
        }
    ]
    summary_path = out_dir / "aak_siso_certificate_summary.csv"
    write_summary(summary_path, summary)

    print("finite Hankel matrix:", f"{rows} x {cols}")
    print("target rank:", rank)
    print("true poles:", np.round(true_poles, 8).tolist())
    print("true weights:", np.round(true_weights, 8).tolist())
    print(
        "leading singular values:", np.round(certificate["hankel_singular_values"][:8], 8).tolist()
    )
    print("sigma_next:", f"{certificate['sigma_next']:.6e}")
    print("rank-r SVD error:", f"{certificate['rank_svd_error']:.6e}")
    print("left Schmidt residual:", f"{certificate['schmidt_left_residual']:.3e}")
    print("right Schmidt residual:", f"{certificate['schmidt_right_residual']:.3e}")
    print("candidate tail error:", f"{candidate['hankelized_tail_error']:.3e}")
    print("candidate rational error:", f"{candidate['rational_error']:.3e}")
    print("candidate pole radius:", f"{candidate['max_pole_radius']:.4f}")
    print("candidate accepted:", candidate["accepted"])
    print("recovered poles:", np.round(np.sort(np.real(selected_poles)), 8).tolist())
    print(f"wrote {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    singular_values = np.asarray(certificate["hankel_singular_values"], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.semilogy(np.arange(1, 13), singular_values[:12], marker="o")
    ax.axvline(rank + 1, linestyle="--", linewidth=1.0, label="first neglected singular value")
    ax.set_title("Finite AAK/Nehari certificate: Hankel singular values")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "aak_certificate_singular_values.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

    fig2, ax2 = plt.subplots(figsize=(8.8, 4.8))
    ax2.plot(certificate["left_schmidt_vector"], marker="o", label="left Schmidt vector")
    ax2.plot(certificate["right_schmidt_vector"], marker="s", label="right Schmidt vector")
    ax2.set_title("Critical finite Schmidt pair")
    ax2.set_xlabel("coefficient index")
    ax2.set_ylabel("amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    path2 = out_dir / "aak_certificate_schmidt_pair.png"
    fig2.savefig(path2, dpi=160)
    print(f"wrote {path2}")

    rational_tail = np.asarray(candidate["rational_tail"], dtype=float)
    fig3, ax3 = plt.subplots(figsize=(9.2, 4.8))
    n = np.arange(tail.size)
    ax3.plot(n, tail, label="true tail", linewidth=2.0)
    ax3.plot(n, rational_tail, "--", label="rank-3 rational candidate")
    ax3.set_title("Exact rank-3 rational tail recovery")
    ax3.set_xlabel("tail index")
    ax3.set_ylabel("coefficient")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    path3 = out_dir / "aak_certificate_tail_recovery.png"
    fig3.savefig(path3, dpi=160)
    print(f"wrote {path3}")

    fig4, ax4 = plt.subplots(figsize=(5.8, 5.8))
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--")
    ax4.add_artist(circle)
    ax4.scatter(np.real(true_poles), np.imag(true_poles), marker="o", label="true poles")
    ax4.scatter(
        np.real(selected_poles), np.imag(selected_poles), marker="x", label="recovered poles"
    )
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlim(-1.1, 1.1)
    ax4.set_ylim(-1.1, 1.1)
    ax4.set_xlabel("real")
    ax4.set_ylabel("imaginary")
    ax4.set_title("Stable rational candidate poles")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    fig4.tight_layout()
    path4 = out_dir / "aak_certificate_poles.png"
    fig4.savefig(path4, dpi=160)
    print(f"wrote {path4}")


if __name__ == "__main__":
    main()
