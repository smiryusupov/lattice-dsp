"""Tutorial: exact rational-tail validation for finite Nehari candidates.

The previous tutorials use a synthetic tail whose effective order is visible from
its Hankel singular values.  This page uses an even cleaner validation case: the
anticausal tail is generated exactly by a known sum of three stable exponentials.
Such a sequence has finite Hankel rank three, so a trustworthy finite-Hankel /
rational workflow should reject ranks that are too small and recover the rank-3
model to near numerical precision.

This is not a full infinite-dimensional AAK/Nehari solver.  It is a controlled
regression/validation example for the finite-dimensional candidate-selection
workflow exposed by the package.
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
    """Return a stable rank-3 exponential tail and its generating poles/weights."""

    if n_terms <= 0:
        raise ValueError("n_terms must be positive")
    poles = np.asarray([0.76, -0.42, 0.18], dtype=np.float64)
    weights = np.asarray([1.25, -0.70, 0.40], dtype=np.float64)
    n = np.arange(n_terms, dtype=np.float64)
    tail = np.sum(weights[:, None] * poles[:, None] ** n[None, :], axis=0)
    return tail.astype(np.float64), poles, weights


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    public_rows = []
    for row in rows:
        public_rows.append(
            {
                "rank": row["rank"],
                "sigma_next": row["sigma_next"],
                "hankelized_tail_error": row["hankelized_tail_error"],
                "rational_error": row["rational_error"],
                "rational_vs_hankelized_error": row["rational_vs_hankelized_error"],
                "max_pole_radius": row["max_pole_radius"],
                "accepted": row["accepted"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)


def sorted_real_poles(poles: np.ndarray) -> np.ndarray:
    poles = np.asarray(poles, dtype=np.complex128)
    return np.sort(np.real_if_close(poles, tol=1000).real)


def main() -> None:
    out_dir = artifact_dir()
    rows = cols = 48
    ranks = [1, 2, 3, 4, 5]
    tail, true_poles, weights = exact_rational_tail(rows + cols - 1)

    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1e-8,
        max_rational_error=1e-8,
        max_pole_radius=0.99,
    )
    candidates = ld.finite_nehari_rational_candidates(
        tail,
        ranks=ranks,
        rows=rows,
        cols=cols,
        criteria=criteria,
    )
    selected = ld.select_finite_nehari_candidate(candidates)

    summary_path = out_dir / "finite_nehari_exact_rational_tail_summary.csv"
    write_summary(summary_path, candidates)

    print("finite Hankel matrix:", f"{rows} x {cols}")
    print("true poles:", np.round(np.sort(true_poles), 6).tolist())
    print("true weights:", np.round(weights, 6).tolist())
    print("candidate ranks:", ranks)
    print(
        "criteria:",
        f"tail_error <= {criteria.max_tail_error:g},",
        f"rational_error <= {criteria.max_rational_error:g},",
        f"pole_radius <= {criteria.max_pole_radius:g}",
    )
    for row in candidates:
        status = "ACCEPT" if row["accepted"] else "reject"
        print(
            "rank={rank}: sigma_next={sigma_next:.3e}, tail_error={hankelized_tail_error:.3e}, "
            "rational_error={rational_error:.3e}, pole_radius={max_pole_radius:.4f}, {status}".format(
                **row, status=status
            )
        )
    print("selected rank:", selected["rank"])
    print("selected poles:", np.round(sorted_real_poles(selected["poles"]), 6).tolist())
    print(f"wrote {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    rank_values = np.asarray([row["rank"] for row in candidates], dtype=int)
    sigma_next = np.asarray([row["sigma_next"] for row in candidates], dtype=float)
    tail_errors = np.asarray([row["hankelized_tail_error"] for row in candidates], dtype=float)
    rational_errors = np.asarray([row["rational_error"] for row in candidates], dtype=float)
    accepted = np.asarray([bool(row["accepted"]) for row in candidates])

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.semilogy(rank_values, sigma_next, marker="o", label="sigma_next")
    ax.semilogy(rank_values, tail_errors, marker="s", label="Hankelized tail error")
    ax.semilogy(rank_values, rational_errors, marker="^", label="rational tail error")
    ax.axvline(3, linestyle="--", linewidth=1.0, label="true rank")
    ax.scatter(rank_values[accepted], rational_errors[accepted], s=90, marker="*", label="accepted")
    ax.set_title("Exact rank-3 rational tail: candidate-selection validation")
    ax.set_xlabel("candidate rank")
    ax.set_ylabel("error / singular value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "finite_nehari_exact_rational_tail_errors.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

    fig2, ax2 = plt.subplots(figsize=(8.2, 4.6))
    unit = plt.Circle((0.0, 0.0), 1.0, fill=False, linestyle="--", alpha=0.6)
    ax2.add_patch(unit)
    ax2.scatter(np.real(true_poles), np.imag(true_poles), marker="x", s=100, label="true poles")
    ax2.scatter(
        np.real(selected["poles"]),
        np.imag(selected["poles"]),
        marker="o",
        s=65,
        label="selected poles",
    )
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_title("Known poles recovered by the selected rational candidate")
    ax2.set_xlabel("real")
    ax2.set_ylabel("imaginary")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    path2 = out_dir / "finite_nehari_exact_rational_tail_poles.png"
    fig2.savefig(path2, dpi=160)
    print(f"wrote {path2}")

    fig3, ax3 = plt.subplots(figsize=(8.8, 4.6))
    n = np.arange(tail.size)
    ax3.plot(n, tail, label="exact tail", linewidth=2.0)
    ax3.plot(
        n, selected["rational_tail"], "--", label=f"selected rank {selected['rank']} rational tail"
    )
    ax3.set_title("Exact tail and selected rational realization")
    ax3.set_xlabel("coefficient index")
    ax3.set_ylabel("coefficient")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    path3 = out_dir / "finite_nehari_exact_rational_tail_fit.png"
    fig3.savefig(path3, dpi=160)
    print(f"wrote {path3}")


if __name__ == "__main__":
    main()
