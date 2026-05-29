"""Tutorial: finite-section AAK/Nehari reduction on a non-exact tail.

The exact rational-tail tutorial is deliberately clean: a rank-3 exponential
sequence is recovered to numerical precision.  Real identification and model-
reduction data is rarely that exact.  This tutorial adds a small deterministic
non-rational residual to a stable rank-3 tail and uses the high-level
``finite_aak_reduce_tail`` helper to select a practical rational candidate.

The point is not to claim full infinite-dimensional AAK/Nehari optimality.  The
point is to show the current finite-section workflow on a more realistic tail:
rank is selected by tolerances, the rational poles remain stable, and the
certificate reports the finite Schmidt-pair residuals for the selected rank.
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


def noisy_rational_tail(n_terms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a stable rank-3 tail plus a small deterministic non-rational residual."""

    poles = np.array([-0.42, 0.18, 0.76], dtype=float)
    weights = np.array([1.25, -0.7, 0.4], dtype=float)
    n = np.arange(n_terms, dtype=float)
    clean = sum(weight * pole**n for weight, pole in zip(weights, poles, strict=True))

    # A small deterministic residual with several damped components.  It keeps the
    # tutorial reproducible while making the finite Hankel matrix effectively full rank.
    residual = 0.012 * (0.83**n) * np.sin(0.41 * n + 0.2) + 0.006 * (0.57**n) * np.cos(1.17 * n)
    tail = np.asarray(clean + residual, dtype=float)
    return tail, np.asarray(clean, dtype=float), poles, weights


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = artifact_dir()
    rows = cols = 48
    ranks = [1, 2, 3, 4, 5, 6, 8]
    tail, clean_tail, true_poles, _ = noisy_rational_tail(rows + cols - 1)

    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=2.0e-2,
        max_rational_error=3.5e-2,
        max_pole_radius=0.99,
    )
    reduction = ld.finite_aak_reduce_tail(
        tail,
        ranks=ranks,
        rows=rows,
        cols=cols,
        criteria=criteria,
        attach_certificate=True,
    )

    candidates = reduction["candidates"]
    selected = reduction["selected"]
    certificate = reduction["certificate"]
    assert certificate is not None

    summary_rows = []
    for row in candidates:
        summary_rows.append(
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
    summary_path = out_dir / "finite_aak_noisy_tail_summary.csv"
    write_summary(summary_path, summary_rows)

    print("finite Hankel matrix:", f"{rows} x {cols}")
    print("true clean poles:", np.round(true_poles, 8).tolist())
    print("candidate ranks:", ranks)
    print(
        "criteria:",
        f"tail_error <= {criteria.max_tail_error:g},",
        f"rational_error <= {criteria.max_rational_error:g},",
        f"pole_radius <= {criteria.max_pole_radius:g}",
    )
    for row in summary_rows:
        status = "ACCEPT" if row["accepted"] else "reject"
        print(
            "rank={rank}: sigma_next={sigma_next:.3e}, tail_error={hankelized_tail_error:.3e}, "
            "rational_error={rational_error:.3e}, pole_radius={max_pole_radius:.4f}, {status}".format(
                **row, status=status
            )
        )
    print("selected rank:", reduction["selected_rank"])
    print("selected accepted:", reduction["accepted"])
    print("selected poles:", np.round(np.sort(np.real(selected["poles"])), 8).tolist())
    print("selected tail error:", f"{selected['hankelized_tail_error']:.3e}")
    print("selected rational error:", f"{selected['rational_error']:.3e}")
    print(
        "Schmidt residuals:",
        f"{certificate['schmidt_left_residual']:.3e}",
        f"{certificate['schmidt_right_residual']:.3e}",
    )
    print(f"wrote {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    rank_values = np.asarray([row["rank"] for row in summary_rows], dtype=int)
    sigma_next = np.asarray([row["sigma_next"] for row in summary_rows], dtype=float)
    tail_errors = np.asarray([row["hankelized_tail_error"] for row in summary_rows], dtype=float)
    rational_errors = np.asarray([row["rational_error"] for row in summary_rows], dtype=float)
    accepted = np.asarray([bool(row["accepted"]) for row in summary_rows])

    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    ax.semilogy(rank_values, sigma_next, marker="o", label="sigma_next")
    ax.semilogy(rank_values, tail_errors, marker="s", label="Hankelized tail error")
    ax.semilogy(rank_values, rational_errors, marker="^", label="rational tail error")
    ax.axhline(criteria.max_tail_error, linestyle="--", linewidth=1.0, label="tail tolerance")
    ax.axhline(
        criteria.max_rational_error, linestyle=":", linewidth=1.0, label="rational tolerance"
    )
    ax.scatter(rank_values[accepted], rational_errors[accepted], s=90, marker="*", label="accepted")
    ax.set_title("Finite-section AAK/Nehari candidate selection on a non-exact tail")
    ax.set_xlabel("candidate rank")
    ax.set_ylabel("error / singular value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "finite_aak_noisy_tail_errors.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

    rational_tail = np.asarray(selected["rational_tail"], dtype=float)
    approx_tail = np.asarray(selected["approximated_tail"], dtype=float)
    n = np.arange(tail.size)
    fig2, ax2 = plt.subplots(figsize=(9.4, 5.0))
    ax2.plot(n, tail, label="non-exact tail", linewidth=2.0)
    ax2.plot(n, clean_tail, linestyle=":", label="clean rank-3 component")
    ax2.plot(n, approx_tail, "--", label="Hankelized selected tail")
    ax2.plot(n, rational_tail, "-.", label="selected rational realization")
    ax2.set_title("Selected finite AAK/Nehari candidate on a non-exact tail")
    ax2.set_xlabel("tail index")
    ax2.set_ylabel("coefficient")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    path2 = out_dir / "finite_aak_noisy_tail_fit.png"
    fig2.savefig(path2, dpi=160)
    print(f"wrote {path2}")

    fig3, ax3 = plt.subplots(figsize=(5.8, 5.8))
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--")
    ax3.add_artist(circle)
    ax3.scatter(np.real(true_poles), np.imag(true_poles), marker="o", label="clean poles")
    ax3.scatter(
        np.real(selected["poles"]), np.imag(selected["poles"]), marker="x", label="selected poles"
    )
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_xlabel("real")
    ax3.set_ylabel("imaginary")
    ax3.set_title("Stable poles for the selected rational candidate")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    path3 = out_dir / "finite_aak_noisy_tail_poles.png"
    fig3.savefig(path3, dpi=160)
    print(f"wrote {path3}")


if __name__ == "__main__":
    main()
