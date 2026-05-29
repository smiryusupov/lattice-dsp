"""Tutorial: selecting a finite SISO AAK/Nehari rational candidate.

The previous tutorials introduced three pieces separately:

* finite Hankel singular values,
* the first neglected Schmidt pair,
* a rational recurrence fitted to a Hankelized anticausal tail.

This tutorial puts them together as a conservative candidate-selection workflow.
For a list of ranks, it builds finite Nehari/AAK-style Hankelized tail
approximations, realizes each one as a low-order rational model, and selects the
first candidate that meets user-chosen accuracy and stability thresholds.

This is still not a production infinite-dimensional AAK/Nehari solver.  It is a
validated finite-dimensional workflow with explicit acceptance criteria: target
singular value, tail error, rational realization error, and pole radius.  The reusable logic lives in ``lattice_dsp.finite_nehari_rational_candidates``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from collections.abc import Iterable

import numpy as np

import lattice_dsp as ld

try:
    from examples.finite_nehari_rational_bridge import synthetic_anticausal_tail
except ModuleNotFoundError:  # pragma: no cover - script execution from examples/
    from finite_nehari_rational_bridge import synthetic_anticausal_tail


CandidateCriteria = ld.FiniteNehariCandidateCriteria


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def candidate_rows(
    tail: np.ndarray,
    *,
    ranks: Iterable[int],
    rows: int,
    cols: int,
    criteria: CandidateCriteria,
) -> list[dict[str, float | int | bool]]:
    """Evaluate finite Nehari/rational candidates for a sequence of ranks.

    This wrapper keeps the tutorial table compact while delegating the reusable
    candidate-selection logic to the package-level API.
    """

    candidates = ld.finite_nehari_rational_candidates(
        tail,
        ranks=ranks,
        rows=rows,
        cols=cols,
        criteria=criteria,
    )
    return [
        {
            "rank": row["rank"],
            "sigma_next": row["sigma_next"],
            "hankelized_tail_error": row["hankelized_tail_error"],
            "rational_error": row["rational_error"],
            "rational_vs_hankelized_error": row["rational_vs_hankelized_error"],
            "max_pole_radius": row["max_pole_radius"],
            "accepted": row["accepted"],
        }
        for row in candidates
    ]


def select_candidate(rows: list[dict[str, float | int | bool]]) -> dict[str, float | int | bool]:
    """Return the first accepted row, or the row with the smallest rational error."""

    return ld.select_finite_nehari_candidate(rows)


def write_summary(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = artifact_dir()
    rows = cols = 48
    ranks = [1, 2, 3, 4, 5, 6]
    criteria = CandidateCriteria(max_tail_error=1e-3, max_rational_error=1e-2, max_pole_radius=0.99)
    tail = synthetic_anticausal_tail(rows + cols - 1)

    results = candidate_rows(tail, ranks=ranks, rows=rows, cols=cols, criteria=criteria)
    selected = select_candidate(results)

    summary_path = out_dir / "aak_siso_candidate_selection_summary.csv"
    write_summary(summary_path, results)

    print("finite Hankel matrix:", f"{rows} x {cols}")
    print("candidate ranks:", ranks)
    print(
        "criteria:",
        f"tail_error <= {criteria.max_tail_error:g},",
        f"rational_error <= {criteria.max_rational_error:g},",
        f"pole_radius <= {criteria.max_pole_radius:g}",
    )
    for row in results:
        status = "ACCEPT" if row["accepted"] else "reject"
        print(
            "rank={rank}: sigma_next={sigma_next:.3e}, tail_error={hankelized_tail_error:.3e}, "
            "rational_error={rational_error:.3e}, pole_radius={max_pole_radius:.4f}, {status}".format(
                **row, status=status
            )
        )
    print("selected rank:", selected["rank"])
    print(f"wrote {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    rank_values = np.asarray([row["rank"] for row in results], dtype=int)
    sigma_next = np.asarray([row["sigma_next"] for row in results], dtype=float)
    tail_errors = np.asarray([row["hankelized_tail_error"] for row in results], dtype=float)
    rational_errors = np.asarray([row["rational_error"] for row in results], dtype=float)
    pole_radii = np.asarray([row["max_pole_radius"] for row in results], dtype=float)
    accepted = np.asarray([bool(row["accepted"]) for row in results])

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.semilogy(rank_values, sigma_next, marker="o", label="sigma_next")
    ax.semilogy(rank_values, tail_errors, marker="s", label="Hankelized tail error")
    ax.semilogy(rank_values, rational_errors, marker="^", label="rational tail error")
    ax.axhline(criteria.max_tail_error, linestyle="--", linewidth=1.0, label="tail tolerance")
    ax.axhline(
        criteria.max_rational_error, linestyle=":", linewidth=1.0, label="rational tolerance"
    )
    ax.scatter(rank_values[accepted], rational_errors[accepted], s=90, marker="*", label="accepted")
    ax.set_title("Finite AAK/Nehari candidate selection by rank")
    ax.set_xlabel("candidate rank")
    ax.set_ylabel("error / singular value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "aak_siso_candidate_selection_errors.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.4))
    ax2.plot(rank_values, pole_radii, marker="o")
    ax2.axhline(
        criteria.max_pole_radius, linestyle="--", linewidth=1.0, label="pole-radius threshold"
    )
    ax2.set_title("Stability diagnostic for rational candidates")
    ax2.set_xlabel("candidate rank")
    ax2.set_ylabel("maximum pole radius")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    path2 = out_dir / "aak_siso_candidate_selection_poles.png"
    fig2.savefig(path2, dpi=160)
    print(f"wrote {path2}")


if __name__ == "__main__":
    main()
