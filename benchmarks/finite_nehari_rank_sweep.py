"""Finite Nehari/AAK rank-sweep benchmark.

This benchmark is a numerical validation and tutorial companion for
``finite_nehari_approximate_tail``.  It does not implement an exact
infinite-dimensional Nehari or AAK solver.  Instead, it measures how the finite
Hankel singular values, unconstrained SVD error, Hankelized error, and tail
approximation error change with rank.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

import lattice_dsp as ld


def synthetic_anticausal_tail(n_terms: int, seed: int = 7) -> np.ndarray:
    """Return a deterministic anticausal tail with several decaying modes.

    The first modes are strong and the later modes are weaker.  This produces a
    clear singular-value decay while still leaving visible approximation error at
    small rank.
    """

    if n_terms <= 0:
        raise ValueError("n_terms must be positive")

    # The seed is used only to make small sign/weight perturbations deterministic.
    rng = np.random.default_rng(seed)
    poles = np.array([0.94, 0.78, -0.58, 0.38, -0.22], dtype=float)
    weights = np.array([1.00, 0.28, -0.18, 0.07, 0.025], dtype=float)
    weights = weights * (1.0 + 0.015 * rng.normal(size=weights.shape))

    n = np.arange(n_terms, dtype=float)
    tail = np.zeros(n_terms, dtype=float)
    for w, p in zip(weights, poles, strict=True):
        tail += w * p**n
    return tail


def run_rank_sweep(
    tail: np.ndarray,
    ranks: list[int],
    rows: int,
    cols: int,
) -> list[dict[str, Any]]:
    """Run finite Nehari/AAK approximation for a list of ranks."""

    results: list[dict[str, Any]] = []
    for rank in ranks:
        t0 = time.perf_counter()
        result = ld.finite_nehari_approximate_tail(
            tail.tolist(),
            rank=int(rank),
            rows=int(rows),
            cols=int(cols),
        )
        elapsed = time.perf_counter() - t0

        sigma_next = float(result["sigma_next"])
        hankelized_error = float(result["hankelized_hankel_error"])
        unconstrained_error = float(result["unconstrained_hankel_error"])
        tail_error = float(result["relative_tail_error"])

        results.append(
            {
                "rank": int(rank),
                "time_s": elapsed,
                "sigma_next": sigma_next,
                "unconstrained_hankel_error": unconstrained_error,
                "hankelized_hankel_error": hankelized_error,
                "hankelized_over_sigma_next": hankelized_error / sigma_next
                if sigma_next > 0
                else None,
                "relative_tail_error": tail_error,
            }
        )
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite Nehari/AAK rank-sweep benchmark.")
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--cols", type=int, default=48)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/finite-nehari-rank-sweep.json")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("reports/finite-nehari-rank-sweep.csv")
    )
    args = parser.parse_args()

    n_terms = args.rows + args.cols - 1
    tail = synthetic_anticausal_tail(n_terms, seed=args.seed)
    rows = run_rank_sweep(tail, args.ranks, args.rows, args.cols)

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rows": args.rows,
        "cols": args.cols,
        "tail_terms": int(n_terms),
        "ranks": args.ranks,
        "seed": args.seed,
        "description": (
            "Finite-dimensional Nehari/AAK teaching benchmark. The SVD error equals sigma_next "
            "for the unconstrained rank-r matrix problem; the Hankelized error is a separate "
            "structured approximation diagnostic."
        ),
    }

    payload = {"metadata": metadata, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.csv_output, rows)

    print(json.dumps(metadata, indent=2))
    print()
    print(
        f"{'rank':>5s} {'time_s':>10s} {'sigma_next':>13s} {'svd_error':>13s} {'hankelized':>13s} {'hank/sigma':>11s} {'tail_rel':>11s}"
    )
    print("-" * 86)
    for row in rows:
        ratio = row["hankelized_over_sigma_next"]
        ratio_text = f"{ratio:11.3f}" if ratio is not None else f"{'n/a':>11s}"
        print(
            f"{row['rank']:5d} "
            f"{row['time_s']:10.5f} "
            f"{row['sigma_next']:13.6e} "
            f"{row['unconstrained_hankel_error']:13.6e} "
            f"{row['hankelized_hankel_error']:13.6e} "
            f"{ratio_text} "
            f"{row['relative_tail_error']:11.3e}"
        )
    print()
    print(f"Wrote {args.output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
