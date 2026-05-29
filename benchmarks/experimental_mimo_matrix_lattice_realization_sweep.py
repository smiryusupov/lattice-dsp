"""Benchmark experimental MIMO state-space to matrix-lattice realization diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from collections.abc import Iterable

# When this benchmark is executed as ``python benchmarks/<script>.py``,
# Python puts ``benchmarks/`` on sys.path rather than the repository root.
# Prefer the checked-out source tree over any older installed lattice-dsp.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

import lattice_dsp as ld  # noqa: E402


def coupled_state_space(
    order: int, channels: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic stable square coupled MIMO state-space model."""

    if order <= 0:
        raise ValueError("order must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(order, order)))
    radii = np.linspace(0.30, 0.90, order)
    A = q @ np.diag(radii) @ np.linalg.inv(q)
    mix_in = rng.normal(size=(order, channels))
    mix_out = rng.normal(size=(channels, order))
    B = 0.35 * mix_in / np.sqrt(order)
    C = 0.35 * mix_out / np.sqrt(order)
    direct = rng.normal(size=(channels, channels))
    D = 0.08 * direct / max(np.linalg.norm(direct, ord=2), 1e-12)
    return A.astype(float), B.astype(float), C.astype(float), D.astype(float)


def state_spectral_radius(A: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(np.asarray(A, dtype=float))
    return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def median_runtime(fn, repeats: int) -> tuple[float, object]:
    times: list[float] = []
    value: object = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - start)
    return float(statistics.median(times)), value


def _float(value: object) -> float:
    return float(value)  # central place for mypy/readability in row construction


def run_case(
    *,
    full_order: int,
    reduced_order: int,
    lattice_order: int,
    channels: int,
    n_markov: int,
    n_freq: int,
    block_rows: int,
    block_cols: int,
    candidate_gains: Iterable[float],
    static_gain_iterations: int,
    repeats: int,
    n_threads: int,
    seed: int,
) -> dict[str, object]:
    """Run one coupled-MIMO realization diagnostic case."""

    A, B, C, D = coupled_state_space(full_order, channels, seed)
    markov = ld.mimo_state_space_markov_response(A, B, C, D, n_markov)

    reduce_s, reduced_obj = median_runtime(
        lambda: ld.finite_hankel_reduce_mimo(
            markov,
            reduced_order=reduced_order,
            block_rows=block_rows,
            block_cols=block_cols,
        ),
        repeats,
    )
    reduced = dict(reduced_obj)  # type: ignore[arg-type]

    def realize() -> dict[str, object]:
        return ld.experimental_mimo_state_space_to_matrix_lattice(
            reduced["A"],
            reduced["B"],
            reduced["C"],
            reduced["D"],
            order=lattice_order,
            n_markov=n_markov,
            n_freq=n_freq,
            candidate_gains=tuple(candidate_gains),
            fit_static_gains=True,
            static_gain_mode="both",
            static_gain_iterations=static_gain_iterations,
            n_threads=n_threads,
        )

    realize_s, fit_obj = median_runtime(realize, repeats)
    fit = dict(fit_obj)  # type: ignore[arg-type]
    raw_error = _float(fit["state_response_relative_error"])
    compensated_error = _float(fit["static_gain_relative_error"])
    improvement = raw_error / max(compensated_error, 1e-30)

    return {
        "full_order": int(full_order),
        "reduced_order": int(reduced_order),
        "lattice_order": int(lattice_order),
        "channels": int(channels),
        "reduce_s": reduce_s,
        "realize_s": realize_s,
        "reduced_state_radius": state_spectral_radius(np.asarray(reduced["A"], dtype=float)),
        "retained_hankel_energy": _float(reduced["retained_hankel_energy"]),
        "selected_gain": _float(fit["selected_gain"]),
        "polar_factor_relative_error": _float(fit["polar_factor_relative_error"]),
        "state_response_relative_error": raw_error,
        "static_gain_relative_error": compensated_error,
        "static_gain_improvement": improvement,
        "static_gain_left_condition": _float(fit["static_gain_left_condition"]),
        "static_gain_right_condition": _float(fit["static_gain_right_condition"]),
        "unitarity_error": _float(fit["unitarity_error"]),
        "max_reflection_singular_value": _float(fit["max_reflection_singular_value"]),
        "target_gain_condition_span": _float(fit["target_gain_condition_span"]),
        "diagnostic_classification": str(fit["diagnostic_classification"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-order", type=int, default=12)
    parser.add_argument("--reduced-orders", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--lattice-orders", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--n-markov", type=int, default=192)
    parser.add_argument("--n-freq", type=int, default=192)
    parser.add_argument("--block-rows", type=int, default=28)
    parser.add_argument("--block-cols", type=int, default=28)
    parser.add_argument(
        "--candidate-gains",
        type=float,
        nargs="+",
        default=[0.15, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85],
    )
    parser.add_argument("--static-gain-iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experimental-mimo-matrix-lattice-realization-sweep.json"),
    )
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.full_order <= 0:
        raise SystemExit("--full-order must be positive")
    if args.channels <= 0:
        raise SystemExit("--channels must be positive")
    if args.n_markov <= 8:
        raise SystemExit("--n-markov must be larger than 8")
    if args.n_freq < 8:
        raise SystemExit("--n-freq must be at least 8")
    if args.block_rows <= 0 or args.block_cols <= 0:
        raise SystemExit("--block-rows and --block-cols must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.static_gain_iterations <= 0:
        raise SystemExit("--static-gain-iterations must be positive")
    if not args.candidate_gains or any(g < 0.0 or not np.isfinite(g) for g in args.candidate_gains):
        raise SystemExit("--candidate-gains must be finite nonnegative values")
    if any(r <= 0 for r in args.reduced_orders):
        raise SystemExit("--reduced-orders must be positive")
    if any(o < 0 for o in args.lattice_orders):
        raise SystemExit("--lattice-orders must be nonnegative")


def main() -> None:
    args = parse_args()
    _validate_args(args)

    rows: list[dict[str, object]] = []
    for reduced_order in args.reduced_orders:
        for lattice_order in args.lattice_orders:
            rows.append(
                run_case(
                    full_order=args.full_order,
                    reduced_order=reduced_order,
                    lattice_order=lattice_order,
                    channels=args.channels,
                    n_markov=args.n_markov,
                    n_freq=args.n_freq,
                    block_rows=args.block_rows,
                    block_cols=args.block_cols,
                    candidate_gains=args.candidate_gains,
                    static_gain_iterations=args.static_gain_iterations,
                    repeats=args.repeats,
                    n_threads=args.n_threads,
                    seed=args.seed + 100 * reduced_order + lattice_order,
                )
            )

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "has_openmp": bool(getattr(ld, "HAS_OPENMP", False)),
        "full_order": args.full_order,
        "channels": args.channels,
        "n_markov": args.n_markov,
        "n_freq": args.n_freq,
        "block_rows": args.block_rows,
        "block_cols": args.block_cols,
        "candidate_gains": list(args.candidate_gains),
        "static_gain_iterations": args.static_gain_iterations,
        "repeats": args.repeats,
        "n_threads": args.n_threads,
        "description": (
            "Experimental MIMO state-space to matrix-lattice realization diagnostic sweep. "
            "The fitted lattice is an all-pass/polar scaffold; static gain compensation is reported "
            "separately and this is not an exact matrix AAK/Nehari solver."
        ),
        "results": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))
    print()
    print(
        f"{'red':>4} {'lat':>4} {'stable':>6} {'realize_s':>10} {'polar_err':>10} "
        f"{'raw_err':>10} {'gain_err':>10} {'improve':>9} {'unitary':>10} {'class':>34}"
    )
    print("-" * 118)
    for row in rows:
        stable = bool(
            float(row["max_reflection_singular_value"]) < 1.0
            and float(row["reduced_state_radius"]) < 1.0
        )
        print(
            f"{int(row['reduced_order']):4d} {int(row['lattice_order']):4d} {str(stable):>6} "
            f"{float(row['realize_s']):10.4f} {float(row['polar_factor_relative_error']):10.3e} "
            f"{float(row['state_response_relative_error']):10.3e} {float(row['static_gain_relative_error']):10.3e} "
            f"{float(row['static_gain_improvement']):9.2f} {float(row['unitarity_error']):10.2e} "
            f"{str(row['diagnostic_classification']):>34}"
        )
    print(f"\nWrote {args.output}")
    if args.csv_output is not None:
        print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
