"""Benchmark matrix-lattice frequency-response evaluation runtime."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

import lattice_dsp as ld
from lattice_dsp import matrix_lattice as matrix_lattice_module


def make_filter(dim: int, order: int, seed: int) -> ld.MatrixLatticeAllPass:
    rng = np.random.default_rng(seed)
    reflections = [
        ld.contractive_matrix_from_raw(
            0.22 * (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))), margin=1e-6
        )
        for _ in range(order)
    ]
    residue = ld.unitary_polar_factor(
        rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    )
    return ld.MatrixLatticeAllPass(reflections, residue=residue)


def median_runtime(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(statistics.median(times))


def response_unitarity_error(response: np.ndarray) -> float:
    eye = np.eye(response.shape[1], dtype=np.complex128)
    return float(max(np.linalg.norm(g.conj().T @ g - eye, ord="fro") for g in response))


def run_case(
    dim: int, order: int, n_freq: int, repeats: int, n_threads: int, seed: int
) -> dict[str, float | int | bool]:
    filt = make_filter(dim, order, seed)
    omega = np.linspace(0.0, np.pi, n_freq)

    # Warm both paths before timing.
    compiled_response = filt.frequency_response(omega, n_threads=n_threads)
    python_response = matrix_lattice_module._frequency_response_numpy(
        filt.stage_blocks, filt.residue, omega
    )  # noqa: SLF001

    compiled_s = median_runtime(
        lambda: filt.frequency_response(omega, n_threads=n_threads), repeats
    )
    python_s = median_runtime(
        lambda: matrix_lattice_module._frequency_response_numpy(
            filt.stage_blocks, filt.residue, omega
        ),
        repeats,  # noqa: SLF001
    )
    rel_diff = float(
        np.linalg.norm(compiled_response - python_response)
        / max(np.linalg.norm(python_response), 1e-30)
    )
    speedup = python_s / compiled_s if compiled_s > 0.0 else float("inf")

    return {
        "dim": dim,
        "order": order,
        "n_freq": n_freq,
        "n_threads": n_threads,
        "compiled_s": compiled_s,
        "python_s": python_s,
        "speedup": speedup,
        "relative_difference": rel_diff,
        "unitarity_error": response_unitarity_error(compiled_response),
        "max_reflection_singular_value": filt.max_reflection_singular_value(),
        "real_scalar_parameter_count": filt.parameter_count(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--n-freq", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--output", type=Path, default=Path("reports/matrix-lattice-runtime.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_freq <= 0:
        raise SystemExit("--n-freq must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    rows = []
    for dim in args.dims:
        for order in args.orders:
            rows.append(
                run_case(
                    dim,
                    order,
                    args.n_freq,
                    args.repeats,
                    args.n_threads,
                    args.seed + 100 * dim + order,
                )
            )

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "has_openmp": bool(getattr(ld, "HAS_OPENMP", False)),
        "n_freq": args.n_freq,
        "repeats": args.repeats,
        "n_threads": args.n_threads,
        "description": "MatrixLatticeAllPass frequency-response benchmark. The compiled C++ path is compared with the NumPy reference evaluator.",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))
    print()
    print(
        f"{'dim':>4} {'order':>5} {'params':>8} {'compiled_s':>11} {'python_s':>10} {'speedup':>8} {'unitarity':>11} {'rel_diff':>10}"
    )
    print("-" * 86)
    for row in rows:
        print(
            f"{row['dim']:4d} {row['order']:5d} {row['real_scalar_parameter_count']:8d} "
            f"{row['compiled_s']:11.5f} {row['python_s']:10.5f} {row['speedup']:8.2f} "
            f"{row['unitarity_error']:11.2e} {row['relative_difference']:10.2e}"
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
