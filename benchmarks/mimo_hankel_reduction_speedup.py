"""Benchmark finite block-Hankel reduction on coupled MIMO systems.

The reducer returns state-space models.  This benchmark therefore measures the
one-time reduction cost and the repeated cost of simulating the full and reduced
state-space systems on batched MIMO input signals.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def state_spectral_radius(A) -> float:
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def coupled_state_space(order: int, outputs: int, inputs: int, seed: int):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(order, order)))
    radii = np.linspace(0.90, 0.18, order)
    signs = np.where(np.arange(order) % 2 == 0, 1.0, -1.0)
    A = q @ np.diag(signs * radii) @ q.T
    B = 0.28 * rng.normal(size=(order, inputs))
    C = 0.28 * rng.normal(size=(outputs, order))
    D = 0.025 * rng.normal(size=(outputs, inputs))
    if inputs == outputs:
        D += 0.035 * (np.ones((outputs, inputs)) - np.eye(outputs))
    return A, B, C, D


def state_space_process_python(A, B, C, D, x):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    x = np.asarray(x, dtype=float)
    batch, samples, _ = x.shape
    n_outputs = D.shape[0]
    n_state = A.shape[0]
    state = np.zeros((batch, n_state), dtype=float)
    y = np.zeros((batch, samples, n_outputs), dtype=float)
    for n in range(samples):
        xn = x[:, n, :]
        y[:, n, :] = state @ C.T + xn @ D.T
        if n_state:
            state = state @ A.T + xn @ B.T
    return y


def state_space_process(A, B, C, D, x, n_threads: int = 0):
    """Use the compiled state-space processor when available.

    The Python fallback keeps the benchmark source readable when somebody runs it
    against an older local extension before rebuilding.
    """

    compiled = getattr(ld, "mimo_state_space_process_batch", None)
    if compiled is None:
        return state_space_process_python(A, B, C, D, x)
    return compiled(A, B, C, D, x, n_threads=n_threads)


def median_time(fn, repeats: int):
    values = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - t0)
    return statistics.median(values), result


def rel_mse(reference, estimate) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    return float(np.sum((reference - estimate) ** 2) / (np.sum(reference**2) + 1e-30))


def snr_db_from_rel_mse(value: float) -> float:
    return 10.0 * math.log10(1.0 / max(value, 1e-300))


def break_even_samples_per_batch(reduction_s: float, full_s: float, reduced_s: float, samples: int):
    full_per_sample = full_s / samples
    reduced_per_sample = reduced_s / samples
    saved = full_per_sample - reduced_per_sample
    if saved <= 0.0:
        return None
    return reduction_s / saved


def run(args):
    rng = np.random.default_rng(args.seed)
    rows = []
    n_threads = getattr(args, "n_threads", 0)
    reuse_count = max(1, int(getattr(args, "reuse_count", 1)))

    for full_order in args.full_orders:
        A, B, C, D = coupled_state_space(
            full_order, args.outputs, args.inputs, args.seed + full_order
        )
        markov = ld.mimo_state_space_markov_response(A, B, C, D, args.n_markov)
        x = rng.normal(size=(args.batch, args.samples, args.inputs))

        full_time, y_full = median_time(
            lambda A=A, B=B, C=C, D=D, x=x: state_space_process(A, B, C, D, x, n_threads),
            args.repeats,
        )

        for reduced_order in args.reduced_orders:
            if reduced_order > min(args.block_rows * args.outputs, args.block_cols * args.inputs):
                continue
            if reduced_order > full_order:
                continue

            t0 = time.perf_counter()
            try:
                reduction = ld.finite_hankel_reduce_mimo(
                    markov,
                    reduced_order=reduced_order,
                    block_rows=args.block_rows,
                    block_cols=args.block_cols,
                )
                reduce_s = time.perf_counter() - t0
            except ValueError as exc:
                rows.append(
                    {
                        "full_order": full_order,
                        "reduced_order": reduced_order,
                        "stable": False,
                        "error": str(exc),
                    }
                )
                continue

            red_time, y_red = median_time(
                lambda r=reduction, x=x: state_space_process(
                    r["A"], r["B"], r["C"], r["D"], x, n_threads
                ),
                args.repeats,
            )
            output_rel = rel_mse(y_full, y_red)
            approx_markov = ld.mimo_state_space_markov_response(
                reduction["A"], reduction["B"], reduction["C"], reduction["D"], args.n_markov
            )
            markov_rel = rel_mse(markov, approx_markov)
            filter_speedup = full_time / red_time if red_time > 0 else float("inf")
            one_shot_end2end = (
                full_time / (reduce_s + red_time) if reduce_s + red_time > 0 else float("inf")
            )
            amortized_denominator = reduce_s + reuse_count * red_time
            amortized_end2end = (
                (reuse_count * full_time) / amortized_denominator
                if amortized_denominator > 0
                else float("inf")
            )
            break_even = break_even_samples_per_batch(reduce_s, full_time, red_time, args.samples)

            rows.append(
                {
                    "full_order": full_order,
                    "reduced_order": reduced_order,
                    "stable": bool(reduction["stable"]),
                    "full_state_radius": state_spectral_radius(A),
                    "reduced_state_radius": state_spectral_radius(reduction["A"]),
                    "retained_hankel_energy": float(reduction["retained_hankel_energy"]),
                    "relative_markov_error": markov_rel,
                    "relative_output_error": output_rel,
                    "output_snr_db": snr_db_from_rel_mse(output_rel),
                    "reduction_time_s": reduce_s,
                    "full_process_median_s": full_time,
                    "reduced_process_median_s": red_time,
                    "process_speedup": filter_speedup,
                    "one_shot_end_to_end_speedup": one_shot_end2end,
                    "amortized_end_to_end_speedup": amortized_end2end,
                    "reuse_count": reuse_count,
                    "break_even_samples_per_batch": break_even,
                }
            )

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "has_openmp": bool(getattr(ld, "HAS_OPENMP", False)),
        "inputs": args.inputs,
        "outputs": args.outputs,
        "batch": args.batch,
        "samples": args.samples,
        "repeats": args.repeats,
        "reuse_count": reuse_count,
        "n_threads": n_threads,
        "state_space_backend": "compiled"
        if getattr(ld, "mimo_state_space_process_batch", None) is not None
        else "python",
        "n_markov": args.n_markov,
        "block_rows": args.block_rows,
        "block_cols": args.block_cols,
        "seed": args.seed,
        "description": "Coupled MIMO finite block-Hankel reduction benchmark. Reduction is a preprocessing cost; processing speedup is measured by batched state-space simulation, and amortized end-to-end speedup assumes the reduced model is reused for reuse_count batches.",
    }
    return {"metadata": metadata, "rows": rows}


def print_table(result):
    print(json.dumps(result["metadata"], indent=2))
    print()
    print(
        f"{'full':>5s} {'red':>5s} {'stable':>7s} {'reduce_s':>10s} {'proc_x':>9s} "
        f"{'one_x':>8s} {'reuse_x':>9s} {'SNR':>8s} {'markov_err':>12s} {'radius':>8s} {'break_even':>12s}"
    )
    print("-" * 110)
    for row in result["rows"]:
        if "error" in row:
            print(
                f"{row['full_order']:5d} {row['reduced_order']:5d} {'False':>7s} "
                f"{'n/a':>10s} {'n/a':>9s} {'n/a':>8s} {'n/a':>9s} "
                f"{'n/a':>8s} {'n/a':>12s} {'n/a':>8s} {'n/a':>12s}"
            )
            continue
        be = row["break_even_samples_per_batch"]
        be_text = "n/a" if be is None else f"{be:.0f}"
        print(
            f"{row['full_order']:5d} {row['reduced_order']:5d} {str(row['stable']):>7s} "
            f"{row['reduction_time_s']:10.4f} {row['process_speedup']:9.2f} "
            f"{row['one_shot_end_to_end_speedup']:8.2f} {row['amortized_end_to_end_speedup']:9.2f} "
            f"{row['output_snr_db']:8.2f} {row['relative_markov_error']:12.3e} "
            f"{row['reduced_state_radius']:8.4f} {be_text:>12s}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-orders", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--reduced-orders", type=int, nargs="+", default=[2, 4, 6, 8, 12])
    parser.add_argument("--inputs", type=int, default=3)
    parser.add_argument("--outputs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--reuse-count",
        type=int,
        default=1,
        help="number of additional batches over which to amortize the one-time reduction cost",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="thread count for the compiled state-space processor; use 0 for the OpenMP default when available",
    )
    parser.add_argument("--n-markov", type=int, default=512)
    parser.add_argument("--block-rows", type=int, default=48)
    parser.add_argument("--block-cols", type=int, default=48)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--output", default="reports/mimo-hankel-reduction-speedup.json")
    args = parser.parse_args()

    result = run(args)
    print_table(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
