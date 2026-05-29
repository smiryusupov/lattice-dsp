"""MIMO long-signal stress for finite block-Hankel reduction and compiled state-space runtime.

This example complements the scalar long-signal and echo-scale stress tutorials.
It focuses on the multichannel niche: a coupled MIMO system is reduced with the
finite block-Hankel workflow, then the reduced state-space model is reused on a
long batched multichannel signal through the compiled C++ runtime.

The comparison numbers are deliberately scale diagnostics.  They show the cost
profile of a dense recursive MIMO state-space model against a long direct-form
MIMO FIR echo model.  They are not an accuracy-equivalence claim and they are not
a matrix-valued AAK/Nehari solver claim.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def median_time(fn, repeats: int) -> tuple[float, object]:
    times: list[float] = []
    result: object | None = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    assert result is not None
    return statistics.median(times), result


def stable_coupled_state_space(
    order: int,
    outputs: int,
    inputs: int,
    radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a deterministic stable dense MIMO state-space model."""
    rng = np.random.default_rng(seed)

    # A random orthogonal similarity keeps the system coupled while making the
    # spectral radius explicit.  The descending radii create a long but stable
    # memory profile.
    q, _ = np.linalg.qr(rng.normal(size=(order, order)))
    radii = np.linspace(radius, 0.15, order, dtype=np.float64)
    A = q @ np.diag(radii) @ q.T

    # Scale B and C by sqrt(order) so the output RMS remains controlled when the
    # state dimension changes.
    scale = max(np.sqrt(order), 1.0)
    B = rng.normal(size=(order, inputs)) / scale
    C = rng.normal(size=(outputs, order)) / scale
    D = 0.015 * rng.normal(size=(outputs, inputs))
    return A.astype(np.float64), B.astype(np.float64), C.astype(np.float64), D.astype(np.float64)


def colored_multichannel_input(
    batch: int,
    samples: int,
    inputs: int,
    seed: int,
) -> np.ndarray:
    """Generate dependency-free multichannel input with light temporal coloring."""
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(batch, samples, inputs)).astype(np.float64)
    x = np.empty_like(white)
    state = np.zeros((batch, inputs), dtype=np.float64)
    for n in range(samples):
        state = 0.88 * state + 0.12 * white[:, n, :]
        x[:, n, :] = state
    rms = np.sqrt(np.mean(x * x, axis=(1, 2), keepdims=True))
    return x / np.maximum(rms, 1e-30)


def dense_state_space_visit_scale(
    batch: int,
    samples: int,
    order: int,
    inputs: int,
    outputs: int,
) -> int:
    """Approximate dense multiply-add visits for a batched MIMO state-space pass."""
    per_sample = order * order + order * inputs + outputs * order + outputs * inputs
    return int(batch) * int(samples) * int(per_sample)


def direct_mimo_fir_visit_scale(
    batch: int,
    samples: int,
    taps: int,
    inputs: int,
    outputs: int,
) -> int:
    """Approximate direct-form MIMO FIR multiply-add visits."""
    return int(batch) * int(samples) * int(taps) * int(inputs) * int(outputs)


def spectral_radius(A: np.ndarray) -> float:
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=int, default=1_000_000, help="number of time samples per stream"
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="number of independent multichannel streams"
    )
    parser.add_argument("--inputs", type=int, default=8, help="input channels")
    parser.add_argument("--outputs", type=int, default=8, help="output channels")
    parser.add_argument(
        "--full-order",
        type=int,
        default=64,
        help="order of the synthetic full MIMO state-space model",
    )
    parser.add_argument(
        "--reduced-order",
        type=int,
        default=16,
        help="finite block-Hankel reduced order used for long-signal processing",
    )
    parser.add_argument(
        "--markov-samples",
        type=int,
        default=320,
        help="number of MIMO Markov parameters used for reduction",
    )
    parser.add_argument(
        "--block-rows", type=int, default=24, help="block rows in the finite block-Hankel matrix"
    )
    parser.add_argument(
        "--block-cols", type=int, default=24, help="block columns in the finite block-Hankel matrix"
    )
    parser.add_argument(
        "--fir-taps",
        type=int,
        default=131_072,
        help="reference MIMO FIR echo-tap count for scale estimates",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.985,
        help="dominant stable pole radius of the generated full model",
    )
    parser.add_argument("--repeats", type=int, default=3, help="median timing repeats")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument(
        "--n-threads",
        type=int,
        default=0,
        help="threads for the compiled state-space runtime; 0 uses backend default",
    )
    parser.add_argument(
        "--time-full",
        action="store_true",
        help="also time the unreduced full-order state-space model",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.inputs <= 0 or args.outputs <= 0:
        raise ValueError("--inputs and --outputs must be positive")
    if args.full_order <= 0 or args.reduced_order <= 0:
        raise ValueError("--full-order and --reduced-order must be positive")
    if args.reduced_order > args.full_order:
        raise ValueError("--reduced-order must not exceed --full-order")
    if args.markov_samples <= args.block_rows + args.block_cols:
        raise ValueError("--markov-samples must exceed --block-rows + --block-cols")
    if args.fir_taps <= 0:
        raise ValueError("--fir-taps must be positive")
    if not (0.0 < args.radius < 1.0):
        raise ValueError("--radius must satisfy 0 < radius < 1")

    A, B, C, D = stable_coupled_state_space(
        args.full_order,
        args.outputs,
        args.inputs,
        args.radius,
        args.seed,
    )

    markov_time, markov_obj = median_time(
        lambda: ld.mimo_state_space_markov_response(A, B, C, D, args.markov_samples),
        1,
    )
    markov = np.asarray(markov_obj, dtype=np.float64)

    reduce_time, result_obj = median_time(
        lambda: ld.finite_hankel_reduce_mimo(
            markov,
            reduced_order=args.reduced_order,
            block_rows=args.block_rows,
            block_cols=args.block_cols,
        ),
        1,
    )
    result = dict(result_obj)
    Ar = np.asarray(result["A"], dtype=np.float64)
    Br = np.asarray(result["B"], dtype=np.float64)
    Cr = np.asarray(result["C"], dtype=np.float64)
    Dr = np.asarray(result["D"], dtype=np.float64)

    approx = np.asarray(
        ld.mimo_state_space_markov_response(Ar, Br, Cr, Dr, args.markov_samples), dtype=np.float64
    )
    relative_markov_error = float(
        np.sum((markov - approx) ** 2) / np.maximum(np.sum(markov * markov), 1e-30)
    )

    x = colored_multichannel_input(args.batch, args.samples, args.inputs, args.seed + 1)

    reduced_time, y_obj = median_time(
        lambda: ld.mimo_state_space_process_batch(Ar, Br, Cr, Dr, x, n_threads=args.n_threads),
        args.repeats,
    )
    y = np.asarray(y_obj, dtype=np.float64)

    reduced_throughput_streams = args.batch * args.samples / max(reduced_time, 1e-30) / 1e6
    reduced_channel_throughput = (
        args.batch * args.samples * args.outputs / max(reduced_time, 1e-30) / 1e6
    )
    reduced_visits = dense_state_space_visit_scale(
        args.batch, args.samples, args.reduced_order, args.inputs, args.outputs
    )
    full_visits = dense_state_space_visit_scale(
        args.batch, args.samples, args.full_order, args.inputs, args.outputs
    )
    fir_visits = direct_mimo_fir_visit_scale(
        args.batch, args.samples, args.fir_taps, args.inputs, args.outputs
    )
    fir_lms_visits = 2 * fir_visits
    hankel_rows = args.block_rows * args.outputs
    hankel_cols = args.block_cols * args.inputs

    rows: list[dict[str, object]] = [
        {
            "method": "mimo_markov_generation_cpp",
            "seconds": markov_time,
            "markov_samples": args.markov_samples,
            "full_order": args.full_order,
            "inputs": args.inputs,
            "outputs": args.outputs,
        },
        {
            "method": "finite_block_hankel_reduce_mimo_cpp",
            "seconds": reduce_time,
            "full_order": args.full_order,
            "reduced_order": args.reduced_order,
            "block_hankel_rows": hankel_rows,
            "block_hankel_cols": hankel_cols,
            "retained_hankel_energy": float(result.get("retained_hankel_energy", np.nan)),
            "relative_markov_error": relative_markov_error,
            "reduced_stable": bool(result.get("stable", False)),
        },
        {
            "method": "mimo_state_space_process_batch_reduced_cpp",
            "seconds": reduced_time,
            "samples": args.samples,
            "batch": args.batch,
            "inputs": args.inputs,
            "outputs": args.outputs,
            "reduced_order": args.reduced_order,
            "throughput_mstreams_per_s": reduced_throughput_streams,
            "throughput_moutput_channels_per_s": reduced_channel_throughput,
            "dense_state_space_visits": reduced_visits,
            "dense_state_space_visit_rate_giga_per_s": reduced_visits
            / max(reduced_time, 1e-30)
            / 1e9,
            "output_rms": float(np.sqrt(np.mean(y * y))),
            "reduced_spectral_radius": spectral_radius(Ar),
        },
        {
            "method": "direct_mimo_fir_echo_scale_estimate_filter_only",
            "samples": args.samples,
            "batch": args.batch,
            "inputs": args.inputs,
            "outputs": args.outputs,
            "fir_taps": args.fir_taps,
            "direct_fir_visits": fir_visits,
            "fir_taps_per_reduced_state": args.fir_taps / args.reduced_order,
        },
        {
            "method": "direct_mimo_fir_lms_scale_estimate_filter_plus_update",
            "samples": args.samples,
            "batch": args.batch,
            "inputs": args.inputs,
            "outputs": args.outputs,
            "fir_taps": args.fir_taps,
            "direct_fir_lms_visits": fir_lms_visits,
            "fir_taps_per_reduced_state": args.fir_taps / args.reduced_order,
        },
    ]

    print("MIMO long-signal finite-Hankel/state-space stress")
    print("=" * 64)
    print(f"batch streams: {args.batch:,}")
    print(f"samples per stream: {args.samples:,}")
    print(f"inputs x outputs: {args.inputs} x {args.outputs}")
    print(f"full MIMO state order: {args.full_order:,}")
    print(f"dominant full-model pole radius target: {args.radius:.6f}")
    print(f"full-model spectral radius: {spectral_radius(A):.6f}")
    print(f"Markov samples for reduction: {args.markov_samples:,}")
    print(f"block-Hankel matrix: {hankel_rows:,} x {hankel_cols:,}")
    print(f"reduced order: {args.reduced_order:,}")
    print(f"Markov generation time: {markov_time:.6f} s")
    print(f"finite MIMO block-Hankel reduction time: {reduce_time:.6f} s")
    print(f"retained Hankel energy: {float(result.get('retained_hankel_energy', np.nan)):.6f}")
    print(f"relative Markov error: {relative_markov_error:.3e}")
    print(f"reduced model stable: {bool(result.get('stable', False))}")
    print(f"reduced spectral radius: {spectral_radius(Ar):.6f}")
    print()
    print("compiled reduced MIMO runtime")
    print("-" * 64)
    print(f"median reduced state-space time: {reduced_time:.6f} s")
    print(f"throughput: {reduced_throughput_streams:.2f} million multichannel samples/s")
    print(f"output-channel throughput: {reduced_channel_throughput:.2f} million output samples/s")
    print(f"dense reduced state-space visits: {reduced_visits:,}")
    print(
        f"dense visit rate: {reduced_visits / max(reduced_time, 1e-30) / 1e9:.2f} billion visits/s"
    )
    print(f"output RMS: {np.sqrt(np.mean(y * y)):.6f}")
    print()
    print("MIMO echo-scale comparison numbers")
    print("-" * 64)
    print(f"reference MIMO FIR taps per input-output path: {args.fir_taps:,}")
    print(f"FIR taps / reduced state order: {args.fir_taps / args.reduced_order:.1f}x")
    print(f"full dense state-space visits at same signal size: {full_visits:,}")
    print(f"reduced dense state-space visits: {reduced_visits:,}")
    print(f"direct MIMO FIR filter visits: {fir_visits:,}")
    print(f"direct MIMO FIR LMS filter+update visits, rough scale: {fir_lms_visits:,}")
    print("note: these are scale diagnostics, not an accuracy equivalence claim")

    if args.time_full:
        full_time, y_full_obj = median_time(
            lambda: ld.mimo_state_space_process_batch(A, B, C, D, x, n_threads=args.n_threads),
            args.repeats,
        )
        y_full = np.asarray(y_full_obj, dtype=np.float64)
        rows.append(
            {
                "method": "mimo_state_space_process_batch_full_cpp",
                "seconds": full_time,
                "samples": args.samples,
                "batch": args.batch,
                "inputs": args.inputs,
                "outputs": args.outputs,
                "full_order": args.full_order,
                "throughput_mstreams_per_s": args.batch
                * args.samples
                / max(full_time, 1e-30)
                / 1e6,
                "throughput_moutput_channels_per_s": args.batch
                * args.samples
                * args.outputs
                / max(full_time, 1e-30)
                / 1e6,
                "dense_state_space_visits": full_visits,
                "dense_state_space_visit_rate_giga_per_s": full_visits
                / max(full_time, 1e-30)
                / 1e9,
                "output_rms": float(np.sqrt(np.mean(y_full * y_full))),
                "speedup_reduced_vs_full": full_time / max(reduced_time, 1e-30),
            }
        )
        print()
        print("optional full-order runtime")
        print("-" * 64)
        print(f"median full state-space time: {full_time:.6f} s")
        print(f"reduced/full runtime speedup: {full_time / max(reduced_time, 1e-30):.2f}x")
        print(f"full output RMS: {np.sqrt(np.mean(y_full * y_full)):.6f}")

    out_dir = artifact_dir()
    csv_path = out_dir / "mimo_long_signal_state_space_stress.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
