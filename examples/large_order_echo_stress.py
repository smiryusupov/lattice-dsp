"""Large echo-scale stress: long signals and high-order stable recursive models.

This example complements ``million_sample_iir_throughput.py``.  The first
throughput example shows a very long tail that collapses to one recursive state.
Here we stress the other axis that matters in echo-style work: a long signal and
a much larger stable recursive order.

The example is intentionally a fixed-filter throughput and scale diagnostic, not
an adaptive echo canceller.  Classical FIR echo cancellation often needs a very
large tap vector to cover delay, early reflections, and a reverberant tail.  An
LMS-style FIR pass touches that large vector at every sample, and adaptation can
roughly double the coefficient traffic because it both filters and updates taps.

A stable lattice/lattice-ladder IIR model has a different cost profile.  It keeps
recursive state and enforces scalar all-pole stability through bounded reflection
coefficients.  This does not remove step-size tuning or identification difficulty
in adaptive systems, but it shows why compact recursive models are attractive
when the physical path has long memory.
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


def median_time(fn, repeats: int) -> tuple[float, np.ndarray]:
    times: list[float] = []
    result: np.ndarray | None = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = np.asarray(fn(), dtype=np.float64)
        times.append(time.perf_counter() - t0)
    assert result is not None
    return statistics.median(times), result


def synthetic_speech_like_signal(samples: int, rng: np.random.Generator) -> np.ndarray:
    """Create a deterministic, dependency-free speech/noise-like input."""
    white = rng.normal(size=samples).astype(np.float64)
    # A tiny AR coloring loop avoids external audio dependencies while making the
    # input less white than a pure RNG sequence.
    x = np.empty_like(white)
    s1 = 0.0
    s2 = 0.0
    for n, v in enumerate(white):
        s1 = 0.92 * s1 + 0.08 * v
        s2 = 0.65 * s2 + 0.35 * s1
        x[n] = s2
    rms = float(np.sqrt(np.mean(x * x)))
    return x / max(rms, 1e-30)


def stable_echo_lattice(
    order: int, max_reflection: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a stable high-order lattice-ladder echo-like model.

    The reflection coefficients are deliberately bounded well inside the unit
    disk.  Ladder taps decay with stage number so the model behaves like a large
    but controlled recursive echo tail instead of an arbitrary unstable IIR.
    """
    rng = np.random.default_rng(seed)
    stages = np.arange(1, order + 1, dtype=np.float64)
    signs = rng.choice(np.array([-1.0, 1.0]), size=order)
    smooth = np.sin(0.071 * stages) + 0.35 * np.sin(0.019 * stages + 0.4)
    envelope = np.exp(-stages / max(order / 3.0, 1.0))
    reflection = max_reflection * envelope * smooth / max(1.0, np.max(np.abs(smooth)))
    reflection *= signs
    reflection = np.clip(reflection, -0.98, 0.98).astype(np.float64)

    ladder_stages = np.arange(order + 1, dtype=np.float64)
    ladder = rng.normal(scale=1.0, size=order + 1) * np.exp(-ladder_stages / max(order / 7.0, 1.0))
    ladder[0] += 1.0
    ladder /= max(np.linalg.norm(ladder), 1e-30)
    ladder *= 0.35
    return reflection, ladder.astype(np.float64)


def maybe_time_fft_tail(x: np.ndarray, echo_taps: int, repeats: int) -> tuple[float, np.ndarray]:
    """Optional FFT/FIR echo-tail timing used only when requested."""
    tail_index = np.arange(echo_taps, dtype=np.float64)
    decay = np.exp(-tail_index / max(echo_taps / 8.0, 1.0))
    h = decay / max(np.linalg.norm(decay), 1e-30)
    n_out = x.size + h.size - 1
    n_fft = 1 << (n_out - 1).bit_length()

    def run() -> np.ndarray:
        spectrum = np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft)
        return np.fft.irfft(spectrum, n_fft)[: x.size]

    return median_time(run, repeats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Large echo-scale stable recursive model stress test."
    )
    parser.add_argument("--samples", type=int, default=1_000_000, help="number of input samples")
    parser.add_argument(
        "--order", type=int, default=512, help="stable lattice-ladder recursive order"
    )
    parser.add_argument(
        "--echo-taps",
        type=int,
        default=131_072,
        help="reference FIR echo-tap count for scale estimates",
    )
    parser.add_argument(
        "--max-reflection", type=float, default=0.45, help="maximum reflection magnitude envelope"
    )
    parser.add_argument("--repeats", type=int, default=3, help="median timing repeats")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--time-fft-tail",
        action="store_true",
        help="also time a dependency-free FFT/FIR tail reference",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.order <= 0:
        raise ValueError("--order must be positive")
    if args.echo_taps <= 0:
        raise ValueError("--echo-taps must be positive")
    if not (0.0 < args.max_reflection < 1.0):
        raise ValueError("--max-reflection must satisfy 0 < max_reflection < 1")

    rng = np.random.default_rng(args.seed)
    x = synthetic_speech_like_signal(args.samples, rng)
    reflection, ladder = stable_echo_lattice(args.order, args.max_reflection, args.seed + 1)

    def run_lattice() -> np.ndarray:
        filt = ld.LatticeLadderIIR(reflection.tolist(), ladder.tolist())
        return filt.process(x)

    iir_time, y = median_time(run_lattice, args.repeats)
    throughput = args.samples / max(iir_time, 1e-30) / 1e6
    stage_rate = args.samples * args.order / max(iir_time, 1e-30) / 1e9
    lattice_stage_visits = args.samples * args.order
    fir_filter_tap_visits = args.samples * args.echo_taps
    fir_lms_tap_visits = 2 * fir_filter_tap_visits
    tap_to_order_ratio = args.echo_taps / args.order

    rows: list[dict[str, object]] = [
        {
            "method": "lattice_ladder_iir_fixed_model",
            "samples": args.samples,
            "order_or_taps": args.order,
            "median_seconds": iir_time,
            "throughput_msamples_per_s": throughput,
            "stage_visits_giga_per_s": stage_rate,
            "output_rms": float(np.sqrt(np.mean(y * y))),
            "max_abs_reflection": float(np.max(np.abs(reflection))),
        },
        {
            "method": "fir_echo_scale_estimate_filter_only",
            "samples": args.samples,
            "order_or_taps": args.echo_taps,
            "coefficient_touches": fir_filter_tap_visits,
            "tap_to_lattice_order_ratio": tap_to_order_ratio,
        },
        {
            "method": "fir_lms_scale_estimate_filter_plus_update",
            "samples": args.samples,
            "order_or_taps": args.echo_taps,
            "coefficient_touches": fir_lms_tap_visits,
            "tap_to_lattice_order_ratio": tap_to_order_ratio,
        },
    ]

    print("large echo-scale stable recursive model stress")
    print("=" * 56)
    print(f"samples: {args.samples:,}")
    print(f"lattice-ladder recursive order: {args.order:,}")
    print(f"max |reflection|: {np.max(np.abs(reflection)):.6f}")
    print(f"recursive state count: {args.order:,}")
    print(f"ladder parameter count: {args.order + 1:,}")
    print(f"median IIR/lattice-ladder time: {iir_time:.6f} s")
    print(f"throughput: {throughput:.2f} million samples/s")
    print(f"stage update rate: {stage_rate:.2f} billion stage-visits/s")
    print(f"output RMS: {np.sqrt(np.mean(y * y)):.6f}")
    print()
    print("echo-scale comparison numbers")
    print("-" * 56)
    print(f"reference FIR echo taps: {args.echo_taps:,}")
    print(f"FIR taps / lattice order: {tap_to_order_ratio:.1f}x")
    print(f"lattice stage visits: {lattice_stage_visits:,}")
    print(f"FIR filter tap visits, direct form: {fir_filter_tap_visits:,}")
    print(f"FIR LMS filter+update tap visits, rough scale: {fir_lms_tap_visits:,}")
    print("note: the tap-visit numbers are scale diagnostics, not an accuracy equivalence claim")

    if args.time_fft_tail:
        fft_time, y_fft = maybe_time_fft_tail(x, args.echo_taps, args.repeats)
        rows.append(
            {
                "method": "optional_fft_fir_tail_reference",
                "samples": args.samples,
                "order_or_taps": args.echo_taps,
                "median_seconds": fft_time,
                "throughput_msamples_per_s": args.samples / max(fft_time, 1e-30) / 1e6,
                "output_rms": float(np.sqrt(np.mean(y_fft * y_fft))),
                "speedup_iir_vs_fft_tail": fft_time / max(iir_time, 1e-30),
            }
        )
        print()
        print("optional FFT/FIR reference")
        print("-" * 56)
        print(f"FFT/FIR median time: {fft_time:.6f} s")
        print(f"IIR/lattice speedup over FFT/FIR tail: {fft_time / max(iir_time, 1e-30):.2f}x")
        print(f"FFT/FIR output RMS: {np.sqrt(np.mean(y_fft * y_fft)):.6f}")

    out_dir = artifact_dir()
    csv_path = out_dir / "large_order_echo_stress.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
