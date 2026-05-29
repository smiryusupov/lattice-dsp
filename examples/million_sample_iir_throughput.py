"""Million-sample throughput: long acoustic-like tail as a low-order IIR.

This example shows the speed motivation for recursive models.  A long FIR tail
can represent an acoustic or decay response by storing many taps.  When the tail
has a compact recursive description, an IIR/lattice filter can process millions
of samples with a small fixed state instead of carrying the full tail length.

The example uses a first-order stable IIR whose impulse response is an exponential
reverberant-like decay,

    h[m] = (1 - r) r**m,    0 < r < 1.

The equivalent FIR approximation truncates that tail to many taps.  The timing
comparison is intentionally local to your machine and NumPy build; it is a
reproducibility aid, not a universal benchmark number.
"""

from __future__ import annotations

import argparse
import csv
import math
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


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fft_convolve_truncated(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Dependency-free full FFT convolution, truncated to len(x)."""
    n_out = int(x.size + h.size - 1)
    n_fft = next_power_of_two(n_out)
    spectrum = np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft)
    return np.fft.irfft(spectrum, n_fft)[: x.size]


def relative_rms_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    err = reference - estimate
    return float(np.sqrt(np.mean(err * err)) / (np.sqrt(np.mean(reference * reference)) + 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description="Million-sample IIR throughput demonstration.")
    parser.add_argument("--samples", type=int, default=1_000_000, help="number of input samples")
    parser.add_argument(
        "--tail-taps", type=int, default=131_072, help="FIR taps used to truncate the IIR tail"
    )
    parser.add_argument(
        "--pole",
        type=float,
        default=0.99992,
        help="stable IIR pole radius for the exponential tail",
    )
    parser.add_argument("--repeats", type=int, default=3, help="median timing repeats")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-fft", action="store_true", help="skip the FFT/FIR reference timing")
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.tail_taps <= 0:
        raise ValueError("--tail-taps must be positive")
    if not (0.0 < args.pole < 1.0):
        raise ValueError("--pole must satisfy 0 < pole < 1")

    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=args.samples).astype(np.float64)

    # Under the lattice convention used by lattice-dsp, a first-order reflection
    # coefficient k gives A(z) = 1 + k z^-1.  The pole is therefore -k.  Choosing
    # k = -pole gives the stable smoother y[n] = (1-pole) x[n] + pole y[n-1].
    reflection = [-float(args.pole)]
    numerator = [1.0 - float(args.pole), 0.0]

    iir_time, y_iir = median_time(
        lambda: ld.LatticeIIR(reflection, numerator).process(x), args.repeats
    )

    rows: list[dict[str, object]] = [
        {
            "method": "lattice_iir_recursive",
            "samples": args.samples,
            "state_or_taps": len(reflection),
            "median_seconds": iir_time,
            "throughput_msamples_per_s": args.samples / max(iir_time, 1e-30) / 1e6,
            "relative_rms_error_vs_iir": 0.0,
        }
    ]

    print("million-sample IIR throughput demonstration")
    print("=" * 52)
    print(f"samples: {args.samples:,}")
    print(f"stable pole radius: {args.pole:.8f}")
    print(f"IIR reflection coefficient: {reflection[0]:.8f}")
    print(f"IIR recursive state count: {len(reflection)}")
    print(f"IIR median time: {iir_time:.6f} s")
    print(f"IIR throughput: {args.samples / max(iir_time, 1e-30) / 1e6:.2f} million samples/s")

    if not args.skip_fft:
        # Finite FIR approximation to the same infinite exponential tail.  This is
        # not meant as the slowest possible comparison: FFT convolution is already
        # a strong baseline for long FIR filtering.  The point is that the IIR has
        # constant state for this structured tail, while the FIR path stores and
        # transforms many coefficients.
        taps = (1.0 - args.pole) * np.power(args.pole, np.arange(args.tail_taps, dtype=np.float64))
        fft_time, y_fir = median_time(lambda: fft_convolve_truncated(x, taps), args.repeats)
        rel_err = relative_rms_error(y_iir, y_fir)
        n_fft = next_power_of_two(args.samples + args.tail_taps - 1)
        speedup = fft_time / max(iir_time, 1e-30)
        rows.append(
            {
                "method": "truncated_fir_fft_convolution",
                "samples": args.samples,
                "state_or_taps": args.tail_taps,
                "median_seconds": fft_time,
                "throughput_msamples_per_s": args.samples / max(fft_time, 1e-30) / 1e6,
                "relative_rms_error_vs_iir": rel_err,
                "fft_length": n_fft,
                "speedup_iir_vs_fft_fir": speedup,
            }
        )
        print()
        print(f"FIR truncation taps: {args.tail_taps:,}")
        print(f"FFT length for FIR reference: {n_fft:,}")
        print(f"FFT/FIR median time: {fft_time:.6f} s")
        print(f"IIR speedup over FFT/FIR reference: {speedup:.2f}x")
        print(f"relative RMS error from truncating the infinite IIR tail: {rel_err:.3e}")
        print(
            f"omitted tail amplitude after {args.tail_taps:,} taps: {math.pow(args.pole, args.tail_taps):.3e}"
        )

    out_dir = artifact_dir()
    csv_path = out_dir / "million_sample_iir_throughput.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
