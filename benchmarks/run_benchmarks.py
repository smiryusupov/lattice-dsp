"""Run small performance benchmarks for lattice-dsp.

The benchmark intentionally measures three different things:

1. A pure-Python reference loop, which shows the cost avoided by the C++ core.
2. SciPy's lfilter, when SciPy is installed, as a mature static-filter baseline.
3. lattice-dsp's C++ pybind11 path, including OpenMP over independent rows.

This is not a scientific paper benchmark. It is a reproducible sanity benchmark
that can be run locally or in CI to catch performance regressions and produce a
JSON artifact for comparison.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lattice_dsp import (
    HAS_OPENMP,
    AdaptiveLatticeLadderNLMS,
    adaptive_process_batch,
    LatticeIIR,
    LatticeLadderIIR,
    numerator_to_ladder,
    process_batch,
    reflection_to_denominator,
)

try:  # pragma: no cover - optional benchmark dependency
    from scipy.signal import lfilter
except Exception:  # pragma: no cover
    lfilter = None


def pure_python_lfilter(b: list[float], a: list[float], x: np.ndarray) -> np.ndarray:
    """Simple direct-form reference for one stream. Kept deliberately unoptimized."""
    y = np.zeros_like(x, dtype=float)
    for n in range(x.size):
        acc = 0.0
        for i, coeff in enumerate(b):
            if n - i >= 0:
                acc += coeff * float(x[n - i])
        for i in range(1, len(a)):
            if n - i >= 0:
                acc -= a[i] * float(y[n - i])
        y[n] = acc / a[0]
    return y


def time_call(fn: Callable[[], object], repeats: int) -> dict[str, float]:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return {
        "min_s": min(timings),
        "median_s": statistics.median(timings),
        "max_s": max(timings),
    }


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    x_batch = rng.normal(size=(args.channels, args.samples)).astype(np.float64)
    x_one = x_batch[0].copy()

    reflection = [0.35, -0.25, 0.15, -0.08]
    taps = [0.2, -0.1, 0.05, 0.0, 0.75]
    denominator = reflection_to_denominator(reflection)

    results: dict[str, object] = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "channels": args.channels,
            "samples": args.samples,
            "repeats": args.repeats,
            "has_openmp": HAS_OPENMP,
            "reflection_update_period": args.reflection_update_period,
            "scale_reflection_mu_by_period": args.scale_reflection_mu_by_period,
        },
        "benchmarks": {},
    }

    one_filter = LatticeIIR(reflection, taps)
    results["benchmarks"]["lattice_direct_cpp_one_stream"] = time_call(  # type: ignore[index]
        lambda: one_filter.process(x_one), args.repeats
    )

    ladder = numerator_to_ladder(reflection, taps)
    one_ladder_filter = LatticeLadderIIR(reflection, ladder)
    results["benchmarks"]["lattice_ladder_cpp_one_stream"] = time_call(  # type: ignore[index]
        lambda: one_ladder_filter.process(x_one), args.repeats
    )

    results["benchmarks"]["lattice_direct_cpp_batch_default_threads"] = time_call(  # type: ignore[index]
        lambda: process_batch(reflection, taps, x_batch, n_threads=0, realization="direct"),
        args.repeats,
    )

    results["benchmarks"]["lattice_direct_cpp_batch_one_thread"] = time_call(  # type: ignore[index]
        lambda: process_batch(reflection, taps, x_batch, n_threads=1, realization="direct"),
        args.repeats,
    )

    results["benchmarks"]["lattice_ladder_cpp_batch_default_threads"] = time_call(  # type: ignore[index]
        lambda: process_batch(reflection, taps, x_batch, n_threads=0, realization="lattice"),
        args.repeats,
    )

    results["benchmarks"]["lattice_ladder_cpp_batch_one_thread"] = time_call(  # type: ignore[index]
        lambda: process_batch(reflection, taps, x_batch, n_threads=1, realization="lattice"),
        args.repeats,
    )

    if args.include_adaptive:
        target = LatticeIIR(reflection, taps)
        desired = np.asarray(target.process(x_one), dtype=np.float64)

        def run_adaptive_analytic_numpy_block() -> None:
            adaptive = AdaptiveLatticeLadderNLMS(
                [0.0] * len(reflection),
                [0.0] * len(taps),
                mu_taps=0.05,
                mu_reflection=0.001,
                freeze_reflection=False,
                gradient_mode="analytic",
                reflection_update_period=args.reflection_update_period,
                scale_reflection_mu_by_period=args.scale_reflection_mu_by_period,
            )
            adaptive.process_adapt(x_one, desired)

        results["benchmarks"]["adaptive_lattice_ladder_nlms_analytic_numpy_block"] = time_call(  # type: ignore[index]
            run_adaptive_analytic_numpy_block, max(1, min(args.repeats, 3))
        )

        def run_adaptive_analytic_list_block() -> None:
            adaptive = AdaptiveLatticeLadderNLMS(
                [0.0] * len(reflection),
                [0.0] * len(taps),
                mu_taps=0.05,
                mu_reflection=0.001,
                freeze_reflection=False,
                gradient_mode="analytic",
                reflection_update_period=args.reflection_update_period,
                scale_reflection_mu_by_period=args.scale_reflection_mu_by_period,
            )
            adaptive.adapt_block(x_one.tolist(), desired.tolist())

        results["benchmarks"]["adaptive_lattice_ladder_nlms_analytic_list_block"] = time_call(  # type: ignore[index]
            run_adaptive_analytic_list_block, max(1, min(args.repeats, 3))
        )

        if args.include_adaptive_sample_loop:

            def run_adaptive_analytic_sample_loop() -> None:
                adaptive = AdaptiveLatticeLadderNLMS(
                    [0.0] * len(reflection),
                    [0.0] * len(taps),
                    mu_taps=0.05,
                    mu_reflection=0.001,
                    freeze_reflection=False,
                    gradient_mode="analytic",
                    reflection_update_period=args.reflection_update_period,
                    scale_reflection_mu_by_period=args.scale_reflection_mu_by_period,
                )
                for xn, dn in zip(x_one, desired, strict=True):
                    adaptive.adapt_sample(float(xn), float(dn))

            results["benchmarks"]["adaptive_lattice_ladder_nlms_analytic_sample_loop"] = time_call(  # type: ignore[index]
                run_adaptive_analytic_sample_loop, max(1, min(args.repeats, 3))
            )

        if args.include_finite_difference_adaptive:

            def run_adaptive_finite_difference_numpy_block() -> None:
                adaptive = AdaptiveLatticeLadderNLMS(
                    [0.0] * len(reflection),
                    [0.0] * len(taps),
                    mu_taps=0.05,
                    mu_reflection=0.001,
                    freeze_reflection=False,
                    gradient_mode="finite_difference",
                    reflection_update_period=args.reflection_update_period,
                    scale_reflection_mu_by_period=args.scale_reflection_mu_by_period,
                )
                adaptive.process_adapt(x_one, desired)

            results["benchmarks"]["adaptive_lattice_ladder_nlms_finite_difference_numpy_block"] = (
                time_call(  # type: ignore[index]
                    run_adaptive_finite_difference_numpy_block, max(1, min(args.repeats, 3))
                )
            )

        if args.include_adaptive_batch:
            batch_channels = min(args.channels, args.adaptive_batch_channels)
            x_adapt_batch = x_batch[:batch_channels].copy()
            desired_batch = np.asarray(
                process_batch(reflection, taps, x_adapt_batch, n_threads=0, realization="direct"),
                dtype=np.float64,
            )

            results["metadata"]["adaptive_batch_channels"] = batch_channels  # type: ignore[index]
            results["benchmarks"]["adaptive_lattice_ladder_nlms_analytic_batch_default_threads"] = (
                time_call(  # type: ignore[index]
                    lambda: adaptive_process_batch(
                        [0.0] * len(reflection),
                        [0.0] * len(taps),
                        x_adapt_batch,
                        desired_batch,
                        mu_taps=0.05,
                        mu_reflection=0.001,
                        freeze_reflection=False,
                        gradient_mode="analytic",
                        reflection_update_period=args.reflection_update_period,
                        scale_reflection_mu_by_period=args.scale_reflection_mu_by_period,
                        n_threads=0,
                    ),
                    max(1, min(args.repeats, 3)),
                )
            )
            results["benchmarks"]["adaptive_lattice_ladder_nlms_analytic_batch_one_thread"] = (
                time_call(  # type: ignore[index]
                    lambda: adaptive_process_batch(
                        [0.0] * len(reflection),
                        [0.0] * len(taps),
                        x_adapt_batch,
                        desired_batch,
                        mu_taps=0.05,
                        mu_reflection=0.001,
                        freeze_reflection=False,
                        gradient_mode="analytic",
                        reflection_update_period=args.reflection_update_period,
                        scale_reflection_mu_by_period=args.scale_reflection_mu_by_period,
                        n_threads=1,
                    ),
                    max(1, min(args.repeats, 3)),
                )
            )

    if args.include_python_reference:
        results["benchmarks"]["pure_python_one_stream"] = time_call(  # type: ignore[index]
            lambda: pure_python_lfilter(taps, denominator, x_one), max(1, min(args.repeats, 3))
        )

    if lfilter is not None:
        results["benchmarks"]["scipy_lfilter_one_stream"] = time_call(  # type: ignore[index]
            lambda: lfilter(taps, denominator, x_one), args.repeats
        )
        results["benchmarks"]["scipy_lfilter_batch_axis1"] = time_call(  # type: ignore[index]
            lambda: lfilter(taps, denominator, x_batch, axis=1), args.repeats
        )
    else:
        results["benchmarks"]["scipy_lfilter"] = {"skipped": "SciPy is not installed"}  # type: ignore[index]

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark-results.json"))
    parser.add_argument(
        "--include-python-reference",
        action="store_true",
        help="Include a deliberately slow pure-Python direct-form reference.",
    )
    parser.add_argument(
        "--include-adaptive",
        action="store_true",
        help="Include the experimental adaptive IIR update path. Slower than static filtering.",
    )
    parser.add_argument(
        "--include-adaptive-sample-loop",
        action="store_true",
        help="Also benchmark the slow Python-level per-sample adapt_sample loop.",
    )
    parser.add_argument(
        "--include-adaptive-batch",
        action="store_true",
        help="Also benchmark independent adaptive problems over a 2-D batch.",
    )
    parser.add_argument(
        "--adaptive-batch-channels",
        type=int,
        default=8,
        help="Maximum number of channels used by the adaptive batch benchmark.",
    )
    parser.add_argument(
        "--reflection-update-period",
        type=int,
        default=1,
        help="Update reflection/raw denominator parameters every K samples in adaptive benchmarks.",
    )
    parser.add_argument(
        "--scale-reflection-mu-by-period",
        action="store_true",
        help="Use mu_reflection * reflection_update_period on denominator-update samples. Useful for fair period sweeps.",
    )
    parser.add_argument(
        "--include-finite-difference-adaptive",
        action="store_true",
        help="Also benchmark the slow finite-difference adaptive gradient reference.",
    )
    args = parser.parse_args()

    if args.channels <= 0 or args.samples <= 0 or args.repeats <= 0:
        raise SystemExit("channels, samples, and repeats must all be positive")
    if args.adaptive_batch_channels <= 0:
        raise SystemExit("adaptive-batch-channels must be positive")
    if args.reflection_update_period <= 0:
        raise SystemExit("reflection-update-period must be positive")

    results = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
