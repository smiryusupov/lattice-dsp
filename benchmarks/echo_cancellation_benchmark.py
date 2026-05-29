"""Small synthetic echo-cancellation metric benchmark.

The goal is to expose ERLE/MSE behavior for adaptive filters on a controlled
echo-like problem.  This is not a production AEC benchmark.  It compares no
cancellation, a simple FIR/NLMS baseline, lattice/IIR only, and small
dependency-free residual suppressor baselines.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from lattice_dsp import (
    HAS_OPENMP,
    HybridEchoCanceller,
    SpectralResidualSuppressor,
    echo_metrics,
    generate_echo_problem,
    residual_attenuator,
)


def time_call(fn: Callable[[], Any], repeats: int) -> tuple[Any, dict[str, float]]:
    timings: list[float] = []
    value: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        timings.append(time.perf_counter() - start)
    return value, {
        "min_s": min(timings),
        "median_s": statistics.median(timings),
        "max_s": max(timings),
    }


def fir_nlms(
    reference: np.ndarray,
    desired: np.ndarray,
    *,
    order: int = 64,
    mu: float = 0.5,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small FIR/NLMS baseline implemented in NumPy/Python."""

    if order <= 0:
        raise ValueError("order must be positive")
    x = np.asarray(reference, dtype=np.float64)
    d = np.asarray(desired, dtype=np.float64)
    if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
        raise ValueError("reference and desired must be equally shaped 1-D arrays")

    w = np.zeros(order, dtype=np.float64)
    xbuf = np.zeros(order, dtype=np.float64)
    y = np.zeros_like(x)
    e = np.zeros_like(x)
    for n, sample in enumerate(x):
        xbuf[1:] = xbuf[:-1]
        xbuf[0] = sample
        y_n = float(np.dot(w, xbuf))
        e_n = float(d[n] - y_n)
        norm = float(np.dot(xbuf, xbuf) + epsilon)
        w += (mu * e_n / norm) * xbuf
        y[n] = y_n
        e[n] = e_n
    return y, e, w


def summarize_case(
    name: str,
    microphone: np.ndarray,
    enhanced: np.ndarray,
    clean_target: np.ndarray,
    timing: dict[str, float] | None = None,
) -> dict[str, float | str]:
    metrics = echo_metrics(microphone, enhanced, clean_target).as_dict()
    row: dict[str, float | str] = {"name": name, **metrics}
    if timing:
        row.update(timing)
    return row


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    # Backward-compatible defaults for tests or callers that construct the
    # Namespace manually instead of using ``build_parser``.
    for name, default in {
        "spectral_frame_size": 512,
        "spectral_hop_size": None,
        "spectral_floor": 0.08,
        "spectral_over_subtract": 1.25,
        "spectral_noise_percentile": 20.0,
        "spectral_smoothing": 0.65,
        "spectral_exponent": 1.0,
        "spectral_mode": "echo_aware",
        "spectral_echo_aware_strength": 0.85,
        "spectral_reference_key": "echo_estimate",
    }.items():
        if not hasattr(args, name):
            setattr(args, name, default)

    problem = generate_echo_problem(
        samples=args.samples,
        sample_rate=args.sample_rate,
        seed=args.seed,
        nonlinear_strength=args.nonlinear_strength,
        nonlinearity=args.nonlinearity,
        near_end_power_ratio=args.near_end_power_ratio,
        noise_snr_db=args.noise_snr_db,
        double_talk=not args.no_double_talk,
    )

    cases: list[dict[str, float | str]] = []
    cases.append(
        summarize_case(
            "no_cancellation",
            problem.microphone,
            problem.microphone,
            problem.clean_target,
            {"min_s": 0.0, "median_s": 0.0, "max_s": 0.0},
        )
    )

    residual_only, residual_only_timing = time_call(
        lambda: residual_attenuator(problem.microphone, gain=args.residual_gain),
        args.repeats,
    )
    cases.append(
        summarize_case(
            "toy_residual_suppressor_only",
            problem.microphone,
            residual_only,
            problem.clean_target,
            residual_only_timing,
        )
    )

    spectral_processor = SpectralResidualSuppressor(
        frame_size=args.spectral_frame_size,
        hop_size=args.spectral_hop_size,
        floor=args.spectral_floor,
        over_subtract=args.spectral_over_subtract,
        noise_percentile=args.spectral_noise_percentile,
        smoothing=args.spectral_smoothing,
        exponent=args.spectral_exponent,
        mode=args.spectral_mode,
        echo_aware_strength=args.spectral_echo_aware_strength,
        reference_key=args.spectral_reference_key,
    )
    spectral_only, spectral_only_timing = time_call(
        lambda: spectral_processor(
            problem.microphone,
            {
                "sample_rate": args.sample_rate,
                "reference": problem.reference,
            },
        ),
        args.repeats,
    )
    cases.append(
        summarize_case(
            "spectral_residual_suppressor_only",
            problem.microphone,
            spectral_only,
            problem.clean_target,
            spectral_only_timing,
        )
    )

    fir_result, fir_timing = time_call(
        lambda: fir_nlms(
            problem.reference,
            problem.microphone,
            order=args.fir_order,
            mu=args.fir_mu,
            epsilon=args.epsilon,
        ),
        args.repeats,
    )
    _, fir_residual, fir_weights = fir_result
    cases.append(
        summarize_case(
            "fir_nlms_baseline",
            problem.microphone,
            fir_residual,
            problem.clean_target,
            fir_timing,
        )
    )

    def make_canceller(*, residual_mode: str | None = None) -> HybridEchoCanceller:
        processor = None
        if residual_mode == "toy":

            def processor(residual: np.ndarray, context: dict[str, Any]) -> np.ndarray:
                return residual_attenuator(residual, gain=args.residual_gain)

        elif residual_mode == "spectral":
            processor = spectral_processor
        elif residual_mode is not None:
            raise ValueError(f"unknown residual_mode: {residual_mode}")

        return HybridEchoCanceller(
            initial_reflection=[0.0] * args.iir_order,
            initial_taps=[0.0] * (args.iir_order + 1),
            mu_taps=args.mu_taps,
            mu_reflection=args.mu_reflection,
            epsilon=args.epsilon,
            reflection_update_period=args.reflection_update_period,
            scale_reflection_mu_by_period=not args.no_scale_reflection_mu_by_period,
            residual_processor=processor,
            sample_rate=args.sample_rate,
        )

    def run_lattice_only() -> Any:
        canceller = make_canceller()
        return canceller.process(
            problem.reference, problem.microphone, clean_target=problem.clean_target
        )

    lattice_result, lattice_timing = time_call(run_lattice_only, args.repeats)
    cases.append(
        summarize_case(
            "lattice_iir_only",
            problem.microphone,
            lattice_result.residual,
            problem.clean_target,
            lattice_timing,
        )
    )

    def run_hybrid() -> Any:
        canceller = make_canceller(residual_mode="toy")
        return canceller.process(
            problem.reference, problem.microphone, clean_target=problem.clean_target
        )

    hybrid_result, hybrid_timing = time_call(run_hybrid, args.repeats)
    cases.append(
        summarize_case(
            "lattice_iir_plus_toy_residual_suppressor",
            problem.microphone,
            hybrid_result.enhanced,
            problem.clean_target,
            hybrid_timing,
        )
    )

    def run_spectral_hybrid() -> Any:
        canceller = make_canceller(residual_mode="spectral")
        return canceller.process(
            problem.reference, problem.microphone, clean_target=problem.clean_target
        )

    spectral_hybrid_result, spectral_hybrid_timing = time_call(run_spectral_hybrid, args.repeats)
    cases.append(
        summarize_case(
            "lattice_iir_plus_spectral_residual_suppressor",
            problem.microphone,
            spectral_hybrid_result.enhanced,
            problem.clean_target,
            spectral_hybrid_timing,
        )
    )

    best = max(cases, key=lambda row: float(row["erle_db"]))
    return {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "has_openmp": HAS_OPENMP,
            "samples": args.samples,
            "sample_rate": args.sample_rate,
            "seed": args.seed,
            "repeats": args.repeats,
            "nonlinearity": args.nonlinearity,
            "nonlinear_strength": args.nonlinear_strength,
            "near_end_power_ratio": args.near_end_power_ratio,
            "noise_snr_db": args.noise_snr_db,
            "iir_order": args.iir_order,
            "fir_order": args.fir_order,
            "reflection_update_period": args.reflection_update_period,
            "scale_reflection_mu_by_period": not args.no_scale_reflection_mu_by_period,
            "residual_gain": args.residual_gain,
            "spectral_frame_size": args.spectral_frame_size,
            "spectral_hop_size": args.spectral_hop_size,
            "spectral_floor": args.spectral_floor,
            "spectral_over_subtract": args.spectral_over_subtract,
            "spectral_noise_percentile": args.spectral_noise_percentile,
            "spectral_smoothing": args.spectral_smoothing,
            "spectral_exponent": args.spectral_exponent,
            "spectral_mode": args.spectral_mode,
            "spectral_echo_aware_strength": args.spectral_echo_aware_strength,
            "spectral_reference_key": args.spectral_reference_key,
            "target_reflection": problem.reflection.tolist(),
            "target_taps": problem.taps.tolist(),
            "fir_weight_norm": float(np.linalg.norm(fir_weights)),
            "final_lattice_reflection": lattice_result.reflection.tolist(),
            "final_hybrid_reflection": hybrid_result.reflection.tolist(),
        },
        "cases": cases,
        "best_by_erle": best,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64_000)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--nonlinearity", choices=["none", "tanh", "cubic", "clipped"], default="tanh"
    )
    parser.add_argument("--nonlinear-strength", type=float, default=0.08)
    parser.add_argument("--near-end-power-ratio", type=float, default=0.02)
    parser.add_argument("--noise-snr-db", type=float, default=30.0)
    parser.add_argument("--no-double-talk", action="store_true")
    parser.add_argument("--iir-order", type=int, default=4)
    parser.add_argument("--fir-order", type=int, default=64)
    parser.add_argument("--fir-mu", type=float, default=0.5)
    parser.add_argument("--mu-taps", type=float, default=0.05)
    parser.add_argument("--mu-reflection", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--reflection-update-period", type=int, default=8)
    parser.add_argument("--no-scale-reflection-mu-by-period", action="store_true")
    parser.add_argument("--residual-gain", type=float, default=0.7)
    parser.add_argument("--spectral-frame-size", type=int, default=512)
    parser.add_argument("--spectral-hop-size", type=int, default=None)
    parser.add_argument("--spectral-floor", type=float, default=0.08)
    parser.add_argument("--spectral-over-subtract", type=float, default=1.25)
    parser.add_argument("--spectral-noise-percentile", type=float, default=20.0)
    parser.add_argument("--spectral-smoothing", type=float, default=0.65)
    parser.add_argument("--spectral-exponent", type=float, default=1.0)
    parser.add_argument("--spectral-mode", choices=["echo_aware", "blind"], default="echo_aware")
    parser.add_argument("--spectral-echo-aware-strength", type=float, default=0.85)
    parser.add_argument(
        "--spectral-reference-key",
        choices=["echo_estimate", "reference"],
        default="echo_estimate",
        help="Context signal used by echo-aware spectral residual suppression.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/echo-benchmark.json"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_benchmark(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
