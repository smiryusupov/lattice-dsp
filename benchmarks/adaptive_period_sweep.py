"""Sweep adaptive reflection-update periods and record speed/quality trade-offs.

The adaptive IIR update has two different costs:

* numerator/ladder updates, which are cheap and can run every sample; and
* reflection/denominator updates, which are more expensive and often noisier.

By default, the script enables period-scaled reflection steps so period K uses
``mu_reflection * K`` on update samples. This makes the speed/quality comparison
fairer than giving long periods K-times fewer effective denominator updates.

This script evaluates that trade-off by running the same identification problem
with several ``reflection_update_period`` values.  It reports runtime plus MSE
and coefficient-error metrics, then writes JSON and optional CSV output.

Example
-------
python benchmarks/adaptive_period_sweep.py --periods 1 2 4 8 16 32 \
    --samples 20000 --repeats 5 --output adaptive-period-sweep.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from lattice_dsp import AdaptiveLatticeLadderNLMS, HAS_OPENMP, LatticeIIR


def parse_periods(values: list[str]) -> list[int]:
    """Parse period CLI values, accepting either space- or comma-separated input."""
    periods: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            period = int(token)
            if period <= 0:
                raise ValueError("reflection update periods must be positive")
            periods.append(period)
    if not periods:
        raise ValueError("at least one period is required")
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(periods))


def mse(x: np.ndarray) -> float:
    return float(np.mean(np.square(x)))


def run_trial(
    period: int,
    x: np.ndarray,
    desired: np.ndarray,
    target_reflection: np.ndarray,
    target_taps: np.ndarray,
    *,
    mu_taps: float,
    mu_reflection: float,
    epsilon: float,
    margin: float,
    tail: int,
    scale_reflection_mu_by_period: bool,
) -> tuple[float, dict[str, float]]:
    adaptive = AdaptiveLatticeLadderNLMS(
        [0.0] * int(target_reflection.size),
        [0.0] * int(target_taps.size),
        mu_taps=mu_taps,
        mu_reflection=mu_reflection,
        epsilon=epsilon,
        margin=margin,
        freeze_reflection=False,
        gradient_mode="analytic",
        reflection_update_period=period,
        scale_reflection_mu_by_period=scale_reflection_mu_by_period,
    )

    start = time.perf_counter()
    y, err = adaptive.process_adapt(x, desired)
    elapsed = time.perf_counter() - start

    err = np.asarray(err, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    final_reflection = np.asarray(adaptive.reflection, dtype=np.float64)
    final_taps = np.asarray(adaptive.taps, dtype=np.float64)
    tail_n = min(tail, err.size)

    quality = {
        "mse_total": mse(err),
        "mse_head": mse(err[:tail_n]),
        "mse_tail": mse(err[-tail_n:]),
        "output_power": mse(y),
        "reflection_l2_error": float(np.linalg.norm(final_reflection - target_reflection)),
        "taps_l2_error": float(np.linalg.norm(final_taps - target_taps)),
        "max_abs_reflection": float(np.max(np.abs(final_reflection)))
        if final_reflection.size
        else 0.0,
        "stability_margin": float(1.0 - np.max(np.abs(final_reflection)))
        if final_reflection.size
        else 1.0,
    }
    return elapsed, quality


def sweep(args: argparse.Namespace) -> dict[str, object]:
    periods = parse_periods(args.periods)
    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=args.samples).astype(np.float64)

    target_reflection = np.asarray(args.target_reflection, dtype=np.float64)
    target_taps = np.asarray(args.target_taps, dtype=np.float64)
    target = LatticeIIR(target_reflection.tolist(), target_taps.tolist())
    desired = np.asarray(target.process(x), dtype=np.float64)
    if args.noise_std > 0.0:
        desired = desired + rng.normal(scale=args.noise_std, size=desired.shape)

    rows: list[dict[str, object]] = []
    baseline_median: float | None = None
    baseline_tail_mse: float | None = None

    for period in periods:
        timings: list[float] = []
        quality: dict[str, float] | None = None
        for _ in range(args.repeats):
            elapsed, quality = run_trial(
                period,
                x,
                desired,
                target_reflection,
                target_taps,
                mu_taps=args.mu_taps,
                mu_reflection=args.mu_reflection,
                epsilon=args.epsilon,
                margin=args.margin,
                tail=args.tail,
                scale_reflection_mu_by_period=args.scale_reflection_mu_by_period,
            )
            timings.append(elapsed)
        assert quality is not None
        median_s = statistics.median(timings)
        if baseline_median is None:
            baseline_median = median_s
            baseline_tail_mse = quality["mse_tail"]

        row: dict[str, object] = {
            "reflection_update_period": period,
            "min_s": min(timings),
            "median_s": median_s,
            "max_s": max(timings),
            "speedup_vs_first_period": (baseline_median / median_s) if median_s > 0.0 else None,
            **quality,
            "tail_mse_ratio_vs_first_period": (
                quality["mse_tail"] / baseline_tail_mse
                if baseline_tail_mse and baseline_tail_mse > 0.0
                else None
            ),
        }
        rows.append(row)

    return {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "has_openmp": HAS_OPENMP,
            "samples": args.samples,
            "repeats": args.repeats,
            "seed": args.seed,
            "periods": periods,
            "mu_taps": args.mu_taps,
            "mu_reflection": args.mu_reflection,
            "scale_reflection_mu_by_period": args.scale_reflection_mu_by_period,
            "epsilon": args.epsilon,
            "margin": args.margin,
            "noise_std": args.noise_std,
            "tail": min(args.tail, args.samples),
            "target_reflection": target_reflection.tolist(),
            "target_taps": target_taps.tolist(),
        },
        "results": rows,
    }


def write_csv(path: Path, results: dict[str, object]) -> None:
    rows = results["results"]
    if not isinstance(rows, list) or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", nargs="+", default=["1", "2", "4", "8", "16", "32"])
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tail", type=int, default=2_000)
    parser.add_argument("--mu-taps", type=float, default=0.05)
    parser.add_argument("--mu-reflection", type=float, default=0.001)
    parser.add_argument(
        "--no-scale-reflection-mu-by-period",
        dest="scale_reflection_mu_by_period",
        action="store_false",
        help="Disable period scaling and keep the same raw mu_reflection for every update period.",
    )
    parser.set_defaults(scale_reflection_mu_by_period=True)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--margin", type=float, default=1e-4)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument(
        "--target-reflection", nargs="+", type=float, default=[0.35, -0.25, 0.15, -0.08]
    )
    parser.add_argument(
        "--target-taps", nargs="+", type=float, default=[0.2, -0.1, 0.05, 0.0, 0.75]
    )
    parser.add_argument("--output", type=Path, default=Path("reports/adaptive-period-sweep.json"))
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()

    if args.samples <= 0 or args.repeats <= 0 or args.tail <= 0:
        raise SystemExit("samples, repeats, and tail must all be positive")
    if len(args.target_taps) != len(args.target_reflection) + 1:
        raise SystemExit("target-taps must have length len(target-reflection) + 1")
    if args.noise_std < 0.0:
        raise SystemExit("noise-std must be non-negative")

    results = sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nWrote {args.output}")
    if args.csv_output is not None:
        write_csv(args.csv_output, results)
        print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
