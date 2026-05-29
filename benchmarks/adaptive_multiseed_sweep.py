"""Run adaptive reflection-period sweeps over multiple random seeds.

Single-seed period sweeps are useful for debugging, but they can over-reward a
period that happens to align well with one signal realization.  This script runs
``benchmarks/adaptive_period_sweep.py``-style trials across many seeds and then
aggregates speed/quality metrics per ``reflection_update_period``.

The main robustness metrics are:

* median tail-MSE ratio across seeds;
* worst-case tail-MSE ratio across seeds;
* median speedup across seeds; and
* minimum stability margin across seeds.

Example
-------
python benchmarks/adaptive_multiseed_sweep.py \
    --seeds 100 101 102 103 104 \
    --periods 1 2 4 8 16 32 \
    --samples 20000 \
    --repeats 3 \
    --output adaptive-multiseed-sweep.json \
    --csv-output adaptive-multiseed-aggregate.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from lattice_dsp import AdaptiveLatticeLadderNLMS, HAS_OPENMP, LatticeIIR
from benchmarks.adaptive_period_sweep import parse_periods


AGGREGATE_COLUMNS = [
    "reflection_update_period",
    "n_seeds",
    "median_s_median",
    "median_s_min",
    "median_s_max",
    "speedup_vs_period1_median",
    "speedup_vs_period1_min",
    "speedup_vs_period1_max",
    "mse_tail_median",
    "mse_tail_max",
    "tail_mse_ratio_median",
    "tail_mse_ratio_max",
    "tail_mse_ratio_p90",
    "mse_total_median",
    "reflection_l2_error_median",
    "reflection_l2_error_max",
    "taps_l2_error_median",
    "taps_l2_error_max",
    "stability_margin_median",
    "stability_margin_min",
]


PER_SEED_COLUMNS = [
    "seed",
    "reflection_update_period",
    "min_s",
    "median_s",
    "max_s",
    "speedup_vs_period1",
    "mse_total",
    "mse_head",
    "mse_tail",
    "tail_mse_ratio_vs_period1",
    "output_power",
    "reflection_l2_error",
    "taps_l2_error",
    "max_abs_reflection",
    "stability_margin",
]


def parse_seeds(values: list[str]) -> list[int]:
    """Parse seed CLI values, accepting either space- or comma-separated input."""
    seeds: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                seeds.append(int(token))
    if not seeds:
        raise ValueError("at least one seed is required")
    return list(dict.fromkeys(seeds))


def mse(x: np.ndarray) -> float:
    return float(np.mean(np.square(x)))


def quantile(values: list[float], q: float) -> float:
    """Small dependency-free quantile helper with linear interpolation."""
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


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

    return elapsed, {
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


def run_seed(
    seed: int,
    periods: list[int],
    *,
    samples: int,
    repeats: int,
    mu_taps: float,
    mu_reflection: float,
    epsilon: float,
    margin: float,
    noise_std: float,
    tail: int,
    target_reflection: np.ndarray,
    target_taps: np.ndarray,
    scale_reflection_mu_by_period: bool,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=samples).astype(np.float64)

    target = LatticeIIR(target_reflection.tolist(), target_taps.tolist())
    desired = np.asarray(target.process(x), dtype=np.float64)
    if noise_std > 0.0:
        desired = desired + rng.normal(scale=noise_std, size=desired.shape)

    rows: list[dict[str, Any]] = []
    baseline_median: float | None = None
    baseline_tail_mse: float | None = None

    for period in periods:
        timings: list[float] = []
        quality: dict[str, float] | None = None
        for _ in range(repeats):
            elapsed, quality = run_trial(
                period,
                x,
                desired,
                target_reflection,
                target_taps,
                mu_taps=mu_taps,
                mu_reflection=mu_reflection,
                epsilon=epsilon,
                margin=margin,
                tail=tail,
                scale_reflection_mu_by_period=scale_reflection_mu_by_period,
            )
            timings.append(elapsed)
        assert quality is not None
        median_s = statistics.median(timings)
        if baseline_median is None:
            baseline_median = median_s
            baseline_tail_mse = quality["mse_tail"]
        rows.append(
            {
                "seed": seed,
                "reflection_update_period": period,
                "min_s": min(timings),
                "median_s": median_s,
                "max_s": max(timings),
                "speedup_vs_period1": (baseline_median / median_s) if median_s > 0.0 else None,
                **quality,
                "tail_mse_ratio_vs_period1": (
                    quality["mse_tail"] / baseline_tail_mse
                    if baseline_tail_mse is not None and baseline_tail_mse > 0.0
                    else None
                ),
            }
        )
    return rows


def aggregate_rows(per_seed_rows: list[dict[str, Any]], periods: list[int]) -> list[dict[str, Any]]:
    """Aggregate per-seed rows into one robust row per period."""
    aggregate: list[dict[str, Any]] = []
    for period in periods:
        rows = [row for row in per_seed_rows if int(row["reflection_update_period"]) == period]
        if not rows:
            continue

        def vals(key: str, current_rows: list[dict[str, Any]] = rows) -> list[float]:
            return [float(row[key]) for row in current_rows if row.get(key) is not None]

        median_s = vals("median_s")
        speedups = vals("speedup_vs_period1")
        tail_ratios = vals("tail_mse_ratio_vs_period1")
        mse_tail = vals("mse_tail")
        mse_total = vals("mse_total")
        refl_err = vals("reflection_l2_error")
        tap_err = vals("taps_l2_error")
        margins = vals("stability_margin")

        aggregate.append(
            {
                "reflection_update_period": period,
                "n_seeds": len(rows),
                "median_s_median": statistics.median(median_s),
                "median_s_min": min(median_s),
                "median_s_max": max(median_s),
                "speedup_vs_period1_median": statistics.median(speedups),
                "speedup_vs_period1_min": min(speedups),
                "speedup_vs_period1_max": max(speedups),
                "mse_tail_median": statistics.median(mse_tail),
                "mse_tail_max": max(mse_tail),
                "tail_mse_ratio_median": statistics.median(tail_ratios),
                "tail_mse_ratio_max": max(tail_ratios),
                "tail_mse_ratio_p90": quantile(tail_ratios, 0.9),
                "mse_total_median": statistics.median(mse_total),
                "reflection_l2_error_median": statistics.median(refl_err),
                "reflection_l2_error_max": max(refl_err),
                "taps_l2_error_median": statistics.median(tap_err),
                "taps_l2_error_max": max(tap_err),
                "stability_margin_median": statistics.median(margins),
                "stability_margin_min": min(margins),
            }
        )
    return aggregate


def sweep(args: argparse.Namespace) -> dict[str, Any]:
    periods = parse_periods(args.periods)
    seeds = parse_seeds(args.seeds)
    target_reflection = np.asarray(args.target_reflection, dtype=np.float64)
    target_taps = np.asarray(args.target_taps, dtype=np.float64)

    per_seed_results: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows = run_seed(
            seed,
            periods,
            samples=args.samples,
            repeats=args.repeats,
            mu_taps=args.mu_taps,
            mu_reflection=args.mu_reflection,
            epsilon=args.epsilon,
            margin=args.margin,
            noise_std=args.noise_std,
            tail=args.tail,
            target_reflection=target_reflection,
            target_taps=target_taps,
            scale_reflection_mu_by_period=args.scale_reflection_mu_by_period,
        )
        per_seed_results.append({"seed": seed, "results": rows})
        flat_rows.extend(rows)

    aggregate = aggregate_rows(flat_rows, periods)
    return {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "has_openmp": HAS_OPENMP,
            "samples": args.samples,
            "repeats": args.repeats,
            "seeds": seeds,
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
        "aggregate_results": aggregate,
        "per_seed_results": per_seed_results,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", default=["100", "101", "102", "103", "104"])
    parser.add_argument("--periods", nargs="+", default=["1", "2", "4", "8", "16", "32"])
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--repeats", type=int, default=3)
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
    parser.add_argument(
        "--output", type=Path, default=Path("reports/adaptive-multiseed-sweep.json")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=None, help="Optional aggregate CSV output"
    )
    parser.add_argument(
        "--per-seed-csv-output", type=Path, default=None, help="Optional per-seed CSV output"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        write_csv(args.csv_output, results["aggregate_results"], AGGREGATE_COLUMNS)
        print(f"Wrote {args.csv_output}")
    if args.per_seed_csv_output is not None:
        rows = [
            row for seed_payload in results["per_seed_results"] for row in seed_payload["results"]
        ]
        write_csv(args.per_seed_csv_output, rows, PER_SEED_COLUMNS)
        print(f"Wrote {args.per_seed_csv_output}")


if __name__ == "__main__":
    main()
