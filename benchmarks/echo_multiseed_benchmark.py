"""Robust multi-seed benchmark for hybrid echo/noise cancellation.

This script repeatedly runs ``echo_cancellation_benchmark.py`` over multiple
random seeds and optional scenario sweeps.  It aggregates ERLE, MSE, runtime,
and pairwise comparisons so the hybrid lattice/IIR claim can be evaluated on
more than one synthetic realization.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from lattice_dsp import HAS_OPENMP

from benchmarks.echo_cancellation_benchmark import build_parser as build_single_parser
from benchmarks.echo_cancellation_benchmark import run_benchmark


def parse_int_values(values: Iterable[str]) -> list[int]:
    """Parse integers from comma- and/or space-separated CLI tokens."""

    parsed: list[int] = []
    for token in values:
        for item in str(token).split(","):
            item = item.strip()
            if item:
                value = int(item)
                if value not in parsed:
                    parsed.append(value)
    if not parsed:
        raise ValueError("at least one integer value is required")
    return parsed


def parse_float_values(values: Iterable[str]) -> list[float]:
    """Parse floats from comma- and/or space-separated CLI tokens."""

    parsed: list[float] = []
    for token in values:
        for item in str(token).split(","):
            item = item.strip()
            if item:
                value = float(item)
                if value not in parsed:
                    parsed.append(value)
    if not parsed:
        raise ValueError("at least one float value is required")
    return parsed


def median(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    if not items:
        return float("nan")
    return float(statistics.median(items))


def percentile(values: Iterable[float], q: float) -> float:
    """Return a simple linearly interpolated percentile for small lists."""

    items = sorted(float(v) for v in values)
    if not items:
        return float("nan")
    if len(items) == 1:
        return items[0]
    q = min(max(float(q), 0.0), 1.0)
    position = q * (len(items) - 1)
    lo = int(position)
    hi = min(lo + 1, len(items) - 1)
    weight = position - lo
    return float((1.0 - weight) * items[lo] + weight * items[hi])


def aggregate_case_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate case rows across seeds for one scenario."""

    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        by_name[str(row["name"])].append(row)

    aggregate: list[dict[str, Any]] = []
    for name in sorted(by_name):
        rows = by_name[name]
        erle = [float(row["erle_db"]) for row in rows]
        seg_erle = [float(row["segmental_erle_median_db"]) for row in rows]
        mse_improvement = [float(row["mse_improvement_db"]) for row in rows]
        output_mse = [float(row["output_mse"]) for row in rows]
        residual_power = [float(row["residual_power_db"]) for row in rows]
        median_s = [float(row.get("median_s", 0.0)) for row in rows]
        aggregate.append(
            {
                "name": name,
                "n_seeds": len(rows),
                "erle_db_median": median(erle),
                "erle_db_min": min(erle),
                "erle_db_max": max(erle),
                "erle_db_p10": percentile(erle, 0.10),
                "segmental_erle_median_db_median": median(seg_erle),
                "mse_improvement_db_median": median(mse_improvement),
                "output_mse_median": median(output_mse),
                "residual_power_db_median": median(residual_power),
                "median_s_median": median(median_s),
                "median_s_min": min(median_s),
                "median_s_max": max(median_s),
            }
        )
    return aggregate


def compare_case_pairs(
    per_seed_cases: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    """Compare two named cases seed-by-seed.

    ``left`` is interpreted as the candidate method and ``right`` as the
    baseline.  Positive ERLE gain means the candidate improved over the
    baseline. Runtime speedup is baseline runtime divided by candidate runtime.
    """

    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in per_seed_cases:
        by_seed[int(row["seed"])][str(row["name"])] = row

    erle_gains: list[float] = []
    runtime_speedups: list[float] = []
    mse_gain_deltas: list[float] = []
    for _seed, rows in by_seed.items():
        if left not in rows or right not in rows:
            continue
        left_row = rows[left]
        right_row = rows[right]
        erle_gains.append(float(left_row["erle_db"]) - float(right_row["erle_db"]))
        mse_gain_deltas.append(
            float(left_row["mse_improvement_db"]) - float(right_row["mse_improvement_db"])
        )
        left_time = max(float(left_row.get("median_s", 0.0)), 1e-15)
        right_time = max(float(right_row.get("median_s", 0.0)), 1e-15)
        runtime_speedups.append(right_time / left_time)

    if not erle_gains:
        return {
            "left": left,
            "right": right,
            "n_seeds": 0,
            "erle_gain_db_median": float("nan"),
            "erle_gain_db_min": float("nan"),
            "mse_improvement_gain_db_median": float("nan"),
            "runtime_speedup_median": float("nan"),
            "runtime_speedup_min": float("nan"),
        }
    return {
        "left": left,
        "right": right,
        "n_seeds": len(erle_gains),
        "erle_gain_db_median": median(erle_gains),
        "erle_gain_db_min": min(erle_gains),
        "erle_gain_db_max": max(erle_gains),
        "mse_improvement_gain_db_median": median(mse_gain_deltas),
        "runtime_speedup_median": median(runtime_speedups),
        "runtime_speedup_min": min(runtime_speedups),
        "runtime_speedup_max": max(runtime_speedups),
    }


def _single_args(
    base: argparse.Namespace, *, seed: int, strength: float, noise_snr_db: float, near: float
) -> argparse.Namespace:
    """Create the namespace expected by ``run_benchmark`` for one scenario."""

    return argparse.Namespace(
        samples=base.samples,
        sample_rate=base.sample_rate,
        seed=seed,
        repeats=base.repeats,
        nonlinearity=base.nonlinearity,
        nonlinear_strength=strength,
        near_end_power_ratio=near,
        noise_snr_db=noise_snr_db,
        no_double_talk=base.no_double_talk,
        iir_order=base.iir_order,
        fir_order=base.fir_order,
        fir_mu=base.fir_mu,
        mu_taps=base.mu_taps,
        mu_reflection=base.mu_reflection,
        epsilon=base.epsilon,
        reflection_update_period=base.reflection_update_period,
        no_scale_reflection_mu_by_period=base.no_scale_reflection_mu_by_period,
        residual_gain=base.residual_gain,
        spectral_frame_size=base.spectral_frame_size,
        spectral_hop_size=base.spectral_hop_size,
        spectral_floor=base.spectral_floor,
        spectral_over_subtract=base.spectral_over_subtract,
        spectral_noise_percentile=base.spectral_noise_percentile,
        spectral_smoothing=base.spectral_smoothing,
        spectral_exponent=base.spectral_exponent,
        spectral_mode=base.spectral_mode,
        spectral_echo_aware_strength=base.spectral_echo_aware_strength,
        spectral_reference_key=base.spectral_reference_key,
    )


def _metadata(
    args: argparse.Namespace,
    *,
    seeds: list[int],
    strengths: list[float],
    noise_levels: list[float],
    near_levels: list[float],
    n_scenarios: int,
) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "has_openmp": HAS_OPENMP,
        "samples": args.samples,
        "sample_rate": args.sample_rate,
        "seeds": seeds,
        "repeats": args.repeats,
        "nonlinearity": args.nonlinearity,
        "nonlinear_strengths": strengths,
        "noise_snr_dbs": noise_levels,
        "near_end_power_ratios": near_levels,
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
        "n_scenarios": n_scenarios,
    }


def _write_checkpoint(
    args: argparse.Namespace,
    *,
    seeds: list[int],
    strengths: list[float],
    noise_levels: list[float],
    near_levels: list[float],
    scenarios: list[dict[str, Any]],
    total_scenarios: int,
) -> None:
    checkpoint_output = getattr(args, "checkpoint_output", None)
    if checkpoint_output is None:
        return
    payload = {
        "metadata": {
            **_metadata(
                args,
                seeds=seeds,
                strengths=strengths,
                noise_levels=noise_levels,
                near_levels=near_levels,
                n_scenarios=total_scenarios,
            ),
            "partial": True,
            "completed_scenarios": len(scenarios),
        },
        "scenarios": scenarios,
    }
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_multiseed_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_int_values(args.seeds)
    strengths = parse_float_values(args.nonlinear_strengths)
    noise_levels = parse_float_values(args.noise_snr_dbs)
    near_levels = parse_float_values(args.near_end_power_ratios)
    scenario_grid = list(itertools.product(strengths, noise_levels, near_levels))
    total_runs = len(scenario_grid) * len(seeds)

    scenarios: list[dict[str, Any]] = []
    completed_runs = 0
    for scenario_index, (strength, noise_snr_db, near) in enumerate(scenario_grid, start=1):
        scenario = {
            "nonlinearity": args.nonlinearity,
            "nonlinear_strength": strength,
            "noise_snr_db": noise_snr_db,
            "near_end_power_ratio": near,
        }
        per_seed_cases: list[dict[str, Any]] = []
        per_seed_metadata: list[dict[str, Any]] = []
        if getattr(args, "progress", True):
            print(
                f"Scenario {scenario_index}/{len(scenario_grid)}: "
                f"nonlinear_strength={strength}, noise_snr_db={noise_snr_db}, "
                f"near_end_power_ratio={near}",
                flush=True,
            )
        for seed in seeds:
            completed_runs += 1
            if getattr(args, "progress", True):
                print(
                    f"  Run {completed_runs}/{total_runs}: seed={seed}",
                    flush=True,
                )
            single_args = _single_args(
                args,
                seed=seed,
                strength=strength,
                noise_snr_db=noise_snr_db,
                near=near,
            )
            payload = run_benchmark(single_args)
            per_seed_metadata.append(payload["metadata"])
            for row in payload["cases"]:
                per_seed_cases.append({"seed": seed, **row})

        aggregate_cases = aggregate_case_rows(per_seed_cases)
        by_name = {row["name"]: row for row in aggregate_cases}
        best = max(aggregate_cases, key=lambda row: float(row["erle_db_median"]))
        comparisons = {
            "lattice_vs_fir": compare_case_pairs(
                per_seed_cases,
                "lattice_iir_only",
                "fir_nlms_baseline",
            ),
            "hybrid_vs_fir": compare_case_pairs(
                per_seed_cases,
                "lattice_iir_plus_toy_residual_suppressor",
                "fir_nlms_baseline",
            ),
            "hybrid_vs_lattice": compare_case_pairs(
                per_seed_cases,
                "lattice_iir_plus_toy_residual_suppressor",
                "lattice_iir_only",
            ),
            "spectral_hybrid_vs_fir": compare_case_pairs(
                per_seed_cases,
                "lattice_iir_plus_spectral_residual_suppressor",
                "fir_nlms_baseline",
            ),
            "spectral_hybrid_vs_lattice": compare_case_pairs(
                per_seed_cases,
                "lattice_iir_plus_spectral_residual_suppressor",
                "lattice_iir_only",
            ),
            "spectral_hybrid_vs_toy_hybrid": compare_case_pairs(
                per_seed_cases,
                "lattice_iir_plus_spectral_residual_suppressor",
                "lattice_iir_plus_toy_residual_suppressor",
            ),
            "lattice_vs_no_cancellation": compare_case_pairs(
                per_seed_cases,
                "lattice_iir_only",
                "no_cancellation",
            ),
        }
        scenarios.append(
            {
                "scenario": scenario,
                "aggregate_cases": aggregate_cases,
                "comparisons": comparisons,
                "best_by_median_erle": best,
                "key_results": {
                    "fir_erle_db_median": by_name["fir_nlms_baseline"]["erle_db_median"],
                    "lattice_erle_db_median": by_name["lattice_iir_only"]["erle_db_median"],
                    "hybrid_erle_db_median": by_name["lattice_iir_plus_toy_residual_suppressor"][
                        "erle_db_median"
                    ],
                    "spectral_hybrid_erle_db_median": by_name[
                        "lattice_iir_plus_spectral_residual_suppressor"
                    ]["erle_db_median"],
                    "lattice_runtime_s_median": by_name["lattice_iir_only"]["median_s_median"],
                    "fir_runtime_s_median": by_name["fir_nlms_baseline"]["median_s_median"],
                    "hybrid_runtime_s_median": by_name["lattice_iir_plus_toy_residual_suppressor"][
                        "median_s_median"
                    ],
                    "spectral_hybrid_runtime_s_median": by_name[
                        "lattice_iir_plus_spectral_residual_suppressor"
                    ]["median_s_median"],
                },
                "per_seed_cases": per_seed_cases,
                "per_seed_metadata": per_seed_metadata,
            }
        )
        _write_checkpoint(
            args,
            seeds=seeds,
            strengths=strengths,
            noise_levels=noise_levels,
            near_levels=near_levels,
            scenarios=scenarios,
            total_scenarios=len(scenario_grid),
        )

    return {
        "metadata": _metadata(
            args,
            seeds=seeds,
            strengths=strengths,
            noise_levels=noise_levels,
            near_levels=near_levels,
            n_scenarios=len(scenarios),
        ),
        "scenarios": scenarios,
    }


def write_scenario_csv(payload: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "nonlinearity",
        "nonlinear_strength",
        "noise_snr_db",
        "near_end_power_ratio",
        "fir_erle_db_median",
        "lattice_erle_db_median",
        "hybrid_erle_db_median",
        "spectral_hybrid_erle_db_median",
        "fir_runtime_s_median",
        "lattice_runtime_s_median",
        "hybrid_runtime_s_median",
        "spectral_hybrid_runtime_s_median",
        "lattice_vs_fir_erle_gain_db_median",
        "lattice_vs_fir_erle_gain_db_min",
        "lattice_vs_fir_runtime_speedup_median",
        "hybrid_vs_fir_erle_gain_db_median",
        "hybrid_vs_fir_erle_gain_db_min",
        "hybrid_vs_fir_runtime_speedup_median",
        "hybrid_vs_lattice_erle_gain_db_median",
        "spectral_hybrid_vs_fir_erle_gain_db_median",
        "spectral_hybrid_vs_fir_erle_gain_db_min",
        "spectral_hybrid_vs_fir_runtime_speedup_median",
        "spectral_hybrid_vs_lattice_erle_gain_db_median",
        "spectral_hybrid_vs_toy_hybrid_erle_gain_db_median",
        "best_by_median_erle",
    ]

    def cmp_value(comparisons: dict[str, Any], name: str, metric: str) -> Any:
        return comparisons.get(name, {}).get(metric, float("nan"))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in payload["scenarios"]:
            scenario = item["scenario"]
            comparisons = item["comparisons"]
            row = {
                **scenario,
                **item["key_results"],
                "lattice_vs_fir_erle_gain_db_median": cmp_value(
                    comparisons, "lattice_vs_fir", "erle_gain_db_median"
                ),
                "lattice_vs_fir_erle_gain_db_min": cmp_value(
                    comparisons, "lattice_vs_fir", "erle_gain_db_min"
                ),
                "lattice_vs_fir_runtime_speedup_median": cmp_value(
                    comparisons, "lattice_vs_fir", "runtime_speedup_median"
                ),
                "hybrid_vs_fir_erle_gain_db_median": cmp_value(
                    comparisons, "hybrid_vs_fir", "erle_gain_db_median"
                ),
                "hybrid_vs_fir_erle_gain_db_min": cmp_value(
                    comparisons, "hybrid_vs_fir", "erle_gain_db_min"
                ),
                "hybrid_vs_fir_runtime_speedup_median": cmp_value(
                    comparisons, "hybrid_vs_fir", "runtime_speedup_median"
                ),
                "hybrid_vs_lattice_erle_gain_db_median": cmp_value(
                    comparisons, "hybrid_vs_lattice", "erle_gain_db_median"
                ),
                "spectral_hybrid_vs_fir_erle_gain_db_median": cmp_value(
                    comparisons, "spectral_hybrid_vs_fir", "erle_gain_db_median"
                ),
                "spectral_hybrid_vs_fir_erle_gain_db_min": cmp_value(
                    comparisons, "spectral_hybrid_vs_fir", "erle_gain_db_min"
                ),
                "spectral_hybrid_vs_fir_runtime_speedup_median": cmp_value(
                    comparisons, "spectral_hybrid_vs_fir", "runtime_speedup_median"
                ),
                "spectral_hybrid_vs_lattice_erle_gain_db_median": cmp_value(
                    comparisons, "spectral_hybrid_vs_lattice", "erle_gain_db_median"
                ),
                "spectral_hybrid_vs_toy_hybrid_erle_gain_db_median": cmp_value(
                    comparisons, "spectral_hybrid_vs_toy_hybrid", "erle_gain_db_median"
                ),
                "best_by_median_erle": item["best_by_median_erle"]["name"],
            }
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    single = build_single_parser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", default=["100", "101", "102", "103", "104"])
    parser.add_argument("--nonlinear-strengths", nargs="+", default=["0.0", "0.04", "0.08", "0.16"])
    parser.add_argument("--noise-snr-dbs", nargs="+", default=["40", "30", "20"])
    parser.add_argument("--near-end-power-ratios", nargs="+", default=["0.0", "0.02"])
    for action in single._actions:
        if not action.option_strings or action.dest in {
            "help",
            "seed",
            "nonlinear_strength",
            "noise_snr_db",
            "near_end_power_ratio",
            "output",
        }:
            continue
        kwargs: dict[str, Any] = {
            "dest": action.dest,
            "default": action.default,
            "help": argparse.SUPPRESS,
        }
        if isinstance(action, argparse._StoreTrueAction):
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = getattr(action, "type", None)
            if action.choices is not None:
                kwargs["choices"] = action.choices
        parser.add_argument(
            *action.option_strings, **{k: v for k, v in kwargs.items() if v is not None}
        )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/echo-multiseed-benchmark.json")
    )
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small interactive quick validation sweep instead of the full grid.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable progress messages during long sweeps.",
    )
    parser.set_defaults(progress=True)
    return parser


def _apply_quick_preset(args: argparse.Namespace) -> None:
    if not getattr(args, "quick", False):
        return
    args.seeds = ["100", "101"]
    args.nonlinear_strengths = ["0.0", "0.08"]
    args.noise_snr_dbs = ["30"]
    args.near_end_power_ratios = ["0.02"]
    args.samples = min(int(args.samples), 16000)
    args.repeats = 1


def main() -> None:
    args = build_parser().parse_args()
    _apply_quick_preset(args)
    payload = run_multiseed_benchmark(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.csv_output is not None:
        write_scenario_csv(payload, args.csv_output)
        print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
