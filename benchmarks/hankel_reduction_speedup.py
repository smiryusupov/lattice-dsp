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


def median_time(fn, repeats: int):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def stable_reflection(order: int, rng: np.random.Generator) -> np.ndarray:
    # Slow decay gives the reducer something nontrivial to compress while staying
    # comfortably inside the scalar lattice stability region.
    decay = np.exp(-np.arange(order) / max(8.0, order / 5.0))
    signs = rng.choice([-1.0, 1.0], size=order)
    jitter = rng.uniform(0.55, 1.0, size=order)
    return 0.72 * decay * signs * jitter


def numerator_for_order(order: int) -> np.ndarray:
    n = np.zeros(order + 1, dtype=float)
    n[0] = 1.0
    if order >= 2:
        n[1] = -0.18
        n[2] = 0.08
    if order >= 5:
        n[4] = -0.04
    return n


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    err = reference - estimate
    p_ref = float(np.mean(reference * reference))
    p_err = float(np.mean(err * err))
    return 10.0 * math.log10((p_ref + 1e-30) / (p_err + 1e-30))


def pole_radius_from_reflection(reflection: list[float]) -> float:
    if not reflection:
        return 0.0
    denominator = np.asarray(ld.reflection_to_denominator(reflection), dtype=float)
    roots = np.roots(denominator)
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def process(
    reflection: np.ndarray | list[float], numerator: np.ndarray | list[float], x: np.ndarray
):
    return ld.process_batch(list(map(float, reflection)), list(map(float, numerator)), x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark finite-Hankel SISO IIR reduction amortization."
    )
    parser.add_argument("--full-orders", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--reduced-orders", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-impulse", type=int, default=768)
    parser.add_argument("--hankel-rows", type=int, default=96)
    parser.add_argument("--hankel-cols", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/hankel-reduction-speedup.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.channels, args.samples)).astype(np.float64)
    rows_out: list[dict[str, float | int | bool | str | None]] = []

    for full_order in args.full_orders:
        reflection = stable_reflection(full_order, rng)
        numerator = numerator_for_order(full_order)

        full_time, y_full = median_time(
            lambda reflection=reflection, numerator=numerator: process(reflection, numerator, x),
            args.repeats,
        )
        full_per_sample = full_time / (args.channels * args.samples)

        for reduced_order in args.reduced_orders:
            if reduced_order >= full_order:
                continue
            if reduced_order > min(args.hankel_rows, args.hankel_cols):
                continue

            reduce_time, reduced = median_time(
                lambda ro=reduced_order, reflection=reflection, numerator=numerator: (
                    ld.finite_hankel_reduce_iir(
                        reflection.tolist(),
                        numerator.tolist(),
                        reduced_order=ro,
                        n_impulse=args.n_impulse,
                        rows=args.hankel_rows,
                        cols=args.hankel_cols,
                    )
                ),
                1,
            )

            if not reduced["stable"] or not reduced["reflection"]:
                rows_out.append(
                    {
                        "full_order": full_order,
                        "reduced_order": reduced_order,
                        "stable": bool(reduced["stable"]),
                        "reduction_time_s": reduce_time,
                        "error": "reduced model was not stable in reflection coordinates",
                    }
                )
                continue

            reduced_reflection = list(map(float, reduced["reflection"]))
            reduced_numerator = list(map(float, reduced["numerator"]))
            reduced_time, y_reduced = median_time(
                lambda rr=reduced_reflection, rn=reduced_numerator: process(rr, rn, x),
                args.repeats,
            )
            reduced_per_sample = reduced_time / (args.channels * args.samples)
            delta = full_per_sample - reduced_per_sample
            break_even_samples_per_channel = (
                reduce_time / delta / args.channels if delta > 0 else None
            )

            rel_mse = float(np.mean((y_full - y_reduced) ** 2) / (np.mean(y_full**2) + 1e-30))
            rows_out.append(
                {
                    "full_order": full_order,
                    "reduced_order": reduced_order,
                    "stable": bool(reduced["stable"]),
                    "method": reduced["method"],
                    "retained_hankel_energy": float(reduced["retained_hankel_energy"]),
                    "relative_impulse_error": float(reduced["relative_impulse_error"]),
                    "rel_mse_on_random_batch": rel_mse,
                    "snr_db_on_random_batch": snr_db(y_full, y_reduced),
                    "max_pole_radius": pole_radius_from_reflection(reduced_reflection),
                    "reduction_time_s": reduce_time,
                    "full_filter_median_s": full_time,
                    "reduced_filter_median_s": reduced_time,
                    "filter_speedup": full_time / reduced_time if reduced_time > 0 else None,
                    "full_time_per_sample_s": full_per_sample,
                    "reduced_time_per_sample_s": reduced_per_sample,
                    "break_even_samples_per_channel": break_even_samples_per_channel,
                }
            )

    result = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "has_openmp": bool(ld.HAS_OPENMP),
            "channels": args.channels,
            "samples": args.samples,
            "repeats": args.repeats,
            "n_impulse": args.n_impulse,
            "hankel_rows": args.hankel_rows,
            "hankel_cols": args.hankel_cols,
            "seed": args.seed,
            "description": "Finite-Hankel reduction amortization benchmark. Reduction is a preprocessing cost; speedup applies when the reduced model is reused.",
        },
        "rows": rows_out,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result["metadata"], indent=2))
    print()
    print(
        f"{'full':>5s} {'red':>5s} {'stable':>7s} {'reduce_s':>10s} "
        f"{'full_s':>10s} {'red_s':>10s} {'speedup':>8s} {'SNR':>8s} {'break_even/ch':>15s}"
    )
    print("-" * 95)
    for row in rows_out:
        if "error" in row:
            print(
                f"{row['full_order']:5d} {row['reduced_order']:5d} {str(row['stable']):>7s} {row['reduction_time_s']:10.4f}  ERROR: {row['error']}"
            )
            continue
        be = row["break_even_samples_per_channel"]
        be_text = "n/a" if be is None else f"{be:.0f}"
        print(
            f"{row['full_order']:5d} {row['reduced_order']:5d} {str(row['stable']):>7s} "
            f"{row['reduction_time_s']:10.4f} {row['full_filter_median_s']:10.4f} "
            f"{row['reduced_filter_median_s']:10.4f} {row['filter_speedup']:8.2f} "
            f"{row['snr_db_on_random_batch']:8.2f} {be_text:>15s}"
        )

    print()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
