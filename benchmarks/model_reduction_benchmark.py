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


def pole_radius(reflection):
    if len(reflection) == 0:
        return 0.0
    a = np.asarray(ld.reflection_to_denominator(reflection.tolist()), dtype=float)
    roots = np.roots(a)
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def snr_db(reference, estimate):
    err = reference - estimate
    p_ref = float(np.mean(reference * reference))
    p_err = float(np.mean(err * err))
    return 10.0 * math.log10((p_ref + 1e-30) / (p_err + 1e-30))


def process_batch(reflection, x):
    numerator = [1.0] + [0.0] * len(reflection)
    return ld.process_batch(reflection.tolist(), numerator, x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-order", type=int, default=16)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6, 8, 12, 16])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="reports/model-reduction-results.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Stable high-order all-pole model via reflection coefficients.
    # Magnitudes decay with order so lower-order reductions have a fair chance.
    decay = np.exp(-np.arange(args.full_order) / 5.0)
    full_reflection = 0.65 * decay * rng.uniform(-1.0, 1.0, size=args.full_order)

    x = rng.normal(size=(args.channels, args.samples)).astype(np.float64)

    full_time, y_full = median_time(
        lambda: process_batch(full_reflection, x),
        args.repeats,
    )

    rows = []

    for order in args.orders:
        if order > args.full_order:
            continue

        # Method 1: stable lattice truncation.
        # Keep the first k reflection coefficients.
        trunc_reflection = full_reflection[:order].copy()

        trunc_time, y_trunc = median_time(
            lambda r=trunc_reflection: process_batch(r, x),
            args.repeats,
        )

        rows.append(
            {
                "method": "reflection_truncation",
                "order": order,
                "median_s": trunc_time,
                "speedup_vs_full": full_time / trunc_time,
                "rel_mse": float(np.mean((y_full - y_trunc) ** 2) / (np.mean(y_full**2) + 1e-30)),
                "snr_db": snr_db(y_full, y_trunc),
                "max_pole_radius": pole_radius(trunc_reflection),
            }
        )

        # Method 2: reduced AR refit.
        # Estimate a reduced all-pole model from the full model's first channel.
        # This is often a better reduced model than simple truncation.
        if order > 0:
            r = ld.autocorrelation(y_full[0], order)
            fit_reflection = np.asarray(
                ld.levinson_durbin_reflection(r, order),
                dtype=float,
            )
        else:
            fit_reflection = np.asarray([], dtype=float)

        fit_time, y_fit = median_time(
            lambda r=fit_reflection: process_batch(r, x),
            args.repeats,
        )

        rows.append(
            {
                "method": "reduced_ar_refit",
                "order": order,
                "median_s": fit_time,
                "speedup_vs_full": full_time / fit_time,
                "rel_mse": float(np.mean((y_full - y_fit) ** 2) / (np.mean(y_full**2) + 1e-30)),
                "snr_db": snr_db(y_full, y_fit),
                "max_pole_radius": pole_radius(fit_reflection),
            }
        )

    result = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "has_openmp": bool(ld.HAS_OPENMP),
            "full_order": args.full_order,
            "channels": args.channels,
            "samples": args.samples,
            "repeats": args.repeats,
            "seed": args.seed,
            "full_median_s": full_time,
            "full_max_pole_radius": pole_radius(full_reflection),
        },
        "rows": rows,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))

    print(json.dumps(result["metadata"], indent=2))
    print()
    print(
        f"{'method':24s} {'order':>5s} {'median_s':>10s} {'speedup':>9s} {'rel_mse':>12s} {'snr_db':>9s} {'pole_r':>8s}"
    )
    print("-" * 88)
    for row in sorted(rows, key=lambda r: (r["method"], r["order"])):
        print(
            f"{row['method']:24s} "
            f"{row['order']:5d} "
            f"{row['median_s']:10.6f} "
            f"{row['speedup_vs_full']:9.2f} "
            f"{row['rel_mse']:12.3e} "
            f"{row['snr_db']:9.2f} "
            f"{row['max_pole_radius']:8.4f}"
        )

    print()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
