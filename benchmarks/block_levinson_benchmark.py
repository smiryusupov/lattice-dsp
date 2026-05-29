"""Benchmark dense block Yule-Walker solve versus block Levinson recursion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def simulate_var(channels: int, order: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coeffs = []
    for lag in range(order):
        raw = rng.standard_normal((channels, channels))
        # Small, diagonally dominant coefficients keep the VAR stable.
        coeffs.append((0.18 / (lag + 1)) * raw / max(channels, 1))
    radius = ld.companion_spectral_radius(np.asarray(coeffs))
    if radius >= 0.85:
        scale = 0.85 / radius
        coeffs = [scale * a for a in coeffs]

    x = np.zeros((samples + 512, channels))
    noise = rng.normal(size=x.shape)
    for n in range(order, x.shape[0]):
        y = noise[n].copy()
        for lag, a_lag in enumerate(coeffs, start=1):
            y -= a_lag @ x[n - lag]
        x[n] = y
    return x[512:]


def median_time(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    x = simulate_var(args.channels, args.order, args.samples, seed=123)
    r = ld.multichannel_autocorrelation(x, order=args.order)

    direct = ld.solve_block_yule_walker_direct(r, order=args.order)
    levinson = ld.block_levinson_durbin(r, order=args.order)
    coeff_diff = float(np.linalg.norm(direct.coefficients - levinson.coefficients))

    direct_time = median_time(
        lambda: ld.solve_block_yule_walker_direct(r, order=args.order), args.repeats
    )
    levinson_time = median_time(lambda: ld.block_levinson_durbin(r, order=args.order), args.repeats)

    result = {
        "channels": args.channels,
        "order": args.order,
        "samples": args.samples,
        "repeats": args.repeats,
        "direct_dense_seconds_median": direct_time,
        "block_levinson_seconds_median": levinson_time,
        "speedup_direct_over_levinson": direct_time / levinson_time
        if levinson_time
        else float("inf"),
        "coefficient_difference_norm": coeff_diff,
        "max_reflection_spectral_norm": float(np.max(levinson.reflection_spectral_norms)),
    }

    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
