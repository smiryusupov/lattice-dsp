from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np

import lattice_dsp as ld


def median_time(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    times: list[float] = []
    result: Any = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def impulse_from_poles(poles: np.ndarray, weights: np.ndarray, n_terms: int) -> np.ndarray:
    n = np.arange(n_terms, dtype=np.float64)
    return np.sum(weights[:, None] * poles[:, None] ** n[None, :], axis=0)


def numerator_from_impulse_and_denominator(
    impulse: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    order = denominator.size - 1
    numerator = np.zeros(order + 1, dtype=np.float64)
    for i in range(order + 1):
        numerator[i] = sum(float(denominator[j]) * float(impulse[i - j]) for j in range(i + 1))
    return numerator


def compressible_iir(order: int, rng: np.random.Generator, n_impulse: int) -> dict[str, np.ndarray]:
    """Build a stable real-pole IIR with decaying modal weights.

    The construction gives the reduction methods a meaningful compressible model:
    a few slow poles dominate, while many smaller modes are cheap to discard.
    """

    if order <= 1:
        raise ValueError("order must be greater than one")

    slow = np.array([0.92, 0.78, -0.58, 0.42], dtype=np.float64)
    remaining = order - slow.size
    if remaining > 0:
        grid = np.linspace(0.30, 0.04, remaining)
        signs = np.where(np.arange(remaining) % 2 == 0, 1.0, -1.0)
        small = signs * grid
        poles = np.concatenate([slow, small])
    else:
        poles = slow[:order]

    # Small deterministic jitter avoids perfectly repeated benchmark cases while
    # preserving stability and reproducibility.
    jitter = rng.uniform(-0.008, 0.008, size=poles.size)
    poles = np.clip(poles + jitter, -0.94, 0.94)

    weights = np.zeros(order, dtype=np.float64)
    weights[: min(4, order)] = np.array([1.0, 0.28, -0.17, 0.08], dtype=np.float64)[: min(4, order)]
    if order > 4:
        weights[4:] = (
            0.035
            * np.exp(-np.arange(order - 4) / 4.0)
            * np.where(np.arange(order - 4) % 2 == 0, 1.0, -1.0)
        )

    denominator = np.asarray(np.poly(poles), dtype=np.float64)
    impulse = impulse_from_poles(poles, weights, n_impulse)
    numerator = numerator_from_impulse_and_denominator(impulse, denominator)
    reflection = np.asarray(ld.denominator_to_reflection(denominator.tolist()), dtype=np.float64)
    return {
        "poles": poles,
        "weights": weights,
        "denominator": denominator,
        "numerator": numerator,
        "reflection": reflection,
        "impulse": impulse,
    }


def process(
    reflection: np.ndarray | list[float], numerator: np.ndarray | list[float], x: np.ndarray
) -> np.ndarray:
    return np.asarray(
        ld.process_batch(list(map(float, reflection)), list(map(float, numerator)), x),
        dtype=np.float64,
    )


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    err = reference - estimate
    p_ref = float(np.mean(reference * reference))
    p_err = float(np.mean(err * err))
    return 10.0 * math.log10((p_ref + 1e-30) / (p_err + 1e-30))


def pole_radius_from_denominator(denominator: np.ndarray | list[float]) -> float:
    denominator_arr = np.asarray(denominator, dtype=np.float64)
    roots = np.roots(denominator_arr)
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def frequency_response(
    denominator: np.ndarray, numerator: np.ndarray, n_freq: int = 512
) -> np.ndarray:
    w = np.linspace(0.0, math.pi, n_freq)
    z = np.exp(-1j * w)
    num = np.zeros_like(z, dtype=np.complex128)
    den = np.zeros_like(z, dtype=np.complex128)
    for k, coeff in enumerate(numerator):
        num += coeff * z**k
    for k, coeff in enumerate(denominator):
        den += coeff * z**k
    return num / den


def max_magnitude_error_db(
    full_denominator: np.ndarray,
    full_numerator: np.ndarray,
    reduced_denominator: np.ndarray,
    reduced_numerator: np.ndarray,
) -> float:
    h_full = frequency_response(full_denominator, full_numerator)
    h_reduced = frequency_response(reduced_denominator, reduced_numerator)
    full_db = 20.0 * np.log10(np.maximum(np.abs(h_full), 1e-14))
    reduced_db = 20.0 * np.log10(np.maximum(np.abs(h_reduced), 1e-14))
    return float(np.max(np.abs(full_db - reduced_db)))


def break_even_samples_per_channel(
    reduction_time_s: float, full_time_s: float, reduced_time_s: float, channels: int, samples: int
) -> float | None:
    full_per_sample = full_time_s / (channels * samples)
    reduced_per_sample = reduced_time_s / (channels * samples)
    delta = full_per_sample - reduced_per_sample
    if delta <= 0.0:
        return None
    return reduction_time_s / delta / channels


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def evaluate_reduced_model(
    *,
    method: str,
    full_order: int,
    target_order: int,
    reduction_time_s: float,
    full_model: dict[str, np.ndarray],
    full_time_s: float,
    y_full: np.ndarray,
    reduced_reflection: np.ndarray,
    reduced_numerator: np.ndarray,
    reduced_denominator: np.ndarray,
    relative_impulse_error: float,
    accepted: bool,
    stable: bool,
    x: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    reduced_time_s, y_reduced = median_time(
        lambda: process(reduced_reflection, reduced_numerator, x), repeats
    )
    rel_mse = float(np.mean((y_full - y_reduced) ** 2) / (np.mean(y_full**2) + 1e-30))
    end_to_end_speedup = (
        full_time_s / (reduction_time_s + reduced_time_s)
        if reduction_time_s + reduced_time_s > 0
        else None
    )
    be = break_even_samples_per_channel(
        reduction_time_s, full_time_s, reduced_time_s, x.shape[0], x.shape[1]
    )
    return {
        "method": method,
        "full_order": int(full_order),
        "target_order": int(target_order),
        "stable": bool(stable),
        "accepted": bool(accepted),
        "reduction_time_s": float(reduction_time_s),
        "full_filter_median_s": float(full_time_s),
        "reduced_filter_median_s": float(reduced_time_s),
        "filter_speedup": float(full_time_s / reduced_time_s) if reduced_time_s > 0 else None,
        "amortized_end_to_end_speedup": float(end_to_end_speedup)
        if end_to_end_speedup is not None
        else None,
        "break_even_samples_per_channel": float(be) if be is not None else None,
        "relative_impulse_error": float(relative_impulse_error),
        "rel_mse_on_random_batch": rel_mse,
        "snr_db_on_random_batch": snr_db(y_full, y_reduced),
        "max_magnitude_error_db": max_magnitude_error_db(
            full_model["denominator"],
            full_model["numerator"],
            reduced_denominator,
            reduced_numerator,
        ),
        "max_pole_radius": pole_radius_from_denominator(reduced_denominator),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare finite-Hankel and finite-section AAK/Nehari SISO IIR reduction workflows."
    )
    parser.add_argument("--full-orders", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--target-orders", type=int, nargs="+", default=[3, 4, 6, 8, 12])
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-impulse", type=int, default=768)
    parser.add_argument("--hankel-rows", type=int, default=96)
    parser.add_argument("--hankel-cols", type=int, default=96)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--output", default="reports/finite-aak-iir-reduction-speedup.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.channels, args.samples)).astype(np.float64)
    rows_out: list[dict[str, Any]] = []
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1.0,
        max_rational_error=1.0,
        max_pole_radius=0.999,
    )

    for full_order in args.full_orders:
        if full_order <= 1:
            continue
        full_model = compressible_iir(full_order, rng, args.n_impulse)
        full_time_s, y_full = median_time(
            lambda fm=full_model: process(fm["reflection"], fm["numerator"], x),
            args.repeats,
        )

        for target_order in args.target_orders:
            if target_order >= full_order:
                continue
            if target_order > min(args.hankel_rows, args.hankel_cols):
                continue

            # Finite-Hankel / Ho--Kalman baseline.
            try:
                reduce_time, hankel = median_time(
                    lambda ro=target_order, fm=full_model: ld.finite_hankel_reduce_iir(
                        fm["reflection"].tolist(),
                        fm["numerator"].tolist(),
                        reduced_order=ro,
                        n_impulse=args.n_impulse,
                        rows=args.hankel_rows,
                        cols=args.hankel_cols,
                    ),
                    1,
                )
                if bool(hankel["stable"]) and hankel.get("reflection"):
                    rows_out.append(
                        serializable_row(
                            evaluate_reduced_model(
                                method="finite_hankel",
                                full_order=full_order,
                                target_order=target_order,
                                reduction_time_s=reduce_time,
                                full_model=full_model,
                                full_time_s=full_time_s,
                                y_full=y_full,
                                reduced_reflection=np.asarray(
                                    hankel["reflection"], dtype=np.float64
                                ),
                                reduced_numerator=np.asarray(hankel["numerator"], dtype=np.float64),
                                reduced_denominator=np.asarray(
                                    hankel["denominator"], dtype=np.float64
                                ),
                                relative_impulse_error=float(hankel["relative_impulse_error"]),
                                accepted=True,
                                stable=True,
                                x=x,
                                repeats=args.repeats,
                            )
                        )
                    )
                else:
                    rows_out.append(
                        {
                            "method": "finite_hankel",
                            "full_order": full_order,
                            "target_order": target_order,
                            "stable": bool(hankel.get("stable", False)),
                            "accepted": False,
                            "reduction_time_s": float(reduce_time),
                            "error": "reduced model was not stable in scalar lattice coordinates",
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - benchmark rows should report failures.
                rows_out.append(
                    {
                        "method": "finite_hankel",
                        "full_order": full_order,
                        "target_order": target_order,
                        "stable": False,
                        "accepted": False,
                        "error": str(exc),
                    }
                )

            # Finite-section AAK/Nehari candidate using the same target order.
            try:
                reduce_time, aak = median_time(
                    lambda ro=target_order, fm=full_model: ld.finite_aak_reduce_iir(
                        fm["reflection"],
                        fm["numerator"],
                        ranks=[ro],
                        n_impulse=args.n_impulse,
                        rows=args.hankel_rows,
                        cols=args.hankel_cols,
                        criteria=criteria,
                        attach_certificate=True,
                    ),
                    1,
                )
                if bool(aak["stable"]) and aak["reduced_reflection"].size:
                    rows_out.append(
                        serializable_row(
                            evaluate_reduced_model(
                                method="finite_aak_candidate",
                                full_order=full_order,
                                target_order=target_order,
                                reduction_time_s=reduce_time,
                                full_model=full_model,
                                full_time_s=full_time_s,
                                y_full=y_full,
                                reduced_reflection=np.asarray(
                                    aak["reduced_reflection"], dtype=np.float64
                                ),
                                reduced_numerator=np.asarray(
                                    aak["reduced_numerator"], dtype=np.float64
                                ),
                                reduced_denominator=np.asarray(
                                    aak["reduced_denominator"], dtype=np.float64
                                ),
                                relative_impulse_error=float(aak["relative_impulse_error"]),
                                accepted=bool(aak["accepted"]),
                                stable=bool(aak["stable"]),
                                x=x,
                                repeats=args.repeats,
                            )
                        )
                    )
                else:
                    rows_out.append(
                        {
                            "method": "finite_aak_candidate",
                            "full_order": full_order,
                            "target_order": target_order,
                            "stable": bool(aak.get("stable", False)),
                            "accepted": False,
                            "reduction_time_s": float(reduce_time),
                            "error": "selected candidate was not stable in scalar lattice coordinates",
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - benchmark rows should report failures.
                rows_out.append(
                    {
                        "method": "finite_aak_candidate",
                        "full_order": full_order,
                        "target_order": target_order,
                        "stable": False,
                        "accepted": False,
                        "error": str(exc),
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
            "description": (
                "Finite-Hankel versus finite-section AAK/Nehari SISO IIR reduction benchmark. "
                "Both methods are finite-section baselines; neither is claimed to be a full infinite-dimensional solver."
            ),
        },
        "rows": rows_out,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result["metadata"], indent=2))
    print()
    print(
        f"{'method':>21s} {'full':>5s} {'red':>5s} {'stable':>7s} {'reduce_s':>10s} "
        f"{'filter_x':>9s} {'end2end_x':>10s} {'SNR':>8s} {'mag_err':>9s} {'break_even/ch':>15s}"
    )
    print("-" * 115)
    for row in rows_out:
        if "error" in row:
            print(
                f"{row['method']:>21s} {row['full_order']:5d} {row['target_order']:5d} "
                f"{str(row.get('stable', False)):>7s} {float(row.get('reduction_time_s', 0.0)):10.4f}  ERROR: {row['error']}"
            )
            continue
        be = row["break_even_samples_per_channel"]
        be_text = "n/a" if be is None else f"{be:.0f}"
        print(
            f"{row['method']:>21s} {row['full_order']:5d} {row['target_order']:5d} {str(row['stable']):>7s} "
            f"{row['reduction_time_s']:10.4f} {row['filter_speedup']:9.2f} "
            f"{row['amortized_end_to_end_speedup']:10.2f} {row['snr_db_on_random_batch']:8.2f} "
            f"{row['max_magnitude_error_db']:9.3f} {be_text:>15s}"
        )

    print()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
