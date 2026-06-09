# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""User-facing tuning helpers for adaptive lattice-IIR parameters."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ._core import AdaptiveLatticeLadderNLMS


def _parse_periods(periods: Iterable[int]) -> list[int]:
    parsed: list[int] = []
    for value in periods:
        period = int(value)
        if period <= 0:
            raise ValueError("periods must contain positive integers")
        parsed.append(period)
    if not parsed:
        raise ValueError("periods must contain at least one value")
    return list(dict.fromkeys(parsed))


def _as_trial_matrix(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        return np.ascontiguousarray(array.reshape(1, -1))
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    raise ValueError(f"{name} must be a 1-D array or a 2-D trial-by-sample array")


def _tail_count(tail: int | None, n_samples: int) -> int:
    if n_samples <= 0:
        raise ValueError("x and desired must contain at least one sample")
    if tail is None:
        return max(1, min(n_samples, n_samples // 10))
    tail_i = int(tail)
    if tail_i <= 0:
        raise ValueError("tail must be positive")
    return min(tail_i, n_samples)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(float(v) for v in values))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _safe_ratio(value: float, baseline: float) -> float:
    if baseline > 0.0:
        return float(value / baseline)
    return 1.0 if value == 0.0 else math.inf


def _infer_initial_parameters(
    initial_reflection: Sequence[float] | None,
    initial_taps: Sequence[float] | None,
    order: int | None,
) -> tuple[list[float], list[float], int]:
    if initial_reflection is not None:
        reflection = [float(v) for v in initial_reflection]
        inferred_order = len(reflection)
        if order is not None and int(order) != inferred_order:
            raise ValueError("order must match len(initial_reflection)")
    else:
        if order is None:
            if initial_taps is not None:
                inferred_order = len(initial_taps) - 1
            else:
                # A small, practical default that makes the simple API usable.
                # Serious experiments should pass order or initial_reflection.
                inferred_order = 4
        else:
            inferred_order = int(order)
        if inferred_order < 0:
            raise ValueError("order must be non-negative")
        reflection = [0.0] * inferred_order

    if initial_taps is None:
        taps = [0.0] * (inferred_order + 1)
    else:
        taps = [float(v) for v in initial_taps]
        if len(taps) != inferred_order + 1:
            raise ValueError("initial_taps must have length order + 1")

    return reflection, taps, inferred_order


def _mse(array: np.ndarray) -> float:
    return float(np.mean(np.square(array)))


def _recommend_period(
    rows: list[dict[str, Any]],
    *,
    max_tail_mse_ratio: float,
    max_worst_tail_mse_ratio: float,
    min_stability_margin: float,
    prefer: str,
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if row["tail_mse_ratio_median"] <= max_tail_mse_ratio
        and row["tail_mse_ratio_worst"] <= max_worst_tail_mse_ratio
        and row["stability_margin_min"] >= min_stability_margin
    ]
    if not eligible:
        return None

    if prefer == "fastest":
        return min(eligible, key=lambda row: (row["median_s"], row["tail_mse_ratio_worst"]))
    if prefer in {"quality", "accurate"}:
        return min(eligible, key=lambda row: (row["mse_tail_median"], row["median_s"]))
    if prefer == "balanced":
        return max(
            eligible,
            key=lambda row: (
                row["speedup_vs_period1_median"] / max(row["tail_mse_ratio_worst"], 1e-12),
                -row["median_s"],
            ),
        )
    raise ValueError("prefer must be one of: 'fastest', 'quality', 'accurate', 'balanced'")


def tune_reflection_update_period(
    x: Any,
    desired: Any,
    *,
    periods: Iterable[int] = (1, 2, 4, 8, 16, 32),
    initial_reflection: Sequence[float] | None = None,
    initial_taps: Sequence[float] | None = None,
    order: int | None = None,
    mu_taps: float = 0.05,
    mu_reflection: float = 0.001,
    epsilon: float = 1e-8,
    margin: float = 1e-4,
    tail: int | None = None,
    repeats: int = 1,
    max_tail_mse_ratio: float = 1.5,
    max_worst_tail_mse_ratio: float | None = None,
    min_stability_margin: float = 0.05,
    prefer: str = "fastest",
    gradient_mode: str = "analytic",
    scale_reflection_mu_by_period: bool = True,
    min_trials_for_robust: int = 2,
) -> dict[str, Any]:
    """Tune ``reflection_update_period`` for adaptive lattice-ladder NLMS.

    Parameters
    ----------
    x, desired:
        Either 1-D arrays with shape ``(samples,)`` or 2-D arrays with shape
        ``(trials, samples)``.  With 2-D data, each row is treated as an
        independent validation trial, and ``max_worst_tail_mse_ratio`` is
        evaluated across rows.
    periods:
        Candidate denominator/reflection update periods.
    initial_reflection, initial_taps, order:
        Initial model parameters.  If no reflection/taps are provided, a
        zero-initialized order-4 model is used so the convenience API works out
        of the box.  For real experiments, pass ``order`` or explicit initial
        coefficients.
    max_tail_mse_ratio:
        Constraint on the median tail-MSE ratio versus the first candidate
        period, typically period 1.
    max_worst_tail_mse_ratio:
        Constraint on the worst tail-MSE ratio across trials.  Defaults to the
        same value as ``max_tail_mse_ratio``.
    min_trials_for_robust:
        Minimum number of independent validation rows required before the
        recommendation is labelled robust.  With fewer trials, the function
        still returns a period, but the report includes a warning that the
        recommendation is signal-specific.
    prefer:
        ``"fastest"`` chooses the fastest eligible period. ``"quality"`` /
        ``"accurate"`` chooses the lowest median tail MSE. ``"balanced"`` uses
        a simple speedup / worst-tail-ratio score.

    Returns
    -------
    dict
        A report dictionary with ``recommended_period``, ``recommended``,
        ``results``, ``warnings``, and ``metadata`` keys.
    """

    period_list = _parse_periods(periods)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    min_trials_for_robust_i = int(min_trials_for_robust)
    if min_trials_for_robust_i <= 0:
        raise ValueError("min_trials_for_robust must be positive")
    if max_worst_tail_mse_ratio is None:
        max_worst_tail_mse_ratio = max_tail_mse_ratio

    x_trials = _as_trial_matrix("x", x)
    desired_trials = _as_trial_matrix("desired", desired)
    if x_trials.shape != desired_trials.shape:
        raise ValueError("x and desired must have the same shape")

    n_trials, n_samples = x_trials.shape
    tail_n = _tail_count(tail, n_samples)
    is_robust_recommendation = n_trials >= min_trials_for_robust_i
    recommendation_scope = "robust" if is_robust_recommendation else "single_signal"
    warnings: list[str] = []
    if not is_robust_recommendation:
        warnings.append(
            "Only one/few validation trial(s) were provided, so the selected "
            "reflection_update_period is signal-specific rather than a robust "
            "default. Pass a 2-D trial-by-sample array to validate worst-case "
            "behavior across independent signals."
        )
    reflection0, taps0, inferred_order = _infer_initial_parameters(
        initial_reflection, initial_taps, order
    )

    rows: list[dict[str, Any]] = []
    baseline_time: float | None = None
    baseline_tail_mse: list[float] | None = None

    for period in period_list:
        timings: list[float] = []
        metrics: list[dict[str, float]] = []
        for repeat_idx in range(int(repeats)):
            trial_metrics: list[dict[str, float]] = []
            start = time.perf_counter()
            for trial_idx in range(n_trials):
                adaptive = AdaptiveLatticeLadderNLMS(
                    reflection0,
                    taps0,
                    mu_taps=mu_taps,
                    mu_reflection=mu_reflection,
                    epsilon=epsilon,
                    margin=margin,
                    freeze_reflection=False,
                    gradient_mode=gradient_mode,
                    reflection_update_period=period,
                    scale_reflection_mu_by_period=scale_reflection_mu_by_period,
                )
                y, err = adaptive.process_adapt(x_trials[trial_idx], desired_trials[trial_idx])
                y_arr = np.asarray(y, dtype=np.float64)
                err_arr = np.asarray(err, dtype=np.float64)
                final_reflection = np.asarray(adaptive.reflection, dtype=np.float64)
                max_abs_reflection = (
                    float(np.max(np.abs(final_reflection))) if final_reflection.size else 0.0
                )
                trial_metrics.append(
                    {
                        "mse_total": _mse(err_arr),
                        "mse_head": _mse(err_arr[:tail_n]),
                        "mse_tail": _mse(err_arr[-tail_n:]),
                        "output_power": _mse(y_arr),
                        "max_abs_reflection": max_abs_reflection,
                        "stability_margin": float(1.0 - max_abs_reflection),
                    }
                )
            timings.append(time.perf_counter() - start)
            if repeat_idx == int(repeats) - 1:
                metrics = trial_metrics

        median_s = _median(timings)
        if baseline_time is None:
            baseline_time = median_s
            baseline_tail_mse = [metric["mse_tail"] for metric in metrics]
        assert baseline_tail_mse is not None

        tail_ratios = [
            _safe_ratio(metric["mse_tail"], baseline_tail_mse[idx])
            for idx, metric in enumerate(metrics)
        ]
        mse_tail_values = [metric["mse_tail"] for metric in metrics]
        mse_total_values = [metric["mse_total"] for metric in metrics]
        stability_values = [metric["stability_margin"] for metric in metrics]
        max_reflection_values = [metric["max_abs_reflection"] for metric in metrics]

        rows.append(
            {
                "reflection_update_period": period,
                "n_trials": n_trials,
                "min_s": float(min(timings)),
                "median_s": median_s,
                "max_s": float(max(timings)),
                "speedup_vs_period1_median": float(baseline_time / median_s)
                if median_s > 0.0
                else math.inf,
                "mse_tail_median": _median(mse_tail_values),
                "mse_tail_worst": float(max(mse_tail_values)),
                "mse_total_median": _median(mse_total_values),
                "tail_mse_ratio_median": _median(tail_ratios),
                "tail_mse_ratio_p90": _quantile(tail_ratios, 0.9),
                "tail_mse_ratio_worst": float(max(tail_ratios)),
                "stability_margin_median": _median(stability_values),
                "stability_margin_min": float(min(stability_values)),
                "max_abs_reflection_median": _median(max_reflection_values),
                "max_abs_reflection_max": float(max(max_reflection_values)),
            }
        )

    recommended = _recommend_period(
        rows,
        max_tail_mse_ratio=float(max_tail_mse_ratio),
        max_worst_tail_mse_ratio=float(max_worst_tail_mse_ratio),
        min_stability_margin=float(min_stability_margin),
        prefer=prefer,
    )

    return {
        "recommended_period": None
        if recommended is None
        else int(recommended["reflection_update_period"]),
        "recommended": recommended,
        "results": rows,
        "warnings": warnings,
        "metadata": {
            "periods": period_list,
            "order": inferred_order,
            "n_trials": n_trials,
            "min_trials_for_robust": min_trials_for_robust_i,
            "is_robust_recommendation": is_robust_recommendation,
            "recommendation_scope": recommendation_scope,
            "samples": n_samples,
            "tail": tail_n,
            "repeats": int(repeats),
            "mu_taps": float(mu_taps),
            "mu_reflection": float(mu_reflection),
            "epsilon": float(epsilon),
            "margin": float(margin),
            "gradient_mode": gradient_mode,
            "scale_reflection_mu_by_period": bool(scale_reflection_mu_by_period),
            "max_tail_mse_ratio": float(max_tail_mse_ratio),
            "max_worst_tail_mse_ratio": float(max_worst_tail_mse_ratio),
            "min_stability_margin": float(min_stability_margin),
            "prefer": prefer,
        },
    }


__all__ = ["tune_reflection_update_period"]
