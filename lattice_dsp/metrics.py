# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Metrics for echo cancellation, noise cancellation, and system identification.

The functions in this module intentionally avoid audio-specific dependencies.
They operate on NumPy arrays and are useful for synthetic benchmarks where the
clean target or true echo component is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def as_float_array(name: str, value: Any) -> np.ndarray:
    """Return ``value`` as a contiguous 1-D float64 array."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return np.ascontiguousarray(array)


def power(signal: Any, *, epsilon: float = 0.0) -> float:
    """Return mean-square power of a signal."""

    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        raise ValueError("signal must be non-empty")
    value = float(np.mean(np.square(x)))
    return max(value, float(epsilon))


def power_db(signal: Any, *, reference: float = 1.0, epsilon: float = 1e-12) -> float:
    """Return signal power in dB relative to ``reference``."""

    ref = max(float(reference), float(epsilon))
    return float(10.0 * np.log10(power(signal, epsilon=epsilon) / ref))


def mse(reference: Any, estimate: Any) -> float:
    """Return mean-squared error between equally shaped arrays."""

    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape")
    return power(ref - est)


def improvement_db(before_mse: float, after_mse: float, *, epsilon: float = 1e-12) -> float:
    """Return MSE improvement in dB from ``before_mse`` to ``after_mse``."""

    before = max(float(before_mse), float(epsilon))
    after = max(float(after_mse), float(epsilon))
    return float(10.0 * np.log10(before / after))


def erle_db(echo_or_input_error: Any, residual_error: Any, *, epsilon: float = 1e-12) -> float:
    """Return echo-return-loss enhancement in dB.

    In synthetic experiments, ``echo_or_input_error`` is usually the microphone
    signal minus the clean near-end/noise target before cancellation.  The
    ``residual_error`` is the enhanced output minus the same clean target after
    cancellation.  Positive values mean echo/error power was reduced.
    """

    before = power(echo_or_input_error, epsilon=epsilon)
    after = power(residual_error, epsilon=epsilon)
    return improvement_db(before, after, epsilon=epsilon)


def segmental_erle_db(
    echo_or_input_error: Any,
    residual_error: Any,
    *,
    frame_length: int = 1024,
    hop_length: int | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return frame-wise ERLE values.

    The last incomplete frame is ignored.  ``hop_length`` defaults to
    ``frame_length`` for non-overlapping frames.
    """

    before = as_float_array("echo_or_input_error", echo_or_input_error)
    after = as_float_array("residual_error", residual_error)
    if before.shape != after.shape:
        raise ValueError("echo_or_input_error and residual_error must have the same shape")
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    hop = frame_length if hop_length is None else int(hop_length)
    if hop <= 0:
        raise ValueError("hop_length must be positive")
    values: list[float] = []
    for start in range(0, before.size - frame_length + 1, hop):
        stop = start + frame_length
        values.append(erle_db(before[start:stop], after[start:stop], epsilon=epsilon))
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class EchoMetrics:
    """Scalar metrics for a synthetic echo/noise-cancellation run."""

    input_mse: float
    output_mse: float
    mse_improvement_db: float
    erle_db: float
    residual_power_db: float
    segmental_erle_median_db: float

    def as_dict(self) -> dict[str, float]:
        return {
            "input_mse": self.input_mse,
            "output_mse": self.output_mse,
            "mse_improvement_db": self.mse_improvement_db,
            "erle_db": self.erle_db,
            "residual_power_db": self.residual_power_db,
            "segmental_erle_median_db": self.segmental_erle_median_db,
        }


def echo_metrics(
    microphone: Any,
    enhanced: Any,
    clean_target: Any,
    *,
    frame_length: int = 1024,
    epsilon: float = 1e-12,
) -> EchoMetrics:
    """Compute synthetic echo-cancellation metrics.

    ``clean_target`` is the desired signal that should remain after echo/noise
    cancellation.  In real recordings it is usually unavailable; in synthetic
    benchmarks it lets us compute ERLE and MSE improvement consistently.
    """

    mic = as_float_array("microphone", microphone)
    out = as_float_array("enhanced", enhanced)
    clean = as_float_array("clean_target", clean_target)
    if mic.shape != out.shape or mic.shape != clean.shape:
        raise ValueError("microphone, enhanced, and clean_target must have the same shape")

    input_error = mic - clean
    output_error = out - clean
    input_mse = power(input_error, epsilon=epsilon)
    output_mse = power(output_error, epsilon=epsilon)
    seg = segmental_erle_db(
        input_error,
        output_error,
        frame_length=min(frame_length, mic.size),
        epsilon=epsilon,
    )
    seg_median = float(np.median(seg)) if seg.size else erle_db(input_error, output_error)
    return EchoMetrics(
        input_mse=input_mse,
        output_mse=output_mse,
        mse_improvement_db=improvement_db(input_mse, output_mse, epsilon=epsilon),
        erle_db=erle_db(input_error, output_error, epsilon=epsilon),
        residual_power_db=power_db(output_error, epsilon=epsilon),
        segmental_erle_median_db=seg_median,
    )


__all__ = [
    "EchoMetrics",
    "echo_metrics",
    "erle_db",
    "improvement_db",
    "mse",
    "power",
    "power_db",
    "segmental_erle_db",
]
