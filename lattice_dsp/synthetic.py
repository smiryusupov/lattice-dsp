"""Synthetic signal generators for echo/noise-cancellation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from collections.abc import Sequence

import numpy as np

from ._core import reflection_to_denominator


@dataclass(frozen=True)
class EchoProblem:
    """Synthetic echo/noise-cancellation problem.

    Attributes
    ----------
    reference:
        Far-end/reference signal.
    microphone:
        Observed microphone signal containing echo plus clean target.
    clean_target:
        Desired signal that should remain after cancellation; near-end plus
        background noise in this synthetic setup.
    linear_echo:
        Echo produced by the stable linear IIR path.
    nonlinear_echo:
        Residual nonlinear echo component produced from a distorted reference.
    near_end:
        Optional local/near-end signal.
    noise:
        Background noise component.
    denominator, numerator:
        Coefficients of the linear echo path.
    reflection, taps:
        Stable lattice/lattice-ladder parameters used to construct the path.
    sample_rate:
        Sample rate in Hz.
    """

    reference: np.ndarray
    microphone: np.ndarray
    clean_target: np.ndarray
    linear_echo: np.ndarray
    nonlinear_echo: np.ndarray
    near_end: np.ndarray
    noise: np.ndarray
    denominator: np.ndarray
    numerator: np.ndarray
    reflection: np.ndarray
    taps: np.ndarray
    sample_rate: int


def _as_coefficients(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    return np.ascontiguousarray(array)


def iir_filter(
    numerator: Sequence[float], denominator: Sequence[float], signal: Sequence[float]
) -> np.ndarray:
    """Small direct-form IIR reference implementation for synthetic data."""

    b = _as_coefficients("numerator", numerator)
    a = _as_coefficients("denominator", denominator)
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1-D")
    if not np.isclose(a[0], 1.0):
        b = b / a[0]
        a = a / a[0]

    y = np.zeros_like(x, dtype=np.float64)
    for n in range(x.size):
        acc = 0.0
        for i, bi in enumerate(b):
            if n >= i:
                acc += float(bi) * float(x[n - i])
        for i in range(1, a.size):
            if n >= i:
                acc -= float(a[i]) * float(y[n - i])
        y[n] = acc
    return y


def _colored_noise(rng: np.random.Generator, samples: int, pole: float = 0.85) -> np.ndarray:
    white = rng.normal(size=samples)
    y = np.empty(samples, dtype=np.float64)
    state = 0.0
    for n, sample in enumerate(white):
        state = pole * state + sample
        y[n] = state
    y -= float(np.mean(y))
    std = float(np.std(y))
    return y / std if std > 0 else y


def _scale_to_power(signal: np.ndarray, target_power: float) -> np.ndarray:
    current = float(np.mean(signal * signal))
    if current <= 0.0 or target_power <= 0.0:
        return np.zeros_like(signal)
    return signal * float(np.sqrt(target_power / current))


def generate_echo_problem(
    *,
    samples: int = 64_000,
    sample_rate: int = 16_000,
    seed: int = 1234,
    reflection: Sequence[float] = (0.35, -0.25, 0.15, -0.08),
    taps: Sequence[float] = (0.2, -0.1, 0.05, 0.0, 0.75),
    nonlinear_strength: float = 0.08,
    nonlinearity: Literal["none", "tanh", "cubic", "clipped"] = "tanh",
    near_end_power_ratio: float = 0.02,
    noise_snr_db: float = 30.0,
    double_talk: bool = True,
) -> EchoProblem:
    """Generate a reproducible synthetic nonlinear echo problem.

    The linear path is stable because it is parameterized by reflection
    coefficients.  The nonlinear component is intentionally small; the goal is
    to make the linear lattice/IIR stage useful while leaving a residual that a
    spectral or downstream residual processor could handle later.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if nonlinear_strength < 0.0:
        raise ValueError("nonlinear_strength must be non-negative")
    if near_end_power_ratio < 0.0:
        raise ValueError("near_end_power_ratio must be non-negative")

    rng = np.random.default_rng(seed)
    reflection_array = _as_coefficients("reflection", reflection)
    taps_array = _as_coefficients("taps", taps)
    denominator = np.asarray(reflection_to_denominator(reflection_array), dtype=np.float64)
    numerator = taps_array.copy()

    reference = _colored_noise(rng, samples, pole=0.65)
    linear_echo = iir_filter(numerator, denominator, reference)

    if nonlinearity == "none" or nonlinear_strength == 0.0:
        nonlinear_drive = np.zeros_like(reference)
    elif nonlinearity == "tanh":
        nonlinear_drive = np.tanh(1.5 * reference) - reference
    elif nonlinearity == "cubic":
        nonlinear_drive = reference**3
        nonlinear_drive -= float(np.mean(nonlinear_drive))
    elif nonlinearity == "clipped":
        nonlinear_drive = np.clip(reference, -0.75, 0.75) - reference
    else:  # pragma: no cover - Literal typing should prevent this.
        raise ValueError(f"unsupported nonlinearity: {nonlinearity}")
    nonlinear_echo = nonlinear_strength * iir_filter(numerator, denominator, nonlinear_drive)

    echo = linear_echo + nonlinear_echo
    echo_power = float(np.mean(echo * echo))

    if double_talk and near_end_power_ratio > 0.0:
        near_raw = _colored_noise(rng, samples, pole=0.92)
        # Make near-end sparse-ish so the benchmark resembles far-end dominant
        # echo periods mixed with short local activity.
        envelope = np.zeros(samples, dtype=np.float64)
        segment = max(sample_rate // 2, 1)
        for start in range(0, samples, 2 * segment):
            stop = min(start + segment, samples)
            envelope[start:stop] = 1.0
        near_end = _scale_to_power(near_raw * envelope, echo_power * near_end_power_ratio)
    else:
        near_end = np.zeros(samples, dtype=np.float64)

    if np.isfinite(noise_snr_db):
        noise_power = echo_power / (10.0 ** (float(noise_snr_db) / 10.0))
    else:
        noise_power = 0.0
    noise = _scale_to_power(rng.normal(size=samples), noise_power)

    clean_target = near_end + noise
    microphone = echo + clean_target
    return EchoProblem(
        reference=np.ascontiguousarray(reference),
        microphone=np.ascontiguousarray(microphone),
        clean_target=np.ascontiguousarray(clean_target),
        linear_echo=np.ascontiguousarray(linear_echo),
        nonlinear_echo=np.ascontiguousarray(nonlinear_echo),
        near_end=np.ascontiguousarray(near_end),
        noise=np.ascontiguousarray(noise),
        denominator=np.ascontiguousarray(denominator),
        numerator=np.ascontiguousarray(numerator),
        reflection=np.ascontiguousarray(reflection_array),
        taps=np.ascontiguousarray(taps_array),
        sample_rate=sample_rate,
    )


__all__ = ["EchoProblem", "generate_echo_problem", "iir_filter"]
