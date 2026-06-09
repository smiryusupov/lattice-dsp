# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Small application-layer helpers for synthetic echo/noise experiments.

The core package intentionally stays dependency-light.  This module provides a
thin bridge between stable adaptive lattice/IIR filters and simple residual
processors such as fixed-gain or deterministic spectral suppressors.  These
helpers are examples and diagnostics, not production acoustic echo cancellation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._core import AdaptiveLatticeLadderNLMS

ResidualProcessor = Callable[[np.ndarray, Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True)
class EchoCancellationResult:
    """Container returned by :class:`HybridEchoCanceller`.

    Attributes
    ----------
    echo_estimate:
        Linear echo/noise estimate produced by the adaptive lattice/IIR stage.
    residual:
        Microphone signal minus the linear estimate.  In echo cancellation this
        is the signal passed to an optional residual suppressor.
    enhanced:
        Residual after the optional residual processor.  If no residual
        processor is supplied, this is equal to ``residual``.
    error:
        Alias for ``residual``; kept for adaptive-filter terminology.
    reflection, taps, denominator, numerator:
        Final adaptive filter parameters after processing the block.
    metrics:
        Optional scalar diagnostics.  When a clean target is provided, the
        metrics include input/residual/enhanced MSE and improvement in dB.
    """

    echo_estimate: np.ndarray
    residual: np.ndarray
    enhanced: np.ndarray
    error: np.ndarray
    reflection: np.ndarray
    taps: np.ndarray
    denominator: np.ndarray
    numerator: np.ndarray
    metrics: dict[str, float]


def _as_1d_float(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return np.ascontiguousarray(array)


def mse(reference: Any, estimate: Any) -> float:
    """Return mean-squared error between two equally shaped arrays."""

    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape")
    return float(np.mean(np.square(ref - est)))


def improvement_db(before_mse: float, after_mse: float, *, epsilon: float = 1e-12) -> float:
    """Return improvement in dB from ``before_mse`` to ``after_mse``.

    Positive values mean the after-MSE is lower.  ``epsilon`` prevents division
    by zero in silent or perfectly reconstructed synthetic examples.
    """

    before = max(float(before_mse), float(epsilon))
    after = max(float(after_mse), float(epsilon))
    return float(10.0 * np.log10(before / after))


class ResidualAttenuator:
    """Dependency-free residual suppressor that applies a fixed gain.

    This is deliberately simple.  It is useful as a minimal placeholder for
    a residual echo suppressor, but it will attenuate near-end speech as well
    as residual echo.
    """

    def __init__(self, gain: float = 0.7) -> None:
        if gain < 0.0:
            raise ValueError("gain must be non-negative")
        self.gain = float(gain)

    def __call__(
        self, residual: np.ndarray, context: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        del context
        return np.asarray(residual, dtype=np.float64) * self.gain


def residual_attenuator(signal: Any, *, gain: float = 0.7) -> np.ndarray:
    """Return ``signal`` multiplied by ``gain`` as a tiny residual baseline."""

    return ResidualAttenuator(gain=gain)(np.asarray(signal, dtype=np.float64), {})


class SpectralResidualSuppressor:
    """Small deterministic STFT-domain residual echo suppressor.

    This class is intentionally **not** a production denoiser.  It provides a
    reproducible, dependency-free residual-model baseline beyond the toy gain
    attenuator.

    By default the suppressor is **echo-aware** when a context dictionary with
    ``reference`` or ``echo_estimate`` is supplied.  It estimates time-frequency
    coherence between the residual and the echo/reference signal, then attenuates
    components that remain correlated with the far-end path.  This is safer for
    echo-cancellation benchmarks than blind spectral gating, because near-end
    speech/noise should be preserved when it is not coherent with the far-end.

    If no useful context is supplied, the class falls back to a blind STFT soft
    gate.  The blind mode is useful as a denoising sanity check but can damage the
    desired near-end target and should not be interpreted as residual echo
    suppression.

    Parameters
    ----------
    frame_size:
        STFT frame length.  If the signal is shorter, it is padded internally.
    hop_size:
        Frame hop.  Defaults to one quarter of ``frame_size``.
    mode:
        ``"echo_aware"`` uses far-end/echo coherence when context is available;
        ``"blind"`` always uses the old percentile-floor soft gate.
    floor:
        Minimum gain.  A value of ``0`` can create artifacts; ``0.05`` to
        ``0.2`` is usually safer for examples.
    over_subtract:
        Blind-mode multiplier applied to the estimated residual floor.
    noise_percentile:
        Blind-mode percentile across time used as the residual floor estimate.
    smoothing:
        Temporal gain smoothing in ``[0, 1)``.  Larger values vary more slowly.
    exponent:
        Optional gain curve exponent.  In echo-aware mode this is applied to the
        coherence before suppression; in blind mode it shapes the soft gate.
    echo_aware_strength:
        Suppression strength for coherence-based echo-aware mode.
    reference_key:
        Preferred context key for the far-end/echo-correlated signal.  If this
        key is absent, ``echo_estimate`` and then ``reference`` are tried.
    echo_gate:
        If true, scale echo-aware suppression by a block-level echo-confidence
        estimate.  This helps real-audio/double-talk cases where the residual
        is dominated by near-end content and a fixed global suppression strength
        can over-suppress the target.
    echo_gate_threshold:
        Echo-confidence threshold below which suppression is reduced toward
        ``echo_gate_floor``.
    echo_gate_transition:
        Width of the soft transition from reduced to full suppression.
    echo_gate_floor:
        Minimum fraction of ``echo_aware_strength`` kept when the gate is closed.
    """

    def __init__(
        self,
        *,
        frame_size: int = 512,
        hop_size: int | None = None,
        mode: str = "echo_aware",
        floor: float = 0.08,
        over_subtract: float = 1.25,
        noise_percentile: float = 20.0,
        smoothing: float = 0.65,
        exponent: float = 1.0,
        echo_aware_strength: float = 0.7,
        reference_key: str = "echo_estimate",
        echo_gate: bool = False,
        echo_gate_threshold: float = 0.05,
        echo_gate_transition: float = 0.20,
        echo_gate_floor: float = 0.0,
        epsilon: float = 1e-10,
    ) -> None:
        if frame_size <= 8:
            raise ValueError("frame_size must be greater than 8")
        if hop_size is None:
            hop_size = frame_size // 4
        if hop_size <= 0 or hop_size > frame_size:
            raise ValueError("hop_size must be in the range [1, frame_size]")
        if mode not in {"echo_aware", "blind"}:
            raise ValueError("mode must be 'echo_aware' or 'blind'")
        if not 0.0 <= floor <= 1.0:
            raise ValueError("floor must be in [0, 1]")
        if over_subtract < 0.0:
            raise ValueError("over_subtract must be non-negative")
        if not 0.0 <= noise_percentile <= 100.0:
            raise ValueError("noise_percentile must be in [0, 100]")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        if exponent <= 0.0:
            raise ValueError("exponent must be positive")
        if echo_aware_strength < 0.0:
            raise ValueError("echo_aware_strength must be non-negative")
        if echo_gate_threshold < 0.0:
            raise ValueError("echo_gate_threshold must be non-negative")
        if echo_gate_transition <= 0.0:
            raise ValueError("echo_gate_transition must be positive")
        if not 0.0 <= echo_gate_floor <= 1.0:
            raise ValueError("echo_gate_floor must be in [0, 1]")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.mode = str(mode)
        self.floor = float(floor)
        self.over_subtract = float(over_subtract)
        self.noise_percentile = float(noise_percentile)
        self.smoothing = float(smoothing)
        self.exponent = float(exponent)
        self.echo_aware_strength = float(echo_aware_strength)
        self.reference_key = str(reference_key)
        self.echo_gate = bool(echo_gate)
        self.echo_gate_threshold = float(echo_gate_threshold)
        self.echo_gate_transition = float(echo_gate_transition)
        self.echo_gate_floor = float(echo_gate_floor)
        self.last_echo_gate_scale = 1.0
        self.last_echo_confidence = 1.0
        self.epsilon = float(epsilon)

    def __call__(
        self, residual: np.ndarray, context: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        if self.mode == "echo_aware":
            reference = self._reference_from_context(context)
            if reference is not None:
                return self.process_echo_aware(residual, reference)
        return self.process(residual)

    def _reference_from_context(self, context: Mapping[str, Any] | None) -> np.ndarray | None:
        if not context:
            return None
        candidates = [self.reference_key, "echo_estimate", "reference"]
        seen: set[str] = set()
        for key in candidates:
            if key in seen:
                continue
            seen.add(key)
            value = context.get(key)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.ndim == 1 and array.size > 0:
                return np.ascontiguousarray(array)
        return None

    def _stft(self, signal: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        frame_size = self.frame_size
        hop = self.hop_size
        length = int(signal.size)
        n_frames = max(1, int(np.ceil(max(0, length - frame_size) / hop)) + 1)
        padded_length = (n_frames - 1) * hop + frame_size
        padded = np.zeros(padded_length, dtype=np.float64)
        padded[:length] = signal
        window = np.hanning(frame_size).astype(np.float64)
        if not np.any(window):
            window = np.ones(frame_size, dtype=np.float64)
        spectra = []
        for i in range(n_frames):
            start = i * hop
            spectra.append(np.fft.rfft(padded[start : start + frame_size] * window))
        return np.stack(spectra, axis=0), length, window

    def _istft(self, spec: np.ndarray, length: int, window: np.ndarray) -> np.ndarray:
        frame_size = self.frame_size
        hop = self.hop_size
        n_frames = int(spec.shape[0])
        padded_length = (n_frames - 1) * hop + frame_size
        out = np.zeros(padded_length, dtype=np.float64)
        norm = np.zeros(padded_length, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            frame = np.fft.irfft(spec[i], n=frame_size).real
            out[start : start + frame_size] += frame * window
            norm[start : start + frame_size] += window * window
        # Standard overlap-add normalization divides by the squared-window
        # sum.  Hann windows have very small values near block boundaries, and
        # after spectral gain modification those tiny denominators can amplify
        # harmless reconstruction leakage into very large edge samples.  Use a
        # relative floor so gain-modified frames stay bounded; this only affects
        # the first/last few samples or pathological low-overlap regions.
        norm_peak = float(np.max(norm)) if norm.size else 0.0
        norm_floor = max(self.epsilon, 1e-4 * norm_peak)
        out /= np.maximum(norm, norm_floor)
        return out[:length]

    def process(self, signal: Any) -> np.ndarray:
        """Blindly suppress a 1-D residual signal and return an equal-length array.

        This method does not use reference/echo context.  For echo-cancellation
        applications, prefer calling the object with a context dictionary so
        :meth:`process_echo_aware` can preserve uncorrelated near-end content.
        """

        x = _as_1d_float("signal", signal)
        if x.size == 0:
            return x.copy()

        spec, length, window = self._stft(x)
        mag = np.abs(spec)
        floor_estimate = np.percentile(mag, self.noise_percentile, axis=0)

        raw_gain = 1.0 - (self.over_subtract * floor_estimate[None, :]) / (mag + self.epsilon)
        raw_gain = np.clip(raw_gain, self.floor, 1.0)
        if self.exponent != 1.0:
            raw_gain = np.power(raw_gain, self.exponent)
            raw_gain = np.clip(raw_gain, self.floor, 1.0)

        if self.smoothing > 0.0 and raw_gain.shape[0] > 1:
            raw_gain = self._smooth_gain(raw_gain)

        return self._istft(spec * raw_gain, length, window)

    def _echo_gate_scale(self, residual: np.ndarray, reference: np.ndarray) -> float:
        """Return a block-level scale for echo-aware suppression strength.

        The reference is normally the adaptive echo estimate.  When its power is
        small relative to the residual, the block is likely near-end dominated
        or the linear stage has little confidence.  In that case, fixed spectral
        suppression can reduce target quality even when it lowers leakage
        diagnostics.  The gate keeps suppression conservative for those blocks.
        """

        if not self.echo_gate:
            self.last_echo_confidence = 1.0
            self.last_echo_gate_scale = 1.0
            return 1.0
        r_power = float(np.mean(np.square(residual))) if residual.size else 0.0
        x_power = float(np.mean(np.square(reference))) if reference.size else 0.0
        confidence = x_power / max(x_power + r_power, self.epsilon)
        scale = (confidence - self.echo_gate_threshold) / self.echo_gate_transition
        scale = float(np.clip(scale, 0.0, 1.0))
        scale = self.echo_gate_floor + (1.0 - self.echo_gate_floor) * scale
        self.last_echo_confidence = float(confidence)
        self.last_echo_gate_scale = float(scale)
        return float(scale)

    def process_echo_aware(self, residual: Any, reference: Any) -> np.ndarray:
        """Suppress far-end-coherent residual components.

        ``reference`` can be the original far-end signal or the linear echo
        estimate.  The latter is usually preferred because it already includes
        the learned acoustic path coloration.
        """

        r = _as_1d_float("residual", residual)
        x = _as_1d_float("reference", reference)
        if r.size == 0:
            return r.copy()
        if x.size != r.size:
            raise ValueError("reference must have the same length as residual")

        r_spec, length, window = self._stft(r)
        x_spec, _, _ = self._stft(x)
        if x_spec.shape != r_spec.shape:
            raise RuntimeError("internal STFT shape mismatch")

        n_frames, n_bins = r_spec.shape
        gain = np.ones((n_frames, n_bins), dtype=np.float64)
        alpha = self.smoothing
        s_rr = np.zeros(n_bins, dtype=np.float64)
        s_xx = np.zeros(n_bins, dtype=np.float64)
        s_rx = np.zeros(n_bins, dtype=np.complex128)
        strength = min(self.echo_aware_strength * self._echo_gate_scale(r, x), 2.0)

        for i in range(n_frames):
            rr = np.abs(r_spec[i]) ** 2
            xx = np.abs(x_spec[i]) ** 2
            rx = r_spec[i] * np.conj(x_spec[i])
            if i == 0:
                s_rr = rr
                s_xx = xx
                s_rx = rx
            else:
                s_rr = alpha * s_rr + (1.0 - alpha) * rr
                s_xx = alpha * s_xx + (1.0 - alpha) * xx
                s_rx = alpha * s_rx + (1.0 - alpha) * rx
            coherence = (np.abs(s_rx) ** 2) / (s_rr * s_xx + self.epsilon)
            coherence = np.clip(coherence, 0.0, 1.0)
            shaped = np.power(coherence, self.exponent)
            gain[i] = np.clip(1.0 - strength * shaped, self.floor, 1.0)

        if self.smoothing > 0.0 and n_frames > 1:
            # A second light pass smooths the final gain while keeping the
            # coherence estimator itself causal/reproducible.
            gain = self._smooth_gain(gain)

        return self._istft(r_spec * gain, length, window)

    def _smooth_gain(self, gain: np.ndarray) -> np.ndarray:
        if gain.shape[0] <= 1 or self.smoothing <= 0.0:
            return gain
        smoothed = np.empty_like(gain)
        smoothed[0] = gain[0]
        alpha = self.smoothing
        for i in range(1, gain.shape[0]):
            smoothed[i] = alpha * smoothed[i - 1] + (1.0 - alpha) * gain[i]
        return smoothed


def spectral_residual_suppressor(signal: Any, **kwargs: Any) -> np.ndarray:
    """Convenience wrapper around :class:`SpectralResidualSuppressor`."""

    return SpectralResidualSuppressor(**kwargs).process(signal)


class HybridEchoCanceller:
    """Linear adaptive lattice/IIR stage plus optional residual processor.

    This helper is intentionally small.  It models the common hybrid strategy:

    1. a stable adaptive lattice/IIR filter estimates the linear echo path from
       a reference/far-end signal to the microphone signal;
    2. the residual signal is passed to an optional dependency-free processor
       such as a fixed attenuator, a spectral suppressor, or any callable
       supplied by the user.

    The residual processor is called as ``processor(residual, context)``.  The
    context dictionary contains ``reference``, ``microphone``, ``echo_estimate``,
    ``sample_rate``, and ``linear_filter``.
    """

    def __init__(
        self,
        initial_reflection: Sequence[float],
        initial_taps: Sequence[float],
        *,
        mu_taps: float = 0.05,
        mu_reflection: float = 0.001,
        epsilon: float = 1e-8,
        margin: float = 1e-4,
        freeze_reflection: bool = False,
        gradient_mode: str = "analytic",
        reflection_update_period: int = 8,
        scale_reflection_mu_by_period: bool = True,
        residual_processor: ResidualProcessor | None = None,
        sample_rate: int | None = None,
        safety_max_power_ratio: float | None = None,
        safety_max_abs: float | None = None,
    ) -> None:
        self.linear_filter = AdaptiveLatticeLadderNLMS(
            initial_reflection,
            initial_taps,
            mu_taps=mu_taps,
            mu_reflection=mu_reflection,
            epsilon=epsilon,
            margin=margin,
            freeze_reflection=freeze_reflection,
            gradient_mode=gradient_mode,
            reflection_update_period=reflection_update_period,
            scale_reflection_mu_by_period=scale_reflection_mu_by_period,
        )
        if safety_max_power_ratio is not None and safety_max_power_ratio <= 0.0:
            raise ValueError("safety_max_power_ratio must be positive or None")
        if safety_max_abs is not None and safety_max_abs <= 0.0:
            raise ValueError("safety_max_abs must be positive or None")
        self.residual_processor = residual_processor
        self.sample_rate = sample_rate
        self.safety_max_power_ratio = (
            None if safety_max_power_ratio is None else float(safety_max_power_ratio)
        )
        self.safety_max_abs = None if safety_max_abs is None else float(safety_max_abs)

    def reset(self, value: float = 0.0) -> None:
        """Reset the internal adaptive filter state."""

        self.linear_filter.reset(float(value))

    def _safety_check(
        self, candidate: np.ndarray, microphone: np.ndarray
    ) -> tuple[bool, float, float]:
        """Return whether ``candidate`` is safe relative to ``microphone``.

        The guard is intentionally simple and deterministic.  It is meant for
        real-audio evaluation where occasional adaptive divergence should fall
        back to a conservative signal instead of dominating aggregate metrics
        with huge floating-point values.
        """

        y = np.asarray(candidate, dtype=np.float64)
        mic = np.asarray(microphone, dtype=np.float64)
        if y.shape != mic.shape or not np.all(np.isfinite(y)):
            return False, float("inf"), float("inf")
        max_abs = float(np.max(np.abs(y))) if y.size else 0.0
        mic_power = float(np.mean(np.square(mic))) if mic.size else 0.0
        out_power = float(np.mean(np.square(y))) if y.size else 0.0
        ratio = out_power / max(mic_power, 1e-24)
        if self.safety_max_abs is not None and max_abs > self.safety_max_abs:
            return False, ratio, max_abs
        if self.safety_max_power_ratio is not None and ratio > self.safety_max_power_ratio:
            return False, ratio, max_abs
        return True, ratio, max_abs

    def process(
        self,
        reference: Any,
        microphone: Any,
        *,
        clean_target: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> EchoCancellationResult:
        """Process one reference/microphone block.

        Parameters
        ----------
        reference:
            Far-end/reference signal used by the linear adaptive filter.
        microphone:
            Microphone signal containing echo/noise plus desired near-end
            content.  During supervised synthetic tests this can be the desired
            response of a known echo path.
        clean_target:
            Optional clean signal used only for metrics.  In echo cancellation
            this is usually unavailable in real deployments, but it is useful
            for simulations and benchmarks.
        context:
            Optional extra values passed to the residual processor.
        """

        x = _as_1d_float("reference", reference)
        mic = _as_1d_float("microphone", microphone)
        if x.shape != mic.shape:
            raise ValueError("reference and microphone must have the same shape")

        echo_estimate_raw, residual_raw = self.linear_filter.process_adapt(x, mic)
        echo_estimate = np.asarray(echo_estimate_raw, dtype=np.float64)
        residual = np.asarray(residual_raw, dtype=np.float64)

        safety_triggered = 0.0
        safety_stage = 0.0
        residual_safe, residual_ratio, residual_max_abs = self._safety_check(residual, mic)
        enhanced_ratio = residual_ratio
        enhanced_max_abs = residual_max_abs
        if not residual_safe:
            # The linear adaptive stage diverged.  Fall back to no cancellation
            # for this block so benchmarks/users get a bounded, conservative
            # result instead of a numerically explosive residual.
            safety_triggered = 1.0
            safety_stage = 1.0
            echo_estimate = np.zeros_like(mic)
            residual = mic.copy()
            enhanced = mic.copy()
        else:
            processor_context: dict[str, Any] = {
                "reference": x,
                "microphone": mic,
                "echo_estimate": echo_estimate,
                "sample_rate": self.sample_rate,
                "linear_filter": self.linear_filter,
            }
            if context:
                processor_context.update(dict(context))

            if self.residual_processor is None:
                enhanced = residual.copy()
            else:
                enhanced = _as_1d_float(
                    "residual_processor output",
                    self.residual_processor(residual, processor_context),
                )
                if enhanced.shape != residual.shape:
                    raise ValueError(
                        "residual_processor must return a 1-D array with the same shape"
                    )

            enhanced_safe, enhanced_ratio, enhanced_max_abs = self._safety_check(enhanced, mic)
            if not enhanced_safe:
                # The optional residual processor produced an unsafe signal.
                # Keep the bounded linear residual rather than discarding the
                # whole adaptive cancellation.
                safety_triggered = 1.0
                safety_stage = 2.0
                enhanced = residual.copy()

        metrics: dict[str, float] = {
            "reference_power": float(np.mean(np.square(x))),
            "microphone_power": float(np.mean(np.square(mic))),
            "linear_residual_power": float(np.mean(np.square(residual))),
            "enhanced_power": float(np.mean(np.square(enhanced))),
            "safety_triggered": safety_triggered,
            "safety_stage": safety_stage,
            "safety_residual_power_ratio": float(residual_ratio),
            "safety_enhanced_power_ratio": float(enhanced_ratio),
            "safety_residual_max_abs": float(residual_max_abs),
            "safety_enhanced_max_abs": float(enhanced_max_abs),
        }
        if clean_target is not None:
            clean = _as_1d_float("clean_target", clean_target)
            if clean.shape != mic.shape:
                raise ValueError("clean_target must have the same shape as microphone")
            input_mse = mse(clean, mic)
            residual_mse = mse(clean, residual)
            enhanced_mse = mse(clean, enhanced)
            metrics.update(
                {
                    "input_mse": input_mse,
                    "linear_residual_mse": residual_mse,
                    "enhanced_mse": enhanced_mse,
                    "linear_improvement_db": improvement_db(input_mse, residual_mse),
                    "enhanced_improvement_db": improvement_db(input_mse, enhanced_mse),
                }
            )

        return EchoCancellationResult(
            echo_estimate=echo_estimate,
            residual=residual,
            enhanced=enhanced,
            error=residual,
            reflection=np.asarray(self.linear_filter.reflection, dtype=np.float64),
            taps=np.asarray(self.linear_filter.taps, dtype=np.float64),
            denominator=np.asarray(self.linear_filter.denominator, dtype=np.float64),
            numerator=np.asarray(self.linear_filter.numerator, dtype=np.float64),
            metrics=metrics,
        )


def make_residual_processor_from_model(model: Any) -> ResidualProcessor:
    """Wrap a user-provided residual model as a residual processor.

    The model may expose ``predict(residual, context)``, be directly callable as
    ``model(residual, context)``, or be callable as ``model(residual)``.  This
    keeps downstream residual processors out of the core while making custom
    callables easy to plug in.
    """

    def processor(residual: np.ndarray, context: Mapping[str, Any]) -> np.ndarray:
        if hasattr(model, "predict"):
            output = model.predict(residual, context)  # noqa: B009 - user-supplied protocol
        else:
            try:
                output = model(residual, context)
            except TypeError:
                output = model(residual)
        return np.asarray(output, dtype=np.float64)

    return processor


__all__ = [
    "EchoCancellationResult",
    "HybridEchoCanceller",
    "ResidualAttenuator",
    "ResidualProcessor",
    "SpectralResidualSuppressor",
    "improvement_db",
    "make_residual_processor_from_model",
    "mse",
    "residual_attenuator",
    "spectral_residual_suppressor",
]
