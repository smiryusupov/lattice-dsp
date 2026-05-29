"""Multichannel autoregressive estimation and block Levinson utilities.

The functions in this module work with vector autoregressive (VAR) models of
order ``p`` written as

``x[n] + A[0] x[n-1] + ... + A[p-1] x[n-p] = e[n]``.

The autocorrelation convention is ``R[k] = E{x[n] x[n-k]^H}`` for
``k >= 0``.  With this convention, the block Yule-Walker equations are

``sum_j A[j] R[i-j] = -R[i]`` for ``i = 1, ..., p``.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np


ArrayLike = np.ndarray | Iterable[float] | Iterable[complex]


@dataclass(frozen=True)
class MultichannelARResult:
    """Result returned by multichannel AR solvers.

    Parameters
    ----------
    coefficients:
        AR coefficient matrices with shape ``(order, channels, channels)``.
        The model convention is ``x[n] + sum_j A[j] @ x[n-j-1] = e[n]``.
    prediction_error:
        Forward prediction-error covariance matrix.
    reflection:
        Forward matrix reflection / partial-correlation coefficients.  For the
        direct dense solver this is empty because reflection coefficients are an
        order-recursive quantity.
    backward_coefficients:
        Backward AR coefficient matrices from the block Levinson recursion.
    backward_error:
        Backward prediction-error covariance matrix.
    method:
        Human-readable solver label.
    backward_reflection:
        Backward matrix reflection coefficients used by causal multichannel
        lattice prediction.  Direct dense solvers leave this empty.
    """

    coefficients: np.ndarray
    prediction_error: np.ndarray
    reflection: np.ndarray
    backward_coefficients: np.ndarray
    backward_error: np.ndarray
    method: str
    backward_reflection: np.ndarray | None = None

    @property
    def order(self) -> int:
        """AR model order."""

        return int(self.coefficients.shape[0])

    @property
    def channels(self) -> int:
        """Number of channels in the vector process."""

        return int(self.coefficients.shape[1])

    @property
    def reflection_spectral_norms(self) -> np.ndarray:
        """Largest singular value of each forward matrix reflection coefficient."""

        if self.reflection.size == 0:
            return np.empty(0, dtype=float)
        return np.asarray([np.linalg.svd(k, compute_uv=False)[0] for k in self.reflection])

    @property
    def backward_reflection_spectral_norms(self) -> np.ndarray:
        """Largest singular value of each backward matrix reflection coefficient."""

        if self.backward_reflection is None or self.backward_reflection.size == 0:
            return np.empty(0, dtype=float)
        return np.asarray([np.linalg.svd(k, compute_uv=False)[0] for k in self.backward_reflection])


def _as_autocorrelation_array(r: ArrayLike) -> np.ndarray:
    arr = np.asarray(r)
    if arr.ndim != 3:
        raise ValueError("autocorrelation must have shape (order + 1, channels, channels)")
    if arr.shape[1] != arr.shape[2]:
        raise ValueError("autocorrelation blocks must be square matrices")
    if arr.shape[0] < 1:
        raise ValueError("at least the zero-lag autocorrelation block is required")
    return arr.astype(np.result_type(arr.dtype, np.complex128), copy=False)


def _validate_order(r: np.ndarray, order: int | None) -> int:
    p = r.shape[0] - 1 if order is None else int(order)
    if p < 0:
        raise ValueError("order must be non-negative")
    if p >= r.shape[0]:
        raise ValueError("order exceeds available autocorrelation lags")
    return p


def _right_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return ``a @ inv(b)`` without forming the inverse explicitly."""

    return np.linalg.solve(b.T, a.T).T


def _lag_block(r: np.ndarray, lag: int) -> np.ndarray:
    """Return R[lag] for positive/negative lags under Hermitian symmetry."""

    return r[lag] if lag >= 0 else r[-lag].conj().T


def multichannel_autocorrelation(
    x: ArrayLike,
    order: int,
    *,
    biased: bool = True,
    demean: bool = True,
) -> np.ndarray:
    """Estimate multichannel autocorrelation matrices.

    Parameters
    ----------
    x:
        Input sequence with shape ``(samples, channels)``.  One-dimensional
        input is accepted and treated as a single channel.
    order:
        Maximum lag to estimate.
    biased:
        If ``True``, divide every lag by ``samples``.  If ``False``, divide lag
        ``k`` by ``samples - k``.
    demean:
        Remove the sample mean from each channel before estimating covariance.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(order + 1, channels, channels)`` using the
        convention ``R[k] = E{x[n] x[n-k]^H}``.
    """

    data = np.asarray(x)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("x must have shape (samples, channels)")
    n, channels = data.shape
    p = int(order)
    if p < 0:
        raise ValueError("order must be non-negative")
    if n <= p:
        raise ValueError("number of samples must be greater than order")

    dtype = np.result_type(data.dtype, np.complex128)
    z = data.astype(dtype, copy=False)
    if demean:
        z = z - np.mean(z, axis=0, keepdims=True)

    r = np.empty((p + 1, channels, channels), dtype=dtype)
    for lag in range(p + 1):
        denom = n if biased else n - lag
        r[lag] = (z[lag:].T @ z[: n - lag].conj()) / denom
    return r


def block_toeplitz_from_autocorrelation(
    r: ArrayLike,
    order: int | None = None,
    *,
    regularization: float = 0.0,
) -> np.ndarray:
    """Build the block Toeplitz matrix for the block Yule-Walker equations.

    The returned matrix has shape ``(order * channels, order * channels)`` and
    contains blocks ``R[i-j]`` with Hermitian symmetry for negative lags.
    """

    rr = _as_autocorrelation_array(r)
    p = _validate_order(rr, order)
    channels = rr.shape[1]
    if p == 0:
        return np.empty((0, 0), dtype=rr.dtype)

    t = np.empty((p * channels, p * channels), dtype=rr.dtype)
    eye = np.eye(channels, dtype=rr.dtype)
    reg = float(regularization)
    for row in range(p):
        for col in range(p):
            lag = (col + 1) - (row + 1)
            block = _lag_block(rr, lag).copy()
            if row == col and reg:
                block = block + reg * eye
            t[row * channels : (row + 1) * channels, col * channels : (col + 1) * channels] = block
    return t


def solve_block_yule_walker_direct(
    r: ArrayLike,
    order: int | None = None,
    *,
    regularization: float = 0.0,
) -> MultichannelARResult:
    """Solve multichannel Yule-Walker equations with a dense block solve.

    This is the most direct baseline for vector AR estimation.  It is useful for
    testing and for small problems, but it ignores the Toeplitz structure except
    when constructing the matrix.
    """

    rr = _as_autocorrelation_array(r)
    p = _validate_order(rr, order)
    channels = rr.shape[1]
    dtype = rr.dtype
    if p == 0:
        err = rr[0] + float(regularization) * np.eye(channels, dtype=dtype)
        empty = np.empty((0, channels, channels), dtype=dtype)
        return MultichannelARResult(empty, err, empty, empty, err, "direct", empty.copy())

    t = block_toeplitz_from_autocorrelation(rr, p, regularization=regularization)
    rhs = np.hstack([-rr[lag] for lag in range(1, p + 1)])

    # Solve A_stack @ T = RHS by transposing the linear system.  This avoids an
    # explicit inverse and keeps the row-wise VAR convention intact.
    coeff_stack_h = np.linalg.solve(t.conj().T, rhs.conj().T)
    coeff_stack = coeff_stack_h.conj().T
    coefficients = np.stack(
        [coeff_stack[:, i * channels : (i + 1) * channels] for i in range(p)],
        axis=0,
    )

    pred_err = rr[0].copy()
    if regularization:
        pred_err = pred_err + float(regularization) * np.eye(channels, dtype=dtype)
    for lag, a_lag in enumerate(coefficients, start=1):
        pred_err = pred_err + a_lag @ rr[lag].conj().T
    pred_err = 0.5 * (pred_err + pred_err.conj().T)
    empty_reflection = np.empty((0, channels, channels), dtype=dtype)
    return MultichannelARResult(
        coefficients=coefficients,
        prediction_error=pred_err,
        reflection=empty_reflection,
        backward_coefficients=np.empty((0, channels, channels), dtype=dtype),
        backward_error=pred_err.copy(),
        method="direct",
        backward_reflection=empty_reflection.copy(),
    )


def block_levinson_durbin(
    r: ArrayLike,
    order: int | None = None,
    *,
    regularization: float = 0.0,
) -> MultichannelARResult:
    """Estimate a vector AR model with the block Levinson-Durbin recursion.

    This implements the forward/backward block recursion often associated with
    multichannel Levinson, Whittle, and Wiggins-Robinson/LWR algorithms.  It
    returns the same forward AR coefficients as the dense block Yule-Walker
    solve for well-conditioned covariance sequences, while also exposing the
    order-recursive matrix reflection coefficients.
    """

    rr = _as_autocorrelation_array(r)
    p = _validate_order(rr, order)
    channels = rr.shape[1]
    dtype = rr.dtype
    eye = np.eye(channels, dtype=dtype)

    if p == 0:
        err0 = rr[0].copy() + float(regularization) * eye
        empty = np.empty((0, channels, channels), dtype=dtype)
        return MultichannelARResult(empty, err0, empty, empty, err0, "block_levinson", empty.copy())

    forward: list[np.ndarray] = []
    backward: list[np.ndarray] = []
    reflections: list[np.ndarray] = []
    backward_reflections: list[np.ndarray] = []
    ef = rr[0].copy() + float(regularization) * eye
    eb = rr[0].copy() + float(regularization) * eye

    for stage in range(1, p + 1):
        delta = rr[stage].copy()
        for lag in range(1, stage):
            delta = delta + forward[lag - 1] @ rr[stage - lag]

        k_forward = -_right_solve(delta, eb)
        k_backward = -_right_solve(delta.conj().T, ef)

        old_forward = list(forward)
        old_backward = list(backward)
        new_forward: list[np.ndarray] = []
        new_backward: list[np.ndarray] = []

        for lag in range(1, stage):
            new_forward.append(old_forward[lag - 1] + k_forward @ old_backward[stage - lag - 1])
            new_backward.append(old_backward[lag - 1] + k_backward @ old_forward[stage - lag - 1])
        new_forward.append(k_forward)
        new_backward.append(k_backward)

        ef_next = ef - k_forward @ eb @ k_forward.conj().T
        eb_next = eb - k_backward @ ef @ k_backward.conj().T
        ef_next = 0.5 * (ef_next + ef_next.conj().T)
        eb_next = 0.5 * (eb_next + eb_next.conj().T)

        forward = new_forward
        backward = new_backward
        ef = ef_next
        eb = eb_next
        reflections.append(k_forward)
        backward_reflections.append(k_backward)

    return MultichannelARResult(
        coefficients=np.stack(forward, axis=0),
        prediction_error=ef,
        reflection=np.stack(reflections, axis=0),
        backward_coefficients=np.stack(backward, axis=0),
        backward_error=eb,
        method="block_levinson",
        backward_reflection=np.stack(backward_reflections, axis=0),
    )


def _as_reflection_stack(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"{name} must have shape (order, channels, channels)")
    if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr.astype(np.result_type(arr.dtype, np.complex128), copy=False))


class MIMOLatticePredictor:
    """Causal online multichannel lattice predictor.

    The predictor implements the vector lattice recursion

    ``f_0[n] = b_0[n] = y[n]``

    ``f_m[n] = f_{m-1}[n] + K_m b_{m-1}[n-1]``

    ``b_m[n] = b_{m-1}[n-1] + L_m f_{m-1}[n]``

    where ``K_m`` and ``L_m`` are forward and backward matrix reflection
    coefficients.  Calling :meth:`predict` uses only the stored backward-error
    state from previous samples and therefore gives ``y_hat[n]`` before
    ``y[n]`` is observed.  Calling :meth:`update` then consumes the observed
    vector and returns the forward prediction error.

    The class is a runtime filter/predictor.  Estimating the reflection
    matrices with :func:`block_levinson_durbin` is a separate batch operation.
    """

    def __init__(self, forward_reflection: ArrayLike, backward_reflection: ArrayLike) -> None:
        kf = _as_reflection_stack(forward_reflection, "forward_reflection")
        kb = _as_reflection_stack(backward_reflection, "backward_reflection")
        if kb.shape != kf.shape:
            raise ValueError("forward_reflection and backward_reflection must have the same shape")
        self.forward_reflection = kf
        self.backward_reflection = kb
        self.order = int(kf.shape[0])
        self.channels = int(kf.shape[1])
        self._dtype = np.result_type(kf.dtype, kb.dtype, np.complex128)
        self.reset()

    @classmethod
    def from_levinson(cls, result: MultichannelARResult) -> MIMOLatticePredictor:
        """Create a predictor from :func:`block_levinson_durbin` output."""

        if result.method != "block_levinson":
            raise ValueError(
                "MIMOLatticePredictor.from_levinson expects block_levinson_durbin output"
            )
        if result.reflection.shape[0] != result.order:
            raise ValueError(
                "result does not contain forward reflection coefficients for every order"
            )
        if (
            result.backward_reflection is None
            or result.backward_reflection.shape != result.reflection.shape
        ):
            raise ValueError("result does not contain matching backward reflection coefficients")
        return cls(result.reflection, result.backward_reflection)

    def reset(self) -> None:
        """Reset the internal backward-error state to zero history."""

        self._backward_state = [
            np.zeros(self.channels, dtype=self._dtype) for _ in range(self.order + 1)
        ]

    def _coerce_sample(self, sample: ArrayLike) -> np.ndarray:
        vec = np.asarray(sample, dtype=self._dtype)
        if vec.ndim != 1 or vec.shape[0] != self.channels:
            raise ValueError(f"sample must have shape ({self.channels},)")
        if not np.all(np.isfinite(vec.real)) or not np.all(np.isfinite(vec.imag)):
            raise ValueError("sample must contain only finite values")
        return vec

    def _forward_error(self, sample: np.ndarray, *, update_state: bool) -> np.ndarray:
        previous_backward = self._backward_state
        new_backward: list[np.ndarray] = [sample.copy()] + [
            np.zeros(self.channels, dtype=self._dtype) for _ in range(self.order)
        ]
        f = sample.copy()
        for stage in range(self.order):
            old_f = f.copy()
            f = old_f + self.forward_reflection[stage] @ previous_backward[stage]
            b = previous_backward[stage] + self.backward_reflection[stage] @ old_f
            new_backward[stage + 1] = b
        if update_state:
            self._backward_state = new_backward
        return f

    def predict(self) -> np.ndarray:
        """Return the one-step prediction using only previous samples."""

        zero = np.zeros(self.channels, dtype=self._dtype)
        return -self._forward_error(zero, update_state=False)

    def update(self, sample: ArrayLike) -> np.ndarray:
        """Consume ``sample`` and return its forward prediction error."""

        return self._forward_error(self._coerce_sample(sample), update_state=True)

    def step(self, sample: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(prediction, error)`` for one observed vector and update state."""

        prediction = self.predict()
        error = self.update(sample)
        return prediction, error

    def process(self, x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Process a sequence and return ``(prediction, error)`` arrays.

        ``x`` must have shape ``(samples, channels)``.  The first samples use
        zero-history initial conditions, so the output has the same number of
        samples as the input.
        """

        data = np.asarray(x, dtype=self._dtype)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[1] != self.channels:
            raise ValueError(f"x must have shape (samples, {self.channels})")
        if not np.all(np.isfinite(data.real)) or not np.all(np.isfinite(data.imag)):
            raise ValueError("x must contain only finite values")
        prediction = np.empty_like(data, dtype=self._dtype)
        error = np.empty_like(data, dtype=self._dtype)
        for n, sample in enumerate(data):
            prediction[n], error[n] = self.step(sample)
        return prediction, error


def causal_mimo_lattice_predict(
    x: ArrayLike,
    forward_reflection: ArrayLike,
    backward_reflection: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch convenience wrapper for :class:`MIMOLatticePredictor`.

    Returns ``(prediction, error)`` with the same sample count as ``x``.  The
    computation is causal: each row of ``prediction`` is formed before the
    corresponding row of ``x`` is used to update the lattice state.
    """

    return MIMOLatticePredictor(forward_reflection, backward_reflection).process(x)


def multichannel_prediction_error(x: ArrayLike, coefficients: ArrayLike) -> np.ndarray:
    """Return residuals of a multichannel AR model on a sequence.

    The output has shape ``(samples - order, channels)`` and follows
    ``e[n] = x[n] + sum_j A[j] @ x[n-j-1]``.
    """

    data = np.asarray(x)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("x must have shape (samples, channels)")

    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 3 or coeffs.shape[1] != coeffs.shape[2]:
        raise ValueError("coefficients must have shape (order, channels, channels)")
    p, channels, _ = coeffs.shape
    if data.shape[1] != channels:
        raise ValueError("x and coefficients disagree on channel count")
    if data.shape[0] <= p:
        raise ValueError("number of samples must be greater than AR order")

    dtype = np.result_type(data.dtype, coeffs.dtype, np.complex128)
    z = data.astype(dtype, copy=False)
    a = coeffs.astype(dtype, copy=False)
    residual = np.empty((z.shape[0] - p, channels), dtype=dtype)
    for out_idx, n in enumerate(range(p, z.shape[0])):
        e = z[n].copy()
        for lag in range(1, p + 1):
            e = e + a[lag - 1] @ z[n - lag]
        residual[out_idx] = e
    return residual


def matrix_ar_frequency_response(coefficients: ArrayLike, w: ArrayLike) -> np.ndarray:
    """Evaluate ``H(e^jw) = inv(I + sum_j A[j] e^{-jw(j+1)})``.

    Parameters
    ----------
    coefficients:
        AR coefficient matrices with shape ``(order, channels, channels)``.
    w:
        Frequencies in radians/sample.

    Returns
    -------
    numpy.ndarray
        Frequency response with shape ``(len(w), channels, channels)``.
    """

    a = np.asarray(coefficients)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError("coefficients must have shape (order, channels, channels)")
    freqs = np.asarray(w, dtype=float)
    if freqs.ndim == 0:
        freqs = freqs[None]
    if freqs.ndim != 1:
        raise ValueError("w must be a scalar or one-dimensional array")

    p, channels, _ = a.shape
    dtype = np.result_type(a.dtype, np.complex128)
    eye = np.eye(channels, dtype=dtype)
    h = np.empty((freqs.size, channels, channels), dtype=dtype)
    for i, omega in enumerate(freqs):
        denom = eye.copy()
        for lag in range(1, p + 1):
            denom = denom + a[lag - 1].astype(dtype, copy=False) * np.exp(-1j * omega * lag)
        h[i] = np.linalg.inv(denom)
    return h


def companion_spectral_radius(coefficients: ArrayLike) -> float:
    """Return the spectral radius of the VAR companion matrix.

    Values below one indicate a stable causal VAR model under the package's
    coefficient convention.
    """

    a = np.asarray(coefficients)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError("coefficients must have shape (order, channels, channels)")
    p, channels, _ = a.shape
    if p == 0:
        return 0.0

    dtype = np.result_type(a.dtype, np.complex128)
    comp = np.zeros((p * channels, p * channels), dtype=dtype)
    # x[n] = -A1 x[n-1] - ... - Ap x[n-p] + e[n]
    comp[:channels, :] = np.hstack([-a[lag].astype(dtype, copy=False) for lag in range(p)])
    if p > 1:
        comp[channels:, :-channels] = np.eye((p - 1) * channels, dtype=dtype)
    eig = np.linalg.eigvals(comp)
    return float(np.max(np.abs(eig)))


__all__ = [
    "MIMOLatticePredictor",
    "MultichannelARResult",
    "block_levinson_durbin",
    "causal_mimo_lattice_predict",
    "block_toeplitz_from_autocorrelation",
    "companion_spectral_radius",
    "matrix_ar_frequency_response",
    "multichannel_autocorrelation",
    "multichannel_prediction_error",
    "solve_block_yule_walker_direct",
]
