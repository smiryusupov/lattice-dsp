"""Matrix-valued lattice/all-pass utilities.

This module is intentionally a foundation layer, not a full wireless
precoding framework.  It provides the reusable pieces needed to work with
complex matrix reflection coefficients whose singular values are bounded by
one.  Those contractive matrices parameterize lossless/lattice stages and are
useful for compact MIMO/time-domain DSP representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import numpy as np

try:  # pragma: no cover - exercised through the compiled extension when built.
    from ._core import matrix_lattice_frequency_response as _core_matrix_lattice_frequency_response
except Exception:  # pragma: no cover
    _core_matrix_lattice_frequency_response = None


def _as_square_complex_matrix(
    value: np.ndarray | Sequence[Sequence[complex]], name: str
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D complex matrix")
    if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr)


def matrix_spectral_norm(matrix: np.ndarray | Sequence[Sequence[complex]]) -> float:
    """Return the largest singular value of a complex matrix."""

    mat = _as_square_complex_matrix(matrix, "matrix")
    return float(np.linalg.svd(mat, compute_uv=False)[0])


def is_matrix_reflection_stable(
    matrix: np.ndarray | Sequence[Sequence[complex]], *, margin: float = 1e-9
) -> bool:
    """Return ``True`` when ``||K||_2 < 1 - margin``.

    For matrix lattice/all-pass stages, the scalar ``|k| < 1`` stability test
    generalizes to a strict contraction: all singular values of ``K`` must be
    below one.
    """

    if margin < 0.0 or margin >= 1.0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and in [0, 1)")
    return matrix_spectral_norm(matrix) < 1.0 - margin


def project_matrix_reflection(
    matrix: np.ndarray | Sequence[Sequence[complex]], *, margin: float = 1e-6
) -> np.ndarray:
    """Project a square complex matrix to a strict spectral-norm contraction."""

    if margin < 0.0 or margin >= 1.0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and in [0, 1)")
    mat = _as_square_complex_matrix(matrix, "matrix")
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    limit = np.nextafter(1.0 - margin, 0.0)
    s = np.minimum(s, limit)
    return np.ascontiguousarray((u * s) @ vh)


def contractive_matrix_from_raw(
    raw: np.ndarray | Sequence[Sequence[complex]], *, margin: float = 1e-6
) -> np.ndarray:
    """Map an unconstrained complex matrix to a stable matrix reflection.

    This is the matrix analogue of using ``tanh`` to bound scalar reflection
    coefficients.  The singular vectors are retained and singular values are
    mapped through ``(1 - margin) * tanh(s)``.
    """

    if margin < 0.0 or margin >= 1.0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and in [0, 1)")
    mat = _as_square_complex_matrix(raw, "raw")
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    s = (1.0 - margin) * np.tanh(s)
    return np.ascontiguousarray((u * s) @ vh)


def unitary_polar_factor(matrix: np.ndarray | Sequence[Sequence[complex]]) -> np.ndarray:
    """Return the unitary polar factor of a square complex matrix."""

    mat = _as_square_complex_matrix(matrix, "matrix")
    u, _, vh = np.linalg.svd(mat, full_matrices=False)
    return np.ascontiguousarray(u @ vh)


def psd_matrix_sqrt(
    matrix: np.ndarray | Sequence[Sequence[complex]], *, clip_negative: bool = True
) -> np.ndarray:
    """Hermitian positive-semidefinite matrix square root."""

    mat = _as_square_complex_matrix(matrix, "matrix")
    hermitian = 0.5 * (mat + mat.conj().T)
    eigvals, eigvecs = np.linalg.eigh(hermitian)
    if clip_negative:
        eigvals = np.maximum(eigvals, 0.0)
    elif np.any(eigvals < -1e-12):
        raise ValueError("matrix has negative eigenvalues and is not PSD")
    root = (eigvecs * np.sqrt(eigvals)) @ eigvecs.conj().T
    return np.ascontiguousarray(root)


def matrix_lattice_stage_blocks(
    reflection: np.ndarray | Sequence[Sequence[complex]],
    *,
    margin: float = 1e-9,
    project: bool = False,
) -> np.ndarray:
    """Return the four block matrices ``[T11, T12, T21, T22]`` for one stage.

    For a contractive matrix reflection ``K``, the lossless stage is

    ``[[K, sqrt(I - K K^H)], [sqrt(I - K^H K), -K^H]]``.
    """

    k = _as_square_complex_matrix(reflection, "reflection")
    if project:
        k = project_matrix_reflection(k, margin=margin)
    if not is_matrix_reflection_stable(k, margin=margin):
        raise ValueError("reflection matrix must satisfy spectral_norm(K) < 1 - margin")
    eye = np.eye(k.shape[0], dtype=np.complex128)
    t11 = k
    t12 = psd_matrix_sqrt(eye - k @ k.conj().T)
    t21 = psd_matrix_sqrt(eye - k.conj().T @ k)
    t22 = -k.conj().T
    return np.ascontiguousarray(np.stack([t11, t12, t21, t22], axis=0))


def _frequency_response_numpy(
    stage_blocks: np.ndarray, residue: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    n_freq = omega.size
    m = residue.shape[0]
    out = np.empty((n_freq, m, m), dtype=np.complex128)
    eye = np.eye(m, dtype=np.complex128)
    for i, w in enumerate(omega):
        z = np.exp(-1j * float(w))
        g = residue.copy()
        # stage_blocks[0] is the outermost stage; apply inner stages first.
        for t11, t12, t21, t22 in stage_blocks[::-1]:
            zg = z * g
            x = np.linalg.solve(eye - t22 @ zg, t21)
            g = t11 + t12 @ zg @ x
        out[i] = g
    return out


@dataclass(frozen=True)
class MatrixLatticeAllPass:
    """Complex matrix-valued lattice/all-pass filter representation.

    Parameters
    ----------
    reflections:
        Sequence of square complex matrix reflection coefficients.  Each matrix
        must be a strict contraction, i.e. ``||K_i||_2 < 1``.
    residue:
        Final unitary matrix.  If omitted, the identity matrix is used.
    margin:
        Stability margin for reflection matrices.
    project:
        If true, reflection matrices are projected to the stable set and the
        residue is replaced by its unitary polar factor.
    """

    reflections: tuple[np.ndarray, ...]
    residue: np.ndarray
    margin: float = 1e-9
    stage_blocks: np.ndarray | None = None

    def __init__(
        self,
        reflections: Iterable[np.ndarray | Sequence[Sequence[complex]]],
        residue: np.ndarray | Sequence[Sequence[complex]] | None = None,
        *,
        margin: float = 1e-9,
        project: bool = False,
    ) -> None:
        if margin < 0.0 or margin >= 1.0 or not np.isfinite(margin):
            raise ValueError("margin must be finite and in [0, 1)")
        mats = tuple(_as_square_complex_matrix(k, "reflection") for k in reflections)
        if not mats:
            if residue is None:
                raise ValueError("at least one reflection or an explicit residue is required")
            res = _as_square_complex_matrix(residue, "residue")
            dim = res.shape[0]
        else:
            dim = mats[0].shape[0]
            if any(k.shape != (dim, dim) for k in mats):
                raise ValueError("all reflection matrices must have the same square shape")
            if project:
                mats = tuple(project_matrix_reflection(k, margin=margin) for k in mats)
            for k in mats:
                if not is_matrix_reflection_stable(k, margin=margin):
                    raise ValueError(
                        "each reflection matrix must satisfy spectral_norm(K) < 1 - margin"
                    )
            res = (
                np.eye(dim, dtype=np.complex128)
                if residue is None
                else _as_square_complex_matrix(residue, "residue")
            )
        if res.shape != (dim, dim):
            raise ValueError("residue shape must match reflection matrices")
        if project:
            res = unitary_polar_factor(res)
        blocks = (
            np.ascontiguousarray(
                np.stack([matrix_lattice_stage_blocks(k, margin=margin) for k in mats], axis=0)
            )
            if mats
            else np.empty((0, 4, dim, dim), dtype=np.complex128)
        )
        object.__setattr__(self, "reflections", tuple(np.ascontiguousarray(k) for k in mats))
        object.__setattr__(self, "residue", np.ascontiguousarray(res))
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "stage_blocks", blocks)

    @property
    def order(self) -> int:
        return len(self.reflections)

    @property
    def dimension(self) -> int:
        return int(self.residue.shape[0])

    def parameter_count(self, *, real_scalars: bool = True, include_residue: bool = True) -> int:
        """Return coefficient count for reporting/feedback estimates."""

        complex_count = self.order * self.dimension * self.dimension
        if include_residue:
            complex_count += self.dimension * self.dimension
        return int(2 * complex_count if real_scalars else complex_count)

    def max_reflection_singular_value(self) -> float:
        if not self.reflections:
            return 0.0
        return float(max(matrix_spectral_norm(k) for k in self.reflections))

    def frequency_response(
        self, omega: np.ndarray | Sequence[float], *, n_threads: int = 0
    ) -> np.ndarray:
        """Evaluate the matrix all-pass response at radian frequencies."""

        w = np.ascontiguousarray(np.asarray(omega, dtype=np.float64).reshape(-1))
        if self.stage_blocks is None:
            raise RuntimeError("internal stage blocks are not initialized")
        if _core_matrix_lattice_frequency_response is not None:
            return _core_matrix_lattice_frequency_response(
                self.stage_blocks, self.residue, w, n_threads
            )
        return _frequency_response_numpy(self.stage_blocks, self.residue, w)

    def unitarity_error(self, omega: np.ndarray | Sequence[float]) -> float:
        """Maximum ``||G(w)^H G(w) - I||_F`` over supplied frequencies."""

        response = self.frequency_response(omega)
        eye = np.eye(self.dimension, dtype=np.complex128)
        err = 0.0
        for g in response:
            err = max(err, float(np.linalg.norm(g.conj().T @ g - eye, ord="fro")))
        return err

    def to_online_filter(self) -> OnlineMatrixLatticeAllPass:
        """Return a causal streaming runtime for this all-pass lattice."""

        return OnlineMatrixLatticeAllPass(self)

    def impulse_response(self, n_samples: int) -> np.ndarray:
        """Return the causal matrix impulse response of the online realization.

        The returned array has shape ``(n_samples, outputs, inputs)``.  Entry
        ``h[t, i, j]`` is the output in channel ``i`` at delay ``t`` after a
        unit impulse in input channel ``j``.  The response is truncated to the
        requested length; matrix-lattice all-pass responses are generally IIR,
        so longer lengths expose more of the decaying tail.
        """

        if int(n_samples) != n_samples or int(n_samples) <= 0:
            raise ValueError("n_samples must be a positive integer")
        n = int(n_samples)
        h = np.empty((n, self.dimension, self.dimension), dtype=np.complex128)
        zero = np.zeros(self.dimension, dtype=np.complex128)
        for input_channel in range(self.dimension):
            runtime = self.to_online_filter()
            impulse = zero.copy()
            impulse[input_channel] = 1.0
            h[0, :, input_channel] = runtime.process_sample(impulse)
            for sample_idx in range(1, n):
                h[sample_idx, :, input_channel] = runtime.process_sample(zero)
        return np.ascontiguousarray(h)


class _OnlineMatrixLatticeStage:
    """One causal stage used internally by OnlineMatrixLatticeAllPass."""

    def __init__(
        self, block: np.ndarray, inner: _OnlineMatrixLatticeStage | _OnlineMatrixLatticeResidue
    ) -> None:
        self.t11 = np.ascontiguousarray(block[0])
        self.t12 = np.ascontiguousarray(block[1])
        self.t21 = np.ascontiguousarray(block[2])
        self.t22 = np.ascontiguousarray(block[3])
        self.inner = inner
        self.dimension = int(self.t11.shape[0])
        self._delay = np.zeros(self.dimension, dtype=np.complex128)

    def reset(self) -> None:
        self._delay.fill(0.0)
        self.inner.reset()

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        delayed_inner_output = self._delay
        output = self.t11 @ sample + self.t12 @ delayed_inner_output
        inner_input = self.t21 @ sample + self.t22 @ delayed_inner_output
        self._delay = np.ascontiguousarray(self.inner.process_sample(inner_input))
        return np.ascontiguousarray(output)


class _OnlineMatrixLatticeResidue:
    """Terminal unitary matrix used internally by OnlineMatrixLatticeAllPass."""

    def __init__(self, residue: np.ndarray) -> None:
        self.residue = np.ascontiguousarray(residue)
        self.dimension = int(residue.shape[0])

    def reset(self) -> None:
        return None

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.residue @ sample)


class OnlineMatrixLatticeAllPass:
    """Causal streaming realization of :class:`MatrixLatticeAllPass`.

    The existing :class:`MatrixLatticeAllPass` object defines the transfer
    function.  This runtime wrapper realizes the same Schur/lattice cascade in
    time, with one vector delay per section.  For an input vector ``u[n]`` the
    output at time ``n`` depends only on ``u[n]`` and previous internal states;
    future samples are never inspected.

    The finite prefix energy of a processed block can differ from the input
    energy because some energy remains in the delay states.  Over a long stream,
    or after appending enough zero samples to let the tail decay, the
    input-output energy agrees up to numerical precision for a lossless/all-pass
    lattice.
    """

    def __init__(self, lattice: MatrixLatticeAllPass) -> None:
        if not isinstance(lattice, MatrixLatticeAllPass):
            raise TypeError("lattice must be a MatrixLatticeAllPass instance")
        self.lattice = lattice
        self.dimension = lattice.dimension
        self.order = lattice.order
        inner: _OnlineMatrixLatticeStage | _OnlineMatrixLatticeResidue = (
            _OnlineMatrixLatticeResidue(lattice.residue)
        )
        # stage_blocks[0] is the outermost section in the frequency-response
        # recursion, so build the runtime cascade from the innermost section out.
        for block in lattice.stage_blocks[::-1]:
            inner = _OnlineMatrixLatticeStage(block, inner)
        self._outer = inner

    def reset(self) -> None:
        """Reset all section delay states to zero."""

        self._outer.reset()

    def _coerce_sample(self, sample: np.ndarray | Sequence[complex]) -> np.ndarray:
        vec = np.asarray(sample, dtype=np.complex128)
        if vec.ndim != 1 or vec.shape[0] != self.dimension:
            raise ValueError(f"sample must have shape ({self.dimension},)")
        if not np.all(np.isfinite(vec.real)) or not np.all(np.isfinite(vec.imag)):
            raise ValueError("sample must contain only finite values")
        return np.ascontiguousarray(vec)

    def process_sample(self, sample: np.ndarray | Sequence[complex]) -> np.ndarray:
        """Process one vector sample causally and update internal states."""

        return self._outer.process_sample(self._coerce_sample(sample))

    def process(self, x: np.ndarray | Sequence[Sequence[complex]], *, drain: int = 0) -> np.ndarray:
        """Process a sequence of vector samples.

        Parameters
        ----------
        x:
            Array with shape ``(samples, channels)``.  One-dimensional input is
            accepted only for a one-channel lattice.
        drain:
            Number of trailing zero-vector samples to append to expose the
            decaying all-pass tail.  The returned array has ``samples + drain``
            rows.
        """

        if drain < 0:
            raise ValueError("drain must be nonnegative")
        data = np.asarray(x, dtype=np.complex128)
        if data.ndim == 1:
            if self.dimension != 1:
                raise ValueError(f"x must have shape (samples, {self.dimension})")
            data = data[:, None]
        if data.ndim != 2 or data.shape[1] != self.dimension:
            raise ValueError(f"x must have shape (samples, {self.dimension})")
        if not np.all(np.isfinite(data.real)) or not np.all(np.isfinite(data.imag)):
            raise ValueError("x must contain only finite values")
        out = np.empty((data.shape[0] + int(drain), self.dimension), dtype=np.complex128)
        for idx, sample in enumerate(data):
            out[idx] = self.process_sample(sample)
        zero = np.zeros(self.dimension, dtype=np.complex128)
        for tail_idx in range(int(drain)):
            out[data.shape[0] + tail_idx] = self.process_sample(zero)
        return out


def online_matrix_lattice_allpass_process(
    x: np.ndarray | Sequence[Sequence[complex]],
    reflections: Iterable[np.ndarray | Sequence[Sequence[complex]]],
    residue: np.ndarray | Sequence[Sequence[complex]] | None = None,
    *,
    margin: float = 1e-9,
    project: bool = False,
    drain: int = 0,
) -> np.ndarray:
    """Convenience wrapper for causal matrix-lattice all-pass processing."""

    lattice = MatrixLatticeAllPass(reflections, residue=residue, margin=margin, project=project)
    return OnlineMatrixLatticeAllPass(lattice).process(x, drain=drain)


def matrix_lattice_impulse_response_convolution(
    x: np.ndarray | Sequence[Sequence[complex]],
    impulse_response: np.ndarray | Sequence[Sequence[Sequence[complex]]],
    *,
    drain: int = 0,
) -> np.ndarray:
    """Apply a truncated causal MIMO impulse response in the time domain.

    Parameters
    ----------
    x:
        Input array with shape ``(samples, inputs)``.
    impulse_response:
        Matrix impulse response with shape ``(taps, outputs, inputs)``.
    drain:
        Number of trailing zero-input samples to append to the returned output.

    Notes
    -----
    This helper is deliberately time-domain.  It is useful for validating the
    causal streaming runtime against a finite impulse-response truncation and
    for building finite-block adjoint diagnostics without FFT-domain circular
    convolution.
    """

    data = np.asarray(x, dtype=np.complex128)
    h = np.asarray(impulse_response, dtype=np.complex128)
    if data.ndim == 1:
        data = data[:, None]
    if h.ndim != 3:
        raise ValueError("impulse_response must have shape (taps, outputs, inputs)")
    if data.ndim != 2 or data.shape[1] != h.shape[2]:
        raise ValueError(f"x must have shape (samples, {h.shape[2]})")
    if int(drain) != drain or int(drain) < 0:
        raise ValueError("drain must be a nonnegative integer")
    if not np.all(np.isfinite(data.real)) or not np.all(np.isfinite(data.imag)):
        raise ValueError("x must contain only finite values")
    if not np.all(np.isfinite(h.real)) or not np.all(np.isfinite(h.imag)):
        raise ValueError("impulse_response must contain only finite values")

    n_out = data.shape[0] + int(drain)
    out = np.zeros((n_out, h.shape[1]), dtype=np.complex128)
    for delay in range(h.shape[0]):
        limit = min(data.shape[0], n_out - delay)
        if limit <= 0:
            break
        out[delay : delay + limit] += data[:limit] @ h[delay].T
    return np.ascontiguousarray(out)


def matrix_lattice_finite_adjoint(
    y: np.ndarray | Sequence[Sequence[complex]],
    impulse_response: np.ndarray | Sequence[Sequence[Sequence[complex]]],
    *,
    output_length: int | None = None,
) -> np.ndarray:
    """Apply the finite-block adjoint of a truncated causal MIMO response.

    If ``h[k]`` denotes a causal matrix impulse response, forward convolution is

    ``y[n] = sum_k h[k] x[n-k]``.

    This helper applies the corresponding finite-block adjoint

    ``x_adj[n] = sum_k h[k]^H y[n+k]``

    using only time-domain sums.  The operation is generally **noncausal** as a
    synthesis/inverse step because it depends on future output samples.  It is
    intended for finite-record diagnostics such as paraunitary reconstruction
    tests, where the whole transformed record is available.
    """

    data = np.asarray(y, dtype=np.complex128)
    h = np.asarray(impulse_response, dtype=np.complex128)
    if data.ndim == 1:
        data = data[:, None]
    if h.ndim != 3:
        raise ValueError("impulse_response must have shape (taps, outputs, inputs)")
    if data.ndim != 2 or data.shape[1] != h.shape[1]:
        raise ValueError(f"y must have shape (samples, {h.shape[1]})")
    if output_length is None:
        out_len = data.shape[0]
    elif int(output_length) == output_length and int(output_length) >= 0:
        out_len = int(output_length)
    else:
        raise ValueError("output_length must be a nonnegative integer or None")
    if not np.all(np.isfinite(data.real)) or not np.all(np.isfinite(data.imag)):
        raise ValueError("y must contain only finite values")
    if not np.all(np.isfinite(h.real)) or not np.all(np.isfinite(h.imag)):
        raise ValueError("impulse_response must contain only finite values")

    out = np.zeros((out_len, h.shape[2]), dtype=np.complex128)
    for delay in range(h.shape[0]):
        limit = min(out_len, data.shape[0] - delay)
        if limit <= 0:
            break
        out[:limit] += data[delay : delay + limit] @ h[delay].conj()
    return np.ascontiguousarray(out)


def mimo_state_space_frequency_response(
    A: np.ndarray | Sequence[Sequence[float]],
    B: np.ndarray | Sequence[Sequence[float]],
    C: np.ndarray | Sequence[Sequence[float]],
    D: np.ndarray | Sequence[Sequence[float]],
    omega: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Evaluate a discrete-time MIMO state-space frequency response.

    The convention matches :func:`lattice_dsp.mimo_state_space_markov_response`:

    ``H(z) = D + C z^-1 (I - A z^-1)^-1 B``.

    Parameters are converted to dense NumPy arrays.  The returned array has
    shape ``(n_frequency, outputs, inputs)``.
    """

    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    c = np.asarray(C, dtype=np.float64)
    d = np.asarray(D, dtype=np.float64)
    w = np.asarray(omega, dtype=np.float64).reshape(-1)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be a square matrix")
    n_state = a.shape[0]
    if b.ndim != 2 or b.shape[0] != n_state:
        raise ValueError("B must have shape (state, inputs)")
    if c.ndim != 2 or c.shape[1] != n_state:
        raise ValueError("C must have shape (outputs, state)")
    if d.ndim != 2 or d.shape != (c.shape[0], b.shape[1]):
        raise ValueError("D must have shape (outputs, inputs)")
    if not all(np.all(np.isfinite(x)) for x in (a, b, c, d, w)):
        raise ValueError("state-space matrices and frequencies must be finite")
    response = np.empty((w.size, d.shape[0], d.shape[1]), dtype=np.complex128)
    eye = np.eye(n_state, dtype=np.complex128)
    ac = a.astype(np.complex128, copy=False)
    bc = b.astype(np.complex128, copy=False)
    cc = c.astype(np.complex128, copy=False)
    dc = d.astype(np.complex128, copy=False)
    for i, wi in enumerate(w):
        zinv = np.exp(-1j * float(wi))
        if n_state == 0:
            response[i] = dc
        else:
            response[i] = dc + cc @ (zinv * np.linalg.solve(eye - zinv * ac, bc))
    return response


def _as_square_response_stack(
    value: np.ndarray | Sequence[Sequence[Sequence[complex]]], name: str
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"{name} must have shape (frequency, channels, channels)")
    if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr)


def fit_static_matrix_gains(
    target_response: np.ndarray | Sequence[Sequence[Sequence[complex]]],
    lattice_response: np.ndarray | Sequence[Sequence[Sequence[complex]]],
    *,
    mode: str = "both",
    n_iter: int = 20,
    regularization: float = 1e-12,
) -> dict[str, object]:
    """Fit static matrix gains around a matrix-lattice response.

    The model is

    ``target_response[w] ~= left_gain @ lattice_response[w] @ right_gain``.

    ``mode='left'`` fixes ``right_gain = I``; ``mode='right'`` fixes
    ``left_gain = I``; ``mode='both'`` alternates least-squares updates for both
    factors.  This is a diagnostic used to separate all-pass/lattice mismatch
    from static nonunitary gain mismatch.  It is not a dynamic realization
    solver.
    """

    target = _as_square_response_stack(target_response, "target_response")
    lattice = _as_square_response_stack(lattice_response, "lattice_response")
    if target.shape != lattice.shape:
        raise ValueError("target_response and lattice_response must have the same shape")
    if mode not in {"left", "right", "both"}:
        raise ValueError("mode must be 'left', 'right', or 'both'")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if regularization < 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and nonnegative")

    m = target.shape[1]
    eye = np.eye(m, dtype=np.complex128)

    def solve_left(right: np.ndarray) -> np.ndarray:
        s1 = np.zeros((m, m), dtype=np.complex128)
        s2 = regularization * eye.copy()
        for h, u in zip(target, lattice, strict=True):
            x = u @ right
            s1 += h @ x.conj().T
            s2 += x @ x.conj().T
        return np.ascontiguousarray(s1 @ np.linalg.pinv(s2))

    def solve_right(left: np.ndarray) -> np.ndarray:
        s1 = regularization * eye.copy()
        s2 = np.zeros((m, m), dtype=np.complex128)
        for h, u in zip(target, lattice, strict=True):
            y = left @ u
            s1 += y.conj().T @ y
            s2 += y.conj().T @ h
        return np.ascontiguousarray(np.linalg.pinv(s1) @ s2)

    def response_for(left_gain: np.ndarray, right_gain: np.ndarray) -> np.ndarray:
        return np.asarray([left_gain @ u @ right_gain for u in lattice], dtype=np.complex128)

    left = eye.copy()
    right = eye.copy()
    if mode == "left":
        left = solve_left(right)
    elif mode == "right":
        right = solve_right(left)
    else:
        candidate_pairs: list[tuple[float, np.ndarray, np.ndarray]] = []

        def record(left_gain: np.ndarray, right_gain: np.ndarray) -> None:
            candidate_pairs.append(
                (
                    _relative_frobenius_error(target, response_for(left_gain, right_gain)),
                    left_gain.copy(),
                    right_gain.copy(),
                )
            )

        record(eye, eye)
        record(solve_left(eye), eye)
        record(eye, solve_right(eye))

        # A right-only least-squares initialization converges much more
        # reliably than starting the bilinear ALS from two identities when the
        # true target has a substantial static right rotation.
        right = solve_right(eye)
        left = solve_left(right)
        for _ in range(int(n_iter)):
            right = solve_right(left)
            # ``left @ U @ right`` has a scalar ambiguity.  Normalize the two
            # factors to comparable Frobenius norms without changing the
            # product.
            left_norm = np.linalg.norm(left)
            right_norm = np.linalg.norm(right)
            if left_norm > 0.0 and right_norm > 0.0:
                alpha = float(np.sqrt(right_norm / left_norm))
                left *= alpha
                right /= alpha
            record(left, right)
            left = solve_left(right)
            record(left, right)

        _, left, right = min(candidate_pairs, key=lambda item: item[0])

    compensated = response_for(left, right)
    raw_relative_error = _relative_frobenius_error(target, lattice)
    compensated_relative_error = _relative_frobenius_error(target, compensated)
    return {
        "mode": mode,
        "left_gain": np.ascontiguousarray(left),
        "right_gain": np.ascontiguousarray(right),
        "compensated_response": np.ascontiguousarray(compensated),
        "raw_relative_error": raw_relative_error,
        "compensated_relative_error": compensated_relative_error,
        "left_gain_condition": float(np.linalg.cond(left)),
        "right_gain_condition": float(np.linalg.cond(right)),
        "iterations": int(n_iter if mode == "both" else 1),
    }


def polar_factor_response(
    response: np.ndarray | Sequence[Sequence[Sequence[complex]]],
) -> np.ndarray:
    """Return the unitary polar factor of each square response matrix."""

    arr = np.asarray(response, dtype=np.complex128)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError("response must have shape (frequency, channels, channels)")
    out = np.empty_like(arr)
    for i, h in enumerate(arr):
        u, _, vh = np.linalg.svd(h, full_matrices=False)
        out[i] = u @ vh
    return np.ascontiguousarray(out)


def _relative_frobenius_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - estimate) / max(np.linalg.norm(reference), 1e-30))


def _state_space_markov_response_numpy(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, n_samples: int
) -> np.ndarray:
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    out = np.empty((n_samples, D.shape[0], D.shape[1]), dtype=np.float64)
    out[0] = D
    power = np.eye(A.shape[0], dtype=np.float64)
    for k in range(1, n_samples):
        out[k] = C @ power @ B
        power = power @ A
    return out


def matrix_lattice_scaffold_from_markov(
    markov: np.ndarray | Sequence[Sequence[Sequence[float]]],
    *,
    order: int,
    gain: float = 0.55,
    margin: float = 1e-6,
) -> MatrixLatticeAllPass:
    """Build a stable matrix-lattice all-pass scaffold from MIMO Markov matrices.

    This is an initializer/diagnostic, not an exact realization theorem.  Early
    Markov matrices supply coupling directions; each is projected to a strict
    matrix-reflection contraction.  The residue is the unitary polar factor of
    the direct term ``M_0``.
    """

    arr = np.asarray(markov, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError("markov must have shape (samples, channels, channels)")
    if order < 0:
        raise ValueError("order must be nonnegative")
    if gain < 0.0 or not np.isfinite(gain):
        raise ValueError("gain must be a finite nonnegative value")
    if margin < 0.0 or margin >= 1.0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and in [0, 1)")
    channels = arr.shape[1]
    reflections: list[np.ndarray] = []
    for k in range(order):
        idx = min(k + 1, arr.shape[0] - 1)
        raw = arr[idx].astype(np.complex128, copy=False)
        norm = np.linalg.norm(raw, ord=2)
        scaled = (
            np.zeros((channels, channels), dtype=np.complex128)
            if norm == 0.0
            else float(gain) * raw / norm
        )
        reflections.append(contractive_matrix_from_raw(scaled, margin=margin))
    residue = unitary_polar_factor(
        arr[0].astype(np.complex128) + 1e-12 * np.eye(channels, dtype=np.complex128)
    )
    return MatrixLatticeAllPass(reflections, residue=residue, margin=margin)


def experimental_mimo_state_space_to_matrix_lattice(
    A: np.ndarray | Sequence[Sequence[float]],
    B: np.ndarray | Sequence[Sequence[float]],
    C: np.ndarray | Sequence[Sequence[float]],
    D: np.ndarray | Sequence[Sequence[float]],
    *,
    order: int,
    n_markov: int = 256,
    n_freq: int = 256,
    candidate_gains: Sequence[float] | None = None,
    fit_static_gains: bool = False,
    static_gain_mode: str = "both",
    static_gain_iterations: int = 20,
    margin: float = 1e-6,
    n_threads: int = 0,
) -> dict[str, object]:
    """Experimental square MIMO state-space to matrix-lattice all-pass fit.

    The fitted object is a :class:`MatrixLatticeAllPass`, so the target is the
    unitary/polar part of the state-space response, not the full nonunitary gain
    response.  The routine builds Markov-initialized lattice scaffolds for a
    grid of reflection gains and returns the candidate with the smallest
    relative Frobenius error against the target polar factor.  When
    ``fit_static_gains`` is true, the routine also fits static left/right gain
    matrices around the selected lattice response,
    ``H(w) ~= L @ G_lattice(w) @ R``.  This separates all-pass scaffold quality
    from nonunitary static gain mismatch.

    This is a practical diagnostic/initializer for matrix-lattice
    realization experiments.  It is not a full matrix AAK/Nehari solver and it is not a
    proof that an arbitrary stable MIMO state-space model has been exactly
    realized in matrix-lattice coordinates.
    """

    if order < 0:
        raise ValueError("order must be nonnegative")
    if n_markov <= order + 1:
        raise ValueError("n_markov must be larger than order + 1")
    if n_freq < 8:
        raise ValueError("n_freq must be at least 8")
    if margin < 0.0 or margin >= 1.0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and in [0, 1)")
    gains = tuple(
        float(x)
        for x in (candidate_gains if candidate_gains is not None else np.linspace(0.0, 0.85, 9))
    )
    if not gains or any((not np.isfinite(g)) or g < 0.0 for g in gains):
        raise ValueError("candidate_gains must contain finite nonnegative values")

    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    c = np.asarray(C, dtype=np.float64)
    d = np.asarray(D, dtype=np.float64)
    # Reuse validation in the response helper and then enforce square transfer matrices.
    omega = np.linspace(0.0, np.pi, int(n_freq), dtype=np.float64)
    h_state = mimo_state_space_frequency_response(a, b, c, d, omega)
    if h_state.shape[1] != h_state.shape[2]:
        raise ValueError("matrix-lattice realization requires a square MIMO transfer matrix")

    markov = _state_space_markov_response_numpy(a, b, c, d, int(n_markov))
    target = polar_factor_response(h_state)
    singular_values = np.linalg.svd(h_state, compute_uv=False)
    gain_condition_span = float(np.max(singular_values) / max(np.min(singular_values), 1e-30))

    candidates: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for gain in gains:
        lattice = matrix_lattice_scaffold_from_markov(markov, order=order, gain=gain, margin=margin)
        response = lattice.frequency_response(omega, n_threads=n_threads)
        err = _relative_frobenius_error(target, response)
        candidate = {
            "gain": gain,
            "polar_factor_relative_error": err,
            "max_reflection_singular_value": lattice.max_reflection_singular_value(),
            "unitarity_error": lattice.unitarity_error(omega),
            "lattice": lattice,
        }
        candidates.append(candidate)
        if best is None or err < float(best["polar_factor_relative_error"]):
            best = candidate

    assert best is not None
    best_lattice = best["lattice"]
    assert isinstance(best_lattice, MatrixLatticeAllPass)
    best_response = best_lattice.frequency_response(omega, n_threads=n_threads)
    state_response_relative_error = _relative_frobenius_error(h_state, best_response)
    static_gain_fit: dict[str, object] | None = None
    static_gain_relative_error = state_response_relative_error
    static_gain_improvement = 1.0
    static_gain_left_condition = 1.0
    static_gain_right_condition = 1.0
    gain_compensated_response = best_response
    if fit_static_gains:
        static_gain_fit = fit_static_matrix_gains(
            h_state,
            best_response,
            mode=static_gain_mode,
            n_iter=static_gain_iterations,
        )
        static_gain_relative_error = float(static_gain_fit["compensated_relative_error"])
        static_gain_improvement = state_response_relative_error / max(
            static_gain_relative_error, 1e-30
        )
        static_gain_left_condition = float(static_gain_fit["left_gain_condition"])
        static_gain_right_condition = float(static_gain_fit["right_gain_condition"])
        gain_compensated_response = np.asarray(
            static_gain_fit["compensated_response"], dtype=np.complex128
        )

    if float(best["polar_factor_relative_error"]) < 0.25:
        diagnostic_classification = "good_allpass_polar_fit"
    elif (
        fit_static_gains
        and static_gain_improvement >= 2.0
        and static_gain_relative_error < state_response_relative_error
    ):
        diagnostic_classification = "mostly_static_gain_or_nonunitary_mismatch"
    else:
        diagnostic_classification = "poor_lattice_scaffold_fit"

    return {
        "method": "experimental_mimo_state_space_to_matrix_lattice",
        "status": "experimental_polar_allpass_fit",
        "order": int(order),
        "dimension": int(h_state.shape[1]),
        "selected_gain": float(best["gain"]),
        "candidate_gains": np.asarray(gains, dtype=np.float64),
        "candidate_errors": np.asarray(
            [float(cand["polar_factor_relative_error"]) for cand in candidates], dtype=np.float64
        ),
        "candidate_max_reflection_singular_values": np.asarray(
            [float(cand["max_reflection_singular_value"]) for cand in candidates], dtype=np.float64
        ),
        "polar_factor_relative_error": float(best["polar_factor_relative_error"]),
        "unitarity_error": float(best["unitarity_error"]),
        "max_reflection_singular_value": best_lattice.max_reflection_singular_value(),
        "target_gain_condition_span": gain_condition_span,
        "state_response_relative_error": state_response_relative_error,
        "fit_static_gains": bool(fit_static_gains),
        "static_gain_mode": static_gain_mode,
        "static_gain_relative_error": static_gain_relative_error,
        "static_gain_improvement": static_gain_improvement,
        "static_gain_left_condition": static_gain_left_condition,
        "static_gain_right_condition": static_gain_right_condition,
        "static_gain_fit": static_gain_fit,
        "diagnostic_classification": diagnostic_classification,
        "frequency_grid": omega,
        "state_response": h_state,
        "target_polar_response": target,
        "lattice_response": best_response,
        "gain_compensated_response": gain_compensated_response,
        "lattice": best_lattice,
        "reflections": np.asarray(best_lattice.reflections, dtype=np.complex128),
        "residue": np.asarray(best_lattice.residue, dtype=np.complex128),
        "note": "experimental all-pass/polar fit; not an exact matrix AAK/Nehari solver",
    }


__all__ = [
    "MatrixLatticeAllPass",
    "OnlineMatrixLatticeAllPass",
    "contractive_matrix_from_raw",
    "experimental_mimo_state_space_to_matrix_lattice",
    "fit_static_matrix_gains",
    "is_matrix_reflection_stable",
    "matrix_lattice_scaffold_from_markov",
    "matrix_lattice_stage_blocks",
    "matrix_lattice_finite_adjoint",
    "matrix_lattice_impulse_response_convolution",
    "online_matrix_lattice_allpass_process",
    "matrix_spectral_norm",
    "mimo_state_space_frequency_response",
    "polar_factor_response",
    "project_matrix_reflection",
    "psd_matrix_sqrt",
    "unitary_polar_factor",
]
