"""Matrix-valued lattice/all-pass response demo."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lattice_dsp import MatrixLatticeAllPass, contractive_matrix_from_raw, unitary_polar_factor


def _artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figures(
    omega: np.ndarray, response: np.ndarray, filt: MatrixLatticeAllPass, impulse: np.ndarray
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()
    singular_values = np.linalg.svd(response, compute_uv=False)
    eye = np.eye(filt.dimension)
    unitarity_error = np.array([np.linalg.norm(hi.conj().T @ hi - eye) for hi in response])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for idx in range(singular_values.shape[1]):
        ax.plot(omega, singular_values[:, idx], label=f"σ{idx + 1}")
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("singular value")
    ax.set_title("All singular values stay at one for an all-pass response")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "matrix_lattice_allpass_singular_values.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.semilogy(omega, np.maximum(unitarity_error, 1e-18))
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("||HᴴH - I||₂")
    ax.set_title("Frequency-by-frequency unitarity residual")
    fig.tight_layout()
    path = out_dir / "matrix_lattice_allpass_unitarity_error.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.semilogy(np.maximum(np.linalg.norm(impulse, axis=(1, 2)), 1e-18))
    ax.set_xlabel("sample")
    ax.set_ylabel("impulse-response block Frobenius norm")
    ax.set_title("Causal online realization has a decaying all-pass tail")
    fig.tight_layout()
    path = out_dir / "matrix_lattice_allpass_streaming_impulse.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    freq_indices = [0, len(omega) // 4, len(omega) // 2]
    fig, axes = plt.subplots(1, len(freq_indices), figsize=(9.2, 3.2))
    for ax, idx in zip(axes, freq_indices, strict=True):
        im = ax.imshow(np.abs(response[idx]))
        ax.set_title(f"|H(e^{{jω}})| at ω={omega[idx]:.2f}")
        ax.set_xlabel("input")
        ax.set_ylabel("output")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    path = out_dir / "matrix_lattice_allpass_entry_magnitudes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


rng = np.random.default_rng(123)
dim = 3
order = 4

reflections = [
    contractive_matrix_from_raw(
        0.35 * (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    )
    for _ in range(order)
]
residue = unitary_polar_factor(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))

filt = MatrixLatticeAllPass(reflections, residue=residue)
omega = np.linspace(0.0, np.pi, 256)
response = filt.frequency_response(omega)

# Streaming realization check: process one impulse per input channel and compare
# the truncated impulse-response frequency response with the direct evaluator.
n_impulse = 512
impulse_response = np.empty((n_impulse, dim, dim), dtype=np.complex128)
for input_channel in range(dim):
    runtime = filt.to_online_filter()
    impulse = np.zeros((n_impulse, dim), dtype=np.complex128)
    impulse[0, input_channel] = 1.0
    y = runtime.process(impulse)
    impulse_response[:, :, input_channel] = y
streaming_probe = np.linspace(0.0, np.pi, 32)
powers = np.exp(-1j * np.outer(streaming_probe, np.arange(n_impulse)))
streaming_response = np.einsum("wn,nij->wij", powers, impulse_response)
streaming_relative_error = np.linalg.norm(
    streaming_response - filt.frequency_response(streaming_probe)
) / np.linalg.norm(filt.frequency_response(streaming_probe))

print("dimension:", filt.dimension)
print("order:", filt.order)
print("max reflection singular value:", round(filt.max_reflection_singular_value(), 6))
print("real scalar parameter count:", filt.parameter_count())
print("max unitarity error:", f"{filt.unitarity_error(omega):.3e}")
print("streaming impulse/frequency relative error:", f"{streaming_relative_error:.3e}")
print("response shape:", response.shape)

_save_figures(omega, response, filt, impulse_response)
