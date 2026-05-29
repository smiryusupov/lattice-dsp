"""Compact matrix-valued unitary response representation.

This demo is intentionally general: it treats a matrix lattice all-pass filter
as a compact representation of a frequency-dependent unitary/MIMO response.
The same primitive can appear in filter banks, array processing, multichannel
audio, learned orthogonal convolutions, or wideband MIMO systems.
"""

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
    *,
    omega: np.ndarray,
    response: np.ndarray,
    stored_real_scalars: int,
    lattice_real_scalars: int,
    filt: MatrixLatticeAllPass,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()
    singular_values = np.linalg.svd(response[::16], compute_uv=False)
    omega_probe = omega[::16]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    labels = ["dense frequency grid", "matrix lattice"]
    ax.bar(labels, [stored_real_scalars, lattice_real_scalars])
    ax.set_yscale("log")
    ax.set_ylabel("stored real scalars")
    ax.set_title("Compact storage for a structured unitary response")
    for label, value in zip(labels, [stored_real_scalars, lattice_real_scalars], strict=True):
        ax.text(label, value, f"{value}", ha="center", va="bottom")
    fig.tight_layout()
    path = out_dir / "matrix_unitary_compression_storage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for idx in range(singular_values.shape[1]):
        ax.plot(omega_probe, singular_values[:, idx], label=f"σ{idx + 1}")
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("singular value")
    ax.set_title("Compression keeps the unitary-response diagnostic")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    path = out_dir / "matrix_unitary_compression_singular_values.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    entry_magnitude = np.abs(response[:, 0, 0])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(omega, entry_magnitude)
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("|H00(e^{jω})|")
    ax.set_title("A single entry is frequency dependent even though H is unitary")
    fig.tight_layout()
    path = out_dir / "matrix_unitary_compression_entry_response.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    span = min(256, x.shape[0])
    ax.plot(np.real(x[:span, 0]), label="input ch0 real")
    ax.plot(np.real(y[:span, 0]), label="streaming output ch0 real", alpha=0.8)
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
    ax.set_title("The compact lattice also runs as a causal time-domain filter")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "matrix_unitary_compression_streaming_trace.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


rng = np.random.default_rng(7)
dim = 4
order = 3
n_frequencies = 1024
stream_samples = 768
tail = 512

reflections = [
    contractive_matrix_from_raw(
        0.28 * (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    )
    for _ in range(order)
]
residue = unitary_polar_factor(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
unitary_response = MatrixLatticeAllPass(reflections, residue=residue)

omega = 2.0 * np.pi * np.arange(n_frequencies) / n_frequencies
h = unitary_response.frequency_response(omega)
x = rng.normal(size=(stream_samples, dim)) + 1j * rng.normal(size=(stream_samples, dim))
y = unitary_response.to_online_filter().process(x, drain=tail)
streaming_energy_error = abs(float(np.vdot(y, y).real) - float(np.vdot(x, x).real)) / float(
    np.vdot(x, x).real
)

stored_real_scalars = n_frequencies * dim * dim * 2
lattice_real_scalars = unitary_response.parameter_count(real_scalars=True, include_residue=True)
print("MIMO dimension:", dim)
print("frequency bins:", n_frequencies)
print("matrix lattice order:", order)
print("store all frequency-bin matrices, real scalars:", stored_real_scalars)
print("matrix lattice representation, real scalars:", lattice_real_scalars)
print("compression ratio:", round(stored_real_scalars / lattice_real_scalars, 1), "x")
print("unitarity error:", f"{unitary_response.unitarity_error(omega[::32]):.3e}")
print("streaming samples:", stream_samples)
print("tail samples:", tail)
print("streaming energy error with tail:", f"{streaming_energy_error:.3e}")
print("example response[0] first row:", np.round(h[0, 0], 3))
print(
    "causal runtime: the compact response is not only stored on a grid; it can process vector samples online"
)

_save_figures(
    omega=omega,
    response=h,
    stored_real_scalars=stored_real_scalars,
    lattice_real_scalars=lattice_real_scalars,
    filt=unitary_response,
    x=x,
    y=y,
)
