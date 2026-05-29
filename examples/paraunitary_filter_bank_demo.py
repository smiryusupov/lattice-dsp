"""Paraunitary / perfect-reconstruction matrix-lattice demo.

A matrix lattice all-pass filter is unitary at every frequency.  This example
uses the causal online runtime for the forward analysis transform and a
finite-record time-domain adjoint for synthesis.  The adjoint check is a block
operation because it needs future transformed samples; the forward analysis path
is genuinely streaming.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lattice_dsp import (
    MatrixLatticeAllPass,
    contractive_matrix_from_raw,
    matrix_lattice_finite_adjoint,
    unitary_polar_factor,
)


def _artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_filter(rng: np.random.Generator, channels: int, order: int) -> MatrixLatticeAllPass:
    reflections = [
        contractive_matrix_from_raw(
            0.32
            * (rng.normal(size=(channels, channels)) + 1j * rng.normal(size=(channels, channels)))
        )
        for _ in range(order)
    ]
    residue = unitary_polar_factor(
        rng.normal(size=(channels, channels)) + 1j * rng.normal(size=(channels, channels))
    )
    return MatrixLatticeAllPass(reflections, residue=residue)


def _analysis_streaming(x: np.ndarray, filt: MatrixLatticeAllPass, *, tail: int) -> np.ndarray:
    runtime = filt.to_online_filter()
    return runtime.process(x, drain=tail)


def _synthesis_finite_adjoint(
    y: np.ndarray, filt: MatrixLatticeAllPass, *, tail: int, output_length: int
) -> np.ndarray:
    h = filt.impulse_response(tail)
    return matrix_lattice_finite_adjoint(y, h, output_length=output_length)


def _save_figures(
    x: np.ndarray, y: np.ndarray, x_hat: np.ndarray, filt: MatrixLatticeAllPass
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()
    n_samples = x.shape[0]
    output_energy = np.sum(np.abs(y[:n_samples]) ** 2, axis=0)
    input_energy = np.sum(np.abs(x) ** 2, axis=0)
    reconstruction_error = np.abs(x_hat - x)
    omega_probe = np.linspace(0.0, np.pi, 96)
    singular_values = np.linalg.svd(filt.frequency_response(omega_probe), compute_uv=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    idx = np.arange(x.shape[1])
    width = 0.36
    ax.bar(idx - width / 2, input_energy, width=width, label="analysis input")
    ax.bar(idx + width / 2, output_energy, width=width, label="first output prefix")
    ax.set_xlabel("channel")
    ax.set_ylabel("energy")
    ax.set_title("Streaming analysis moves energy across channels")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "paraunitary_filter_bank_channel_energy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.semilogy(np.maximum(np.mean(reconstruction_error, axis=1), 1e-18))
    ax.set_xlabel("sample")
    ax.set_ylabel("mean absolute reconstruction error")
    ax.set_title("Finite-block time-domain adjoint reconstructs the input")
    fig.tight_layout()
    path = out_dir / "paraunitary_filter_bank_reconstruction_error.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for sv in range(singular_values.shape[1]):
        ax.plot(omega_probe, singular_values[:, sv], label=f"σ{sv + 1}")
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("singular value")
    ax.set_title("Analysis response is unitary at each frequency")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "paraunitary_filter_bank_singular_values.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    span = min(400, n_samples)
    ax.plot(np.real(x[:span, 0]), label="input ch0 real")
    ax.plot(np.real(y[:span, 0]), label="streaming analysis ch0 real", alpha=0.8)
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
    ax.set_title("Causal paraunitary analysis trace")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "paraunitary_filter_bank_streaming_trace.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


rng = np.random.default_rng(2025)
channels = 4
order = 3
n_samples = 4096
tail = 1024

filt = _make_filter(rng, channels, order)
x = rng.normal(size=(n_samples, channels)) + 1j * rng.normal(size=(n_samples, channels))
y = _analysis_streaming(x, filt, tail=tail)
x_hat = _synthesis_finite_adjoint(y, filt, tail=tail, output_length=n_samples)

energy_in = float(np.vdot(x, x).real)
energy_out = float(np.vdot(y, y).real)
relative_reconstruction_error = float(np.linalg.norm(x_hat - x) / np.linalg.norm(x))
relative_energy_error = abs(energy_out - energy_in) / energy_in

print("channels:", channels)
print("order:", order)
print("samples:", n_samples)
print("tail samples:", tail)
print("max reflection singular value:", round(filt.max_reflection_singular_value(), 6))
print("unitarity error:", f"{filt.unitarity_error(np.linspace(0.0, np.pi, 128)):.3e}")
print("relative reconstruction error:", f"{relative_reconstruction_error:.3e}")
print("relative energy error with streamed tail:", f"{relative_energy_error:.3e}")
print("causal analysis: output at n uses current input and stored lattice states")
print(
    "finite adjoint: synthesis is time-domain but noncausal because it uses the full transformed record"
)
print("takeaway: matrix lattice all-pass stages act as streaming paraunitary analysis transforms")

_save_figures(x, y, x_hat, filt)
