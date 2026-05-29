"""ML-adjacent streaming unitary convolution demo.

Orthogonal/unitary convolutions are useful in ML because they preserve signal
norms and keep forward/adjoint maps well conditioned.  This example uses the
causal online matrix-lattice all-pass runtime as a streaming multichannel
unitary convolution block.  A finite-record time-domain adjoint is used only for
an offline reconstruction diagnostic.
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
            0.25
            * (rng.normal(size=(channels, channels)) + 1j * rng.normal(size=(channels, channels)))
        )
        for _ in range(order)
    ]
    residue = unitary_polar_factor(
        rng.normal(size=(channels, channels)) + 1j * rng.normal(size=(channels, channels))
    )
    return MatrixLatticeAllPass(reflections, residue=residue)


def _forward_streaming(batch: np.ndarray, filt: MatrixLatticeAllPass, *, tail: int) -> np.ndarray:
    out = np.empty((batch.shape[0], batch.shape[1] + tail, batch.shape[2]), dtype=np.complex128)
    for item in range(batch.shape[0]):
        out[item] = filt.to_online_filter().process(batch[item], drain=tail)
    return out


def _finite_adjoint(
    batch: np.ndarray, filt: MatrixLatticeAllPass, *, tail: int, output_length: int
) -> np.ndarray:
    h = filt.impulse_response(tail)
    out = np.empty((batch.shape[0], output_length, batch.shape[2]), dtype=np.complex128)
    for item in range(batch.shape[0]):
        out[item] = matrix_lattice_finite_adjoint(batch[item], h, output_length=output_length)
    return out


def _save_figures(
    *,
    input_norms: np.ndarray,
    output_norms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_hat: np.ndarray,
    omega_probe: np.ndarray,
    singular_values: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    batch = np.arange(len(input_norms))
    ax.plot(batch, input_norms, marker="o", label="input")
    ax.plot(batch, output_norms, marker="x", linestyle="--", label="streaming output with tail")
    ax.set_xlabel("batch item")
    ax.set_ylabel("flattened signal norm")
    ax.set_title("Streaming unitary convolution preserves each batch-item norm")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "ml_unitary_convolution_batch_norms.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for idx in range(singular_values.shape[1]):
        ax.plot(omega_probe, singular_values[:, idx], label=f"σ{idx + 1}")
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("singular value")
    ax.set_title("Frequency response stays unitary")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    path = out_dir / "ml_unitary_convolution_singular_values.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    channel_energy_in = np.mean(np.abs(x) ** 2, axis=(0, 1))
    channel_energy_out = np.mean(np.abs(y[:, : x.shape[1]]) ** 2, axis=(0, 1))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    idx = np.arange(len(channel_energy_in))
    width = 0.36
    ax.bar(idx - width / 2, channel_energy_in, width=width, label="input")
    ax.bar(idx + width / 2, channel_energy_out, width=width, label="streaming output prefix")
    ax.set_xlabel("channel")
    ax.set_ylabel("mean energy")
    ax.set_title("Energy may move across channels while total norm is preserved")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "ml_unitary_convolution_channel_energy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    reconstruction = np.linalg.norm((x_hat - x).reshape(x.shape[0], -1), axis=1) / input_norms
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.semilogy(np.maximum(reconstruction, 1e-18), marker="o")
    ax.set_xlabel("batch item")
    ax.set_ylabel("relative finite-adjoint reconstruction error")
    ax.set_title("Time-domain adjoint recovers the input")
    fig.tight_layout()
    path = out_dir / "ml_unitary_convolution_adjoint_error.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    span = min(256, x.shape[1])
    ax.plot(np.real(x[0, :span, 0]), label="input ch0 real")
    ax.plot(np.real(y[0, :span, 0]), label="streaming output ch0 real", alpha=0.8)
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
    ax.set_title("Causal online convolution trace")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "ml_unitary_convolution_streaming_trace.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


rng = np.random.default_rng(314)
batch_size = 8
sequence_length = 1024
channels = 6
order = 4
tail = 1024

filt = _make_filter(rng, channels, order)
x = rng.normal(size=(batch_size, sequence_length, channels)) + 1j * rng.normal(
    size=(batch_size, sequence_length, channels)
)
y = _forward_streaming(x, filt, tail=tail)
x_hat = _finite_adjoint(y, filt, tail=tail, output_length=sequence_length)

input_norms = np.linalg.norm(x.reshape(batch_size, -1), axis=1)
output_norms = np.linalg.norm(y.reshape(batch_size, -1), axis=1)
max_norm_error = float(np.max(np.abs(output_norms - input_norms) / input_norms))
max_adjoint_error = float(
    np.max(np.linalg.norm((x_hat - x).reshape(batch_size, -1), axis=1) / input_norms)
)

omega_probe = np.linspace(0.0, np.pi, 64)
response = filt.frequency_response(omega_probe)
singular_values = np.linalg.svd(response, compute_uv=False)

print("batch size:", batch_size)
print("sequence length:", sequence_length)
print("channels:", channels)
print("order:", order)
print("tail samples:", tail)
print("real scalar parameters:", filt.parameter_count())
print("max streaming norm-preservation error:", f"{max_norm_error:.3e}")
print("max finite-adjoint reconstruction error:", f"{max_adjoint_error:.3e}")
print("singular value range:", f"[{singular_values.min():.6f}, {singular_values.max():.6f}]")
print("causal forward: output at n uses current input and previous lattice state")
print("finite adjoint: reconstruction is time-domain but noncausal over the block")
print(
    "takeaway: matrix lattice filters can parameterize streaming norm-preserving convolution blocks"
)

_save_figures(
    input_norms=input_norms,
    output_norms=output_norms,
    x=x,
    y=y,
    x_hat=x_hat,
    omega_probe=omega_probe,
    singular_values=singular_values,
)
