"""Tutorial: coupled MIMO matrix-lattice filtering with streaming analysis.

A :class:`lattice_dsp.MatrixLatticeAllPass` is a square, multichannel,
frequency-dependent all-pass mixing system.  This example applies the forward
analysis transform with the causal online runtime, then uses a finite-record
noncausal adjoint in the time domain to check reconstruction.  The example is
about the matrix-lattice runtime and diagnostics, not about model reduction.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_coupled_lattice(
    channels: int = 3, order: int = 5, seed: int = 202
) -> ld.MatrixLatticeAllPass:
    """Return a deterministic coupled matrix-lattice all-pass filter."""

    rng = np.random.default_rng(seed)
    reflections = []
    for stage in range(order):
        raw = rng.normal(size=(channels, channels)) + 1j * rng.normal(size=(channels, channels))
        reflections.append(ld.contractive_matrix_from_raw((0.18 + 0.03 * stage) * raw, margin=1e-6))
    residue = ld.unitary_polar_factor(
        rng.normal(size=(channels, channels)) + 1j * rng.normal(size=(channels, channels))
    )
    return ld.MatrixLatticeAllPass(reflections, residue=residue)


def coupled_complex_signal(samples: int = 1024, channels: int = 3, seed: int = 203) -> np.ndarray:
    """Generate a correlated complex multichannel input block."""

    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(samples, 2)) + 1j * rng.normal(size=(samples, 2))
    mixing = np.array(
        [
            [1.0 + 0.0j, 0.35 - 0.10j],
            [0.55 + 0.20j, -0.65 + 0.30j],
            [-0.20 + 0.45j, 0.85 + 0.05j],
        ],
        dtype=np.complex128,
    )[:channels, :]
    x = latent @ mixing.T
    x += 0.08 * (rng.normal(size=(samples, channels)) + 1j * rng.normal(size=(samples, channels)))
    return np.ascontiguousarray(x)


def apply_matrix_lattice_streaming(
    x: np.ndarray, filt: ld.MatrixLatticeAllPass, *, tail: int = 256
) -> np.ndarray:
    """Apply the forward matrix-lattice all-pass with the causal online runtime."""

    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 2 or x.shape[1] != filt.dimension:
        raise ValueError("x must have shape (samples, filter.dimension)")
    return filt.to_online_filter().process(x, drain=tail)


def apply_matrix_lattice_finite_adjoint_time_domain(
    y: np.ndarray,
    filt: ld.MatrixLatticeAllPass,
    *,
    tail: int = 256,
    output_length: int | None = None,
) -> np.ndarray:
    """Apply the finite-record time-domain adjoint used for reconstruction checks."""

    y = np.asarray(y, dtype=np.complex128)
    if y.ndim != 2 or y.shape[1] != filt.dimension:
        raise ValueError("y must have shape (samples, filter.dimension)")
    h = filt.impulse_response(tail)
    return ld.matrix_lattice_finite_adjoint(y, h, output_length=output_length)


def normalized_covariance_magnitude(x: np.ndarray) -> np.ndarray:
    """Return absolute normalized channel covariance."""

    x = np.asarray(x, dtype=np.complex128)
    centered = x - np.mean(x, axis=0, keepdims=True)
    cov = centered.conj().T @ centered / max(x.shape[0] - 1, 1)
    scale = np.sqrt(np.outer(np.real(np.diag(cov)), np.real(np.diag(cov)))) + 1e-30
    return np.abs(cov) / scale


def main() -> None:
    out_dir = artifact_dir()
    channels = 3
    order = 5
    samples = 2048
    tail = 768

    filt = make_coupled_lattice(channels=channels, order=order)
    x = coupled_complex_signal(samples=samples, channels=channels)

    y = apply_matrix_lattice_streaming(x, filt, tail=tail)
    h = filt.impulse_response(tail)
    y_truncated = ld.matrix_lattice_impulse_response_convolution(x, h, drain=tail)
    x_hat = apply_matrix_lattice_finite_adjoint_time_domain(
        y, filt, tail=tail, output_length=samples
    )

    omega = np.linspace(0.0, np.pi, 512)
    response = filt.frequency_response(
        omega, n_threads=int(os.environ.get("LATTICE_DSP_N_THREADS", "1"))
    )
    singular_values = np.linalg.svd(response, compute_uv=False)
    unitarity_error = filt.unitarity_error(omega)
    energy_error = abs(float(np.vdot(y, y).real) - float(np.vdot(x, x).real)) / max(
        float(np.vdot(x, x).real), 1e-30
    )
    reconstruction_error = float(np.linalg.norm(x_hat - x) / max(np.linalg.norm(x), 1e-30))
    streaming_vs_truncated_error = float(
        np.linalg.norm(y - y_truncated) / max(np.linalg.norm(y), 1e-30)
    )

    cov_in = normalized_covariance_magnitude(x)
    cov_out = normalized_covariance_magnitude(y[:samples])

    summary = {
        "channels": channels,
        "order": order,
        "samples": samples,
        "tail_samples": tail,
        "max_reflection_singular_value": filt.max_reflection_singular_value(),
        "real_scalar_parameter_count": filt.parameter_count(),
        "unitarity_error": unitarity_error,
        "streaming_vs_truncated_impulse_error": streaming_vs_truncated_error,
        "energy_relative_error_with_tail": energy_error,
        "finite_adjoint_reconstruction_error": reconstruction_error,
        "input_mean_offdiag_cov": float(
            (np.sum(cov_in) - np.trace(cov_in)) / (channels * (channels - 1))
        ),
        "output_mean_offdiag_cov": float(
            (np.sum(cov_out) - np.trace(cov_out)) / (channels * (channels - 1))
        ),
    }

    csv_path = out_dir / "coupled_mimo_lattice_filter_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print("channels:", channels)
    print("matrix-lattice order:", order)
    print("samples:", samples)
    print("tail samples for energy/reconstruction:", tail)
    print("max reflection singular value:", f"{summary['max_reflection_singular_value']:.4f}")
    print("real scalar parameter count:", summary["real_scalar_parameter_count"])
    print("max unitarity error:", f"{unitarity_error:.3e}")
    print("streaming vs truncated impulse error:", f"{streaming_vs_truncated_error:.3e}")
    print("energy relative error with tail:", f"{energy_error:.3e}")
    print("finite-adjoint reconstruction error:", f"{reconstruction_error:.3e}")
    print(
        "input/output mean off-diagonal covariance:",
        f"{summary['input_mean_offdiag_cov']:.3f}",
        f"{summary['output_mean_offdiag_cov']:.3f}",
    )
    print(
        "causal analysis: y[n] is produced by OnlineMatrixLatticeAllPass before future x samples are seen"
    )
    print("finite adjoint: reconstruction uses the whole transformed block and is noncausal")
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for i in range(channels):
        ax.plot(omega, singular_values[:, i], label=f"s{i + 1}")
    ax.set_title("Matrix-lattice singular values over frequency")
    ax.set_xlabel("radian frequency")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "coupled_mimo_lattice_singular_values.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {fig_path}")

    fig2, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].imshow(cov_in, vmin=0.0, vmax=1.0)
    axes[0].set_title("input |normalized covariance|")
    axes[0].set_xlabel("channel")
    axes[0].set_ylabel("channel")
    im1 = axes[1].imshow(cov_out, vmin=0.0, vmax=1.0)
    axes[1].set_title("streaming output |normalized covariance|")
    axes[1].set_xlabel("channel")
    fig2.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.82)
    fig2_path = out_dir / "coupled_mimo_lattice_covariance.png"
    fig2.savefig(fig2_path, dpi=160, bbox_inches="tight")
    print(f"wrote {fig2_path}")

    fig3, ax3 = plt.subplots(figsize=(8.0, 4.5))
    span = min(320, samples)
    ax3.plot(np.real(x[:span, 0]), label="input ch0 real")
    ax3.plot(np.real(y[:span, 0]), label="streaming output ch0 real", alpha=0.8)
    ax3.set_title("Causal matrix-lattice analysis on one channel")
    ax3.set_xlabel("sample")
    ax3.set_ylabel("amplitude")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3_path = out_dir / "coupled_mimo_lattice_streaming_trace.png"
    fig3.savefig(fig3_path, dpi=160)
    print(f"wrote {fig3_path}")


if __name__ == "__main__":
    main()
