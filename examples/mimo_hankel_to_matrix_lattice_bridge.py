"""Tutorial: bridge diagnostics from MIMO block-Hankel reduction to matrix lattices.

This is deliberately a bridge diagnostic, not a matrix AAK/Nehari or exact
matrix-lattice realization solver.  A general stable MIMO state-space model has
frequency-dependent gains, while :class:`lattice_dsp.MatrixLatticeAllPass`
represents unitary/all-pass scattering.  The useful question at this stage is:
can the reduced MIMO model provide stable matrix-lattice *initialization data*
and how far is that all-pass scaffold from the reduced model's unitary/polar
part?
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

import lattice_dsp as ld

try:
    from examples.mimo_coupled_model_reduction import coupled_state_space, state_spectral_radius
except Exception:  # pragma: no cover - direct script execution uses examples/ on sys.path.
    from mimo_coupled_model_reduction import coupled_state_space, state_spectral_radius


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_space_frequency_response(A, B, C, D, omega: np.ndarray) -> np.ndarray:
    """Evaluate ``D + C z^-1 (I - A z^-1)^-1 B`` on the unit circle."""

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    omega = np.asarray(omega, dtype=float).reshape(-1)
    n_outputs, n_inputs = D.shape
    n_state = A.shape[0]
    response = np.empty((omega.size, n_outputs, n_inputs), dtype=np.complex128)
    eye = np.eye(n_state, dtype=np.complex128)
    for i, w in enumerate(omega):
        zinv = np.exp(-1j * float(w))
        if n_state:
            response[i] = D + C @ (zinv * np.linalg.solve(eye - zinv * A, B))
        else:
            response[i] = D
    return response


def polar_factor_per_frequency(response: np.ndarray) -> np.ndarray:
    """Return the unitary polar factor of each square frequency-response matrix."""

    response = np.asarray(response, dtype=np.complex128)
    if response.ndim != 3 or response.shape[1] != response.shape[2]:
        raise ValueError("response must have shape (frequency, channels, channels)")
    out = np.empty_like(response)
    for i, h in enumerate(response):
        u, _, vh = np.linalg.svd(h, full_matrices=False)
        out[i] = u @ vh
    return out


def matrix_lattice_scaffold_from_markov(
    markov: np.ndarray, *, order: int, gain: float = 0.55
) -> ld.MatrixLatticeAllPass:
    """Build a stable matrix-lattice scaffold from early reduced Markov matrices.

    The result is an all-pass/unitary scaffold.  It is not an exact realization
    of the reduced MIMO model; the Markov matrices only provide coupling
    directions for contractive reflection initializers.
    """

    markov = np.asarray(markov, dtype=float)
    if markov.ndim != 3 or markov.shape[1] != markov.shape[2]:
        raise ValueError("markov must have shape (samples, channels, channels)")
    channels = markov.shape[1]
    reflections = []
    for k in range(order):
        idx = min(k + 1, markov.shape[0] - 1)
        raw = markov[idx]
        norm = np.linalg.norm(raw, ord=2)
        scaled = (
            np.zeros((channels, channels), dtype=np.complex128)
            if norm == 0.0
            else gain * raw / norm
        )
        reflections.append(ld.contractive_matrix_from_raw(scaled, margin=1e-5))
    residue = ld.unitary_polar_factor(markov[0] + 1e-6 * np.eye(channels))
    return ld.MatrixLatticeAllPass(reflections, residue=residue, margin=1e-9)


def response_relative_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.linalg.norm(reference - estimate) / max(np.linalg.norm(reference), 1e-30))


def main() -> None:
    out_dir = artifact_dir()

    full_order = 12
    reduced_order = 6
    channels = 3
    n_markov = 220
    block_rows = block_cols = 28
    lattice_order = 6
    omega = np.linspace(0.0, np.pi, 384)

    A, B, C, D = coupled_state_space(order=full_order, outputs=channels, inputs=channels, seed=31)
    markov = ld.mimo_state_space_markov_response(A, B, C, D, n_markov)
    reduced = ld.finite_hankel_reduce_mimo(
        markov, reduced_order=reduced_order, block_rows=block_rows, block_cols=block_cols
    )
    reduced_markov = ld.mimo_state_space_markov_response(
        reduced["A"], reduced["B"], reduced["C"], reduced["D"], n_markov
    )

    lattice = matrix_lattice_scaffold_from_markov(reduced_markov, order=lattice_order)
    h_reduced = state_space_frequency_response(
        reduced["A"], reduced["B"], reduced["C"], reduced["D"], omega
    )
    polar_target = polar_factor_per_frequency(h_reduced)
    h_lattice = lattice.frequency_response(omega, n_threads=1)

    gain_singular = np.linalg.svd(h_reduced, compute_uv=False)
    polar_error = response_relative_error(polar_target, h_lattice)
    gain_flatness = float(np.max(gain_singular) / max(np.min(gain_singular), 1e-30))
    markov_error = float(np.sum((markov - reduced_markov) ** 2) / (np.sum(markov**2) + 1e-30))
    lattice_unitarity = lattice.unitarity_error(omega)

    summary = {
        "full_order": full_order,
        "reduced_order": reduced_order,
        "channels": channels,
        "lattice_order": lattice_order,
        "reduced_state_radius": state_spectral_radius(reduced["A"]),
        "retained_hankel_energy": float(reduced["retained_hankel_energy"]),
        "relative_markov_error": markov_error,
        "lattice_max_reflection_singular_value": lattice.max_reflection_singular_value(),
        "lattice_unitarity_error": lattice_unitarity,
        "polar_factor_relative_error": polar_error,
        "reduced_response_gain_condition_span": gain_flatness,
        "note": "matrix-lattice scaffold, not exact realization",
    }

    csv_path = out_dir / "mimo_hankel_to_matrix_lattice_bridge_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print("full state order:", full_order)
    print("reduced state order:", reduced_order)
    print("channels:", channels)
    print("reduced state radius:", f"{summary['reduced_state_radius']:.4f}")
    print("retained block-Hankel energy:", f"{summary['retained_hankel_energy']:.6f}")
    print("relative Markov error:", f"{markov_error:.3e}")
    print("matrix-lattice scaffold order:", lattice_order)
    print(
        "max scaffold reflection singular value:",
        f"{summary['lattice_max_reflection_singular_value']:.4f}",
    )
    print("scaffold unitarity error:", f"{lattice_unitarity:.3e}")
    print("polar-factor relative error:", f"{polar_error:.3e}")
    print("reduced response gain condition span:", f"{gain_flatness:.3e}")
    print("bridge status: scaffold diagnostic, not an exact matrix-lattice realization")
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for i in range(channels):
        ax.plot(omega, gain_singular[:, i], label=f"gain s{i + 1}")
    ax.set_title("Reduced MIMO response singular values")
    ax.set_xlabel("radian frequency")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "mimo_bridge_reduced_response_gains.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {fig_path}")

    per_freq_error = np.linalg.norm(polar_target - h_lattice, axis=(1, 2)) / np.maximum(
        np.linalg.norm(polar_target, axis=(1, 2)), 1e-30
    )
    fig2, ax2 = plt.subplots(figsize=(8.0, 4.5))
    ax2.plot(omega, per_freq_error)
    ax2.set_title("Matrix-lattice scaffold error against reduced-model polar factor")
    ax2.set_xlabel("radian frequency")
    ax2.set_ylabel("relative Frobenius error")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2_path = out_dir / "mimo_bridge_polar_error.png"
    fig2.savefig(fig2_path, dpi=160)
    print(f"wrote {fig2_path}")

    fig3, ax3 = plt.subplots(figsize=(8.0, 4.5))
    hsv = np.asarray(reduced["hankel_singular_values"], dtype=float)
    idx = np.arange(1, min(40, hsv.size) + 1)
    ax3.semilogy(idx, hsv[: idx.size], marker="o")
    ax3.axvline(reduced_order, linestyle="--", linewidth=1.2)
    ax3.set_title("Block-Hankel singular values feeding the bridge")
    ax3.set_xlabel("index")
    ax3.set_ylabel("singular value")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3_path = out_dir / "mimo_bridge_block_hankel_singular_values.png"
    fig3.savefig(fig3_path, dpi=160)
    print(f"wrote {fig3_path}")


if __name__ == "__main__":
    main()
