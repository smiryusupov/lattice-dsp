"""Tutorial: experimental MIMO state-space to matrix-lattice realization.

This example is intentionally experimental.  A general stable MIMO state-space
model is not automatically a matrix-lattice all-pass model: it can have
frequency-dependent gain.  The helper in this tutorial therefore fits a stable
matrix-lattice all-pass to the *unitary polar factor* of the reduced MIMO
response.  The returned lattice is useful as a realization scaffold and diagnostic
initialization, not as a proof of exact matrix AAK/Nehari optimality.
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


def main() -> None:
    out_dir = artifact_dir()

    full_order = 12
    reduced_order = 6
    channels = 3
    n_markov = 240
    block_rows = block_cols = 28
    lattice_order = 6
    n_freq = 384
    candidate_gains = np.linspace(0.15, 0.85, 8)

    A, B, C, D = coupled_state_space(order=full_order, outputs=channels, inputs=channels, seed=71)
    markov = ld.mimo_state_space_markov_response(A, B, C, D, n_markov)
    reduced = ld.finite_hankel_reduce_mimo(
        markov, reduced_order=reduced_order, block_rows=block_rows, block_cols=block_cols
    )

    fit = ld.experimental_mimo_state_space_to_matrix_lattice(
        reduced["A"],
        reduced["B"],
        reduced["C"],
        reduced["D"],
        order=lattice_order,
        n_markov=n_markov,
        n_freq=n_freq,
        candidate_gains=candidate_gains,
        fit_static_gains=True,
        static_gain_mode="both",
        static_gain_iterations=20,
        n_threads=1,
    )

    summary = {
        "full_order": full_order,
        "reduced_order": reduced_order,
        "channels": channels,
        "lattice_order": lattice_order,
        "selected_gain": fit["selected_gain"],
        "reduced_state_radius": state_spectral_radius(reduced["A"]),
        "retained_hankel_energy": float(reduced["retained_hankel_energy"]),
        "polar_factor_relative_error": fit["polar_factor_relative_error"],
        "state_response_relative_error": fit["state_response_relative_error"],
        "static_gain_relative_error": fit["static_gain_relative_error"],
        "static_gain_improvement": fit["static_gain_improvement"],
        "static_gain_left_condition": fit["static_gain_left_condition"],
        "static_gain_right_condition": fit["static_gain_right_condition"],
        "diagnostic_classification": fit["diagnostic_classification"],
        "unitarity_error": fit["unitarity_error"],
        "max_reflection_singular_value": fit["max_reflection_singular_value"],
        "target_gain_condition_span": fit["target_gain_condition_span"],
        "note": fit["note"],
    }

    csv_path = out_dir / "experimental_mimo_matrix_lattice_realization_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print("full state order:", full_order)
    print("reduced state order:", reduced_order)
    print("channels:", channels)
    print("lattice realization order:", lattice_order)
    print("selected reflection gain:", f"{float(fit['selected_gain']):.4f}")
    print("reduced state radius:", f"{summary['reduced_state_radius']:.4f}")
    print("retained block-Hankel energy:", f"{summary['retained_hankel_energy']:.6f}")
    print("polar-factor fit error:", f"{float(fit['polar_factor_relative_error']):.3e}")
    print("raw state-response error:", f"{float(fit['state_response_relative_error']):.3e}")
    print("static-gain compensated error:", f"{float(fit['static_gain_relative_error']):.3e}")
    print("static-gain improvement:", f"{float(fit['static_gain_improvement']):.2f}x")
    print(
        "static gain conditions:",
        f"{float(fit['static_gain_left_condition']):.3f}",
        f"{float(fit['static_gain_right_condition']):.3f}",
    )
    print("diagnostic classification:", fit["diagnostic_classification"])
    print("lattice unitarity error:", f"{float(fit['unitarity_error']):.3e}")
    print("max reflection singular value:", f"{float(fit['max_reflection_singular_value']):.4f}")
    print("target gain condition span:", f"{float(fit['target_gain_condition_span']):.3e}")
    print("status: experimental all-pass/polar realization scaffold, not exact matrix AAK/Nehari")
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    omega = np.asarray(fit["frequency_grid"], dtype=float)
    target = np.asarray(fit["target_polar_response"], dtype=np.complex128)
    response = np.asarray(fit["lattice_response"], dtype=np.complex128)
    state_response = np.asarray(fit["state_response"], dtype=np.complex128)
    compensated = np.asarray(fit["gain_compensated_response"], dtype=np.complex128)
    per_freq_error = np.linalg.norm(target - response, axis=(1, 2)) / np.maximum(
        np.linalg.norm(target, axis=(1, 2)), 1e-30
    )
    raw_state_error = np.linalg.norm(state_response - response, axis=(1, 2)) / np.maximum(
        np.linalg.norm(state_response, axis=(1, 2)), 1e-30
    )
    compensated_error = np.linalg.norm(state_response - compensated, axis=(1, 2)) / np.maximum(
        np.linalg.norm(state_response, axis=(1, 2)), 1e-30
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.semilogy(omega, per_freq_error, label="lattice vs polar target")
    ax.semilogy(omega, raw_state_error, label="raw lattice vs state response")
    ax.semilogy(omega, compensated_error, label="after static gains")
    ax.set_title("Experimental matrix-lattice realization diagnostics")
    ax.set_xlabel("radian frequency")
    ax.set_ylabel("relative Frobenius error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "experimental_mimo_matrix_lattice_realization_error.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {fig_path}")

    fig2, ax2 = plt.subplots(figsize=(8.0, 4.5))
    ax2.plot(np.asarray(fit["candidate_gains"]), np.asarray(fit["candidate_errors"]), marker="o")
    ax2.axvline(float(fit["selected_gain"]), linestyle="--", linewidth=1.2)
    ax2.set_title("Reflection gain search")
    ax2.set_xlabel("candidate gain")
    ax2.set_ylabel("polar-factor relative error")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2_path = out_dir / "experimental_mimo_matrix_lattice_gain_search.png"
    fig2.savefig(fig2_path, dpi=160)
    print(f"wrote {fig2_path}")

    reflections = np.asarray(fit["reflections"], dtype=np.complex128)
    reflection_svs = (
        np.array([np.linalg.svd(k, compute_uv=False) for k in reflections])
        if reflections.size
        else np.empty((0, channels))
    )
    fig3, ax3 = plt.subplots(figsize=(8.0, 4.5))
    if reflection_svs.size:
        for j in range(reflection_svs.shape[1]):
            ax3.plot(
                np.arange(1, reflection_svs.shape[0] + 1),
                reflection_svs[:, j],
                marker="o",
                label=f"sv{j + 1}",
            )
    ax3.set_title("Matrix-reflection singular values")
    ax3.set_xlabel("lattice stage")
    ax3.set_ylabel("singular value")
    ax3.set_ylim(0.0, 1.02)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3_path = out_dir / "experimental_mimo_matrix_lattice_reflections.png"
    fig3.savefig(fig3_path, dpi=160)
    print(f"wrote {fig3_path}")


if __name__ == "__main__":
    main()
