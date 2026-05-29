"""Tutorial: calibrating matrix-lattice static gain diagnostics.

This example uses a known matrix-lattice all-pass response, then wraps it in
static nonunitary gains.  The diagnostic fit should recover a near-perfect
compensated response.  This is a calibration case for the experimental
MIMO state-space to matrix-lattice realization scaffold: it separates true
all-pass/lattice mismatch from static gain mismatch.
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


def known_lattice(dimension: int = 3, order: int = 4) -> ld.MatrixLatticeAllPass:
    rng = np.random.default_rng(421)
    reflections = []
    for stage in range(order):
        raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
        reflections.append(ld.contractive_matrix_from_raw(0.30 * raw / (1.0 + stage)))
    residue_raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    residue = ld.unitary_polar_factor(residue_raw)
    return ld.MatrixLatticeAllPass(reflections, residue=residue)


def main() -> None:
    out_dir = artifact_dir()
    dimension = 3
    order = 4
    n_freq = 384
    omega = np.linspace(0.0, np.pi, n_freq)

    lattice = known_lattice(dimension=dimension, order=order)
    response = lattice.frequency_response(omega, n_threads=1)

    rng = np.random.default_rng(422)
    left_true = np.diag([1.8, 0.75, 0.35]).astype(np.complex128)
    right_true = ld.unitary_polar_factor(
        rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    )
    target = np.asarray([left_true @ h @ right_true for h in response], dtype=np.complex128)

    allpass_fit = ld.fit_static_matrix_gains(response, response, mode="both", n_iter=6)
    gain_fit = ld.fit_static_matrix_gains(target, response, mode="both", n_iter=40)

    summary = {
        "dimension": dimension,
        "order": order,
        "known_lattice_unitarity_error": lattice.unitarity_error(omega),
        "allpass_raw_error": allpass_fit["raw_relative_error"],
        "allpass_compensated_error": allpass_fit["compensated_relative_error"],
        "gain_wrapped_raw_error": gain_fit["raw_relative_error"],
        "gain_wrapped_compensated_error": gain_fit["compensated_relative_error"],
        "gain_improvement": float(gain_fit["raw_relative_error"])
        / max(float(gain_fit["compensated_relative_error"]), 1e-30),
        "left_gain_condition": gain_fit["left_gain_condition"],
        "right_gain_condition": gain_fit["right_gain_condition"],
        "max_reflection_singular_value": lattice.max_reflection_singular_value(),
    }

    csv_path = out_dir / "experimental_mimo_matrix_lattice_calibration_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print("known lattice dimension:", dimension)
    print("known lattice order:", order)
    print("known lattice unitarity error:", f"{summary['known_lattice_unitarity_error']:.3e}")
    print("max reflection singular value:", f"{summary['max_reflection_singular_value']:.4f}")
    print("all-pass calibration raw error:", f"{summary['allpass_raw_error']:.3e}")
    print("all-pass calibration compensated error:", f"{summary['allpass_compensated_error']:.3e}")
    print("gain-wrapped raw error:", f"{summary['gain_wrapped_raw_error']:.3e}")
    print("gain-wrapped compensated error:", f"{summary['gain_wrapped_compensated_error']:.3e}")
    print("static-gain improvement:", f"{summary['gain_improvement']:.2e}x")
    print(
        "left/right fitted gain condition:",
        f"{summary['left_gain_condition']:.3f}",
        f"{summary['right_gain_condition']:.3f}",
    )
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    compensated = np.asarray(gain_fit["compensated_response"], dtype=np.complex128)
    raw_per_freq = np.linalg.norm(target - response, axis=(1, 2)) / np.maximum(
        np.linalg.norm(target, axis=(1, 2)), 1e-30
    )
    compensated_per_freq = np.linalg.norm(target - compensated, axis=(1, 2)) / np.maximum(
        np.linalg.norm(target, axis=(1, 2)), 1e-30
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.semilogy(omega, raw_per_freq, label="raw lattice vs gain-wrapped target")
    ax.semilogy(omega, compensated_per_freq, label="after fitted static gains")
    ax.set_title("Static gain calibration for matrix-lattice response")
    ax.set_xlabel("radian frequency")
    ax.set_ylabel("relative Frobenius error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / "experimental_mimo_matrix_lattice_calibration_error.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {fig_path}")

    sv = np.linalg.svd(np.asarray(gain_fit["left_gain"]), compute_uv=False)
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.0))
    ax2.plot(np.arange(1, sv.size + 1), sv, marker="o")
    ax2.set_title("Fitted left static-gain singular values")
    ax2.set_xlabel("singular-value index")
    ax2.set_ylabel("singular value")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2_path = out_dir / "experimental_mimo_matrix_lattice_calibration_gain.png"
    fig2.savefig(fig2_path, dpi=160)
    print(f"wrote {fig2_path}")


if __name__ == "__main__":
    main()
