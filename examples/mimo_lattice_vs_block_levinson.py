"""Side-by-side MIMO lattice and block Levinson demonstrations.

The two algorithms are complementary rather than interchangeable:

* block Levinson estimates vector AR predictors from block Toeplitz covariance;
* matrix lattice all-pass filters represent unitary/paraunitary MIMO responses.

Both expose matrix reflection-style parameters, which is why they belong in the
same package family.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def _artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def simulate_var(coefficients: list[np.ndarray], samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = len(coefficients)
    channels = coefficients[0].shape[0]
    x = np.zeros((samples + 256, channels))
    noise = rng.normal(size=x.shape)
    for n in range(order, x.shape[0]):
        y = noise[n].copy()
        for lag, a_lag in enumerate(coefficients, start=1):
            y -= a_lag @ x[n - lag]
        x[n] = y
    return x[256:]


def _save_figures(
    *,
    w: np.ndarray,
    h: np.ndarray,
    levinson: ld.MultichannelARResult,
    lattice_reflection_norms: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()
    eye = np.eye(h.shape[1])
    unitarity_error = np.array([np.linalg.norm(hi.conj().T @ hi - eye) for hi in h])
    singular_values = np.linalg.svd(h, compute_uv=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(
        np.arange(1, len(levinson.reflection_spectral_norms) + 1),
        levinson.reflection_spectral_norms,
        marker="o",
    )
    axes[0].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[0].set_title("block Levinson AR")
    axes[0].set_xlabel("stage")
    axes[0].set_ylabel("reflection norm")
    axes[1].plot(
        np.arange(1, len(lattice_reflection_norms) + 1), lattice_reflection_norms, marker="o"
    )
    axes[1].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[1].set_title("matrix all-pass lattice")
    axes[1].set_xlabel("stage")
    axes[1].set_ylabel("reflection norm")
    fig.suptitle("Two matrix reflection diagnostics, two different models")
    fig.tight_layout()
    path = out_dir / "mimo_lattice_vs_block_reflection_norms.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for idx in range(singular_values.shape[1]):
        ax.plot(w, singular_values[:, idx], label=f"σ{idx + 1}")
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("singular value")
    ax.set_title("Matrix-lattice response stays unitary across frequency")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "mimo_lattice_vs_block_singular_values.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.semilogy(w, np.maximum(unitarity_error, 1e-18))
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("||HᴴH - I||₂")
    ax.set_title("All-pass unitarity residual")
    fig.tight_layout()
    path = out_dir / "mimo_lattice_vs_block_unitarity_error.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    # Classical MIMO AR / block Levinson side.
    ar_coefficients = [
        np.array([[0.36, 0.09, 0.02], [-0.04, 0.31, 0.08], [0.03, -0.06, 0.25]]),
        np.array([[-0.12, 0.02, 0.00], [0.01, -0.10, -0.02], [0.02, 0.03, -0.08]]),
    ]
    x = simulate_var(ar_coefficients, samples=30000, seed=11)
    r = ld.multichannel_autocorrelation(x, order=2)
    direct = ld.solve_block_yule_walker_direct(r, order=2)
    levinson = ld.block_levinson_durbin(r, order=2)
    block_diff = np.linalg.norm(direct.coefficients - levinson.coefficients)

    # Matrix all-pass lattice side.
    rng = np.random.default_rng(12)
    channels = 3
    order = 3
    reflections = []
    for _ in range(order):
        raw = 0.25 * (
            rng.standard_normal((channels, channels))
            + 1j * rng.standard_normal((channels, channels))
        )
        reflections.append(ld.contractive_matrix_from_raw(raw))
    residue = ld.unitary_polar_factor(
        rng.standard_normal((channels, channels)) + 1j * rng.standard_normal((channels, channels))
    )
    filt = ld.MatrixLatticeAllPass(reflections, residue)
    w = np.linspace(0, np.pi, 256)
    h = filt.frequency_response(w)
    eye = np.eye(channels)
    unitarity_error = max(np.linalg.norm(hi.conj().T @ hi - eye) for hi in h)
    lattice_reflection_norms = np.array(
        [np.linalg.svd(k, compute_uv=False)[0] for k in reflections]
    )

    print("channels:", channels)
    print("block Levinson order:", levinson.order)
    print("block Levinson/direct coefficient difference:", f"{block_diff:.3e}")
    print("block Levinson reflection norms:", np.round(levinson.reflection_spectral_norms, 6))
    print("matrix all-pass lattice order:", order)
    print("matrix all-pass max reflection norm:", f"{lattice_reflection_norms.max():.6f}")
    print("matrix all-pass unitarity error:", f"{unitarity_error:.3e}")
    print(
        "takeaway: block Levinson validates MIMO AR prediction; matrix lattice covers unitary MIMO filtering"
    )

    _save_figures(w=w, h=h, levinson=levinson, lattice_reflection_norms=lattice_reflection_norms)


if __name__ == "__main__":
    main()
