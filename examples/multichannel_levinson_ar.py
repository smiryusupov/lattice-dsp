"""Multichannel Levinson-Durbin estimation for a vector AR process."""

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
    x = np.zeros((samples + 512, channels))
    noise = rng.normal(size=x.shape)
    for n in range(order, x.shape[0]):
        value = noise[n].copy()
        for lag, a_lag in enumerate(coefficients, start=1):
            value -= a_lag @ x[n - lag]
        x[n] = value
    return x[512:]


def _save_figures(
    *,
    true_coefficients: np.ndarray,
    estimated_coefficients: np.ndarray,
    reflection_norms: np.ndarray,
    prediction_error: np.ndarray,
    residual_covariance: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()

    order = estimated_coefficients.shape[0]
    fig, axes = plt.subplots(order, 2, figsize=(8.0, 3.2 * order), squeeze=False)
    limit = float(max(np.max(np.abs(true_coefficients)), np.max(np.abs(estimated_coefficients))))
    for lag in range(order):
        for col, (name, values) in enumerate(
            (("true", true_coefficients[lag]), ("estimated", estimated_coefficients[lag].real))
        ):
            ax = axes[lag, col]
            im = ax.imshow(values, vmin=-limit, vmax=limit)
            ax.set_title(f"lag {lag + 1}: {name} coefficient matrix")
            ax.set_xlabel("input channel")
            ax.set_ylabel("output channel")
            fig.colorbar(im, ax=ax, shrink=0.78)
    fig.tight_layout()
    path = out_dir / "multichannel_levinson_coefficients.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    stages = np.arange(1, len(reflection_norms) + 1)
    ax.plot(stages, reflection_norms, marker="o")
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Levinson stage")
    ax.set_ylabel("reflection spectral norm")
    ax.set_title("Block reflection norms stay below the stability boundary")
    ax.set_xticks(stages)
    fig.tight_layout()
    path = out_dir / "multichannel_levinson_reflection_norms.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, name, matrix in (
        (axes[0], "Levinson prediction error", prediction_error.real),
        (axes[1], "sample residual covariance", residual_covariance.real),
    ):
        im = ax.imshow(matrix)
        ax.set_title(name)
        ax.set_xlabel("channel")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.tight_layout()
    path = out_dir / "multichannel_levinson_residual_covariance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    true_coefficients = [
        np.array([[0.42, 0.08, -0.04], [-0.05, 0.33, 0.06], [0.02, -0.07, 0.28]]),
        np.array([[-0.16, 0.03, 0.02], [0.01, -0.12, -0.03], [0.00, 0.04, -0.10]]),
    ]
    x = simulate_var(true_coefficients, samples=40000, seed=7)
    r = ld.multichannel_autocorrelation(x, order=2, biased=True, demean=True)

    direct = ld.solve_block_yule_walker_direct(r, order=2)
    levinson = ld.block_levinson_durbin(r, order=2)
    residual = ld.multichannel_prediction_error(x, levinson.coefficients)

    true_stack = np.asarray(true_coefficients)
    coeff_rel_error = np.linalg.norm(levinson.coefficients.real - true_stack) / np.linalg.norm(
        true_stack
    )
    solver_agreement = np.linalg.norm(levinson.coefficients - direct.coefficients)
    residual_covariance = np.cov(residual.T)

    print("channels:", x.shape[1])
    print("order:", levinson.order)
    print(
        "companion spectral radius:", f"{ld.companion_spectral_radius(levinson.coefficients):.6f}"
    )
    print("direct/block-Levinson coefficient difference:", f"{solver_agreement:.3e}")
    print("relative coefficient error vs true VAR:", f"{coeff_rel_error:.3e}")
    print("reflection spectral norms:", np.round(levinson.reflection_spectral_norms, 6))
    print("prediction error covariance trace:", f"{np.trace(levinson.prediction_error).real:.6f}")
    print("sample residual variance trace:", f"{np.trace(residual_covariance).real:.6f}")
    print("takeaway: block Levinson gives a classical MIMO AR/lattice baseline")

    _save_figures(
        true_coefficients=true_stack,
        estimated_coefficients=levinson.coefficients,
        reflection_norms=levinson.reflection_spectral_norms,
        prediction_error=levinson.prediction_error,
        residual_covariance=residual_covariance,
    )


if __name__ == "__main__":
    main()
