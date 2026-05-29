"""Online causal MIMO lattice prediction from block Levinson reflections."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def _artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def simulate_var(coefficients: np.ndarray, samples: int = 12000, seed: int = 31) -> np.ndarray:
    """Generate a stable coupled vector AR process."""

    rng = np.random.default_rng(seed)
    order, channels, _ = coefficients.shape
    x = np.zeros((samples + 512, channels), dtype=np.float64)
    noise = rng.normal(scale=0.35, size=x.shape)
    for n in range(order, x.shape[0]):
        value = noise[n].copy()
        for lag in range(1, order + 1):
            value -= coefficients[lag - 1] @ x[n - lag]
        x[n] = value
    return x[512:]


def _normalized_covariance(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=0, keepdims=True)
    cov = centered.T @ centered.conj() / max(x.shape[0] - 1, 1)
    scale = np.sqrt(np.outer(np.real(np.diag(cov)), np.real(np.diag(cov)))) + 1e-30
    return np.real(cov / scale)


def _save_figures(
    *,
    x: np.ndarray,
    prediction: np.ndarray,
    error: np.ndarray,
    forward_norms: np.ndarray,
    backward_norms: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()
    n_view = min(240, x.shape[0])

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.plot(np.arange(n_view), x[:n_view, 0], label="observed channel 0")
    ax.plot(np.arange(n_view), prediction[:n_view, 0].real, label="one-step prediction")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
    ax.set_title("Causal MIMO lattice prediction uses only previous vectors")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "causal_mimo_lattice_prediction_trace.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    stages = np.arange(1, len(forward_norms) + 1)
    ax.plot(stages, forward_norms, marker="o", label="forward K")
    ax.plot(stages, backward_norms, marker="s", label="backward L")
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("lattice stage")
    ax.set_ylabel("spectral norm")
    ax.set_title("Matrix reflection norms for online MIMO prediction")
    ax.set_xticks(stages)
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "causal_mimo_lattice_reflection_norms.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    input_cov = _normalized_covariance(x)
    error_cov = _normalized_covariance(error)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, title, matrix in (
        (axes[0], "input normalized covariance", input_cov),
        (axes[1], "prediction-error normalized covariance", error_cov),
    ):
        im = ax.imshow(matrix, vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("channel")
        ax.set_ylabel("channel")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    path = out_dir / "causal_mimo_lattice_residual_covariance.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    true_coefficients = np.asarray(
        [
            [[0.34, 0.08, -0.03], [-0.05, 0.30, 0.06], [0.02, -0.06, 0.27]],
            [[-0.12, 0.03, 0.01], [0.02, -0.10, -0.02], [0.00, 0.04, -0.08]],
        ],
        dtype=np.float64,
    )
    x = simulate_var(true_coefficients)
    order = true_coefficients.shape[0]

    # Offline/batch estimation step: obtain matrix reflection coefficients from a finite record.
    r = ld.multichannel_autocorrelation(x, order=order)
    levinson = ld.block_levinson_durbin(r, order=order)

    # Online/runtime step: predict each vector before updating the state with that vector.
    predictor = ld.MIMOLatticePredictor.from_levinson(levinson)
    prediction, error = predictor.process(x)
    direct_error = ld.multichannel_prediction_error(x, levinson.coefficients)
    online_direct_difference = np.linalg.norm(error[order:] - direct_error) / max(
        np.linalg.norm(direct_error), 1e-30
    )

    input_cov = np.cov(x.T)
    error_cov = np.cov(error[order:].real.T)
    input_offdiag = (np.sum(np.abs(input_cov)) - np.trace(np.abs(input_cov))) / (
        x.shape[1] * (x.shape[1] - 1)
    )
    error_offdiag = (np.sum(np.abs(error_cov)) - np.trace(np.abs(error_cov))) / (
        x.shape[1] * (x.shape[1] - 1)
    )

    print("channels:", x.shape[1])
    print("order:", order)
    print(
        "companion spectral radius:", f"{ld.companion_spectral_radius(levinson.coefficients):.6f}"
    )
    print("forward reflection norms:", np.round(levinson.reflection_spectral_norms, 6))
    print("backward reflection norms:", np.round(levinson.backward_reflection_spectral_norms, 6))
    print("online lattice/direct AR residual difference:", f"{online_direct_difference:.3e}")
    print(
        "input/error mean absolute off-diagonal covariance:",
        f"{input_offdiag:.4f}",
        f"{error_offdiag:.4f}",
    )
    print(
        "takeaway: after batch coefficient estimation, the MIMO lattice predictor is causal and online"
    )

    _save_figures(
        x=x,
        prediction=prediction,
        error=error[order:],
        forward_norms=levinson.reflection_spectral_norms,
        backward_norms=levinson.backward_reflection_spectral_norms,
    )


if __name__ == "__main__":
    main()
