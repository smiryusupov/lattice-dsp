"""Matrix AR spectral estimation from multichannel autocorrelation."""

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


def spectrum_from_response(h: np.ndarray, noise_cov: np.ndarray) -> np.ndarray:
    s = np.empty_like(h)
    for i, hi in enumerate(h):
        s[i] = hi @ noise_cov @ hi.conj().T
    return s


def _db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.real(values), 1e-14))


def _save_figures(
    w: np.ndarray, s_true: np.ndarray, s_est: np.ndarray, fit: ld.MultichannelARResult
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for channel in range(s_est.shape[1]):
        ax.plot(
            w, _db(s_true[:, channel, channel]), linestyle="--", label=f"true S{channel}{channel}"
        )
        ax.plot(w, _db(s_est[:, channel, channel]), label=f"estimated S{channel}{channel}")
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("auto spectrum (dB)")
    ax.set_title("Matrix AR auto-spectra")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "matrix_ar_auto_spectra.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(w, np.abs(s_true[:, 0, 1]), linestyle="--", label="true |S01|")
    ax.plot(w, np.abs(s_est[:, 0, 1]), label="estimated |S01|")
    ax.set_xlabel("rad/sample")
    ax.set_ylabel("cross-spectrum magnitude")
    ax.set_title("Cross-channel spectral coupling")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "matrix_ar_cross_spectrum.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    stages = np.arange(1, len(fit.reflection_spectral_norms) + 1)
    ax.plot(stages, fit.reflection_spectral_norms, marker="o")
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Levinson stage")
    ax.set_ylabel("reflection spectral norm")
    ax.set_title("Estimated matrix reflection norms")
    ax.set_xticks(stages)
    fig.tight_layout()
    path = out_dir / "matrix_ar_reflection_norms.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    true_coefficients = [
        np.array([[0.55, -0.08], [0.10, 0.38]]),
        np.array([[-0.22, 0.04], [-0.03, -0.14]]),
    ]
    x = simulate_var(true_coefficients, samples=60000, seed=9)
    r = ld.multichannel_autocorrelation(x, order=2)
    fit = ld.block_levinson_durbin(r, order=2)

    w = np.linspace(0, np.pi, 256)
    h_true = ld.matrix_ar_frequency_response(np.asarray(true_coefficients), w)
    h_est = ld.matrix_ar_frequency_response(fit.coefficients, w)
    s_true = spectrum_from_response(h_true, np.eye(2))
    s_est = spectrum_from_response(h_est, fit.prediction_error.real)

    rel_spectrum_error = np.linalg.norm(s_est - s_true) / np.linalg.norm(s_true)
    peak_bin = int(np.argmax(np.real(s_est[:, 0, 0])))

    print("channels:", 2)
    print("order:", fit.order)
    print("frequency bins:", len(w))
    print("companion spectral radius:", f"{ld.companion_spectral_radius(fit.coefficients):.6f}")
    print("relative spectral-matrix error:", f"{rel_spectrum_error:.3e}")
    print("channel-0 spectral peak rad/sample:", f"{w[peak_bin]:.4f}")
    print(
        "prediction error eigenvalues:", np.round(np.linalg.eigvalsh(fit.prediction_error).real, 6)
    )
    print("takeaway: multichannel Levinson supports matrix AR spectral estimates")

    _save_figures(w, s_true, s_est, fit)


if __name__ == "__main__":
    main()
