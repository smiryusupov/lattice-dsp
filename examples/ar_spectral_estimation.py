"""AR-style spectral estimation with reflection coefficients.

A stable all-pole process is generated from known reflection coefficients.  A
small adaptive predictor estimates a stable recursive model, then the true and
learned spectra are compared.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lattice_dsp import AdaptiveLatticeLadderNLMS, LatticeIIR, reflection_to_denominator


def response_db(
    numerator: np.ndarray, denominator: np.ndarray, n_fft: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    omega = np.linspace(0.0, np.pi, n_fft)
    z = np.exp(-1j * omega)
    b = sum(float(coef) * z**i for i, coef in enumerate(numerator))
    a = sum(float(coef) * z**i for i, coef in enumerate(denominator))
    h = b / np.maximum(np.abs(a), 1e-12)
    return omega / np.pi, 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))


def artifact_dir() -> Path:
    """Return the directory for generated figures/data."""

    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    rng = np.random.default_rng(33)
    samples = 10_000
    true_reflection = [0.72, -0.48, 0.24]
    innovation = rng.normal(scale=0.25, size=samples)
    x = np.asarray(
        LatticeIIR(true_reflection, [1.0, 0.0, 0.0, 0.0]).process(innovation), dtype=float
    )

    # One-step prediction: input is delayed signal, desired is current signal.
    predictor_input = np.r_[0.0, x[:-1]]
    predictor = AdaptiveLatticeLadderNLMS(
        initial_reflection=[0.0, 0.0, 0.0],
        initial_taps=[0.0, 0.0, 0.0, 0.0],
        mu_taps=0.04,
        mu_reflection=0.001,
        margin=1e-4,
    )
    error = np.asarray(predictor.adapt_block(predictor_input, x), dtype=float)

    true_den = np.asarray(reflection_to_denominator(true_reflection), dtype=float)
    learned_den = np.asarray(predictor.denominator, dtype=float)

    print("true reflection:", np.round(true_reflection, 4).tolist())
    print("learned reflection:", np.round(predictor.reflection, 4).tolist())
    print("initial prediction MSE:", float(np.mean(error[:1000] ** 2)))
    print("final prediction MSE:", float(np.mean(error[-1000:] ** 2)))

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    freq, true_db = response_db(np.array([1.0]), true_den)
    _, learned_db = response_db(np.array([1.0]), learned_den)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freq, true_db, label="true all-pole spectrum")
    ax.plot(freq, learned_db, "--", label="learned stable spectrum")
    ax.set_title("AR spectral estimate from reflection coefficients")
    ax.set_xlabel("normalized frequency × π rad/sample")
    ax.set_ylabel("magnitude (dB)")
    ax.legend()
    fig.tight_layout()
    out = artifact_dir() / "ar_spectral_estimation.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
