"""Tutorial: diagonal MIMO equals independent SISO filters.

A useful sanity check for MIMO DSP is the diagonal case.  If every matrix
Markov parameter or reflection matrix is diagonal, the MIMO system does not mix
channels; it is just several scalar SISO systems running side by side.

This tutorial performs two checks:

* five stable SISO lattice IIR filters are placed on the diagonal of a MIMO
  Markov sequence and compared with independent SISO filtering; and
* a three-channel online MIMO lattice predictor with diagonal reflection
  matrices is compared sample-by-sample with three independent one-channel
  online lattice predictors.
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


def diagonal_markov_from_siso(
    filters: list[tuple[np.ndarray, np.ndarray]], n_impulse: int
) -> np.ndarray:
    channels = len(filters)
    markov = np.zeros((n_impulse, channels, channels), dtype=float)
    for ch, (reflection, numerator) in enumerate(filters):
        denominator = ld.reflection_to_denominator(reflection.tolist())
        h = ld.iir_impulse_response(denominator, numerator.tolist(), n_impulse)
        markov[:, ch, ch] = np.asarray(h, dtype=float)
    return markov


def mimo_convolve(markov: np.ndarray, x: np.ndarray) -> np.ndarray:
    samples, inputs = x.shape
    _, outputs, markov_inputs = markov.shape
    if inputs != markov_inputs:
        raise ValueError("input dimension does not match Markov parameters")
    y = np.zeros((samples, outputs), dtype=float)
    horizon = markov.shape[0]
    for n in range(samples):
        max_lag = min(horizon, n + 1)
        for lag in range(max_lag):
            y[n] += markov[lag] @ x[n - lag]
    return y


def online_mimo_vs_independent_siso(
    x: np.ndarray,
    forward_scalars: np.ndarray,
    backward_scalars: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare a diagonal online MIMO lattice with independent one-channel predictors."""

    kf = np.asarray([np.diag(row) for row in forward_scalars], dtype=float)
    kb = np.asarray([np.diag(row) for row in backward_scalars], dtype=float)
    mimo = ld.MIMOLatticePredictor(kf, kb)
    siso = [
        ld.MIMOLatticePredictor(
            forward_scalars[:, ch].reshape(-1, 1, 1),
            backward_scalars[:, ch].reshape(-1, 1, 1),
        )
        for ch in range(x.shape[1])
    ]

    prediction_mimo = np.empty_like(x, dtype=float)
    prediction_siso = np.empty_like(x, dtype=float)
    error_mimo = np.empty_like(x, dtype=float)
    error_siso = np.empty_like(x, dtype=float)

    for n, sample in enumerate(x):
        # The prediction is intentionally requested before the current sample is
        # passed into update().  This is the causal one-step prediction contract.
        prediction_mimo[n] = mimo.predict().real
        error_mimo[n] = mimo.update(sample).real
        for ch, predictor in enumerate(siso):
            prediction_siso[n, ch] = predictor.predict()[0].real
            error_siso[n, ch] = predictor.update(np.array([sample[ch]]))[0].real

    return prediction_mimo, prediction_siso, error_mimo, error_siso


def main() -> None:
    out_dir = artifact_dir()
    rng = np.random.default_rng(2026)

    channels = 5
    samples = 700
    n_impulse = 160

    # Five different stable SISO lattice IIR filters.  Each channel has its own
    # reflection coefficients and numerator, but there is no cross-channel mixing.
    filters: list[tuple[np.ndarray, np.ndarray]] = []
    for ch in range(channels):
        scale = 0.58 - 0.04 * ch
        reflection = scale * np.array([0.55, -0.42, 0.28, -0.16], dtype=float)
        numerator = np.array([1.0, 0.08 * (ch + 1), -0.04, 0.015, 0.0], dtype=float)
        filters.append((reflection, numerator))

    x = rng.normal(size=(samples, channels))
    siso_y = np.zeros_like(x)
    for ch, (reflection, numerator) in enumerate(filters):
        filt = ld.LatticeIIR(reflection.tolist(), numerator.tolist())
        siso_y[:, ch] = filt.process(x[:, ch])

    markov = diagonal_markov_from_siso(filters, n_impulse=n_impulse)
    mimo_y = mimo_convolve(markov, x)

    # Ignore the very end of the impulse truncation error by using a long enough
    # impulse response.  With these stable filters, the residual is near roundoff.
    error = mimo_y - siso_y
    max_abs_error = float(np.max(np.abs(error)))
    rms_error = float(np.sqrt(np.mean(error**2)))
    off_diagonal_energy = float(np.sum(markov[:, ~np.eye(channels, dtype=bool)] ** 2))
    diagonal_energy = float(np.sum(markov[:, np.eye(channels, dtype=bool)] ** 2))

    rows = []
    for ch in range(channels):
        rows.append(
            {
                "channel": ch,
                "max_abs_error": float(np.max(np.abs(error[:, ch]))),
                "rms_error": float(np.sqrt(np.mean(error[:, ch] ** 2))),
                "max_abs_reflection": float(np.max(np.abs(filters[ch][0]))),
            }
        )

    csv_path = out_dir / "mimo_diagonal_equals_independent_siso_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    online_channels = 3
    online_x = rng.normal(size=(samples, online_channels))
    forward_scalars = np.array(
        [
            [0.18, -0.12, 0.07],
            [-0.05, 0.03, -0.02],
            [0.015, -0.01, 0.012],
        ],
        dtype=float,
    )
    backward_scalars = np.array(
        [
            [0.16, -0.10, 0.06],
            [-0.04, 0.02, -0.01],
            [0.012, -0.008, 0.010],
        ],
        dtype=float,
    )
    pred_mimo, pred_siso, err_mimo, err_siso = online_mimo_vs_independent_siso(
        online_x,
        forward_scalars,
        backward_scalars,
    )
    online_prediction_diff = pred_mimo - pred_siso
    online_error_diff = err_mimo - err_siso
    max_online_prediction_diff = float(np.max(np.abs(online_prediction_diff)))
    max_online_error_diff = float(np.max(np.abs(online_error_diff)))

    online_rows = []
    for ch in range(online_channels):
        online_rows.append(
            {
                "channel": ch,
                "max_abs_prediction_difference": float(
                    np.max(np.abs(online_prediction_diff[:, ch]))
                ),
                "max_abs_error_difference": float(np.max(np.abs(online_error_diff[:, ch]))),
                "max_abs_forward_reflection": float(np.max(np.abs(forward_scalars[:, ch]))),
                "max_abs_backward_reflection": float(np.max(np.abs(backward_scalars[:, ch]))),
            }
        )

    online_csv_path = out_dir / "mimo_diagonal_online_predictor_summary.csv"
    with online_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(online_rows[0]))
        writer.writeheader()
        writer.writerows(online_rows)

    print("channels:", channels)
    print("samples:", samples)
    print("impulse truncation length:", n_impulse)
    print("off-diagonal Markov energy:", f"{off_diagonal_energy:.3e}")
    print("diagonal Markov energy:", f"{diagonal_energy:.3e}")
    print("max |MIMO output - independent SISO output|:", f"{max_abs_error:.3e}")
    print("RMS error:", f"{rms_error:.3e}")
    print("online predictor channels:", online_channels)
    print(
        "max |online diagonal MIMO prediction - independent SISO prediction|:",
        f"{max_online_prediction_diff:.3e}",
    )
    print(
        "max |online diagonal MIMO error - independent SISO error|:", f"{max_online_error_diff:.3e}"
    )
    print(f"wrote {csv_path}")
    print(f"wrote {online_csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    t = np.arange(160)
    fig, axes = plt.subplots(channels, 1, figsize=(9, 8), sharex=True)
    for ch, ax in enumerate(axes):
        ax.plot(t, siso_y[: t.size, ch], label="independent SISO", linewidth=1.7)
        ax.plot(t, mimo_y[: t.size, ch], "--", label="diagonal MIMO", linewidth=1.2)
        ax.set_ylabel(f"ch {ch}")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Diagonal MIMO reproduces five independent SISO lattice IIR outputs")
    axes[-1].set_xlabel("sample")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig_path = out_dir / "mimo_diagonal_outputs.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {fig_path}")

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.4))
    for ch in range(channels):
        ax2.semilogy(np.abs(error[:, ch]) + 1e-18, label=f"channel {ch}")
    ax2.set_title("Absolute difference between diagonal MIMO and independent SISO outputs")
    ax2.set_xlabel("sample")
    ax2.set_ylabel("absolute error")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2)
    fig2.tight_layout()
    fig2_path = out_dir / "mimo_diagonal_error.png"
    fig2.savefig(fig2_path, dpi=160)
    print(f"wrote {fig2_path}")

    fig3, ax3 = plt.subplots(figsize=(8.5, 4.5))
    for ch in range(online_channels):
        ax3.semilogy(np.abs(online_prediction_diff[:, ch]) + 1e-18, label=f"channel {ch}")
    ax3.set_title("Online diagonal MIMO predictor equals independent one-channel predictors")
    ax3.set_xlabel("sample")
    ax3.set_ylabel("absolute prediction difference")
    ax3.grid(True, alpha=0.3)
    ax3.legend(ncol=online_channels)
    fig3.tight_layout()
    fig3_path = out_dir / "mimo_diagonal_online_prediction_error.png"
    fig3.savefig(fig3_path, dpi=160)
    print(f"wrote {fig3_path}")


if __name__ == "__main__":
    main()
