"""Online coupled MIMO prediction versus independent SISO baselines.

The diagonal-MIMO tutorial checks that matrix lattice prediction reduces to
independent one-channel prediction when all reflection matrices are diagonal.
This tutorial checks the complementary case: when the training signal has true
cross-channel dynamics, a full online MIMO lattice predictor can use those
off-diagonal terms while independent SISO predictors cannot.
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


def simulate_coupled_var(coefficients: np.ndarray, samples: int, seed: int = 71) -> np.ndarray:
    """Generate a stable coupled vector AR process."""

    rng = np.random.default_rng(seed)
    order, channels, _ = coefficients.shape
    burn_in = 512
    x = np.zeros((samples + burn_in, channels), dtype=np.float64)
    noise = rng.normal(scale=0.30, size=x.shape)
    for n in range(order, x.shape[0]):
        value = noise[n].copy()
        for lag in range(1, order + 1):
            value -= coefficients[lag - 1] @ x[n - lag]
        x[n] = value
    return x[burn_in:]


def normalized_covariance(x: np.ndarray) -> np.ndarray:
    centered = np.asarray(x, dtype=np.float64) - np.mean(x, axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov))) + 1e-30
    return cov / scale


def mean_abs_offdiag(matrix: np.ndarray) -> float:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(np.abs(matrix[mask])))


def fit_full_mimo_predictor(
    train: np.ndarray, order: int
) -> tuple[ld.MultichannelARResult, ld.MIMOLatticePredictor]:
    r = ld.multichannel_autocorrelation(train, order=order)
    result = ld.block_levinson_durbin(r, order=order)
    return result, ld.MIMOLatticePredictor.from_levinson(result)


def fit_independent_siso_predictors(train: np.ndarray, order: int) -> list[ld.MIMOLatticePredictor]:
    predictors: list[ld.MIMOLatticePredictor] = []
    for ch in range(train.shape[1]):
        r = ld.multichannel_autocorrelation(train[:, [ch]], order=order)
        result = ld.block_levinson_durbin(r, order=order)
        predictors.append(ld.MIMOLatticePredictor.from_levinson(result))
    return predictors


def process_independent_siso(
    predictors: list[ld.MIMOLatticePredictor], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.empty_like(x, dtype=np.float64)
    error = np.empty_like(x, dtype=np.float64)
    for n, sample in enumerate(x):
        for ch, predictor in enumerate(predictors):
            prediction[n, ch] = predictor.predict()[0].real
            error[n, ch] = predictor.update(np.array([sample[ch]]))[0].real
    return prediction, error


def diagonal_ablation_predictor(result: ld.MultichannelARResult) -> ld.MIMOLatticePredictor:
    """Keep only per-channel reflection entries from a full MIMO fit."""

    kf = np.asarray([np.diag(np.diag(stage)) for stage in result.reflection], dtype=np.complex128)
    if result.backward_reflection is None:
        raise ValueError("block-Levinson result does not contain backward reflections")
    kb = np.asarray(
        [np.diag(np.diag(stage)) for stage in result.backward_reflection], dtype=np.complex128
    )
    return ld.MIMOLatticePredictor(kf, kb)


def residual_rms(error: np.ndarray, warmup: int) -> float:
    return float(np.sqrt(np.mean(np.asarray(error[warmup:]).real ** 2)))


def residual_rms_by_channel(error: np.ndarray, warmup: int) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(error[warmup:]).real ** 2, axis=0))


def save_summary_csv(
    out_dir: Path,
    *,
    warmup: int,
    full_error: np.ndarray,
    diagonal_error: np.ndarray,
    siso_error: np.ndarray,
) -> Path:
    rows: list[dict[str, object]] = []
    errors = {
        "full_mimo": full_error,
        "diagonal_ablation": diagonal_error,
        "independent_siso": siso_error,
    }
    for name, err in errors.items():
        by_channel = residual_rms_by_channel(err, warmup)
        cov = normalized_covariance(err[warmup:].real)
        for ch, rms in enumerate(by_channel):
            rows.append(
                {
                    "model": name,
                    "channel": ch,
                    "residual_rms": float(rms),
                    "mean_abs_offdiag_residual_correlation": mean_abs_offdiag(cov),
                }
            )
    path = out_dir / "online_coupled_mimo_vs_siso_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_figures(
    out_dir: Path,
    *,
    true_coefficients: np.ndarray,
    test: np.ndarray,
    full_prediction: np.ndarray,
    siso_prediction: np.ndarray,
    full_error: np.ndarray,
    diagonal_error: np.ndarray,
    siso_error: np.ndarray,
    warmup: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    n_view = min(260, test.shape[0])
    t = np.arange(n_view)
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    ax.plot(t, test[:n_view, 0], label="observed channel 0", linewidth=1.6)
    ax.plot(t, full_prediction[:n_view, 0].real, label="full MIMO prediction", linewidth=1.3)
    ax.plot(
        t,
        siso_prediction[:n_view, 0].real,
        "--",
        label="independent SISO prediction",
        linewidth=1.2,
    )
    ax.set_xlabel("test sample")
    ax.set_ylabel("amplitude")
    ax.set_title("Online prediction: full MIMO can use cross-channel history")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out_dir / "online_coupled_mimo_prediction_trace.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    labels = ["full MIMO", "diagonal\nablation", "independent\nSISO"]
    values = [
        residual_rms(full_error, warmup),
        residual_rms(diagonal_error, warmup),
        residual_rms(siso_error, warmup),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("residual RMS")
    ax.set_title("Coupled online MIMO prediction reduces residual energy")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    path = out_dir / "online_coupled_mimo_rms_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    matrices = [
        ("full MIMO residual", normalized_covariance(full_error[warmup:].real)),
        ("independent SISO residual", normalized_covariance(siso_error[warmup:].real)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax, (title, matrix) in zip(axes, matrices, strict=True):
        im = ax.imshow(matrix, vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("channel")
        ax.set_ylabel("channel")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.84)
    path = out_dir / "online_coupled_mimo_residual_covariance.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    order = true_coefficients.shape[0]
    fig, axes = plt.subplots(1, order, figsize=(4.2 * order, 3.8))
    if order == 1:
        axes = [axes]
    max_abs = float(np.max(np.abs(true_coefficients)))
    for lag, ax in enumerate(axes):
        im = ax.imshow(true_coefficients[lag], vmin=-max_abs, vmax=max_abs)
        ax.set_title(f"true A[{lag + 1}]")
        ax.set_xlabel("source channel")
        ax.set_ylabel("target channel")
    fig.colorbar(im, ax=np.ravel(axes).tolist(), shrink=0.84)
    path = out_dir / "online_coupled_mimo_coefficient_matrices.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    out_dir = artifact_dir()

    # Stable coupled VAR(2).  The off-diagonal entries are the part independent
    # SISO predictors cannot represent.
    true_coefficients = np.asarray(
        [
            [[0.55, 0.30, 0.00], [-0.25, 0.45, 0.22], [0.18, -0.12, 0.40]],
            [[-0.18, 0.08, 0.02], [0.05, -0.14, -0.05], [-0.03, 0.07, -0.10]],
        ],
        dtype=np.float64,
    )
    order, channels, _ = true_coefficients.shape
    train_samples = 6000
    test_samples = 2200
    x = simulate_coupled_var(true_coefficients, train_samples + test_samples)
    train = x[:train_samples]
    test = x[train_samples:]
    warmup = order

    full_result, full_predictor = fit_full_mimo_predictor(train, order)
    full_prediction, full_error = full_predictor.process(test)

    diagonal_predictor = diagonal_ablation_predictor(full_result)
    diagonal_prediction, diagonal_error = diagonal_predictor.process(test)

    siso_predictors = fit_independent_siso_predictors(train, order)
    siso_prediction, siso_error = process_independent_siso(siso_predictors, test)

    full_rms = residual_rms(full_error, warmup)
    diagonal_rms = residual_rms(diagonal_error, warmup)
    siso_rms = residual_rms(siso_error, warmup)
    relative_improvement = (siso_rms - full_rms) / max(siso_rms, 1e-30)

    full_cov = normalized_covariance(full_error[warmup:].real)
    siso_cov = normalized_covariance(siso_error[warmup:].real)
    full_offdiag = mean_abs_offdiag(full_cov)
    siso_offdiag = mean_abs_offdiag(siso_cov)
    offdiag_reduction = (siso_offdiag - full_offdiag) / max(siso_offdiag, 1e-30)

    csv_path = save_summary_csv(
        out_dir,
        warmup=warmup,
        full_error=full_error,
        diagonal_error=diagonal_error,
        siso_error=siso_error,
    )

    print("channels:", channels)
    print("order:", order)
    print("training samples:", train_samples)
    print("test samples:", test_samples)
    print(
        "true companion spectral radius:", f"{ld.companion_spectral_radius(true_coefficients):.6f}"
    )
    print(
        "fitted companion spectral radius:",
        f"{ld.companion_spectral_radius(full_result.coefficients):.6f}",
    )
    print("full MIMO reflection norms:", np.round(full_result.reflection_spectral_norms, 6))
    print("full MIMO residual RMS:", f"{full_rms:.6f}")
    print("diagonal-ablation residual RMS:", f"{diagonal_rms:.6f}")
    print("independent SISO residual RMS:", f"{siso_rms:.6f}")
    print("relative RMS improvement vs independent SISO:", f"{100.0 * relative_improvement:.2f}%")
    print("mean abs off-diagonal residual correlation, full MIMO:", f"{full_offdiag:.6f}")
    print("mean abs off-diagonal residual correlation, independent SISO:", f"{siso_offdiag:.6f}")
    print(
        "off-diagonal residual correlation reduction vs independent SISO:",
        f"{100.0 * offdiag_reduction:.2f}%",
    )
    print("causal contract: prediction is requested before update(y_n) for every test vector")
    print(f"wrote {csv_path}")

    save_figures(
        out_dir,
        true_coefficients=true_coefficients,
        test=test,
        full_prediction=full_prediction,
        siso_prediction=siso_prediction,
        full_error=full_error,
        diagonal_error=diagonal_error,
        siso_error=siso_error,
        warmup=warmup,
    )


if __name__ == "__main__":
    main()
