"""Streaming multichannel audio decorrelation with a real matrix-lattice all-pass filter.

A real-coefficient matrix lattice all-pass filter can redistribute energy across
channels and frequency while preserving total power.  This example uses the
causal online runtime, so each output frame depends only on the current input
frame and previous lattice states.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lattice_dsp import MatrixLatticeAllPass, contractive_matrix_from_raw, unitary_polar_factor


def _artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_real_filter(rng: np.random.Generator, channels: int, order: int) -> MatrixLatticeAllPass:
    reflections = [
        contractive_matrix_from_raw(0.45 * rng.normal(size=(channels, channels)))
        for _ in range(order)
    ]
    residue = unitary_polar_factor(rng.normal(size=(channels, channels)))
    return MatrixLatticeAllPass(reflections, residue=residue)


def _apply_real_streaming_filter(x: np.ndarray, filt: MatrixLatticeAllPass) -> np.ndarray:
    runtime = filt.to_online_filter()
    y = runtime.process(x)
    return np.real_if_close(y, tol=1000).real


def _mean_abs_off_diagonal_correlation(x: np.ndarray) -> float:
    corr = np.corrcoef(x, rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(np.abs(upper)))


def _save_figures(x: np.ndarray, y: np.ndarray, input_corr: float, output_corr: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()
    corr_x = np.corrcoef(x, rowvar=False)
    corr_y = np.corrcoef(y, rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, title, corr in (
        (axes[0], f"input correlation\nmean |offdiag|={input_corr:.3f}", corr_x),
        (axes[1], f"output correlation\nmean |offdiag|={output_corr:.3f}", corr_y),
    ):
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("channel")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.tight_layout()
    path = out_dir / "multichannel_audio_decorrelator_correlation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    segment = slice(0, 700)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x[segment, 0], label="input ch0")
    ax.plot(y[segment, 0], linestyle="--", label="output ch0")
    ax.set_xlabel("sample")
    ax.set_ylabel("normalized amplitude")
    ax.set_title("Same-energy decorrelation changes waveform shape")
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "multichannel_audio_decorrelator_waveform.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(["input", "output"], [input_corr, output_corr])
    ax.set_ylabel("mean |off-diagonal correlation|")
    ax.set_title("Correlation decreases while total energy is preserved")
    fig.tight_layout()
    path = out_dir / "multichannel_audio_decorrelator_corr_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


rng = np.random.default_rng(4)
channels = 4
order = 5
n_samples = 8192

# A smooth shared source creates highly correlated channels.  Small delays and
# independent noise make it more realistic than exact copies.
white = rng.normal(size=n_samples + 128)
source = np.convolve(white, np.ones(64) / 64.0, mode="valid")[:n_samples]
input_channels = []
for ch in range(channels):
    delayed = np.roll(source, 3 * ch)
    input_channels.append(0.9 * delayed + 0.1 * rng.normal(size=n_samples))
x = np.stack(input_channels, axis=1)
x = (x - x.mean(axis=0)) / x.std(axis=0)

filt = _make_real_filter(rng, channels, order)
y = _apply_real_streaming_filter(x, filt)
y = (y - y.mean(axis=0)) / y.std(axis=0)

input_corr = _mean_abs_off_diagonal_correlation(x)
output_corr = _mean_abs_off_diagonal_correlation(y)
energy_ratio = float(np.sum(y * y) / np.sum(x * x))

print("channels:", channels)
print("order:", order)
print("max reflection singular value:", round(filt.max_reflection_singular_value(), 6))
print("input mean |offdiag corr|:", round(input_corr, 4))
print("output mean |offdiag corr|:", round(output_corr, 4))
print("decorrelation factor:", round(input_corr / output_corr, 2), "x")
print("normalized energy ratio:", f"{energy_ratio:.6f}")
print(
    "takeaway: causal MIMO all-pass filtering can decorrelate channels without using future samples"
)

_save_figures(x, y, input_corr, output_corr)
