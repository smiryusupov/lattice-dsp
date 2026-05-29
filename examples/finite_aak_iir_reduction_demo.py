"""Tutorial: finite-section AAK/Nehari reduction of a stable SISO IIR filter.

The previous finite AAK/Nehari tutorials work directly with an abstract tail
sequence.  This tutorial closes the loop for DSP users: start with a stable
higher-order lattice/IIR filter, compute its impulse response, select a reduced
rational model with ``finite_aak_reduce_iir``, and compare impulse response,
frequency response, and filtering speed.

This is still a finite-section reduction candidate, not a full
infinite-dimensional AAK/Nehari solver.  Its purpose is practical: demonstrate
how the current finite Hankel/Schmidt-pair/rational workflow can produce a
stable lower-order IIR model that is cheaper to run.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import time
from pathlib import Path
from collections.abc import Callable

import numpy as np

import lattice_dsp as ld


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def impulse_from_poles(poles: np.ndarray, weights: np.ndarray, n_terms: int) -> np.ndarray:
    n = np.arange(n_terms, dtype=float)
    return np.sum(weights[:, None] * poles[:, None] ** n[None, :], axis=0)


def numerator_from_impulse_and_denominator(
    impulse: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    order = denominator.size - 1
    numerator = np.zeros(order + 1, dtype=float)
    for i in range(order + 1):
        numerator[i] = sum(float(denominator[j]) * float(impulse[i - j]) for j in range(i + 1))
    return numerator


def synthetic_high_order_iir() -> dict[str, np.ndarray]:
    """Return a stable full model whose impulse response has compressible modes."""

    poles = np.array([0.91, 0.72, -0.55, 0.38, -0.25, 0.14, -0.08, 0.03], dtype=float)
    weights = np.array([1.0, 0.25, -0.18, 0.07, -0.035, 0.015, -0.007, 0.003], dtype=float)
    denominator = np.asarray(np.poly(poles), dtype=float)
    impulse = impulse_from_poles(poles, weights, 512)
    numerator = numerator_from_impulse_and_denominator(impulse, denominator)
    reflection = np.asarray(ld.denominator_to_reflection(denominator.tolist()), dtype=float)
    return {
        "poles": poles,
        "weights": weights,
        "denominator": denominator,
        "numerator": numerator,
        "reflection": reflection,
        "impulse": impulse,
    }


def frequency_response(
    denominator: np.ndarray, numerator: np.ndarray, n_freq: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    w = np.linspace(0.0, math.pi, n_freq)
    z = np.exp(-1j * w)
    num = np.zeros_like(z, dtype=np.complex128)
    den = np.zeros_like(z, dtype=np.complex128)
    for k, coeff in enumerate(numerator):
        num += coeff * z**k
    for k, coeff in enumerate(denominator):
        den += coeff * z**k
    return w, num / den


def median_time(fn: Callable[[], np.ndarray], repeats: int) -> tuple[float, np.ndarray]:
    times: list[float] = []
    result: np.ndarray | None = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    assert result is not None
    return statistics.median(times), result


def process(reflection: np.ndarray, numerator: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray(ld.process_batch(reflection.tolist(), numerator.tolist(), x), dtype=float)


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    error = reference - estimate
    return 10.0 * math.log10(
        (float(np.mean(reference * reference)) + 1e-30) / (float(np.mean(error * error)) + 1e-30)
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = artifact_dir()
    model = synthetic_high_order_iir()
    rows = cols = 96
    n_impulse = 192
    ranks = [2, 3, 4, 5, 6, 8]
    criteria = ld.FiniteNehariCandidateCriteria(
        max_tail_error=1.0e-3,
        max_rational_error=5.0e-3,
        max_pole_radius=0.99,
    )

    reduction = ld.finite_aak_reduce_iir(
        model["reflection"],
        model["numerator"],
        ranks=ranks,
        n_impulse=n_impulse,
        rows=rows,
        cols=cols,
        criteria=criteria,
        attach_certificate=True,
    )

    selected = reduction["selected"]
    reduced_reflection = np.asarray(reduction["reduced_reflection"], dtype=float)
    reduced_numerator = np.asarray(reduction["reduced_numerator"], dtype=float)
    reduced_denominator = np.asarray(reduction["reduced_denominator"], dtype=float)

    rng = np.random.default_rng(123)
    channels = 32
    samples = 30000
    x = rng.normal(size=(channels, samples)).astype(np.float64)
    full_time, y_full = median_time(
        lambda: process(model["reflection"], model["numerator"], x), repeats=3
    )
    reduced_time, y_reduced = median_time(
        lambda: process(reduced_reflection, reduced_numerator, x), repeats=3
    )
    filter_speedup = full_time / reduced_time
    snr = snr_db(y_full, y_reduced)
    rel_mse = float(np.mean((y_full - y_reduced) ** 2) / (np.mean(y_full**2) + 1e-30))

    w, h_full = frequency_response(model["denominator"], model["numerator"])
    _, h_reduced = frequency_response(reduced_denominator, reduced_numerator)
    full_db = 20.0 * np.log10(np.maximum(np.abs(h_full), 1e-14))
    reduced_db = 20.0 * np.log10(np.maximum(np.abs(h_reduced), 1e-14))
    max_mag_error_db = float(np.max(np.abs(full_db - reduced_db)))

    summary_rows = []
    for row in reduction["candidates"]:
        summary_rows.append(
            {
                "rank": row["rank"],
                "sigma_next": row["sigma_next"],
                "hankelized_tail_error": row["hankelized_tail_error"],
                "rational_error": row["rational_error"],
                "max_pole_radius": row["max_pole_radius"],
                "accepted": row["accepted"],
            }
        )
    summary_rows.append(
        {
            "rank": "selected",
            "sigma_next": selected["sigma_next"],
            "hankelized_tail_error": reduction["relative_impulse_error"],
            "rational_error": selected["rational_error"],
            "max_pole_radius": selected["max_pole_radius"],
            "accepted": reduction["accepted"],
        }
    )
    csv_path = out_dir / "finite_aak_iir_reduction_summary.csv"
    write_csv(csv_path, summary_rows)

    print("full IIR order:", model["reflection"].size)
    print("finite Hankel matrix:", f"{rows} x {cols}")
    print("candidate ranks:", ranks)
    print("selected rank:", reduction["selected_rank"])
    print("selected accepted:", reduction["accepted"])
    print("selected pole radius:", f"{selected['max_pole_radius']:.4f}")
    print("relative impulse error:", f"{reduction['relative_impulse_error']:.3e}")
    print("batch output SNR:", f"{snr:.2f} dB")
    print("batch output rel MSE:", f"{rel_mse:.3e}")
    print("max magnitude error:", f"{max_mag_error_db:.3f} dB")
    print("full filter median time:", f"{full_time:.4f} s")
    print("reduced filter median time:", f"{reduced_time:.4f} s")
    print("filter speedup:", f"{filter_speedup:.2f}x")
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    singular_values = np.asarray(selected["hankel_singular_values"], dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(np.arange(1, 21), singular_values[:20], marker="o")
    ax.axvline(
        reduction["selected_rank"] + 1, linestyle="--", linewidth=1.0, label="first neglected index"
    )
    ax.set_title("Hankel singular values of the full IIR impulse response")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "finite_aak_iir_singular_values.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")

    fig2, ax2 = plt.subplots(figsize=(9.2, 5.0))
    n = np.arange(80)
    ax2.plot(n, reduction["full_impulse_response"][:80], label="full impulse", linewidth=2.0)
    ax2.plot(n, reduction["reduced_impulse_response"][:80], "--", label="reduced impulse")
    ax2.set_title("Full and selected finite AAK/Nehari IIR impulse responses")
    ax2.set_xlabel("sample")
    ax2.set_ylabel("amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    path2 = out_dir / "finite_aak_iir_impulse_response.png"
    fig2.savefig(path2, dpi=160)
    print(f"wrote {path2}")

    fig3, ax3 = plt.subplots(figsize=(9.2, 5.0))
    ax3.plot(w / math.pi, full_db, label="full IIR", linewidth=2.0)
    ax3.plot(w / math.pi, reduced_db, "--", label="selected reduced IIR")
    ax3.set_title("Magnitude response after finite AAK/Nehari IIR reduction")
    ax3.set_xlabel("normalized frequency ×π rad/sample")
    ax3.set_ylabel("magnitude (dB)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    path3 = out_dir / "finite_aak_iir_magnitude_response.png"
    fig3.savefig(path3, dpi=160)
    print(f"wrote {path3}")

    fig4, ax4 = plt.subplots(figsize=(5.8, 5.8))
    circle = plt.Circle((0.0, 0.0), 1.0, fill=False, linestyle="--")
    ax4.add_artist(circle)
    ax4.scatter(np.real(model["poles"]), np.imag(model["poles"]), label="full poles")
    ax4.scatter(
        np.real(selected["poles"]), np.imag(selected["poles"]), marker="x", label="reduced poles"
    )
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlim(-1.05, 1.05)
    ax4.set_ylim(-1.05, 1.05)
    ax4.set_xlabel("real")
    ax4.set_ylabel("imaginary")
    ax4.set_title("Stable poles before and after reduction")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    fig4.tight_layout()
    path4 = out_dir / "finite_aak_iir_poles.png"
    fig4.savefig(path4, dpi=160)
    print(f"wrote {path4}")


if __name__ == "__main__":
    main()
