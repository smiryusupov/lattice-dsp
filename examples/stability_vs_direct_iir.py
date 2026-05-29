"""Why reflection-parameterized adaptive IIR is useful.

This example compares two toy adaptive IIR identifiers:

1. a deliberately simple direct-form denominator update; and
2. `AdaptiveLatticeLadderNLMS`, whose denominator is represented by bounded
   reflection/PARCOR coefficients.

The direct-form update is intentionally aggressive so the pole radius can drift
outside the unit circle.  This illustrates that the direct denominator
coefficients are numerically awkward stability coordinates: stability is a root
location condition, not a per-coefficient box constraint.

Even FIR LMS requires careful learning-rate selection: `mu` controls convergence
speed, misadjustment, and possible divergence.  Adaptive IIR keeps that tuning
problem and adds a structural stability problem because denominator updates move
poles.

The lattice update keeps `|k_i| < 1 - margin` by construction.  The example is
therefore about parameterization and stability enforcement, not about claiming
that this specific toy adaptation always has the lowest MSE.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lattice_dsp import AdaptiveLatticeLadderNLMS, LatticeIIR, reflection_to_denominator


def max_pole_radius(denominator: np.ndarray) -> float:
    """Return max absolute pole radius for A(z)=1+a1 z^-1+...+aN z^-N."""

    coeffs = np.asarray(denominator, dtype=float)
    if coeffs.ndim != 1 or coeffs.size < 2:
        return 0.0
    roots = np.roots(coeffs)
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def direct_form_identifier(
    x: np.ndarray,
    desired: np.ndarray,
    *,
    order: int = 2,
    mu_num: float = 0.03,
    mu_den: float = 0.08,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Tiny approximate-gradient adaptive direct-form IIR identifier.

    This is not a recommended algorithm.  It exists to show why unconstrained
    denominator updates are dangerous in adaptive IIR examples.
    """

    b = np.zeros(order + 1, dtype=float)
    a = np.zeros(order, dtype=float)  # denominator is [1, a1, a2, ...]
    x_hist = np.zeros(order + 1, dtype=float)
    y_hist = np.zeros(order, dtype=float)
    y = np.zeros_like(x, dtype=float)
    error = np.zeros_like(x, dtype=float)
    pole_radii: list[float] = []

    for n, sample in enumerate(x):
        x_hist[1:] = x_hist[:-1]
        x_hist[0] = sample
        y_n = float(np.dot(b, x_hist) - np.dot(a, y_hist))
        e_n = float(desired[n] - y_n)

        norm = float(np.dot(x_hist, x_hist) + np.dot(y_hist, y_hist) + epsilon)
        b += (mu_num * e_n / norm) * x_hist
        # Frozen-gradient update for y = b*x - a*y_history.
        # This can move the denominator outside the stability region.
        a -= (mu_den * e_n / norm) * y_hist

        y[n] = y_n
        error[n] = e_n
        y_hist[1:] = y_hist[:-1]
        y_hist[0] = y_n
        pole_radii.append(max_pole_radius(np.r_[1.0, a]))

        if not np.isfinite(y_n) or pole_radii[-1] > 1.25:
            error[n + 1 :] = np.nan
            pole_radii.extend([pole_radii[-1]] * (x.size - n - 1))
            break

    return y, error, pole_radii


def artifact_dir() -> Path:
    """Return the directory for generated figures/data."""

    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=10_000)

    target_reflection = [0.82, -0.55]
    target_numerator = [0.35, -0.05, 0.6]
    desired = np.asarray(LatticeIIR(target_reflection, target_numerator).process(x), dtype=float)

    target_den = np.asarray(reflection_to_denominator(target_reflection), dtype=float)
    target_radius = max_pole_radius(target_den)

    _, direct_error, direct_radii = direct_form_identifier(x, desired)

    lattice = AdaptiveLatticeLadderNLMS(
        initial_reflection=[0.0, 0.0],
        initial_taps=[0.0, 0.0, 0.0],
        mu_taps=0.06,
        mu_reflection=0.0015,
        margin=1e-4,
    )
    lattice_error = np.asarray(lattice.adapt_block(x, desired), dtype=float)
    learned_den = np.asarray(lattice.denominator, dtype=float)

    print("target denominator:", np.round(target_den, 5).tolist())
    print("target max pole radius:", round(target_radius, 5))
    print()
    print("direct-form max pole radius:", round(float(np.nanmax(direct_radii)), 5))
    print("direct-form final finite MSE:", float(np.nanmean(direct_error[-1000:] ** 2)))
    print()
    print("lattice learned reflection:", np.round(lattice.reflection, 5).tolist())
    print("lattice learned denominator:", np.round(learned_den, 5).tolist())
    print("lattice max pole radius:", round(max_pole_radius(learned_den), 5))
    print("lattice final MSE:", float(np.mean(lattice_error[-1000:] ** 2)))
    print()
    print("Takeaway: the lattice path may still need tuning, but denominator stability")
    print("is enforced by the reflection parameterization instead of hoped for after")
    print("a direct coefficient update.")
    print("A low finite-window MSE is not itself a stability certificate.")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(direct_radii, label="direct-form adaptive denominator")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="unit circle")
    ax.set_title("Direct denominator update can cross the stability boundary")
    ax.set_xlabel("sample")
    ax.set_ylabel("max pole radius")
    ax.legend()
    fig.tight_layout()
    out = artifact_dir() / "stability_vs_direct_iir.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
