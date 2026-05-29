"""Track a slowly drifting stable IIR system.

The target filter changes its reflection coefficients over time.  The adaptive
model updates numerator taps and bounded raw reflection parameters, so the
learned denominator remains stable while tracking the drift.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lattice_dsp import AdaptiveLatticeLadderNLMS, LatticeIIR


def artifact_dir() -> Path:
    """Return the directory for generated figures/data."""

    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    rng = np.random.default_rng(21)
    samples = 12_000
    n = np.arange(samples)
    x = rng.normal(size=samples)

    k1 = 0.35 + 0.18 * np.sin(2.0 * np.pi * n / samples)
    k2 = -0.28 + 0.10 * np.cos(2.0 * np.pi * n / samples)
    true_reflection_path = np.column_stack([k1, k2])
    numerator = [0.45, -0.1, 0.55]

    target = LatticeIIR(true_reflection_path[0].tolist(), numerator)
    adaptive = AdaptiveLatticeLadderNLMS(
        initial_reflection=[0.0, 0.0],
        initial_taps=[0.0, 0.0, 0.0],
        mu_taps=0.04,
        mu_reflection=0.001,
        margin=1e-4,
        reflection_update_period=4,
        scale_reflection_mu_by_period=True,
    )

    desired = np.zeros(samples, dtype=float)
    error = np.zeros(samples, dtype=float)
    learned = np.zeros_like(true_reflection_path)

    for i in range(samples):
        target.set_reflection_preserve_state(true_reflection_path[i].tolist())
        desired[i] = target.process_sample(float(x[i]))
        _, error[i] = adaptive.adapt_sample(float(x[i]), float(desired[i]))
        learned[i] = np.asarray(adaptive.reflection, dtype=float)

    print("true final reflection:", np.round(true_reflection_path[-1], 4).tolist())
    print("learned final reflection:", np.round(learned[-1], 4).tolist())
    print("initial MSE:", float(np.mean(error[:1000] ** 2)))
    print("final MSE:", float(np.mean(error[-1000:] ** 2)))
    print("minimum stability margin:", float(1.0 - np.max(np.abs(learned))))

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(true_reflection_path[:, 0], label="true k1")
    ax.plot(true_reflection_path[:, 1], label="true k2")
    ax.plot(learned[:, 0], "--", label="learned k1")
    ax.plot(learned[:, 1], "--", label="learned k2")
    ax.set_title("Tracking a drifting stable IIR system")
    ax.set_xlabel("sample")
    ax.set_ylabel("reflection coefficient")
    ax.legend(ncol=2)
    fig.tight_layout()
    out = artifact_dir() / "tracking_drifting_iir_system.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
