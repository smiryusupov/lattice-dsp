"""Algorithm-selection map for lattice-dsp.

This example is an onboarding companion to ``docs/theory/choosing_algorithms``.
It prints a compact decision table and writes the same information as CSV so the
example gallery has a concrete artifact.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import lattice_dsp as ld


@dataclass(frozen=True)
class Choice:
    goal: str
    recommended_api: str
    first_diagnostic: str
    scope: str


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def choices() -> list[Choice]:
    return [
        Choice(
            "stable scalar IIR filtering",
            "LatticeIIR, reflection_to_denominator",
            "reflection bounds and pole radius",
            "stable scalar recursive filtering",
        ),
        Choice(
            "stable IIR with a numerator",
            "LatticeLadderIIR, numerator_to_ladder",
            "sample-by-sample agreement with B(z)/A(z)",
            "lattice denominator plus ladder numerator",
        ),
        Choice(
            "adaptive recursive system identification",
            "AdaptiveLatticeLadderNLMS, LatticeLadderRLS",
            "learning curve, final MSE, learned pole radius",
            "controlled online identification examples",
        ),
        Choice(
            "AR spectral estimation",
            "burg_denominator, levinson_durbin_denominator",
            "prediction error and spectrum residuals",
            "stationary AR modeling diagnostics",
        ),
        Choice(
            "finite SISO model reduction",
            "hankel_singular_values, finite_hankel_reduce_iir",
            "Hankel singular-value decay and response error",
            "finite-Hankel/Ho-Kalman baseline",
        ),
        Choice(
            "finite Nehari/AAK-style diagnostics",
            "finite_aak_siso_certificate, finite_aak_reduce_iir",
            "tail error, rational error, pole radius",
            "finite-section candidate workflow",
        ),
        Choice(
            "MIMO state-space reduction",
            "mimo_state_space_markov_response, finite_hankel_reduce_mimo",
            "block-Hankel singular values and state-space response error",
            "finite block-Hankel baseline",
        ),
        Choice(
            "matrix-lattice all-pass experiments",
            "MatrixLatticeAllPass, contractive_matrix_from_raw",
            "frequency-response singular values",
            "all-pass/paraunitary tutorial scaffold",
        ),
    ]


def write_csv(path: Path, rows: list[Choice]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["goal", "recommended_api", "first_diagnostic", "scope"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def smoke_checks() -> dict[str, object]:
    """Run tiny checks that touch representative public APIs."""

    reflection = np.array([0.55, -0.25], dtype=float)
    denominator = np.asarray(ld.reflection_to_denominator(reflection), dtype=float)
    impulse = np.asarray(ld.iir_impulse_response(denominator, [1.0], 48), dtype=float)
    hsv = np.asarray(ld.hankel_singular_values(impulse, rows=12, cols=12), dtype=float)

    autocorr = ld.autocorrelation(impulse, 4)
    ar_den = np.asarray(ld.levinson_durbin_denominator(autocorr, 2), dtype=float)

    return {
        "denominator": np.round(denominator, 6).tolist(),
        "leading_hankel_sv": np.round(hsv[:3], 6).tolist(),
        "levinson_denominator": np.round(ar_den, 6).tolist(),
    }


def main() -> None:
    rows = choices()
    print("lattice-dsp algorithm-selection map")
    print("=" * 40)
    for i, row in enumerate(rows, start=1):
        print(f"{i}. {row.goal}")
        print(f"   use: {row.recommended_api}")
        print(f"   diagnostic: {row.first_diagnostic}")
        print(f"   scope: {row.scope}")

    checks = smoke_checks()
    print("\nrepresentative tiny smoke checks")
    for key, value in checks.items():
        print(f"{key}: {value}")

    out = artifact_dir() / "algorithm_selection_map.csv"
    write_csv(out, rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
