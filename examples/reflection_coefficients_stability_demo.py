"""Reflection coefficients as a stability coordinate system.

This example complements ``stability_vs_direct_iir.py``.  Instead of an adaptive
failure case, it shows the static parameterization: scalar all-pole denominators
built from reflection/PARCOR coefficients with ``|k_i| < 1`` have poles inside
the unit disk, while direct denominator coefficients do not expose such a simple
per-coefficient stability rule.
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


def max_pole_radius(denominator: np.ndarray) -> float:
    roots = np.roots(np.asarray(denominator, dtype=float))
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "parameter_type",
                "parameters",
                "max_abs_reflection",
                "max_pole_radius",
                "stable",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = artifact_dir()

    reflection_cases = {
        "mild reflection vector": np.array([0.20, -0.30, 0.10]),
        "moderate reflection vector": np.array([0.75, -0.45, 0.25]),
        "near-boundary reflection vector": np.array([0.95, -0.80, 0.65]),
    }
    direct_denominators = {
        "stable direct denominator": np.array([1.0, -1.85, 0.95]),
        "nearby unstable direct denominator": np.array([1.0, -2.05, 1.05]),
        "another unstable direct denominator": np.array([1.0, 0.0, 1.05]),
    }

    rows: list[dict[str, object]] = []
    for label, k in reflection_cases.items():
        den = np.asarray(ld.reflection_to_denominator(k), dtype=float)
        radius = max_pole_radius(den)
        rows.append(
            {
                "label": label,
                "parameter_type": "reflection",
                "parameters": np.round(k, 5).tolist(),
                "max_abs_reflection": float(np.max(np.abs(k))),
                "max_pole_radius": radius,
                "stable": radius < 1.0,
            }
        )

    for label, den in direct_denominators.items():
        radius = max_pole_radius(den)
        try:
            reflection = np.asarray(ld.denominator_to_reflection(den), dtype=float)
            max_abs_reflection = float(np.max(np.abs(reflection)))
        except Exception:
            max_abs_reflection = float("nan")
        rows.append(
            {
                "label": label,
                "parameter_type": "direct denominator",
                "parameters": np.round(den, 5).tolist(),
                "max_abs_reflection": max_abs_reflection,
                "max_pole_radius": radius,
                "stable": radius < 1.0,
            }
        )

    print("reflection coefficients and scalar IIR stability")
    print("=" * 52)
    for row in rows:
        print(f"{row['label']}:")
        print(f"  type: {row['parameter_type']}")
        print(f"  parameters: {row['parameters']}")
        print(f"  max |reflection|: {row['max_abs_reflection']:.6g}")
        print(f"  max pole radius: {row['max_pole_radius']:.6g}")
        print(f"  stable: {row['stable']}")

    csv_path = out_dir / "reflection_coefficients_stability_summary.csv"
    write_rows(csv_path, rows)
    print(f"\nwrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    labels = [str(row["label"]) for row in rows]
    pole_radii = [float(row["max_pole_radius"]) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x, pole_radii)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_title("Pole radius from reflection versus direct denominator examples")
    ax.set_ylabel("max pole radius")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig_path = out_dir / "reflection_coefficients_stability_pole_radius.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
