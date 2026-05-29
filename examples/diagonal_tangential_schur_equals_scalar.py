"""Diagonal tangential Schur data reduce to independent scalar Pick tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import lattice_dsp as ld


def _artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figures(full_pick: np.ndarray, block_error: float, channel_min_eigs: list[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib is not installed; skipped figures")
        return

    out_dir = _artifact_dir()

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(np.abs(full_pick))
    ax.set_xlabel("condition")
    ax.set_ylabel("condition")
    ax.set_title("Diagonal tangential data produce a block Pick matrix")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(out_dir / "diagonal_tangential_pick_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(np.arange(1, len(channel_min_eigs) + 1), channel_min_eigs)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("channel")
    ax.set_ylabel("minimum scalar Pick eigenvalue")
    ax.set_title(f"Independent scalar feasibility checks; off-block max={block_error:.1e}")
    fig.tight_layout()
    fig.savefig(out_dir / "diagonal_scalar_pick_min_eigenvalues.png", dpi=160)
    plt.close(fig)


channels = 3
points_per_channel = 3
gains = np.array([0.25, -0.40 + 0.05j, 0.35j], dtype=np.complex128)
points_by_channel = [
    np.array([-0.20, 0.05 + 0.10j, 0.26j]),
    np.array([0.00, 0.18 - 0.12j, -0.30j]),
    np.array([0.12, -0.22 + 0.05j, 0.32j]),
]

points: list[complex] = []
directions: list[np.ndarray] = []
values: list[np.ndarray] = []
for channel in range(channels):
    basis = np.eye(channels, dtype=np.complex128)[channel]
    for z in points_by_channel[channel]:
        points.append(complex(z))
        directions.append(basis.copy())
        values.append(gains[channel] * basis)

full_data = ld.RightTangentialSchurData(np.array(points), np.array(directions), np.array(values))
full_pick = ld.right_tangential_pick_matrix(full_data)

# Since the conditions are ordered by channel and directions are coordinate
# vectors, the full Pick matrix should be block diagonal with scalar Pick blocks.
channel_min_eigs: list[float] = []
block_diag = np.zeros_like(full_pick)
for channel in range(channels):
    sl = slice(channel * points_per_channel, (channel + 1) * points_per_channel)
    scalar_values = np.full((points_per_channel, 1), gains[channel], dtype=np.complex128)
    scalar_data = ld.RightTangentialSchurData(
        points_by_channel[channel], np.ones((points_per_channel, 1)), scalar_values
    )
    scalar_pick = ld.right_tangential_pick_matrix(scalar_data)
    block_diag[sl, sl] = scalar_pick
    channel_min_eigs.append(float(ld.pick_matrix_eigenvalues(scalar_pick)[0]))

block_error = float(np.max(np.abs(full_pick - block_diag)))
constant_solution = np.diag(gains)
residual = ld.max_tangential_residual(full_data, constant_solution)

print("channels:", channels)
print("points per channel:", points_per_channel)
print("total tangential conditions:", full_data.total_conditions)
print("diagonal Schur max singular value:", f"{np.max(np.abs(gains)):.6f}")
print("full Pick min eigenvalue:", f"{ld.pick_matrix_eigenvalues(full_pick)[0]:.6e}")
print("max |full Pick - scalar block diagonal Pick|:", f"{block_error:.3e}")
print("constant diagonal interpolation residual:", f"{residual:.3e}")
print("scalar channel min eigenvalues:", np.round(channel_min_eigs, 6))
print(
    "interpretation: diagonal MIMO tangential data decompose into independent scalar Schur/Pick problems"
)

_save_figures(full_pick, block_error, channel_min_eigs)
