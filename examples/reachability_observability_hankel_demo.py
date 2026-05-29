"""Reachability, observability, and Hankel singular values.

This example shows why Hankel singular values are useful for model reduction.
It builds a small state-space system with one unreachable state direction and
one unobservable state direction.  The input-output Hankel matrix only sees the
state directions that are both reachable and observable.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def reachability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])


def finite_gramians(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, horizon: int = 160
) -> tuple[np.ndarray, np.ndarray]:
    Wc = np.zeros((A.shape[0], A.shape[0]), dtype=float)
    Wo = np.zeros_like(Wc)
    Ak = np.eye(A.shape[0])
    for _ in range(horizon):
        Wc += Ak @ B @ B.T @ Ak.T
        Wo += Ak.T @ C.T @ C @ Ak
        Ak = Ak @ A
    return Wc, Wo


def markov_parameters(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, n_samples: int
) -> np.ndarray:
    out = np.empty(n_samples, dtype=float)
    out[0] = float(D[0, 0])
    for k in range(1, n_samples):
        out[k] = float((C @ np.linalg.matrix_power(A, k - 1) @ B)[0, 0])
    return out


def hankel_from_impulse(impulse: np.ndarray, rows: int, cols: int, offset: int = 1) -> np.ndarray:
    return np.array(
        [[impulse[i + j + offset] for j in range(cols)] for i in range(rows)], dtype=float
    )


def write_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["quantity", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = artifact_dir()

    # Four states, but only two state directions are both reachable and observable.
    A = np.diag([0.82, 0.55, 0.25, 0.12])
    B = np.array([[1.0], [0.45], [0.0], [0.8]])  # state 3 is unreachable
    C = np.array([[1.0, 0.35, 0.9, 0.0]])  # state 4 is unobservable
    D = np.array([[0.0]])

    R = reachability_matrix(A, B)
    obs = observability_matrix(A, C)
    rank_R = int(np.linalg.matrix_rank(R, tol=1e-10))
    rank_O = int(np.linalg.matrix_rank(obs, tol=1e-10))

    Wc, Wo = finite_gramians(A, B, C)
    gramian_hsv = np.sqrt(np.maximum(np.real(np.linalg.eigvals(Wc @ Wo)), 0.0))
    gramian_hsv = np.sort(gramian_hsv)[::-1]

    impulse = markov_parameters(A, B, C, D, n_samples=80)
    H = hankel_from_impulse(impulse, rows=24, cols=24)
    finite_hsv = np.linalg.svd(H, compute_uv=False)
    numerical_hankel_rank = int(np.sum(finite_hsv > 1e-8))

    rows: list[dict[str, float | int | str]] = [
        {"quantity": "state_dimension", "value": A.shape[0]},
        {"quantity": "reachability_rank", "value": rank_R},
        {"quantity": "observability_rank", "value": rank_O},
        {"quantity": "finite_hankel_rank_tol_1e-8", "value": numerical_hankel_rank},
        {"quantity": "leading_gramian_hsv", "value": f"{gramian_hsv[0]:.8g}"},
        {"quantity": "second_gramian_hsv", "value": f"{gramian_hsv[1]:.8g}"},
        {"quantity": "third_gramian_hsv", "value": f"{gramian_hsv[2]:.8g}"},
    ]
    csv_path = out_dir / "reachability_observability_hankel_summary.csv"
    write_summary(csv_path, rows)

    print("state dimension:", A.shape[0])
    print("reachability rank:", rank_R)
    print("observability rank:", rank_O)
    print("finite Hankel numerical rank (tol=1e-8):", numerical_hankel_rank)
    print("Gramian Hankel singular values:", [round(float(v), 8) for v in gramian_hsv])
    print("finite Hankel singular values:", [round(float(v), 8) for v in finite_hsv[:6]])
    print()
    print("Interpretation: one state is unreachable, one is unobservable, and the")
    print("input-output Hankel matrix only has two significant directions.")
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    idx = np.arange(1, min(10, finite_hsv.size) + 1)
    ax.semilogy(idx, finite_hsv[: idx.size], marker="o")
    ax.set_title("Finite Hankel singular values identify minimal input-output order")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "reachability_observability_hankel_singular_values.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
