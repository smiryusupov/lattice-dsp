from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Algorithm:
    name: str
    kind: str
    mu: float = 0.0
    lam: float = 0.995
    p0: float = 100.0
    eps: float = 1e-6


def artifact_dir() -> Path:
    path = Path(os.environ.get("LATTICE_DSP_ARTIFACT_DIR", "reports/example-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_regressors(n: int = 180, order: int = 4, seed: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n + order + 2)
    for i in range(1, len(x)):
        x[i] = 0.92 * x[i - 1] + 0.45 * x[i]
    u = np.zeros((n, order), dtype=float)
    for i in range(n):
        u[i] = x[i + order - 1 : i - 1 : -1] if i > 0 else x[order - 1 :: -1]
    return u / np.sqrt(np.mean(u * u))


def run_adaptive(
    u: np.ndarray, w_true: np.ndarray, disturbance: np.ndarray, alg: Algorithm
) -> dict[str, np.ndarray]:
    n, p = u.shape
    w = np.zeros(p, dtype=float)
    P = alg.p0 * np.eye(p)

    pred_error = np.zeros(n)
    posterior_error = np.zeros(n)
    weight_error = np.zeros(n)
    weights = np.zeros((n, p))

    for i, ui in enumerate(u):
        d = float(ui @ w_true + disturbance[i])
        e = d - float(ui @ w)
        pred_error[i] = e

        if alg.kind == "lms":
            w = w + alg.mu * ui * e
        elif alg.kind == "nlms":
            w = w + (alg.mu / (alg.eps + float(ui @ ui))) * ui * e
        elif alg.kind == "rls":
            Pu = P @ ui
            gain = Pu / (alg.lam + float(ui @ Pu))
            w = w + gain * e
            P = (P - np.outer(gain, ui) @ P) / alg.lam
            P = 0.5 * (P + P.T)
        else:
            raise ValueError(f"Unknown algorithm kind: {alg.kind}")

        posterior_error[i] = d - float(ui @ w)
        weight_error[i] = np.linalg.norm(w - w_true)
        weights[i] = w

    return {
        "prediction_error": pred_error,
        "posterior_error": posterior_error,
        "weight_error": weight_error,
        "weights": weights,
    }


def sensitivity_matrix(
    u: np.ndarray, w_true: np.ndarray, alg: Algorithm, *, output: str
) -> tuple[np.ndarray, float, np.ndarray]:
    n = u.shape[0]
    zero = np.zeros(n)
    base = run_adaptive(u, w_true, zero, alg)[output]
    M = np.zeros((n, n), dtype=float)
    for j in range(n):
        impulse = np.zeros(n)
        impulse[j] = 1.0
        response = run_adaptive(u, w_true, impulse, alg)[output]
        M[:, j] = response - base
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return M, float(S[0] ** 2), Vt[0]


def cumulative_gain(noise_error: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
    num = np.cumsum(noise_error * noise_error)
    den = np.cumsum(disturbance * disturbance) + 1e-30
    return num / den


def main() -> None:
    out_dir = artifact_dir()
    rng = np.random.default_rng(4)

    n = 180
    order = 4
    u = make_regressors(n=n, order=order)
    w_true = np.array([0.70, -0.42, 0.25, -0.12])
    noise_rms = 0.04

    algorithms = [
        Algorithm("small-step LMS", "lms", mu=0.020),
        Algorithm("NLMS", "nlms", mu=0.25),
        Algorithm("RLS", "rls", lam=0.995, p0=100.0),
    ]

    random_disturbance = noise_rms * rng.normal(size=n)

    sensitivity: dict[str, dict[str, object]] = {}
    for alg in algorithms:
        M, gain, v_unit = sensitivity_matrix(u, w_true, alg, output="prediction_error")
        sensitivity[alg.name] = {"matrix": M, "gain": gain, "v_unit": v_unit}

    # Worst-case disturbance for the RLS prediction-error map.  Scale it to the same
    # energy as a length-n Gaussian sequence with RMS = noise_rms.
    rls_worst_unit = np.asarray(sensitivity["RLS"]["v_unit"], dtype=float)
    rls_worst_disturbance = noise_rms * np.sqrt(n) * rls_worst_unit

    zero = np.zeros(n)
    rows = []
    trajectories: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for alg in algorithms:
        clean = run_adaptive(u, w_true, zero, alg)
        random_run = run_adaptive(u, w_true, random_disturbance, alg)
        worst_run = run_adaptive(u, w_true, rls_worst_disturbance, alg)

        random_noise_error = random_run["prediction_error"] - clean["prediction_error"]
        worst_noise_error = worst_run["prediction_error"] - clean["prediction_error"]

        rows.append(
            {
                "algorithm": alg.name,
                "mu_or_lambda": alg.mu if alg.kind != "rls" else alg.lam,
                "operator_worst_case_gain": float(sensitivity[alg.name]["gain"]),
                "random_noise_gain": float(
                    np.sum(random_noise_error**2) / (np.sum(random_disturbance**2) + 1e-30)
                ),
                "rls_worst_disturbance_gain": float(
                    np.sum(worst_noise_error**2) / (np.sum(rls_worst_disturbance**2) + 1e-30)
                ),
                "clean_final_weight_error": float(clean["weight_error"][-1]),
                "random_final_weight_error": float(random_run["weight_error"][-1]),
                "rls_worst_final_weight_error": float(worst_run["weight_error"][-1]),
                "random_tail_mse": float(np.mean(random_run["prediction_error"][-50:] ** 2)),
                "rls_worst_tail_mse": float(np.mean(worst_run["prediction_error"][-50:] ** 2)),
            }
        )
        trajectories[alg.name] = {"clean": clean, "random": random_run, "rls_worst": worst_run}

    csv_path = out_dir / "hinf_lms_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for alg in algorithms:
        ax.semilogy(
            t, trajectories[alg.name]["clean"]["weight_error"], label=f"{alg.name}: no disturbance"
        )
        ax.semilogy(
            t,
            trajectories[alg.name]["random"]["weight_error"],
            linestyle="--",
            label=f"{alg.name}: random disturbance",
        )
    ax.set_title("Average-case view: convergence under benign random disturbance")
    ax.set_xlabel("iteration")
    ax.set_ylabel("weight-error norm")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    convergence_path = out_dir / "hinf_lms_random_convergence.png"
    fig.savefig(convergence_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for alg in algorithms:
        clean = trajectories[alg.name]["clean"]
        worst = trajectories[alg.name]["rls_worst"]
        noise_error = worst["prediction_error"] - clean["prediction_error"]
        ax.plot(t, cumulative_gain(noise_error, rls_worst_disturbance), label=alg.name)
    ax.set_title("Worst-case view: cumulative gain under an RLS-aligned disturbance")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cumulative noise-error energy / disturbance energy")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    gain_path = out_dir / "hinf_lms_cumulative_gain.png"
    fig.savefig(gain_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(t, rls_worst_disturbance, label="disturbance")
    for alg in algorithms:
        clean = trajectories[alg.name]["clean"]
        worst = trajectories[alg.name]["rls_worst"]
        noise_error = worst["prediction_error"] - clean["prediction_error"]
        ax.plot(t, noise_error, label=f"{alg.name} noise-induced error", alpha=0.85)
    ax.set_title("A disturbance can be chosen to excite an estimator's weak direction")
    ax.set_xlabel("iteration")
    ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    disturbance_path = out_dir / "hinf_lms_worst_case_disturbance.png"
    fig.savefig(disturbance_path, dpi=160)
    plt.close(fig)

    print("H-infinity-inspired LMS/RLS diagnostic")
    print(f"samples={n}, order={order}, noise_rms={noise_rms}")
    print()
    print(
        f"{'algorithm':18s} {'worst gain':>12s} {'random gain':>12s} {'RLS-worst gain':>15s} {'random tail MSE':>16s}"
    )
    print("-" * 82)
    for row in rows:
        print(
            f"{row['algorithm']:18s} "
            f"{row['operator_worst_case_gain']:12.3f} "
            f"{row['random_noise_gain']:12.3f} "
            f"{row['rls_worst_disturbance_gain']:15.3f} "
            f"{row['random_tail_mse']:16.3e}"
        )
    print()
    print("Interpretation:")
    print("  RLS usually converges fastest under benign random noise, but the sensitivity")
    print("  matrix exposes directions where disturbance energy is amplified more strongly.")
    print("  Small-step LMS is slower, yet its disturbance-to-error gain can be smaller.")
    print("  This illustrates the Hassibi-Sayed-Kailath message: LMS can be understood")
    print("  through a worst-case energy-gain lens, not only as crude least-squares descent.")
    print()
    print(f"Wrote {convergence_path}")
    print(f"Wrote {gain_path}")
    print(f"Wrote {disturbance_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
