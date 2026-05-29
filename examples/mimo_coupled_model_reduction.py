"""Tutorial: coupled MIMO finite-Hankel model reduction.

The diagonal MIMO tutorial shows that independent SISO filters are a special
case of MIMO.  This tutorial moves to the genuinely coupled case: every input
can affect every output through a shared stable state-space model.

The reducer works with Markov matrices, builds a finite block-Hankel matrix, and
returns a reduced state-space realization.  This is the reference MIMO finite-section baseline and is separate from
matrix AAK/Nehari optimality claims.
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


def coupled_state_space(order: int = 12, outputs: int = 3, inputs: int = 3, seed: int = 31):
    """Return a stable, visibly coupled MIMO state-space system.

    The construction uses a random orthogonal basis and deliberately dense B/C/D
    matrices so off-diagonal input-output channels are nonzero.  The eigenvalues
    of A are inside the unit disk by construction.
    """

    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(order, order)))
    radii = np.linspace(0.88, 0.20, order)
    signs = np.where(np.arange(order) % 2 == 0, 1.0, -1.0)
    A = q @ np.diag(signs * radii) @ q.T

    B = 0.32 * rng.normal(size=(order, inputs))
    C = 0.32 * rng.normal(size=(outputs, order))
    D = 0.03 * rng.normal(size=(outputs, inputs))

    # Add deterministic cross-channel structure so the system is not close to diagonal.
    if outputs == inputs:
        D += 0.04 * (np.ones((outputs, inputs)) - np.eye(outputs))
    return A, B, C, D


def state_spectral_radius(A) -> float:
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def state_space_process_python(A, B, C, D, x):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    x = np.asarray(x, dtype=float)

    batch, samples, _ = x.shape
    n_outputs = D.shape[0]
    n_state = A.shape[0]
    state = np.zeros((batch, n_state), dtype=float)
    y = np.zeros((batch, samples, n_outputs), dtype=float)

    for n in range(samples):
        xn = x[:, n, :]
        y[:, n, :] = state @ C.T + xn @ D.T
        if n_state:
            state = state @ A.T + xn @ B.T
    return y


def state_space_process(A, B, C, D, x):
    """Process batched MIMO signals through a state-space model.

    The installed package exposes a compiled C++/OpenMP processor.  The small
    Python fallback keeps the tutorial readable if someone opens it against an
    older local extension before rebuilding.
    """

    compiled = getattr(ld, "mimo_state_space_process_batch", None)
    if compiled is None:
        return state_space_process_python(A, B, C, D, x)
    return compiled(A, B, C, D, x)


def relative_channel_error(reference, estimate):
    """Return per-output/input relative Markov errors."""

    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    num = np.sum((reference - estimate) ** 2, axis=0)
    den = np.sum(reference**2, axis=0) + 1e-30
    return num / den


def main() -> None:
    out_dir = artifact_dir()

    full_order = 12
    inputs = outputs = 3
    n_markov = 220
    block_rows = block_cols = 28
    reduced_orders = [2, 4, 6, 8]

    A, B, C, D = coupled_state_space(full_order, outputs, inputs)
    markov = ld.mimo_state_space_markov_response(A, B, C, D, n_markov)

    rng = np.random.default_rng(123)
    x = rng.normal(size=(16, 3000, inputs))
    y_full = state_space_process(A, B, C, D, x)

    summary = []
    reduced_markov = {}
    reduced_outputs = {}
    reduced_results = {}

    for order in reduced_orders:
        result = ld.finite_hankel_reduce_mimo(
            markov,
            reduced_order=order,
            block_rows=block_rows,
            block_cols=block_cols,
        )
        approx_markov = ld.mimo_state_space_markov_response(
            result["A"], result["B"], result["C"], result["D"], n_markov
        )
        y_reduced = state_space_process(result["A"], result["B"], result["C"], result["D"], x)

        rel_markov = float(np.sum((markov - approx_markov) ** 2) / (np.sum(markov**2) + 1e-30))
        rel_output = float(np.sum((y_full - y_reduced) ** 2) / (np.sum(y_full**2) + 1e-30))
        output_snr = float(10.0 * np.log10(1.0 / max(rel_output, 1e-300)))
        coupling_error = relative_channel_error(markov, approx_markov)

        reduced_markov[order] = approx_markov
        reduced_outputs[order] = y_reduced
        reduced_results[order] = result
        summary.append(
            {
                "order": order,
                "stable": bool(result["stable"]),
                "state_radius": state_spectral_radius(result["A"]),
                "retained_hankel_energy": float(result["retained_hankel_energy"]),
                "relative_markov_error": rel_markov,
                "relative_output_error": rel_output,
                "output_snr_db": output_snr,
                "max_channel_markov_error": float(np.max(coupling_error)),
            }
        )

    csv_path = out_dir / "mimo_coupled_model_reduction_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    hsv = np.asarray(reduced_results[reduced_orders[-1]]["hankel_singular_values"], dtype=float)
    print("full state order:", full_order)
    print("inputs:", inputs, "outputs:", outputs)
    print("full state spectral radius:", f"{state_spectral_radius(A):.4f}")
    print("block Hankel matrix:", f"{block_rows * outputs} x {block_cols * inputs}")
    print("leading block-Hankel singular values:", [round(float(v), 6) for v in hsv[:10]])
    for row in summary:
        print(
            "order={order}: stable={stable}, radius={state_radius:.4f}, retained={retained_hankel_energy:.6f}, "
            "markov_error={relative_markov_error:.3e}, output_snr={output_snr_db:.2f} dB".format(
                **row
            )
        )
    print(f"wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed; skipped figures")
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    idx = np.arange(1, min(35, hsv.size) + 1)
    ax.semilogy(idx, hsv[: idx.size], marker="o")
    ax.set_title("Coupled MIMO block-Hankel singular values")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "mimo_coupled_block_hankel_singular_values.png"
    fig.savefig(fig_path, dpi=160)
    print(f"wrote {fig_path}")

    selected_order = 6
    fig2, axes = plt.subplots(outputs, inputs, figsize=(9, 7), sharex=True)
    t = np.arange(80)
    for y in range(outputs):
        for u in range(inputs):
            ax = axes[y, u]
            ax.plot(t, markov[: t.size, y, u], linewidth=1.8, label="full")
            ax.plot(
                t,
                reduced_markov[selected_order][: t.size, y, u],
                "--",
                linewidth=1.2,
                label=f"reduced {selected_order}",
            )
            ax.grid(True, alpha=0.25)
            if y == outputs - 1:
                ax.set_xlabel(f"input {u}")
            if u == 0:
                ax.set_ylabel(f"output {y}")
    axes[0, 0].legend(loc="upper right")
    fig2.suptitle("Coupled MIMO Markov responses", y=1.02)
    fig2.tight_layout()
    fig2_path = out_dir / "mimo_coupled_markov_responses.png"
    fig2.savefig(fig2_path, dpi=160)
    print(f"wrote {fig2_path}")

    err = relative_channel_error(markov, reduced_markov[selected_order])
    fig3, ax3 = plt.subplots(figsize=(5.2, 4.4))
    im = ax3.imshow(err)
    ax3.set_title(f"Relative Markov error per channel, order {selected_order}")
    ax3.set_xlabel("input")
    ax3.set_ylabel("output")
    fig3.colorbar(im, ax=ax3)
    fig3.tight_layout()
    fig3_path = out_dir / "mimo_coupled_error_heatmap.png"
    fig3.savefig(fig3_path, dpi=160)
    print(f"wrote {fig3_path}")

    fig4, ax4 = plt.subplots(figsize=(5.2, 5.2))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax4.plot(np.cos(theta), np.sin(theta), linewidth=1.0)
    ax4.scatter(
        np.real(np.linalg.eigvals(A)), np.imag(np.linalg.eigvals(A)), marker="o", label="full"
    )
    ax4.scatter(
        np.real(np.linalg.eigvals(reduced_results[selected_order]["A"])),
        np.imag(np.linalg.eigvals(reduced_results[selected_order]["A"])),
        marker="x",
        label=f"reduced {selected_order}",
    )
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_title("MIMO state poles / eigenvalues")
    ax4.set_xlabel("real")
    ax4.set_ylabel("imag")
    ax4.legend()
    ax4.grid(True, alpha=0.25)
    fig4.tight_layout()
    fig4_path = out_dir / "mimo_coupled_state_poles.png"
    fig4.savefig(fig4_path, dpi=160)
    print(f"wrote {fig4_path}")


if __name__ == "__main__":
    main()
