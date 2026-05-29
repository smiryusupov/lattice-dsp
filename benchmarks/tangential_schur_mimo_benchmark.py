"""Benchmark finite MIMO tangential-Schur/Pick and J-inner diagnostics.

The benchmark focuses on the pure Python/NumPy tangential-Schur layer that
connects MIMO interpolation data with Pick certificates and J-inner Potapov
factors.  It measures three public operations:

* building and diagonalizing the right-tangential Pick matrix;
* recovering a compatible constant Schur matrix and checking interpolation;
* building/evaluating elementary J-inner Potapov products on the unit circle.

A separate diagonal sanity block compares a full diagonal-MIMO Pick matrix with
independent scalar Pick blocks.  That comparison is a benchmark counterpart to
the diagonal-MIMO-equals-SISO examples: the full matrix formulation should agree
numerically with the independent scalar decomposition while exposing the extra
cost of treating a diagonal problem as dense MIMO data.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable

import numpy as np


def _load_lattice_api():
    """Import public lattice_dsp API, with a source-tree fallback for docs smoke runs.

    A source checkout without the compiled extension cannot import the top-level
    package because ``lattice_dsp._core`` is missing.  The tangential-Schur module
    itself is pure Python, so benchmark/docs generation can still run the finite
    Pick/J-inner diagnostics directly from ``lattice_dsp/tangential_schur.py``.
    Installed wheels use the public top-level API.
    """

    def from_source_tree():
        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "lattice_dsp" / "tangential_schur.py"
        spec = importlib.util.spec_from_file_location(
            "_lattice_dsp_tangential_schur_bench", module_path
        )
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError("could not load lattice_dsp/tangential_schur.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return SimpleNamespace(
            RightTangentialSchurData=module.RightTangentialSchurData,
            right_tangential_pick_matrix=module.right_tangential_pick_matrix,
            pick_matrix_eigenvalues=module.pick_matrix_eigenvalues,
            is_tangential_schur_solvable=module.is_tangential_schur_solvable,
            constant_schur_solution=module.constant_schur_solution,
            max_tangential_residual=module.max_tangential_residual,
            potapov_product_from_rank_one_data=module.potapov_product_from_rank_one_data,
            j_unitarity_residual=module.j_unitarity_residual,
        )

    try:
        import lattice_dsp as ld  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name != "lattice_dsp._core":
            raise
        return from_source_tree()
    # During local docs generation another older lattice_dsp build may be
    # importable on sys.path.  Use the source-tree implementation when the
    # newly added tangential API is absent.
    if not hasattr(ld, "RightTangentialSchurData"):
        return from_source_tree()
    return ld


ld = _load_lattice_api()


def median_time(fn: Callable[[], object], repeats: int) -> tuple[float, object]:
    values: list[float] = []
    result: object = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - t0)
    return statistics.median(values), result


def random_points(rng: np.random.Generator, n_points: int, radius: float) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    # Spread points in the disk but keep them away from the boundary by default.
    radii = radius * np.sqrt(rng.uniform(0.05, 1.0, size=n_points))
    return radii * np.exp(1j * angles)


def constant_contraction(rng: np.random.Generator, dim: int, scale: float) -> np.ndarray:
    raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    sigma_max = np.linalg.svd(raw, compute_uv=False)[0]
    return scale * raw / sigma_max


def full_mimo_data(
    dim: int, n_points: int, multiplicity: int, *, seed: int, radius: float, scale: float
):
    rng = np.random.default_rng(seed)
    s0 = constant_contraction(rng, dim, scale)
    points = random_points(rng, n_points, radius)
    directions = rng.normal(size=(n_points, dim, multiplicity)) + 1j * rng.normal(
        size=(n_points, dim, multiplicity)
    )
    values = np.einsum("oi,nir->nor", s0, directions)
    data = ld.RightTangentialSchurData(points, directions, values)
    return data, s0


def run_full_mimo_case(
    *,
    dim: int,
    n_points: int,
    multiplicity: int,
    repeats: int,
    boundary_grid: int,
    seed: int,
    radius: float,
    scale: float,
) -> dict[str, object]:
    data, s0 = full_mimo_data(
        dim,
        n_points,
        multiplicity,
        seed=seed,
        radius=radius,
        scale=scale,
    )

    pick_build_s, pick = median_time(lambda: ld.right_tangential_pick_matrix(data), repeats)
    pick_eig_s, eig = median_time(lambda: ld.pick_matrix_eigenvalues(pick), repeats)
    psd_check_s, solvable = median_time(lambda: ld.is_tangential_schur_solvable(data), repeats)
    constant_solution_s, recovered = median_time(lambda: ld.constant_schur_solution(data), repeats)
    residual = float(ld.max_tangential_residual(data, recovered))
    constant_error = float(np.linalg.norm(recovered - s0) / max(1.0, np.linalg.norm(s0)))
    sigma = float(np.linalg.svd(recovered, compute_uv=False)[0])

    potapov_build_s, product = median_time(
        lambda: ld.potapov_product_from_rank_one_data(data), repeats
    )
    omega = np.linspace(0.0, 2.0 * np.pi, boundary_grid, endpoint=False)
    z_grid = np.exp(1j * omega)
    potapov_eval_s, theta = median_time(lambda: product.evaluate(z_grid), repeats)
    j_check_s, j_residuals = median_time(lambda: ld.j_unitarity_residual(theta, product.j), repeats)
    max_j = float(np.max(j_residuals))

    total_conditions = int(data.total_conditions)
    total_time_s = float(
        pick_build_s
        + pick_eig_s
        + psd_check_s
        + constant_solution_s
        + potapov_build_s
        + potapov_eval_s
        + j_check_s
    )
    return {
        "case": "full_mimo_tangential_schur",
        "dim": dim,
        "points": n_points,
        "multiplicity": multiplicity,
        "total_conditions": total_conditions,
        "pick_size": int(pick.shape[0]),
        "j_dimension": int(product.dimension),
        "boundary_grid": boundary_grid,
        "pick_build_s": float(pick_build_s),
        "pick_eig_s": float(pick_eig_s),
        "psd_check_s": float(psd_check_s),
        "constant_solution_s": float(constant_solution_s),
        "potapov_build_s": float(potapov_build_s),
        "potapov_eval_s": float(potapov_eval_s),
        "j_check_s": float(j_check_s),
        "time_s": total_time_s,
        "min_pick_eigenvalue": float(np.min(eig)),
        "max_pick_eigenvalue": float(np.max(eig)),
        "solvable": bool(solvable),
        "constant_solution_sigma_max": sigma,
        "max_tangential_residual": residual,
        "constant_solution_relative_error": constant_error,
        "j_inner_residual": max_j,
    }


def diagonal_mimo_coordinate_data(
    dim: int, points_per_channel: int, *, seed: int, radius: float, scale: float
):
    rng = np.random.default_rng(seed)
    gains = scale * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=dim))
    points_blocks = [random_points(rng, points_per_channel, radius) for _ in range(dim)]
    points = np.concatenate(points_blocks)
    directions = np.zeros((dim * points_per_channel, dim), dtype=np.complex128)
    values = np.zeros_like(directions)
    for ch in range(dim):
        start = ch * points_per_channel
        stop = start + points_per_channel
        directions[start:stop, ch] = 1.0
        values[start:stop, ch] = gains[ch]
    data = ld.RightTangentialSchurData(points, directions, values)
    return data, points_blocks, gains


def scalar_pick_matrix(points: np.ndarray, gain: complex) -> np.ndarray:
    z = np.asarray(points, dtype=np.complex128)
    pick = np.empty((z.size, z.size), dtype=np.complex128)
    for i, zi in enumerate(z):
        for j, zj in enumerate(z):
            pick[i, j] = (1.0 - np.conj(gain) * gain) / (1.0 - np.conj(zi) * zj)
    return 0.5 * (pick + pick.conj().T)


def block_diag(mats: list[np.ndarray]) -> np.ndarray:
    size = sum(mat.shape[0] for mat in mats)
    out = np.zeros((size, size), dtype=np.complex128)
    offset = 0
    for mat in mats:
        n = mat.shape[0]
        out[offset : offset + n, offset : offset + n] = mat
        offset += n
    return out


def run_diagonal_scalar_block_case(
    *,
    dim: int,
    points_per_channel: int,
    repeats: int,
    seed: int,
    radius: float,
    scale: float,
) -> dict[str, object]:
    data, point_blocks, gains = diagonal_mimo_coordinate_data(
        dim,
        points_per_channel,
        seed=seed,
        radius=radius,
        scale=scale,
    )

    full_pick_s, full_pick = median_time(lambda: ld.right_tangential_pick_matrix(data), repeats)
    full_eig_s, full_eig = median_time(lambda: ld.pick_matrix_eigenvalues(full_pick), repeats)

    def scalar_blocks():
        blocks = [scalar_pick_matrix(points, gains[ch]) for ch, points in enumerate(point_blocks)]
        eig = np.concatenate([np.linalg.eigvalsh(block) for block in blocks])
        return blocks, eig

    scalar_blocks_s, (blocks, scalar_eig) = median_time(scalar_blocks, repeats)
    expected = block_diag(blocks)
    block_error = float(np.linalg.norm(full_pick - expected) / max(1.0, np.linalg.norm(expected)))
    eig_error = float(
        np.linalg.norm(np.sort(full_eig) - np.sort(scalar_eig))
        / max(1.0, np.linalg.norm(np.sort(scalar_eig)))
    )
    speedup = scalar_blocks_s / full_pick_s if full_pick_s > 0.0 else float("inf")
    return {
        "case": "diagonal_mimo_vs_scalar_blocks",
        "dim": dim,
        "points": int(dim * points_per_channel),
        "points_per_channel": points_per_channel,
        "multiplicity": 1,
        "total_conditions": int(data.total_conditions),
        "pick_size": int(full_pick.shape[0]),
        "full_mimo_pick_s": float(full_pick_s),
        "scalar_blocks_pick_s": float(scalar_blocks_s),
        "pick_eig_s": float(full_eig_s),
        "time_s": float(full_pick_s + full_eig_s),
        "scalar_block_time_s": float(scalar_blocks_s),
        "speedup_scalar_blocks_vs_full_mimo": float(speedup),
        "min_pick_eigenvalue": float(np.min(full_eig)),
        "diagonal_block_relative_error": block_error,
        "diagonal_block_eigenvalue_relative_error": eig_error,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_or_nan(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def plot_benchmark(rows: list[dict[str, object]], artifact_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return []

    written: list[Path] = []
    artifact_dir.mkdir(parents=True, exist_ok=True)

    full_rows = [row for row in rows if row.get("case") == "full_mimo_tangential_schur"]
    if full_rows:
        labels = [f"d={r['dim']}\nN={r['points']} r={r['multiplicity']}" for r in full_rows]
        x = np.arange(len(full_rows))
        timing_keys = [
            "pick_build_s",
            "pick_eig_s",
            "constant_solution_s",
            "potapov_build_s",
            "potapov_eval_s",
        ]
        fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * len(full_rows) + 3.0), 4.8))
        bottom = np.zeros(len(full_rows))
        for key in timing_keys:
            values = np.array([finite_or_nan(row.get(key)) for row in full_rows], dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            ax.bar(x, values, bottom=bottom, label=key.replace("_", " "))
            bottom += values
        ax.set_title("MIMO tangential-Schur runtime breakdown")
        ax.set_ylabel("seconds")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize="small")
        fig.tight_layout()
        out = artifact_dir / "tangential_schur_mimo_runtime_breakdown.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

        fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * len(full_rows) + 3.0), 4.8))
        metrics = [
            "max_tangential_residual",
            "j_inner_residual",
            "constant_solution_relative_error",
        ]
        width = 0.24
        for idx, key in enumerate(metrics):
            values = np.array([finite_or_nan(row.get(key)) for row in full_rows], dtype=float)
            ax.bar(x + (idx - 1) * width, values, width=width, label=key.replace("_", " "))
        ax.set_title("MIMO tangential-Schur numerical residuals")
        ax.set_ylabel("residual")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize="small")
        fig.tight_layout()
        out = artifact_dir / "tangential_schur_mimo_residuals.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    diag_rows = [row for row in rows if row.get("case") == "diagonal_mimo_vs_scalar_blocks"]
    if diag_rows:
        labels = [f"d={r['dim']}\npts/ch={r['points_per_channel']}" for r in diag_rows]
        x = np.arange(len(diag_rows))
        full = np.array(
            [finite_or_nan(row.get("full_mimo_pick_s")) for row in diag_rows], dtype=float
        )
        scalar = np.array(
            [finite_or_nan(row.get("scalar_blocks_pick_s")) for row in diag_rows], dtype=float
        )
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(7.0, 0.65 * len(diag_rows) + 3.0), 4.6))
        ax.bar(x - width / 2, full, width=width, label="full dense MIMO Pick")
        ax.bar(x + width / 2, scalar, width=width, label="independent scalar blocks")
        ax.set_title("Diagonal MIMO Pick matrix versus scalar-block decomposition")
        ax.set_ylabel("seconds")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize="small")
        fig.tight_layout()
        out = artifact_dir / "tangential_schur_diagonal_mimo_vs_scalar_blocks.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

        fig, ax = plt.subplots(figsize=(max(7.0, 0.65 * len(diag_rows) + 3.0), 4.6))
        errors = np.array(
            [finite_or_nan(row.get("diagonal_block_relative_error")) for row in diag_rows],
            dtype=float,
        )
        eig_errors = np.array(
            [
                finite_or_nan(row.get("diagonal_block_eigenvalue_relative_error"))
                for row in diag_rows
            ],
            dtype=float,
        )
        # Exact zeros are possible in the diagonal sanity check.  Plot a tiny
        # floor so the log-scale figure remains visible without changing the
        # downloadable numeric data.
        errors_plot = np.maximum(np.nan_to_num(errors, nan=0.0), 1e-18)
        eig_errors_plot = np.maximum(np.nan_to_num(eig_errors, nan=0.0), 1e-18)
        ax.bar(x - width / 2, errors_plot, width=width, label="Pick block error")
        ax.bar(x + width / 2, eig_errors_plot, width=width, label="eigenvalue error")
        ax.set_title("Diagonal-MIMO reduction-to-scalar numerical check")
        ax.set_ylabel("relative error")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize="small")
        fig.tight_layout()
        out = artifact_dir / "tangential_schur_diagonal_scalar_block_errors.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--points", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--diagonal-points-per-channel", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--boundary-grid", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--radius", type=float, default=0.65)
    parser.add_argument("--scale", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--output", type=Path, default=Path("reports/tangential-schur-mimo.json"))
    parser.add_argument(
        "--csv-output", type=Path, default=Path("reports/tangential-schur-mimo.csv")
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(dim <= 0 for dim in args.dims):
        raise SystemExit("--dims must contain positive integers")
    if any(n <= 0 for n in args.points):
        raise SystemExit("--points must contain positive integers")
    if args.multiplicity <= 0:
        raise SystemExit("--multiplicity must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.boundary_grid <= 0:
        raise SystemExit("--boundary-grid must be positive")
    if not (0.0 < args.radius < 1.0):
        raise SystemExit("--radius must lie in (0, 1)")
    if not (0.0 < args.scale < 1.0):
        raise SystemExit("--scale must lie in (0, 1)")

    rows: list[dict[str, object]] = []
    for dim in args.dims:
        for n_points in args.points:
            rows.append(
                run_full_mimo_case(
                    dim=dim,
                    n_points=n_points,
                    multiplicity=args.multiplicity,
                    repeats=args.repeats,
                    boundary_grid=args.boundary_grid,
                    seed=args.seed + 1000 * dim + 17 * n_points,
                    radius=args.radius,
                    scale=args.scale,
                )
            )
        for points_per_channel in args.diagonal_points_per_channel:
            rows.append(
                run_diagonal_scalar_block_case(
                    dim=dim,
                    points_per_channel=points_per_channel,
                    repeats=args.repeats,
                    seed=args.seed + 3000 * dim + 19 * points_per_channel,
                    radius=args.radius,
                    scale=args.scale,
                )
            )

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "description": "Finite MIMO tangential-Schur/Pick and J-inner diagnostic benchmark.",
        "dims": args.dims,
        "points": args.points,
        "multiplicity": args.multiplicity,
        "diagonal_points_per_channel": args.diagonal_points_per_channel,
        "boundary_grid": args.boundary_grid,
        "repeats": args.repeats,
        "radius": args.radius,
        "scale": args.scale,
        "seed": args.seed,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows, args.csv_output)

    written_plots: list[Path] = []
    if not args.no_plots:
        written_plots = plot_benchmark(rows, args.output.parent)

    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))
    print()
    print(
        f"{'case':>34} {'dim':>4} {'points':>6} {'cond':>6} {'total_s':>10} "
        f"{'pick_s':>10} {'pot_eval_s':>11} {'j_resid':>10} {'interp_resid':>13}"
    )
    print("-" * 124)
    for row in rows:
        print(
            f"{str(row['case']):>34} {int(row['dim']):4d} {int(row['points']):6d} "
            f"{int(row['total_conditions']):6d} {finite_or_nan(row.get('time_s')):10.4e} "
            f"{finite_or_nan(row.get('pick_build_s', row.get('full_mimo_pick_s'))):10.4e} "
            f"{finite_or_nan(row.get('potapov_eval_s')):11.4e} "
            f"{finite_or_nan(row.get('j_inner_residual')):10.2e} "
            f"{finite_or_nan(row.get('max_tangential_residual', row.get('diagonal_block_relative_error'))):13.2e}"
        )
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.csv_output}")
    for path in written_plots:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
