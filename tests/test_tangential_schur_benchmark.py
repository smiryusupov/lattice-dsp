from __future__ import annotations

import numpy as np

from benchmarks.tangential_schur_mimo_benchmark import (
    block_diag,
    full_mimo_data,
    run_diagonal_scalar_block_case,
    run_full_mimo_case,
    scalar_pick_matrix,
)


def test_full_mimo_tangential_schur_benchmark_case_is_accurate() -> None:
    row = run_full_mimo_case(
        dim=2,
        n_points=3,
        multiplicity=1,
        repeats=1,
        boundary_grid=32,
        seed=11,
        radius=0.55,
        scale=0.45,
    )
    assert row["case"] == "full_mimo_tangential_schur"
    assert row["solvable"] is True
    assert row["min_pick_eigenvalue"] > -1e-10
    assert row["max_tangential_residual"] < 1e-10
    assert row["constant_solution_relative_error"] < 1e-10
    assert row["j_inner_residual"] < 1e-10
    assert row["time_s"] >= 0.0


def test_diagonal_mimo_benchmark_reduces_to_scalar_pick_blocks() -> None:
    row = run_diagonal_scalar_block_case(
        dim=3,
        points_per_channel=3,
        repeats=1,
        seed=13,
        radius=0.55,
        scale=0.45,
    )
    assert row["case"] == "diagonal_mimo_vs_scalar_blocks"
    assert row["diagonal_block_relative_error"] < 1e-14
    assert row["diagonal_block_eigenvalue_relative_error"] < 1e-14
    assert row["speedup_scalar_blocks_vs_full_mimo"] >= 0.0


def test_scalar_pick_block_helper_matches_known_diagonal_assembly() -> None:
    points = np.array([0.0, 0.2 + 0.1j, -0.25j])
    gains = [0.2, -0.35j]
    blocks = [scalar_pick_matrix(points, gain) for gain in gains]
    assembled = block_diag(blocks)
    assert assembled.shape == (6, 6)
    np.testing.assert_allclose(assembled[:3, :3], blocks[0])
    np.testing.assert_allclose(assembled[3:, 3:], blocks[1])
    np.testing.assert_allclose(assembled[:3, 3:], 0.0)


def test_full_mimo_data_shapes() -> None:
    data, s0 = full_mimo_data(3, 4, 2, seed=17, radius=0.5, scale=0.4)
    assert data.input_dim == 3
    assert data.output_dim == 3
    assert data.n_points == 4
    assert data.total_conditions == 8
    assert s0.shape == (3, 3)
