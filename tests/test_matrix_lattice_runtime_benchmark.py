from __future__ import annotations

from benchmarks.matrix_lattice_runtime import run_case


def test_matrix_lattice_runtime_case_matches_numpy_reference() -> None:
    row = run_case(dim=2, order=2, n_freq=32, repeats=1, n_threads=1, seed=50)
    assert row["dim"] == 2
    assert row["order"] == 2
    assert row["relative_difference"] < 1e-11
    assert row["unitarity_error"] < 1e-9
    assert row["real_scalar_parameter_count"] == 2 * (2 * 2 * 2 + 2 * 2)
