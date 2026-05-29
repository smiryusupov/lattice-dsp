from __future__ import annotations

import math

from benchmarks.experimental_mimo_matrix_lattice_realization_sweep import run_case


def test_realization_sweep_case_reports_finite_diagnostics() -> None:
    row = run_case(
        full_order=6,
        reduced_order=2,
        lattice_order=2,
        channels=2,
        n_markov=48,
        n_freq=32,
        block_rows=8,
        block_cols=8,
        candidate_gains=(0.15, 0.45),
        static_gain_iterations=4,
        repeats=1,
        n_threads=1,
        seed=123,
    )

    assert row["full_order"] == 6
    assert row["reduced_order"] == 2
    assert row["lattice_order"] == 2
    assert row["channels"] == 2
    assert 0.0 <= float(row["selected_gain"])
    assert float(row["reduced_state_radius"]) < 1.0
    assert float(row["retained_hankel_energy"]) > 0.0
    assert float(row["max_reflection_singular_value"]) < 1.0
    assert float(row["unitarity_error"]) < 1e-8
    for key in (
        "reduce_s",
        "realize_s",
        "polar_factor_relative_error",
        "state_response_relative_error",
        "static_gain_relative_error",
        "static_gain_improvement",
        "target_gain_condition_span",
    ):
        assert math.isfinite(float(row[key]))
        assert float(row[key]) >= 0.0
    assert row["diagnostic_classification"] in {
        "good_allpass_polar_fit",
        "mostly_static_gain_or_nonunitary_mismatch",
        "poor_lattice_scaffold_fit",
    }
