import numpy as np

from benchmarks.finite_nehari_rank_sweep import run_rank_sweep, synthetic_anticausal_tail


def test_synthetic_anticausal_tail_has_requested_length():
    tail = synthetic_anticausal_tail(17, seed=3)
    assert tail.shape == (17,)
    assert np.all(np.isfinite(tail))


def test_rank_sweep_reports_monotone_unconstrained_errors():
    rows = cols = 10
    tail = synthetic_anticausal_tail(rows + cols - 1, seed=5)
    results = run_rank_sweep(tail, ranks=[1, 2, 3], rows=rows, cols=cols)

    assert [r["rank"] for r in results] == [1, 2, 3]
    for row in results:
        assert row["sigma_next"] > 0.0
        assert row["unconstrained_hankel_error"] >= row["sigma_next"] - 1e-10
        assert row["unconstrained_hankel_error"] <= row["sigma_next"] + 1e-8
        assert row["hankelized_hankel_error"] >= 0.0
        assert row["relative_tail_error"] >= 0.0

    svd_errors = [r["unconstrained_hankel_error"] for r in results]
    assert svd_errors[0] >= svd_errors[1] >= svd_errors[2]
