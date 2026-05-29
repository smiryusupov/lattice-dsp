import numpy as np

import lattice_dsp as ld
from examples import mimo_coupled_model_reduction as tutorial
from benchmarks import mimo_hankel_reduction_speedup as bench


def test_coupled_state_space_is_stable_and_coupled():
    A, B, C, D = tutorial.coupled_state_space(order=8, outputs=3, inputs=3, seed=4)
    assert tutorial.state_spectral_radius(A) < 1.0
    assert np.linalg.norm(D - np.diag(np.diag(D))) > 0.0
    markov = ld.mimo_state_space_markov_response(A, B, C, D, 40)
    offdiag_energy = np.sum(markov[:, ~np.eye(3, dtype=bool)] ** 2)
    diag_energy = np.sum(markov[:, np.eye(3, dtype=bool)] ** 2)
    assert offdiag_energy > 0.01 * diag_energy


def test_coupled_mimo_reduction_improves_with_order():
    A, B, C, D = tutorial.coupled_state_space(order=8, outputs=3, inputs=3, seed=5)
    markov = ld.mimo_state_space_markov_response(A, B, C, D, 100)
    low = ld.finite_hankel_reduce_mimo(markov, reduced_order=2, block_rows=14, block_cols=14)
    high = ld.finite_hankel_reduce_mimo(markov, reduced_order=6, block_rows=14, block_cols=14)
    assert low["stable"] is True
    assert high["stable"] is True
    assert high["relative_markov_error"] < low["relative_markov_error"]
    assert high["retained_hankel_energy"] >= low["retained_hankel_energy"]


def test_state_space_process_shapes_and_reduction_output_error():
    A, B, C, D = tutorial.coupled_state_space(order=6, outputs=2, inputs=2, seed=6)
    markov = ld.mimo_state_space_markov_response(A, B, C, D, 80)
    result = ld.finite_hankel_reduce_mimo(markov, reduced_order=4, block_rows=12, block_cols=12)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 120, 2))
    y_full = tutorial.state_space_process(A, B, C, D, x)
    y_reduced = tutorial.state_space_process(result["A"], result["B"], result["C"], result["D"], x)
    assert y_full.shape == (4, 120, 2)
    assert y_reduced.shape == y_full.shape
    rel = np.sum((y_full - y_reduced) ** 2) / np.sum(y_full**2)
    assert rel < 0.15


def test_mimo_hankel_reduction_benchmark_small_run():
    class Args:
        full_orders = [6]
        reduced_orders = [2, 4]
        inputs = 2
        outputs = 2
        batch = 3
        samples = 80
        repeats = 1
        reuse_count = 5
        n_markov = 80
        block_rows = 10
        block_cols = 10
        seed = 123

    result = bench.run(Args())
    assert result["metadata"]["inputs"] == 2
    assert result["metadata"]["reuse_count"] == 5
    assert len(result["rows"]) == 2
    assert all("relative_markov_error" in row for row in result["rows"])
    assert all("one_shot_end_to_end_speedup" in row for row in result["rows"])
    assert all("amortized_end_to_end_speedup" in row for row in result["rows"])
    assert result["rows"][1]["relative_markov_error"] <= result["rows"][0]["relative_markov_error"]
