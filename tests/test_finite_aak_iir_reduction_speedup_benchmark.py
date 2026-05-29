import numpy as np

from benchmarks import finite_aak_iir_reduction_speedup as bench


def test_break_even_samples_per_channel():
    value = bench.break_even_samples_per_channel(
        reduction_time_s=0.1,
        full_time_s=1.0,
        reduced_time_s=0.5,
        channels=10,
        samples=1000,
    )
    assert value == 200.0


def test_break_even_none_when_reduced_is_not_faster():
    value = bench.break_even_samples_per_channel(
        reduction_time_s=0.1,
        full_time_s=0.5,
        reduced_time_s=0.5,
        channels=10,
        samples=1000,
    )
    assert value is None


def test_compressible_iir_is_stable_and_has_expected_shapes():
    rng = np.random.default_rng(123)
    model = bench.compressible_iir(order=8, rng=rng, n_impulse=128)
    assert model["reflection"].shape == (8,)
    assert model["numerator"].shape == (9,)
    assert model["impulse"].shape == (128,)
    assert np.max(np.abs(model["poles"])) < 1.0
