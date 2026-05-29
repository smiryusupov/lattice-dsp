import numpy as np
import pytest

from lattice_dsp import AdaptiveLatticeLadderNLMS, LatticeIIR, adaptive_process_batch


def test_reflection_update_period_property_and_validation():
    adaptive = AdaptiveLatticeLadderNLMS([0.0], [1.0, 0.0], reflection_update_period=4)
    assert adaptive.reflection_update_period == 4
    adaptive.reflection_update_period = 2
    assert adaptive.reflection_update_period == 2
    assert adaptive.scale_reflection_mu_by_period is False
    adaptive.scale_reflection_mu_by_period = True
    assert adaptive.scale_reflection_mu_by_period is True
    with pytest.raises(ValueError, match="reflection_update_period"):
        adaptive.reflection_update_period = 0
    with pytest.raises(ValueError, match="reflection_update_period"):
        AdaptiveLatticeLadderNLMS([0.0], [1.0, 0.0], reflection_update_period=0)


def test_reflection_update_period_skips_raw_gradient_between_updates():
    rng = np.random.default_rng(12345)
    x = rng.normal(size=20)
    desired = np.asarray(LatticeIIR([0.35], [1.0, 0.0]).process(x), dtype=float)

    adaptive = AdaptiveLatticeLadderNLMS(
        [0.0],
        [1.0, 0.0],
        mu_taps=0.0,
        mu_reflection=0.02,
        reflection_update_period=4,
    )

    # Sample 0 is an update point; sample 1 is skipped and should expose a zero
    # raw gradient while still advancing the internal signal histories.
    adaptive.adapt_sample(float(x[0]), float(desired[0]))
    adaptive.adapt_sample(float(x[1]), float(desired[1]))
    np.testing.assert_allclose(adaptive.last_raw_gradient, [0.0], atol=1e-15)


def test_reflection_update_period_still_learns_with_decimated_updates():
    rng = np.random.default_rng(2027)
    x = rng.normal(size=5000)
    desired = np.asarray(LatticeIIR([0.45], [1.0, 0.0]).process(x), dtype=float)

    adaptive = AdaptiveLatticeLadderNLMS(
        [0.0],
        [1.0, 0.0],
        mu_taps=0.0,
        mu_reflection=0.04,
        reflection_update_period=8,
    )
    y, error = adaptive.process_adapt(x, desired)

    assert y.shape == x.shape
    assert np.mean(error[-500:] ** 2) < 0.35 * np.mean(error[:500] ** 2)
    assert abs(adaptive.reflection[0]) < 1.0 - adaptive.margin


def test_scaled_reflection_period_improves_decimated_denominator_learning():
    rng = np.random.default_rng(2028)
    x = rng.normal(size=6000)
    desired = np.asarray(LatticeIIR([0.45], [1.0, 0.0]).process(x), dtype=float)

    common = dict(
        initial_reflection=[0.0],
        initial_taps=[1.0, 0.0],
        mu_taps=0.0,
        mu_reflection=0.004,
        reflection_update_period=8,
    )
    fixed = AdaptiveLatticeLadderNLMS(**common, scale_reflection_mu_by_period=False)
    scaled = AdaptiveLatticeLadderNLMS(**common, scale_reflection_mu_by_period=True)

    _, fixed_error = fixed.process_adapt(x, desired)
    _, scaled_error = scaled.process_adapt(x, desired)

    fixed_tail = np.mean(fixed_error[-500:] ** 2)
    scaled_tail = np.mean(scaled_error[-500:] ** 2)
    assert scaled_tail < fixed_tail
    assert abs(scaled.reflection[0] - 0.45) < abs(fixed.reflection[0] - 0.45)


def test_adaptive_process_batch_accepts_reflection_update_period():
    rng = np.random.default_rng(222)
    x = rng.normal(size=(2, 256)).astype(np.float64)
    desired = np.asarray(
        [LatticeIIR([0.2], [1.0, 0.0]).process(row) for row in x], dtype=np.float64
    )

    y, e, final_reflection, final_taps = adaptive_process_batch(
        [0.0],
        [1.0, 0.0],
        x,
        desired,
        mu_taps=0.0,
        mu_reflection=0.02,
        reflection_update_period=4,
        scale_reflection_mu_by_period=True,
        n_threads=1,
    )

    assert y.shape == x.shape
    assert e.shape == x.shape
    assert final_reflection.shape == (2, 1)
    assert final_taps.shape == (2, 2)
