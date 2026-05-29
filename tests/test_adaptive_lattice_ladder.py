import numpy as np

from lattice_dsp import AdaptiveLatticeLadderNLMS, LatticeIIR


def test_adaptive_lattice_reflection_updates_reduce_all_pole_error():
    rng = np.random.default_rng(123)
    x = rng.normal(size=4000)

    target = LatticeIIR([0.45], [1.0, 0.0])
    desired = np.asarray(target.process(x), dtype=float)

    adaptive = AdaptiveLatticeLadderNLMS(
        [0.0],
        [1.0, 0.0],
        mu_taps=0.0,
        mu_reflection=0.02,
        margin=1e-4,
    )
    errors = np.asarray(adaptive.adapt_block(x.tolist(), desired.tolist()), dtype=float)

    assert np.mean(errors[-500:] ** 2) < 0.1 * np.mean(errors[:500] ** 2)
    assert abs(adaptive.reflection[0] - 0.45) < 0.03
    assert abs(adaptive.reflection[0]) < 1.0


def test_adaptive_lattice_freeze_reflection_keeps_denominator_fixed():
    rng = np.random.default_rng(321)
    x = rng.normal(size=1024)
    desired = 0.75 * x

    adaptive = AdaptiveLatticeLadderNLMS(
        [0.2],
        [0.0, 0.0],
        mu_taps=0.4,
        mu_reflection=0.2,
        freeze_reflection=True,
    )
    initial_reflection = np.asarray(adaptive.reflection, dtype=float)
    errors = np.asarray(adaptive.adapt_block(x.tolist(), desired.tolist()), dtype=float)

    np.testing.assert_allclose(adaptive.reflection, initial_reflection, atol=1e-15)
    assert np.mean(errors[-200:] ** 2) < np.mean(errors[:200] ** 2)


def test_adaptive_lattice_raw_mapping_keeps_near_boundary_stable():
    rng = np.random.default_rng(456)
    x = rng.normal(size=512)
    desired = np.zeros_like(x)

    adaptive = AdaptiveLatticeLadderNLMS(
        [0.95, -0.9],
        [1.0, 0.0, 0.0],
        mu_taps=0.0,
        mu_reflection=0.5,
        margin=1e-3,
    )
    _ = adaptive.adapt_block(x.tolist(), desired.tolist())

    assert all(abs(k) < 1.0 - adaptive.margin for k in adaptive.reflection)
    assert all(np.isfinite(k) for k in adaptive.raw_reflection)
    assert len(adaptive.last_raw_gradient) == 2
