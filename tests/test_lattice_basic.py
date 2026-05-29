import math

import numpy as np
import pytest

from lattice_dsp import (
    AdaptiveNotch,
    LatticeIIR,
    LatticeLadderNLMS,
    bounded_reflection_from_raw,
    denominator_to_reflection,
    process_batch,
    reflection_to_denominator,
)


def reference_lfilter(b, a, x):
    """Small pure-NumPy reference for y = b/a * x, a[0] = 1."""
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    for n in range(len(x)):
        acc = 0.0
        for i in range(len(b)):
            if n - i >= 0:
                acc += b[i] * x[n - i]
        for i in range(1, len(a)):
            if n - i >= 0:
                acc -= a[i] * y[n - i]
        y[n] = acc / a[0]
    return y


def test_zero_order_identity():
    x = np.array([1.0, -2.0, 3.0, 0.5])
    f = LatticeIIR([], [1.0])
    np.testing.assert_allclose(f.process(x), x)


def test_rejects_unstable_reflection():
    with pytest.raises(ValueError):
        LatticeIIR([1.0], [0.0, 1.0])


def test_reflection_denominator_roundtrip():
    reflection = np.array([0.45, -0.2, 0.1])
    denominator = reflection_to_denominator(reflection.tolist())
    restored = denominator_to_reflection(denominator)
    np.testing.assert_allclose(restored, reflection, atol=1e-12)


def test_reflection_denominator_is_stable():
    reflection = np.array([0.7, -0.4, 0.25, -0.15])
    denominator = np.array(reflection_to_denominator(reflection.tolist()))
    poles = np.roots(denominator)
    assert np.max(np.abs(poles)) < 1.0


def test_process_matches_reference_iir():
    rng = np.random.default_rng(42)
    x = rng.normal(size=512)
    reflection = [0.35, -0.2]
    numerator = [0.5, 0.1, -0.05]
    denominator = reflection_to_denominator(reflection)
    f = LatticeIIR(reflection, numerator)
    np.testing.assert_allclose(
        f.process(x), reference_lfilter(numerator, denominator, x), atol=1e-12
    )


def test_bounded_reflection_is_stable():
    raw = [-100.0, 0.0, 100.0]
    k = bounded_reflection_from_raw(raw, margin=1e-3)
    assert all(abs(v) < 1.0 for v in k)


def test_batch_matches_channel_loop():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 128))
    reflection = [0.2, -0.1]
    taps = [0.3, 0.1, 0.8]
    y_batch = process_batch(reflection, taps, x)
    y_loop = []
    for row in x:
        f = LatticeIIR(reflection, taps)
        y_loop.append(f.process(row))
    np.testing.assert_allclose(y_batch, np.vstack(y_loop))


def test_ladder_nlms_reduces_simple_gain_error():
    rng = np.random.default_rng(1)
    x = rng.normal(size=512)
    desired = 0.7 * x
    af = LatticeLadderNLMS([], [0.0], mu=0.5)
    errors = np.array(af.adapt_block(x.tolist(), desired.tolist()))
    assert np.mean(errors[-100:] ** 2) < np.mean(errors[:100] ** 2)
    assert abs(af.taps[0] - 0.7) < 0.1


def test_adaptive_notch_tracks_single_tone():
    theta_true = 0.31 * math.pi
    n = np.arange(5000)
    x = np.sin(theta_true * n)
    notch = AdaptiveNotch(theta=0.8, pole_radius=0.98, mu=0.005)
    _ = notch.process(x)
    assert abs(notch.theta - theta_true) < 0.03
