import numpy as np

from lattice_dsp import (
    LatticeIIR,
    LatticeLadderIIR,
    ladder_to_numerator,
    numerator_to_ladder,
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


def test_ladder_numerator_roundtrip():
    reflection = [0.35, -0.2, 0.1]
    ladder = [0.4, -0.3, 0.2, 0.1]
    numerator = ladder_to_numerator(reflection, ladder)
    restored = numerator_to_ladder(reflection, numerator)
    np.testing.assert_allclose(restored, ladder, atol=1e-12)


def test_lattice_ladder_matches_transfer_function():
    rng = np.random.default_rng(123)
    x = rng.normal(size=1024)
    reflection = [0.4, -0.25, 0.12]
    ladder = [0.3, -0.2, 0.05, 0.8]
    numerator = ladder_to_numerator(reflection, ladder)
    denominator = reflection_to_denominator(reflection)

    f = LatticeLadderIIR(reflection, ladder)
    y = f.process(x)
    y_ref = reference_lfilter(numerator, denominator, x)
    np.testing.assert_allclose(y, y_ref, atol=1e-12)
    np.testing.assert_allclose(f.numerator, numerator, atol=1e-12)


def test_lattice_realization_matches_direct_realization_with_converted_taps():
    rng = np.random.default_rng(456)
    x = rng.normal(size=2048)
    reflection = [0.25, -0.15, 0.05]
    direct_taps = [0.2, -0.1, 0.0, 0.75]
    ladder = numerator_to_ladder(reflection, direct_taps)

    direct = LatticeIIR(reflection, direct_taps)
    lattice = LatticeLadderIIR(reflection, ladder)
    np.testing.assert_allclose(lattice.numerator, direct_taps, atol=1e-12)
    np.testing.assert_allclose(lattice.process(x), direct.process(x), atol=1e-12)


def test_process_batch_lattice_realization_matches_direct():
    rng = np.random.default_rng(789)
    x = rng.normal(size=(8, 256))
    reflection = [0.2, -0.1]
    taps = [0.3, 0.1, 0.8]

    y_direct = process_batch(reflection, taps, x, realization="direct")
    y_lattice = process_batch(reflection, taps, x, realization="lattice")
    np.testing.assert_allclose(y_lattice, y_direct, atol=1e-12)
