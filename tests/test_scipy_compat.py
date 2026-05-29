import numpy as np
import pytest

from lattice_dsp import LatticeIIR, reflection_to_denominator

scipy_signal = pytest.importorskip("scipy.signal")


def test_lattice_iir_matches_scipy_lfilter():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=2048)
    reflection = [0.55, -0.22, 0.08]
    numerator = [0.1, -0.05, 0.2, 0.7]
    denominator = reflection_to_denominator(reflection)

    ours = LatticeIIR(reflection, numerator).process(x)
    scipy = scipy_signal.lfilter(numerator, denominator, x)

    np.testing.assert_allclose(ours, scipy, atol=1e-11, rtol=1e-11)
