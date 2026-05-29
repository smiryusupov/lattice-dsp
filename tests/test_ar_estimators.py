from __future__ import annotations

import numpy as np

from lattice_dsp import (
    LatticeIIR,
    autocorrelation,
    burg_reflection,
    levinson_durbin_reflection,
    reflection_to_denominator,
)


def test_levinson_and_burg_return_stable_reflection_coefficients():
    rng = np.random.default_rng(123)
    x = rng.normal(size=4096)
    y = np.asarray(LatticeIIR([0.45, -0.2], [1.0, 0.0, 0.0]).process(x), dtype=float)

    r = autocorrelation(y, 2)
    k_lev = np.asarray(levinson_durbin_reflection(r, 2), dtype=float)
    k_burg = np.asarray(burg_reflection(y, 2), dtype=float)

    assert k_lev.shape == (2,)
    assert k_burg.shape == (2,)
    assert np.all(np.abs(k_lev) < 1.0)
    assert np.all(np.abs(k_burg) < 1.0)
    assert len(reflection_to_denominator(k_burg)) == 3


def test_burg_tracks_known_stable_ar_process():
    rng = np.random.default_rng(456)
    true_reflection = np.asarray([0.62, -0.34, 0.18, -0.08], dtype=float)
    excitation = rng.normal(scale=0.2, size=20000)
    ar = LatticeIIR(true_reflection.tolist(), [1.0, 0.0, 0.0, 0.0, 0.0])
    x = np.asarray(ar.process(excitation), dtype=float)[2000:]

    k_burg = np.asarray(burg_reflection(x, 4), dtype=float)
    r = autocorrelation(x, 4, True)
    k_lev = np.asarray(levinson_durbin_reflection(r, 4), dtype=float)

    assert np.all(np.abs(k_burg) < 1.0)
    np.testing.assert_allclose(k_burg, true_reflection, atol=0.04)
    np.testing.assert_allclose(k_burg, k_lev, atol=0.04)
