from __future__ import annotations

import numpy as np

from lattice_dsp import LatticeIIR, LatticeLadderRLS, rls_process_batch


def test_lattice_ladder_rls_identifies_fixed_denominator_numerator():
    rng = np.random.default_rng(5)
    x = rng.normal(size=3000)
    reflection = [0.4, -0.15]
    target_taps = np.array([0.35, -0.2, 0.55])
    desired = np.asarray(LatticeIIR(reflection, target_taps).process(x), dtype=float)

    rls = LatticeLadderRLS(reflection, [0.0, 0.0, 0.0], forgetting_factor=0.995)
    _, e = rls.process_adapt(x, desired)

    assert np.mean(np.asarray(e)[-500:] ** 2) < 1e-6
    assert np.allclose(rls.taps, target_taps, atol=2e-2)


def test_rls_process_batch_shapes():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(3, 512))
    reflection = [0.3]
    desired = x.copy()
    y, e, taps = rls_process_batch(reflection, [0.0, 0.0], x, desired)
    assert y.shape == x.shape
    assert e.shape == x.shape
    assert taps.shape == (3, 2)
