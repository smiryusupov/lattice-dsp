"""Adaptive stable IIR system identification.

This example identifies both numerator taps and reflection coefficients. The
reflection updates are performed through unconstrained raw variables that are
mapped with tanh, so the learned denominator remains stable during adaptation.
"""

from __future__ import annotations

import numpy as np

from lattice_dsp import AdaptiveLatticeLadderNLMS, LatticeIIR

rng = np.random.default_rng(42)
x = rng.normal(size=6000)

true_reflection = [0.45, -0.25]
true_numerator = [0.4, -0.1, 0.7]
target = LatticeIIR(true_reflection, true_numerator)
desired = np.asarray(target.process(x), dtype=float)

adaptive = AdaptiveLatticeLadderNLMS(
    initial_reflection=[0.0, 0.0],
    initial_taps=[0.0, 0.0, 0.0],
    mu_taps=0.08,
    mu_reflection=0.002,
    margin=1e-4,
)
errors = np.asarray(adaptive.adapt_block(x.tolist(), desired.tolist()), dtype=float)

print("target reflection:", true_reflection)
print("learned reflection:", np.round(adaptive.reflection, 4).tolist())
print("target numerator:", true_numerator)
print("learned numerator:", np.round(adaptive.numerator, 4).tolist())
print("initial MSE:", float(np.mean(errors[:500] ** 2)))
print("final MSE:", float(np.mean(errors[-500:] ** 2)))
