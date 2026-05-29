"""AR-style prediction with stable reflection updates.

The predictor estimates x[n] from previous samples using a stable IIR model.
This is a compact example; production AR estimation should add model-order
selection, stationarity checks, and validation on held-out data.
"""

from __future__ import annotations

import numpy as np

from lattice_dsp import AdaptiveLatticeLadderNLMS, LatticeIIR

rng = np.random.default_rng(7)
innovation = rng.normal(scale=0.3, size=5000)
source = LatticeIIR([0.65, -0.35], [1.0, 0.0, 0.0])
x = np.asarray(source.process(innovation), dtype=float)

# Predict x[n] from x[n-1]. The input stream is delayed by one sample.
predictor_input = np.concatenate([[0.0], x[:-1]])
desired = x

predictor = AdaptiveLatticeLadderNLMS(
    initial_reflection=[0.0, 0.0],
    initial_taps=[0.0, 0.0, 0.0],
    mu_taps=0.05,
    mu_reflection=0.001,
    margin=1e-4,
)
errors = np.asarray(predictor.adapt_block(predictor_input.tolist(), desired.tolist()), dtype=float)

print("learned reflection:", np.round(predictor.reflection, 4).tolist())
print("learned numerator:", np.round(predictor.numerator, 4).tolist())
print("initial prediction MSE:", float(np.mean(errors[:500] ** 2)))
print("final prediction MSE:", float(np.mean(errors[-500:] ** 2)))
