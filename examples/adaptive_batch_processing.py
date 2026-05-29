"""Run independent adaptive IIR identification problems in a single C++/OpenMP call."""

import numpy as np

from lattice_dsp import LatticeIIR, adaptive_process_batch

rng = np.random.default_rng(42)
channels = 8
samples = 4000
x = rng.normal(size=(channels, samples))

true_reflection = [0.35, -0.2]
true_taps = [0.5, -0.15, 0.1]

desired = np.vstack(
    [np.asarray(LatticeIIR(true_reflection, true_taps).process(row), dtype=float) for row in x]
)

y, error, final_reflection, final_taps = adaptive_process_batch(
    [0.0, 0.0],
    [0.0, 0.0, 0.0],
    x,
    desired,
    mu_taps=0.08,
    mu_reflection=0.002,
    n_threads=0,
)

print("initial MSE:", np.mean(error[:, :500] ** 2))
print("final MSE:", np.mean(error[:, -500:] ** 2))
print("mean final reflection:", final_reflection.mean(axis=0))
print("mean final taps:", final_taps.mean(axis=0))
