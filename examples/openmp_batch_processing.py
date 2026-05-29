"""Multi-channel batch processing demo.

Rows are independent channels and can be parallelized by OpenMP in the C++ core.
"""

import numpy as np

from lattice_dsp import HAS_OPENMP, process_batch

rng = np.random.default_rng(7)
channels = 64
samples = 50_000
x = rng.normal(size=(channels, samples))

reflection = [0.3, -0.2, 0.1]
taps = [0.1, -0.05, 0.2, 0.9]
y = process_batch(reflection, taps, x, n_threads=0)

print("OpenMP enabled:", HAS_OPENMP)
print("input shape:   ", x.shape)
print("output shape:  ", y.shape)
print("output RMS:    ", float(np.sqrt(np.mean(y * y))))
