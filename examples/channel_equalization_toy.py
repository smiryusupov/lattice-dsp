"""Toy fixed-denominator equalization/system-identification demo.

This is intentionally small: it estimates numerator coefficients while keeping a
stable reflection-parameterized denominator fixed. Full decision-directed channel
equalization will be a later application layer.
"""

import numpy as np

from lattice_dsp import LatticeIIR, LatticeLadderNLMS

rng = np.random.default_rng(5)
symbols = rng.choice([-1.0, 1.0], size=6000)

# A compact stable IIR channel.
channel_reflection = [0.45, -0.25]
channel_num = [0.8, 0.35, -0.1]
channel = LatticeIIR(channel_reflection, channel_num)
received = channel.process(symbols) + 0.02 * rng.normal(size=symbols.size)

# Estimate a stable inverse-ish numerator with a fixed denominator for the demo.
equalizer = LatticeLadderNLMS(channel_reflection, [0.0, 0.0, 0.0], mu=0.08)
errors = np.array(equalizer.adapt_block(received.tolist(), symbols.tolist()))

print("estimated equalizer numerator:", [round(v, 5) for v in equalizer.numerator])
print("initial MSE:", float(np.mean(errors[:500] ** 2)))
print("final MSE:", float(np.mean(errors[-500:] ** 2)))
