"""Fixed stable lattice denominator + adaptive ladder taps demo.

Run after installing the package:
    python examples/system_identification.py
"""

import numpy as np

from lattice_dsp import LatticeIIR, LatticeLadderNLMS


rng = np.random.default_rng(42)
n = 4000
x = rng.normal(size=n)

# Synthetic unknown stable lattice-ladder system.
true_reflection = [0.35, -0.25]
true_taps = [0.15, -0.2, 0.8]
plant = LatticeIIR(true_reflection, true_taps)
d = plant.process(x) + 0.01 * rng.normal(size=n)

# Estimate ladder taps while keeping the stable denominator fixed.
adaptive = LatticeLadderNLMS(true_reflection, [0.0, 0.0, 0.0], mu=0.2)
errors = np.array(adaptive.adapt_block(x.tolist(), d.tolist()))

print("true taps:     ", true_taps)
print("estimated taps:", [round(v, 5) for v in adaptive.taps])
print("initial MSE:   ", float(np.mean(errors[:500] ** 2)))
print("final MSE:     ", float(np.mean(errors[-500:] ** 2)))
