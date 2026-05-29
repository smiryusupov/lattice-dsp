"""Reflection/PARCOR <-> denominator conversion demo."""

import numpy as np

from lattice_dsp import denominator_to_reflection, reflection_to_denominator

reflection = [0.7, -0.4, 0.25]
denominator = reflection_to_denominator(reflection)
restored = denominator_to_reflection(denominator)
poles = np.roots(denominator)

print("reflection:", reflection)
print("denominator:", [round(v, 6) for v in denominator])
print("restored:", [round(v, 6) for v in restored])
print("max pole radius:", float(np.max(np.abs(poles))))
