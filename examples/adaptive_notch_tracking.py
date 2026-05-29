"""Adaptive notch tracking demo.

Run after installing the package:
    python examples/adaptive_notch_tracking.py
"""

import numpy as np

from lattice_dsp import AdaptiveNotch

rng = np.random.default_rng(123)
fs = 8_000
n = np.arange(8_000)
theta_true = 0.31 * np.pi
frequency_hz = theta_true * fs / (2 * np.pi)

x = np.sin(theta_true * n) + 0.05 * rng.normal(size=n.size)
notch = AdaptiveNotch(theta=0.8, pole_radius=0.98, mu=0.005)
y = notch.process(x)

print("true frequency [Hz]:     ", round(frequency_hz, 2))
print("estimated frequency [Hz]:", round(notch.theta * fs / (2 * np.pi), 2))
print("input RMS:               ", float(np.sqrt(np.mean(x * x))))
print("output RMS:              ", float(np.sqrt(np.mean(y[-2000:] * y[-2000:]))))
