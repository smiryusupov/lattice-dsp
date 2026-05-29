"""Tune the adaptive reflection-update period on one identification problem."""

from __future__ import annotations

import numpy as np

import lattice_dsp

rng = np.random.default_rng(1234)
x = rng.normal(size=20_000)

# Synthetic target for the example. In a real system, ``desired`` is measured.
target = lattice_dsp.LatticeIIR([0.35, -0.25, 0.15, -0.08], [0.2, -0.1, 0.05, 0.0, 0.75])
desired = np.asarray(target.process(x), dtype=float)

result = lattice_dsp.tune_reflection_update_period(
    x,
    desired,
    periods=[1, 2, 4, 8, 16, 32],
    order=4,
    max_tail_mse_ratio=1.5,
    max_worst_tail_mse_ratio=2.0,
)

print("recommended period:", result["recommended_period"])
print("scope:", result["metadata"]["recommendation_scope"])
print("warnings:", result["warnings"])
print("recommended row:", result["recommended"])

# Robust validation mode: stack multiple independent validation trials as rows.
X = np.vstack([x, rng.normal(size=x.size), rng.normal(size=x.size)])
D = np.vstack([desired, target.process(X[1]), target.process(X[2])]).astype(float)
robust = lattice_dsp.tune_reflection_update_period(
    X,
    D,
    periods=[1, 2, 4, 8, 16, 32],
    order=4,
    max_tail_mse_ratio=1.5,
    max_worst_tail_mse_ratio=2.0,
)
print("robust recommended period:", robust["recommended_period"])
print("robust scope:", robust["metadata"]["recommendation_scope"])
