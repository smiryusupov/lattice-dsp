"""Burg and Levinson-Durbin AR estimation demo."""

from __future__ import annotations

import numpy as np

from lattice_dsp import (
    LatticeIIR,
    autocorrelation,
    burg_reflection,
    levinson_durbin_reflection,
    reflection_to_denominator,
)


def main() -> None:
    rng = np.random.default_rng(123)
    n = 12000
    true_reflection = [0.62, -0.34, 0.18, -0.08]
    # AR process: white excitation through a stable all-pole lattice/IIR.
    excitation = rng.normal(scale=0.2, size=n)
    ar = LatticeIIR(true_reflection, [1.0, 0.0, 0.0, 0.0, 0.0])
    x = np.asarray(ar.process(excitation), dtype=float)
    x = x[1000:]  # drop startup transient

    order = 4
    r = autocorrelation(x, order, True)
    k_lev = np.asarray(levinson_durbin_reflection(r, order))
    k_burg = np.asarray(burg_reflection(x, order))

    print("true reflection:     ", np.round(true_reflection, 4))
    print("Levinson reflection: ", np.round(k_lev, 4))
    print("Burg reflection:     ", np.round(k_burg, 4))
    print("Burg denominator:    ", np.round(reflection_to_denominator(k_burg), 4))


if __name__ == "__main__":
    main()
