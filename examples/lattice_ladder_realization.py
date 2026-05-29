"""Compare direct-form and synthesis lattice-ladder realizations.

This example starts from SciPy-style direct numerator taps, converts them to
ladder taps for the true synthesis lattice-ladder structure, and verifies that
both realizations produce the same transfer-function behavior.
"""

from __future__ import annotations

import numpy as np

from lattice_dsp import LatticeIIR, LatticeLadderIIR, numerator_to_ladder


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=2048)

    reflection = [0.35, -0.2, 0.1]
    direct_numerator = [0.2, -0.1, 0.05, 0.75]
    ladder_taps = numerator_to_ladder(reflection, direct_numerator)

    direct = LatticeIIR(reflection, direct_numerator)
    lattice = LatticeLadderIIR(reflection, ladder_taps)

    y_direct = direct.process(x)
    y_lattice = lattice.process(x)
    max_abs_err = float(np.max(np.abs(y_direct - y_lattice)))

    print("reflection:", reflection)
    print("direct numerator:", direct_numerator)
    print("ladder taps:", ladder_taps)
    print("lattice numerator reconstruction:", lattice.numerator)
    print(f"max |direct - lattice|: {max_abs_err:.3e}")


if __name__ == "__main__":
    main()
