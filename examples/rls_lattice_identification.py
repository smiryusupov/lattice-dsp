"""Fixed-denominator RLS adaptation of stable lattice/IIR numerator taps."""

from __future__ import annotations

import numpy as np

from lattice_dsp import LatticeIIR, LatticeLadderNLMS, LatticeLadderRLS


def main() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    reflection = [0.55, -0.25]
    target_taps = [0.25, -0.15, 0.65]
    desired = np.asarray(LatticeIIR(reflection, target_taps).process(x), dtype=float)

    nlms = LatticeLadderNLMS(reflection, [0.0, 0.0, 0.0], mu=0.12)
    rls = LatticeLadderRLS(reflection, [0.0, 0.0, 0.0], forgetting_factor=0.995)

    _, e_nlms = nlms.process_adapt(x, desired)
    _, e_rls = rls.process_adapt(x, desired)

    print("target taps:", np.round(target_taps, 4))
    print("NLMS taps:  ", np.round(nlms.taps, 4))
    print("RLS taps:   ", np.round(rls.taps, 4))
    print("tail MSE NLMS:", float(np.mean(np.asarray(e_nlms)[-1000:] ** 2)))
    print("tail MSE RLS: ", float(np.mean(np.asarray(e_rls)[-1000:] ** 2)))


if __name__ == "__main__":
    main()
