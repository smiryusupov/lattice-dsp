"""Installed-wheel smoke test for cibuildwheel.

This intentionally avoids importing repository-only modules such as
``examples`` or ``benchmarks``. The full source-tree test suite runs in the
regular CI workflow; wheel jobs only need to prove that the built wheel imports
and that representative compiled-extension paths execute.
"""

from __future__ import annotations

import pathlib

import numpy as np

import lattice_dsp as ld
import lattice_dsp._core as core


def main() -> None:
    print("lattice_dsp=", pathlib.Path(ld.__file__).resolve())
    print("_core=", pathlib.Path(core.__file__).resolve())
    print("version=", ld.__version__)
    print("HAS_OPENMP=", ld.HAS_OPENMP)

    reflection = [0.2, -0.1]
    numerator = [0.5, 0.1, -0.05]
    x = np.linspace(-1.0, 1.0, 16)

    filt = ld.LatticeIIR(reflection, numerator)
    y_single = np.asarray(filt.process(x))
    assert y_single.shape == x.shape
    assert np.all(np.isfinite(y_single))

    y_batch = np.asarray(ld.process_batch(reflection, numerator, np.vstack([x, x])))
    assert y_batch.shape == (2, x.size)
    np.testing.assert_allclose(y_batch[0], y_single, atol=1e-12)

    denominator = ld.reflection_to_denominator(reflection)
    restored = ld.denominator_to_reflection(denominator)
    np.testing.assert_allclose(restored, reflection, atol=1e-12)

    singular_values = np.asarray(ld.hankel_singular_values([1.0, 0.3, -0.1, 0.05], 2, 2))
    assert singular_values.shape == (2,)
    assert np.all(singular_values >= -1e-12)

    identity = np.eye(1)
    markov = np.asarray(
        ld.mimo_state_space_markov_response(identity * 0.2, identity, identity, identity * 0.0, 3)
    )
    assert markov.shape == (3, 1, 1)
    assert np.all(np.isfinite(markov))


if __name__ == "__main__":
    main()
