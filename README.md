# lattice-dsp

[![PyPI](https://img.shields.io/pypi/v/lattice-dsp.svg)](https://pypi.org/project/lattice-dsp/)
[![Python](https://img.shields.io/pypi/pyversions/lattice-dsp.svg)](https://pypi.org/project/lattice-dsp/)
[![Docs](https://readthedocs.org/projects/lattice-dsp/badge/?version=latest)](https://lattice-dsp.readthedocs.io/en/latest/)
[![CI](https://github.com/smiryusupov/lattice-dsp/actions/workflows/ci.yml/badge.svg)](https://github.com/smiryusupov/lattice-dsp/actions/workflows/ci.yml)

**Stable IIR lattice filters, adaptive recursive DSP, and matrix/MIMO lattice experiments for Python.**

`lattice-dsp` is a compact C++/Python toolkit for signal-processing workflows where stability-aware recursive filters matter. It focuses on reflection/PARCOR coefficients, lattice and lattice-ladder IIR filters, adaptive IIR experiments, AR spectral diagnostics, finite-Hankel model-reduction diagnostics, and experimental matrix/MIMO lattice tools.

Install with `lattice-dsp`; import with `lattice_dsp`.

```bash
python -m pip install lattice-dsp
```

```python
import lattice_dsp as ld
```

## Why lattice-dsp?

Direct-form IIR denominator updates can move poles outside the unit circle. Lattice parameterizations provide a safer coordinate system: bounded reflection coefficients correspond to stable recursive filters.

This package is intentionally narrower than SciPy or general audio/room-simulation libraries. It is for users who want stable recursive DSP building blocks, finite-section model-reduction diagnostics, and experimental MIMO/state-space lattice workflows in one tested Python package with a compiled backend.

## What is included

- Stable scalar `LatticeIIR` and `LatticeLadderIIR` filters.
- Reflection/PARCOR coefficient conversion for stable IIR denominators.
- Adaptive lattice-ladder NLMS/RLS-style examples.
- Burg, Levinson-Durbin, AR, and spectral diagnostics.
- Batch processing for many independent filters/signals, with optional OpenMP.
- Finite-Hankel SISO/MIMO reduction helpers and Nehari/AAK-style finite-section diagnostics.
- MIMO Markov tensor and state-space processing utilities.
- Matrix-lattice/all-pass examples and causal online MIMO lattice prediction.
- Tangential Schur/Pick/J-inner diagnostic helpers.
- Sphinx tutorials, examples, benchmarks, and interoperability recipes.

## Quick example: stable IIR from reflection coefficients

```python
import numpy as np
import lattice_dsp as ld

rng = np.random.default_rng(0)
x = rng.normal(size=4096)

reflection = [0.35, -0.2, 0.1]
numerator = [0.25, 0.0, 0.15, 0.7]

print(ld.reflection_to_denominator(reflection))

filt = ld.LatticeIIR(reflection, numerator)
y = filt.process(x)
```

## Quick example: adaptive stable IIR identification

```python
import numpy as np
import lattice_dsp as ld

rng = np.random.default_rng(1)
x = rng.normal(size=8000)

target = ld.LatticeIIR([0.55, -0.32], [0.4, -0.1, 0.65])
desired = np.asarray(target.process(x), dtype=float)

adaptive = ld.AdaptiveLatticeLadderNLMS(
    initial_reflection=[0.0, 0.0],
    initial_taps=[0.0, 0.0, 0.0],
    mu_taps=0.06,
    mu_reflection=0.0015,
    margin=1e-4,
)

error = np.asarray(adaptive.adapt_block(x, desired), dtype=float)

print("learned reflection:", np.round(adaptive.reflection, 4))
print("final MSE:", np.mean(error[-1000:] ** 2))
```

## MIMO and model-reduction scope

`lattice-dsp` includes MIMO/state-space and matrix-lattice tools, but the scope is deliberately conservative.

Implemented:

- finite-Hankel SISO reduction as a C++ finite-section Ho-Kalman-style baseline;
- MIMO block-Hankel reduction from Markov tensors;
- compiled batched MIMO state-space simulation;
- causal online MIMO lattice prediction from fitted matrix reflection coefficients;
- matrix-lattice/all-pass runtime examples and finite-record adjoint diagnostics;
- finite definite tangential Schur/Pick and J-inner diagnostic utilities.

Out of scope for the current alpha:

- full infinite-dimensional AAK/Nehari solvers;
- full matrix-valued AAK/Nehari optimal reduction;
- generalized indefinite matrix Schur synthesis;
- production acoustic echo cancellation;
- complete room-acoustics, wireless, or control-system frameworks.

The docs label offline estimation, causal runtime, inductive evaluation, and transductive/block diagnostics separately so examples are clear about data use.

## Documentation

Start here:

- [Documentation](https://lattice-dsp.readthedocs.io/en/latest/)
- [Learning path](docs/learning_path.rst)
- [Interoperability recipes](docs/interoperability.rst)
- [Algorithm-selection guide](docs/theory/choosing_algorithms.rst)
- [Hardy/Hankel/state-space guide](docs/theory/hardy_hankel_state_space.rst)
- [API reference](docs/api.rst)

Representative examples:

- [`examples/reflection_conversion.py`](examples/reflection_conversion.py)
- [`examples/lattice_ladder_realization.py`](examples/lattice_ladder_realization.py)
- [`examples/adaptive_iir_system_identification.py`](examples/adaptive_iir_system_identification.py)
- [`examples/periodogram_vs_ar_spectrum.py`](examples/periodogram_vs_ar_spectrum.py)
- [`examples/causal_mimo_lattice_prediction.py`](examples/causal_mimo_lattice_prediction.py)
- [`examples/mimo_coupled_model_reduction.py`](examples/mimo_coupled_model_reduction.py)
- [`examples/matrix_lattice_allpass.py`](examples/matrix_lattice_allpass.py)

## Development

```bash
python -m pip install -e ".[dev,examples,benchmark,docs]"
pytest
ruff check .
ruff format .
```

Build docs locally:

```bash
./scripts/build_docs_with_results.sh
xdg-open docs/_build/html/index.html
```

The native extension is built with `scikit-build-core`, CMake, pybind11, NumPy, and optional OpenMP. If OpenMP is unavailable, the package falls back to serial batch processing.

## Public validation

For a full release-style check:

```bash
python -m pip install -e ".[dev,examples,benchmark,docs]"
./scripts/release_validation.sh
```

For a quicker public-claim check:

```bash
PYTHONPATH="$PWD" pytest -q tests/test_release_trust_claims.py
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).

## Citation

If this package supports your research or teaching, please cite it using [`CITATION.cff`](CITATION.cff).