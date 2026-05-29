# Contributing

This project is currently a research-oriented alpha toolkit for stable lattice/lattice-ladder DSP filters, adaptive recursive filtering, finite model-reduction diagnostics, and matrix/MIMO lattice experiments.
The most valuable contributions are:

- correctness tests against known references,
- stable adaptive IIR algorithms,
- careful C++ implementations with pybind11 bindings,
- application examples for prediction, equalization, and system identification,
- reproducible benchmarks.

Before submitting algorithm code, please include:

1. a short reference or derivation in comments or docs,
2. tests for numerical behavior,
3. an example that shows the intended use case.

Please keep implementation work clean-room. Do not copy code from sources without
an explicit compatible license.
