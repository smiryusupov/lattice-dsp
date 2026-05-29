# Contributing and development

This page is for contributors who want to build `lattice-dsp` from source,
run checks locally, or add a feature. It describes the public development
workflow rather than any private maintainer process.

## Local setup

```bash
python -m pip install -e ".[dev,examples,benchmark]"
pytest
pre-commit install
```

The quoted extras form works in shells such as `zsh` that otherwise treat square
brackets as glob patterns.

The package is built with scikit-build-core, CMake, and pybind11. OpenMP is used
when the compiler and platform provide it. If OpenMP is unavailable, the package
still builds and uses serial batch processing for the affected routines.

## Linting and formatting

```bash
ruff check .
ruff format .
```

The repository includes `.pre-commit-config.yaml` so Ruff checks can run through
`pre-commit` before commits.

## Where changes usually belong

Most low-level filtering changes touch the C++ layer first and then the Python
API surface:

1. declare or update C++ APIs in `include/lattice_dsp/lattice.hpp`,
2. implement them in `src/lattice.cpp`,
3. expose Python bindings in `src/bindings.cpp`,
4. add Python regression tests in `tests/`, and
5. add a small example or benchmark when the change affects user-visible
   behavior or performance.

Pure-Python algorithms, documentation-only changes, and examples may not need
all of these steps.

## Numerical stability expectations

Adaptive recursive filters can become hard to test and explain if intermediate
IIR models become unstable. Public APIs should therefore prefer one of these
stability strategies:

- reflection/PARCOR coefficients constrained by `|k| < 1`,
- unconstrained raw parameters mapped through `tanh`, or
- explicit projection policies for algorithms that update denominator-like
  parameters directly.

When a new method can produce unstable intermediate filters, document that
behavior and add tests that make the expected failure or projection mode clear.

## Parallelism expectations

OpenMP should be used for independent rows, independent signals, or independent
filter instances. A single recursive IIR stream should not be described as
sample-parallel unless a specific algorithmic transformation makes that true.

## Clean-room contributions

Do not copy third-party source code unless its license is compatible and the
attribution requirements are clear. Implement algorithms from public
specifications, papers, books, and independently written tests.
