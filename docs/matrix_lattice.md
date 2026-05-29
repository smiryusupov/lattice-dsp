# Matrix lattice / MIMO all-pass utilities

`lattice-dsp` v1.0 adds a matrix-valued lattice/all-pass foundation.  This is a
natural extension of scalar reflection coefficients: instead of requiring
`|k| < 1`, each matrix reflection coefficient `K_i` must be a strict
contraction:

```text
||K_i||_2 < 1
```

This condition is easy to check using singular values and gives a compact,
stability-aware parameterization for complex MIMO/time-domain DSP structures such as paraunitary filter banks, multichannel all-pass transforms, stable unitary convolutions, array processing blocks, and compact frequency-dependent matrix responses.

## What is included

- `MatrixLatticeAllPass`
- `contractive_matrix_from_raw()`
- `project_matrix_reflection()`
- `matrix_lattice_stage_blocks()`
- `unitary_polar_factor()`
- OpenMP-backed frequency-response evaluation through the C++ extension

## Scope

The package provides reusable matrix-lattice primitives, not an end-to-end
application framework.  Wireless precoding, array processing, learned unitary
convolutions, filter banks, and spatial-audio transforms can all use similar
matrix-valued all-pass ideas, but this module intentionally stays at the DSP
primitive layer.

## Example

```python
import numpy as np
from lattice_dsp import MatrixLatticeAllPass, contractive_matrix_from_raw

rng = np.random.default_rng(0)
K = [contractive_matrix_from_raw(0.3 * (rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))))]
f = MatrixLatticeAllPass(K)
omega = np.linspace(0, np.pi, 128)
G = f.frequency_response(omega)
print(f.unitarity_error(omega))
```

Run:

```bash
python examples/matrix_lattice_allpass.py
python examples/matrix_unitary_response_compression.py
python examples/paraunitary_filter_bank_demo.py
python examples/ml_unitary_convolution_demo.py
python examples/multichannel_audio_decorrelator.py
```
