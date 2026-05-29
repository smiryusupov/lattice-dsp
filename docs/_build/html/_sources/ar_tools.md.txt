# Burg and Levinson-Durbin AR tools

`lattice-dsp` includes C++/pybind11 implementations of common AR estimation
utilities that fit naturally with reflection/PARCOR coefficients:

- `autocorrelation(x, max_lag, biased=True)`
- `levinson_durbin_reflection(r, order)`
- `levinson_durbin_denominator(r, order)`
- `levinson_durbin_error(r, order)`
- `burg_reflection(x, order)`
- `burg_denominator(x, order)`

The reflection outputs can be passed directly to `LatticeIIR` or converted with
`reflection_to_denominator`. This keeps AR models stable by construction.

```python
import numpy as np
from lattice_dsp import autocorrelation, levinson_durbin_reflection, burg_reflection

x = np.random.default_rng(0).normal(size=4096)
r = autocorrelation(x, 8)
k_levinson = levinson_durbin_reflection(r, 8)
k_burg = burg_reflection(x, 8)
```

Burg is useful when you want a stable AR estimate directly from a short signal.
Levinson-Durbin is useful when you already have an autocorrelation estimate.
