# Fixed-denominator RLS lattice adaptation

`LatticeLadderRLS` is a fast C++ RLS adaptive filter for numerator/ladder taps
with a fixed stable denominator parameterized by reflection coefficients.

This is intentionally scoped: it is a serious adaptive-filter baseline, not a
complete acoustic echo canceller. Stability comes from the reflection-coded
recursive denominator; RLS updates only the feed-forward numerator basis.

```python
from lattice_dsp import LatticeLadderRLS

adaptive = LatticeLadderRLS(
    reflection=[0.5, -0.2],
    initial_taps=[0.0, 0.0, 0.0],
    forgetting_factor=0.995,
    initial_inverse_covariance=1000.0,
)
y, e = adaptive.process_adapt(x, desired)
```

For many independent channels, use the OpenMP-enabled batch function:

```python
from lattice_dsp import rls_process_batch

y, e, final_taps = rls_process_batch(reflection, initial_taps, x_batch, d_batch)
```

Input batches are channel-by-sample matrices.
