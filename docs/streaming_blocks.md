# Streaming and block helpers

The core filter classes are stateful and can already process block-by-block.
`lattice_dsp.streaming` adds small Python wrappers that make streaming demos
clear without moving the inner loops out of C++.

```python
from lattice_dsp import BlockProcessor, AdaptiveBlockProcessor

fixed = BlockProcessor(reflection, taps)
y0 = fixed.process(x0)
y1 = fixed.process(x1)

adaptive = AdaptiveBlockProcessor(reflection, initial_taps, kind="rls")
result0 = adaptive.process_adapt(x0, d0)
result1 = adaptive.process_adapt(x1, d1)
```

Available adaptive kinds:

- `nlms`: fixed-denominator NLMS numerator adaptation.
- `rls`: fixed-denominator RLS numerator adaptation.
- `adaptive_iir`: experimental stable denominator + numerator adaptation.

For offline multi-channel work, prefer C++ batch functions such as
`process_batch`, `adaptive_process_batch`, and `rls_process_batch`.
