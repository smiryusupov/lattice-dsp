# Applications layer

The public package is centered on stable lattice/IIR DSP.  Application helpers are intentionally small and dependency-light.

## What belongs in the package

Good examples for this repository:

- stable IIR system identification;
- adaptive notch tracking;
- AR/prediction experiments;
- compact channel equalization prototypes;
- batched simulation and filter-bank processing;
- synthetic echo-like metric demos.

## Scope boundary

This package is not a production acoustic echo canceller.  Real AEC needs delay tracking, double-talk detection, long echo-path modeling, nonlinear handling, residual echo suppression, AGC/noise integration, and careful speech-preservation evaluation.

The included `HybridEchoCanceller` is therefore best understood as:

```text
reference -> stable adaptive lattice/IIR -> echo estimate
microphone - echo estimate -> residual
optional small residual processor -> enhanced output
```

That pattern is useful for experiments, but it is not a complete telephony stack.

## Minimal usage

```python
from lattice_dsp import HybridEchoCanceller

canceller = HybridEchoCanceller(
    initial_reflection=[0.0, 0.0, 0.0, 0.0],
    initial_taps=[0.0, 0.0, 0.0, 0.0, 0.0],
    reflection_update_period=8,
)

result = canceller.process(reference, microphone, clean_target=clean_target)
print(result.metrics)
```

Use `docs/echo_cancellation.md` for ERLE and metric interpretation.
