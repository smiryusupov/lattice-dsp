# Adaptive stable IIR workflows

Adaptive IIR filters are attractive because they can model resonant or recursive systems with fewer parameters than long FIR filters.  The difficulty is stability: direct updates to denominator coefficients can move poles outside the unit circle.

`lattice-dsp` avoids this failure mode by using reflection/PARCOR coefficients.  For an all-pole denominator produced by the Schur/step-up recursion, the sufficient stability condition is simple:

```text
|k_i| < 1 for every reflection coefficient k_i
```

The adaptive API keeps unconstrained raw variables internally and maps them through a bounded `tanh` transform:

```text
k_i = (1 - margin) tanh(raw_i)
```

This does not make every adaptive experiment successful, but it prevents a common and severe failure: denominator instability caused by unconstrained direct coefficient updates.

## Key classes

- `LatticeIIR`: stable reflection-parameterized IIR filter.
- `LatticeLadderIIR`: synthesis lattice-ladder realization.
- `LatticeLadderNLMS`: fixed-denominator adaptive numerator/ladder NLMS.
- `AdaptiveLatticeLadderNLMS`: experimental adaptive numerator plus bounded reflection updates.
- `AdaptiveNotch`: small second-order adaptive notch filter.

## Recommended examples

```bash
python examples/stability_vs_direct_iir.py
python examples/adaptive_iir_system_identification.py
python examples/tracking_drifting_iir_system.py
python examples/adaptive_prediction_ar.py
python examples/ar_spectral_estimation.py
```

## Practical tuning guidance

- Start with numerator-only adaptation (`LatticeLadderNLMS`) when possible.
- Use small `mu_reflection`; denominator updates are more sensitive than numerator updates.
- Use `reflection_update_period > 1` to reduce gradient cost once behavior is stable.
- Enable `scale_reflection_mu_by_period=True` when skipping reflection updates.
- Track final MSE and stability margin, not only one scalar score.

## What stable parameterization does not solve

Reflection-bounded updates guarantee denominator stability, not task success.  They do not automatically solve:

- poor excitation;
- wrong model order;
- long sparse impulse responses better modeled by FIR/partitioned FIR;
- nonlinear systems;
- delay/reference mismatch;
- nonstationary targets that change faster than the adaptation can track.
