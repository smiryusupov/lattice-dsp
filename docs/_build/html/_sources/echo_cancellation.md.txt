# Synthetic echo-cancellation metric demo

`lattice-dsp` is **not** a production acoustic echo canceller.  The echo helpers are kept as small synthetic examples for adaptive-filter experiments and metric interpretation.

The included echo demo answers a narrow question:

> If a synthetic microphone signal contains a known echo/error component, how much does an adaptive filter reduce that error power?

Run the small example:

```bash
python examples/echo_cancellation_erle_demo.py
```

Or run the JSON benchmark:

```bash
python benchmarks/echo_cancellation_benchmark.py \
  --samples 64000 \
  --sample-rate 16000 \
  --no-double-talk \
  --near-end-power-ratio 0 \
  --noise-snr-db 60 \
  --output echo-benchmark.json
```

## Synthetic signal model

`generate_echo_problem(...)` creates:

- a far-end/reference signal `x`;
- a stable IIR echo path parameterized by reflection coefficients;
- optional nonlinear residual echo;
- optional near-end/double-talk signal;
- optional background noise;
- a microphone signal containing echo plus any near-end/noise components;
- a clean target used only for metrics.

Because the clean target is known, the package can compute error-power metrics that would not be available in a live deployment.

## What is ERLE?

ERLE means **Echo Return Loss Enhancement**.  In this package the basic form is:

```text
ERLE dB = 10 log10( power(input_error) / power(output_error) )
```

where:

```text
input_error  = microphone - clean_target
output_error = enhanced_or_residual - clean_target
```

Positive ERLE means the output residual error has less power than the input echo/error.  For example, `10 dB` means the residual error power is about ten times lower; `20 dB` means about one hundred times lower.

## Why ERLE is interesting

ERLE is useful because it gives a simple scalar view of how much echo/error power was removed in a controlled experiment.  It is especially useful for:

- synthetic regression tests;
- comparing adaptive-filter variants;
- checking convergence across random seeds;
- validating whether a filter order or step size is obviously too weak.

## Why ERLE is not enough

ERLE can be misleading.  A processor can improve ERLE by suppressing everything, including desired near-end speech.  That is why the package also reports:

- `mse_improvement_db`;
- `segmental_erle_median_db`;
- `residual_power_db`;
- per-case timing.

For real AEC quality, listening tests and speech-preservation metrics matter.  This repository presents synthetic echo examples as adaptive-filter diagnostics, not production AEC performance results.

## Compared cases

`benchmarks/echo_cancellation_benchmark.py` compares small baselines:

1. `no_cancellation` — microphone passed through unchanged;
2. `toy_residual_suppressor_only` — fixed gain attenuation, included as a failure-mode baseline;
3. `spectral_residual_suppressor_only` — deterministic dependency-free spectral suppression;
4. `fir_nlms_baseline` — simple FIR/NLMS reference;
5. `lattice_iir_only` — adaptive stable lattice/IIR linear stage;
6. `lattice_iir_plus_toy_residual_suppressor`;
7. `lattice_iir_plus_spectral_residual_suppressor`.

These are educational baselines.  They are not intended to compete with WebRTC AEC, SpeexDSP, AirPods-class voice pickup, or Microsoft AEC Challenge systems.

## Recommended interpretation

Use this demo to say:

> “Here is how a stable adaptive lattice/IIR model behaves on a controlled echo-like problem, and here is the ERLE it obtains.”

Do not use it to say:

> “This package is a production echo canceller.”
