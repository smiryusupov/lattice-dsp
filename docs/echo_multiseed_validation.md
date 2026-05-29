# Echo/noise cancellation multi-seed validation

The single-run nonlinear echo benchmark is useful for debugging and is not an application-level validation by itself.  The multi-seed benchmark repeats
that experiment across independent synthetic signal realizations and optional
scenario sweeps.

The goal is to validate the hybrid idea:

```text
stable adaptive lattice/IIR = linear echo-path model
optional residual processor = nonlinear echo/noise/speech residual stage
```

## Run a small robust benchmark

```bash
python benchmarks/echo_multiseed_benchmark.py \
  --seeds 100 101 102 103 104 \
  --nonlinear-strengths 0.0 0.08 0.16 \
  --noise-snr-dbs 40 30 \
  --near-end-power-ratios 0.0 0.02 \
  --samples 64000 \
  --repeats 3 \
  --reflection-update-period 8 \
  --output echo-multiseed-benchmark.json \
  --csv-output echo-multiseed-summary-raw.csv
```

Then generate a readable report:

```bash
python benchmarks/echo_multiseed_report.py echo-multiseed-benchmark.json \
  --markdown-output echo-multiseed-report.md \
  --csv-output echo-multiseed-summary.csv
```

## What the report compares

For each scenario, the benchmark aggregates these cases over seeds:

- `no_cancellation`
- `toy_residual_suppressor_only`
- `fir_nlms_baseline`
- `lattice_iir_only`
- `lattice_iir_plus_toy_residual_suppressor`

The report highlights:

- median ERLE for FIR, lattice/IIR, and hybrid cases
- worst-seed ERLE gain relative to FIR/NLMS
- median runtime speedup relative to FIR/NLMS
- hybrid gain over lattice/IIR-only processing
- best method by median ERLE for each scenario

## Interpreting results

The most important distinction is between two interpretations:

1. `lattice_iir_only` tests whether the stable adaptive IIR stage models the
   linear echo path efficiently.
2. `lattice_iir_plus_toy_residual_suppressor` tests the hybrid architecture,
   where the residual stage is a simple dependency-free suppressor,
   or other nonlinear/noise suppressor.

The toy residual suppressor is intentionally simple and dependency-free.  It is
not meant to be a production nonlinear echo suppressor.  Its role is to verify
that the API and benchmark pipeline can evaluate residual-stage improvements.

## Suggested result wording

Use scenario-specific wording.  Prefer phrasing like:

> In synthetic multi-seed benchmarks, the lattice/IIR stage can model the stable
> linear echo path efficiently.  The hybrid API exposes the remaining residual
> for residual suppression, enabling explicit speed/quality
> comparisons against FIR/NLMS baselines.

## Spectral residual comparisons

The multi-seed benchmark includes additional comparisons for the deterministic spectral residual stage:

- `spectral_hybrid_vs_fir`
- `spectral_hybrid_vs_lattice`
- `spectral_hybrid_vs_toy_hybrid`

These comparisons help answer whether the residual stage itself is doing useful work beyond the stable lattice/IIR linear model and beyond the fixed-gain toy baseline.
