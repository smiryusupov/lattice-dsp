# Benchmarking strategy

The benchmark suite makes the package niche measurable by comparing the
compiled C++/pybind11 implementation against two baselines:

1. a deliberately simple pure-Python direct-form reference, and
2. SciPy `scipy.signal.lfilter` when SciPy is installed.

It also compares the package's own direct-form reference realization against the
true synthesis lattice-ladder realization.

`lattice-dsp` does not replace SciPy for static filtering.
SciPy is mature, heavily optimized, and broad. The goal is to show the value of
this project for workflows that combine stable reflection-parameterized IIR
filters, streaming APIs, adaptive algorithms, and batched independent streams.

## Run locally

```bash
python -m pip install -e .[benchmark]
python benchmarks/run_benchmarks.py --channels 64 --samples 50000 --repeats 5 --output benchmark-results.json
```

To include the slow pure-Python reference:

```bash
python benchmarks/run_benchmarks.py --include-python-reference
```

To include the experimental adaptive IIR update path:

```bash
python benchmarks/run_benchmarks.py --include-adaptive
```

To compare the analytic adaptive gradient against the slow finite-difference
reference:

```bash
python benchmarks/run_benchmarks.py --include-adaptive --include-finite-difference-adaptive
```

## What to expect

- A single IIR stream is sequential because recursive filters depend on previous
  state.
- OpenMP helps when there are many independent rows: channels, filter-bank
  members, Monte Carlo trials, or parameter sweeps.
- `process_batch(..., n_threads=1)` is useful as a baseline for OpenMP overhead.
- `process_batch(..., n_threads=0)` lets OpenMP choose the default number of
  threads.
- `realization="direct"` benchmarks the reference direct-form path.
- `realization="lattice"` benchmarks the synthesis lattice-ladder path after
  converting direct numerator taps to ladder taps.
- `--include-adaptive` is intentionally separate because adaptive denominator
  updates are much more expensive than static filtering.

## JSON output

The benchmark script writes a JSON file with metadata and timing summaries:

```json
{
  "metadata": {
    "channels": 64,
    "samples": 50000,
    "has_openmp": true
  },
  "benchmarks": {
    "lattice_cpp_batch_default_threads": {
      "min_s": 0.01,
      "median_s": 0.02,
      "max_s": 0.03
    }
  }
}
```

Actual numbers are machine-dependent, so compare results on the same machine or
inside the same CI runner.

## Adaptive block benchmarks

Adaptive filtering is recursive within each stream, so one adaptive problem is
not parallelized over samples. The benchmark suite separates several costs:

- `adaptive_lattice_ladder_nlms_analytic_numpy_block`: one C++ call over a
  NumPy vector, returning output and error arrays.
- `adaptive_lattice_ladder_nlms_analytic_list_block`: legacy list/vector API,
  useful to show Python conversion overhead.
- `adaptive_lattice_ladder_nlms_analytic_sample_loop`: optional slow baseline
  that crosses the Python/C++ boundary once per sample.
- `adaptive_lattice_ladder_nlms_analytic_batch_*`: optional independent
  channel-by-sample adaptation with OpenMP over rows.

Example:

```bash
python benchmarks/run_benchmarks.py --include-adaptive --samples 10000 --repeats 3
python benchmarks/run_benchmarks.py --include-adaptive --include-adaptive-sample-loop --samples 10000 --repeats 3
python benchmarks/run_benchmarks.py --include-adaptive --include-adaptive-batch --adaptive-batch-channels 8 --samples 10000 --repeats 3
python benchmarks/run_benchmarks.py --include-adaptive --reflection-update-period 8 --samples 10000 --repeats 3
```

`--reflection-update-period K` keeps numerator/tap updates sample-by-sample but
updates reflection/raw denominator parameters only every `K` samples. This is a
practical control for adaptive IIR work because denominator updates are both
more expensive and more sensitive than numerator updates.

## Adaptive reflection-update period sweep

`reflection_update_period` is a speed/quality control for stable adaptive IIR
identification. Numerator/tap updates still run every sample, while the more
expensive denominator/reflection update runs every `K` samples. The sweep script
uses period-scaled reflection steps by default, so period `K` uses
`mu_reflection * K` on update samples. This avoids unfairly under-training longer
periods. Add `--no-scale-reflection-mu-by-period` to inspect the fixed-step
behavior.

Run:

```bash
python benchmarks/adaptive_period_sweep.py \
  --periods 1 2 4 8 16 32 \
  --samples 20000 \
  --repeats 5 \
  --output adaptive-period-sweep.json \
  --csv-output adaptive-period-sweep.csv
```

Important fields in the JSON/CSV output:

- `median_s`: median runtime for one adaptive identification run.
- `speedup_vs_first_period`: runtime improvement relative to the first period.
- `mse_tail`: mean squared error over the final `tail` samples.
- `tail_mse_ratio_vs_first_period`: convergence-quality change relative to period `1`.
- `reflection_l2_error` and `taps_l2_error`: final coefficient errors against
  the synthetic target used by the benchmark.
- `stability_margin`: `1 - max(abs(reflection))`; positive values confirm the
  bounded reflection parameterization remains stable.

Use this sweep to report adaptive performance under a stated data distribution. A larger period can be much faster, and the tail MSE plus coefficient errors show whether that period is acceptable for the application.


## Adaptive period reports

After running `benchmarks/adaptive_period_sweep.py`, generate a compact report:

```bash
python benchmarks/adaptive_period_report.py adaptive-period-sweep-scaled.json \
  --markdown-output adaptive-period-report.md \
  --csv-output adaptive-period-pareto.csv \
  --plot-output adaptive-period-tradeoff.png \
  --max-tail-mse-ratio 1.5
```

The report marks the recommended `reflection_update_period` according to a
quality threshold and writes the Pareto frontier.  Use this to distinguish a
speedup without also reporting the convergence/quality cost.

## Multi-seed adaptive-period validation

Use the multi-seed sweep when selecting a default `reflection_update_period` for
examples or documentation.  It aggregates each period across several random
inputs and reports median and worst-case quality ratios.

```bash
python benchmarks/adaptive_multiseed_sweep.py \
  --seeds 100 101 102 103 104 \
  --periods 1 2 4 8 16 32 \
  --samples 20000 \
  --repeats 3 \
  --output adaptive-multiseed-sweep.json \
  --csv-output adaptive-multiseed-aggregate.csv

python benchmarks/adaptive_multiseed_report.py adaptive-multiseed-sweep.json \
  --markdown-output adaptive-multiseed-report.md \
  --csv-output adaptive-multiseed-pareto.csv \
  --max-tail-mse-ratio-median 1.5 \
  --max-tail-mse-ratio-worst 2.0
```

Prefer a conservative period if the fastest period has a high worst-case
tail-MSE ratio, even when its median ratio looks acceptable.


## API-level tuning

Use `lattice_dsp.tune_reflection_update_period` when you already have validation data in memory and want a programmatic recommendation instead of a CLI report. The helper supports 1-D data and 2-D trial-by-sample matrices. A 1-D run returns a valid signal-specific recommendation and includes a warning in `result["warnings"]`; a 2-D run can be labelled robust when it meets `min_trials_for_robust`.


## Echo/noise cancellation benchmark

`benchmarks/echo_cancellation_benchmark.py` validates the application-facing
hybrid direction: stable lattice/IIR linear modelling plus optional residual
suppression. It writes a JSON file with ERLE, MSE improvement, residual power,
and timing for no cancellation, FIR/NLMS, lattice/IIR, residual-only, and
hybrid cases.

```bash
python benchmarks/echo_cancellation_benchmark.py --samples 64000 --output echo-benchmark.json
```

The included residual suppressors are simple dependency-free baselines for making ERLE/MSE behavior visible; they are not production AEC quality metrics.


## Multi-seed echo validation

Robust echo/noise-cancellation validation is available with:

```bash
python benchmarks/echo_multiseed_benchmark.py \
  --seeds 100 101 102 103 104 \
  --nonlinear-strengths 0.0 0.08 0.16 \
  --noise-snr-dbs 40 30 \
  --near-end-power-ratios 0.0 0.02 \
  --output echo-multiseed-benchmark.json \
  --csv-output echo-multiseed-summary-raw.csv

python benchmarks/echo_multiseed_report.py echo-multiseed-benchmark.json \
  --markdown-output echo-multiseed-report.md \
  --csv-output echo-multiseed-summary.csv
```

See `docs/echo_multiseed_validation.md` for interpretation guidance.
